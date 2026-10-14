"""
Cache storage backends for the Token Analyzer.

CacheBackend abstracts *where a cached value physically lives* — separate
from TokenizerAdapter (Milestone 3), which abstracts *where the tokenization
computation comes from*. Same architectural move (one fixed contract, swappable
implementations), different question being answered.

Values crossing this boundary are always plain, JSON-serializable dicts —
never raw dataclass instances — so the contract stays honest about what a
network-based backend (Redis) can actually carry. Each concrete backend owns
its own (de)serialization internally; callers never branch on which backend
is active.
"""

from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Protocol


class CacheBackend(ABC):
    """Contract every cache storage backend must satisfy."""

    @abstractmethod
    def get(self, key: str) -> Optional[dict]:
        """Return the cached value for `key`, or None on a miss (including an
        expired entry — an expired entry must behave identically to a
        never-cached one from the caller's perspective)."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: dict, ttl_seconds: Optional[int] = None) -> None:
        """Store `value` under `key`. If `ttl_seconds` is given, the entry
        must no longer be returned by `get()` after that many seconds."""
        raise NotImplementedError


class InMemoryCacheBackend(CacheBackend):
    """Process-local cache backed by a plain dict.

    Used for local development and as the default in tests (no external
    dependency, no network) — mirrors Milestone 3's FakeAdapter reasoning:
    a red test here should mean "my caching logic is wrong," not "Redis
    wasn't running." NOT shared across multiple server processes — see the
    Milestone 6 knowledge_graph.md entry for why that's a hit-rate problem,
    not a correctness one, given /analyze's determinism.

    Deliberately deep-copies on both set() and get() even though nothing in
    this process *currently* mutates a returned result — this makes the
    aliasing bug (a caller mutating a cache-shared object and corrupting it
    for every future caller) structurally impossible rather than merely
    against the rules, at negligible cost for this project's data sizes.
    This also keeps behavior identical to RedisCacheBackend, which gets this
    guarantee for free via serialization.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict, Optional[float]]] = {}

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            # Expired: evict lazily on read rather than running a background
            # sweep — simplest correct behavior for this project's scale.
            del self._store[key]
            return None
        return copy.deepcopy(value)

    def set(self, key: str, value: dict, ttl_seconds: Optional[int] = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (copy.deepcopy(value), expires_at)


class RedisLikeClient(Protocol):
    """The minimal slice of the real redis-py client's interface this
    backend depends on. Typing against this (not the concrete redis.Redis
    class) is what lets tests substitute a hand-built fake with zero network
    involvement, and is itself the same test-double principle used for
    TiktokenAdapter's toy Encoding in Milestone 3."""

    def get(self, key: str) -> Optional[bytes]: ...

    def set(self, key: str, value: str, ex: Optional[int] = None) -> object: ...


class RedisCacheBackend(CacheBackend):
    """Shared cache backed by Redis. Values are JSON-encoded before crossing
    the network (Redis has no concept of a Python dict) and decoded back on
    read. TTL is delegated to Redis's own native expiry (`EX` seconds) rather
    than reimplemented client-side, since Redis already does this correctly
    and atomically.

    The Redis client is injected, not constructed here — this keeps the
    backend itself dependency-free of *how* a client is configured (host,
    port, auth, connection pool sizing are the caller's concern) and,
    critically, testable against a fake with no real network call.

    Deliberately catches every exception from the underlying client — a
    broader catch than `compare()`'s "only TokenAnalyzerError, let real bugs
    propagate loudly" rule from Milestone 3, and deliberately so: caching was
    decided (Milestone 6, Observability discussion) to be a soft, optional
    dependency, so ANY failure reaching Redis (connection refused, timeout,
    auth failure — we can't enumerate every driver-specific exception type
    the injected client might raise) must degrade to "treat this as a cache
    miss," never propagate and break the request. `compare()`'s narrow catch
    exists to avoid *hiding real bugs in our own code*; this broad catch
    exists because failures in an already-optional external dependency
    should never be fatal to begin with — a different failure category,
    a deliberately different catching strategy.
    """

    def __init__(self, client: RedisLikeClient) -> None:
        self._client = client

    def get(self, key: str) -> Optional[dict]:
        try:
            raw = self._client.get(key)
        except Exception:
            _mark_cache_backend_unreachable()
            return None  # degrade to "cache miss" — analyze() still runs
        _mark_cache_backend_reachable()
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: dict, ttl_seconds: Optional[int] = None) -> None:
        raw = json.dumps(value)
        try:
            self._client.set(key, raw, ex=ttl_seconds)
        except Exception:
            _mark_cache_backend_unreachable()
            # Silently no-op: failing to WRITE to the cache is just a missed
            # optimization for future requests, never a reason to fail this one.
        else:
            _mark_cache_backend_reachable()


def _mark_cache_backend_reachable() -> None:
    from ..observability.metrics import CACHE_BACKEND_REACHABLE

    CACHE_BACKEND_REACHABLE.set(1)


def _mark_cache_backend_unreachable() -> None:
    from ..observability.metrics import CACHE_BACKEND_REACHABLE

    CACHE_BACKEND_REACHABLE.set(0)