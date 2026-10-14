"""
Ties the cache to the existing analyze() pipeline (Milestone 3) via the
placement decided in Milestone 2: cache check happens BEFORE the adapter
registry is ever touched. On a hit, the adapter registry is bypassed
entirely; on a miss, analyze() runs as before and the result is stored.

Also where request/cache-hit/latency metrics get recorded (Milestone 6,
Observability) — this is the one chokepoint every /analyze call passes
through regardless of cache backend, so it's the natural place to measure.

*** PLACEHOLDER FLAG, same spirit as analyzer.py's cost table (Milestone 3) ***
`get_tokenizer_version()` below is a stand-in — wire this to
`get_adapter(model_name).version` in your real project (see this session's
Part 1 discussion: TiktokenAdapter.version = tiktoken.__version__,
HuggingFaceAdapter.version = a hash of the loaded Tokenizer's own state).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from ..observability.metrics import (
    ANALYZE_LATENCY_SECONDS,
    CACHE_HIT_COUNT,
    CACHE_MISS_COUNT,
    REQUEST_COUNT,
)
from .cache_backend import CacheBackend
from .keys import compute_cache_key

# Hedge against the ONE invalidation trigger that can't be put in the key:
# the cost/pricing table (Milestone 3's flagged placeholder) can drift while
# text/model_name/tokenizer_version stay identical. 6 hours bounds how long a
# stale price can be served without requiring an active invalidation push.
DEFAULT_TTL_SECONDS = 6 * 60 * 60


def get_tokenizer_version(model_name: str) -> str:
    """PLACEHOLDER — see module docstring."""
    return "unversioned-placeholder"


def analyze_cached(
    text: str,
    model_name: str,
    backend: CacheBackend,
    analyze_fn: Any,  # Callable[[str, str], AnalysisResult] — analyzer.analyze
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
):
    """Cache-checking wrapper around analyzer.analyze(). Same signature and
    return type as analyze() itself — a drop-in substitution, same principle
    as TokenizerAdapter's interchangeable concrete backends."""
    REQUEST_COUNT.labels(model_name=model_name).inc()
    start = time.perf_counter()

    tokenizer_version = get_tokenizer_version(model_name)
    key = compute_cache_key(text, model_name, tokenizer_version)

    cached = backend.get(key)
    if cached is not None:
        CACHE_HIT_COUNT.labels(model_name=model_name).inc()
        result = _dict_to_result(cached)
        ANALYZE_LATENCY_SECONDS.labels(model_name=model_name, cache_result="hit").observe(
            time.perf_counter() - start
        )
        return result

    CACHE_MISS_COUNT.labels(model_name=model_name).inc()
    result = analyze_fn(text, model_name)  # registry lookup + adapter call happen here, only on a miss
    backend.set(key, dataclasses.asdict(result), ttl_seconds=ttl_seconds)
    ANALYZE_LATENCY_SECONDS.labels(model_name=model_name, cache_result="miss").observe(
        time.perf_counter() - start
    )
    return result


def _dict_to_result(data: dict):
    from ..analyzer import AnalysisResult, TokenString  # adjust import paths to your real modules

    tokens = [TokenString(**t) for t in data["tokens"]]
    return AnalysisResult(**{**data, "tokens": tokens})