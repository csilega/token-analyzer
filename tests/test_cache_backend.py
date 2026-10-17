"""
Unit tests for both CacheBackend implementations. RedisCacheBackend is tested
against hand-built fake clients (healthy and down): no real Redis needed.
The 'down' tests are the evidence for Milestone 6's claim that a Redis
outage degrades to a cache miss and never breaks a request.
"""
from token_analyzer.caching.cache_backend import InMemoryCacheBackend, RedisCacheBackend
from token_analyzer.observability.metrics import CACHE_BACKEND_REACHABLE

SAMPLE = {"model_name": "gpt-4", "token_count": 1,
          "tokens": [{"token_id": 1, "text": "hi"}]}


def _reachable_gauge() -> float:
    # prometheus_client internal attribute: same caveat as the Session 7
    # Histogram fix. If a client upgrade breaks this, inspect the Gauge again.
    return CACHE_BACKEND_REACHABLE._value.get()


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.last_ex = "never-called"

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value.encode("utf-8")  # real Redis hands back bytes
        self.last_ex = ex
        return True


class _DownRedis:
    def get(self, key):
        raise ConnectionError("simulated: Redis unreachable")

    def set(self, key, value, ex=None):
        raise ConnectionError("simulated: Redis unreachable")


# ---------- InMemoryCacheBackend ----------

def test_in_memory_miss_returns_none():
    assert InMemoryCacheBackend().get("absent") is None


def test_in_memory_round_trip():
    cache = InMemoryCacheBackend()
    cache.set("k", SAMPLE)
    assert cache.get("k") == SAMPLE


def test_in_memory_expired_entry_behaves_like_a_miss():
    cache = InMemoryCacheBackend()
    cache.set("k", SAMPLE, ttl_seconds=0)  # expires immediately
    assert cache.get("k") is None


def test_in_memory_mutating_returned_value_cannot_corrupt_cache():
    cache = InMemoryCacheBackend()
    cache.set("k", SAMPLE)
    first = cache.get("k")
    first["tokens"][0]["text"] = "CORRUPTED"
    assert cache.get("k")["tokens"][0]["text"] == "hi"


def test_in_memory_mutating_original_after_set_cannot_corrupt_cache():
    cache = InMemoryCacheBackend()
    original = {"tokens": [{"token_id": 1, "text": "hi"}]}
    cache.set("k", original)
    original["tokens"][0]["text"] = "CORRUPTED"
    assert cache.get("k")["tokens"][0]["text"] == "hi"


# ---------- RedisCacheBackend: healthy ----------

def test_redis_round_trip_through_json():
    cache = RedisCacheBackend(_FakeRedis())
    cache.set("k", SAMPLE)
    assert cache.get("k") == SAMPLE


def test_redis_ttl_is_delegated_to_redis():
    client = _FakeRedis()
    RedisCacheBackend(client).set("k", SAMPLE, ttl_seconds=60)
    assert client.last_ex == 60


def test_redis_healthy_miss_still_reports_reachable():
    """A miss and an outage both return None. The gauge is the ONLY signal
    that tells them apart, so it has to be asserted, not just executed."""
    assert RedisCacheBackend(_FakeRedis()).get("absent") is None
    assert _reachable_gauge() == 1


# ---------- RedisCacheBackend: down (graceful degradation) ----------

def test_redis_down_get_degrades_to_miss():
    assert RedisCacheBackend(_DownRedis()).get("k") is None
    assert _reachable_gauge() == 0


def test_redis_down_set_is_silent_no_op():
    RedisCacheBackend(_DownRedis()).set("k", SAMPLE)  # must not raise
    assert _reachable_gauge() == 0


def test_reachability_gauge_recovers_after_outage():
    RedisCacheBackend(_DownRedis()).get("k")
    assert _reachable_gauge() == 0
    RedisCacheBackend(_FakeRedis()).get("k")
    assert _reachable_gauge() == 1