"""
Metrics for the Token Analyzer API — the answer to questions logs can't
answer *cheaply* (see the Milestone 6 knowledge_graph.md entry for the full
reasoning): "what's our cache hit rate over the last hour," "what's our p99
/analyze latency right now" — computed via small, fixed-memory running
counters/histograms rather than scanning historical log lines.

Uses the real `prometheus_client` library rather than hand-rolling Counter/
Histogram — we already derived *why* a Counter is just an O(1) running total
and a Histogram is a fixed set of bucket counters (not a stored list of raw
values) through first-principles reasoning; this is applying a real,
production-standard implementation of that same idea, the same way `slowapi`
(Milestone 4) was a real implementation of the rate-limiting concept
discussed first.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Counters: simple running totals, O(1) increment, no history stored ---

REQUEST_COUNT = Counter(
    "token_analyzer_requests_total",
    "Total number of /analyze requests received.",
    labelnames=["model_name"],
)

CACHE_HIT_COUNT = Counter(
    "token_analyzer_cache_hits_total",
    "Total number of /analyze requests served from cache.",
    labelnames=["model_name"],
)

CACHE_MISS_COUNT = Counter(
    "token_analyzer_cache_misses_total",
    "Total number of /analyze requests NOT served from cache (adapter ran).",
    labelnames=["model_name"],
)

# --- Histogram: fixed latency buckets, approximates any percentile without
# ever storing an individual raw latency value. Buckets chosen for this
# service's expected range (single-request tokenization, not batch jobs) —
# revisit if real production data shows most traffic clustering outside
# this range.
ANALYZE_LATENCY_SECONDS = Histogram(
    "token_analyzer_analyze_latency_seconds",
    "Time to serve an /analyze request, cache hit or miss.",
    labelnames=["model_name", "cache_result"],  # cache_result: "hit" | "miss"
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# --- Gauge: Redis reachability is explicitly NOT part of /health/ready
# (caching is an optimization, not a hard dependency — a Redis outage must
# not pull processes out of the load balancer's pool). Surfaced here
# instead, as a metric a dashboard/alert can watch independently of routing.
CACHE_BACKEND_REACHABLE = Gauge(
    "token_analyzer_cache_backend_reachable",
    "1 if the configured cache backend (e.g. Redis) was reachable on the "
    "most recent attempt, 0 otherwise. NOT used for readiness/pool "
    "membership — caching is an optimization, not a correctness requirement.",
)