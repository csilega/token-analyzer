"""
Liveness and readiness endpoints.

Deliberately two separate routes even though — per this session's Milestone
6 discussion — they currently return identical results for this service:
`token_analyzer`'s one external-ish dependency (the cache backend) was
explicitly designed to be non-blocking, so there's no external reachability
check that should ever exclude a process from the load balancer's routing
pool. Kept structurally separate anyway so that IF a genuine hard dependency
is ever added, only /health/ready needs to change — liveness must stay
permanently free of any dependency check, so it can never trigger a
pointless restart-storm over an external outage restarting wouldn't fix.

Neither route does real work — they exist to prove the process can still
respond through its normal request-handling path (see the knowledge_graph.md
entry on why a trivial endpoint still catches a genuine deadlock: a hung
process blocks EVERYTHING running in it, including the cheapest possible
handler, which is exactly the signal a timeout-based caller is checking for).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health/live")
def liveness() -> dict:
    """Is this process still responding at all? No dependency checks, on
    purpose — a failure here should mean 'restart me,' and restarting can
    only ever fix a problem inside this process, never an external one."""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness() -> dict:
    """Can this process correctly serve /analyze right now? For this
    service specifically: yes, whenever it's alive — see module docstring.
    Redis reachability is intentionally NOT checked here; see the
    cache_backend_reachable metric for that signal instead."""
    return {"status": "ready"}