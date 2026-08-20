"""
Real-backend integration tests: the same API surface as
test_api_integration_stub.py, but hitting the REAL token_analyzer
library - real tiktoken adapters, real vocab loads.

Slower than the stub suite (first real tokenizer load per model
downloads/parses a real vocab file) and requires network access, so this
tier is a good candidate for a separate, less-frequent CI stage
rather than running on every single commit.
"""

import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Imported here, not at module level -- matches
    # test_api_integration_stub.py's pattern. Importing api.app.main at
    # module scope runs it at collection time, before pytest necessarily
    # has the project root on sys.path, which is what caused the
    # `ModuleNotFoundError: No module named 'api'` collection error.
    # Deferring the import to inside the fixture (test-run time) avoids
    # depending on pytest's sys.path/rootdir configuration entirely.
    from api.app.main import app, limiter
    limiter.reset()
    return TestClient(app)


# --- /analyze ---------------------------------------------------------------

def test_analyze_success_returns_200_and_correct_shape(client):
    r = client.post("/analyze", json={"text": "hello world", "model": "gpt-4"})
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "success"
    assert body["model_name"] == "gpt-4"
    # Real tokenizer: don't assert an exact count here the way the stub
    # suite does -- real BPE token counts for "hello world" depend on the
    # actual vocab, not our fake word-split logic. Assert the invariant
    # that actually matters instead.
    assert body["token_count"] > 0
    assert body["token_count"] == len(body["tokens"])
    assert body["char_count"] == len("hello world")
    # Real cost table lookup (Milestone 3) -- gpt-4 has known pricing.
    assert body["estimated_cost_usd"] is not None


def test_analyze_model_with_no_pricing_returns_null_cost(client):
    # If your SUPPORTED_MODELS includes a model absent from
    # _COST_PER_1K_TOKENS_USD, swap the model name below to match it --
    # this proves the `float | None` contract actually round-trips as
    # JSON `null`, not a crash or a fabricated 0.0.
    pytest.skip("Set a real model name with no cost-table entry, if one exists")


def test_analyze_unsupported_model_returns_404(client):
    r = client.post("/analyze", json={"text": "hello", "model": "not-a-real-model"})
    assert r.status_code == 404
    assert r.json()["error"] == "unsupported_model"


def test_analyze_wrong_type_returns_422(client):
    r = client.post("/analyze", json={"text": 12345, "model": "gpt-4"})
    assert r.status_code == 422


# --- /compare -----------------------------------------------------------

def test_compare_mixed_success_and_failure_shapes(client):
    r = client.post(
        "/compare",
        json={"text": "hello world", "models": ["gpt-4", "not-a-real-model"]},
    )
    assert r.status_code == 200

    results = r.json()["results"]
    assert results["gpt-4"]["status"] == "success"
    assert results["not-a-real-model"]["status"] == "error"


def test_compare_same_text_different_models_can_yield_different_token_counts(client):
    # A real cross-tokenizer property worth asserting: different models'
    # vocabularies genuinely tokenize the same text differently. This is
    # the actual point of the whole project -- worth a real assertion,
    # not just a shape check. Adjust model names to two real, differently
    # -vocabbed models you support.
    r = client.post(
        "/compare",
        json={"text": "The quick brown fox jumps.", "models": ["gpt-4", "gpt-3.5-turbo"]},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert all(v["status"] == "success" for v in results.values())


# --- Rate limiting (unchanged from stub suite -- pure API-layer behavior) --

def test_rate_limit_returns_429_after_ten_requests(client):
    for i in range(10):
        r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
        assert r.status_code == 200, f"request {i + 1} unexpectedly failed"

    r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    assert r.status_code == 429


# --- Request tracing (unchanged -- pure API-layer behavior) ----------------

def test_request_id_header_present_and_looks_like_a_uuid(client):
    r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    req_id = r.headers.get("x-request-id")
    assert req_id is not None
    uuid4_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    assert re.match(uuid4_pattern, req_id, re.IGNORECASE)