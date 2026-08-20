"""
Fast integration tests: real FastAPI app, real HTTP-shaped requests via
TestClient, but the tokenizer adapter is monkeypatched to a lightweight
fake - no real tiktoken loads, no network calls.

Purpose: catch bugs in the wiring itself (status codes, schema shapes,
the exception handler, rate limiting, request tracing) fast and reliably,
independent of tokenizer backend availability. This is the tier that
should run on every commit / every CI push.

These do NOT prove the real tokenizer integration works end-to-end
see test_api_integration_real.py for that.
"""

import re

import pytest
from fastapi.testclient import TestClient

from token_analyzer.adapters.base import TokenString


class _FakeAdapter:
    """A fast, deterministic stand-in for a real tokenizer adapter --
    no network, no real BPE/vocab logic, just enough to exercise the API
    layer's wiring."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def get_token_strings(self, text: str) -> list[TokenString]:
        return [TokenString(token_id=i, text=w) for i, w in enumerate(text.split())]


@pytest.fixture
def client(monkeypatch):
    """Patches get_adapter at the point analyzer.py actually calls it
    (token_analyzer.analyzer.get_adapter, the name bound by that
    module's `from token_analyzer.registry import get_adapter`), so
    every call to analyze()/compare() in these tests uses the fake
    adapter for any 'gpt-4'-like known model, and still raises
    UnsupportedModelError for anything else matching real behavior."""

    from token_analyzer.exceptions import UnsupportedModelError

    def fake_get_adapter(model_name: str):
        if model_name not in {"gpt-4", "gpt-4-turbo"}:
            raise UnsupportedModelError(f"Unsupported model: {model_name}")
        return _FakeAdapter()

    import token_analyzer.analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "get_adapter", fake_get_adapter)

    from api.app.main import app, limiter

    # slowapi's limiter storage is a module-level singleton -- without
    # resetting it, requests made by earlier tests in this file would
    # count toward this test's quota too, causing order-dependent
    # failures. Reset ensures every test starts with a clean quota.
    limiter.reset()

    return TestClient(app)


# --- /analyze ---------------------------------------------------------------

def test_analyze_success_returns_200_and_correct_shape(client):
    r = client.post("/analyze", json={"text": "hello world", "model": "gpt-4"})
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "success"
    assert body["model_name"] == "gpt-4"
    assert body["token_count"] == 2
    assert body["char_count"] == len("hello world")
    assert len(body["tokens"]) == 2
    assert body["tokens"][0].keys() == {"token_id", "text"}


def test_analyze_unsupported_model_returns_404(client):
    r = client.post("/analyze", json={"text": "hello", "model": "not-a-real-model"})
    assert r.status_code == 404
    assert r.json()["error"] == "unsupported_model"


def test_analyze_wrong_type_returns_422_automatically(client):
    # Unquoted int -- genuinely the wrong JSON type, not a numeric string.
    r = client.post("/analyze", json={"text": 12345, "model": "gpt-4"})
    assert r.status_code == 422


def test_analyze_numeric_string_is_accepted_not_a_bug(client):
    # "12345" IS a valid string -- str only constrains type, not content.
    # This must succeed; it is not the same case as the test above.
    r = client.post("/analyze", json={"text": "12345", "model": "gpt-4"})
    assert r.status_code == 200


def test_analyze_missing_field_returns_422(client):
    r = client.post("/analyze", json={"text": "hello"})  # missing `model`
    assert r.status_code == 422


# --- /compare -----------------------------------------------------------

def test_compare_mixed_success_and_failure_shapes(client):
    r = client.post(
        "/compare",
        json={"text": "hello world", "models": ["gpt-4", "not-a-real-model"]},
    )
    assert r.status_code == 200  # compare() isolates failures -- never a 404 here

    results = r.json()["results"]
    assert results["gpt-4"]["status"] == "success"
    assert results["not-a-real-model"]["status"] == "error"
    assert "error" in results["not-a-real-model"]


# --- Rate limiting --------------------------------------------------------

def test_rate_limit_returns_429_after_ten_requests(client):
    for i in range(10):
        r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
        assert r.status_code == 200, f"request {i + 1} unexpectedly failed"

    r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    assert r.status_code == 429


# --- Request tracing --------------------------------------------------------

def test_request_id_header_present_and_looks_like_a_uuid(client):
    r = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    req_id = r.headers.get("x-request-id")
    assert req_id is not None
    uuid4_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    assert re.match(uuid4_pattern, req_id, re.IGNORECASE)


def test_request_id_differs_across_requests(client):
    r1 = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    r2 = client.post("/analyze", json={"text": "hi", "model": "gpt-4"})
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]