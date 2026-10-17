"""
Proves the broad TokenAnalyzerError handler returns a structured 500 for a
tokenizer cold-load failure, WITHOUT leaking internal error details.
"""
import pytest
import tiktoken
import asyncio
import json
from fastapi.testclient import TestClient

from api.app.main import app, limiter
from token_analyzer.registry import get_adapter
from api.app.main import token_analyzer_error_handler
from token_analyzer.exceptions import TokenAnalyzerError


INTERNAL_DETAIL = "simulated: secret-internal-host.example unreachable"


@pytest.fixture
def client(monkeypatch):
    def _boom(model_name):
        raise ConnectionError(INTERNAL_DETAIL)

    monkeypatch.setattr(tiktoken, "encoding_for_model", _boom)
    get_adapter.cache_clear()   # api/tests doesn't inherit tests/conftest.py
    limiter.reset()
    yield TestClient(app)
    get_adapter.cache_clear()


def test_tokenizer_load_failure_returns_structured_500(client):
    response = client.post("/analyze", json={"text": "hello", "model": "gpt-4"})
    assert response.status_code == 500
    assert response.json()["error"] == "tokenizer_load_failed"


def test_tokenizer_load_failure_does_not_leak_internals(client):
    response = client.post("/analyze", json={"text": "hello", "model": "gpt-4"})
    assert INTERNAL_DETAIL not in response.text

class _FutureDomainError(TokenAnalyzerError):
    """Stands in for any TokenAnalyzerError subclass added later."""


def test_unknown_domain_error_falls_back_to_generic_safe_500():
    exc = _FutureDomainError("secret internal detail")
    response = asyncio.run(token_analyzer_error_handler(None, exc))
    assert response.status_code == 500
    assert json.loads(response.body)["error"] == "internal_error"
    assert b"secret internal detail" not in response.body