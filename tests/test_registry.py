"""
Unit tests for registry.get_adapter().

tiktoken.encoding_for_model() touches the network (downloads merge tables
on first use). We monkeypatch it so these tests verify OUR logic
(model-name-to-adapter mapping, caching, error handling) without any
network dependency. A failing test here should always mean our registry
logic broke, never "the network was down."
"""

import pytest

from token_analyzer.adapters.huggingface_adapter import HuggingFaceAdapter
from token_analyzer.adapters.tiktoken_adapter import TiktokenAdapter
from token_analyzer.exceptions import UnsupportedModelError
from token_analyzer.registry import SUPPORTED_MODELS, get_adapter


class _FakeEncoding:
    """Stand-in for a real tiktoken.Encoding -- just needs to exist as an
    object; its internals are never exercised by these tests."""
    def __init__(self, name):
        self.name = name


class _FakeHFTokenizer:
    """Stand-in for a real tokenizers.Tokenizer -- same idea as
    _FakeEncoding above, avoiding any real HuggingFace Hub network call."""
    def __init__(self, repo_id):
        self.repo_id = repo_id


def test_get_adapter_returns_huggingface_adapter(monkeypatch):
    monkeypatch.setattr(
        "token_analyzer.registry.Tokenizer.from_pretrained",
        lambda repo_id: _FakeHFTokenizer(repo_id),
    )
    adapter = get_adapter("bert-base-uncased")
    assert isinstance(adapter, HuggingFaceAdapter)


def test_supported_models_includes_huggingface_models():
    assert "bert-base-uncased" in SUPPORTED_MODELS
    assert "gpt2" in SUPPORTED_MODELS


def test_get_adapter_returns_tiktoken_adapter(monkeypatch):
    monkeypatch.setattr(
        "token_analyzer.registry.tiktoken.encoding_for_model",
        lambda model_name: _FakeEncoding(model_name),
    )
    adapter = get_adapter("gpt-4")
    assert isinstance(adapter, TiktokenAdapter)


def test_get_adapter_caches_same_instance(monkeypatch):
    call_count = 0

    def fake_encoding_for_model(model_name):
        nonlocal call_count
        call_count += 1
        return _FakeEncoding(model_name)

    monkeypatch.setattr(
        "token_analyzer.registry.tiktoken.encoding_for_model",
        fake_encoding_for_model,
    )

    first = get_adapter("gpt-4")
    second = get_adapter("gpt-4")

    assert first is second
    assert call_count == 1  # the underlying constructor ran exactly once


def test_get_adapter_different_models_not_cached_together(monkeypatch):
    monkeypatch.setattr(
        "token_analyzer.registry.tiktoken.encoding_for_model",
        lambda model_name: _FakeEncoding(model_name),
    )
    gpt4 = get_adapter("gpt-4")
    gpt35 = get_adapter("gpt-3.5-turbo")
    assert gpt4 is not gpt35


def test_get_adapter_raises_for_unknown_model():
    with pytest.raises(UnsupportedModelError) as exc_info:
        get_adapter("definitely-not-a-real-model")
    assert "definitely-not-a-real-model" in str(exc_info.value)


def test_supported_models_is_frozenset():
    # Guards against someone accidentally changing this to a mutable list,
    # which would let calling code mutate the registry's source of truth.
    assert isinstance(SUPPORTED_MODELS, frozenset)
    assert "gpt-4" in SUPPORTED_MODELS