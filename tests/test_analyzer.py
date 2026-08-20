"""
Unit tests for analyzer.analyze() and analyzer.compare().

These tests substitute a small FakeAdapter for the real TokenizerAdapter,
via monkeypatching analyzer.get_adapter. This verifies analyze()/compare()'s
OWN logic (count aggregation, cost math, per-model error isolation)
completely independent of whether tiktoken/HuggingFace correctly tokenize
anything. A failing test here should always mean our analyzer logic broke.
"""

import pytest

from token_analyzer.adapters.base import TokenizerAdapter, TokenString
from token_analyzer.analyzer import AnalysisError, AnalysisResult, analyze, compare
from token_analyzer.exceptions import UnsupportedModelError


class FakeAdapter(TokenizerAdapter):
    """Deterministic stand-in: always splits text on whitespace and
    returns one fabricated token ID per word. No real tokenization logic
    -- exists purely so analyze()'s math can be checked against known,
    hand-computable inputs."""

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode() expects str")
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in token_ids)

    def get_token_strings(self, text: str) -> list[TokenString]:
        return [
            TokenString(token_id=i, text=word)
            for i, word in enumerate(text.split())
        ]


@pytest.fixture
def fake_adapter_for(monkeypatch):
    """Patches analyzer.get_adapter to return FakeAdapter for a given
    model name, regardless of what registry.py would normally do."""

    def _patch(model_name: str):
        monkeypatch.setattr(
            "token_analyzer.analyzer.get_adapter",
            lambda name: FakeAdapter() if name == model_name else (_ for _ in ()).throw(
                UnsupportedModelError(name)
            ),
        )

    return _patch


def test_analyze_rejects_non_str_text():
    with pytest.raises(TypeError):
        analyze(12345, "gpt-4")


def test_analyze_computes_token_and_char_counts(fake_adapter_for):
    fake_adapter_for("fake-model")
    result = analyze("hello there world", "fake-model")

    assert isinstance(result, AnalysisResult)
    assert result.token_count == 3          # 3 words -> 3 fake tokens
    assert result.char_count == len("hello there world")
    assert result.tokens == [
        TokenString(0, "hello"),
        TokenString(1, "there"),
        TokenString(2, "world"),
    ]


def test_analyze_computes_cost_for_known_model(fake_adapter_for):
    fake_adapter_for("gpt-4")  # gpt-4 IS in the cost table: $0.03 / 1k tokens
    result = analyze("one two three four", "gpt-4")  # 4 fake tokens

    assert result.token_count == 4
    assert result.estimated_cost_usd == pytest.approx((4 / 1000) * 0.03)


def test_analyze_cost_is_none_for_model_without_pricing(fake_adapter_for):
    fake_adapter_for("some-unpriced-model")
    result = analyze("one two", "some-unpriced-model")

    assert result.estimated_cost_usd is None


def test_analyze_propagates_unsupported_model_error(fake_adapter_for):
    fake_adapter_for("only-this-model-exists")
    with pytest.raises(UnsupportedModelError):
        analyze("hello", "a-totally-different-model")


def test_compare_rejects_non_str_text():
    with pytest.raises(TypeError):
        compare(12345, ["gpt-4"])


def test_compare_aggregates_multiple_models(monkeypatch):
    monkeypatch.setattr(
        "token_analyzer.analyzer.get_adapter", lambda name: FakeAdapter()
    )
    results = compare("one two three", ["model-a", "model-b"])

    assert set(results.keys()) == {"model-a", "model-b"}
    assert all(isinstance(r, AnalysisResult) for r in results.values())
    assert all(r.token_count == 3 for r in results.values())


def test_compare_isolates_failures_without_aborting(monkeypatch):
    def selective_get_adapter(name):
        if name == "good-model":
            return FakeAdapter()
        raise UnsupportedModelError(name)

    monkeypatch.setattr(
        "token_analyzer.analyzer.get_adapter", selective_get_adapter
    )
    results = compare("one two", ["good-model", "bad-model"])

    assert isinstance(results["good-model"], AnalysisResult)
    assert isinstance(results["bad-model"], AnalysisError)
    assert "bad-model" in results["bad-model"].error