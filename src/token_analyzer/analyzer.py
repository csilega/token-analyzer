"""
Core analysis logic: the library's public entry points.

analyze(text, model_name) -> AnalysisResult      single-model metrics
compare(text, model_names) -> dict[str, AnalysisResult | AnalysisError]
                                                  cross-model comparison
"""

from dataclasses import dataclass

from token_analyzer.adapters.base import TokenString
from token_analyzer.exceptions import TokenAnalyzerError, UnsupportedModelError
from token_analyzer.registry import get_adapter

# USD per 1,000 tokens. Placeholder figures -- pricing changes over time
# and is NOT something this library should be trusted as the source of
# truth for. Flagging this explicitly rather than letting it look
# authoritative: a real system would fetch this from a config file or
# provider API, not hardcode it
_COST_PER_1K_TOKENS_USD = {
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01,
    "gpt-4o": 0.005,
    "gpt-3.5-turbo": 0.0015,
}


@dataclass(frozen=True)
class AnalysisResult:
    model_name: str
    text: str
    token_count: int
    char_count: int
    tokens: list[TokenString]
    estimated_cost_usd: float | None  # None if model has no known pricing


@dataclass(frozen=True)
class AnalysisError:
    """Represents a per-model failure inside a compare() call -- lets one
    bad model name fail without aborting the whole comparison."""
    model_name: str
    error: str


def analyze(text: str, model_name: str) -> AnalysisResult:
    """
    Run full token analysis for `text` against a single model.

    Raises:
        TypeError: if text is not a str.
        UnsupportedModelError: if model_name has no registered adapter.
    """
    if not isinstance(text, str):
        raise TypeError(f"analyze() expects str text, got {type(text).__name__}")

    adapter = get_adapter(model_name)  # raises UnsupportedModelError itself
    token_ids = adapter.encode(text)
    tokens = adapter.get_token_strings(text)

    cost_per_1k = _COST_PER_1K_TOKENS_USD.get(model_name)
    estimated_cost = (
        (len(token_ids) / 1000) * cost_per_1k if cost_per_1k is not None else None
    )

    return AnalysisResult(
        model_name=model_name,
        text=text,
        token_count=len(token_ids),
        char_count=len(text),
        tokens=tokens,
        estimated_cost_usd=estimated_cost,
    )


def compare(
    text: str, model_names: list[str]
) -> dict[str, AnalysisResult | AnalysisError]:
    """
    Run analyze() for `text` against every model in `model_names`.

    A single unsupported/failing model does not abort the whole
    comparison -- its slot in the result dict holds an AnalysisError
    instead of raising, so callers comparing e.g. 5 models still get
    results for the 4 that succeeded.
    """
    if not isinstance(text, str):
        raise TypeError(f"compare() expects str text, got {type(text).__name__}")

    results: dict[str, AnalysisResult | AnalysisError] = {}
    for model_name in model_names:
        try:
            results[model_name] = analyze(text, model_name)
        except TokenAnalyzerError as e:
            results[model_name] = AnalysisError(model_name=model_name, error=str(e))
    return results