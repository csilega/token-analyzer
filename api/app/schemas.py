"""
Pydantic schemas: the API's executable contract.

These are deliberately separate from token_analyzer's internal dataclasses
(AnalysisResult, AnalysisError, TokenString) even though the fields
currently mirror them 1:1. The API's public shape and the library's
internal shape are allowed to diverge over time without either one
breaking the other.
"""

from typing import Literal

from pydantic import BaseModel, Field

from token_analyzer.registry import SUPPORTED_MODELS

# Upper bound on submitted text length, in characters.
#
# Derivation: 128,000 tokens (the largest context window among the models
# this API currently supports) x ~4 chars/token (a standard rough estimate
# for English text) = 512,000 chars. This exists purely to protect THIS
# service's own CPU time (the per-token loop in get_token_strings()) and
# memory (the token ID list, the TokenString list, the eventual response
# body) from a single oversized request -- it has nothing to do with a
# downstream model's attention cost, since no model ever runs here.
#
# Not a source of truth for real context-window sizes, which vary by
# model and change over time -- revisit if SUPPORTED_MODELS grows to
# include a model with a meaningfully larger window.
MAX_TEXT_LENGTH = 512_000
MAX_MODELS_PER_COMPARE = len(SUPPORTED_MODELS)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    model: str


class CompareRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    models: list[str] = Field(..., max_length=MAX_MODELS_PER_COMPARE)


class TokenStringResponse(BaseModel):
    token_id: int
    text: str | None


class AnalyzeSuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    model_name: str
    text: str
    token_count: int
    char_count: int
    tokens: list[TokenStringResponse]
    estimated_cost_usd: float | None


class AnalyzeErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    model_name: str
    error: str


class UnsupportedModelErrorResponse(BaseModel):
    error: Literal["unsupported_model"]
    detail: str


class CompareResponse(BaseModel):
    results: dict[str, AnalyzeSuccessResponse | AnalyzeErrorResponse]