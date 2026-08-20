"""
Pydantic schemas: the API's executable contract.

These are deliberately separate from token_analyzer's internal dataclasses
(AnalysisResult, AnalysisError, TokenString) even though the fields
currently mirror them 1:1. The API's public shape and the library's
internal shape are allowed to diverge over time without either one
breaking the other.
"""

from typing import Literal

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str
    model: str


class CompareRequest(BaseModel):
    text: str
    models: list[str]


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


class CompareResponse(BaseModel):
    results: dict[str, AnalyzeSuccessResponse | AnalyzeErrorResponse]