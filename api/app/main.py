"""
FastAPI application: routes + the global exception-handler translation
layer from library exceptions -> HTTP responses.

Deliberately thin. All real logic lives in token_analyzer (Milestone 3);
this file's only job is: validate input (Pydantic), call the library,
shape the output (Pydantic), translate known library exceptions into the
correct HTTP status codes.
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from token_analyzer.analyzer import analyze, compare
from token_analyzer.analyzer import AnalysisError as LibAnalysisError
from token_analyzer.exceptions import UnsupportedModelError

from .context import request_id_var
from .logging_config import configure_logging
from .schemas import (
    AnalyzeErrorResponse,
    AnalyzeRequest,
    AnalyzeSuccessResponse,
    CompareRequest,
    CompareResponse,
    TokenStringResponse,
    UnsupportedModelErrorResponse
)

configure_logging()
logger = logging.getLogger("token_analyzer.api")


app = FastAPI(title="Token Analyzer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limiting ---------------------------------------------------------
# Keyed by IP address (get_remote_address) -- a known, accepted limitation:
# clients sharing a NAT gateway (e.g. one office network) share a quota.
# Fixing this properly means per-client API keys, which requires an auth
# layer this milestone doesn't build. Documented as a design tradeoff, not
# silently ignored -- revisit if/when auth is added.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Request tracing middleware --------------------------------------------
# Runs once, at the very start of every request, before any route or
# dependency. Generates one ID per request, sets it into the ContextVar
# (so anything downstream -- routes, this file's logger, or a log call
# buried inside token_analyzer -- can retrieve it via .get()), and also
# echoes it back as a response header for the client's own debugging.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        logger.info(f"request started: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"request finished: status={response.status_code}")
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        # Reset, not just leave set -- middleware runs on a shared worker,
        # and a stale ContextVar value could otherwise leak into a
        # different, later request that happens to reuse the same context.
        request_id_var.reset(token)


# --- Global exception handler ------------------------------------------
# Registered once, at the app level. Any route that lets an
# UnsupportedModelError escape -- present or future -- is caught here,
# automatically, with zero per-route error-handling code required.
@app.exception_handler(UnsupportedModelError)
async def unsupported_model_handler(request: Request, exc: UnsupportedModelError):
    return JSONResponse(status_code=404, content={"error": "unsupported_model", "detail": str(exc)})


# --- Helpers -------------------------------------------------------------
def _to_success_response(result) -> AnalyzeSuccessResponse:
    """Convert a library AnalysisResult -> the public AnalyzeSuccessResponse."""
    return AnalyzeSuccessResponse(
        model_name=result.model_name,
        text=result.text,
        token_count=result.token_count,
        char_count=result.char_count,
        tokens=[
            TokenStringResponse(token_id=t.token_id, text=t.text) for t in result.tokens
        ],
        estimated_cost_usd=result.estimated_cost_usd,
    )


# --- Routes ----------------------------------------------------------------
@app.post(
    "/analyze",
    response_model=AnalyzeSuccessResponse,
    responses={404: {"model": UnsupportedModelErrorResponse, "description": "Unsupported model"}},
)
@limiter.limit("10/minute")
def analyze_endpoint(request: Request, body: AnalyzeRequest) -> AnalyzeSuccessResponse:
    # Plain `def`, not `async def`: analyze() is 100% CPU-bound (the
    # tokenizer's one-time network fetch is already cached by
    # get_adapter()'s lru_cache from Milestone 3 -- nothing left here to
    # await). FastAPI runs plain-`def` routes in a worker thread pool,
    # off the event loop, so a long tokenization job doesn't block every
    # other concurrent request the way it would under `async def` with
    # no real await inside it.
    #
    # `request: Request` is required here -- not for us, but because
    # slowapi's @limiter.limit inspects the route's parameters for the
    # raw Starlette Request to identify the caller (get_remote_address).
    # The Pydantic body is named `body` instead of `request` to avoid
    # colliding with it.
    #
    # Note: analyze()'s TypeError path is unreachable here -- AnalyzeRequest
    # already guarantees body.text is a str before this line runs.
    # UnsupportedModelError is intentionally left to escape uncaught: the
    # global exception_handler above catches it and returns 404.
    result = analyze(body.text, body.model)
    logger.info(f"analyze succeeded: model={body.model} token_count={result.token_count}")
    return _to_success_response(result)


@app.post("/compare", response_model=CompareResponse)
@limiter.limit("10/minute")
def compare_endpoint(request: Request, body: CompareRequest) -> CompareResponse:
    # compare() never raises UnsupportedModelError itself -- it catches
    # per-model TokenAnalyzerError internally and stores an AnalysisError
    # in that model's slot instead (Milestone 3's partial-failure
    # isolation). So every entry here is guaranteed success-or-error,
    # never an uncaught exception -- there is no 404 case for this route.
    raw_results = compare(body.text, body.models)

    shaped: dict[str, AnalyzeSuccessResponse | AnalyzeErrorResponse] = {}
    for model_name, result in raw_results.items():
        if isinstance(result, LibAnalysisError):
            shaped[model_name] = AnalyzeErrorResponse(
                model_name=result.model_name, error=result.error
            )
        else:
            shaped[model_name] = _to_success_response(result)

    logger.info(
        f"compare finished: total={len(shaped)} "
        f"succeeded={sum(1 for r in shaped.values() if r.status == 'success')}"
    )
    return CompareResponse(results=shaped)