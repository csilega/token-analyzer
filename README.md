# Token Analyzer

Token Analyzer helps you understand how different LLMs tokenize text — token counts, per-token costs, and cross-model comparisons. I built it end-to-end as a first-principles learning project, not as a thin wrapper around existing libraries.

## Status

**Milestones 0–4 of 10 are complete.**

| Milestone | Status |
|---|---|
| 0 — Foundations and mental models (why tokenization exists, context windows, cost) | ✅ |
| 1 — Tokenization algorithms (BPE, WordPiece, SentencePiece, byte-level BPE) | ✅ |
| 2 — System architecture and design | ✅ |
| 3 — Core analyzer engine (adapter pattern, tiktoken and HuggingFace backends) | ✅ |
| 4 — API layer (FastAPI, validation, error handling, rate limiting, tracing, docs) | ✅ |
| 5 — Frontend and visualization | ⬜ next |
| 6–9 — Production hardening, formal testing strategy, CI/CD, final docs | ⬜ |

## Why I built it this way

I derive and defend every non-trivial design decision here before I implement it, rather than copying a pattern from a tutorial. A few examples that shaped the actual code:

- **Byte-level BPE tokens can be genuinely undecodable in isolation** — a token that's only valid as part of a multi-byte UTF-8 sequence. `get_token_strings()` returns `TokenString(token_id, text: str | None)`, where `None` is a signal that a token has no standalone string form. I found this by simulating a real split-character emoji case, proving two naive fixes (`OrderedDict` repackaging and prefix-diffing) both failed, and landing on strict-decode detection instead.
- **`compare()` isolates per-model failures** instead of letting one bad model name abort the whole batch — a small instance of the bulkhead resiliency pattern.
- **API routes use plain `def`, not `async def`.** The tokenization work is 100% CPU-bound once a tokenizer is cached, so an `async def` route with no real `await` inside it would block the entire event loop instead of using FastAPI's worker thread pool.
- **A `ContextVar`-based request ID** propagates from API middleware into library log calls with zero changes to the library's function signatures. Verified in a sandbox test that a log call inside the library picks up the API layer's request ID.

## Tech stack

Python 3.13, FastAPI, Pydantic, tiktoken, HuggingFace `tokenizers`, pytest, and slowapi.

## Setup

Install the package in editable mode:

```bash
pip install -e .
```

## Running the API

Start the server:

```bash
uvicorn api.app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive, auto-generated API documentation.

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "model": "gpt-4"}'
```

## Running tests

Run the core library tests:

```bash
pytest tests/ -v          # unit tests plus real-tokenizer integration tests
```

Run the API layer tests:

```bash
pytest api/tests/ -v      # stub-backed (fast) tests plus real-backend integration tests
```
