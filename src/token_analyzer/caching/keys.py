"""Cache key computation — the compound key design from Milestone 2."""

from __future__ import annotations

import hashlib


def compute_cache_key(text: str, model_name: str, tokenizer_version: str) -> str:
    """A fixed-size, collision-resistant key derived from every input that
    affects the correct output. `tokenizer_version` guards against a silent
    tokenizer/vocab update serving a stale-but-cached-as-correct result
    (Milestone 2's invalidation trigger #1). Hashing (rather than using the
    raw text as the key) keeps key size bounded regardless of input length —
    relevant once this key is used as a Redis key or a dict key holding
    arbitrarily long user text.
    """
    raw = f"{model_name}\x00{tokenizer_version}\x00{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()