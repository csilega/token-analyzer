"""
Maps model names (e.g. "gpt-4") to the correct TokenizerAdapter instance,
caching instances per model name so repeated calls don't rebuild
encodings/merge tables from scratch.
"""

from functools import lru_cache

import tiktoken
from tokenizers import Tokenizer

from token_analyzer.adapters.base import TokenizerAdapter
from token_analyzer.adapters.huggingface_adapter import HuggingFaceAdapter
from token_analyzer.adapters.tiktoken_adapter import TiktokenAdapter
from token_analyzer.exceptions import UnsupportedModelError

# Models we know how to build a tiktoken-backed adapter for.
# Extend this as support grows
_TIKTOKEN_MODELS = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-3.5-turbo",
}

# Maps OUR model_name strings to the HuggingFace Hub repo ID that
# Tokenizer.from_pretrained() needs. Kept as an explicit mapping (rather
# than assuming model_name == repo_id) so our public-facing names don't
# have to match HF's naming exactly.
_HUGGINGFACE_MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "gpt2": "gpt2",
}

SUPPORTED_MODELS = frozenset(_TIKTOKEN_MODELS | _HUGGINGFACE_MODELS.keys())


@lru_cache(maxsize=None)
def get_adapter(model_name: str) -> TokenizerAdapter:
    """
    Return a cached TokenizerAdapter for `model_name`.

    The first call for a given model_name constructs the adapter (which
    may involve loading/downloading merge tables or tokenizer files);
    every subsequent call for that exact string returns the same cached
    instance.

    Raises:
        UnsupportedModelError: if model_name has no registered adapter.
    """
    if model_name in _TIKTOKEN_MODELS:
        encoding = tiktoken.encoding_for_model(model_name)
        return TiktokenAdapter(encoding)

    if model_name in _HUGGINGFACE_MODELS:
        repo_id = _HUGGINGFACE_MODELS[model_name]
        tokenizer = Tokenizer.from_pretrained(repo_id)
        return HuggingFaceAdapter(tokenizer)

    raise UnsupportedModelError(model_name)