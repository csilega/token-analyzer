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
from token_analyzer.exceptions import UnsupportedModelError, TokenizerLoadError

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
    ...
    Raises:
        UnsupportedModelError: if model_name has no registered adapter.
        TokenizerLoadError: if the adapter is registered but its backend
            failed to load (e.g. a network failure fetching vocab/merge data).
    """
    if model_name in _TIKTOKEN_MODELS:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            raise TokenizerLoadError(
                f"Failed to load tiktoken encoding for {model_name}: {e}"
            ) from e
        return TiktokenAdapter(encoding)

    if model_name in _HUGGINGFACE_MODELS:
        repo_id = _HUGGINGFACE_MODELS[model_name]
        try:
            tokenizer = Tokenizer.from_pretrained(repo_id)
        except Exception as e:
            raise TokenizerLoadError(
                f"Failed to load HuggingFace tokenizer for {model_name}: {e}"
            ) from e
        return HuggingFaceAdapter(tokenizer)

    raise UnsupportedModelError(model_name)