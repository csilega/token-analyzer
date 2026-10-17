"""
HuggingFace `tokenizers`-backed implementation of TokenizerAdapter.
"""
import hashlib
from tokenizers import Tokenizer

from token_analyzer.adapters.base import TokenizerAdapter, TokenString


class HuggingFaceAdapter(TokenizerAdapter):
    """
    Wraps a HuggingFace `tokenizers.Tokenizer`.

    Note: WordPiece/Unigram-family HF tokenizers (e.g. BERT) operate on
    whole characters, not raw bytes -- so the "split-character" problem
    that motivated TokenString.text=None largely does not arise for them.
    Byte-level BPE HF tokenizers (e.g. GPT-2 style) CAN still exhibit it,
    same as tiktoken. We apply the same strict-decode detection either way
    so the contract is uniform regardless of backend.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects str, got {type(text).__name__}")
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def get_token_strings(self, text: str) -> list[TokenString]:
        token_ids = self.encode(text)
        result: list[TokenString] = []
        for token_id in token_ids:
            # HF's decode() on a single ID also applies its own
            # byte-fallback/replacement handling under the hood; we detect
            # unreliable single-token decodes the same way as tiktoken --
            # by comparing against a strict UTF-8 decode of the token's
            # own byte representation where available.
            single = self._tokenizer.decode([token_id])
            if "\ufffd" in single:
                # ANY replacement character means this token's bytes aren't
                # valid UTF-8 on their own -> honest None. Deliberately NOT
                # "entirely replacement chars" (see the Session 7 regression:
                # that let "\ufffda\ufffd" through as fabricated text). A real
                # U+FFFD in user input also lands here; an over-cautious None
                # is preferred to a possibly-fabricated string.
                result.append(TokenString(token_id=token_id, text=None))
            else:
                result.append(TokenString(token_id=token_id, text=single))
        return result

    @property
    def version(self) -> str:
        # NOT tokenizers.__version__ — that tracks the installed package,
        # not the actual downloaded tokenizer.json content, which can
        # drift on the Hub independently of any local package version.
        # Hash the adapter's own loaded state instead, so this changes
        # if and only if the real vocab/merges actually changed.
        serialized = self._tokenizer.to_str()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]