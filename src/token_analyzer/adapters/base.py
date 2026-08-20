"""
Core adapter contract for the Token Analyzer.

Every tokenizer backend (tiktoken, HuggingFace, SentencePiece, ...) is wrapped
by a concrete subclass of TokenizerAdapter, translating that library's native
API into this shared shape. Nothing in this file knows about HTTP, FastAPI,
or any web concept — this is a plain, framework-agnostic contract, usable
from any Python script with zero server involved
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenString:
    """
    One token's displayable-string representation.

    `text` is the token's own decoded string ONLY when that token is
    independently decodable. Byte-level BPE can split a single multi-byte
    character's bytes across two or more adjacent tokens; in that case no
    single token has a valid standalone string on its own. `text` is `None`
    for exactly those tokens, signaling to any consumer at this token must 
    be rendered as part of a merged span with its neighbors, not as an
    isolated highlight box.
    """
    token_id: int
    text: str | None


class TokenizerAdapter(ABC):
    """
    Abstract contract every tokenizer backend must satisfy.

    This class cannot be instantiated directly (ABC + @abstractmethod
    enforce that at the language level) — only a subclass that implements
    every method below can be instantiated. This is what lets the rest of
    the system (registry, analyzer, the API layer) depend on
    ONE interface, never on tiktoken/HuggingFace/SentencePiece specifics.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Convert raw text into a list of token IDs.

        Raises:
            TypeError: if `text` is not a str. This is a plain Python
                exception, not an HTTP error — the API layer
                is responsible for translating this into a 4xx response.
                This library has no knowledge that HTTP exists.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back into text.

        This is the ONLY method that should ever be trusted to correctly
        reconstruct text from token IDs — decoding a single token ID in
        isolation is not equivalent to decoding it as part of its full
        sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def get_token_strings(self, text: str) -> list[TokenString]:
        """
        Return one TokenString per token in `text`, in order.

        Implementations must NOT decode each token ID in isolation and
        assume the result is correct — for byte-level BPE backends, some
        tokens have no valid standalone string (see TokenString.text).
        """
        raise NotImplementedError
