"""
tiktoken-backed implementation of TokenizerAdapter.
"""

import tiktoken

from token_analyzer.adapters.base import TokenizerAdapter, TokenString


class TiktokenAdapter(TokenizerAdapter):
    """
    Wraps a tiktoken.Encoding, translating its native API into the
    TokenizerAdapter contract.
    """

    def __init__(self, encoding: tiktoken.Encoding):
        self._encoding = encoding

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects str, got {type(text).__name__}")
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)

    def get_token_strings(self, text: str) -> list[TokenString]:
        token_ids = self.encode(text)
        result: list[TokenString] = []
        for token_id in token_ids:
            raw_bytes = self._encoding.decode_single_token_bytes(token_id)
            try:
                # Strict mode: raises instead of silently inserting U+FFFD.
                # Success here means these bytes are a complete, valid
                # character (or characters) on their own.
                decoded = raw_bytes.decode("utf-8", errors="strict")
                result.append(TokenString(token_id=token_id, text=decoded))
            except UnicodeDecodeError:
                # This token's bytes are only a fragment of a multi-byte
                # character that some neighboring token(s) complete.
                # No correct standalone string exists for it.
                result.append(TokenString(token_id=token_id, text=None))
        return result

    @property
    def version(self) -> str:
        # Safe: tiktoken's vocab/merge data is bundled in the installed
        # package, not fetched remotely — the package version and the
        # actual data in use are always in lockstep.
        return f"tiktoken-{tiktoken.__version__}"