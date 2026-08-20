"""
Tests for TiktokenAdapter.

Split into two clearly separated groups:

1. Unit tests (TestTiktokenAdapterWithToyEncoding) - use a small,
   hand-built tiktoken.Encoding with NO network dependency. The vocab is
   deliberately engineered to force a split-character case on purpose,
   every run, regardless of what OpenAI ever does to cl100k_base. A
   failing test here always means our adapter logic broke.

2. Integration test (test_real_cl100k_base_...) - exercises the real
   cl100k_base encoding. Needs network on first run (downloads/caches
   merge tables). Skipped gracefully (not failed) when that's unavailable,
   since a skip here means "couldn't verify against the real thing right
   now," not "our code is broken."
"""

import tiktoken
import pytest

from token_analyzer.adapters.tiktoken_adapter import TiktokenAdapter
from token_analyzer.adapters.base import TokenString


def _build_toy_encoding() -> tiktoken.Encoding:
    """
    Base vocab: every single byte (0-255) is its own token, matching
    real byte-level BPE's foundation. Plus a couple of hand-picked merges
    so common substrings collapse into single tokens - but 'é' (bytes
    0xC3 0xA9) is deliberately NEVER merged into one token, guaranteeing
    a split-character case every time this encoding is used.
    """
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    next_id = 256
    for merge in [b"ca", b"caf"]:
        mergeable_ranks[merge] = next_id
        next_id += 1

    return tiktoken.Encoding(
        name="toy_split_char",
        pat_str=r"""[^\s]+|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


@pytest.fixture
def toy_adapter() -> TiktokenAdapter:
    return TiktokenAdapter(_build_toy_encoding())


class TestTiktokenAdapterWithToyEncoding:
    """Unit tests: no network, fully deterministic, split-char case
    guaranteed by construction."""

    def test_encode_rejects_non_str(self, toy_adapter):
        with pytest.raises(TypeError):
            toy_adapter.encode(12345)

    def test_encode_returns_list_of_ints(self, toy_adapter):
        ids = toy_adapter.encode("caf")
        assert ids == [257]  # the merged "caf" token, per our toy vocab

    def test_decode_round_trips_exactly(self, toy_adapter):
        text = "café"
        ids = toy_adapter.encode(text)
        assert toy_adapter.decode(ids) == text

    def test_get_token_strings_flags_split_character_as_none(self, toy_adapter):
        result = toy_adapter.get_token_strings("café")

        # "caf" merges into one valid token -> real string, not None
        assert result[0] == TokenString(token_id=257, text="caf")

        # é's two bytes (0xC3, 0xA9) are NEVER merged in our toy vocab,
        # so neither can be independently decoded -- both MUST be None.
        assert result[1].text is None
        assert result[2].text is None

        # And critically: the byte values still round-trip correctly
        # when decoded TOGETHER, proving no information was lost --
        # only the per-token isolated view is undefined, not the data.
        ids = [ts.token_id for ts in result]
        assert toy_adapter.decode(ids) == "café"

    def test_get_token_strings_does_not_flag_ascii_as_none(self, toy_adapter):
        # Plain ASCII has no multi-byte split risk -- every token here
        # should decode to a real string, never None.
        result = toy_adapter.get_token_strings("hi")
        assert all(ts.text is not None for ts in result)


def _cl100k_base_is_reachable() -> bool:
    try:
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _cl100k_base_is_reachable(),
    reason="cl100k_base not reachable (no network or host unavailable) -- "
    "skipping integration test, not treating as a code failure.",
)
def test_real_cl100k_base_handles_emoji_split():
    """
    Integration test: confirms our adapter's None-detection logic behaves
    correctly against the ACTUAL cl100k_base encoding, not just our toy
    vocab. Emoji are reliably multi-token in real byte-level BPE, so this
    should always find at least one None - but if OpenAI's real vocab
    ever changes such that it doesn't, that's genuinely useful information
    this test is designed to surface, not noise to suppress.
    """
    adapter = TiktokenAdapter(tiktoken.get_encoding("cl100k_base"))
    result = adapter.get_token_strings("🎉")

    assert any(ts.text is None for ts in result)

    ids = [ts.token_id for ts in result]
    assert adapter.decode(ids) == "🎉"