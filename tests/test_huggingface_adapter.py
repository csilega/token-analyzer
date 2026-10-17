"""
Unit tests for HuggingFaceAdapter, using a hand-built fake Tokenizer.
No HuggingFace Hub, no network: same reasoning as the toy tiktoken Encoding
in test_tiktoken_adapter.py.
"""
import pytest

from token_analyzer.adapters.huggingface_adapter import HuggingFaceAdapter


class _FakeEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeHFTokenizer:
    """Mimics the slice of tokenizers.Tokenizer that the adapter uses.

    `encodings` maps an input string to its token IDs; `pieces` maps each ID to
    what HF's decode([id]) would return, including replacement characters
    for byte fragments that aren't valid UTF-8 on their own.
    """

    def __init__(self, encodings, pieces, serialized="fake-tokenizer-v1"):
        self._encodings = encodings
        self._pieces = pieces
        self._serialized = serialized

    def encode(self, text):
        return _FakeEncoding(self._encodings[text])

    def decode(self, ids):
        return "".join(self._pieces[i] for i in ids)

    def to_str(self):
        return self._serialized


PIECES = {
    1: "hello",
    2: "\ufffd\ufffd",          # entirely replacement chars (a pure byte fragment)
    3: "\ufffda\ufffd",         # MIXED: the exact shape the Session 7 regression let through
}
ENCODINGS = {"hello": [1], "mixed": [1, 2, 3]}


@pytest.fixture
def adapter():
    return HuggingFaceAdapter(_FakeHFTokenizer(ENCODINGS, PIECES))


def test_encode_returns_ids(adapter):
    assert adapter.encode("mixed") == [1, 2, 3]


def test_encode_rejects_non_str(adapter):
    with pytest.raises(TypeError):
        adapter.encode(123)


def test_decode_joins_pieces(adapter):
    assert adapter.decode([1]) == "hello"


def test_clean_token_gets_real_string(adapter):
    [token] = adapter.get_token_strings("hello")
    assert token.token_id == 1
    assert token.text == "hello"


def test_fully_corrupted_token_is_honest_null(adapter):
    tokens = adapter.get_token_strings("mixed")
    assert tokens[1].text is None


def test_partially_corrupted_token_is_honest_null(adapter):
    """Regression test for the Session 7 bug. Under the broken
    `single.strip('\\ufffd') == ''` heuristic, token 3 came back as the
    fabricated string '\\ufffda\\ufffd'. This test kills that mutant."""
    tokens = adapter.get_token_strings("mixed")
    assert tokens[2].text is None


def test_version_is_stable_for_identical_tokenizer_state():
    a = HuggingFaceAdapter(_FakeHFTokenizer(ENCODINGS, PIECES, serialized="same"))
    b = HuggingFaceAdapter(_FakeHFTokenizer(ENCODINGS, PIECES, serialized="same"))
    assert a.version == b.version
    assert len(a.version) == 16


def test_version_changes_when_tokenizer_content_changes():
    """Hub-side drift must change the version, and therefore the cache key."""
    old = HuggingFaceAdapter(_FakeHFTokenizer(ENCODINGS, PIECES, serialized="vocab-v1"))
    new = HuggingFaceAdapter(_FakeHFTokenizer(ENCODINGS, PIECES, serialized="vocab-v2"))
    assert old.version != new.version