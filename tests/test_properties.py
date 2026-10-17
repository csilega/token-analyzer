"""
Property-based tests (Hypothesis). These generate hundreds of random strings
per run, including emoji, mixed scripts, and rare Unicode, and check invariants
that must hold for ANY input, not just inputs we thought to write by hand.

Uses the real cl100k_base encoding, so this is integration-tier: it skips
cleanly when the encoding can't be fetched (same pattern as
test_tiktoken_adapter.py).
"""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from token_analyzer.registry import get_adapter


def _cl100k_available() -> bool:
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cl100k_available(), reason="cl100k_base not reachable"
)

# conftest.py's autouse fixture (lru_cache reset) is function-scoped, so it runs
# once per test, not once per generated example. That's harmless here, but
# Hypothesis warns about it by default, so we acknowledge it explicitly.
PROPERTY_SETTINGS = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@PROPERTY_SETTINGS
@given(st.text())
def test_round_trip_is_lossless(text):
    """Property 1: byte-level BPE never loses information."""
    adapter = get_adapter("gpt-4")
    assert adapter.decode(adapter.encode(text)) == text


@PROPERTY_SETTINGS
@given(st.text())
def test_token_strings_reassemble_to_exact_input(text):
    """Property 2: honest-null. Real strings must reassemble to the exact input."""
    adapter = get_adapter("gpt-4")
    tokens = adapter.get_token_strings(text)

    # Token IDs must line up exactly with encode(): same tokens, same order.
    assert [t.token_id for t in tokens] == adapter.encode(text)

    # Only when every token claims a real string can we demand exact reassembly.
    if all(t.text is not None for t in tokens):
        assert "".join(t.text for t in tokens) == text