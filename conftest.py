"""
Repo-wide test setup.

Module-level code here runs during pytest's collection phase, before any
test module (and therefore api.app.main / api.app.config) is imported.
"""
import os

import pytest

# Required by api.app.config.Settings. setdefault: a real environment
# variable (e.g. set by CI) still wins over this test default.
os.environ.setdefault("CORS_ALLOW_ORIGINS", "http://localhost:5173")


def _cl100k_reachable() -> bool:
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Single source of truth for what 'real_tokenizer' tests do offline.

    Locally: skip them if the vocabulary can't be loaded.
    In CI (REQUIRE_REAL_BACKEND=1): never skip -- a skip there would be a
    silent pass, so an unreachable host fails the run loudly instead.
    """
    marked = [i for i in items if i.get_closest_marker("real_tokenizer")]
    if not marked or _cl100k_reachable():
        return

    if os.getenv("REQUIRE_REAL_BACKEND") == "1":
        pytest.exit(
            "REQUIRE_REAL_BACKEND=1 but cl100k_base is unreachable -- "
            "real-backend tests cannot be verified. Rerun when the host is back.",
            returncode=1,
        )

    skip = pytest.mark.skip(
        reason="cl100k_base unreachable (set REQUIRE_REAL_BACKEND=1 to fail instead of skip)"
    )
    for item in marked:
        item.add_marker(skip)