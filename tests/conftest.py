"""
Shared pytest fixtures.
"""

import pytest

from token_analyzer.registry import get_adapter


@pytest.fixture(autouse=True)
def clear_adapter_cache():
    """
    get_adapter() is wrapped in @lru_cache, which is module-level state
    that persists across the whole test run. Without clearing it before
    each test, one test's cached adapter can silently leak into another
    test's assertions -- a real form of test pollution. autouse=True means
    every test gets this for free with no explicit opt-in required.
    """
    get_adapter.cache_clear()
    yield
    get_adapter.cache_clear()