import pytest

from llmcost import engine


@pytest.fixture(autouse=True)
def _reset_warned_pairs():
    """Warn-once state is process-global; isolate it per test."""
    engine._WARNED_PAIRS.clear()
    yield
    engine._WARNED_PAIRS.clear()
