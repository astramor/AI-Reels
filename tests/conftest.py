import os

import pytest


_TEST_API_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
)


def _ensure_test_env() -> None:
    """Set test-only API keys before test modules import core.config."""
    for key in _TEST_API_KEYS:
        os.environ.setdefault(key, "testenv")


_ensure_test_env()


@pytest.fixture(autouse=True)
def test_llm_env(monkeypatch):
    """Make LLM tests independent of manually exported API keys."""
    for key in _TEST_API_KEYS:
        if not os.environ.get(key):
            monkeypatch.setenv(key, "testenv")
