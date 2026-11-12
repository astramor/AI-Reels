import pytest

from core.llm_client import UnifiedLLM


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", True),
        ("openrouter", True),
        ("gemini", True),
        ("gemini-api", True),
        ("ollama", False),
        ("llama-server", False),
    ],
)
def test_is_cloud_llm(provider, expected):
    llm = UnifiedLLM({"llm_provider": provider})
    assert llm.is_cloud_llm is expected


class FailingPrimaryLLM(UnifiedLLM):
    def _chat_with_provider(
        self,
        provider,
        sys_p,
        user_p,
        temp=0.1,
        json_mode=False,
        max_tokens=None,
    ):
        if provider == self.provider:
            raise RuntimeError("primary provider failed")
        return "fallback response"


def test_chat_uses_fallback_provider_on_error():
    llm = FailingPrimaryLLM(
        {
            "llm_provider": "openai",
            "llm_fallback_provider": "ollama",
            "openai_api_key": "testenv",
            "ollama_model": "test-model",
        }
    )
    assert llm.chat("system", "user") == "fallback response"


def test_empty_yaml_openai_key_uses_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "testenv")
    llm = UnifiedLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "",
            "openai_base_url": "http://127.0.0.1:18080/v1",
        }
    )
    assert llm.openai_key == "testenv"


def test_empty_yaml_openrouter_key_uses_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "testenv")
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "",
            "openai_api_key": "",
        }
    )
    assert llm.openrouter_key == "testenv"


def test_empty_yaml_gemini_key_uses_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "testenv")
    llm = UnifiedLLM(
        {
            "llm_provider": "gemini",
            "gemini_api_key": "",
        }
    )
    assert llm.gemini_key == "testenv"


def test_env_key_wins_over_non_empty_yaml(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm = UnifiedLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "yaml-key",
        }
    )
    assert llm.openai_key == "env-key"


def test_non_empty_yaml_key_is_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = UnifiedLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "yaml-key",
        }
    )
    assert llm.openai_key == "yaml-key"


def test_explicit_config_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm = UnifiedLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "cli-key",
            "openai_api_key_explicit": True,
        }
    )
    assert llm.openai_key == "cli-key"
