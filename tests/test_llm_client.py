from types import SimpleNamespace

import pytest

import core.llm_client as llm_client
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


class FakeOpenAIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.calls = []
        self.response = '{"ok": true}'
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response))]
        )


class RetryOpenAIClient(FakeOpenAIClient):
    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise RuntimeError("Unsupported parameter: response_format")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )


class InvalidJSONOpenAIClient(FakeOpenAIClient):
    def __init__(self, base_url, api_key):
        super().__init__(base_url, api_key)
        self.response = "not json"


def make_error_client(error):
    class ErrorOpenAIClient(FakeOpenAIClient):
        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            raise error

    return ErrorOpenAIClient


def test_openai_key_alone_does_not_authenticate_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    constructed = []

    class IsolationClient:
        def __init__(self, base_url, api_key):
            constructed.append((base_url, api_key))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: None)
            )

    monkeypatch.setattr(llm_client, "OpenAI", IsolationClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "",
            "openrouter_model": "deepseek/deepseek-v4.1-flash",
        }
    )
    assert llm.openrouter_key == ""
    assert llm.openai_key == "fake-openai-key"

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        llm._init_provider("openrouter")

    assert constructed == []


def test_openai_provider_still_uses_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    constructed = []

    class OpenAIClient:
        def __init__(self, base_url, api_key):
            constructed.append((base_url, api_key))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: None)
            )

    monkeypatch.setattr(llm_client, "OpenAI", OpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "",
            "openai_base_url": "https://api.openai.com/v1",
        }
    )
    llm._init_provider("openai")

    assert constructed == [("https://api.openai.com/v1", "fake-openai-key")]


def test_openrouter_supported_json_mode_sends_response_format(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_json_mode": "supported",
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_openrouter_chat_json_explicit_selection_uses_selected_model(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_model": "deepseek/deepseek-v4.1-flash-1017",
            "openrouter_api_key": "fake-key",
            "openrouter_json_mode": "supported",
            "llm_fallback_provider": "ollama",
            "llm_provider_explicit": True,
            "model_explicit": True,
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert client.calls[0]["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_openrouter_unsupported_json_mode_omits_response_format(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_json_mode": "unsupported",
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert "response_format" not in client.calls[0]


def test_openrouter_unknown_json_mode_omits_response_format(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert "response_format" not in client.calls[0]


def test_openrouter_supported_parameters_metadata_enables_json_mode(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_supported_parameters": ["response_format", "temperature"],
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_openrouter_unsupported_parameters_metadata_omits_json_mode(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_supported_parameters": ["temperature"],
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert "response_format" not in client.calls[0]


def test_openrouter_retries_once_without_response_format(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", RetryOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_json_mode": "supported",
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw == '{"ok": true}'
    client = llm._clients["openrouter"]
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in client.calls[1]
    assert client.calls[0]["model"] == client.calls[1]["model"]


@pytest.mark.parametrize(
    "error",
    [
        "401 authentication failed",
        "402 billing error",
        "unrelated provider error",
    ],
)
def test_openrouter_non_unsupported_errors_do_not_retry(monkeypatch, error):
    monkeypatch.setattr(llm_client, "OpenAI", make_error_client(RuntimeError(error)))
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
            "openrouter_json_mode": "supported",
            "llm_fallback_provider": "openrouter",
        }
    )

    raw = llm.chat_json("system", "user")
    assert raw is None
    client = llm._clients["openrouter"]
    assert len(client.calls) == 1


def test_openrouter_invalid_json_output_raises_clear_error(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", InvalidJSONOpenAIClient)
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "fake-key",
        }
    )

    with pytest.raises(ValueError, match="Invalid JSON output"):
        llm.chat_json("system", "user")
