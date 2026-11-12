from types import SimpleNamespace

import pytest

from build_highlight_spans import apply_cli_model_overrides
from check_models import load_gemini_key
from core.config import Settings, load_yaml_defaults
from core.llm_client import UnifiedLLM
from domains.llm.summarizer import SermonSummarizer


def test_build_cli_defaults_do_not_override_config():
    config = {
        "llm_model": "config-model",
        "gemini_model": "config-gemini",
        "gemini_api_key": "config-key",
    }
    args = SimpleNamespace(llm_model=None, gemini_model=None, gemini_api_key=None)
    result = apply_cli_model_overrides(dict(config), args)
    assert result == config


def test_build_cli_explicit_overrides_are_applied():
    args = SimpleNamespace(
        llm_model="cli-model",
        gemini_model="cli-gemini",
        gemini_api_key="fake-key",
    )
    result = apply_cli_model_overrides({}, args)
    assert result["llm_model"] == "cli-model"
    assert result["gemini_model"] == "cli-gemini"
    assert result["gemini_api_key"] == "fake-key"
    assert result["gemini_api_key_explicit"] is True


def test_build_cli_explicit_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")
    args = SimpleNamespace(
        llm_model=None,
        gemini_model=None,
        gemini_api_key="cli-gemini",
    )
    config = apply_cli_model_overrides({}, args)
    llm = UnifiedLLM({"llm_provider": "gemini", **config})
    assert llm.gemini_key == "cli-gemini"


@pytest.mark.parametrize(
    "provider,config,expected",
    [
        ("openai", {"llm_provider": "openai", "openai_model": "openai-model"}, "openai-model"),
        ("openrouter", {"llm_provider": "openrouter", "openrouter_model": "openrouter-model"}, "openrouter-model"),
        ("gemini", {"llm_provider": "gemini", "gemini_model": "gemini-model"}, "gemini-model"),
        ("ollama", {"llm_provider": "ollama", "ollama_model": "ollama-model"}, "ollama-model"),
        ("llama-server", {"llm_provider": "llama-server", "llama_server_model": "llama-model"}, "llama-model"),
    ],
)
def test_provider_specific_model_resolution(provider, config, expected):
    llm = UnifiedLLM(config)
    assert llm.provider == provider
    assert llm.model == expected


def test_sermon_summarizer_uses_settings_provider_and_model():
    settings = Settings(
        llm_provider="openai",
        openai_model="openai-model",
        openai_api_key="fake-openai-key",
    )
    summarizer = SermonSummarizer(settings)
    assert summarizer.llm.provider == "openai"
    assert summarizer.model_name == "openai-model"


def test_sermon_summarizer_effective_key_uses_environment_first(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    settings = Settings(
        llm_provider="openai",
        openai_model="openai-model",
        openai_api_key="yaml-key",
    )
    summarizer = SermonSummarizer(settings)
    assert summarizer.llm.openai_key == "env-key"


def test_sermon_summarizer_effective_key_uses_yaml_when_environment_absent(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        llm_provider="openai",
        openai_model="openai-model",
        openai_api_key="yaml-key",
    )
    summarizer = SermonSummarizer(settings)
    assert summarizer.llm.openai_key == "yaml-key"


def test_empty_or_null_yaml_api_keys_do_not_erase_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "openai_api_key: ''\ngemini_api_key: null\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")

    defaults = load_yaml_defaults()
    assert "openai_api_key" not in defaults
    assert "gemini_api_key" not in defaults

    settings = Settings(**defaults)
    assert settings.openai_api_key == "env-openai"
    assert settings.gemini_api_key.get_secret_value() == "env-gemini"


def test_environment_api_keys_win_over_non_empty_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "openai_api_key: yaml-openai\ngemini_api_key: yaml-gemini\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")

    defaults = load_yaml_defaults()
    assert "openai_api_key" not in defaults
    assert "gemini_api_key" not in defaults

    settings = Settings(**defaults)
    assert settings.openai_api_key == "env-openai"
    assert settings.gemini_api_key.get_secret_value() == "env-gemini"


def test_non_empty_yaml_api_keys_are_used_when_environment_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "openai_api_key: yaml-openai\ngemini_api_key: yaml-gemini\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    defaults = load_yaml_defaults()
    assert defaults["openai_api_key"] == "yaml-openai"
    assert defaults["gemini_api_key"] == "yaml-gemini"

    settings = Settings(**defaults)
    assert settings.openai_api_key == "yaml-openai"
    assert settings.gemini_api_key.get_secret_value() == "yaml-gemini"


def test_check_models_uses_environment_with_empty_or_null_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "gemini_api_key: ''\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")

    key, source = load_gemini_key()
    assert key == "fake-gemini"
    assert source == "GEMINI_API_KEY"


def test_check_models_environment_wins_over_non_empty_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "gemini_api_key: config-gemini\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")

    key, source = load_gemini_key()
    assert key == "fake-gemini"
    assert source == "GEMINI_API_KEY"


def test_check_models_yaml_fallback_when_environment_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "gemini_api_key: config-gemini\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    key, source = load_gemini_key()
    assert key == "config-gemini"
    assert source == "config.yaml:gemini_api_key"
