from types import SimpleNamespace

import pytest

import batch_reels
from build_highlight_spans import apply_cli_model_overrides, llm_generate_posts
from core.config import Settings
from core.llm_client import UnifiedLLM
from domains.llm.summarizer import SermonSummarizer
from srt_summarizer_local import _map_model_to_provider_field


def test_openrouter_model_and_base_url_resolution():
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_model": "deepseek/deepseek-v4.1-flash",
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "openrouter_api_key": "fake-openrouter-key",
        }
    )
    assert llm.provider == "openrouter"
    assert llm.model == "deepseek/deepseek-v4.1-flash"
    assert llm.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert llm.is_openrouter is True


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openrouter", "deepseek/deepseek-v4.1-flash"),
        ("gemini", "gemini-2.0-flash"),
        ("ollama", "qwen2.5:14b-instruct-q6_K"),
    ],
)
def test_switching_provider_does_not_reuse_llm_model(provider, expected):
    llm = UnifiedLLM({"llm_provider": provider, "llm_model": "ds41"})
    assert llm.model == expected


def test_openai_legacy_llm_model_fallback_is_preserved():
    llm = UnifiedLLM({"llm_provider": "openai", "llm_model": "ds41"})
    assert llm.model == "ds41"


def test_openrouter_explicit_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    llm = UnifiedLLM(
        {
            "llm_provider": "openrouter",
            "openrouter_api_key": "cli-key",
            "openrouter_api_key_explicit": True,
        }
    )
    assert llm.openrouter_key == "cli-key"


def test_openrouter_environment_key_wins_over_yaml(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    llm = UnifiedLLM(
        {"llm_provider": "openrouter", "openrouter_api_key": "yaml-key"}
    )
    assert llm.openrouter_key == "env-key"


def test_openrouter_yaml_key_used_when_environment_absent(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = UnifiedLLM(
        {"llm_provider": "openrouter", "openrouter_api_key": "yaml-key"}
    )
    assert llm.openrouter_key == "yaml-key"


def test_openrouter_empty_yaml_key_uses_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    llm = UnifiedLLM(
        {"llm_provider": "openrouter", "openrouter_api_key": ""}
    )
    assert llm.openrouter_key == "env-key"


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


def test_explicit_provider_failure_does_not_silently_fall_back():
    llm = FailingPrimaryLLM(
        {
            "llm_provider": "openrouter",
            "llm_provider_explicit": True,
            "openrouter_api_key": "fake-key",
            "llm_fallback_provider": "ollama",
            "ollama_model": "test-model",
        }
    )
    with pytest.raises(RuntimeError):
        llm.chat("system", "user")


def test_explicit_provider_failure_with_same_fallback_still_raises():
    llm = FailingPrimaryLLM(
        {
            "llm_provider": "ollama",
            "llm_provider_explicit": True,
            "ollama_model": "test-model",
            "llm_fallback_provider": "ollama",
        }
    )
    with pytest.raises(RuntimeError):
        llm.chat("system", "user")


def test_cloud_fallback_is_logged(capsys):
    llm = FailingPrimaryLLM(
        {
            "llm_provider": "openai",
            "openai_api_key": "fake-key",
            "llm_fallback_provider": "gemini",
            "gemini_api_key": "fake-key",
            "gemini_model": "gemini-2.0-flash",
        }
    )
    result = llm.chat("system", "user")
    assert result == "fallback response"
    assert "Switching to cloud fallback provider: gemini" in capsys.readouterr().out


def test_explicit_model_failure_does_not_silently_fall_back():
    llm = FailingPrimaryLLM(
        {
            "llm_provider": "openrouter",
            "model_explicit": True,
            "openrouter_model": "deepseek/deepseek-v4.1-flash",
            "openrouter_api_key": "fake-key",
            "llm_fallback_provider": "ollama",
            "ollama_model": "test-model",
        }
    )
    with pytest.raises(RuntimeError):
        llm.chat("system", "user")


def test_sermon_summarizer_openrouter_selection():
    settings = Settings(
        llm_provider="openrouter",
        openrouter_model="deepseek/deepseek-v4.1-flash",
        openrouter_api_key="fake-openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    summarizer = SermonSummarizer(settings)
    assert summarizer.provider == "openrouter"
    assert summarizer.model_name == "deepseek/deepseek-v4.1-flash"


def test_build_cli_openrouter_model_mapping():
    args = SimpleNamespace(
        llm_provider="openrouter",
        llm_model="deepseek/deepseek-v4.1-flash",
        gemini_model=None,
        gemini_api_key=None,
        openrouter_api_key="fake-openrouter-key",
    )
    result = apply_cli_model_overrides({}, args)
    assert result["llm_provider"] == "openrouter"
    assert result["llm_provider_explicit"] is True
    assert result["model_explicit"] is True
    assert result["openrouter_model"] == "deepseek/deepseek-v4.1-flash"
    assert result["llm_model"] == "deepseek/deepseek-v4.1-flash"
    assert result["openrouter_api_key"] == "fake-openrouter-key"
    assert result["openrouter_api_key_explicit"] is True


def test_cloud_provider_uses_cloud_caption_prompts():
    captured = []

    class FakeLLM:
        is_cloud_llm = True

        def chat(self, sys_p, user_p, temp=0.1):
            captured.append((sys_p, user_p))
            return "HEADLINE: Test\nBODY: Body"

    info = {
        "date": "2026-11-16",
        "prediger": "Test",
        "thema": "Test",
        "specials": "",
    }
    llm_generate_posts(FakeLLM(), "Titel", "Kontext", info)
    assert captured[0][0] == "Du bist ein eleganter Copywriter."
    assert "Montagmorgen" in captured[0][1]


def test_build_cli_gemini_model_sets_model_explicit():
    args = SimpleNamespace(
        llm_provider="gemini",
        llm_model=None,
        gemini_model="gemini-2.0-flash",
        gemini_api_key=None,
        openrouter_api_key=None,
    )
    result = apply_cli_model_overrides({}, args)
    assert result["gemini_model"] == "gemini-2.0-flash"
    assert result["model_explicit"] is True


def test_local_model_mapping_openrouter():
    conf = {}
    _map_model_to_provider_field(conf, "openrouter", "deepseek/deepseek-v4.1-flash")
    assert conf["openrouter_model"] == "deepseek/deepseek-v4.1-flash"
    assert conf["llm_model"] == "deepseek/deepseek-v4.1-flash"


def test_execute_summarizer_handoff_passes_provider_model_and_key(tmp_path, monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append([str(part) for part in cmd])
        return 0

    monkeypatch.setattr(batch_reels, "run", fake_run)
    args = SimpleNamespace(
        summarizer_script="srt_summarizer_local.py",
        fix_spelling=False,
        auto_sermon=False,
        min_sermon_minutes=10,
        llm_provider="openrouter",
        gemini_model=None,
        llm_model="deepseek/deepseek-v4.1-flash",
        gemini_api_key=None,
        openrouter_api_key="fake-openrouter-key",
        sermon_start=None,
        sermon_end=None,
        max_highlights=8,
        strict_snap_tol=8,
        dry_run=True,
    )
    batch_reels.execute_summarizer(
        tmp_path / "sermon.srt",
        tmp_path / "summ",
        tmp_path / "predigt.srt",
        {},
        args,
    )
    command = captured[0]
    assert "--llm-provider" in command
    assert command[command.index("--llm-provider") + 1] == "openrouter"
    assert "--model" in command
    assert command[command.index("--model") + 1] == "deepseek/deepseek-v4.1-flash"
    assert "--openrouter-api-key" in command
    assert command[command.index("--openrouter-api-key") + 1] == "fake-openrouter-key"


def test_execute_span_builder_handoff_passes_provider_model_and_key(tmp_path, monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append([str(part) for part in cmd])
        return 0

    monkeypatch.setattr(batch_reels, "run", fake_run)
    args = SimpleNamespace(
        spans_script="build_highlight_spans.py",
        target_count=6,
        span_min=15,
        span_max=30,
        title_min=3,
        title_max=10,
        llm_provider="openrouter",
        llm_model="deepseek/deepseek-v4.1-flash",
        gemini_model=None,
        gemini_api_key=None,
        openrouter_api_key="fake-openrouter-key",
        llm_titles=True,
        church_yaml=None,
        dry_run=True,
    )
    batch_reels.execute_span_builder(
        tmp_path / "highlights.md",
        tmp_path / "predigt.srt",
        tmp_path / "spans.md",
        tmp_path / "spans.json",
        args,
    )
    command = captured[0]
    assert "--llm-provider" in command
    assert command[command.index("--llm-provider") + 1] == "openrouter"
    assert "--llm-model" in command
    assert command[command.index("--llm-model") + 1] == "deepseek/deepseek-v4.1-flash"
    assert "--openrouter-api-key" in command
    assert command[command.index("--openrouter-api-key") + 1] == "fake-openrouter-key"
