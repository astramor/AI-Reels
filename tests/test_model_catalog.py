import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import check_models
from core.model_catalog import (
    CatalogError,
    get_models,
    list_gemini_models,
    list_ollama_models,
    list_openrouter_models,
    search_models,
)


class FakePager:
    def __init__(self, pages):
        self.pages = pages

    def __iter__(self):
        for page in self.pages:
            yield from page


class FakeGeminiModels:
    def __init__(self, pages):
        self.pages = pages

    def list(self, config=None):
        return FakePager(self.pages)


class FakeGeminiClient:
    def __init__(self, pages):
        self.models = FakeGeminiModels(pages)


def gemini_model(name, display_name, actions, input_limit=None, output_limit=None):
    return SimpleNamespace(
        name=name,
        display_name=display_name,
        supported_actions=actions,
        input_token_limit=input_limit,
        output_token_limit=output_limit,
    )


def openrouter_entry(
    model_id,
    name,
    input_modalities,
    output_modalities,
    context_length=100000,
    prompt="0.0000003",
    completion="0.0000012",
    max_completion_tokens=None,
    supported_parameters=None,
):
    entry = {
        "id": model_id,
        "name": name,
        "context_length": context_length,
        "architecture": {
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
        },
        "pricing": {"prompt": prompt, "completion": completion},
        "top_provider": {"max_completion_tokens": max_completion_tokens},
    }
    if supported_parameters is not None:
        entry["supported_parameters"] = supported_parameters
    return entry


def test_gemini_pagination_and_generation_method_filtering():
    pages = [
        [
            gemini_model(
                "publishers/google/models/gemini-2.0-flash",
                "Gemini 2.0 Flash",
                ["generateContent"],
                1_048_576,
                8192,
            )
        ],
        [
            gemini_model(
                "publishers/google/models/gemini-embedding-001",
                "Gemini Embedding",
                ["embedContent"],
            ),
            gemini_model(
                "publishers/google/models/gemini-unknown",
                "Gemini Unknown",
                None,
            ),
        ],
    ]

    models = list_gemini_models(FakeGeminiClient(pages))
    assert [m["id"] for m in models] == ["gemini-2.0-flash", "gemini-unknown"]
    assert models[0]["supports_generate_content"] is True
    assert models[1]["supports_generate_content"] is None
    assert models[0]["input_token_limit"] == 1_048_576


def test_openrouter_catalog_filtering_and_search():
    entries = [
        openrouter_entry(
            "deepseek/deepseek-v4.1-flash",
            "DeepSeek: DeepSeek V4.1 Flash",
            ["text", "image"],
            ["text"],
            1_048_576,
            "0.0000003",
            "0.0000012",
        ),
        openrouter_entry(
            "google/gemini-3.1-flash-image",
            "Gemini Flash Image",
            ["image", "text"],
            ["image", "text"],
            131072,
        ),
        openrouter_entry(
            "deepseek/deepseek-v4.1-flash-1017",
            "DeepSeek: DeepSeek V4.1 Flash 1017",
            ["text", "image"],
            ["text"],
            1_048_576,
        ),
        {
            "id": "unknown/model",
            "name": "Unknown Model",
            "context_length": None,
            "architecture": {},
            "pricing": {},
        },
        openrouter_entry(
            "image/only",
            "Image Only",
            ["image"],
            ["text"],
        ),
    ]

    models = list_openrouter_models(lambda: {"data": entries})
    ids = [m["id"] for m in models]
    assert "deepseek/deepseek-v4.1-flash" in ids
    assert "deepseek/deepseek-v4.1-flash-1017" in ids
    assert "unknown/model" in ids
    assert "google/gemini-3.1-flash-image" not in ids
    assert "image/only" not in ids

    deepseek = search_models(models, "DEEPSEEK")
    assert {m["id"] for m in deepseek} == {
        "deepseek/deepseek-v4.1-flash",
        "deepseek/deepseek-v4.1-flash-1017",
    }
    flash = search_models(models, "flash")
    assert "deepseek/deepseek-v4.1-flash" in {m["id"] for m in flash}

    flash_model = next(m for m in models if m["id"] == "deepseek/deepseek-v4.1-flash")
    assert flash_model["prompt_usd_per_million_tokens"] == pytest.approx(0.3)
    assert flash_model["completion_usd_per_million_tokens"] == pytest.approx(1.2)
    assert flash_model["pricing_unit"] == "USD per 1M tokens"
    assert flash_model["compatibility"] == "text-compatible"

    unknown = next(m for m in models if m["id"] == "unknown/model")
    assert unknown["compatibility"] == "unknown"
    assert unknown["prompt_usd_per_million_tokens"] is None


def test_openrouter_json_mode_capability_metadata():
    entries = [
        openrouter_entry(
            "supported/model",
            "Supported",
            ["text"],
            ["text"],
            supported_parameters=["response_format", "temperature"],
        ),
        openrouter_entry(
            "unsupported/model",
            "Unsupported",
            ["text"],
            ["text"],
            supported_parameters=["temperature"],
        ),
        {
            "id": "unknown/model",
            "name": "Unknown",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        },
    ]
    models = list_openrouter_models(lambda: {"data": entries})
    by_id = {m["id"]: m for m in models}
    assert by_id["supported/model"]["json_mode_supported"] is True
    assert by_id["unsupported/model"]["json_mode_supported"] is False
    assert by_id["unknown/model"]["json_mode_supported"] is None


def test_openrouter_malformed_response():
    with pytest.raises(CatalogError):
        list_openrouter_models(lambda: {"data": "not-a-list"})
    with pytest.raises(CatalogError):
        list_openrouter_models(lambda: {"other": []})


def test_ollama_local_model_listing():
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="qwen2.5:14b-instruct-q6_K",
                    size=123,
                    modified_at="2026-01-01",
                    details=SimpleNamespace(
                        family="qwen2",
                        parameter_size="14B",
                        quantization_level="Q6_K",
                    ),
                )
            ]
        )
    )
    models = list_ollama_models(client)
    assert models[0]["id"] == "qwen2.5:14b-instruct-q6_K"
    assert models[0]["family"] == "qwen2"
    assert models[0]["quantization_level"] == "Q6_K"


def test_ollama_modified_at_is_json_serializable():
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="qwen2.5:14b-instruct-q6_K",
                    size=123,
                    modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    details=SimpleNamespace(
                        family="qwen2",
                        parameter_size="14B",
                        quantization_level="Q6_K",
                    ),
                )
            ]
        )
    )
    models = list_ollama_models(client)
    assert models[0]["modified_at"] == "2026-01-01T00:00:00+00:00"
    json.dumps(models)


def test_cache_refresh_stale_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    cache_path = tmp_path / "model_cache.json"

    def fresh_fetcher():
        return {
            "data": [
                openrouter_entry(
                    "deepseek/deepseek-v4.1-flash",
                    "DeepSeek: DeepSeek V4.1 Flash",
                    ["text"],
                    ["text"],
                )
            ]
        }

    models, metadata = get_models(
        "openrouter", refresh=True, cache_path=cache_path, fetcher=fresh_fetcher
    )
    assert metadata["stale"] is False
    assert models[0]["id"] == "deepseek/deepseek-v4.1-flash"

    cache_text = cache_path.read_text(encoding="utf-8")
    assert "secret-test-key" not in cache_text
    assert "openrouter_api_key" not in cache_text

    data = json.loads(cache_text)
    data["openrouter"]["fetched_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=30)
    ).isoformat()
    cache_path.write_text(json.dumps(data), encoding="utf-8")

    def failing_fetcher():
        raise CatalogError("offline")

    stale_models, stale_metadata = get_models(
        "openrouter",
        refresh=False,
        cache_path=cache_path,
        max_age_hours=24,
        fetcher=failing_fetcher,
    )
    assert stale_metadata["stale"] is True
    assert stale_models == models

    missing_cache = tmp_path / "missing.json"
    with pytest.raises(CatalogError):
        get_models(
            "openrouter",
            refresh=False,
            cache_path=missing_cache,
            fetcher=failing_fetcher,
        )

    refreshed_models, refreshed_metadata = get_models(
        "openrouter", refresh=True, cache_path=cache_path, fetcher=fresh_fetcher
    )
    assert refreshed_metadata["stale"] is False
    assert refreshed_models == models


def test_check_models_select_unknown_model_offline(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise CatalogError("catalog unavailable")

    monkeypatch.setattr(check_models, "get_models", unavailable)
    code = check_models.main(
        ["--provider", "openrouter", "--select", "some/unknown-model"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "some/unknown-model" in captured.out
    assert 'openrouter_model: "some/unknown-model"' in captured.out
    assert "explicit model ID is still usable" in captured.err


def test_check_models_select_model_not_in_catalog_warns(monkeypatch, capsys):
    monkeypatch.setattr(
        check_models,
        "get_models",
        lambda *args, **kwargs: (
            [{"id": "deepseek/deepseek-v4.1-flash"}],
            {"fetched_at": "2026-11-12T00:00:00+00:00", "stale": False, "age_hours": 1.0},
        ),
    )
    code = check_models.main(
        ["--provider", "openrouter", "--select", "some/unknown-model"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "some/unknown-model" in captured.out
    assert "not in the catalog" in captured.err


def test_check_models_json_output_ollama_datetime(monkeypatch, capsys):
    monkeypatch.setattr(
        check_models,
        "get_models",
        lambda *args, **kwargs: (
            [
                {
                    "id": "qwen2.5:14b-instruct-q6_K",
                    "modified_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                }
            ],
            {"fetched_at": "2026-11-12T00:00:00+00:00", "stale": False, "age_hours": 1.0},
        ),
    )
    code = check_models.main(["--provider", "ollama", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "2026-01-01" in payload["models"][0]["modified_at"]


def test_check_models_gemini_missing_key_diagnostic(monkeypatch, capsys):
    def missing_key():
        raise CatalogError("GEMINI_API_KEY fehlt")

    def unavailable(*args, **kwargs):
        raise CatalogError("Gemini client is required")

    monkeypatch.setattr(check_models, "_gemini_client", missing_key)
    monkeypatch.setattr(check_models, "get_models", unavailable)
    code = check_models.main(["--provider", "gemini"])
    assert code == 1
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY fehlt" in captured.err


def test_check_models_select_refresh_missing_gemini_key_still_selects(monkeypatch, capsys):
    def missing_key():
        raise CatalogError("GEMINI_API_KEY fehlt")

    def unavailable(*args, **kwargs):
        raise CatalogError("Gemini client is required")

    monkeypatch.setattr(check_models, "_gemini_client", missing_key)
    monkeypatch.setattr(check_models, "get_models", unavailable)
    code = check_models.main(
        ["--provider", "gemini", "--select", "gemini-2.0-flash", "--refresh"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "gemini-2.0-flash" in captured.out
    assert "GEMINI_API_KEY fehlt" in captured.err


def test_load_openrouter_key_does_not_reuse_openai_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    key, source = check_models.load_openrouter_key()
    assert key == ""
    assert source is None


def test_check_models_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        check_models,
        "get_models",
        lambda *args, **kwargs: (
            [{"id": "deepseek/deepseek-v4.1-flash", "name": "DeepSeek Flash"}],
            {"fetched_at": "2026-11-12T00:00:00+00:00", "stale": False, "age_hours": 1.0},
        ),
    )
    code = check_models.main(["--provider", "openrouter", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "openrouter"
    assert payload["models"][0]["id"] == "deepseek/deepseek-v4.1-flash"
    assert payload["cache"]["stale"] is False


def test_save_provider_model_preserves_other_lines(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "# keep this comment\n"
        'llm_provider: "openai"\n'
        'openai_model: "old-model"\n'
        'openai_base_url: "http://127.0.0.1:18080/v1"\n',
        encoding="utf-8",
    )

    check_models.save_provider_model(
        "openrouter", "deepseek/deepseek-v4.1-flash", str(config_path)
    )
    text = config_path.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert 'llm_provider: "openrouter"' in text
    assert 'openrouter_model: "deepseek/deepseek-v4.1-flash"' in text
    assert 'openai_model: "old-model"' in text
    assert 'openai_base_url: "http://127.0.0.1:18080/v1"' in text
