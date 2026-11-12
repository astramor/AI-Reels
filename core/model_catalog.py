#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core/model_catalog.py
---------------------
On-demand model discovery for Ollama, Gemini, and OpenRouter.

The catalog is fetched only when the CLI asks for it. Results are cached in a
bounded JSON file with a timestamp; credentials are never stored in the cache.
"""

import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_PATH = pathlib.Path(".cache/model_catalog.json")
DEFAULT_MAX_AGE_HOURS = 24.0
DEFAULT_TIMEOUT = 20.0

SUPPORTED_PROVIDERS = ("ollama", "gemini", "openrouter")


class CatalogError(RuntimeError):
    """Raised when a model catalog cannot be loaded."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _cache_file(cache_path: Optional[pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(cache_path or DEFAULT_CACHE_PATH)


def _load_cache(cache_path: Optional[pathlib.Path]) -> Dict[str, Any]:
    path = _cache_file(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(
    cache_path: Optional[pathlib.Path],
    provider: str,
    models: List[Dict[str, Any]],
    source: str,
) -> None:
    path = _cache_file(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_cache(path)
    data[provider] = {
        "fetched_at": _iso_now(),
        "source": source,
        "models": models,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _cache_metadata(entry: Dict[str, Any], max_age_hours: float) -> Dict[str, Any]:
    fetched_at = _parse_timestamp(str(entry.get("fetched_at", "")))
    age_hours = None
    stale = True
    if fetched_at is not None:
        age_hours = max(0.0, (_utc_now() - fetched_at).total_seconds() / 3600.0)
        stale = age_hours > max_age_hours
    return {
        "fetched_at": entry.get("fetched_at"),
        "source": entry.get("source"),
        "age_hours": age_hours,
        "stale": stale,
    }


def fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Fetch a public JSON catalog without sending credentials."""
    request = urllib.request.Request(url, headers={"User-Agent": "pythonreels-model-catalog"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise CatalogError(f"HTTP {exc.code} while fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise CatalogError(f"Network error while fetching {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CatalogError(f"Timeout while fetching {url}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Malformed JSON from {url}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"Unexpected catalog shape from {url}")
    return data


def _gemini_model_id(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return str(name).rsplit("/models/", 1)[-1]


def list_gemini_models(client: Any) -> List[Dict[str, Any]]:
    """List Gemini models supporting generateContent, including pagination."""
    if client is None:
        raise CatalogError("Gemini client is required")
    try:
        pager = client.models.list(config={"page_size": 100})
    except Exception as exc:
        raise CatalogError(f"Gemini model listing failed: {exc}") from exc

    models: List[Dict[str, Any]] = []
    try:
        for model in pager:
            name = getattr(model, "name", None)
            display_name = getattr(model, "display_name", None)
            actions = getattr(model, "supported_actions", None)
            supports_generate = None if actions is None else "generateContent" in actions
            if supports_generate is False:
                continue
            models.append(
                {
                    "id": _gemini_model_id(name),
                    "name": name,
                    "display_name": display_name,
                    "input_token_limit": getattr(model, "input_token_limit", None),
                    "output_token_limit": getattr(model, "output_token_limit", None),
                    "supported_actions": actions,
                    "supports_generate_content": supports_generate,
                }
            )
    except Exception as exc:
        raise CatalogError(f"Gemini pagination failed: {exc}") from exc
    return models


def _price_per_million(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) * 1_000_000.0
    except (TypeError, ValueError):
        return None


def list_openrouter_models(fetcher: Optional[Any] = None) -> List[Dict[str, Any]]:
    """List OpenRouter text-input/text-output chat-completion models."""
    data = fetcher() if fetcher else fetch_json(OPENROUTER_MODELS_URL)
    if not isinstance(data, dict):
        raise CatalogError("OpenRouter catalog response is not a JSON object")
    entries = data.get("data")
    if not isinstance(entries, list):
        raise CatalogError("OpenRouter catalog response has no 'data' list")

    models: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        architecture = entry.get("architecture") or {}
        input_modalities = architecture.get("input_modalities")
        output_modalities = architecture.get("output_modalities")

        if input_modalities is None or output_modalities is None:
            compatibility = "unknown"
        elif "text" in input_modalities and set(output_modalities) == {"text"}:
            compatibility = "text-compatible"
        else:
            compatibility = "unsupported"
        if compatibility == "unsupported":
            continue

        pricing = entry.get("pricing") or {}
        top_provider = entry.get("top_provider") or {}
        supported_parameters = entry.get("supported_parameters")
        json_mode_supported = None
        if isinstance(supported_parameters, list):
            json_mode_supported = "response_format" in supported_parameters
        models.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "context_length": entry.get("context_length"),
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "compatibility": compatibility,
                "supported_parameters": supported_parameters,
                "json_mode_supported": json_mode_supported,
                "prompt_usd_per_million_tokens": _price_per_million(pricing.get("prompt")),
                "completion_usd_per_million_tokens": _price_per_million(pricing.get("completion")),
                "max_completion_tokens": top_provider.get("max_completion_tokens"),
                "pricing_unit": "USD per 1M tokens",
            }
        )
    return models


def list_ollama_models(client: Any = None) -> List[Dict[str, Any]]:
    """List locally available Ollama models."""
    if client is None:
        try:
            import ollama
        except ImportError as exc:
            raise CatalogError("Ollama SDK is not installed") from exc
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = ollama.Client(host=host)
    try:
        response = client.list()
    except Exception as exc:
        raise CatalogError(f"Ollama model listing failed: {exc}") from exc

    entries = getattr(response, "models", None)
    if entries is None and isinstance(response, dict):
        entries = response.get("models")
    if not isinstance(entries, (list, tuple)):
        raise CatalogError("Ollama response has no model list")

    models: List[Dict[str, Any]] = []
    for entry in entries:
        details = getattr(entry, "details", None) or {}
        modified_at = getattr(entry, "modified_at", None)
        if isinstance(modified_at, datetime):
            modified_at = modified_at.isoformat()
        models.append(
            {
                "id": getattr(entry, "model", None) or getattr(entry, "name", None),
                "size": getattr(entry, "size", None),
                "modified_at": modified_at,
                "family": getattr(details, "family", None),
                "parameter_size": getattr(details, "parameter_size", None),
                "quantization_level": getattr(details, "quantization_level", None),
            }
        )
    return models


def search_models(models: Iterable[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = str(query).strip().lower()
    if not q:
        return list(models)
    return [
        model
        for model in models
        if q in str(model.get("id", "")).lower()
        or q in str(model.get("name", "")).lower()
        or q in str(model.get("display_name", "")).lower()
    ]


def get_models(
    provider: str,
    refresh: bool = False,
    cache_path: Optional[pathlib.Path] = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    fetcher: Optional[Any] = None,
    client: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return models and cache metadata; fetch only when needed."""
    provider = str(provider).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise CatalogError(f"Unsupported provider: {provider}")

    cache = _load_cache(cache_path)
    entry = cache.get(provider)

    if refresh:
        models, source = _fetch_models(provider, fetcher=fetcher, client=client)
        _save_cache(cache_path, provider, models, source)
        cache = _load_cache(cache_path)
        return models, _cache_metadata(cache[provider], max_age_hours)

    if entry and isinstance(entry.get("models"), list):
        metadata = _cache_metadata(entry, max_age_hours)
        return entry["models"], metadata

    models, source = _fetch_models(provider, fetcher=fetcher, client=client)
    _save_cache(cache_path, provider, models, source)
    cache = _load_cache(cache_path)
    return models, _cache_metadata(cache[provider], max_age_hours)


def _fetch_models(
    provider: str,
    fetcher: Optional[Any] = None,
    client: Any = None,
) -> Tuple[List[Dict[str, Any]], str]:
    if provider == "openrouter":
        return list_openrouter_models(fetcher=fetcher), OPENROUTER_MODELS_URL
    if provider == "gemini":
        return list_gemini_models(client), "gemini-api:models.list"
    if provider == "ollama":
        return list_ollama_models(client), "ollama-local"
    raise CatalogError(f"Unsupported provider: {provider}")
