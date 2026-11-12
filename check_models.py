#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_models.py
---------------
On-demand model discovery and explicit selection for Ollama, Gemini, and
OpenRouter.

Examples:
  python3 check_models.py --provider gemini
  python3 check_models.py --provider openrouter --search deepseek
  python3 check_models.py --provider openrouter --search flash --json
  python3 check_models.py --provider openrouter --refresh
  python3 check_models.py --provider openrouter --select deepseek/deepseek-v4.1-flash
  python3 check_models.py --provider openrouter --select deepseek/deepseek-v4.1-flash --save

Credentials are read from CLI overrides, environment variables, or config.yaml
in that order. They are never printed or written to the catalog cache.
"""

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from core.config import load_yaml_defaults
from core.model_catalog import (
    CatalogError,
    SUPPORTED_PROVIDERS,
    get_models,
    search_models,
)


def _key_from_env_and_yaml(
    env_name: str,
    yaml_key: str,
    fallback_env: Optional[str] = None,
    fallback_yaml: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value, env_name

    defaults = load_yaml_defaults()
    yaml_value = defaults.get(yaml_key)
    if yaml_value not in (None, ""):
        return str(yaml_value).strip(), f"config.yaml:{yaml_key}"

    if fallback_env:
        env_value = os.getenv(fallback_env, "").strip()
        if env_value:
            return env_value, fallback_env
    if fallback_yaml:
        yaml_value = defaults.get(fallback_yaml)
        if yaml_value not in (None, ""):
            return str(yaml_value).strip(), f"config.yaml:{fallback_yaml}"

    return "", None


def load_gemini_key() -> Tuple[str, Optional[str]]:
    """Read the Gemini key without validating unrelated Settings fields."""
    return _key_from_env_and_yaml("GEMINI_API_KEY", "gemini_api_key")


def load_openrouter_key() -> Tuple[str, Optional[str]]:
    """Read the OpenRouter-specific key only."""
    return _key_from_env_and_yaml("OPENROUTER_API_KEY", "openrouter_api_key")


def _gemini_client() -> Any:
    key, _source = load_gemini_key()
    if not key:
        raise CatalogError(
            "GEMINI_API_KEY fehlt (oder gemini_api_key in config.yaml)."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise CatalogError("google-genai ist nicht installiert.") from exc
    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        raise CatalogError(f"Gemini-Client konnte nicht erstellt werden: {exc}") from exc


def config_snippet(provider: str, model: str) -> str:
    if provider == "openrouter":
        return (
            'llm_provider: "openrouter"\n'
            f'openrouter_model: "{model}"\n'
            'openrouter_base_url: "https://openrouter.ai/api/v1"'
        )
    if provider == "gemini":
        return (
            'llm_provider: "gemini"\n'
            f'gemini_model: "{model}"'
        )
    if provider == "ollama":
        return (
            'llm_provider: "ollama"\n'
            f'ollama_model: "{model}"'
        )
    return f'llm_provider: "{provider}"\nllm_model: "{model}"'


def save_provider_model(provider: str, model: str, config_path: str = "config.yaml") -> None:
    """Update only the provider and provider-specific model lines.

    Other lines and comments are preserved. This is intentionally a line-based
    update so PyYAML's comment loss does not rewrite the whole file.
    """
    path = pathlib.Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    model_key = {
        "openrouter": "openrouter_model",
        "gemini": "gemini_model",
        "ollama": "ollama_model",
    }.get(provider, "llm_model")

    def replace_key(key: str, value: str) -> bool:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*.*$")
        for index, line in enumerate(lines):
            if pattern.match(line) and not line.lstrip().startswith("#"):
                lines[index] = f'{key}: "{value}"'
                return True
        return False

    if not replace_key("llm_provider", provider):
        lines.append(f'llm_provider: "{provider}"')
    if not replace_key(model_key, model):
        lines.append(f'{model_key}: "{model}"')

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "unknown" if value is None else str(value)


def print_cache_status(metadata: Dict[str, Any]) -> None:
    if metadata.get("stale"):
        print(
            f"Cache: STALE ({_fmt(metadata.get('age_hours'))}h old, "
            f"fetched {_fmt(metadata.get('fetched_at'))}); use --refresh for a fresh catalog."
        )
    else:
        print(
            f"Cache: fresh ({_fmt(metadata.get('age_hours'))}h old, "
            f"fetched {_fmt(metadata.get('fetched_at'))})."
        )


def print_models(provider: str, models: List[Dict[str, Any]]) -> None:
    print(f"Provider: {provider} ({len(models)} model(s))")
    if not models:
        print("No models found.")
        return

    if provider == "gemini":
        for model in models:
            print(
                f"{_fmt(model.get('id'))} | {_fmt(model.get('display_name'))} | "
                f"input={_fmt(model.get('input_token_limit'))} | "
                f"output={_fmt(model.get('output_token_limit'))} | "
                f"generateContent={_fmt(model.get('supports_generate_content'))}"
            )
    elif provider == "openrouter":
        for model in models:
            print(
                f"{_fmt(model.get('id'))} | {_fmt(model.get('name'))} | "
                f"context={_fmt(model.get('context_length'))} | "
                f"input={_fmt(model.get('input_modalities'))} | "
                f"output={_fmt(model.get('output_modalities'))} | "
                f"prompt={_fmt(model.get('prompt_usd_per_million_tokens'))} USD/1M | "
                f"completion={_fmt(model.get('completion_usd_per_million_tokens'))} USD/1M | "
                f"compatibility={_fmt(model.get('compatibility'))} | "
                f"json_mode={_fmt(model.get('json_mode_supported'))}"
            )
    elif provider == "ollama":
        for model in models:
            print(
                f"{_fmt(model.get('id'))} | size={_fmt(model.get('size'))} | "
                f"family={_fmt(model.get('family'))} | "
                f"quantization={_fmt(model.get('quantization_level'))} | "
                f"modified={_fmt(model.get('modified_at'))}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List, search, or select an LLM model for the reel pipeline."
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default="gemini",
        help="Provider to query (default: gemini).",
    )
    parser.add_argument("--search", default=None, help="Case-insensitive ID/name search.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and fetch a fresh catalog.")
    parser.add_argument("--cache", default=None, help="Catalog cache path (default: .cache/model_catalog.json).")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="Cache staleness threshold in hours (default: 24).",
    )
    parser.add_argument("--select", metavar="MODEL_ID", default=None, help="Select an exact model ID.")
    parser.add_argument("--save", action="store_true", help="Save the selected provider/model as config defaults.")
    parser.add_argument("--config", default="config.yaml", help="Config file for --save.")
    args = parser.parse_args(argv)

    if args.save and not args.select:
        print("--save requires --select MODEL_ID", file=sys.stderr)
        return 2

    provider = args.provider
    client = None
    gemini_error = None
    if provider == "gemini":
        try:
            client = _gemini_client()
        except CatalogError as exc:
            gemini_error = exc
            if args.refresh and not args.select:
                print(f"Catalog error: {exc}", file=sys.stderr)
                return 1
            # A cached Gemini catalog or an explicit selection may still work offline.
            client = None

    if args.select:
        models: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {
            "fetched_at": None,
            "source": None,
            "age_hours": None,
            "stale": True,
        }
        try:
            models, metadata = get_models(
                provider,
                refresh=args.refresh,
                cache_path=args.cache,
                max_age_hours=args.max_age_hours,
                client=client,
            )
        except CatalogError as exc:
            detail = gemini_error or exc
            print(f"Warning: catalog unavailable: {detail}", file=sys.stderr)

        found = any(model.get("id") == args.select for model in models)
        if models and not found:
            print(
                f"Warning: {args.select!r} is not in the catalog; "
                "the catalog may be stale or incomplete.",
                file=sys.stderr,
            )
        elif not models:
            print(
                "Warning: no catalog available; the explicit model ID is still usable.",
                file=sys.stderr,
            )

        print(f"Selected provider: {provider}")
        print(f"Selected model: {args.select}")
        print("Config snippet:")
        print(config_snippet(provider, args.select))

        if args.save:
            save_provider_model(provider, args.select, args.config)
            print(f"Saved provider/model defaults to {args.config}")
        return 0

    try:
        models, metadata = get_models(
            provider,
            refresh=args.refresh,
            cache_path=args.cache,
            max_age_hours=args.max_age_hours,
            client=client,
        )
    except CatalogError as exc:
        detail = gemini_error or exc
        print(f"Catalog error: {detail}", file=sys.stderr)
        return 1

    if args.search:
        models = search_models(models, args.search)

    if args.json:
        print(
            json.dumps(
                {"provider": provider, "models": models, "cache": metadata},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return 0

    print_cache_status(metadata)
    print_models(provider, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
