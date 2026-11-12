#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_models.py
---------------
Listet verfügbare Gemini-Modelle. Der API-Key wird über die zentrale
Settings-Ladung gelesen: GEMINI_API_KEY hat Vorrang; nicht-leere
config.yaml-Werte dienen als Fallback, leere/null Werte werden ignoriert.
"""

import os
import sys
from typing import Optional, Tuple

from google import genai

from core.config import Settings, load_yaml_defaults


def load_gemini_key() -> Tuple[str, Optional[str]]:
    """Read the Gemini key through the same Settings path as production."""
    defaults = load_yaml_defaults()
    settings = Settings(**defaults)
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    key = key.strip()
    if key:
        if "gemini_api_key" in defaults:
            return key, "config.yaml:gemini_api_key"
        return key, "GEMINI_API_KEY"
    return "", None


def main() -> int:
    api_key, source = load_gemini_key()

    if not api_key:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            print("OPENAI_API_KEY ist gesetzt, aber check_models.py prüft Gemini-Modelle.")
            print("Setze GEMINI_API_KEY oder gemini_api_key in config.yaml.")
        else:
            print("Kein Gemini-API-Key gefunden.")
            print("Setze GEMINI_API_KEY oder gemini_api_key in config.yaml.")
        return 1

    print(f"--- Prüfe Gemini-Modelle (Key aus {source}) ---")

    try:
        client = genai.Client(api_key=api_key)
        print("\nGefundene Modelle:")
        print("-" * 60)

        for model in client.models.list():
            name = getattr(model, "name", str(model))
            display_name = getattr(model, "display_name", "")

            if "gemini" in str(name).lower():
                print(f"✅ ID: {name}")
                if display_name:
                    print(f"   Name: {display_name}")
                print("-" * 20)

    except Exception as exc:
        print(f"\n❌ Fehler: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
