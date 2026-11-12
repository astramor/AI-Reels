#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_models.py
---------------
Listet verfügbare Gemini-Modelle. Der API-Key wird primär aus der
Umgebungsvariable GEMINI_API_KEY gelesen; config.yaml dient nur als Fallback.
"""

import os
import sys
from typing import Optional, Tuple

import yaml
from google import genai


def load_gemini_key() -> Tuple[str, Optional[str]]:
    """Read GEMINI_API_KEY first, then fall back to config.yaml."""
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, "GEMINI_API_KEY"

    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        key = str(conf.get("gemini_api_key", "") or "").strip()
        if key:
            return key, "config.yaml:gemini_api_key"
    except Exception as exc:
        print(f"Warnung: config.yaml konnte nicht gelesen werden: {exc}", file=sys.stderr)

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
