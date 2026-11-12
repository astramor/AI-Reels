#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
from typing import Any, Dict, Optional

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_MODEL = "qwen2.5:14b-instruct-q6_K"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLAMA_SERVER_BASE_URL = "http://127.0.0.1:8080/v1"


def clean_thinking(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think.*? response", "", text, flags=re.DOTALL)
    text = re.sub(r"<think.*", "", text, flags=re.DOTALL)
    return text.strip()


def clean_json_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _get_config_value(config: Dict[str, Any], key: str, env_name: str) -> str:
    value = config.get(key)
    if not value and key in ("openrouter_api_key", "openai_api_key"):
        nested = config.get("openrouter")
        if isinstance(nested, dict):
            value = nested.get("api_key")
    if not value:
        value = os.getenv(env_name, "")
    return value or ""


def _get_model_for_provider(config: Dict[str, Any], provider: str) -> str:
    """Resolve the model name for the active provider.

    Supports provider-specific fields as well as the legacy ``llm_model``
    fallback. OpenRouter may also use the nested ``openrouter.model`` form.
    """
    provider = provider.lower()
    if provider == "gemini-api":
        provider = "gemini"

    if provider == "gemini":
        return (
            config.get("gemini_model")
            or config.get("llm_model")
            or DEFAULT_GEMINI_MODEL
        )

    if provider == "openrouter":
        nested = config.get("openrouter")
        if isinstance(nested, dict) and nested.get("model"):
            return nested["model"]
        return (
            config.get("openrouter_model")
            or config.get("llm_model")
            or DEFAULT_MODEL
        )

    if provider == "openai":
        return config.get("openai_model") or config.get("llm_model") or DEFAULT_MODEL

    if provider == "ollama":
        return config.get("ollama_model") or config.get("llm_model") or DEFAULT_MODEL

    if provider == "llama-server":
        return (
            config.get("llama_server_model")
            or config.get("llm_model")
            or DEFAULT_MODEL
        )

    return config.get("llm_model") or DEFAULT_MODEL


class UnifiedLLM:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        raw_provider = str(self.config.get("llm_provider", "ollama")).lower()
        if raw_provider == "gemini-api":
            self.provider = "gemini"
        elif raw_provider == "openai":
            self.provider = "openai"
        elif raw_provider == "openrouter":
            self.provider = "openrouter"
        elif raw_provider == "llama-server":
            self.provider = "llama-server"
        elif raw_provider == "ollama":
            self.provider = "ollama"
        else:
            self.provider = raw_provider

        fallback = str(self.config.get("llm_fallback_provider", "ollama")).lower()
        self.fallback_provider = "gemini" if fallback == "gemini-api" else fallback

        self.is_gemini = self.provider == "gemini"
        self.is_openai = self.provider in ("openai", "openrouter", "llama-server")
        self.is_openrouter = self.provider in ("openrouter", "openai")

        self.gemini_key = _get_config_value(
            self.config, "gemini_api_key", "GEMINI_API_KEY"
        )
        self.openai_key = _get_config_value(
            self.config, "openai_api_key", "OPENAI_API_KEY"
        )
        self.openrouter_key = _get_config_value(
            self.config, "openrouter_api_key", "OPENROUTER_API_KEY"
        )

        self.gemini_model = _get_model_for_provider(self.config, "gemini")
        self.model = _get_model_for_provider(self.config, self.provider)
        self.openrouter_base_url = (
            self.config.get("openai_base_url")
            or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        )
        self.llama_server_base_url = (
            self.config.get("llama_server_base_url")
            or os.getenv("LLAMA_SERVER_BASE_URL", DEFAULT_LLAMA_SERVER_BASE_URL)
        )
        self.ollama_host = (
            self.config.get("ollama_host")
            or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        )

        self._clients: Dict[str, Any] = {}
        self._models: Dict[str, str] = {}

    @property
    def is_cloud_llm(self) -> bool:
        return self.provider in ("openrouter", "openai", "gemini", "gemini-api")

    def _init_provider(self, provider: str) -> None:
        provider = provider.lower()
        if provider == "gemini-api":
            provider = "gemini"
        if provider in self._clients:
            return

        if provider == "gemini":
            if not genai:
                raise ImportError("Bitte `pip install google-genai` für Gemini-Support.")
            if not self.gemini_key:
                raise ValueError("GEMINI_API_KEY fehlt.")
            self._clients[provider] = genai.Client(api_key=self.gemini_key)
            self._models[provider] = _get_model_for_provider(self.config, provider)
        elif provider == "openrouter":
            if not OpenAI:
                raise ImportError("Bitte `pip install openai` für OpenRouter-Support.")
            key = self.openrouter_key or self.openai_key
            if not key:
                raise ValueError("OPENROUTER_API_KEY oder OPENAI_API_KEY fehlt.")
            self._clients[provider] = OpenAI(
                base_url=self.openrouter_base_url,
                api_key=key,
            )
            self._models[provider] = _get_model_for_provider(self.config, provider)
        elif provider == "openai":
            if not OpenAI:
                raise ImportError("Bitte `pip install openai` für OpenAI-Support.")
            key = self.openai_key or self.openrouter_key
            if not key:
                raise ValueError("OPENAI_API_KEY oder OPENROUTER_API_KEY fehlt.")
            self._clients[provider] = OpenAI(
                base_url=self.openrouter_base_url,
                api_key=key,
            )
            self._models[provider] = _get_model_for_provider(self.config, provider)
        elif provider == "llama-server":
            if not OpenAI:
                raise ImportError("Bitte `pip install openai` für llama-server-Support.")
            key = self.openai_key or self.openrouter_key or "not-needed"
            self._clients[provider] = OpenAI(
                base_url=self.llama_server_base_url,
                api_key=key,
            )
            self._models[provider] = _get_model_for_provider(self.config, provider)
        elif provider == "ollama":
            if not OllamaClient:
                raise ImportError("Bitte `pip install ollama` für Ollama-Support.")
            self._clients[provider] = OllamaClient(host=self.ollama_host)
            self._models[provider] = _get_model_for_provider(self.config, provider)
        else:
            raise ValueError(f"Unbekannter LLM-Provider: {provider}")

    def _chat_with_provider(
        self,
        provider: str,
        sys_p: str,
        user_p: str,
        temp: float = 0.1,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        provider = provider.lower()
        if provider == "gemini-api":
            provider = "gemini"
        self._init_provider(provider)
        client = self._clients[provider]
        model = self._models[provider]

        if provider == "gemini":
            try:
                time.sleep(1)
                config_kwargs = {
                    "system_instruction": sys_p,
                    "temperature": temp,
                }
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"
                res = client.models.generate_content(
                    model=model,
                    contents=user_p,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return clean_thinking(res.text)
            except Exception as e:
                print(f"❌ Gemini Error: {e}")
                return ""
        elif provider in ("openrouter", "openai", "llama-server"):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    "temperature": temp,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                res = client.chat.completions.create(**kwargs)
                return clean_thinking(res.choices[0].message.content)
            except Exception as e:
                print(f"❌ {provider} Error: {e}")
                return ""
        else:
            try:
                options = {"temperature": temp, "num_ctx": 8192}
                if max_tokens:
                    options["num_predict"] = max_tokens
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    "options": options,
                    "keep_alive": "0",
                }
                if json_mode:
                    kwargs["format"] = "json"
                resp = client.chat(**kwargs)
                return clean_thinking(resp["message"]["content"])
            except Exception as e:
                print(f"❌ Ollama Error: {e}")
                return ""

    def chat(
        self,
        sys_p: str,
        user_p: str,
        temp: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        primary = self.provider
        try:
            text = self._chat_with_provider(
                primary, sys_p, user_p, temp=temp, max_tokens=max_tokens
            )
        except Exception as e:
            print(f"❌ Provider {primary} nicht verfügbar: {e}")
            text = ""

        if not text and self.fallback_provider and self.fallback_provider != primary:
            try:
                text = self._chat_with_provider(
                    self.fallback_provider,
                    sys_p,
                    user_p,
                    temp=temp,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                print(
                    f"❌ Fallback-Provider {self.fallback_provider} nicht verfügbar: {e}"
                )
                text = ""
        return text

    def chat_json(
        self,
        sys_p: str,
        user_p: str,
        temp: float = 0.2,
    ) -> Optional[str]:
        if self.is_gemini:
            text = self._chat_with_provider(
                "gemini", sys_p, user_p, temp=temp, json_mode=True
            )
            if not text and self.fallback_provider and self.fallback_provider != "gemini":
                try:
                    text = self._chat_with_provider(
                        self.fallback_provider,
                        sys_p,
                        user_p,
                        temp=temp,
                        json_mode=True,
                    )
                except Exception as e:
                    print(
                        f"❌ Fallback-Provider {self.fallback_provider} nicht verfügbar: {e}"
                    )
                    text = ""
            return clean_json_text(text) or None

        text = self.chat(sys_p, user_p, temp=temp)
        return clean_json_text(text) or None

    def unload(self) -> None:
        if self.provider in ("gemini", "openrouter", "openai", "llama-server"):
            return
        if self.provider == "ollama":
            client = self._clients.get("ollama")
            if client:
                try:
                    client.generate(model=self.model, keep_alive="0")
                except Exception as e:
                    print(f"❌ Modell-Entladen fehlgeschlagen: {e}")
