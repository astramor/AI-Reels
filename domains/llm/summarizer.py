#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib
import re
import time
from typing import List, Tuple, Generator, Dict

import srt
from loguru import logger
from core.prompts import (
    JSON_HIGHLIGHT_SYSTEM_PROMPT,
    JSON_HIGHLIGHT_USER_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_MERGE_PROMPT
)
from domains.llm.errors import LLMApiError, LLMInvalidJsonError
from domains.highlights.errors import HighlightValidationError, EmptyTranscriptError
from domains.highlights.models import HighlightPayload

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class SermonSummarizer:
    """
    Klasse zur Zusammenfassung von Videos und Extraktion von Highlights mittels LLMs.
    Unterstützt Ollama und Gemini. Liefert nun konsequent striktes JSON.
    """

    def __init__(self, settings):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.is_gemini = (self.provider == "gemini")
        
        if self.is_gemini:
            self.model_name = settings.gemini_model
            if not genai:
                raise ImportError("Bitte `pip install google-genai` für Gemini-Support.")
            api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini Mode: {self.model_name}")
        else:
            self.model_name = settings.llm_model
            if not OllamaClient:
                raise ImportError("Bitte `pip install ollama` für Ollama-Support.")
            self.client = OllamaClient(host="http://127.0.0.1:11434")
            logger.info(f"Ollama Mode: {self.model_name}")

    def _get_highlight_schema(self) -> types.Schema:
        """Definiert die exakte JSON-Struktur, die wir von Gemini erwarten (Structured Outputs)."""
        return types.Schema(
            type=types.Type.OBJECT,
            properties={
                "highlights": types.Schema(
                    type=types.Type.ARRAY,
                    description="Eine Liste der besten Video-Highlights.",
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "title": types.Schema(type=types.Type.STRING, description="Kurzer, packender Titel"),
                            "start": types.Schema(type=types.Type.NUMBER, description="Startzeit in Sekunden (Float)"),
                            "end": types.Schema(type=types.Type.NUMBER, description="Endzeit in Sekunden (Float)"),
                            "reason": types.Schema(type=types.Type.STRING, description="Warum dieser Clip stark ist"),
                            "hook": types.Schema(type=types.Type.STRING, description="Optionaler Einstiegssatz"),
                            "confidence": types.Schema(type=types.Type.NUMBER, description="Confidence (0.0-1.0)")
                        },
                        required=["title", "start", "end", "reason"]
                    )
                )
            },
            required=["highlights"]
        )

    def extract_json_from_llm_response(self, raw_text: str) -> dict:
        """
        Extrahiert ein JSON-Objekt aus einem potenziell unsauberen LLM-Output.
        Entfernt Markdown-Formatierungen wie ```json ... ```.
        """
        if not raw_text:
            raise LLMInvalidJsonError("LLM-Ausgabe war leer.")
            
        # 1. Versuche direkten Parse (falls der LLM brav war)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. Suche nach Markdown JSON-Blöcken
        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Fallback: Suche nach dem ersten { und dem letzten }
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(raw_text[start_idx:end_idx+1])
            except json.JSONDecodeError as e:
                raise LLMInvalidJsonError(f"Konnte JSON auch aus Fallback-Bereich nicht parsen: {e}")
                
        raise LLMInvalidJsonError("Kein JSON-Objekt im Text gefunden.")

    def chat_json(self, sys_p: str, user_p: str, temp: float = 0.2) -> str:
        """Sendet einen Request an das LLM mit Anweisung für JSON-Ausgabe."""
        if self.is_gemini:
            try:
                time.sleep(1) # Rate limiting / Safety
                res = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_p,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_p,
                        temperature=temp,
                        response_mime_type="application/json",
                        response_schema=self._get_highlight_schema(),
                    ),
                )
                return res.text
            except Exception as e:
                raise LLMApiError(f"Gemini API Fehler: {e}")
        else:
            try:
                options = {
                    "temperature": temp,
                    "num_ctx": 16384, # Größerer Kontext für lange Transkripte
                    "num_predict": 2048
                }
                resp = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    options=options,
                    format="json" # Ollama JSON mode
                )
                return resp["message"]["content"]
            except Exception as e:
                raise LLMApiError(f"Ollama API Fehler: {e}")

    def _clean_thinking(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    def chat(self, sys_p: str, user_p: str, temp: float = 0.1) -> str:
        """Klassischer Text-Chat (wird noch für Summary benutzt)."""
        if self.is_gemini:
            try:
                time.sleep(1)
                res = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_p,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_p, temperature=temp
                    ),
                )
                return self._clean_thinking(res.text)
            except Exception as e:
                logger.error(f"Gemini Error: {e}")
                return ""
        else:
            try:
                resp = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    options={"temperature": temp, "num_ctx": 8192},
                )
                return self._clean_thinking(resp["message"]["content"])
            except Exception as e:
                logger.error(f"Ollama Error: {e}")
                return ""

    def unload(self):
        """Entlädt das Modell aus dem VRAM (nur für Ollama)."""
        if not self.is_gemini and self.client:
            logger.info(f"Unloading model {self.model_name} from VRAM...")
            try:
                self.client.generate(model=self.model_name, keep_alive=0)
            except Exception as e:
                logger.warning(f"Modell-Entladen fehlgeschlagen: {e}")

    def _srt_items(self, srt_path: pathlib.Path) -> Generator[Tuple[srt.Subtitle, str], None, None]:
        for it in srt.parse(srt_path.read_text(encoding="utf-8")):
            yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())

    def validate_highlight_payload(self, payload: dict) -> HighlightPayload:
        """Validiert das geparste JSON-Dictionary mittels Pydantic."""
        try:
            return HighlightPayload.model_validate(payload)
        except Exception as e:
            raise HighlightValidationError(f"JSON erfüllt nicht das Pydantic Schema: {e}")

    def process(self, srt_path_str: str, out_dir_str: str, max_highlights: int = 8):
        srt_path = pathlib.Path(srt_path_str)
        out_dir = pathlib.Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        items = list(self._srt_items(srt_path))
        if not items:
            raise EmptyTranscriptError("Das SRT-Transkript ist leer.")

        # Für JSON-Prompt nutzen wir Floats, da das LLM Floats zurückgeben soll.
        transcript_for_json = "\n".join([f"[{it.start.total_seconds():.2f}s] {txt}" for it, txt in items])
        (out_dir / "transcript_predigt.txt").write_text(transcript_for_json, encoding="utf-8")

        # 1. Highlights via strict JSON
        logger.info("--- Highlights Scouting (Strict JSON) ---")
        user_msg = JSON_HIGHLIGHT_USER_PROMPT.format(transcript=transcript_for_json)
        
        try:
            logger.info("Frage LLM nach JSON-Highlights...")
            raw_response = self.chat_json(JSON_HIGHLIGHT_SYSTEM_PROMPT, user_msg, temp=0.2)
            
            # Artefakt 1: Raw LLM Response
            (out_dir / "highlights_raw_llm.txt").write_text(raw_response, encoding="utf-8")
            
            # Extrahieren & Bereinigen
            parsed_json = self.extract_json_from_llm_response(raw_response)
            
            # Artefakt 2: Parsed JSON
            (out_dir / "highlights_parsed.json").write_text(json.dumps(parsed_json, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Validieren via Pydantic
            validated_payload = self.validate_highlight_payload(parsed_json)
            
            # Artefakt 3: Validated Highlights (Final)
            final_json_path = out_dir / "highlights.json"
            final_json_path.write_text(validated_payload.model_dump_json(indent=2), encoding="utf-8")
            logger.success(f"Erfolgreich {len(validated_payload.highlights)} valide Highlights generiert.")
            
        except (LLMApiError, LLMInvalidJsonError, HighlightValidationError) as e:
            logger.error(f"Fehler bei der Highlight-Generierung: {e}")
            raise NoValidHighlightsError(f"Abbruch: {e}")

        # 2. Summary erstellen (Beibehalten der klassischen Logik)
        c_size = 40000 if self.is_gemini else 6000
        chunks = [transcript_for_json[i : i + c_size] for i in range(0, len(transcript_for_json), c_size)]
        parts = []
        logger.info(f"--- Summary ({len(chunks)} Chunks) ---")
        for c in chunks:
            parts.append(self.chat("Du bist Redakteur.", SUMMARY_CHUNK_PROMPT.format(chunk=c), temp=0.1))

        full_sum = parts[0] if len(parts) == 1 else self.chat(
            "Du bist Redakteur.", SUMMARY_MERGE_PROMPT.format(parts="\n".join(parts)), temp=0.1
        )
        (out_dir / "summary.md").write_text(full_sum, encoding="utf-8")

        return {
            "summary_path": out_dir / "summary.md",
            "highlights_path": final_json_path
        }
