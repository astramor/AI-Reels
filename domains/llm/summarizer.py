#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib
import re
from typing import List, Tuple, Generator, Dict

import srt
from loguru import logger
from core.prompts import (
    JSON_HIGHLIGHT_SYSTEM_PROMPT,
    JSON_HIGHLIGHT_USER_PROMPT,
    JSON_HIGHLIGHT_REDUCE_SYSTEM_PROMPT,
    JSON_HIGHLIGHT_REDUCE_USER_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_MERGE_PROMPT
)
from core.llm_client import UnifiedLLM, clean_thinking
from domains.llm.errors import LLMApiError, LLMInvalidJsonError
from domains.highlights.errors import HighlightValidationError, EmptyTranscriptError
from domains.highlights.models import HighlightPayload


class SermonSummarizer:
    """
    Klasse zur Zusammenfassung von Videos und Extraktion von Highlights mittels LLMs.
    Unterstützt Ollama, Gemini und OpenAI-kompatible Provider (OpenRouter, llama-server usw.).
    Nutzt ein robustes Zwei-Stufen-Verfahren (Scouting & Curation).
    """

    def __init__(self, settings):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.is_gemini = self.provider in ("gemini", "gemini-api")
        self.is_openai = self.provider in ("openai", "openrouter", "llama-server")

        config = {
            "llm_provider": self.provider,
            "llm_model": settings.llm_model,
            "gemini_model": settings.gemini_model,
            "openrouter_model": settings.openrouter_model,
            "openai_model": settings.openai_model,
            "ollama_model": settings.ollama_model,
            "llama_server_model": settings.llama_server_model,
            "gemini_api_key": settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else "",
            "openai_api_key": settings.openai_api_key,
            "openai_base_url": settings.openai_base_url,
        }
        self.llm = UnifiedLLM(config)
        self.model_name = self.llm.gemini_model if self.is_gemini else self.llm.model
        logger.info(f"LLM Mode: {self.llm.provider} / {self.model_name}")



    def extract_json_from_llm_response(self, raw_text: str) -> dict:
        """Extrahiert JSON aus potenziell unsauberem LLM-Output."""
        if not raw_text:
            raise LLMInvalidJsonError("LLM-Ausgabe war leer.")
            
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(raw_text[start_idx:end_idx+1])
            except json.JSONDecodeError as e:
                raise LLMInvalidJsonError(f"Konnte JSON aus Fallback nicht parsen: {e}")
                
        raise LLMInvalidJsonError("Kein JSON-Objekt gefunden.")

    def chat_json(self, sys_p: str, user_p: str, temp: float = 0.2) -> str:
        """Sendet einen Request an das LLM mit Anweisung für JSON-Ausgabe."""
        raw = self.llm.chat_json(sys_p, user_p, temp=temp)
        if not raw:
            raise LLMApiError("LLM-Ausgabe war leer.")
        return raw

    def chat(self, sys_p: str, user_p: str, temp: float = 0.1) -> str:
        """Klassischer Text-Chat (für Summary)."""
        return self.llm.chat(sys_p, user_p, temp=temp)



    def unload(self):
        self.llm.unload()

    def _srt_items(self, srt_path: pathlib.Path) -> Generator[Tuple[srt.Subtitle, str], None, None]:
        for it in srt.parse(srt_path.read_text(encoding="utf-8")):
            yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())

    def validate_highlight_payload(self, payload: dict) -> HighlightPayload:
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

        transcript_for_json = "\n".join([f"[{it.start.total_seconds():.2f}s] {txt}" for it, txt in items])
        (out_dir / "transcript_predigt.txt").write_text(transcript_for_json, encoding="utf-8")

        # 1. Summary erstellen
        c_size = 40000 if self.llm.is_cloud_llm else 6000
        chunks = [transcript_for_json[i : i + c_size] for i in range(0, len(transcript_for_json), c_size)]
        parts = []
        logger.info(f"--- Summary ({len(chunks)} Chunks) ---")
        for c in chunks:
            parts.append(self.chat("Du bist Redakteur.", SUMMARY_CHUNK_PROMPT.format(chunk=c), temp=0.1))

        full_sum = parts[0] if len(parts) == 1 else self.chat(
            "Du bist Redakteur.", SUMMARY_MERGE_PROMPT.format(parts="\n".join(parts)), temp=0.1
        )
        (out_dir / "summary.md").write_text(full_sum, encoding="utf-8")

        # 2. Highlights Scouting (Stage 1: Candidates via Chunking)
        logger.info("--- Highlights Scouting (Stage 1: Candidates) ---")
        longlist_candidates = []
        
        # Cloud-Provider erhalten große Chunks (40000), Ollama kleinere (6000).
        scout_chunks = [transcript_for_json[i : i + c_size] for i in range(0, len(transcript_for_json), c_size)]
        
        for i, chunk in enumerate(scout_chunks):
            logger.info(f"Scout Chunk {i + 1}/{len(scout_chunks)}...")
            user_msg = JSON_HIGHLIGHT_USER_PROMPT.format(transcript=chunk)
            try:
                raw_res = self.chat_json(JSON_HIGHLIGHT_SYSTEM_PROMPT, user_msg, temp=0.3)
                parsed = self.extract_json_from_llm_response(raw_res)
                longlist_candidates.extend(parsed.get("highlights", []))
            except Exception as e:
                logger.warning(f"Fehler in Scout Chunk {i+1}: {e}")

        (out_dir / "highlights_stage1_parsed.json").write_text(
            json.dumps({"highlights": longlist_candidates}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 3. Highlights Selection (Stage 2: Curation)
        logger.info("--- Highlights Selection (Stage 2: Curation) ---")
        try:
            # Wir geben dem Curation-Schritt das Summary und die Longlist
            reduce_user_msg = JSON_HIGHLIGHT_REDUCE_USER_PROMPT.format(
                summary=full_sum,
                candidates=json.dumps({"highlights": longlist_candidates[:25]}, indent=2, ensure_ascii=False)
            )
            
            logger.info("Wähle die besten 5-6 Clips für Social Media aus...")
            raw_response_s2 = self.chat_json(JSON_HIGHLIGHT_REDUCE_SYSTEM_PROMPT, reduce_user_msg, temp=0.1)
            (out_dir / "highlights_raw_llm.txt").write_text(raw_response_s2, encoding="utf-8")
            
            parsed_json = self.extract_json_from_llm_response(raw_response_s2)
            (out_dir / "highlights_parsed.json").write_text(json.dumps(parsed_json, indent=2, ensure_ascii=False), encoding="utf-8")
            
            validated_payload = self.validate_highlight_payload(parsed_json)
            
            # Falls zu wenige Ergebnisse, füllen wir aus der Longlist auf
            if len(validated_payload.highlights) < 4 and len(longlist_candidates) > 0:
                logger.warning("Zu wenige Highlights kuratiert. Fülle aus Kandidaten-Pool auf...")
                existing_starts = [h.start for h in validated_payload.highlights]
                for cand in longlist_candidates:
                    if len(validated_payload.highlights) >= 6: break
                    if cand["start"] not in existing_starts:
                        try:
                            # Validierung des einzelnen Kandidaten
                            validated_payload.highlights.append(cand)
                            existing_starts.append(cand["start"])
                        except: pass

            final_json_path = out_dir / "highlights.json"
            final_json_path.write_text(validated_payload.model_dump_json(indent=2), encoding="utf-8")
            logger.success(f"Erfolgreich {len(validated_payload.highlights)} Top-Highlights kuratiert.")
            
        except (LLMApiError, LLMInvalidJsonError, HighlightValidationError) as e:
            logger.error(f"Fehler bei der Kuratierung: {e}")
            raise NoValidHighlightsError(f"Abbruch: {e}")

        return {
            "summary_path": out_dir / "summary.md",
            "highlights_path": final_json_path
        }
