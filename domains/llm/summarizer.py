#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib
import re
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Generator

import srt
from loguru import logger
from core.time_utils import timedelta_to_hms
from core.prompts import OLLAMA_SUMMARIZER_PROMPTS, GEMINI_SUMMARIZER_PROMPTS

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
    Klasse zur Zusammenfassung von Predigten und Extraktion von Highlights mittels LLMs.
    Unterstützt Ollama und Gemini.
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
            self.prompts = GEMINI_SUMMARIZER_PROMPTS
            logger.info(f"Gemini Mode: {self.model_name}")
        else:
            self.model_name = settings.llm_model
            if not OllamaClient:
                raise ImportError("Bitte `pip install ollama` für Ollama-Support.")
            self.client = OllamaClient(host="http://127.0.0.1:11434")
            self.prompts = OLLAMA_SUMMARIZER_PROMPTS
            logger.info(f"Ollama Mode: {self.model_name}")

    def _get_highlight_schema(self) -> types.Schema:
        """Definiert die exakte JSON-Struktur, die wir von Gemini erwarten."""
        return types.Schema(
            type=types.Type.ARRAY,
            description="Eine Liste der besten Video-Highlights.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start_timestamp": types.Schema(
                        type=types.Type.STRING, 
                        description="Der exakte Start-Zeitstempel im Format HH:MM:SS"
                    ),
                    "end_timestamp": types.Schema(
                        type=types.Type.STRING, 
                        description="Der exakte End-Zeitstempel im Format HH:MM:SS"
                    ),
                    "quote": types.Schema(
                        type=types.Type.STRING, 
                        description="Das bereinigte, virale Zitat"
                    )
                },
                required=["start_timestamp", "end_timestamp", "quote"]
            )
        )

    def extract_highlights_gemini(self, sys_prompt: str, user_prompt: str, temp: float = 0.2) -> list:
        """
        Führt den Gemini-Aufruf mit striktem JSON-Schema durch.
        Gibt eine Liste von Dictionaries zurück: [{'start_timestamp': '...', 'end_timestamp': '...', 'quote': '...'}]
        """
        if not self.is_gemini:
            raise NotImplementedError("Diese Methode erfordert den Gemini Provider.")
            
        try:
            time.sleep(1) # Rate limiting safety
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=temp,
                    response_mime_type="application/json",
                    response_schema=self._get_highlight_schema(),
                ),
            )
            return json.loads(res.text)
        except Exception as e:
            logger.error(f"Gemini API / Parsing Fehler: {e}")
            return []

    def _clean_thinking(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def chat(self, sys_p: str, user_p: str, temp: float = 0.1) -> str:
        if self.is_gemini:
            try:
                time.sleep(1) # Rate limiting / Safety
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
                options = {
                    "temperature": temp,
                    "num_ctx": 8192,
                    "num_predict": 1500,
                    "repeat_penalty": 1.15,
                }
                resp = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    options=options,
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
                # Ollama Trick: keep_alive=0 entlädt das Modell sofort
                self.client.generate(model=self.model_name, keep_alive=0)
            except Exception as e:
                logger.warning(f"Modell-Entladen fehlgeschlagen: {e}")

    def _srt_items(self, srt_path: pathlib.Path) -> Generator[Tuple[srt.Subtitle, str], None, None]:
        for it in srt.parse(srt_path.read_text(encoding="utf-8")):
            yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())

    def _build_highlight_candidates(self, items: List[Tuple[srt.Subtitle, str]]) -> List[str]:
        return [
            f"[{timedelta_to_hms(it.start)}] {txt[:200]}"
            for it, txt in items
            if len(txt.split()) > 5
        ]

    def _find_smart_end(self, start_sec: int, items: List[Tuple[srt.Subtitle, str]], min_dur: int = 18, max_dur: int = 60) -> str:
        """Findet das nächste logische Satzende im SRT für einen Smart Cut."""
        end_sec = start_sec + 20 # Fallback
        for it, txt in items:
            s_end = it.end.total_seconds()
            if s_end > start_sec + min_dur:
                if re.search(r"[.!?]", txt):
                    end_sec = int(s_end)
                    break
            if s_end > start_sec + max_dur:
                end_sec = int(s_end)
                break
        
        # In HMS umwandeln
        td = timedelta(seconds=end_sec)
        return timedelta_to_hms(td)

    def _sanitize_highlights(self, text: str, items: List[Tuple[srt.Subtitle, str]], tol: int = 8) -> List[str]:
        out = []
        sec_map = {int(it.start.total_seconds()): timedelta_to_hms(it.start) for it, txt in items}
        line_re = re.compile(r".*?\[(\d{1,2})[:.](\d{1,2})[:.]?(\d{1,2})?\].*?[:\s]+(.*)")

        for line in text.splitlines():
            if "Zitat:" in line:
                line = line.replace("Das Zitat:", "")
            m = line_re.match(line)
            if not m:
                continue
            h, m_val, s_val, txt = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3) or 0),
                m.group(4),
            )
            sec = h * 3600 + m_val * 60 + s_val

            used = set()
            for l in out:
                 found_ts = re.search(r"\[(\d{1,2}):(\d{1,2}):(\d{1,2})\]", l)
                 if found_ts:
                     used.add(int(found_ts.group(1))*3600 + int(found_ts.group(2))*60 + int(found_ts.group(3)))
            
            if any(s in used for s in range(sec - 15, sec + 15)):
                continue

            found_start = None
            for off in range(-tol, tol + 1):
                if (sec + off) in sec_map:
                    found_start = sec_map[sec + off]
                    start_sec_actual = sec + off
                    break

            if found_start:
                clean = (
                    re.sub(r"^(Und|Aber|Denn)\s+", "", txt, flags=re.IGNORECASE)
                    .strip('"')
                    .strip()
                )
                if len(clean) > 8:
                    # SMART CUT LOGIC HIER
                    found_end = self._find_smart_end(start_sec_actual, items)
                    out.append(f"- [{found_start} -> {found_end}] {clean}")
        return out

    def process(self, srt_path_str: str, out_dir_str: str, max_highlights: int = 8, strict_snap_tol: int = 8):
        srt_path = pathlib.Path(srt_path_str)
        out_dir = pathlib.Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        items = list(self._srt_items(srt_path))
        transcript = "\n".join([f"[{timedelta_to_hms(it.start)}] {txt}" for it, txt in items])
        (out_dir / "transcript_predigt.txt").write_text(transcript, encoding="utf-8")

        c_size = 40000 if self.is_gemini else 6000

        # 1. Summary erstellen
        chunks = [transcript[i : i + c_size] for i in range(0, len(transcript), c_size)]
        parts = []
        logger.info(f"--- Summary ({len(chunks)} Chunks) ---")
        for c in chunks:
            parts.append(
                self.chat("Du bist Redakteur.", self.prompts["chunk"].format(chunk=c), temp=0.1)
            )

        full_sum = (
            parts[0]
            if len(parts) == 1
            else self.chat(
                "Du bist Redakteur.", self.prompts["merge"].format(parts="\n".join(parts)), temp=0.1
            )
        )
        (out_dir / "summary.md").write_text(full_sum, encoding="utf-8")

        # 2. Highlights scouting
        logger.info("--- Highlights Scouting ---")
        hl_chunks = []
        curr, l = [], 0
        cands = self._build_highlight_candidates(items)
        for c in cands:
            if l + len(c) > c_size:
                hl_chunks.append("\n".join(curr))
                curr, l = [], 0
            curr.append(c)
            l += len(c)
        if curr:
            hl_chunks.append("\n".join(curr))

        # --- Highlights scouting (Map) ---
        if self.is_gemini:
            longlist_objects = []
            for i, c in enumerate(hl_chunks):
                logger.info(f"Scout Chunk {i + 1}...")
                user_msg = self.prompts["scout_user"].format(chunk=c)
                
                chunk_results = self.extract_highlights_gemini(
                    sys_prompt=self.prompts["scout_system"],
                    user_prompt=user_msg,
                    temp=0.3
                )
                longlist_objects.extend(chunk_results)

            # JSON als String formatieren, um ihn in die nächste Prompt-Stufe zu geben
            longlist_text = json.dumps(longlist_objects, indent=2, ensure_ascii=False)

            # --- Reduce anwenden (Editor) ---
            logger.info(f"Reducing ({len(longlist_objects)} Candidates)...")
            user_msg_reduce = self.prompts["editor_user"].format(n=max_highlights, longlist=longlist_text)
            
            final_objects = self.extract_highlights_gemini(
                sys_prompt=self.prompts["editor_system"],
                user_prompt=user_msg_reduce,
                temp=0.2
            )

            # Saubere Markdown-Datei mit Start -> Ende Syntax generieren
            final_clean = [f"- [{obj['start_timestamp']} -> {obj['end_timestamp']}] {obj['quote']}" for obj in final_objects]
            (out_dir / "highlights.md").write_text("\n".join(final_clean), encoding="utf-8")
        else:
            longlist = []
            for i, c in enumerate(hl_chunks):
                logger.info(f"Scout Chunk {i + 1}...")
                res = self.chat("Scout", self.prompts["map"].format(block=c), temp=0.3)
                longlist.extend([l for l in res.splitlines() if "[" in l])

            (out_dir / "highlights_longlist.md").write_text(
                "\n".join(longlist), encoding="utf-8"
            )

            # 3. Reduce anwenden
            logger.info(f"Reducing ({len(longlist)} Candidates)...")
            final_raw = self.chat(
                "Editor",
                self.prompts["reduce"].format(longlist="\n".join(longlist), n=max_highlights),
                temp=0.2,
            )
            final_clean = self._sanitize_highlights(final_raw, items, strict_snap_tol)

            if len(final_clean) < 6:
                logger.warning("Auffüllen aus Longlist...")
                more = self._sanitize_highlights("\n".join(longlist), items)
                for m in more:
                    if len(final_clean) >= 6:
                        break
                    if m not in final_clean:
                        final_clean.append(m)

            (out_dir / "highlights.md").write_text("\n".join(final_clean), encoding="utf-8")
        logger.info("Zusammenfassung und Highlights erfolgreich erstellt.")
        return {
            "summary_path": out_dir / "summary.md",
            "highlights_path": out_dir / "highlights.md"
        }
