#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
srt_summarizer_local.py (Vollständig entkoppelte Prompts)
---------------------------------------------------------
- WEICHE: Ollama (Strikte Regeln) / Gemini (Fließender Stil)
- PROMPTS: Komplett getrennte Logik für Zusammenfassung UND Highlights.
"""

import argparse, pathlib, re, math, json, sys, yaml, time, difflib
from datetime import timedelta
import srt
from loguru import logger
from core.time_utils import timedelta_to_hms, parse_time_to_seconds, hms_to_seconds
from core.prompts import OLLAMA_SUMMARIZER_PROMPTS as OLLAMA_PROMPTS, GEMINI_SUMMARIZER_PROMPTS as GEMINI_PROMPTS

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# ====================== DEFAULT MODEL ======================
DEFAULT_MODEL = "qwen2.5:14b-instruct-q6_K"

# ====================== Helpers ======================


def clean_thinking(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ====================== LLM Wrapper ======================


class UnifiedLLM:
    def __init__(self, config):
        self.provider = config.get("llm_provider", "ollama").lower()

        if self.provider == "gemini":
            # Hier nehmen wir gemini_model, falls vorhanden, sonst den Fallback
            self.model_name = config.get(
                "gemini_model", config.get("llm_model", "gemini-3.1-pro-preview")
            )
            if not genai:
                raise ImportError("Bitte `pip install google-genai`")
            self.client = genai.Client(api_key=config.get("gemini_api_key", ""))
            self.is_gemini = True
            logger.info(f"Gemini Mode: {self.model_name}")
        else:
            self.model_name = config.get("llm_model", DEFAULT_MODEL)
            if not OllamaClient:
                raise ImportError("Bitte `pip install ollama`")
            self.client = OllamaClient(host="http://127.0.0.1:11434")
            self.is_gemini = False
            logger.info(f"Ollama Mode: {self.model_name}")

    def chat(self, sys_p, user_p, temp=0.1):
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
                return clean_thinking(res.text)
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
                return clean_thinking(resp["message"]["content"])
            except Exception as e:
                logger.error(f"Ollama Error: {e}")
                return ""


# ====================== Core Logic ======================


def srt_items(path):
    for it in srt.parse(pathlib.Path(path).read_text(encoding="utf-8")):
        yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())


def build_highlight_candidates(sermon_items):
    return [
        f"[{timedelta_to_hms(it.start)}] {txt[:200]}"
        for it, txt in sermon_items
        if len(txt.split()) > 5
    ]


def sanitize_highlights(text, items, tol=8):
    out = []
    sec_map = {int(it.start.total_seconds()): timedelta_to_hms(it.start) for it, txt in items}
    used = set()
    line_re = re.compile(r".*?\[(\d{1,2})[:.](\d{1,2})[:.]?(\d{1,2})?\].*?[:\s]+(.*)")

    for line in text.splitlines():
        if "Zitat:" in line:
            line = line.replace("Das Zitat:", "")
        m = line_re.match(line)
        if not m:
            continue
        h, m_val, s_val, txt = (
            m.group(1),
            m.group(2),
            m.group(3) or 0,
            m.group(4),
        )
        sec = int(hms_to_seconds(h, m_val, s_val))

        if any(s in used for s in range(sec - 15, sec + 15)):
            continue

        found = None
        for off in range(-tol, tol + 1):
            if (sec + off) in sec_map:
                found = sec_map[sec + off]
                break

        if found:
            clean = (
                re.sub(r"^(Und|Aber|Denn)\s+", "", txt, flags=re.IGNORECASE)
                .strip('"')
                .strip()
            )
            if len(clean) > 8:
                out.append(f"- [{found}] {clean}")
                used.add(sec)
    return out


def run(args):
    conf = {}
    if pathlib.Path("config.yaml").exists():
        conf = (
            yaml.safe_load(pathlib.Path("config.yaml").read_text(encoding="utf-8"))
            or {}
        )

    # NUR überschreiben, wenn das Argument auch wirklich beim Aufruf gesetzt wurde
    # Wir prüfen, ob args.model ungleich dem DEFAULT_MODEL ist
    if args.model and args.model != DEFAULT_MODEL:
        if conf.get("llm_provider") == "gemini":
            conf["gemini_model"] = args.model
        else:
            conf["llm_model"] = args.model

    llm = UnifiedLLM(conf)

    # 3. Jetzt erst llm benutzen
    prompts = GEMINI_PROMPTS if llm.is_gemini else OLLAMA_PROMPTS
    prompt_map = prompts["map"]
    prompt_reduce = prompts["reduce"]
    prompt_chunk = prompts["chunk"]
    prompt_merge = prompts["merge"]

    c_size = 40000 if llm.is_gemini else 6000

    items = list(srt_items(args.srt))
    transcript = "\n".join([f"[{timedelta_to_hms(it.start)}] {txt}" for it, txt in items])
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript_predigt.txt").write_text(transcript, encoding="utf-8")

    # 2. Summary erstellen
    chunks = [transcript[i : i + c_size] for i in range(0, len(transcript), c_size)]
    parts = []
    logger.info(f"--- Summary ({len(chunks)} Chunks) ---")
    for c in chunks:
        parts.append(
            llm.chat("Du bist Redakteur.", prompt_chunk.format(chunk=c), temp=0.1)
        )

    full_sum = (
        parts[0]
        if len(parts) == 1
        else llm.chat(
            "Du bist Redakteur.", prompt_merge.format(parts="\n".join(parts)), temp=0.1
        )
    )
    (out_dir / "summary.md").write_text(full_sum, encoding="utf-8")

    # 3. Highlights (Scout/Editor) extrahieren
    logger.info("--- Highlights Scouting ---")
    hl_chunks = []
    curr, l = [], 0
    cands = build_highlight_candidates(items)
    for c in cands:
        if l + len(c) > c_size:
            hl_chunks.append("\n".join(curr))
            curr, l = [], 0
        curr.append(c)
        l += len(c)
    if curr:
        hl_chunks.append("\n".join(curr))

    longlist = []
    for i, c in enumerate(hl_chunks):
        logger.info(f"Scout Chunk {i + 1}...")
        res = llm.chat("Scout", prompt_map.format(block=c), temp=0.3)
        longlist.extend([l for l in res.splitlines() if "[" in l])

    (out_dir / "highlights_longlist.md").write_text(
        "\n".join(longlist), encoding="utf-8"
    )

    # 4. Reduce anwenden
    logger.info(f"Reducing ({len(longlist)} Candidates)...")
    final_raw = llm.chat(
        "Editor",
        prompt_reduce.format(longlist="\n".join(longlist), n=args.max_highlights),
        temp=0.2,
    )
    final_clean = sanitize_highlights(final_raw, items, args.strict_snap_tol)

    if len(final_clean) < 6:
        logger.warning("Auffüllen aus Longlist...")
        more = sanitize_highlights("\n".join(longlist), items)
        for m in more:
            if len(final_clean) >= 6:
                break
            if m not in final_clean:
                final_clean.append(m)

    (out_dir / "highlights.md").write_text("\n".join(final_clean), encoding="utf-8")
    logger.info("Fertig.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out_dir", default="out_test")
    ap.add_argument("--sermon-start")
    ap.add_argument("--sermon-end")
    ap.add_argument("--auto-sermon", action="store_true")
    ap.add_argument("--export-sermon-srt")
    ap.add_argument("--strict-snap-tol", type=int, default=8)
    ap.add_argument("--max-highlights", type=int, default=8)
    ap.add_argument("--min-gap-sec")
    ap.add_argument("--min-sermon-minutes")
    ap.add_argument("--fix-spelling", action="store_true")
    ap.add_argument("--preset")
    ap.add_argument("--timeout")
    ap.add_argument("--strict-times", action="store_true")
    ap.add_argument("--strict-chapters", action="store_true")
    ap.add_argument("--strict-drop-on-miss", action="store_true")
    ap.add_argument("--chapters-snap-tol")
    ap.add_argument("--chapters-drop-on-miss", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--church-yaml")
    ap.add_argument("--social-only", action="store_true")

    ap.add_argument("--model", default=DEFAULT_MODEL)
    run(ap.parse_args())
