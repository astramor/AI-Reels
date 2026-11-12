#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
srt_summarizer_local.py
-----------------------
Zentrale Zusammenfassung und Highlight-Extraktion.

Nutzt UnifiedLLM aus core/llm_client.py. Cloud-Provider (OpenRouter/OpenAI/
DeepSeek/Gemini) erhalten große Chunks und kreative Cloud-Prompts; lokale
Ollama-Provider erhalten kleinere Chunks und strikte Ollama-Prompts.

Ausgaben:
- summary.md
- highlights_longlist.md
- highlights.md
- highlights.json (für build_highlight_spans.py / batch_reels.py)
- optional exportiertes, gefiltertes Predigt-SRT
"""

import argparse
import json
import pathlib
import re
import sys
from datetime import timedelta

import srt
import yaml
from loguru import logger

from core.llm_client import UnifiedLLM
from core.prompts import (
    OLLAMA_PROMPT_MAP,
    OLLAMA_PROMPT_REDUCE,
    OLLAMA_PROMPT_CHUNK,
    OLLAMA_PROMPT_MERGE,
    GEMINI_PROMPT_MAP,
    GEMINI_PROMPT_REDUCE,
    GEMINI_PROMPT_CHUNK,
    GEMINI_PROMPT_MERGE,
)
from core.time_utils import parse_time_to_seconds, seconds_to_hms


def srt_items(path):
    for it in srt.parse(pathlib.Path(path).read_text(encoding="utf-8")):
        yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())


def build_highlight_candidates(sermon_items):
    return [
        f"[{seconds_to_hms(it.start.total_seconds())}] {txt[:200]}"
        for it, txt in sermon_items
        if len(txt.split()) > 5
    ]


def sanitize_highlights(text, items, tol=8):
    out = []
    sec_map = {
        int(round(it.start.total_seconds())): seconds_to_hms(it.start.total_seconds())
        for it, _txt in items
    }
    used = set()
    line_re = re.compile(
        r"\[(\d{1,2})[:.](\d{1,2})(?:[:.](\d{1,2}))?\]\s*(.*)"
    )

    for line in text.splitlines():
        if "Zitat:" in line:
            line = line.replace("Das Zitat:", "")

        m = line_re.search(line)
        if not m:
            continue

        h = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3) or 0)
        txt = m.group(4).strip()
        sec = h * 3600 + minute * 60 + second

        if any(s in used for s in range(sec - 15, sec + 15)):
            continue

        found = None
        for off in range(-tol, tol + 1):
            if (sec + off) in sec_map:
                found = sec_map[sec + off]
                break

        if found:
            clean = re.sub(
                r"^(Und|Aber|Denn)\s+", "", txt, flags=re.IGNORECASE
            ).strip('"').strip()
            if len(clean) > 8:
                out.append(f"- [{found}] {clean}")
                used.add(sec)

    return out


def filter_sermon_items(items, start_sec, end_sec):
    """Keep only subtitles overlapping the requested sermon window.

    Overlapping subtitles are clipped to the window so the exported SRT is
    valid and contains only the processed sermon section.
    """
    filtered = []
    for it, txt in items:
        s = max(it.start.total_seconds(), start_sec)
        e = min(it.end.total_seconds(), end_sec)
        if e > s:
            filtered.append(
                (
                    srt.Subtitle(
                        index=len(filtered) + 1,
                        start=timedelta(seconds=s),
                        end=timedelta(seconds=e),
                        content=it.content,
                    ),
                    txt,
                )
            )
    return filtered


def _map_model_to_provider_field(conf, provider, model):
    """Write the explicit model into the provider-specific config field."""
    if provider == "openrouter":
        nested = conf.get("openrouter")
        if isinstance(nested, dict):
            nested["model"] = model
        else:
            conf["openrouter_model"] = model
    elif provider == "openai":
        conf["openai_model"] = model
    elif provider == "gemini":
        conf["gemini_model"] = model
    elif provider == "ollama":
        conf["ollama_model"] = model
    elif provider == "llama-server":
        conf["llama_server_model"] = model
    else:
        conf["llm_model"] = model


def run(args):
    conf = {}
    if pathlib.Path("config.yaml").exists():
        conf = yaml.safe_load(
            pathlib.Path("config.yaml").read_text(encoding="utf-8")
        ) or {}

    # 1. Provider override must happen BEFORE UnifiedLLM is initialized.
    if args.llm_provider:
        conf["llm_provider"] = args.llm_provider

    provider = str(conf.get("llm_provider", "ollama")).lower()
    if provider == "gemini-api":
        provider = "gemini"

    # 2. Explicit model override mapped to the active provider's field.
    if args.model:
        _map_model_to_provider_field(conf, provider, args.model)

    llm = UnifiedLLM(conf)

    # 3. Prompt and chunk selection based on cloud/local, not Gemini-only.
    prompt_map = GEMINI_PROMPT_MAP if llm.is_cloud_llm else OLLAMA_PROMPT_MAP
    prompt_reduce = (
        GEMINI_PROMPT_REDUCE if llm.is_cloud_llm else OLLAMA_PROMPT_REDUCE
    )
    prompt_chunk = GEMINI_PROMPT_CHUNK if llm.is_cloud_llm else OLLAMA_PROMPT_CHUNK
    prompt_merge = GEMINI_PROMPT_MERGE if llm.is_cloud_llm else OLLAMA_PROMPT_MERGE
    c_size = 40000 if llm.is_cloud_llm else 6000

    items_all = list(srt_items(args.srt))
    if not items_all:
        raise ValueError("Das SRT-Transkript ist leer.")

    t0 = items_all[0][0].start.total_seconds()
    t1 = items_all[-1][0].end.total_seconds()

    if args.sermon_start:
        t0 = parse_time_to_seconds(args.sermon_start)
    if args.sermon_end:
        t1 = parse_time_to_seconds(args.sermon_end)

    if t1 <= t0:
        raise ValueError(
            f"Ungültiges Predigt-Fenster: start={t0:.2f}s, end={t1:.2f}s"
        )

    items = filter_sermon_items(items_all, t0, t1)
    if not items:
        raise ValueError("Keine Untertitel im angegebenen Predigt-Fenster.")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Export the processed/filtered sermon SRT.
    if args.export_sermon_srt:
        export_path = pathlib.Path(args.export_sermon_srt)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            srt.compose([it for it, _txt in items]), encoding="utf-8"
        )
        logger.info(f"Predigt-SRT exportiert: {export_path}")

    transcript = "\n".join(
        [
            f"[{seconds_to_hms(it.start.total_seconds())}] {txt}"
            for it, txt in items
        ]
    )
    (out_dir / "transcript_predigt.txt").write_text(transcript, encoding="utf-8")

    # 5. Summary
    chunks = [
        transcript[i : i + c_size]
        for i in range(0, len(transcript), c_size)
    ]
    parts = []
    logger.info(f"--- Summary ({len(chunks)} Chunks, c_size={c_size}) ---")
    for chunk in chunks:
        parts.append(
            llm.chat(
                "Du bist Redakteur.",
                prompt_chunk.format(chunk=chunk),
                temp=0.1,
            )
        )

    if len(parts) == 1:
        full_sum = parts[0]
    else:
        full_sum = llm.chat(
            "Du bist Redakteur.",
            prompt_merge.format(parts="\n".join(parts)),
            temp=0.1,
        )
    (out_dir / "summary.md").write_text(full_sum, encoding="utf-8")

    # 6. Highlights scouting
    logger.info("--- Highlights Scouting ---")
    candidates = build_highlight_candidates(items)
    hl_chunks = []
    current_lines = []
    current_len = 0

    for candidate in candidates:
        if current_len + len(candidate) > c_size:
            hl_chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(candidate)
        current_len += len(candidate)

    if current_lines:
        hl_chunks.append("\n".join(current_lines))

    longlist = []
    for i, chunk in enumerate(hl_chunks):
        logger.info(f"Scout Chunk {i + 1}/{len(hl_chunks)}...")
        res = llm.chat("Scout", prompt_map.format(block=chunk), temp=0.3)
        longlist.extend([line for line in res.splitlines() if "[" in line])

    (out_dir / "highlights_longlist.md").write_text(
        "\n".join(longlist), encoding="utf-8"
    )

    # 7. Reduce / curation
    logger.info(f"Reducing ({len(longlist)} Candidates)...")
    final_raw = llm.chat(
        "Editor",
        prompt_reduce.format(
            longlist="\n".join(longlist),
            n=args.max_highlights,
        ),
        temp=0.2,
    )
    final_clean = sanitize_highlights(final_raw, items, args.strict_snap_tol)

    if len(final_clean) < 6:
        logger.warning("Auffüllen aus Longlist...")
        more = sanitize_highlights(
            "\n".join(longlist), items, args.strict_snap_tol
        )
        for entry in more:
            if len(final_clean) >= 6:
                break
            if entry not in final_clean:
                final_clean.append(entry)

    if not final_clean and candidates:
        logger.warning("Keine kuratierten Highlights; nutze Kandidaten-Fallback.")
        final_clean = [
            f"- {candidate}" for candidate in candidates[: args.max_highlights]
        ]

    (out_dir / "highlights.md").write_text(
        "\n".join(final_clean), encoding="utf-8"
    )

    # 8. Write strict JSON for build_highlight_spans.py / batch_reels.py.
    payload = []
    for line in final_clean:
        m = re.search(
            r"\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.*)", line
        )
        if not m:
            continue

        start = float(
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + int(m.group(3))
        )
        text = m.group(4).strip()

        end = min(start + 30.0, t1)
        if end <= start:
            end = start + 1.0

        payload.append(
            {
                "title": text[:100],
                "start": start,
                "end": end,
                "reason": text,
                "hook": text,
                "confidence": 0.8,
            }
        )

    (out_dir / "highlights.json").write_text(
        json.dumps({"highlights": payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.success(f"Highlights gespeichert in: {out_dir / 'highlights.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out_dir", default="out_summ")
    ap.add_argument("--max-highlights", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-provider", default="")
    ap.add_argument("--export-sermon-srt")
    ap.add_argument("--sermon-start")
    ap.add_argument("--sermon-end")
    ap.add_argument("--strict-snap-tol", type=int, default=8)
    ap.add_argument("--fix-spelling", action="store_true")
    ap.add_argument("--auto-sermon", action="store_true")
    ap.add_argument("--min-sermon-minutes", type=int)
    ap.add_argument("--preset")
    ap.add_argument("--min-gap-sec", type=int)
    ap.add_argument("--timeout", type=int)

    args = ap.parse_args()

    try:
        run(args)
    except Exception as e:
        logger.exception(f"Fehler im Summarizer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
