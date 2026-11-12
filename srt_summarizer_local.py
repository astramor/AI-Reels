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
import math
import pathlib
import re
import sys

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
from core.time_utils import parse_time_to_seconds, seconds_to_hms_ms
from domains.transcription.processor import filter_sermon_items


_SERMON_TIME_RE = re.compile(r"\d+(?:\.\d+)?(?::\d+(?:\.\d+)?){0,2}")


def _parse_sermon_boundary(value: str, label: str) -> float:
    text = str(value).strip()
    if not _SERMON_TIME_RE.fullmatch(text):
        raise ValueError(f"Ungültige {label}: {value!r}")
    parsed = parse_time_to_seconds(text)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Ungültige {label}: {value!r}")
    return parsed


def srt_items(path):
    for it in srt.parse(pathlib.Path(path).read_text(encoding="utf-8")):
        yield it, re.sub(r"\s+", " ", it.content.replace("\n", " ").strip())


def build_highlight_candidates(sermon_items):
    return [
        {
            "start": float(it.start.total_seconds()),
            "text": txt[:200],
        }
        for it, txt in sermon_items
        if len(txt.split()) > 5
    ]


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().strip('"').lower()


def _text_match_kind(candidate_text, output_text):
    candidate = _normalize_text(candidate_text)
    output = _normalize_text(output_text)
    if not candidate or not output:
        return None
    if candidate == output:
        return "exact"
    if output in candidate or candidate in output:
        return "substring"
    return None


def _select_candidate(candidates, txt, context):
    """Select one candidate only when the text match is unambiguous."""
    if len(candidates) == 1:
        return candidates[0]

    exact_matches = [
        r for r in candidates if _text_match_kind(r["text"], txt) == "exact"
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        logger.warning(
            f"{context}: mehrdeutiger exakter Texttreffer; Referenz übersprungen."
        )
        return None

    substring_matches = [
        r for r in candidates if _text_match_kind(r["text"], txt) == "substring"
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        logger.warning(
            f"{context}: mehrdeutiger Teiltext-Treffer; Referenz übersprungen."
        )
        return None

    logger.warning(f"{context}: kein eindeutiger Texttreffer; Referenz übersprungen.")
    return None


def _find_candidate(sec, txt, records, tol, legacy_whole_second):
    """Map an LLM timestamp back to the original fractional subtitle start.

    Fractional LLM timestamps must match exactly. Legacy whole-second output
    uses the same-second/nearest compatibility window, but only unambiguous
    text matches are accepted when more than one candidate is eligible.
    """
    exact = [r for r in records if abs(r["start"] - sec) < 1e-6]
    if exact:
        return _select_candidate(exact, txt, "exakte Zeitstempel-Referenz")

    if not legacy_whole_second:
        logger.warning(
            f"Exakte Zeitstempel-Referenz {sec:.3f}: kein exakter Kandidat."
        )
        return None

    sec_int = int(round(sec))
    same_second = [
        r
        for r in records
        if int(round(r["start"])) == sec_int or int(r["start"]) == sec_int
    ]
    if same_second:
        return _select_candidate(
            same_second, txt, "legacy whole-second same-second"
        )

    near = [r for r in records if abs(r["start"] - sec) <= tol]
    if near:
        return _select_candidate(near, txt, "legacy whole-second near")

    logger.warning(
        f"Legacy-Zeitstempel {sec:.3f}: kein passender Kandidat."
    )
    return None


def sanitize_highlights(text, items, tol=8):
    records = [
        {"start": float(it.start.total_seconds()), "text": txt}
        for it, txt in items
    ]
    out = []
    used_starts = []
    line_re = re.compile(
        r"\[(\d{1,2})[:.](\d{1,2})(?:[:.](\d{1,2}(?:\.\d+)?))?\]\s*(.*)"
    )

    for line in text.splitlines():
        if "Zitat:" in line:
            line = line.replace("Das Zitat:", "")

        m = line_re.search(line)
        if not m:
            continue

        h = int(m.group(1))
        minute = int(m.group(2))
        second_text = m.group(3) or "0"
        second = float(second_text)
        txt = m.group(4).strip()
        sec = h * 3600 + minute * 60 + second
        legacy_whole_second = "." not in second_text

        candidate = _find_candidate(
            sec, txt, records, tol, legacy_whole_second
        )
        if not candidate:
            continue

        start = candidate["start"]
        if any(abs(start - used) < 15.0 for used in used_starts):
            continue

        clean = re.sub(
            r"^(Und|Aber|Denn)\s+", "", txt, flags=re.IGNORECASE
        ).strip('"').strip()
        if len(clean) > 8:
            out.append({"start": start, "text": clean})
            used_starts.append(start)
        else:
            logger.warning(
                f"Zeitstempel-Referenz {sec:.3f}: Text zu kurz/unsicher; übersprungen."
            )

    return out


def _dedupe_highlights(highlights):
    seen = set()
    out = []
    for highlight in highlights:
        key = (round(float(highlight["start"]), 6), highlight["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(highlight)
    return out


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
    if (
        not isinstance(args.max_highlights, int)
        or isinstance(args.max_highlights, bool)
        or args.max_highlights <= 0
    ):
        raise ValueError("max_highlights muss eine positive Ganzzahl sein")

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
        t0 = _parse_sermon_boundary(args.sermon_start, "sermon_start")
    if args.sermon_end:
        t1 = _parse_sermon_boundary(args.sermon_end, "sermon_end")

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
            f"[{seconds_to_hms_ms(it.start.total_seconds())}] {txt}"
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
    candidate_lines = [
        f"[{seconds_to_hms_ms(c['start'])}] {c['text']}" for c in candidates
    ]
    hl_chunks = []
    current_lines = []
    current_len = 0

    for candidate in candidate_lines:
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

    if len(final_clean) < args.max_highlights:
        logger.warning("Auffüllen aus Longlist...")
        more = sanitize_highlights(
            "\n".join(longlist), items, args.strict_snap_tol
        )
        for entry in more:
            if len(final_clean) >= args.max_highlights:
                break
            if entry not in final_clean:
                final_clean.append(entry)

    if not final_clean and candidates:
        logger.warning("Keine kuratierten Highlights; nutze Kandidaten-Fallback.")
        final_clean = candidates[: args.max_highlights]

    final_clean = _dedupe_highlights(final_clean)
    final_clean = final_clean[: args.max_highlights]

    (out_dir / "highlights.md").write_text(
        "\n".join(
            f"- [{seconds_to_hms_ms(h['start'])}] {h['text']}"
            for h in final_clean
        ),
        encoding="utf-8",
    )

    # 8. Write strict JSON for build_highlight_spans.py / batch_reels.py.
    payload = []
    for highlight in final_clean:
        start = float(highlight["start"])
        text = highlight["text"]

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
    ap.add_argument("--llm-provider", default=None)
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
