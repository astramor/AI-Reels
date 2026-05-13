#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Optional
import srt
from loguru import logger

class SubtitleProcessor:
    """
    Klasse für die Verarbeitung von Transkriptionsdaten und die Generierung von Untertiteln.
    """

    def __init__(self):
        pass

    def fmt_ass_time(self, seconds: float) -> str:
        seconds = max(0, seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds * 100) % 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def snap_start_to_json_word(
        self, json_path: Path, approximate_start: float, window: float = 0.5
    ) -> float:
        if not json_path or not json_path.exists():
            return approximate_start
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidates = []
            for seg in data.get("segments", []):
                if seg["end"] < (approximate_start - window):
                    continue
                if seg["start"] > (approximate_start + window):
                    break
                for w in seg.get("words", []):
                    diff = abs(w["start"] - approximate_start)
                    if diff <= window:
                        candidates.append((diff, w["start"]))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_diff, best_start = candidates[0]
                if best_diff < 0.3:
                    adjusted_start = max(0, best_start - 0.05)
                    logger.info(
                        f"   [Timing Fix] Snap: {approximate_start:.2f}s -> {adjusted_start:.2f}s"
                    )
                    return adjusted_start
                else:
                    logger.info(
                        f"   [Timing] Snap abgelehnt (Diff {best_diff:.2f}s zu groß). Nutze SRT-Zeit: {approximate_start:.2f}s"
                    )

        except Exception as e:
            logger.warning(f"Warnung beim Time-Snapping: {e}")

        return approximate_start

    def generate_karaoke_ass(
        self, json_path: Path, start_abs: float, duration: float, out_ass: Path, margin_v: int = 250
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = []
        COL_YELLOW = "&H0000FFFF&"
        COL_WHITE = "&H00FFFFFF&"

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,85,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        all_words = []
        for seg in data.get("segments", []):
            for w in seg.get("words", []):
                if w["end"] > start_abs:
                    w_clean = w.copy()
                    if w_clean["start"] < start_abs:
                        w_clean["start"] = start_abs
                    all_words.append(w_clean)

        if not all_words:
            logger.warning("   -> Warnung: Keine Wörter im Zeitraum gefunden.")
            return

        chunks = []
        current_chunk = []
        MAX_WORDS_PER_SCREEN = 6

        for i, w in enumerate(all_words):
            current_chunk.append(w)
            word_text = w["word"].strip()
            is_punctuation = word_text.endswith((".", "!", "?", ",", ":"))

            if (
                len(current_chunk) >= MAX_WORDS_PER_SCREEN
                or (is_punctuation and len(current_chunk) > 3)
                or i == len(all_words) - 1
            ):
                chunks.append(current_chunk)
                current_chunk = []

        event_count = 0

        for chunk in chunks:
            if not chunk:
                continue

            chunk_end_abs = chunk[-1]["end"]
            chunk_end_rel = max(0.0, chunk_end_abs - start_abs)

            current_idx = chunks.index(chunk)
            if current_idx < len(chunks) - 1:
                next_start_abs = chunks[current_idx + 1][0]["start"]
                next_start_rel = max(0.0, next_start_abs - start_abs)
                if (next_start_rel - chunk_end_rel) < 0.8:
                    chunk_end_rel = next_start_rel

            for i, w in enumerate(chunk):
                w_start_rel = max(0.0, w["start"] - start_abs)
                if i < len(chunk) - 1:
                    hl_end_rel = max(0.0, chunk[i + 1]["start"] - start_abs)
                else:
                    hl_end_rel = chunk_end_rel

                line_str = ""
                for j, other_w in enumerate(chunk):
                    txt = other_w["word"].strip()
                    if j == i:
                        line_str += f"{{\\c{COL_YELLOW}}}{txt} "
                    elif j < i:
                        line_str += f"{{\\c{COL_WHITE}}}{txt} "
                    else:
                        line_str += f"{{\\c{COL_WHITE}}}{txt} "

                event_start = 0.0 if event_count == 0 else w_start_rel
                if hl_end_rel <= event_start:
                    hl_end_rel = event_start + 0.1

                events.append(
                    f"Dialogue: 0,{self.fmt_ass_time(event_start)},{self.fmt_ass_time(hl_end_rel)},Default,,0,0,0,,{line_str.strip()}"
                )
                event_count += 1

        with open(out_ass, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))
        logger.info(f"   -> {event_count} Karaoke-Events erstellt (Präzisions-Modus).")

    def load_srt_file(self, srt_path: Path) -> List[srt.Subtitle]:
        return list(srt.parse(srt_path.read_text(encoding="utf-8")))

    def find_smart_end(
        self, srt_items: List[srt.Subtitle], start_sec: float, min_dur: float = 18.0, max_dur: float = 60.0
    ) -> float:
        """Findet das nächste logische Satzende im SRT für einen Smart Cut."""
        import re
        end_sec = start_sec + 20.0 # Fallback
        for it in srt_items:
            s_end = it.end.total_seconds()
            if s_end > start_sec + min_dur:
                if re.search(r"[.!?]", it.content):
                    return s_end
            if s_end > start_sec + max_dur:
                return s_end
        return end_sec
