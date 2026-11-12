#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
import srt
from loguru import logger


def filter_sermon_items(items, start_sec: float, end_sec: float):
    """Keep only subtitles overlapping the requested sermon window.

    Subtitles are clipped to the window so the exported SRT is valid and
    contains only the processed sermon section.
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
                        content=txt,
                    ),
                    txt,
                )
            )
    return filtered


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

        clip_end_abs = start_abs + duration
        all_words = []
        for seg in data.get("segments", []):
            for w in seg.get("words", []):
                if w["end"] > start_abs and w["start"] < clip_end_abs:
                    w_clean = w.copy()
                    if w_clean["start"] < start_abs:
                        w_clean["start"] = start_abs
                    if w_clean["end"] > clip_end_abs:
                        w_clean["end"] = clip_end_abs
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

    def load_words_from_json(self, json_path: Path) -> List[dict]:
        """Extract the flat word list from a WhisperX JSON file."""
        if not json_path or not json_path.exists():
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Warnung beim Laden der Wortliste: {e}")
            return []

        words = []
        for seg in data.get("segments", []):
            words.extend(seg.get("words", []))
        if not words:
            words = data.get("words", [])
        words = [
            w for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
        words.sort(key=lambda w: float(w["start"]))
        return words

    def find_prosodic_smart_end(
        self,
        words: list[dict],
        start_sec: float,
        min_dur: float = 18.0,
        max_dur: float = 55.0,
        min_pause_sec: float = 0.35,
    ) -> float:
        """Find a sentence boundary using punctuation, pauses, and target clip length."""
        if not words:
            return start_sec + max_dur

        forbidden_next = {"aber", "doch", "allerdings", "denn", "weil", "obwohl", "sondern", "und"}
        best_end = None
        best_score = -1.0

        for i, w in enumerate(words):
            w_end = w.get("end", 0.0)
            dur = w_end - start_sec
            if dur < min_dur:
                continue
            if dur > max_dur:
                break

            text = w.get("word", "").strip()
            if not text.endswith((".", "!", "?")):
                continue

            next_word = words[i + 1] if i + 1 < len(words) else None
            pause = (next_word["start"] - w_end) if next_word and "start" in next_word else 1.0
            next_clean = next_word.get("word", "").strip().lower().strip(",.!?") if next_word else ""

            if next_clean in forbidden_next:
                continue

            # Prefer longer pauses and target clip lengths around 30 seconds
            pause_score = min(pause / min_pause_sec, 3.0)
            length_penalty = abs(dur - 30.0) * 0.02
            score = pause_score - length_penalty

            if pause >= min_pause_sec and score > best_score:
                best_score = score
                best_end = w_end

        return (best_end + 0.25) if best_end else min(start_sec + max_dur, words[-1].get("end", start_sec + max_dur))

    def snap_end_to_speech_pause(
        self,
        words: List[dict],
        chosen_end: float,
        start_sec: float,
        min_duration: float,
        max_duration: float,
        tolerance: float = 2.5,
        min_pause_sec: float = 0.35,
    ) -> float:
        """Snap a chosen clip end to a nearby word end followed by a real speech pause."""
        if not words:
            return chosen_end

        words = sorted(words, key=lambda w: float(w.get("start", 0.0) or 0.0))
        effective_min = start_sec + min_duration
        effective_max = start_sec + max_duration
        search_min = max(effective_min, chosen_end - tolerance)
        search_max = min(effective_max, chosen_end + tolerance)

        candidates = []
        for i, w in enumerate(words):
            try:
                w_end = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue

            if w_end < search_min or w_end > search_max:
                continue

            next_word = words[i + 1] if i + 1 < len(words) else None
            if next_word is not None:
                try:
                    next_start = float(next_word["start"])
                except (KeyError, TypeError, ValueError):
                    next_start = w_end
                pause = next_start - w_end
            else:
                pause = 1.0

            if pause < min_pause_sec:
                continue

            word_text = str(w.get("word", "")).strip()
            has_punctuation = word_text.endswith((".", "!", "?"))
            distance = abs(w_end - chosen_end)

            # Prefer punctuation, longer pauses, and closeness to the chosen end.
            score = (
                (1.0 if has_punctuation else 0.0)
                + min(pause, 1.0)
                - distance * 0.1
            )
            candidates.append((score, w_end))

        if not candidates:
            return chosen_end

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] + 0.15

    def snap_start_to_sentence_begin(
        self,
        words: List[dict],
        approx_start_sec: float,
        search_window: float = 3.0,
        min_pause: float = 0.30,
    ) -> float:
        """Snap a chosen clip start to a nearby clean sentence beginning."""
        if not words:
            return approx_start_sec

        words = sorted(words, key=lambda w: float(w.get("start", 0.0) or 0.0))
        search_min = max(0.0, approx_start_sec - search_window)
        search_max = approx_start_sec + search_window

        candidates = []
        for i, w in enumerate(words):
            try:
                w_start = float(w["start"])
            except (KeyError, TypeError, ValueError):
                continue

            if w_start < search_min or w_start > search_max:
                continue

            current_text = str(w.get("word", "")).strip()
            if not current_text:
                continue

            current_clean = current_text.strip(",.!?").lower()
            starts_upper = current_text[0].isupper()
            not_conjunction = current_clean not in {
                "und",
                "aber",
                "denn",
                "weil",
                "oder",
            }

            if i == 0:
                pause = 1.0
                has_punctuation = False
                is_first = True
            else:
                prev = words[i - 1]
                try:
                    prev_end = float(prev["end"])
                except (KeyError, TypeError, ValueError):
                    prev_end = w_start

                pause = w_start - prev_end
                if pause < min_pause:
                    continue

                prev_text = str(prev.get("word", "")).strip()
                has_punctuation = prev_text.endswith((".", "!", "?"))
                is_first = False

            distance = abs(w_start - approx_start_sec)
            score = (
                (3.0 if has_punctuation else 0.0)
                + (2.0 if starts_upper and not_conjunction else 0.0)
                + min(pause, 1.0)
                - distance * 0.3
                + (1.0 if is_first else 0.0)
            )
            candidates.append((score, w_start))

        if not candidates:
            return approx_start_sec

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_start = candidates[0][1]
        return max(0.0, best_start - 0.10)
