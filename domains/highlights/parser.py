#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from core.time_utils import hms_to_seconds, parse_time_to_seconds
from core.span_contract import ensure_no_finalized_overlap, parse_finalized_flag
from domains.highlights.models import HighlightPayload, HighlightItem

@dataclass
class Highlight:
    start: float
    end: Optional[float] = None
    label: str = ""
    finalized: bool = False

class HighlightParser:
    """
    Klasse zum Parsen von Highlight-Definitionen aus JSON-Dateien.
    Gibt die klassische Highlight-Struktur für die Rendering-Pipeline zurück.
    """

    def __init__(self):
        pass

    def load_highlights_from_json(self, json_path: Path) -> List[Highlight]:
        if not json_path.exists():
            return []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Legacy fallback rule:
        # - reel_spans.json without metadata -> finalized
        # - any other artifact without metadata -> candidate
        # Explicit finalized metadata always wins, even after renaming/copying.
        legacy_finalized = json_path.name == "reel_spans.json"

        if isinstance(data, dict) and "highlights" in data:
            if legacy_finalized:
                # Legacy dict payload named reel_spans.json: keep its spans
                # finalized and do not run candidate overlap selection.
                items = data["highlights"]
                default_finalized = True
            else:
                payload = HighlightPayload.model_validate(data)
                items = payload.highlights
                default_finalized = False
        elif isinstance(data, list):
            items = data
            default_finalized = legacy_finalized
        elif isinstance(data, dict):
            items = data.get("highlights", [])
            default_finalized = legacy_finalized
        else:
            return []

        hl: List[Highlight] = []
        for item in items:
            if isinstance(item, HighlightItem):
                start = float(item.start)
                end = float(item.end)
                label = item.title
                finalized = bool(item.finalized)
            elif isinstance(item, dict):
                start_val = item.get("start", 0.0)
                end_val = item.get("end")
                start = parse_time_to_seconds(start_val)
                end = parse_time_to_seconds(end_val) if end_val is not None else None
                label = str(item.get("title", item.get("label", ""))).strip()
                finalized = parse_finalized_flag(item.get("finalized"), default_finalized)
            else:
                continue
            hl.append(Highlight(start=start, end=end, label=label, finalized=finalized))
        ensure_no_finalized_overlap(hl)
        return hl

    def load_highlights_from_md(self, md_path: Path) -> List[Highlight]:
        """Parse the legacy markdown highlight format."""
        if not md_path.exists():
            return []

        text = md_path.read_text(encoding="utf-8")
        range_re = re.compile(
            r"\[(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*(?:->|→)\s*"
            r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\]\s*(.*)"
        )
        single_re = re.compile(
            r"\[(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\]\s*(.*)"
        )

        hl: List[Highlight] = []
        for line in text.splitlines():
            m = range_re.search(line)
            if m:
                start = hms_to_seconds(m.group(1), m.group(2), m.group(3))
                end = hms_to_seconds(m.group(4), m.group(5), m.group(6))
                hl.append(Highlight(start=start, end=end, label=m.group(7).strip(), finalized=True))
                continue

            m = single_re.search(line)
            if m:
                start = hms_to_seconds(m.group(1), m.group(2), m.group(3))
                hl.append(Highlight(start=start, end=None, label=m.group(4).strip(), finalized=False))
        ensure_no_finalized_overlap(hl)
        return hl
