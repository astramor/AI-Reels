#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from core.time_utils import hms_to_seconds
from domains.highlights.models import HighlightPayload

@dataclass
class Highlight:
    start: float
    end: Optional[float] = None
    label: str = ""

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
            
        # Pydantic Payload validieren (falls die Datei manuell editiert wurde)
        payload = HighlightPayload.model_validate(data)
        
        hl: List[Highlight] = []
        for item in payload.highlights:
            hl.append(Highlight(
                start=item.start,
                end=item.end,
                label=item.title
            ))
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
                hl.append(Highlight(start=start, end=end, label=m.group(7).strip()))
                continue

            m = single_re.search(line)
            if m:
                start = hms_to_seconds(m.group(1), m.group(2), m.group(3))
                hl.append(Highlight(start=start, end=None, label=m.group(4).strip()))
        return hl
