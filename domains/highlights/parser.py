#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from core.time_utils import hms_to_seconds

@dataclass
class Highlight:
    start: int
    end: Optional[int] = None
    label: str = ""

class HighlightParser:
    """
    Klasse zum Parsen von Highlight-Definitionen aus Markdown-Dateien.
    """

    LINE_RE = re.compile(
        r"^\s*(?:[-–—*]\s*)?\[(\d{2}):(\d{2}):(\d{2})(?:\s*(?:->|→)\s*(\d{2}):(\d{2}):(\d{2}))?\]\s*(.+?)\s*$"
    )

    def __init__(self):
        pass

    def load_highlights_from_md(self, md_path: Path) -> List[Highlight]:
        text = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        hl: List[Highlight] = []
        for line in text.splitlines():
            m = self.LINE_RE.match(line)
            if not m:
                continue
            g = m.groups()
            start = hms_to_seconds(g[0], g[1], g[2])
            end = hms_to_seconds(g[3], g[4], g[5]) if g[3] and g[4] and g[5] else None
            hl.append(Highlight(start, end, g[6].strip()))
        return hl
