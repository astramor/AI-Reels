#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
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
