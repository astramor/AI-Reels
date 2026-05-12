import pytest
import re
from pathlib import Path
from domains.highlights.parser import HighlightParser
from core.config import settings
from batch_reels import load_srt_file, get_srt_slice

def test_smoke_pipeline_integration():
    """
    Smoke-Test: Prüft das Zusammenspiel von Parser, SRT-Loading und Filename-Generierung.
    Kein echtes Rendering, keine Subprozesse.
    """
    # 1. Parser-Check
    md_path = Path("tests/fixtures/sample_highlights.md")
    if not md_path.exists():
        pytest.skip("Fixture sample_highlights.md nicht gefunden")
        
    parser = HighlightParser()
    highlights = parser.load_highlights_from_md(md_path)
    
    assert len(highlights) == 5
    # Erster Clip: [00:00:10 -> 00:00:20] Erster Clip
    assert highlights[0].start == 10.0
    assert highlights[0].end == 20.0
    assert highlights[0].label == "Erster Clip"

    # 2. SRT-Check
    srt_path = Path("tests/fixtures/sample.srt")
    if not srt_path.exists():
        pytest.skip("Fixture sample.srt nicht gefunden")
        
    subs = load_srt_file(srt_path)
    assert len(subs) == 2
    assert "Hallo Welt" in subs[0].content

    # 3. SRT-Slicing Check (Helper aus batch_reels)
    # Ein Highlight bei 10-20s sollte beide SRT-Items (10-12s und 12.1-15s) enthalten
    slice_subs = get_srt_slice(subs, 10.0, 20.0)
    assert len(slice_subs) == 2
    assert slice_subs[0].content.strip() == "Hallo Welt."
    assert "Test" in slice_subs[1].content

    # 4. Filename-Check (Logik aus video_highlight_pipeline_smart)
    idx = 1
    h = highlights[0]
    # Konservative Nachbildung der Filename-Logik
    safe_title = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", "_", h.label).strip("_")[:60]
    out_file_name = f"{settings.prefix}{idx:02d}_{safe_title}.mp4"
    
    assert out_file_name == "highlight_01_Erster_Clip.mp4"
    assert "highlight_" in out_file_name
    assert out_file_name.endswith(".mp4")

def test_pfeil_parsing_smoke():
    """Prüft ob der Pfeil im Parser korrekt verarbeitet wird (Teil des Smoke-Tests)."""
    md_path = Path("tests/fixtures/sample_highlights.md")
    parser = HighlightParser()
    highlights = parser.load_highlights_from_md(md_path)
    
    # Clip 5: [00:05:00 → 00:06:00] Clip mit Pfeil
    h5 = highlights[4]
    assert h5.start == 300.0
    assert h5.end == 360.0
    assert "Pfeil" in h5.label
