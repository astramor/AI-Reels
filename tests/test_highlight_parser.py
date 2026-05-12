import pytest
from pathlib import Path
from domains.highlights.parser import HighlightParser, Highlight

def test_load_highlights_from_md(tmp_path):
    md_content = """
- [00:00:10 -> 00:00:20] Erster Clip
- [00:01:05 -> 00:01:15] Zweiter Clip
* [00:02:30 -> 00:03:00] Dritter Clip
[00:04:00] Vierter Clip ohne Ende
- [00:05:00 → 00:06:00] Clip mit Pfeil
    """
    md_file = tmp_path / "highlights.md"
    md_file.write_text(md_content, encoding="utf-8")
    
    parser = HighlightParser()
    highlights = parser.load_highlights_from_md(md_file)
    
    assert len(highlights) == 5
    
    assert highlights[0].start == 10.0
    assert highlights[0].end == 20.0
    assert highlights[0].label == "Erster Clip"
    
    assert highlights[3].start == 240.0
    assert highlights[3].end is None
    assert highlights[3].label == "Vierter Clip ohne Ende"
    
    assert highlights[4].start == 300.0
    assert highlights[4].end == 360.0
    assert highlights[4].label == "Clip mit Pfeil"

def test_load_highlights_from_actual_fixture():
    fixture_path = Path("tests/fixtures/sample_highlights.md")
    if not fixture_path.exists():
        pytest.skip("Fixture file not found")
        
    parser = HighlightParser()
    highlights = parser.load_highlights_from_md(fixture_path)
    assert len(highlights) == 5
    assert highlights[1].label == "Zweiter Clip"
