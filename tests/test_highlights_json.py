import pytest
import json
from domains.highlights.models import HighlightPayload, HighlightItem
from domains.llm.summarizer import SermonSummarizer
from core.config import settings
from pydantic import ValidationError
from domains.llm.errors import LLMInvalidJsonError

@pytest.fixture
def mock_summarizer():
    return SermonSummarizer(settings)

def test_valid_json_accepted():
    payload = {
        "highlights": [
            {
                "title": "Guter Clip",
                "start": 10.0,
                "end": 45.0,
                "reason": "Sehr spannend",
                "confidence": 0.9
            }
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert len(model.highlights) == 1
    assert model.highlights[0].title == "Guter Clip"

def test_markdown_codeblock_cleaned(mock_summarizer):
    raw = "```json\n{\"highlights\": [{\"title\": \"Test\", \"start\": 0.0, \"end\": 35.0, \"reason\": \"ok\"}]}\n```"
    parsed = mock_summarizer.extract_json_from_llm_response(raw)
    assert "highlights" in parsed

def test_invalid_json_rejected(mock_summarizer):
    raw = "Das ist nur ein Text ohne JSON."
    with pytest.raises(LLMInvalidJsonError):
        mock_summarizer.extract_json_from_llm_response(raw)

def test_missing_highlights_field_rejected():
    payload = {"clips": []}
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "highlights" in str(excinfo.value)

def test_negative_start_time_rejected():
    payload = {
        "highlights": [
            {
                "title": "Clip",
                "start": -5.0,
                "end": 35.0,
                "reason": "Test"
            }
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "Input should be greater than or equal to 0" in str(excinfo.value)

def test_end_before_start_rejected():
    payload = {
        "highlights": [
            {
                "title": "Clip",
                "start": 50.0,
                "end": 40.0,
                "reason": "Test"
            }
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "end (40.0) muss größer als start (50.0) sein" in str(excinfo.value)

def test_overlapping_clips_reduced():
    payload = {
        "highlights": [
            {
                "title": "Clip 1",
                "start": 10.0,
                "end": 50.0,
                "reason": "Test",
                "confidence": 0.5
            },
            {
                "title": "Clip 2", # Dieser startet bei 15s (Überschneidung mit Clip 1)
                "start": 15.0,
                "end": 60.0,
                "reason": "Test",
                "confidence": 0.9 # Höhere Confidence, sollte Clip 1 ersetzen
            },
            {
                "title": "Clip 3", # Kein Overlap
                "start": 100.0,
                "end": 140.0,
                "reason": "Test",
                "confidence": 0.8
            }
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert len(model.highlights) == 2
    assert model.highlights[0].title == "Clip 2"
    assert model.highlights[1].title == "Clip 3"
