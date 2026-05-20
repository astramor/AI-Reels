import pytest
from pydantic import ValidationError
from domains.highlights.models import HighlightPayload, HighlightItem
from domains.llm.summarizer import SermonSummarizer
from domains.llm.errors import LLMInvalidJsonError
from core.config import settings

@pytest.fixture
def summarizer():
    return SermonSummarizer(settings)

# ==========================================
# TEST FALL 1: Valid JSON payload accepted
# ==========================================
def test_1_valid_json_payload_accepted():
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
    assert model.highlights[0].start == 10.0
    assert model.highlights[0].end == 45.0

# ==========================================
# TEST FALL 2: Missing root field "highlights"
# ==========================================
def test_2_missing_root_field_rejected():
    payload = {"clips": []}
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "highlights" in str(excinfo.value)
    assert "Field required" in str(excinfo.value)

# ==========================================
# TEST FALL 3: Highlights not a list
# ==========================================
def test_3_highlights_not_a_list_rejected():
    payload = {
        "highlights": {
            "title": "Clip",
            "start": 10.0,
            "end": 45.0,
            "reason": "Test"
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "Input should be a valid list" in str(excinfo.value)

# ==========================================
# TEST FALL 4: Empty highlights rejected
# ==========================================
def test_4_empty_highlights_rejected():
    payload = {"highlights": []}
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "List should have at least 1 item" in str(excinfo.value)

# ==========================================
# TEST FALL 5: Start/End as numbers
# ==========================================
def test_5_start_end_must_be_numbers():
    payload = {
        "highlights": [
            {
                "title": "Clip",
                "start": "not_a_number",
                "end": 45.0,
                "reason": "Test"
            }
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "Input should be a valid number" in str(excinfo.value)

# ==========================================
# TEST FALL 6: Negative start rejected
# ==========================================
def test_6_negative_start_rejected():
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

# ==========================================
# TEST FALL 7: End <= Start rejected
# ==========================================
def test_7_end_before_or_equal_to_start_rejected():
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

# ==========================================
# TEST FALL 8: Duration out of bounds (marked)
# ==========================================
def test_8_extreme_duration_accepted_but_logged():
    payload = {
        "highlights": [
            {
                "title": "Short Clip",
                "start": 10.0,
                "end": 15.0, # 5s duration
                "reason": "Too short"
            },
            {
                "title": "Long Clip",
                "start": 20.0,
                "end": 150.0, # 130s duration
                "reason": "Too long"
            }
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert len(model.highlights) == 2
    assert model.highlights[0].title == "Short Clip"
    assert model.highlights[1].title == "Long Clip"

# ==========================================
# TEST FALL 9: Empty title rejected
# ==========================================
def test_9_empty_title_rejected():
    payload = {
        "highlights": [
            {
                "title": "",
                "start": 10.0,
                "end": 40.0,
                "reason": "Test"
            }
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload)
    assert "String should have at least 1 character" in str(excinfo.value)

# ==========================================
# TEST FALL 10: Confidence out of bounds
# ==========================================
def test_10_confidence_out_of_bounds_rejected():
    payload_high = {
        "highlights": [{"title": "T", "start": 10.0, "end": 40.0, "reason": "R", "confidence": 1.5}]
    }
    with pytest.raises(ValidationError) as excinfo:
        HighlightPayload.model_validate(payload_high)
    assert "Input should be less than or equal to 1" in str(excinfo.value)

    payload_low = {
        "highlights": [{"title": "T", "start": 10.0, "end": 40.0, "reason": "R", "confidence": -0.1}]
    }
    with pytest.raises(ValidationError) as excinfo2:
        HighlightPayload.model_validate(payload_low)
    assert "Input should be greater than or equal to 0" in str(excinfo2.value)

# ==========================================
# TEST FALL 11: Duplicates / Overlaps handled
# ==========================================
def test_11_overlapping_and_duplicate_clips_reduced():
    payload = {
        "highlights": [
            {
                "title": "Clip 1",
                "start": 10.0,
                "end": 50.0,
                "reason": "Base",
                "confidence": 0.5
            },
            {
                "title": "Clip 2",
                "start": 15.0,
                "end": 60.0,
                "reason": "Overlap",
                "confidence": 0.9
            },
            {
                "title": "Clip 3",
                "start": 100.0,
                "end": 140.0,
                "reason": "Independent",
                "confidence": 0.8
            }
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert len(model.highlights) == 2
    assert model.highlights[0].title == "Clip 2"
    assert model.highlights[1].title == "Clip 3"

# ==========================================
# TEST FALL 12: Sorting enforced
# ==========================================
def test_12_sorting_by_start_time_enforced():
    payload = {
        "highlights": [
            {
                "title": "Third Clip",
                "start": 200.0,
                "end": 240.0,
                "reason": "C"
            },
            {
                "title": "First Clip",
                "start": 10.0,
                "end": 50.0,
                "reason": "A"
            },
            {
                "title": "Second Clip",
                "start": 100.0,
                "end": 140.0,
                "reason": "B"
            }
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert len(model.highlights) == 3
    assert model.highlights[0].title == "First Clip"
    assert model.highlights[1].title == "Second Clip"
    assert model.highlights[2].title == "Third Clip"

# ==========================================
# ZUSATZ: Markdown Code Fences Extraction
# ==========================================
def test_markdown_codeblock_cleaned(summarizer):
    raw = "```json\n{\"highlights\": [{\"title\": \"Test\", \"start\": 0.0, \"end\": 35.0, \"reason\": \"ok\"}]}\n```"
    parsed = summarizer.extract_json_from_llm_response(raw)
    assert "highlights" in parsed

def test_invalid_json_text_rejected(summarizer):
    raw = "Das ist nur ein Text ohne JSON."
    with pytest.raises(LLMInvalidJsonError):
        summarizer.extract_json_from_llm_response(raw)
