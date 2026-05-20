import pytest
import json
import srt
from datetime import timedelta
from pathlib import Path
from domains.transcription.processor import SubtitleProcessor

@pytest.fixture
def processor():
    return SubtitleProcessor()

@pytest.fixture
def sample_whisper_json(tmp_path):
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "words": [
                    {"word": "Hallo", "start": 0.5, "end": 1.0},
                    {"word": "Welt", "start": 1.5, "end": 2.0},
                    {"word": "hier", "start": 9.8, "end": 10.0}
                ]
            },
            {
                "start": 10.0,
                "end": 20.0,
                "words": [
                    {"word": "ist", "start": 10.5, "end": 11.0},
                    {"word": "ein", "start": 12.0, "end": 12.5},
                    {"word": "Test", "start": 19.0, "end": 19.5}
                ]
            }
        ]
    }
    json_path = tmp_path / "whisper.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return json_path

# ==========================================
# 1. TEST: ASS Timestamp Formatting
# ==========================================
def test_fmt_ass_time(processor):
    # Format is h:mm:ss.cs (cs = centiseconds)
    assert processor.fmt_ass_time(0.0) == "0:00:00.00"
    assert processor.fmt_ass_time(1.5) == "0:00:01.50"
    assert processor.fmt_ass_time(61.0) == "0:01:01.00"
    assert processor.fmt_ass_time(3605.123) == "1:00:05.12"
    assert processor.fmt_ass_time(-10.0) == "0:00:00.00" # Negative should be 0

# ==========================================
# 2. TEST: Snapping Logic to Whisper JSON
# ==========================================
def test_snap_start_to_json_word_exact_match(processor, sample_whisper_json):
    # 1.5 is exactly the start of "Welt"
    snapped = processor.snap_start_to_json_word(sample_whisper_json, 1.5)
    # The code subtracts 0.05 from best_start: max(0, 1.5 - 0.05) = 1.45
    assert snapped == pytest.approx(1.45)

def test_snap_start_to_json_word_close_match(processor, sample_whisper_json):
    # 10.6 is close to 10.5 ("ist") (diff = 0.1 < 0.3)
    snapped = processor.snap_start_to_json_word(sample_whisper_json, 10.6)
    # The code subtracts 0.05 from best_start (10.5): max(0, 10.5 - 0.05) = 10.45
    assert snapped == pytest.approx(10.45)

def test_snap_start_to_json_word_out_of_window(processor, sample_whisper_json):
    # 5.0 has no word near it (window is 0.5 default)
    snapped = processor.snap_start_to_json_word(sample_whisper_json, 5.0)
    # Should return approximate_start unchanged
    assert snapped == 5.0

def test_snap_start_to_json_word_diff_too_large(processor, sample_whisper_json):
    # "ist" starts at 10.5. diff is 0.4 if we pass 10.9 (window is 0.5)
    # Since diff (0.4) > 0.3, snap is rejected and original returned.
    snapped = processor.snap_start_to_json_word(sample_whisper_json, 10.9)
    assert snapped == 10.9

def test_snap_start_to_json_word_no_file(processor):
    snapped = processor.snap_start_to_json_word(None, 10.0)
    assert snapped == 10.0

# ==========================================
# 3. TEST: Smart End in SRT (find_smart_end)
# ==========================================
def create_subtitle(start_sec, end_sec, content, index=1):
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_sec),
        end=timedelta(seconds=end_sec),
        content=content
    )

def test_find_smart_end_finds_punctuation(processor):
    srt_items = [
        create_subtitle(0.0, 5.0, "Dieser Satz ist lang"),
        create_subtitle(5.0, 10.0, "und geht noch weiter"),
        create_subtitle(10.0, 20.0, "und endet hier."), # End of sentence, valid min_dur
        create_subtitle(20.0, 30.0, "Noch ein Satz.")
    ]
    # min_dur is 18.0. start is 0.0.
    # At 20.0, end > 0.0 + 18.0 (20.0 > 18.0). It has punctuation ".".
    smart_end = processor.find_smart_end(srt_items, start_sec=0.0, min_dur=18.0)
    assert smart_end == 20.0

def test_find_smart_end_forces_max_dur(processor):
    srt_items = [
        create_subtitle(0.0, 10.0, "Kein Satzende hier"),
        create_subtitle(10.0, 30.0, "auch hier kein satzende in sicht"),
        create_subtitle(30.0, 40.0, "immer noch nichts"),
        create_subtitle(40.0, 50.0, "ein extrem langer block"),
        create_subtitle(50.0, 70.0, "und hier auch nicht")
    ]
    # start = 0.0, max_dur = 60.0
    # It should hit the max_dur limit before finding punctuation
    # Iteration:
    # 50.0 > 18.0, no punctuation
    # 70.0 > 18.0, no punctuation. But 70.0 > 0.0 + 60.0. So it returns 70.0 immediately.
    smart_end = processor.find_smart_end(srt_items, start_sec=0.0, min_dur=18.0, max_dur=60.0)
    assert smart_end == 70.0

def test_find_smart_end_fallback(processor):
    srt_items = [
        create_subtitle(0.0, 5.0, "Kurz")
    ]
    # Items end before min_dur. Loop finishes. Should return start + 20.0
    smart_end = processor.find_smart_end(srt_items, start_sec=0.0, min_dur=18.0)
    assert smart_end == 20.0
