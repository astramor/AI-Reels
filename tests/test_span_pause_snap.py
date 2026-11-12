import pytest

from domains.transcription.processor import SubtitleProcessor


def test_snap_end_prefers_word_with_real_pause():
    words = [
        {"word": "Satz.", "start": 9.0, "end": 10.0},
        {"word": "Ohne", "start": 10.1, "end": 10.6},
        {"word": "Pause.", "start": 11.6, "end": 12.0},
        {"word": "Nächstes", "start": 12.4, "end": 13.0},
    ]

    processor = SubtitleProcessor()
    snapped = processor.snap_end_to_speech_pause(
        words,
        chosen_end=10.5,
        start_sec=0.0,
        min_duration=5.0,
        max_duration=20.0,
    )

    assert snapped == pytest.approx(12.15)


def test_snap_end_without_words_returns_chosen_end():
    processor = SubtitleProcessor()
    assert processor.snap_end_to_speech_pause(
        [],
        chosen_end=10.0,
        start_sec=0.0,
        min_duration=5.0,
        max_duration=20.0,
    ) == 10.0


def test_snap_start_to_clean_sentence_begin_with_lead_in_padding():
    words = [
        {"word": "Satz.", "start": 9.0, "end": 10.0},
        {"word": "Ohne", "start": 10.1, "end": 10.6},
        {"word": "Pause.", "start": 11.6, "end": 12.0},
        {"word": "Nächstes", "start": 12.4, "end": 13.0},
    ]

    processor = SubtitleProcessor()
    snapped = processor.snap_start_to_sentence_begin(
        words,
        approx_start_sec=10.5,
        search_window=3.0,
        min_pause=0.30,
    )

    assert snapped == pytest.approx(12.3)


def test_snap_start_without_pause_falls_back_to_approx_start():
    words = [
        {"word": "Satz.", "start": 20.0, "end": 20.5},
        {"word": "Ohne", "start": 20.55, "end": 20.9},
        {"word": "Pause.", "start": 21.0, "end": 21.4},
        {"word": "Nächstes", "start": 21.45, "end": 21.8},
    ]

    processor = SubtitleProcessor()
    snapped = processor.snap_start_to_sentence_begin(
        words,
        approx_start_sec=24.0,
        search_window=3.0,
        min_pause=0.30,
    )

    assert snapped == pytest.approx(24.0)


def test_snap_start_without_words_returns_approx_start():
    processor = SubtitleProcessor()
    assert processor.snap_start_to_sentence_begin([], 10.0) == 10.0


def test_start_snap_is_used_for_end_selection():
    words = [
        {"word": "Satz.", "start": 9.0, "end": 10.0},
        {"word": "Ohne", "start": 10.1, "end": 10.6},
        {"word": "Pause.", "start": 11.6, "end": 12.0},
        {"word": "Nächstes", "start": 12.4, "end": 13.0},
        {"word": "Weiter.", "start": 15.8, "end": 16.0},
        {"word": "Danach", "start": 16.8, "end": 17.0},
    ]

    processor = SubtitleProcessor()
    adjusted_start = processor.snap_start_to_sentence_begin(words, 10.5)
    adjusted_end = processor.snap_end_to_speech_pause(
        words,
        chosen_end=16.0,
        start_sec=adjusted_start,
        min_duration=5.0,
        max_duration=20.0,
    )
    original_end = processor.snap_end_to_speech_pause(
        words,
        chosen_end=16.0,
        start_sec=10.5,
        min_duration=5.0,
        max_duration=20.0,
    )

    assert adjusted_start == pytest.approx(12.3)
    assert adjusted_end == pytest.approx(16.0)
    assert original_end == pytest.approx(16.15)
    assert adjusted_end != original_end


def test_millisecond_markdown_preserved_by_batch_and_pipeline(tmp_path):
    md_path = tmp_path / "highlight_spans.md"
    md_path.write_text(
        "- [00:01:12.150 -> 00:01:30.400] Titel!\n",
        encoding="utf-8",
    )

    from batch_reels import parse_highlights_md
    from domains.highlights.parser import HighlightParser

    batch_spans = parse_highlights_md(md_path)
    assert batch_spans[0]["start"] == pytest.approx(72.15)
    assert batch_spans[0]["end"] == pytest.approx(90.40)

    pipeline_marks = HighlightParser().load_highlights_from_md(md_path)
    assert pipeline_marks[0].start == pytest.approx(72.15)
    assert pipeline_marks[0].end == pytest.approx(90.40)
