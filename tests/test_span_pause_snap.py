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
