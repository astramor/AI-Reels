import json
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import srt
from pydantic import ValidationError

from batch_reels import (
    execute_render_orchestration,
    parse_highlights_md,
    parse_spans_json,
    probe_media_duration,
)
from build_highlight_spans import finalize_clip_span
from core.span_contract import (
    ClipSpan,
    SpanValidationError,
    clamp_end_to_max,
    validate_span,
)
from core.time_utils import parse_time_to_seconds
from domains.highlights.models import HighlightPayload
from domains.highlights.parser import Highlight, HighlightParser
from domains.transcription.processor import SubtitleProcessor
from core.config import settings
from domains.video.renderer import FFmpegRenderer
from video_highlight_pipeline_smart import (
    finalize_highlight,
    prepare_render_job,
    probe_duration,
)


# ---------------------------------------------------------------------------
# 1. Central validation contract
# ---------------------------------------------------------------------------

def test_validate_rejects_equal_and_reversed_boundaries():
    for start, end in [(10.0, 10.0), (10.0, 9.0)]:
        with pytest.raises(SpanValidationError):
            validate_span(
                ClipSpan(start=start, end=end),
                media_duration=100.0,
                min_duration=15.0,
                max_duration=60.0,
            )


def test_validate_rejects_negative_nan_and_infinite_timestamps():
    cases = [
        ClipSpan(start=-1.0, end=10.0),
        ClipSpan(start=math.nan, end=10.0),
        ClipSpan(start=0.0, end=math.inf),
    ]
    for span in cases:
        with pytest.raises(SpanValidationError):
            validate_span(
                span,
                media_duration=100.0,
                min_duration=15.0,
                max_duration=60.0,
            )


def test_validate_rejects_end_beyond_media_duration():
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=90.0, end=110.0),
            media_duration=100.0,
            min_duration=15.0,
            max_duration=60.0,
        )


def test_validate_rejects_duration_below_min_and_above_max():
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=10.0, end=20.0),
            media_duration=100.0,
            min_duration=15.0,
            max_duration=60.0,
        )
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=10.0, end=80.0),
            media_duration=100.0,
            min_duration=15.0,
            max_duration=60.0,
        )


def test_validate_accepts_exact_min_and_max_durations():
    assert validate_span(
        ClipSpan(start=10.0, end=25.0),
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    ).duration == pytest.approx(15.0)

    assert validate_span(
        ClipSpan(start=10.0, end=70.0),
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    ).duration == pytest.approx(60.0)


def test_short_tail_policy_is_rejected_by_default_and_opt_in():
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=95.0, end=100.0),
            media_duration=100.0,
            min_duration=15.0,
            max_duration=60.0,
        )

    allowed = validate_span(
        ClipSpan(start=95.0, end=100.0),
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
        allow_short_tail=True,
    )
    assert allowed.start == pytest.approx(95.0)
    assert allowed.end == pytest.approx(100.0)

    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=90.0, end=99.0),
            media_duration=100.0,
            min_duration=15.0,
            max_duration=60.0,
            allow_short_tail=True,
        )


def test_fractional_seconds_are_preserved_by_validation():
    span = validate_span(
        ClipSpan(start=12.345, end=20.678),
        media_duration=100.0,
        min_duration=5.0,
        max_duration=60.0,
    )
    assert span.start == pytest.approx(12.345)
    assert span.end == pytest.approx(20.678)
    assert span.duration == pytest.approx(8.333)


def test_clamp_end_to_max_and_media():
    assert clamp_end_to_max(10.0, 80.0, 60.0, 100.0) == pytest.approx(70.0)
    assert clamp_end_to_max(10.0, 80.0, 60.0, 65.0) == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# 2. Generation-path validation
# ---------------------------------------------------------------------------

def test_finalize_clip_span_rejects_llm_end_below_minimum():
    with pytest.raises(SpanValidationError):
        finalize_clip_span(10.0, 12.0, "Clip", 100.0, 15.0, 60.0)


def test_finalize_clip_span_clamps_llm_end_above_maximum():
    span = finalize_clip_span(10.0, 80.0, "Clip", 100.0, 15.0, 60.0)
    assert span.start == pytest.approx(10.0)
    assert span.end == pytest.approx(70.0)
    assert span.duration == pytest.approx(60.0)


def test_finalize_clip_span_clamps_srt_fallback_beyond_maximum():
    span = finalize_clip_span(10.0, 85.0, "Clip", 100.0, 15.0, 60.0)
    assert span.end == pytest.approx(70.0)


def test_finalize_clip_span_rejects_insufficient_remaining_media():
    with pytest.raises(SpanValidationError):
        finalize_clip_span(95.0, 100.0, "Clip", 100.0, 15.0, 60.0)


def test_start_snap_can_move_earlier_and_later():
    words = [
        {"word": "Satz.", "start": 9.0, "end": 10.0},
        {"word": "Ohne", "start": 10.1, "end": 10.6},
        {"word": "Pause.", "start": 11.6, "end": 12.0},
        {"word": "Nächstes", "start": 12.4, "end": 13.0},
        {"word": "Weiter.", "start": 15.8, "end": 16.0},
        {"word": "Danach", "start": 16.8, "end": 17.0},
    ]
    processor = SubtitleProcessor()

    earlier = processor.snap_start_to_sentence_begin(words, 18.0)
    later = processor.snap_start_to_sentence_begin(words, 10.5)
    assert earlier == pytest.approx(16.7)
    assert later == pytest.approx(12.3)
    assert earlier < 18.0
    assert later > 10.5


def test_end_selection_uses_adjusted_start():
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


def test_missing_whisperx_data_still_validates_and_clamps():
    processor = SubtitleProcessor()
    candidate = Highlight(start=10.0, end=80.0, label="Clip", finalized=False)
    finalized = finalize_highlight(
        candidate,
        words=[],
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    )
    assert finalized is not None
    assert finalized.start == pytest.approx(10.0)
    assert finalized.end == pytest.approx(70.0)
    assert finalized.finalized is True


def test_no_words_no_srt_uses_max_duration_and_validates():
    processor = SubtitleProcessor()
    candidate = Highlight(start=10.0, end=None, label="Clip", finalized=False)
    finalized = finalize_highlight(
        candidate,
        words=[],
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    )
    assert finalized is not None
    assert finalized.start == pytest.approx(10.0)
    assert finalized.end == pytest.approx(70.0)


def test_missing_whisperx_data_with_end_below_minimum_is_rejected():
    processor = SubtitleProcessor()
    candidate = Highlight(start=10.0, end=12.0, label="Clip", finalized=False)
    finalized = finalize_highlight(
        candidate,
        words=[],
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    )
    assert finalized is None


def test_srt_fallback_without_words_is_validated():
    processor = SubtitleProcessor()
    srt_items = [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=0.0),
            end=timedelta(seconds=100.0),
            content="Ein langer Satz.",
        )
    ]
    candidate = Highlight(start=10.0, end=None, label="Clip", finalized=False)
    finalized = finalize_highlight(
        candidate,
        words=[],
        srt_items=srt_items,
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    )
    assert finalized is not None
    assert finalized.end == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# 3. Finalized spans and rendering
# ---------------------------------------------------------------------------

def test_finalized_highlight_bypasses_candidate_snapping():
    processor = SubtitleProcessor()
    words = [
        {"word": "Satz.", "start": 9.0, "end": 10.0},
        {"word": "Weiter.", "start": 15.8, "end": 16.0},
    ]
    h = Highlight(start=12.345, end=20.678, label="Final", finalized=True)
    finalized = finalize_highlight(
        h,
        words=words,
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=5.0,
        max_duration=60.0,
    )
    assert finalized is not None
    assert finalized.start == pytest.approx(12.345)
    assert finalized.end == pytest.approx(20.678)
    assert finalized.finalized is True


def test_prepare_render_job_uses_finalized_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "preroll", 5.0)
    monkeypatch.setattr(settings, "postroll", 5.0)
    monkeypatch.setattr(settings, "max_window", 10.0)

    h = Highlight(start=12.345, end=20.678, label="Final", finalized=True)
    job = prepare_render_job(
        idx=1,
        h=h,
        video_path=tmp_path / "in.mp4",
        out_dir=tmp_path,
        json_path=None,
        temp_dir_path=tmp_path,
        src_w=100,
        src_h=100,
        target_w=1080,
        target_h=1920,
        renderer=FFmpegRenderer(),
        tracker=MagicMock(),
        sub_processor=SubtitleProcessor(),
        music_file=None,
        subtitles_path=None,
        prefix="test_",
    )
    assert job.start_ts == pytest.approx(12.345)
    assert job.duration == pytest.approx(8.333)


def test_prepare_render_job_rejects_non_finalized_highlight(tmp_path):
    h = Highlight(start=10.0, end=20.0, label="Candidate", finalized=False)
    with pytest.raises(ValueError):
        prepare_render_job(
            idx=1,
            h=h,
            video_path=tmp_path / "in.mp4",
            out_dir=tmp_path,
            json_path=None,
            temp_dir_path=tmp_path,
            src_w=100,
            src_h=100,
            target_w=1080,
            target_h=1920,
            renderer=FFmpegRenderer(),
            tracker=MagicMock(),
            sub_processor=SubtitleProcessor(),
            music_file=None,
            subtitles_path=None,
        )


def test_karaoke_subtitles_respect_finalized_duration(tmp_path):
    data = {
        "segments": [
            {
                "words": [
                    {"word": "Früh", "start": 10.0, "end": 11.0},
                    {"word": "Mitte", "start": 18.0, "end": 19.0},
                    {"word": "Spät", "start": 25.0, "end": 26.0},
                ]
            }
        ]
    }
    json_path = tmp_path / "words.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    ass_path = tmp_path / "out.ass"

    SubtitleProcessor().generate_karaoke_ass(
        json_path,
        start_abs=10.0,
        duration=10.0,
        out_ass=ass_path,
    )

    text = ass_path.read_text(encoding="utf-8")
    assert "Früh" in text
    assert "Mitte" in text
    assert "Spät" not in text


def test_fractional_timestamps_reach_ffmpeg_command():
    cmd = FFmpegRenderer().build_ffmpeg_cmd(
        input_video=Path("in.mp4"),
        start_ts=12.345,
        duration=8.678,
        out_file=Path("out.mp4"),
    )
    assert cmd[cmd.index("-ss") + 1] == "00:00:12.345"
    assert cmd[cmd.index("-t") + 1] == "8.678"


# ---------------------------------------------------------------------------
# 4. Candidate/finalized metadata, parsing, and batch handoff
# ---------------------------------------------------------------------------

def test_markdown_range_is_finalized_and_single_timestamp_is_candidate(tmp_path):
    md_path = tmp_path / "highlight_spans.md"
    md_path.write_text(
        "- [00:01:12.150 -> 00:01:30.400] Titel!\n"
        "[00:02:00] Kandidat\n",
        encoding="utf-8",
    )
    marks = HighlightParser().load_highlights_from_md(md_path)
    assert marks[0].finalized is True
    assert marks[1].finalized is False


def test_legacy_reel_spans_json_defaults_to_finalized(tmp_path):
    path = tmp_path / "reel_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": "00:01:12.150",
                    "end": "00:01:30.400",
                    "title": "Legacy",
                }
            ]
        ),
        encoding="utf-8",
    )
    spans = parse_spans_json(path)
    assert spans[0]["start"] == pytest.approx(72.15)
    assert spans[0]["end"] == pytest.approx(90.40)
    assert spans[0]["finalized"] is True

    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].start == pytest.approx(72.15)
    assert marks[0].end == pytest.approx(90.40)
    assert marks[0].finalized is True


def test_legacy_reel_spans_dict_defaults_to_finalized(tmp_path):
    path = tmp_path / "reel_spans.json"
    path.write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "title": "Legacy Dict",
                        "start": 10.0,
                        "end": 20.0,
                        "reason": "Test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].finalized is True
    assert marks[0].start == pytest.approx(10.0)
    assert marks[0].end == pytest.approx(20.0)


def test_legacy_reel_spans_dict_overlap_is_rejected(tmp_path):
    path = tmp_path / "reel_spans.json"
    path.write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "title": "A",
                        "start": 10.0,
                        "end": 20.0,
                        "reason": "Test",
                    },
                    {
                        "title": "B",
                        "start": 15.0,
                        "end": 25.0,
                        "reason": "Test",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpanValidationError):
        HighlightParser().load_highlights_from_json(path)


def test_legacy_highlights_json_defaults_to_candidate(tmp_path):
    path = tmp_path / "highlights.json"
    path.write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "title": "Candidate",
                        "start": 10.0,
                        "end": 20.0,
                        "reason": "Test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].finalized is False


def test_batch_handoff_preserves_finalized_fractional_boundaries(tmp_path, monkeypatch):
    spans_path = tmp_path / "reel_spans.json"
    spans_path.write_text(
        json.dumps(
            [
                {
                    "start": "00:01:12.150",
                    "end": "00:01:30.400",
                    "title": "Clip",
                    "finalized": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    predigt_srt = tmp_path / "predigt.srt"
    predigt_srt.write_text(
        "1\n00:00:00,000 --> 00:01:40,000\nText\n",
        encoding="utf-8",
    )
    summ_dir = tmp_path / "summ"
    summ_dir.mkdir()
    out_reels = tmp_path / "out"
    out_reels.mkdir()

    args = SimpleNamespace(
        confirm_subtitles=False,
        music_dir=None,
        nvenc=False,
        dry_run=True,
        face_sample_rate=10.0,
        face_min_conf=0.4,
        face_smooth_sec=2.5,
        face_keys=15,
        face_first_range=1.0,
        face_first_grid=0.4,
        face_first_steps=3,
        face_first_sample_rate=8.0,
    )

    captured = []
    monkeypatch.setattr("batch_reels.probe_media_duration", lambda path: 100.0)
    monkeypatch.setattr(
        "batch_reels.render_clip",
        lambda item, vid, reels_out, summ_dir, stem, args: captured.append(item),
    )

    execute_render_orchestration(
        tmp_path / "video.mp4",
        "stem",
        summ_dir,
        None,
        predigt_srt,
        spans_path,
        out_reels,
        args,
    )

    assert len(captured) == 1
    assert captured[0]["start"] == pytest.approx(72.15)
    assert captured[0]["end"] == pytest.approx(90.40)
    assert captured[0]["finalized"] is True


def test_parse_highlights_md_preserves_finalized_flag(tmp_path):
    md_path = tmp_path / "highlight_spans.md"
    md_path.write_text(
        "- [00:01:12.150 -> 00:01:30.400] Titel!\n",
        encoding="utf-8",
    )
    spans = parse_highlights_md(md_path)
    assert spans[0]["start"] == pytest.approx(72.15)
    assert spans[0]["end"] == pytest.approx(90.40)
    assert spans[0]["finalized"] is True


# ---------------------------------------------------------------------------
# 5. Finalized immutability, metadata strictness, overlap, media duration
# ---------------------------------------------------------------------------

def test_overlong_finalized_span_is_rejected_not_clamped():
    processor = SubtitleProcessor()
    h = Highlight(start=10.0, end=80.0, label="Final", finalized=True)
    result = finalize_highlight(
        h,
        words=[],
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=100.0,
        min_duration=15.0,
        max_duration=60.0,
    )
    assert result is None
    assert h.start == pytest.approx(10.0)
    assert h.end == pytest.approx(80.0)


def test_valid_finalized_span_is_not_mutated():
    processor = SubtitleProcessor()
    h = Highlight(start=1.237, end=4.891, label="Final", finalized=True)
    result = finalize_highlight(
        h,
        words=[],
        srt_items=[],
        transcription_json=None,
        sub_processor=processor,
        media_duration=10.0,
        min_duration=1.0,
        max_duration=10.0,
    )
    assert result is h
    assert h.start == pytest.approx(1.237)
    assert h.end == pytest.approx(4.891)


def test_prepare_render_job_rejects_overlong_finalized_span(tmp_path):
    h = Highlight(start=10.0, end=80.0, label="Final", finalized=True)
    with pytest.raises(ValueError):
        prepare_render_job(
            idx=1,
            h=h,
            video_path=tmp_path / "in.mp4",
            out_dir=tmp_path,
            json_path=None,
            temp_dir_path=tmp_path,
            src_w=100,
            src_h=100,
            target_w=1080,
            target_h=1920,
            renderer=FFmpegRenderer(),
            tracker=MagicMock(),
            sub_processor=SubtitleProcessor(),
            music_file=None,
            subtitles_path=None,
        )


def test_batch_skips_overlong_finalized_span(tmp_path, monkeypatch):
    spans_path = tmp_path / "reel_spans.json"
    spans_path.write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 80.0,
                    "title": "Long",
                    "finalized": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    predigt_srt = tmp_path / "predigt.srt"
    predigt_srt.write_text(
        "1\n00:00:00,000 --> 00:01:40,000\nText\n",
        encoding="utf-8",
    )
    summ_dir = tmp_path / "summ"
    summ_dir.mkdir()
    out_reels = tmp_path / "out"
    out_reels.mkdir()

    args = SimpleNamespace(
        confirm_subtitles=False,
        music_dir=None,
        nvenc=False,
        dry_run=True,
        face_sample_rate=10.0,
        face_min_conf=0.4,
        face_smooth_sec=2.5,
        face_keys=15,
        face_first_range=1.0,
        face_first_grid=0.4,
        face_first_steps=3,
        face_first_sample_rate=8.0,
    )

    captured = []
    monkeypatch.setattr("batch_reels.probe_media_duration", lambda path: 100.0)
    monkeypatch.setattr(
        "batch_reels.render_clip",
        lambda item, vid, reels_out, summ_dir, stem, args: captured.append(item),
    )

    execute_render_orchestration(
        tmp_path / "video.mp4",
        "stem",
        summ_dir,
        None,
        predigt_srt,
        spans_path,
        out_reels,
        args,
    )

    assert captured == []


def test_explicit_finalized_false_overrides_legacy_reel_spans(tmp_path):
    path = tmp_path / "reel_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": "00:01:12.150",
                    "end": "00:01:30.400",
                    "title": "Legacy",
                    "finalized": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    spans = parse_spans_json(path)
    assert spans[0]["finalized"] is False

    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].finalized is False


def test_renamed_explicit_finalized_true_preserves_metadata(tmp_path):
    path = tmp_path / "renamed_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": "00:01:12.150",
                    "end": "00:01:30.400",
                    "title": "Renamed",
                    "finalized": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    spans = parse_spans_json(path)
    assert spans[0]["finalized"] is True

    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].finalized is True


def test_renamed_unmarked_artifact_defaults_to_candidate(tmp_path):
    path = tmp_path / "renamed_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "title": "Renamed",
                }
            ]
        ),
        encoding="utf-8",
    )
    spans = parse_spans_json(path)
    assert spans[0]["finalized"] is False

    marks = HighlightParser().load_highlights_from_json(path)
    assert marks[0].finalized is False


def test_malformed_finalized_string_is_rejected(tmp_path):
    path = tmp_path / "bad_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "title": "Bad",
                    "finalized": "false",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpanValidationError):
        HighlightParser().load_highlights_from_json(path)

    spans = parse_spans_json(path)
    assert spans == []


def test_malformed_finalized_string_in_dict_payload_is_rejected(tmp_path):
    path = tmp_path / "bad_highlights.json"
    path.write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "title": "Bad",
                        "start": 10.0,
                        "end": 20.0,
                        "reason": "Test",
                        "finalized": "false",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        HighlightParser().load_highlights_from_json(path)

    spans = parse_spans_json(path)
    assert spans == []


def test_finalized_dict_payload_overlap_is_rejected():
    payload = {
        "highlights": [
            {
                "title": "A",
                "start": 10.0,
                "end": 20.0,
                "reason": "Test",
                "finalized": True,
            },
            {
                "title": "B",
                "start": 15.0,
                "end": 25.0,
                "reason": "Test",
                "finalized": True,
            },
        ]
    }
    with pytest.raises(ValidationError):
        HighlightPayload.model_validate(payload)


def test_finalized_dict_payload_preserves_order_and_both_spans(tmp_path):
    payload = {
        "highlights": [
            {
                "title": "B",
                "start": 30.0,
                "end": 40.0,
                "reason": "Test",
                "finalized": True,
            },
            {
                "title": "A",
                "start": 10.0,
                "end": 20.0,
                "reason": "Test",
                "finalized": True,
            },
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert [h.title for h in model.highlights] == ["B", "A"]

    path = tmp_path / "finalized.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    marks = HighlightParser().load_highlights_from_json(path)
    assert [m.label for m in marks] == ["B", "A"]
    assert all(m.finalized for m in marks)


def test_mixed_finalized_and_candidate_payload_still_deduplicates_candidates():
    payload = {
        "highlights": [
            {
                "title": "Final",
                "start": 0.0,
                "end": 10.0,
                "reason": "Test",
                "finalized": True,
            },
            {
                "title": "Cand1",
                "start": 10.0,
                "end": 30.0,
                "reason": "Test",
                "confidence": 0.5,
            },
            {
                "title": "Cand2",
                "start": 15.0,
                "end": 25.0,
                "reason": "Test",
                "confidence": 0.4,
            },
        ]
    }
    model = HighlightPayload.model_validate(payload)
    assert [h.title for h in model.highlights] == ["Final", "Cand1"]
    assert model.highlights[0].finalized is True
    assert model.highlights[1].finalized is False


def test_batch_parse_rejects_overlapping_finalized_spans(tmp_path):
    path = tmp_path / "reel_spans.json"
    path.write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "title": "A",
                    "finalized": True,
                },
                {
                    "start": 15.0,
                    "end": 25.0,
                    "title": "B",
                    "finalized": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpanValidationError):
        parse_spans_json(path)


def test_overlapping_finalized_markdown_is_rejected(tmp_path):
    md_path = tmp_path / "highlight_spans.md"
    md_path.write_text(
        "- [00:00:10 -> 00:00:20] A\n"
        "- [00:00:15 -> 00:00:25] B\n",
        encoding="utf-8",
    )
    with pytest.raises(SpanValidationError):
        parse_highlights_md(md_path)


def test_media_duration_zero_nan_and_unavailable():
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=1.0, end=2.0),
            media_duration=0.0,
            min_duration=1.0,
            max_duration=10.0,
        )
    with pytest.raises(SpanValidationError):
        validate_span(
            ClipSpan(start=1.0, end=2.0),
            media_duration=math.nan,
            min_duration=1.0,
            max_duration=10.0,
        )
    assert validate_span(
        ClipSpan(start=1.0, end=2.0),
        media_duration=None,
        min_duration=1.0,
        max_duration=10.0,
    ).end == pytest.approx(2.0)


def test_finalize_clip_span_with_unavailable_media_duration():
    span = finalize_clip_span(10.0, 30.0, "Clip", None, 15.0, 60.0)
    assert span.start == pytest.approx(10.0)
    assert span.end == pytest.approx(30.0)


def test_finalize_clip_span_with_zero_media_duration_rejects():
    with pytest.raises(SpanValidationError):
        finalize_clip_span(10.0, 20.0, "Clip", 0.0, 15.0, 60.0)


def test_probe_media_duration_rejects_invalid_values(monkeypatch):
    for raw in ("0", "NaN", "inf", "-1"):
        monkeypatch.setattr(
            "batch_reels.run_command",
            lambda cmd, raw=raw: SimpleNamespace(
                stdout=json.dumps({"format": {"duration": raw}})
            ),
        )
        with pytest.raises(RuntimeError):
            probe_media_duration(Path("video.mp4"))


def test_probe_duration_rejects_invalid_values(monkeypatch):
    for raw in ("0", "NaN", "inf", "-1"):
        monkeypatch.setattr(
            "video_highlight_pipeline_smart.ffprobe_json",
            lambda path, raw=raw: {"format": {"duration": raw}},
        )
        with pytest.raises(RuntimeError):
            probe_duration(Path("video.mp4"))


# ---------------------------------------------------------------------------
# 6. Subtitle events within the finalized clip interval
# ---------------------------------------------------------------------------

def _ass_event_times(ass_path):
    events = []
    for line in ass_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",")
        start = parse_time_to_seconds(parts[1])
        end = parse_time_to_seconds(parts[2])
        events.append((start, end))
    return events


def test_karaoke_words_cross_start_and_end_boundaries(tmp_path):
    data = {
        "segments": [
            {
                "words": [
                    {"word": "Vor", "start": 0.5, "end": 1.0},
                    {"word": "Start", "start": 1.1, "end": 1.5},
                    {"word": "Mitte", "start": 2.0, "end": 2.5},
                    {"word": "Ende", "start": 4.7, "end": 5.0},
                    {"word": "Danach", "start": 5.1, "end": 5.5},
                ]
            }
        ]
    }
    json_path = tmp_path / "words.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    ass_path = tmp_path / "out.ass"

    start_abs = 1.237
    duration = 3.654
    SubtitleProcessor().generate_karaoke_ass(
        json_path,
        start_abs=start_abs,
        duration=duration,
        out_ass=ass_path,
    )

    dialogue_text = "\n".join(
        line for line in ass_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue:")
    )
    assert "Start" in dialogue_text
    assert "Mitte" in dialogue_text
    assert "Ende" in dialogue_text
    assert "Vor" not in dialogue_text
    assert "Danach" not in dialogue_text

    events = _ass_event_times(ass_path)
    assert events
    for start, end in events:
        assert start >= 0.0
        assert end > start
        assert end <= duration + 0.01


def test_karaoke_zero_length_events_are_extended(tmp_path):
    data = {
        "segments": [
            {
                "words": [
                    {"word": "A", "start": 1.1, "end": 1.5},
                    {"word": "B", "start": 1.2, "end": 1.8},
                    {"word": "C", "start": 2.0, "end": 2.5},
                ]
            }
        ]
    }
    json_path = tmp_path / "words.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    ass_path = tmp_path / "out.ass"

    start_abs = 1.237
    duration = 2.0
    SubtitleProcessor().generate_karaoke_ass(
        json_path,
        start_abs=start_abs,
        duration=duration,
        out_ass=ass_path,
    )

    events = _ass_event_times(ass_path)
    assert events
    for start, end in events:
        assert start >= 0.0
        assert end > start
        assert end <= duration + 0.01
