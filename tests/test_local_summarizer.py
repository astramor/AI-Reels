import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import srt

import srt_summarizer_local as local_summarizer
from build_highlight_spans import parse_highlights_json, parse_longlist
from srt_summarizer_local import build_highlight_candidates, sanitize_highlights


def make_fractional_srt(path):
    subtitles = [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=12.345),
            end=timedelta(seconds=15.678),
            content="Alpha candidate text with many words",
        ),
        srt.Subtitle(
            index=2,
            start=timedelta(seconds=12.400),
            end=timedelta(seconds=15.900),
            content="Beta candidate text with many words",
        ),
        srt.Subtitle(
            index=3,
            start=timedelta(seconds=30.125),
            end=timedelta(seconds=33.500),
            content="Gamma candidate text with many words",
        ),
    ]
    path.write_text(srt.compose(subtitles), encoding="utf-8")


def make_regular_srt(path, count=10):
    subtitles = []
    for i in range(count):
        start = timedelta(seconds=i * 10)
        end = timedelta(seconds=i * 10 + 9)
        subtitles.append(
            srt.Subtitle(
                index=i + 1,
                start=start,
                end=end,
                content="Predigttext mit vielen Wörtern und Details",
            )
        )
    path.write_text(srt.compose(subtitles), encoding="utf-8")


def fake_local_llm(scout_output="", editor_output=""):
    class FakeLocalLLM:
        is_cloud_llm = False

        def __init__(self, conf):
            self.conf = conf

        def chat(self, sys_p, user_p, temp=0.1):
            if sys_p == "Scout":
                return scout_output
            if sys_p == "Editor":
                return editor_output
            return "summary"

    return FakeLocalLLM


def make_local_args(srt_path, out_dir, max_highlights=8, export=None):
    return SimpleNamespace(
        srt=str(srt_path),
        out_dir=str(out_dir),
        max_highlights=max_highlights,
        model=None,
        llm_provider=None,
        export_sermon_srt=str(export) if export else None,
        sermon_start=None,
        sermon_end=None,
        strict_snap_tol=8,
        fix_spelling=False,
        auto_sermon=False,
        min_sermon_minutes=10,
        preset=None,
        min_gap_sec=12,
        timeout=900,
    )


def test_build_highlight_candidates_preserves_fractional_start():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        )
    ]
    candidates = build_highlight_candidates(items)
    assert candidates[0]["start"] == pytest.approx(12.345)


def test_sanitize_preserves_fractional_start():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        )
    ]
    result = sanitize_highlights(
        "- [00:00:12.345] Alpha candidate text with many words", items
    )
    assert result[0]["start"] == pytest.approx(12.345)


def test_sanitize_maps_legacy_whole_second_to_fractional_start():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        )
    ]
    result = sanitize_highlights(
        "- [00:00:12] Alpha candidate text with many words", items
    )
    assert result[0]["start"] == pytest.approx(12.345)


def test_sanitize_distinguishes_candidates_in_same_integer_second():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.400),
                end=timedelta(seconds=15.900),
                content="Beta candidate text with many words",
            ),
            "Beta candidate text with many words",
        ),
    ]
    result = sanitize_highlights(
        "- [00:00:12] Beta candidate text with many words", items
    )
    assert result[0]["start"] == pytest.approx(12.400)


def test_sanitize_skips_malformed_or_unmatched_timestamps():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        )
    ]
    assert sanitize_highlights("- Alpha candidate text with many words", items) == []
    assert sanitize_highlights("- [00:00:40] Alpha candidate text with many words", items) == []


def test_sanitize_fractional_timestamp_requires_exact_match():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.400),
                end=timedelta(seconds=15.900),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.600),
                end=timedelta(seconds=16.100),
                content="Beta candidate text with many words",
            ),
            "Beta candidate text with many words",
        ),
    ]
    assert (
        sanitize_highlights(
            "- [00:00:12.500] Alpha candidate text with many words", items
        )
        == []
    )


def test_sanitize_exact_fractional_timestamp_maps_exact_candidate():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.345),
                end=timedelta(seconds=15.678),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.400),
                end=timedelta(seconds=15.900),
                content="Beta candidate text with many words",
            ),
            "Beta candidate text with many words",
        ),
    ]
    result = sanitize_highlights(
        "- [00:00:12.345] Different candidate text", items
    )
    assert result[0]["start"] == pytest.approx(12.345)


def test_sanitize_legacy_whole_second_without_text_match_is_skipped():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.200),
                end=timedelta(seconds=15.500),
                content="Alpha candidate text with many words",
            ),
            "Alpha candidate text with many words",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.800),
                end=timedelta(seconds=16.100),
                content="Beta candidate text with many words",
            ),
            "Beta candidate text with many words",
        ),
    ]
    result = sanitize_highlights(
        "- [00:00:12] Unknown candidate text", items
    )
    assert result == []


def test_sanitize_legacy_whole_second_ambiguous_text_match_is_skipped():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.200),
                end=timedelta(seconds=15.500),
                content="Alpha shared phrase one",
            ),
            "Alpha shared phrase one",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.800),
                end=timedelta(seconds=16.100),
                content="Beta shared phrase two",
            ),
            "Beta shared phrase two",
        ),
    ]
    result = sanitize_highlights(
        "- [00:00:12] shared phrase", items
    )
    assert result == []


def test_sanitize_legacy_whole_second_equal_exact_text_is_skipped():
    items = [
        (
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=12.200),
                end=timedelta(seconds=15.500),
                content="Shared phrase for both",
            ),
            "Shared phrase for both",
        ),
        (
            srt.Subtitle(
                index=2,
                start=timedelta(seconds=12.800),
                end=timedelta(seconds=16.100),
                content="Shared phrase for both",
            ),
            "Shared phrase for both",
        ),
    ]
    result = sanitize_highlights(
        "- [00:00:12] Shared phrase for both", items
    )
    assert result == []


def test_local_run_preserves_json_and_srt_precision(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_fractional_srt(srt_path)
    out_dir = tmp_path / "out"
    export_path = tmp_path / "filtered.srt"

    scout = "\n".join(
        [
            "- [00:00:12.345] Alpha candidate text with many words",
            "- [00:00:12] Beta candidate text with many words",
            "- [00:00:30.125] Gamma candidate text with many words",
        ]
    )
    editor = "- [00:00:12] Beta candidate text with many words"

    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(scout, editor))
    local_summarizer.run(
        make_local_args(srt_path, out_dir, max_highlights=3, export=export_path)
    )

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert data["highlights"][0]["start"] == pytest.approx(12.400)

    entries = parse_highlights_json(out_dir / "highlights.json")
    assert entries[0]["sec"] == pytest.approx(12.400)

    markdown_entries = parse_longlist(out_dir / "highlights.md")
    assert markdown_entries[0]["sec"] == pytest.approx(12.400)

    exported = list(srt.parse(export_path.read_text(encoding="utf-8")))
    assert exported[0].start.total_seconds() == pytest.approx(12.345)
    assert exported[1].start.total_seconds() == pytest.approx(12.400)
    assert exported[0].end.total_seconds() == pytest.approx(15.678)


def test_local_max_highlights_one(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)
    out_dir = tmp_path / "out"

    editor = "\n".join(
        [
            "- [00:00:00] Predigttext mit vielen Wörtern und Details",
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:80] Predigttext mit vielen Wörtern und Details",
        ]
    )
    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(editor, editor))
    local_summarizer.run(make_local_args(srt_path, out_dir, max_highlights=1))

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) == 1


def test_local_max_highlights_limits_curated_results(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path, count=20)
    out_dir = tmp_path / "out"

    editor = "\n".join(
        [
            "- [00:00:00] Predigttext mit vielen Wörtern und Details",
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:80] Predigttext mit vielen Wörtern und Details",
            "- [00:02:00] Predigttext mit vielen Wörtern und Details",
            "- [00:02:40] Predigttext mit vielen Wörtern und Details",
        ]
    )
    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(editor, editor))
    local_summarizer.run(make_local_args(srt_path, out_dir, max_highlights=3))

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) == 3


def test_local_fallback_fills_without_exceeding_limit(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)
    out_dir = tmp_path / "out"

    editor = "- [00:00:00] Predigttext mit vielen Wörtern und Details"
    scout = "\n".join(
        [
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:80] Predigttext mit vielen Wörtern und Details",
            "- [00:01:20] Predigttext mit vielen Wörtern und Details",
        ]
    )
    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(scout, editor))
    local_summarizer.run(make_local_args(srt_path, out_dir, max_highlights=3))

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) == 3


def test_local_insufficient_valid_candidates_returns_fewer(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)
    out_dir = tmp_path / "out"

    editor = "- [00:00:00] Predigttext mit vielen Wörtern und Details"
    scout = "\n".join(
        [
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:80] Predigttext mit vielen Wörtern und Details",
            "- [00:00:99] Predigttext mit vielen Wörtern und Details",
        ]
    )
    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(scout, editor))
    local_summarizer.run(make_local_args(srt_path, out_dir, max_highlights=5))

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) == 3


def test_local_duplicate_candidates_are_deduplicated(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)
    out_dir = tmp_path / "out"

    editor = "- [00:00:00] Predigttext mit vielen Wörtern und Details"
    scout = "\n".join(
        [
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:40] Predigttext mit vielen Wörtern und Details",
            "- [00:00:80] Predigttext mit vielen Wörtern und Details",
        ]
    )
    monkeypatch.setattr(local_summarizer, "UnifiedLLM", fake_local_llm(scout, editor))
    local_summarizer.run(make_local_args(srt_path, out_dir, max_highlights=3))

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) == 3
    starts = [h["start"] for h in data["highlights"]]
    assert len(starts) == len(set(starts))


def test_local_max_highlights_rejects_zero_negative_and_invalid(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)

    for value in (0, -1, "3", True):
        with pytest.raises(ValueError):
            local_summarizer.run(
                make_local_args(srt_path, tmp_path / "out", max_highlights=value)
            )


def test_local_run_sets_model_explicit_for_cli_model(tmp_path, monkeypatch):
    srt_path = tmp_path / "sermon.srt"
    make_regular_srt(srt_path)
    out_dir = tmp_path / "out"

    args = make_local_args(srt_path, out_dir)
    args.model = "deepseek/deepseek-v4.1-flash"
    args.llm_provider = "openrouter"

    captured = {}

    class FakeLLM:
        is_cloud_llm = True

        def __init__(self, conf):
            captured.update(conf)
            self.provider = conf.get("llm_provider", "ollama")
            self.model = conf.get("openrouter_model") or conf.get("llm_model")

        def chat(self, sys_p, user_p, temp=0.1):
            if sys_p == "Scout":
                return "- [00:00:00] Predigttext mit vielen Wörtern und Details"
            if sys_p == "Editor":
                return "- [00:00:00] Predigttext mit vielen Wörtern und Details"
            return "summary"

    monkeypatch.setattr(local_summarizer, "UnifiedLLM", FakeLLM)
    local_summarizer.run(args)

    assert captured["model_explicit"] is True
    assert captured["openrouter_model"] == "deepseek/deepseek-v4.1-flash"
