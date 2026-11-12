import json
from datetime import timedelta

import pytest
import srt

from core.config import settings
from core.prompts import JSON_HIGHLIGHT_REDUCE_SYSTEM_PROMPT
from domains.highlights.errors import NoValidHighlightsError
from domains.llm.errors import LLMApiError
from domains.llm.summarizer import SermonSummarizer
from srt_summarizer_local import _parse_sermon_boundary


SCOUT_PAYLOAD = {
    "highlights": [
        {
            "title": "Scout 1",
            "start": 0.0,
            "end": 30.0,
            "reason": "R",
            "hook": "H",
            "confidence": 0.9,
        },
        {
            "title": "Scout 2",
            "start": 40.0,
            "end": 70.0,
            "reason": "R",
            "hook": "H",
            "confidence": 0.9,
        },
        {
            "title": "Scout 3",
            "start": 80.0,
            "end": 110.0,
            "reason": "R",
            "hook": "H",
            "confidence": 0.9,
        },
    ]
}

CURATION_PAYLOAD = {
    "highlights": [
        {
            "title": "Curated",
            "start": 100.0,
            "end": 130.0,
            "reason": "R",
            "hook": "H",
            "confidence": 0.9,
        }
    ]
}


def make_srt(path, count=10, text_len=700):
    subtitles = []
    repetitions = max(1, text_len // len("Predigttext"))
    for i in range(count):
        start = timedelta(seconds=i * 10)
        end = timedelta(seconds=i * 10 + 9)
        subtitles.append(
            srt.Subtitle(
                index=i + 1,
                start=start,
                end=end,
                content="Predigttext" * repetitions,
            )
        )
    path.write_text(srt.compose(subtitles), encoding="utf-8")


def make_summarizer(cloud, chat_calls, json_calls):
    summarizer = SermonSummarizer(settings)
    summarizer.llm.provider = "openai" if cloud else "ollama"

    def fake_chat(sys_p, user_p, temp=0.1):
        chat_calls.append((sys_p, user_p, temp))
        return "summary"

    def fake_chat_json(sys_p, user_p, temp=0.2):
        json_calls.append((sys_p, user_p, temp))
        if sys_p == JSON_HIGHLIGHT_REDUCE_SYSTEM_PROMPT:
            return json.dumps(CURATION_PAYLOAD)
        return json.dumps(SCOUT_PAYLOAD)

    summarizer.chat = fake_chat
    summarizer.chat_json = fake_chat_json
    return summarizer


def test_cloud_chunk_size_and_cloud_prompt(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=700)
    out_dir = tmp_path / "out"

    chat_calls = []
    summarizer = make_summarizer(True, chat_calls, [])
    summarizer.process(str(srt_path), str(out_dir))

    assert len(chat_calls) == 1
    assert "fließenden" in chat_calls[0][1]


def test_local_chunk_size_and_local_prompt(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=700)
    out_dir = tmp_path / "out"

    chat_calls = []
    summarizer = make_summarizer(False, chat_calls, [])
    summarizer.process(str(srt_path), str(out_dir))

    assert len(chat_calls) == 3
    assert "Bulletpoints" in chat_calls[0][1]


def test_export_sermon_srt_filters_and_clips_window(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=20)
    out_dir = tmp_path / "out"
    export_path = tmp_path / "filtered.srt"

    summarizer = make_summarizer(True, [], [])
    result = summarizer.process(
        str(srt_path),
        str(out_dir),
        sermon_start="00:00:20",
        sermon_end="00:00:40",
        export_sermon_srt=str(export_path),
    )

    exported = list(srt.parse(export_path.read_text(encoding="utf-8")))
    assert exported
    assert all(
        it.start.total_seconds() >= 20.0
        and it.end.total_seconds() <= 40.0
        for it in exported
    )
    assert exported[0].start.total_seconds() == pytest.approx(20.0)
    assert exported[0].end.total_seconds() == pytest.approx(29.0)
    transcript = (out_dir / "transcript_predigt.txt").read_text(encoding="utf-8")
    assert "[20.00s]" in transcript
    assert "[30.00s]" in transcript
    assert result["summary_path"].exists()


def test_fallback_fills_from_longlist_when_curation_is_too_small(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=50)
    out_dir = tmp_path / "out"

    summarizer = make_summarizer(True, [], [])
    summarizer.process(str(srt_path), str(out_dir), max_highlights=6)

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    assert len(data["highlights"]) >= 4
    assert any(h["title"] == "Scout 1" for h in data["highlights"])


def test_curation_failure_raises_no_valid_highlights(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=50)
    out_dir = tmp_path / "out"

    summarizer = make_summarizer(True, [], [])
    summarizer.chat_json = lambda sys_p, user_p, temp=0.2: "kein JSON"

    with pytest.raises(NoValidHighlightsError):
        summarizer.process(str(srt_path), str(out_dir))


def test_chat_json_raises_on_empty_llm_output():
    summarizer = SermonSummarizer(settings)
    summarizer.llm.chat_json = lambda sys_p, user_p, temp=0.2: None

    with pytest.raises(LLMApiError):
        summarizer.chat_json("system", "user")


def test_exported_sermon_srt_content_matches_transcript(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    subtitles = [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=20),
            end=timedelta(seconds=25),
            content="Zeile eins\nZeile zwei",
        )
    ]
    srt_path.write_text(srt.compose(subtitles), encoding="utf-8")
    out_dir = tmp_path / "out"
    export_path = tmp_path / "filtered.srt"

    summarizer = make_summarizer(True, [], [])
    summarizer.process(
        str(srt_path),
        str(out_dir),
        sermon_start="00:00:20",
        sermon_end="00:00:30",
        export_sermon_srt=str(export_path),
    )

    exported = list(srt.parse(export_path.read_text(encoding="utf-8")))
    transcript = (out_dir / "transcript_predigt.txt").read_text(encoding="utf-8")
    assert exported[0].content == "Zeile eins Zeile zwei"
    assert "Zeile eins Zeile zwei" in transcript


def test_invalid_sermon_boundaries_raise(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=50)
    summarizer = make_summarizer(True, [], [])

    for start, end in [("abc", "00:00:40"), ("00:00:20", "abc")]:
        with pytest.raises(ValueError):
            summarizer.process(
                str(srt_path),
                str(tmp_path / "out"),
                sermon_start=start,
                sermon_end=end,
            )


def test_local_invalid_sermon_boundary_raises():
    with pytest.raises(ValueError):
        _parse_sermon_boundary("abc", "sermon_start")

    assert _parse_sermon_boundary("00:00:20", "sermon_start") == pytest.approx(20.0)


def test_invalid_longlist_fallback_entries_are_skipped(tmp_path):
    srt_path = tmp_path / "sermon.srt"
    make_srt(srt_path, count=10, text_len=50)
    out_dir = tmp_path / "out"

    summarizer = make_summarizer(True, [], [])

    def fake_chat_json(sys_p, user_p, temp=0.2):
        if sys_p == JSON_HIGHLIGHT_REDUCE_SYSTEM_PROMPT:
            return json.dumps(CURATION_PAYLOAD)
        return json.dumps(
            {
                "highlights": [
                    {"title": "Bad", "start": "x", "end": 30.0, "reason": "R"},
                    {
                        "title": "Good",
                        "start": 40.0,
                        "end": 70.0,
                        "reason": "R",
                        "confidence": 0.9,
                    },
                    {
                        "title": "Good2",
                        "start": 80.0,
                        "end": 110.0,
                        "reason": "R",
                        "confidence": 0.9,
                    },
                ]
            }
        )

    summarizer.chat_json = fake_chat_json
    summarizer.process(str(srt_path), str(out_dir), max_highlights=6)

    data = json.loads((out_dir / "highlights.json").read_text(encoding="utf-8"))
    titles = [h["title"] for h in data["highlights"]]
    assert "Good" in titles
    assert "Good2" in titles
    assert "Bad" not in titles
