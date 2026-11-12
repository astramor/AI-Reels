import re
from pathlib import Path

import pytest

from core.time_utils import parse_time_to_seconds
from domains.highlights.models import HighlightPayload

FIXTURE = Path(__file__).parent / "fixtures" / "model_eval_sermon.txt"
TIMESTAMP_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}) --> (\d{2}:\d{2}:\d{2})\]\s+(.*)$"
)


def load_segments():
    segments = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        match = TIMESTAMP_RE.match(line.strip())
        if match:
            segments.append(
                (
                    parse_time_to_seconds(match.group(1)),
                    parse_time_to_seconds(match.group(2)),
                    match.group(3),
                )
            )
    return segments


def validate_highlights_against_fixture(payload):
    segments = load_segments()
    starts = {start for start, _end, _text in segments}
    ends = {end for _start, end, _text in segments}

    for highlight in payload["highlights"]:
        if highlight["start"] not in starts:
            raise ValueError(
                f"start {highlight['start']} is not a fixture timestamp"
            )
        if highlight["end"] not in ends:
            raise ValueError(
                f"end {highlight['end']} is not a fixture timestamp"
            )

    return HighlightPayload.model_validate(payload)


def test_model_eval_fixture_is_present_and_non_sensitive():
    text = FIXTURE.read_text(encoding="utf-8")
    assert text.strip()
    assert len(text) < 2000


def test_model_eval_fixture_has_explicit_timestamped_segments():
    segments = load_segments()
    assert len(segments) >= 3
    assert all(start < end for start, end, _text in segments)
    assert all(text.strip() for _start, _end, text in segments)
    assert len({start for start, _end, _text in segments}) == len(segments)
    assert len({end for _start, end, _text in segments}) == len(segments)


def test_selected_timestamps_originate_from_fixture():
    payload = {
        "highlights": [
            {
                "title": "Worauf wir uns verlassen",
                "start": 8.0,
                "end": 16.0,
                "reason": "Der Text fragt, worauf wir uns verlassen.",
                "finalized": False,
            },
            {
                "title": "Klarer statt schneller",
                "start": 48.0,
                "end": 56.0,
                "reason": "Der Schlusssatz betont Klarheit statt Tempo.",
                "finalized": False,
            },
        ]
    }

    validated = validate_highlights_against_fixture(payload)
    assert len(validated.highlights) == 2
    assert [highlight.start for highlight in validated.highlights] == [8.0, 48.0]


def test_fabricated_start_timestamp_is_rejected():
    payload = {
        "highlights": [
            {
                "title": "Fabricated start",
                "start": 7.5,
                "end": 16.0,
                "reason": "Invented start timestamp.",
                "finalized": False,
            }
        ]
    }

    with pytest.raises(ValueError, match="not a fixture timestamp"):
        validate_highlights_against_fixture(payload)


def test_fabricated_end_timestamp_is_rejected():
    payload = {
        "highlights": [
            {
                "title": "Fabricated end",
                "start": 8.0,
                "end": 15.0,
                "reason": "Invented end timestamp.",
                "finalized": False,
            }
        ]
    }

    with pytest.raises(ValueError, match="not a fixture timestamp"):
        validate_highlights_against_fixture(payload)


def test_valid_non_overlapping_highlights_survive_validation():
    payload = {
        "highlights": [
            {
                "title": "Früher Abschnitt",
                "start": 0.0,
                "end": 8.0,
                "reason": "Der erste Abschnitt fragt, was wirklich trägt.",
                "confidence": 0.4,
                "finalized": False,
            },
            {
                "title": "Später Abschnitt",
                "start": 16.0,
                "end": 24.0,
                "reason": "Der Abschnitt spricht von eigener Kraft und Müdigkeit.",
                "confidence": 0.6,
                "finalized": False,
            },
        ]
    }

    validated = validate_highlights_against_fixture(payload)
    assert len(validated.highlights) == 2
    assert [highlight.title for highlight in validated.highlights] == [
        "Früher Abschnitt",
        "Später Abschnitt",
    ]


def test_overlapping_highlights_follow_documented_policy():
    payload = {
        "highlights": [
            {
                "title": "Lower confidence",
                "start": 0.0,
                "end": 24.0,
                "reason": "Overlapping candidate with lower confidence.",
                "confidence": 0.4,
                "finalized": False,
            },
            {
                "title": "Higher confidence",
                "start": 8.0,
                "end": 24.0,
                "reason": "Overlapping candidate with higher confidence.",
                "confidence": 0.9,
                "finalized": False,
            },
        ]
    }

    validated = validate_highlights_against_fixture(payload)
    assert len(validated.highlights) == 1
    assert validated.highlights[0].title == "Higher confidence"


def test_fewer_than_three_valid_highlights_are_accepted():
    payload = {
        "highlights": [
            {
                "title": "Ein gültiger Abschnitt",
                "start": 0.0,
                "end": 8.0,
                "reason": "Ein einzelner gültiger Abschnitt reicht.",
                "finalized": False,
            }
        ]
    }

    validated = validate_highlights_against_fixture(payload)
    assert len(validated.highlights) == 1
