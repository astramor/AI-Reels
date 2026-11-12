import json

from build_highlight_spans import balance_highlights_from_data


class MockLLM:
    def __init__(self, scores):
        self.scores = scores

    def chat_json(self, sys_p, user_p, temp=0.2):
        return json.dumps({"scores": self.scores})


class InvalidLLM:
    def chat_json(self, sys_p, user_p, temp=0.2):
        return "kein gültiges JSON"


def test_llm_scoring_prefers_better_later_candidate_and_keeps_gap():
    entries = [
        {"sec": 10.0, "text": "first weak"},
        {"sec": 20.0, "text": "later strong"},
        {"sec": 50.0, "text": "third"},
    ]

    scores = [
        {
            "id": 1,
            "hook_strength": 2,
            "standalone_clarity": 2,
            "emotional_or_practical_value": 2,
        },
        {
            "id": 2,
            "hook_strength": 9,
            "standalone_clarity": 9,
            "emotional_or_practical_value": 9,
        },
        {
            "id": 3,
            "hook_strength": 8,
            "standalone_clarity": 8,
            "emotional_or_practical_value": 8,
        },
    ]

    selected = balance_highlights_from_data(
        entries,
        total_duration=60.0,
        target_count=2,
        llm=MockLLM(scores),
        min_gap_sec=30,
    )

    assert selected[0]["text"] == "later strong"
    assert selected[1]["text"] == "third"
    assert abs(selected[0]["sec"] - selected[1]["sec"]) >= 30


def test_invalid_llm_json_falls_back_to_sector_selection():
    entries = [
        {"sec": 10.0, "text": "first weak"},
        {"sec": 20.0, "text": "later strong"},
        {"sec": 50.0, "text": "third"},
    ]

    selected = balance_highlights_from_data(
        entries,
        total_duration=60.0,
        target_count=2,
        llm=InvalidLLM(),
        min_gap_sec=30,
    )

    assert selected[0]["text"] == "first weak"
    assert selected[1]["text"] == "third"


def test_no_llm_uses_deterministic_fallback():
    entries = [
        {"sec": 10.0, "text": "first weak"},
        {"sec": 20.0, "text": "later strong"},
        {"sec": 50.0, "text": "third"},
    ]

    selected = balance_highlights_from_data(
        entries,
        total_duration=60.0,
        target_count=2,
        llm=None,
        min_gap_sec=30,
    )

    assert selected[0]["text"] == "first weak"
    assert selected[1]["text"] == "third"
