import pytest
from datetime import timedelta
from core.time_utils import (
    parse_time_to_seconds,
    seconds_to_hms,
    seconds_to_hms_ms,
    timedelta_to_hms,
    hms_to_seconds,
    normalize_hms
)

def test_parse_time_to_seconds():
    assert parse_time_to_seconds("00:01:00") == 60.0
    assert parse_time_to_seconds("01:00:00") == 3600.0
    assert parse_time_to_seconds("00:00:01.500") == 1.5
    assert parse_time_to_seconds(120) == 120.0
    assert parse_time_to_seconds(timedelta(minutes=1)) == 60.0
    assert parse_time_to_seconds(None) == 0.0
    assert parse_time_to_seconds("invalid") == 0.0

def test_seconds_to_hms():
    assert seconds_to_hms(60) == "00:01:00"
    assert seconds_to_hms(3661) == "01:01:01"
    assert seconds_to_hms(0) == "00:00:00"
    assert seconds_to_hms(None) == "00:00:00"

def test_seconds_to_hms_ms():
    assert seconds_to_hms_ms(61.5) == "00:01:01.500"
    assert seconds_to_hms_ms(0) == "00:00:00.000"
    assert seconds_to_hms_ms(3600.001) == "01:00:00.001"

def test_timedelta_to_hms():
    assert timedelta_to_hms(timedelta(hours=1, minutes=2, seconds=3)) == "01:02:03"

def test_hms_to_seconds():
    assert hms_to_seconds("01", "02", "03") == 3723.0
    assert hms_to_seconds(1, 0, 0) == 3600.0

def test_normalize_hms():
    assert normalize_hms("1:0") == "00:01:00"
    assert normalize_hms(120) == "00:02:00"
    assert normalize_hms(timedelta(minutes=5)) == "00:05:00"
