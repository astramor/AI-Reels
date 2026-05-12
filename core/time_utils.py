#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta
from typing import Union, Optional


def parse_time_to_seconds(value: Union[str, int, float, timedelta, None]) -> float:
    """
    Konvertiert verschiedene Zeitformate (HH:MM:SS, Sekunden als int/float, timedelta) in Sekunden (float).
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, timedelta):
        return value.total_seconds()

    # String-Parsing (HH:MM:SS oder MM:SS oder SS)
    try:
        parts = str(value).split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except (ValueError, TypeError):
        return 0.0
    return 0.0


def seconds_to_hms(seconds: Union[int, float, None]) -> str:
    """
    Konvertiert Sekunden in das Format HH:MM:SS.
    """
    if seconds is None:
        return "00:00:00"
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def seconds_to_hms_ms(seconds: Union[int, float, None]) -> str:
    """
    Konvertiert Sekunden in das Format HH:MM:SS.mmm (für FFmpeg oder präzises Timing).
    """
    if seconds is None:
        return "00:00:00.000"
    t = max(0.0, float(seconds))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def timedelta_to_hms(td: timedelta) -> str:
    """
    Konvertiert ein timedelta-Objekt in HH:MM:SS.
    """
    return seconds_to_hms(td.total_seconds())


def hms_to_seconds(h: Union[str, int], m: Union[str, int], s: Union[str, int]) -> float:
    """
    Hilfsfunktion für RegEx-Parser (extrahiert h, m, s).
    """
    return float(h) * 3600 + float(m) * 60 + float(s)


def normalize_hms(value: Union[str, int, float, timedelta, None]) -> str:
    """
    Normalisiert verschiedene Zeitformate (z.B. "1:2" -> "00:01:02") in HH:MM:SS.
    """
    return seconds_to_hms(parse_time_to_seconds(value))
