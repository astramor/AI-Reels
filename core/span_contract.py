#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Central clip-span contract.

Policy:
- A finalized span is authoritative and must not be reinterpreted by rendering.
- start/end must be finite numbers, 0 <= start < end.
- end must not exceed the available media duration.
- duration must be between min_duration and max_duration.
- A short-tail exception is only allowed when explicitly requested, when the
  span ends exactly at the media duration, and when the remaining media
  interval is shorter than min_duration.
- Fractional seconds are preserved as floats.

Invalid spans are rejected with SpanValidationError.  The only narrow repair
provided here is clamping an end to max_duration/media_duration before
validation; callers may use it when the selected end is too long.
"""

import math
from dataclasses import dataclass
from typing import Optional


class SpanValidationError(ValueError):
    """Raised when a clip span violates the final span contract."""


@dataclass(frozen=True)
class ClipSpan:
    start: float
    end: float
    title: str = ""
    finalized: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def _finite_float(value, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpanValidationError(
            f"{field_name} ist kein numerischer Wert: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise SpanValidationError(
            f"{field_name} ist nicht endlich: {value!r}"
        )
    return number


def validate_span(
    span: ClipSpan,
    media_duration: Optional[float] = None,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    allow_short_tail: bool = False,
) -> ClipSpan:
    """Validate a finalized clip span and return a normalized ClipSpan."""
    start = _finite_float(span.start, "start")
    end = _finite_float(span.end, "end")
    min_duration = _finite_float(min_duration, "min_duration")
    max_duration = _finite_float(max_duration, "max_duration")

    if min_duration < 0:
        raise SpanValidationError("min_duration muss >= 0 sein")
    if max_duration < min_duration:
        raise SpanValidationError(
            "max_duration muss >= min_duration sein"
        )

    if start < 0:
        raise SpanValidationError(f"start muss >= 0 sein: {start}")
    if end <= start:
        raise SpanValidationError(
            f"end ({end}) muss größer als start ({start}) sein"
        )

    media = None
    if media_duration is not None:
        media = _finite_float(media_duration, "media_duration")
        if media < 0:
            raise SpanValidationError("media_duration muss >= 0 sein")
        if end > media + 1e-6:
            raise SpanValidationError(
                f"end ({end}) überschreitet die Mediendauer ({media})"
            )

    duration = end - start
    if duration > max_duration + 1e-6:
        raise SpanValidationError(
            f"Dauer {duration:.3f}s überschreitet max_duration {max_duration:.3f}s"
        )

    if duration < min_duration - 1e-6:
        short_tail = (
            allow_short_tail
            and media is not None
            and math.isclose(end, media, abs_tol=1e-6)
            and (media - start) < min_duration
        )
        if not short_tail:
            raise SpanValidationError(
                f"Dauer {duration:.3f}s unterschreitet min_duration {min_duration:.3f}s"
            )

    return ClipSpan(
        start=start,
        end=end,
        title=span.title,
        finalized=True,
    )


def clamp_end_to_max(
    start: float,
    end: float,
    max_duration: float,
    media_duration: Optional[float] = None,
) -> float:
    """Narrowly repair an over-long end by clamping it to max/media duration."""
    start = _finite_float(start, "start")
    end = _finite_float(end, "end")
    max_duration = _finite_float(max_duration, "max_duration")

    end = min(end, start + max_duration)
    if media_duration is not None:
        media = _finite_float(media_duration, "media_duration")
        end = min(end, media)
    return end


def parse_finalized_flag(value, default: bool = False) -> bool:
    """Parse an explicit finalized flag; malformed values are rejected."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise SpanValidationError(
        f"finalized muss ein bool sein (true/false), nicht {value!r}"
    )


def ensure_no_finalized_overlap(spans) -> None:
    """Reject overlapping finalized spans instead of silently dropping them."""
    finalized = []
    for span in spans:
        if isinstance(span, ClipSpan):
            finalized_flag = span.finalized
            start = span.start
            end = span.end
            title = span.title
        elif isinstance(span, dict):
            finalized_flag = span.get("finalized", False)
            start = span.get("start")
            end = span.get("end")
            title = span.get("title", "")
        else:
            finalized_flag = getattr(span, "finalized", False)
            start = getattr(span, "start", None)
            end = getattr(span, "end", None)
            title = getattr(span, "title", getattr(span, "label", ""))

        if not finalized_flag or end is None:
            continue

        start_f = _finite_float(start, "start")
        end_f = _finite_float(end, "end")
        finalized.append((start_f, end_f, title))

    for i in range(len(finalized)):
        for j in range(i + 1, len(finalized)):
            s1, e1, t1 = finalized[i]
            s2, e2, t2 = finalized[j]
            if s1 < e2 and s2 < e1:
                raise SpanValidationError(
                    f"Finalisierte Spans '{t1}' und '{t2}' überschneiden sich."
                )
