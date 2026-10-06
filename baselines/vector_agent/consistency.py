"""Arithmetic consistency of a sample with the task table.

For a target stated as percent growth over a base value given in the entity's table row, the
growth is recomputed from the model's own level forecast and that base, so a slip in the
model's arithmetic or a base taken from the wrong period does not decide the point forecast.
"""
from __future__ import annotations

import math
import re

from .parse import Sample

_GROWTH = re.compile(r"growth|yoy", re.IGNORECASE)
_PERCENT = re.compile(r"pct|percent", re.IGNORECASE)


def percent_growth_target(task: dict) -> bool:
    target = task.get("target") or {}
    name = str(target.get("name", ""))
    prompt = str(task.get("prompt", ""))
    return (target.get("type") == "regression" and bool(_GROWTH.search(name))
            and bool(_PERCENT.search(name) or re.search(r"as a percent", prompt, re.IGNORECASE)))


def _row_numbers(row: dict) -> list[float]:
    out = []
    for value in row.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out.append(float(value))
    return out


def ground_growth(sample: Sample, row: dict) -> Sample:
    """Recompute percent growth from the sample's level and a base that appears in the table row."""
    level, base = sample.level, sample.base
    if level is None or sample.point is None:
        return sample
    numbers = _row_numbers(row)
    if base is None or not any(abs(base - v) <= 1e-6 * max(1.0, abs(v)) for v in numbers):
        if len(numbers) != 1:
            return sample  # no unambiguous table base: leave the sample as the model gave it
        base = numbers[0]  # the row's only number is the base the growth is measured against
    if base <= 0:
        return sample
    growth = (level / base - 1.0) * 100.0
    shift = growth - sample.point
    if abs(shift) < 1e-9:
        return sample
    sample.point = growth
    if sample.lo is not None and sample.hi is not None:
        sample.lo += shift
        sample.hi += shift
    return sample
