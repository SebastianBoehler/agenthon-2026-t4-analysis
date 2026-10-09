"""Arithmetic consistency of a sample with the task table.

For a target stated as percent growth over a base value given in the entity's table row, the
growth is recomputed from the model's own level forecast and that base, so a slip in the
model's arithmetic or a base taken from the wrong period does not decide the point forecast.
"""
from __future__ import annotations

import math
import re

from .parse import Sample
from .tables import PERCENT_LEVEL, horizon_sds

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
    if base <= 0 or not 0.2 <= level / base <= 5.0:
        return sample  # level and base are not the same quantity: leave the model's growth alone
    growth = (level / base - 1.0) * 100.0
    shift = growth - sample.point
    if abs(shift) < 1e-9:
        return sample
    sample.point = growth
    if sample.lo is not None and sample.hi is not None:
        sample.lo += shift
        sample.hi += shift
    return sample


_BPS = re.compile(r"bps|basis.points", re.IGNORECASE)
FLOOR_MIN_CHANGES = 8  # fewer steps give too noisy a spread to override the model's band


def bps_change_target(task: dict) -> bool:
    target = task.get("target") or {}
    name = str(target.get("name", ""))
    units = " ".join(str(r.get("unit", "")) for r in task.get("entities") or [] if isinstance(r, dict))
    return target.get("type") == "regression" and "change" in (name + " " + units).lower() and bool(
        _BPS.search(name + " " + units))


def change_sd_floor(task: dict, docs, horizon_days: int | None) -> dict[str, float]:
    """For a basis-point change target whose table row names a percent-level table column (a
    yield series id, say), the random-walk sd of that column's change over the horizon, in bps.
    A model-stated band narrower than this ignores the evidence's own volatility."""
    if not bps_change_target(task) or not horizon_days:
        return {}
    spreads: dict[str, tuple[float, float]] = {}
    for doc in docs:
        for column, value in horizon_sds(doc.text, horizon_days, FLOOR_MIN_CHANGES).items():
            spreads.setdefault(column, value)
    out: dict[str, float] = {}
    for row in task.get("entities") or []:
        for value in row.values():
            hit = spreads.get(str(value)) if isinstance(value, str) else None
            if hit and hit[1] < PERCENT_LEVEL and hit[0] > 0:
                out[str(row["entity_id"])] = hit[0] * 100.0
                break
    return out
