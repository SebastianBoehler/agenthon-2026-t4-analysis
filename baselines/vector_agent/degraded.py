"""Carry-forward answers for entities that have no House forecast at all.

The user approved this recovery policy for absent forecasts and failed self-validation.
Only an entity with no usable House reply after every attempt gets one of these rows during
ordinary recovery; failed self-validation replaces all rows. The evidence trace discloses the
replacement. Points carry the table forward: zero for a change target, the row's latest value
for a level. Labels take the non-event or first label, and bands are deliberately wide.
"""
from __future__ import annotations

import math
import re

from .aggregate import Forecast
from .unit import Unit

_CHANGE = re.compile(r"change|growth|revision|delta|surprise|return|reaction|yoy|mom", re.IGNORECASE)
_SCALE_FIELD = re.compile(_CHANGE.pattern + r"|bps", re.IGNORECASE)
_PRIOR = re.compile(r"latest|prior|previous|last|current|start", re.IGNORECASE)
_NEGATION = re.compile(r"^(?:no|non|not)[_\- ]", re.IGNORECASE)
NOTE = "degraded: no House forecast for this entity; carried forward from the task table"


def _numbers(row: dict) -> dict[str, float]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[str(key)] = float(value)
    return out


def degraded_forecast(unit: Unit, row: dict) -> Forecast:
    eid = str(row["entity_id"])
    nums = _numbers(row)
    prior = next((v for k, v in nums.items() if _PRIOR.search(k)), None)
    same_kind = [abs(v) for k, v in nums.items() if _SCALE_FIELD.search(k)]
    if unit.target_type == "classification":
        labels = unit.labels
        label = next((x for x in labels if _NEGATION.match(x)), labels[0] if labels else None)
        if prior is not None:
            half = max(abs(prior), 1.0)
            return Forecast(eid, label, None, prior - half, prior + half, NOTE, 0)
        return Forecast(eid, label, None, 0.0, 1.0, NOTE, 0)
    if _CHANGE.search(str(unit.target.get("name", ""))):
        half = max(10.0, 3.0 * max(same_kind, default=0.0))
        return Forecast(eid, None, 0.0, -half, half, NOTE, 0)
    point = prior if prior is not None else 0.0
    half = max(abs(point) * 0.5, 1.0)
    return Forecast(eid, None, point, point - half, point + half, NOTE, 0)
