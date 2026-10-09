"""Labels defined as ranges of the forecast number.

Some classification units define each label by a threshold on the same quantity the answer's
point and interval forecast, for example an abnormal return "greater than 1%", "less than -1%"
and "between -1% and 1%". For those, the label is the range holding the most probability under
the pooled forecast distribution: a narrow middle range is rarely the most likely outcome even
when the point forecast falls inside it.
"""
from __future__ import annotations

import math
import re

_NUM = r"([+\-−]?\d+(?:\.\d+)?)\s*(%|percent)?"
_ABOVE = re.compile(r"\b(?:greater than|more than|above|exceeding)\s+" + _NUM, re.IGNORECASE)
_BELOW = re.compile(r"\b(?:less than|below|under)\s+" + _NUM, re.IGNORECASE)
_BETWEEN = re.compile(r"\bbetween\s+" + _NUM + r"\s+and\s+" + _NUM, re.IGNORECASE)
_RELATIVE = re.compile(r"\b(?:above|below|within|of)\s+the\b", re.IGNORECASE)


def _f(text: str) -> float:
    return float(text.replace("−", "-"))


def label_ranges(task: dict) -> dict[str, tuple[float, float]] | None:
    """{label: (lo, hi)} when every label is one numeric range and the ranges tile the real line."""
    target = task.get("target") or {}
    assertions = target.get("label_assertions")
    labels = target.get("labels") or (list(assertions) if isinstance(assertions, dict) else [])
    if target.get("type") != "classification" or not isinstance(assertions, dict) or len(labels) < 2:
        return None
    out: dict[str, tuple[float, float]] = {}
    for label in labels:
        text = str(assertions.get(label, ""))
        if _RELATIVE.search(text):  # "5% above the consensus": a range relative to another number
            return None
        between, above, below = _BETWEEN.search(text), _ABOVE.search(text), _BELOW.search(text)
        if between:
            lo, hi = sorted((_f(between.group(1)), _f(between.group(3))))
        elif above and not below:
            lo, hi = _f(above.group(1)), math.inf
        elif below and not above:
            lo, hi = -math.inf, _f(below.group(1))
        else:
            return None
        out[str(label)] = (lo, hi)
    edges = sorted(out.values())
    if edges[0][0] != -math.inf or edges[-1][1] != math.inf:
        return None
    if any(abs(a[1] - b[0]) > 1e-9 for a, b in zip(edges, edges[1:])):
        return None
    return out
