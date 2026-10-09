"""The drivers a task's prompt tells the forecaster to reason from.

Prompts typically carry one sentence such as "Reason from pre-cutoff drivers - net interest income
trends, credit provisions, ...". Its items name what a good explanation covers, so the reasons
step builds one reason per named driver and retrieves premises for each.
"""
from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?:^|(?<=[.;!?]))\s*([^.;!?]*\breason (?:from|through|about)\b[^.!?]*)", re.IGNORECASE)
_LEAD = re.compile(r"^.*?\breason (?:from|through|about)\b\s*", re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:[,;]|\band from\b|\band\b(?=\s+(?:the|how|what|its|their|each|a|an)\b)|\(|\))\s*", re.IGNORECASE)
_DROP_PREFIX = re.compile(r"^(?:and|or|the|from|of|while)\s+", re.IGNORECASE)
MAX_DRIVERS = 8


def task_drivers(prompt: str) -> list[str]:
    """Driver phrases named by the task's "Reason from ..." sentence (empty when there is none)."""
    match = _SENTENCE.search(prompt or "")
    if not match:
        return []
    text = _LEAD.sub("", match.group(1)).strip()
    for sep in (":", " — ", " - ", " – "):
        if sep in text:
            text = text.split(sep, 1)[1]
            break
    out: list[str] = []
    for item in _SPLIT.split(text):
        item = item.strip(" .")
        while _DROP_PREFIX.match(item):
            item = _DROP_PREFIX.sub("", item).strip()
        if len(item) >= 3 and "corpus" not in item.lower() and item.lower() not in {d.lower() for d in out}:
            out.append(item)
    return out[:MAX_DRIVERS]
