"""Revision-direction labels from the series' own vintage history.

For a two-label up/down task about the next revision of a published estimate, the corpus
vintage table shows, for every reference month, the value as printed on each release date.
The label is the majority direction of past revisions at the same stage (the change from the
n-th to the (n+1)-th published value, where the entity's month currently has n values). A tie
or a missing table leaves the House label unchanged.
"""
from __future__ import annotations

import re

from .unit import Unit

_MONTH = re.compile(r"^\d{4}-\d{2}")
_REVISION = re.compile(r"revis", re.IGNORECASE)


def _direction_labels(unit: Unit) -> tuple[str, str] | None:
    labels = [str(x) for x in unit.labels]
    if len(labels) != 2:
        return None
    up = [x for x in labels if re.search(r"\bup\b", x, re.IGNORECASE)]
    down = [x for x in labels if re.search(r"\bdown\b", x, re.IGNORECASE)]
    return (up[0], down[0]) if len(up) == len(down) == 1 else None


def _is_revision_task(unit: Unit) -> bool:
    target = unit.target or {}
    text = " ".join([str(target.get("name", "")), str(target.get("label_assertions", "")),
                     str(unit.task.get("prompt", ""))])
    return bool(_REVISION.search(text))


def vintage_rows(text: str) -> dict[str, list[float]]:
    """{reference month: published values in release order} from a pipe vintage table whose
    header has at least two release-date columns."""
    rows: dict[str, list[float]] = {}
    header_ok = False
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if not header_ok:
            header_ok = sum(bool(re.search(r"\d{4}-\d{2}-\d{2}", c)) for c in cells[1:]) >= 2
            continue
        if not _MONTH.match(cells[0]):
            continue
        values = []
        for cell in cells[1:]:
            try:
                values.append(float(cell.replace(",", "")))
            except ValueError:
                continue
        rows[cells[0][:7]] = values
    return rows


def _month_of(entity: dict) -> str | None:
    for key, value in entity.items():
        if re.search(r"ref|month|period", str(key), re.IGNORECASE) and isinstance(value, str) \
                and _MONTH.match(value):
            return value[:7]
    return None


def stage_majority(rows: dict[str, list[float]], month: str) -> int:
    """+1 / -1 for the majority direction of same-stage revisions in other months, 0 for a tie."""
    stage = len(rows.get(month, []))
    if stage == 0:
        return 0
    signs = [1 if v[stage] > v[stage - 1] else -1
             for m, v in rows.items() if m != month and len(v) > stage and v[stage] != v[stage - 1]]
    up, down = signs.count(1), signs.count(-1)
    return (up > down) - (down > up)


def revision_labels(unit: Unit) -> dict[str, str]:
    """{entity_id: label} for entities whose vintage history gives a strict majority direction."""
    pair = _direction_labels(unit)
    if unit.target_type != "classification" or pair is None or not _is_revision_task(unit):
        return {}
    out = {}
    for entity in unit.entities:
        eid, month = str(entity["entity_id"]), _month_of(entity)
        if month is None:
            continue
        for doc in unit.docs_for(eid):
            if doc.entity_ids is None:
                continue  # a shared document is not this series' own history
            rows = vintage_rows(doc.text)
            if month in rows:
                sign = stage_majority(rows, month)
                if sign:
                    out[eid] = pair[0] if sign > 0 else pair[1]
                break
    return out
