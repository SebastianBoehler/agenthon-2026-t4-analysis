"""Prompts: unit-wide (all entities) and single-entity forecasts, and the reasons synthesis."""
from __future__ import annotations

import json
from datetime import date

from .evidence import Passage
from .unit import Unit

SYSTEM = (
    "You are a senior quantitative analyst who produces calibrated forecasts for a cross-section "
    "of entities. You reason only from the task table and the evidence documents provided, all "
    "dated on or before the cutoff date, and you do not use knowledge of events after the cutoff "
    "date. You answer with one JSON object and nothing else."
)


def _target_block(unit: Unit) -> str:
    t = unit.target
    lines = [f"Target: {t.get('name', '')} (type: {unit.target_type})."]
    if unit.labels:
        lines.append("Allowed labels: " + ", ".join(unit.labels) + ".")
        for lab, text in (t.get("label_assertions") or {}).items():
            lines.append(f'- "{lab}": {text}')
    lines.append(
        f"Cutoff date: {unit.task.get('cutoff_date', '')}. "
        f"Resolution date: {unit.task.get('resolution_date', 'not stated')}{_horizon(unit)}. "
        f"Interval level: {int(round(unit.interval_level * 100))}%."
    )
    return "\n".join(lines)


def _horizon(unit: Unit) -> str:
    try:
        days = (date.fromisoformat(str(unit.task["resolution_date"]))
                - date.fromisoformat(str(unit.task["cutoff_date"]))).days
    except (KeyError, ValueError):
        return ""
    return f" (horizon: {days} calendar days after the cutoff)" if days > 0 else ""


def _table(unit: Unit) -> str:
    rows = []
    for e in unit.entities:
        row = {k: v for k, v in e.items() if k != "corpus_ref"}
        rows.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(rows)


def _applies(unit: Unit, ids) -> str:
    if ids is None:
        return "all entities"
    return ", ".join(sorted(ids))


def _type_rules(unit: Unit) -> str:
    tt = unit.target_type
    if tt == "classification":
        return (
            "- label: the allowed label you judge most likely; probability: your probability "
            "(0 to 1) that this label is correct.\n"
            "- point_forecast, lo, hi: the numeric quantity the task asks you to forecast, in the "
            "task's units (for example the EPS value, the abnormal return in percent, the revised "
            "value, or the event probability when the task asks for a probability). Keep label "
            "and point_forecast consistent with the label definitions."
        )
    if tt == "ranking":
        return (
            "- point_forecast: the predicted value of the metric the ranking is taken from, in the "
            "task's units (NOT the rank number). The ranking direction is defined in the task "
            "text; a larger value must mean what the task says a larger value means.\n"
            "- lo, hi: interval on that same metric value."
        )
    return "- point_forecast, lo, hi: the target value in the task's stated units."


_METHOD = """\
How to forecast:
1. Baseline. For each entity state the naive baseline: carry the latest pre-cutoff value of the \
same quantity forward. If the table gives a prior value of the target quantity (the previous \
print, the prior-period value, a trailing change of the same kind), that value is the baseline; \
otherwise use no change for a change target, the latest observed print for a level or rate, or \
the base rate for an event. Your forecast is scored relative to that baseline: you gain only by \
being closer to the outcome.
2. Signals. Work through the drivers the task names and what the evidence says about each: \
trends, mean reversion, announced or scheduled information, positioning, guidance, solvency. \
Turn them into an expected direction and size over the forecast horizon.
3. Forecast = baseline + the expected change implied by the evidence. When the evidence points \
one way, move by your best estimate of the size; do not shrink toward the baseline out of \
caution. When the evidence is balanced, stay near the baseline.
4. Keep the cross-section consistent: entities that share drivers should move coherently.
5. Uncertainty. Estimate sigma, the standard deviation of the outcome around your forecast over \
this horizon. When the evidence shows past values of the quantity (or its changes), sigma should \
be at least their typical spread over comparable windows. For market prices, yields and spreads, \
moves over weeks or months are much larger than daily noise: scale daily volatility by the \
square root of the number of trading days in the horizon; around scheduled events (earnings \
releases, policy meetings, data releases) moves are several times a normal day's. Then lo and hi are about \
forecast - 1.645*sigma and forecast + 1.645*sigma (skew them if the risks are one-sided). The \
interval is scored by its width plus 20 times any distance by which the outcome falls outside \
it, so it must contain the outcome 9 times out of 10. The most common error is a sigma that is \
too small: large moves happen.
6. Forecast exactly the quantity the task defines (GAAP versus adjusted, first print versus \
revised, level versus change, percent versus basis points); the prior value in the table shows \
its scale. Numbers are plain JSON numbers in the target's units: no percent signs, no leading \
plus signs, no thousands separators."""


def _row_schema(unit: Unit) -> str:
    fields = [
        '"entity_id": "<id>"',
        '"analysis": "<at most 50 words: target quantity and units, baseline, key signals, expected change, typical move size>"',
        '"baseline": <number or null>',
        '"point_forecast": <number>',
        '"sigma": <number>',
        '"lo": <number>',
        '"hi": <number>',
    ]
    if unit.target_type == "classification":
        fields += ['"label": "<one allowed label>"', '"probability": <number 0-1>']
    fields.append('"evidence": [{"doc_id": "<doc id>", "quote": "<the single most relevant passage, at most 25 words, copied exactly>"}]')
    return "{" + ", ".join(fields) + "}"


def unit_prompt(unit: Unit, docs_text: list[tuple[str, str, str]], focus: list[str] | None = None) -> str:
    """Several entities in one reply. ``docs_text`` holds (doc_id, header, text) blocks; ``focus``
    limits the requested rows to a group of the roster (the whole table stays visible)."""
    wanted = focus or unit.entity_ids
    parts = [
        "TASK\n" + str(unit.task.get("prompt", "")),
        _target_block(unit),
        f"ENTITY TABLE ({len(unit.entities)} rows, one JSON object per row)\n" + _table(unit),
        "EVIDENCE DOCUMENTS (all dated on or before the cutoff)",
    ]
    for doc_id, header, text in docs_text:
        parts.append(f"=== {doc_id} | {header} ===\n{text}")
    parts.append(_METHOD)
    parts.append("Field rules:\n" + _type_rules(unit))
    if focus:
        parts.append("FORECAST ONLY THESE ENTITIES (the rest of the table is context): " + ", ".join(focus))
    parts.append(
        "OUTPUT: exactly one JSON object of the form {\"predictions\": [ROW, ...]} with one ROW for "
        f"each of the {len(wanted)} requested entities, every entity_id exactly once, in table order. "
        "ROW = " + _row_schema(unit)
    )
    return "\n\n".join(parts)


def unit_docs_full(unit: Unit) -> list[tuple[str, str, str]]:
    out = []
    for d in sorted(unit.docs.values(), key=lambda d: (d.entity_ids is not None, d.doc_id)):
        header = f"date {d.doc_date} | applies to: {_applies(unit, d.entity_ids)} | {d.title}"
        out.append((d.doc_id, header, d.text))
    return out


def unit_docs_excerpts(unit: Unit, per_entity: dict[str, list[Passage]]) -> list[tuple[str, str, str]]:
    out = []
    for eid, passages in per_entity.items():
        for p in passages:
            d = unit.docs[p.doc_id]
            header = f"date {d.doc_date} | applies to: {_applies(unit, d.entity_ids)} | excerpt for {eid}"
            out.append((p.doc_id, header, p.text))
    return out


def entity_prompt(unit: Unit, entity: dict, passages: list[Passage]) -> str:
    eid = str(entity["entity_id"])
    row = {k: v for k, v in entity.items() if k != "corpus_ref"}
    parts = [
        "TASK\n" + str(unit.task.get("prompt", "")),
        _target_block(unit),
        f"ENTITY TABLE (all {len(unit.entities)} rows, for context)\n" + _table(unit),
        f"FOCUS ENTITY: {eid}\n" + json.dumps(row, ensure_ascii=False),
        f"EVIDENCE EXCERPTS FOR {eid} (verbatim from documents dated on or before the cutoff)",
    ]
    for i, p in enumerate(passages, 1):
        d = unit.docs[p.doc_id]
        parts.append(f"[E{i}] {p.doc_id} | date {d.doc_date} | {d.title}\n{p.text}")
    parts.append(_METHOD)
    parts.append("Field rules:\n" + _type_rules(unit))
    parts.append(f"OUTPUT: exactly one JSON object for entity {eid}: " + _row_schema(unit))
    return "\n\n".join(parts)


def reasons_prompt(unit: Unit, answers: list[dict], candidates: list[tuple[str, str]]) -> str:
    """``candidates`` holds (key, verbatim text) premises the reasons must be built on."""
    parts = [
        "TASK\n" + str(unit.task.get("prompt", "")),
        _target_block(unit),
        "SUBMITTED FORECASTS (one JSON object per entity)\n"
        + "\n".join(json.dumps(a, ensure_ascii=False) for a in answers),
        "CANDIDATE PREMISES (verbatim passages from the evidence)",
    ]
    for key, text in candidates:
        parts.append(f"[{key}] {text}")
    parts.append(
        "Write up to 3 reasons that explain the submitted forecasts, strongest first. Each reason "
        "rests on ONE candidate premise. mechanism (at most 60 words) explains why that fact moves "
        "the forecast; answer_implication (at most 50 words) states what it implies for the "
        "submitted answer and names the entity_ids it concerns and their submitted labels or "
        "values. Together the reasons should cover the main drivers across the entities. Use "
        "plain prose; do not mention files, paths, or scoring."
    )
    parts.append(
        'OUTPUT: exactly one JSON object {"reasons": [{"premise": "<candidate key, e.g. R3>", '
        '"mechanism": "...", "answer_implication": "...", "entities": ["<entity_id>", ...]}]}'
    )
    return "\n\n".join(parts)
