"""Prompts: unit-wide (all entities) and single-entity forecasts, and the reasons synthesis."""
from __future__ import annotations

import json
from datetime import date

from .consistency import bps_change_target
from .evidence import Passage
from .growth_contract import forecast_method, report_comparative_target
from .tables import summarize
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
        assertions = t.get("label_assertions")
        for lab, text in (assertions.items() if isinstance(assertions, dict) else []):
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
            "value, or the event probability when the task asks for a probability). If the task "
            "defines no numeric quantity at all, set point_forecast to the probability of your label, "
            "lo to 0 and hi to 1. Keep label "
            "and point_forecast consistent with the label definitions.\n"
            "- Apply the label definitions literally: an event label is met when ANY condition it "
            "lists is expected to occur within the task’s forecast window. An announced or pending "
            "transaction is evidence about completion probability, not an event already met. A "
            "comparison label compares exactly the quantities it names."
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
same quantity forward. If the entity's table row itself gives a value of the target quantity for \
an earlier period (the previous print, a trailing change of the same kind as the target), that \
value is the baseline. Otherwise use no change: zero for a change or growth target (the level \
stays at the table's base value), the latest observed print for a level or rate. Figures you \
read or compute from the excerpts are signals, not the baseline. For a label about a direction \
or an event, the baseline is the base rate: how often each label has occurred for comparable \
cases in the history shown (a single latest observation is a noisy guide). For the direction of a \
data revision, comparable cases are routine revisions at the same stage (first to second estimate, \
and so on); a release that revised every month at once is an annual or benchmark update and says \
nothing about the next routine revision. Your forecast is \
scored relative to that baseline: you gain only by being closer to the outcome.
2. Signals. Work through the drivers the task names and what the evidence says about each: \
trends, mean reversion, announced or scheduled information, positioning, guidance, solvency. \
Turn them into an expected direction and size over the forecast horizon. Where a COMPUTED one-step \
backtest shows that the mean of recent rows predicted the next row better than the latest value, \
the series mean-reverts: centre the forecast on that recent mean, then adjust for the other signals.
3. Values in the entity table are authoritative. For a target defined as a change, growth or \
comparison relative to a stated base (for example growth versus a prior-year value given in the \
table), copy that base value from the table row, forecast the level of the quantity for the \
target period, then compute the change against exactly that base. The excerpts also discuss \
other periods (for example the latest quarter versus its own prior-year quarter): never \
substitute those values for the table's base. Forecast the level from the most recent levels of \
the quantity (for example the latest reported quarters), not by applying a growth rate measured \
over a different period or against a different base. First remove one-time items the filings \
disclose (gains or losses on sales of businesses or stakes, legal or restructuring charges, special \
assessments) from those levels. A growth rate measured for another period (for example the latest \
quarter's year-over-year growth) is never the baseline and is never carried forward: when its \
prior-year quarter was unusually weak or strong, the rate reflects that base, not momentum.
4. Forecast = baseline + the expected change implied by the evidence. When the evidence points \
one way, move by your best estimate of the size; do not shrink toward the baseline out of \
caution. When the evidence is balanced, stay near the baseline.
5. Keep the cross-section consistent: entities that share drivers should move coherently.
6. Uncertainty. Estimate sigma, the standard deviation of the outcome around your forecast over \
this horizon: how far outcomes of this kind typically land from what is already known at the \
cutoff. Measure it from comparable past surprises in the evidence: for a revised estimate of an \
already published value, the typical revision size; for a next print, the typical \
print-to-print change; for an outcome over a horizon, the typical change over windows of that \
length. When the evidence states how far the quantity moved over a recent window about as long as \
the horizon, a move of that size is roughly one sigma, not a tail event. Do not use the spread of \
levels across unrelated periods. When a COMPUTED summary gives the \
change sd over the horizon for the series you forecast, that is the floor for sigma (convert units, \
e.g. percentage points to basis points). For market prices, yields and spreads, \
moves over weeks or months are much larger than daily noise: scale daily volatility by the \
square root of the number of trading days in the horizon; around scheduled events (earnings \
releases, policy meetings, data releases) moves are several times a normal day's: an individual \
stock's one-day reaction to its own earnings release commonly moves 3 to 10 percent. Then lo and hi are about \
forecast - 1.645*sigma and forecast + 1.645*sigma (skew them if the risks are one-sided). The \
interval is scored by its width plus 20 times any distance by which the outcome falls outside \
it, so it must contain the outcome 9 times out of 10. The most common error is a sigma that is \
too small: large moves happen.
7. Forecast exactly the quantity the task defines (GAAP versus adjusted, first print versus \
revised, level versus change, percent versus basis points); the prior value in the table shows \
its scale. Numbers are plain JSON numbers in the target's units: no percent signs, no leading \
plus signs, no thousands separators."""


def _words(n_requested: int) -> int:
    """Room for reasoning in each row: more when a reply covers few entities."""
    return 120 if n_requested <= 2 else 70 if n_requested <= 6 else 45


def _row_schema(unit: Unit, words: int) -> str:
    report_base = report_comparative_target(unit.task)
    base_analysis = ("forecasted resolving-report comparative" if report_base else
                     "base value copied from the table if the target is relative to one")
    base_rule = ("your forecast of the comparative as printed in the resolving report, using only pre-cutoff evidence"
                 if report_base else "that base value copied exactly from the table row")
    fields = [
        '"entity_id": "<id>"',
        f'"analysis": "<at most {words} words: target quantity and units, {base_analysis}, baseline, key signals, expected level and change, typical move size>"',
        '"baseline": <number or null>',
        '"point_forecast": <number>',
        '"level_forecast": <number or null: for a target that is a change or growth relative to a base value, your forecast of the underlying level for the target period>',
        f'"base_value": <number or null: {base_rule}>',
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
    parts.append(forecast_method(_METHOD, unit.task))
    parts.append("Field rules:\n" + _type_rules(unit))
    if focus:
        parts.append("FORECAST ONLY THESE ENTITIES (the rest of the table is context): " + ", ".join(focus))
    parts.append(
        "OUTPUT: exactly one JSON object of the form {\"predictions\": [ROW, ...]} with one ROW for "
        f"each of the {len(wanted)} requested entities, every entity_id exactly once, in table order. "
        "ROW = " + _row_schema(unit, _words(len(wanted)))
    )
    return "\n\n".join(parts)


def horizon_days(unit: Unit) -> int | None:
    try:
        days = (date.fromisoformat(str(unit.task["resolution_date"]))
                - date.fromisoformat(str(unit.task["cutoff_date"]))).days
    except (KeyError, ValueError):
        return None
    return days if days > 0 else None


def with_stats(text: str, horizon: int | None = None, bps: bool = False) -> str:
    """A whole document plus the arithmetic summary of its tables (never quotable)."""
    stats = summarize(text, horizon, bps)
    if not stats:
        return text
    return (text.rstrip() + "\n[COMPUTED FROM THE TABLES ABOVE: arithmetic summary added for you, "
            "not part of the document; do not quote it]\n" + stats)


def unit_docs_full(unit: Unit) -> list[tuple[str, str, str]]:
    out = []
    for d in sorted(unit.docs.values(), key=lambda d: (d.entity_ids is not None, d.doc_id)):
        header = f"date {d.doc_date} | applies to: {_applies(unit, d.entity_ids)} | {d.title}"
        out.append((d.doc_id, header, with_stats(d.text, horizon_days(unit), bps_change_target(unit.task))))
    return out


def unit_docs_excerpts(unit: Unit, per_entity: dict[str, list[Passage]]) -> list[tuple[str, str, str]]:
    out = []
    for eid, passages in per_entity.items():
        for p in passages:
            d = unit.docs[p.doc_id]
            header = f"date {d.doc_date} | applies to: {_applies(unit, d.entity_ids)} | excerpt for {eid}"
            whole = p.start == 0 and p.end == len(d.text)
            out.append((p.doc_id, header, with_stats(p.text, horizon_days(unit), bps_change_target(unit.task)) if whole else p.text))
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
        whole = p.start == 0 and p.end == len(d.text)
        parts.append(f"[E{i}] {p.doc_id} | date {d.doc_date} | {d.title}\n{with_stats(p.text, horizon_days(unit), bps_change_target(unit.task)) if whole else p.text}")
    parts.append(forecast_method(_METHOD, unit.task))
    parts.append("Field rules:\n" + _type_rules(unit))
    parts.append(f"OUTPUT: exactly one JSON object for entity {eid}: " + _row_schema(unit, _words(1)))
    return "\n\n".join(parts)


_ROW_SKIP = {"corpus_ref", "cik", "currency"}


def _table_row(row: dict) -> str:
    shown = {k: v for k, v in row.items() if k not in _ROW_SKIP and isinstance(v, (str, int, float, bool))}
    return json.dumps(shown, ensure_ascii=False)[:400]


def reasons_prompt(unit: Unit, answers: list[dict], candidates: list[tuple[str, str, str, str]],
                   drivers: list[str] | None = None) -> str:
    """``candidates`` holds (key, verbatim text, doc_id, driver tag and entities it applies to)
    premises; ``drivers`` are the drivers the task's prompt names."""
    parts = [
        "TASK\n" + str(unit.task.get("prompt", "")),
        _target_block(unit),
        "ENTITY TABLE (task values given with the task; the reader does not see this table, so state any "
        "value you rely on)\n" + "\n".join(_table_row(r) for r in unit.entities),
        "SUBMITTED FORECASTS (one JSON object per entity; 'rationale' is the forecaster's own reasoning "
        "behind that forecast, use it to explain the answer)\n"
        + "\n".join(json.dumps(a, ensure_ascii=False) for a in answers),
    ]
    if drivers:
        parts.append("DRIVERS THE TASK NAMES: " + "; ".join(drivers) + ".")
    parts.append("CANDIDATE PREMISES (verbatim passages from the evidence)")
    for key, text, doc_id, applies in candidates:
        parts.append(f"[{key}] (from {doc_id}; {applies}) {text}")
    if drivers:
        scope = ("Together the 3 reasons must cover EVERY driver the task names and EVERY entity: group related "
                 "named drivers so that each reason addresses one to three of them (list them in \"driver\"), "
                 "the first reason taking the drivers that matter most for these forecasts. A reason is about "
                 "its drivers, not about one entity: apply them to every entity they affect. For each reason:\n"
                 "- premise: the candidate key that states the reason's main driver most directly;\n"
                 "- support: up to 3 other candidate keys, preferably for the reason's other drivers or for "
                 "other entities;\n"
                 "- mechanism (at most 80 words): how each of its drivers moves the forecast; ")
    else:
        scope = ("Each reason is about ONE driver: one of the most important factors behind these forecasts, "
                 "shared by several entities where the evidence allows. Choose drivers so that together the 3 "
                 "reasons address EVERY entity, and apply each driver to every entity it affects. For each "
                 "reason:\n"
                 "- premise: the candidate key that states the driver most directly;\n"
                 "- support: up to 3 other candidate keys showing the same driver, preferably for other "
                 "entities;\n"
                 "- mechanism (at most 80 words): why this driver moves the forecast; ")
    parts.append(
        "Write exactly 3 reasons (fewer only if the candidates cannot support 3), strongest first. " + scope
        + "use only figures stated in the premise or support, attributed only to the entity they are about;\n"
        "- answer_implication (at most 90 words): name each affected entity_id with its submitted label "
        "or point forecast as shown above (with the given task value it is compared with, if any), and how "
        "its drivers lead to that submitted value. The submitted answers are final: explain them, never "
        "argue that one is wrong, too high or too low; a driver pulling the other way is named only "
        "together with what outweighs it in the submitted answer;\n"
        "- entities: the affected entity_ids.\n"
        "Use plain prose; do not mention files, paths, or scoring."
    )
    parts.append(
        'OUTPUT: exactly one JSON object {"reasons": [{"driver": "<short name>", "premise": "<key, e.g. R3>", '
        '"support": ["<key>", ...], "mechanism": "...", "answer_implication": "...", "entities": ["<entity_id>", ...]}]}'
    )
    return "\n\n".join(parts)


_EVIDENCE_START = ("EVIDENCE DOCUMENTS", "EVIDENCE EXCERPTS")
_EVIDENCE_END = ("How to forecast:", "COHORT QUESTION")


def shrink_evidence(prompt: str, factor: float) -> str:
    """Keep the task, table and instructions; keep only the first ``factor`` of the evidence,
    cut at a document boundary."""
    start = min((i for i in (prompt.find(m) for m in _EVIDENCE_START) if i >= 0), default=-1)
    end = min((i for i in (prompt.find(m) for m in _EVIDENCE_END) if i >= 0), default=-1)
    if start < 0 or end <= start:
        return prompt
    body = prompt[start:end]
    keep = int(len(body) * factor)
    cut = max(body.rfind("\n\n=== ", 0, keep), body.rfind("\n\n[E", 0, keep))
    kept = body[: cut if cut > len(body) // 20 else keep]
    return prompt[:start] + kept + "\n\n[further evidence omitted for length]\n\n" + prompt[end:]

