"""``analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json``.

The process writes the answer atomically and then exits immediately, so a House request still in
flight after the schedule's hard stop cannot hold the container past the unit's time limit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .agent import Planner, Schedule, Trace, assemble, fit_answer_cap, forecast, rationales, reasons
from .degraded import degraded_forecast
from .house import Budget, Chat, HouseClient, HouseConfig
from .unit import Unit, load_unit, task_table_text


def validate(unit: Unit, answer: dict) -> list[str]:
    """The structural rules that refuse a whole unit, checked before writing."""
    problems = []
    rows = answer["entity_predictions"]
    ids = [r.get("entity_id") for r in rows]
    if sorted(ids) != sorted(unit.entity_ids) or len(set(ids)) != len(ids):
        problems.append("roster mismatch")
    task_text, ranges = task_table_text(unit.task)
    for r in rows:
        eid = r.get("entity_id")
        iv = r.get("interval") or {}
        nums = [iv.get("lo"), iv.get("hi")] + ([r["point_forecast"]] if "point_forecast" in r else [])
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums) or iv["lo"] > iv["hi"]:
            problems.append(f"{eid}: bad numbers")
        if unit.target_type in ("regression", "ranking") and "point_forecast" not in r:
            problems.append(f"{eid}: missing point_forecast")
        if unit.target_type == "classification" and r.get("label") not in unit.labels:
            problems.append(f"{eid}: label outside vocabulary")
        if not r.get("claims"):
            problems.append(f"{eid}: no claims")
        for c in r.get("claims", []):
            if c["doc_id"] == "task":
                s, e = ranges.get(eid, (0, -1))
                ok = s <= c["span_start"] < c["span_end"] <= e and task_text[c["span_start"]:c["span_end"]] == c["claim"]
            else:
                doc = unit.docs.get(c["doc_id"])
                ok = (doc is not None and doc.citable and doc.admits(eid)
                      and 0 <= c["span_start"] < c["span_end"] <= len(doc.text)
                      and doc.text[c["span_start"]:c["span_end"]] == c["claim"])
            if not ok:
                problems.append(f"{eid}: claim does not resolve verbatim")
    return problems


def card_timeout(task_path: Path) -> float | None:
    """[agent].timeout_sec from the unit's card.toml next to task.json, when the harness mounts it."""
    try:
        import tomllib
        card = tomllib.loads((task_path.parent / "card.toml").read_text(encoding="utf-8"))
        value = float(card["agent"]["timeout_sec"])
    except Exception:  # noqa: BLE001 - no card, or no usable timeout: keep the 600 s schedule
        return None
    return value if math.isfinite(value) else None


PINNED_SEED = 20261006


def _plan_without_house(unit: Unit, seed: int):
    """The planner's evidence (for claims) without any House call; a bare stand-in if even that fails."""
    try:
        plan = Planner(unit, seed)
    except Exception:  # noqa: BLE001
        plan = SimpleNamespace(trace=Trace(), samples={}, passages={}, full=True)
    plan.trace.mode = "degraded"
    plan.samples = defaultdict(list, plan.samples)
    return plan


def run(task_path: Path, corpus_dir: Path, out_path: Path, chat: Chat | None, budget: Budget,
        schedule: Schedule, start: float, log=print) -> dict:
    unit = load_unit(task_path, corpus_dir)
    # The organizers' verification rerun sets a fresh QFBENCH_SEED; a seed meant to repeat must be a
    # constant in the code (SUBMISSION_CLI.md, determinism), so request seeds do not follow it.
    seed = PINNED_SEED
    results, plan = {}, None
    if chat is not None:
        try:
            results, plan = forecast(unit, chat, budget, start, schedule, seed)
        except Exception as exc:  # noqa: BLE001 - every entity then gets a degraded row below
            log(f"forecast step failed: {type(exc).__name__}: {exc}")
    if plan is None:
        plan = _plan_without_house(unit, seed)
    log(f"mode={plan.trace.mode} launched={dict(plan.trace.launched)} "
        f"succeeded={dict(plan.trace.succeeded)} requests={budget.used} "
        f"elapsed={time.monotonic() - start:.1f}s")
    for failure in plan.trace.failures[:20]:
        log(f"  call failure: {failure}")
    for note in plan.trace.notes:
        log(f"  {note}")
    rows_by_id = {str(e["entity_id"]): e for e in unit.entities}
    missing = [eid for eid in unit.entity_ids if eid not in results]
    for eid in missing:  # the organizers require an answer even without the House (README)
        results[eid] = degraded_forecast(unit, rows_by_id[eid])
    if missing:
        log(f"  degraded: {len(missing)} of {len(unit.entity_ids)} entities without a House forecast")
    rows, claims_by_entity = assemble(unit, results, plan)
    cap_note = fit_answer_cap(rows, unit.target_type)
    if cap_note != "within cap":
        log(f"  answer cap: {cap_note}")
    reason_list, reason_note = [], "skipped (no House forecasts)"
    if chat is not None and len(missing) < len(unit.entity_ids):
        try:  # reasons are optional: nothing in them may cost the unit
            reason_list, reason_note = reasons(unit, chat, budget, start, schedule, rows, claims_by_entity, seed,
                                               plan.passages, rationales(unit, rows, plan.samples))
        except Exception as exc:  # noqa: BLE001
            reason_list, reason_note = [], f"dropped ({type(exc).__name__})"
    answer = {
        "task_id": unit.task.get("task_id", ""),
        "schema_version": str(unit.task.get("schema_version", "3")),
        "target_type": unit.target_type,
        "entity_predictions": rows,
        "evidence_trace": (
            f"{plan.trace.mode}: {len(unit.docs)} admitted documents dated on or before "
            f"{unit.task.get('cutoff_date')}; {budget.used} House requests; forecasts are medians "
            f"of sampled House replies; claims quote admitted documents verbatim; reasons {reason_note}."
            + (f" {len(missing)} of {len(unit.entity_ids)} entities had no House forecast and carry the task "
               "table forward (degraded)." if missing else "")
        ),
        "notes": {"samples_per_entity": {eid: results[eid].samples for eid in unit.entity_ids}},
    }
    if reason_list:
        answer["submitted_reasons"] = reason_list
    problems = validate(unit, answer)
    if problems:
        log("  self-validation failed (" + "; ".join(problems[:5]) + "); writing the degraded answer")
        results = {eid: degraded_forecast(unit, rows_by_id[eid]) for eid in unit.entity_ids}
        rows, _ = assemble(unit, results, _plan_without_house(unit, seed))
        answer.update(entity_predictions=rows, notes={"degraded": "self-validation failed"})
        answer.pop("submitted_reasons", None)
        problems = validate(unit, answer)
        if problems:
            raise RuntimeError("answer failed self-validation: " + "; ".join(problems[:10]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(answer, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    log(f"wrote {out_path}: {len(rows)} entities, {sum(len(r['claims']) for r in rows)} claims, "
        f"reasons {reason_note}, {time.monotonic() - start:.1f}s")
    return answer


def main(argv: list[str] | None = None) -> int:
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    budget = Budget()
    try:
        chat = HouseClient(HouseConfig.from_env(), budget)
    except RuntimeError as exc:  # no House route: a degraded answer is still written
        print(f"House unavailable: {exc}", file=sys.stderr)
        chat = None
    run(args.task, args.corpus, args.out, chat, budget, Schedule.for_unit_timeout(card_timeout(args.task)), start)
    return 0


def entrypoint() -> None:
    code = 1
    try:
        code = main()
    except SystemExit as exc:  # argparse
        code = exc.code if isinstance(exc.code, int) else 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero; never write a partial answer
        print(f"analyze failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    entrypoint()
