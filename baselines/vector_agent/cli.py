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
from pathlib import Path

from .agent import Schedule, assemble, forecast, reasons
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
                ok = (doc is not None and doc.admits(eid) and 0 <= c["span_start"] < c["span_end"] <= len(doc.text)
                      and doc.text[c["span_start"]:c["span_end"]] == c["claim"])
            if not ok:
                problems.append(f"{eid}: claim does not resolve verbatim")
    return problems


def run(task_path: Path, corpus_dir: Path, out_path: Path, chat: Chat, budget: Budget,
        schedule: Schedule, start: float, log=print) -> dict:
    unit = load_unit(task_path, corpus_dir)
    seed = int(os.environ.get("QFBENCH_SEED", "20261006") or 20261006)
    results, plan = forecast(unit, chat, budget, start, schedule, seed)
    missing = [eid for eid in unit.entity_ids if eid not in results]
    log(f"mode={plan.trace.mode} launched={dict(plan.trace.launched)} "
        f"succeeded={dict(plan.trace.succeeded)} requests={budget.used} "
        f"elapsed={time.monotonic() - start:.1f}s")
    for failure in plan.trace.failures[:20]:
        log(f"  call failure: {failure}")
    if missing:
        raise RuntimeError(f"no usable House forecast for {len(missing)} entities: {missing}")
    rows, claims_by_entity = assemble(unit, results, plan)
    reason_list, reason_note = reasons(unit, chat, budget, start, schedule, rows, claims_by_entity, seed)
    answer = {
        "task_id": unit.task.get("task_id", ""),
        "schema_version": str(unit.task.get("schema_version", "3")),
        "target_type": unit.target_type,
        "entity_predictions": rows,
        "evidence_trace": (
            f"{plan.trace.mode}: {len(unit.docs)} admitted documents dated on or before "
            f"{unit.task.get('cutoff_date')}; {budget.used} House requests; forecasts are medians "
            f"of sampled House replies; claims quote admitted documents verbatim; reasons {reason_note}."
        ),
        "notes": {"samples_per_entity": {eid: results[eid].samples for eid in unit.entity_ids}},
    }
    if reason_list:
        answer["submitted_reasons"] = reason_list
    problems = validate(unit, answer)
    if problems:
        raise RuntimeError("answer failed self-validation: " + "; ".join(problems[:10]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(answer, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
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
    chat = HouseClient(HouseConfig.from_env(), budget)
    run(args.task, args.corpus, args.out, chat, budget, Schedule(), start)
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
