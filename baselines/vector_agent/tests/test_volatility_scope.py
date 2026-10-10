"""Invented entity-specific histories retain their own volatility floor.

These controls exercise actual forecast assembly without House calls or outcomes.
"""
import math
import statistics
import time
from unittest.mock import patch

import pytest

from baselines.vector_agent import agent, cli
from baselines.vector_agent.house import Budget
from baselines.vector_agent.parse import Sample
from baselines.vector_agent.unit import Doc, Unit


def history(eid, step, column="Y", count=10):
    values = [3 + step * (i % 3) for i in range(count)]
    text = f"Date | {column}\n" + "\n".join(
        f"2019-12-{20+i:02d} | {v:.4f}" for i, v in enumerate(values))
    changes = [b-a for a, b in zip(values, values[1:])]
    sd = statistics.stdev(changes) * math.sqrt(31) * 100
    return Doc(eid or "shared", "2019-12-31", "Invented history", text,
               None if eid is None else frozenset([eid])), sd


@pytest.mark.parametrize("case", ["A-first", "B-first", "reverse-roster", "A-only", "B-only",
                                 "shared", "unmatched", "short"])
def test_native_forecast_floor_uses_the_entity_history(case):
    a, sa = history("A", .01)
    b, sb = history("B", .1)
    shared, ss = history(None, .02)
    docs, expected = [a, b], {"A": sa, "B": sb}
    ids = ["A", "B"]
    if case in ("B-first", "reverse-roster"):
        docs = [b, a]
        if case == "reverse-roster":
            ids.reverse()
    elif case in ("A-only", "B-only"):
        docs, expected = ([a], {"A": sa}) if case == "A-only" else ([b], {"B": sb})
    elif case == "shared":
        docs, expected = [shared], {"A": ss, "B": ss}
    elif case in ("unmatched", "short"):
        d, _ = history("A", .02, column="Z" if case == "unmatched" else "Y",
                       count=5 if case == "short" else 10)
        docs, expected = [d], {}
    task = {"task_id": "invented-volatility", "schema_version": "3",
            "cutoff_date": "2020-01-01", "resolution_date": "2020-02-01",
            "target": {"type": "regression", "name": "yield_change_bps"},
            "entities": [{"entity_id": eid, "series": "Y", "unit": "bps change",
                          "latest_level": 3} for eid in ids]}
    unit = Unit(task, {d.doc_id: d for d in docs})
    plan = agent.Planner(unit, 0)
    for eid in ids:
        plan.samples[eid] = [Sample(eid, 0, -.01, .01, None, None, None, "Invented sample")]
    stopped = agent.Schedule(launch_stop=-1, hard_stop=0, reasons_stop=-1, workers=1)
    budget = Budget()
    with patch.object(agent, "Planner", lambda unit, seed: plan), \
            patch.object(agent, "YIELD_BAND_SCALE", 1.0):  # the floor itself, before global widening
        results, actual = agent.forecast(unit, None, budget, time.monotonic(), stopped)
    assert actual is plan and budget.used == 0 and not plan.trace.launched
    for eid, result in results.items():
        half = agent._Z90 * expected[eid] if eid in expected else .01
        assert result.point == 0
        assert result.lo == pytest.approx(-half, rel=1e-5)
        assert result.hi == pytest.approx(half, rel=1e-5)
    rows, _ = agent.assemble(unit, results, plan)
    assert cli.validate(unit, {"entity_predictions": rows}) == []
