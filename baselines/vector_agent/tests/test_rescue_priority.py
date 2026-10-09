"""Spend a final rescue request on missing coverage before another sample."""
from dataclasses import replace
from pathlib import Path

import pytest

from baselines.vector_agent.agent import Planner
from baselines.vector_agent.parse import Sample
from baselines.vector_agent.unit import load_unit


ROOT = Path(__file__).resolve().parents[3]


def exhausted_plan():
    path = ROOT / "units" / "t4-fomc-curve-20240918"
    plan = Planner(load_unit(path / "task.json", path / "corpus"), 0)
    assert plan.full
    for eid in plan.ids:
        sample = Sample(eid, 0.0, -1.0, 1.0, None, None, None, "")
        plan.samples[eid] = [sample, sample]
    plan.samples[plan.ids[0]].pop()
    plan.samples[plan.ids[1]].clear()
    plan.attempts["g0"] = plan.target + 4
    return plan


def test_final_rescue_request_covers_uncovered_entity():
    plan = exhausted_plan()
    job = plan.next_job(budget_left=1)
    assert job.ids == (plan.ids[1],)


def test_uncovered_priority_applies_across_groups():
    plan = exhausted_plan()
    plan.samples[plan.ids[1]] = [replace(s, entity_id=plan.ids[1]) for s in plan.samples[plan.ids[2]]]
    plan.samples[plan.ids[3]].clear()
    plan.groups = [plan.ids[:3], plan.ids[3:]]
    plan.attempts["g1"] = plan.target + 4
    assert plan.next_job(budget_left=1).ids == (plan.ids[3],)


@pytest.mark.parametrize("restriction,value", [("inflight", 1), ("attempts", 3)])
def test_unavailable_uncovered_entity_does_not_block_eligible_rescue(restriction, value):
    plan = exhausted_plan()
    getattr(plan, restriction)[plan.ids[1]] = value
    assert plan.next_job(budget_left=1).ids == (plan.ids[0],)


def test_equal_coverage_keeps_table_order():
    plan = exhausted_plan()
    plan.samples[plan.ids[1]] = [replace(s, entity_id=plan.ids[1]) for s in plan.samples[plan.ids[0]]]
    assert plan.next_job(budget_left=1).ids == (plan.ids[0],)


@pytest.mark.parametrize("retired", [False, True])
def test_inflight_or_retired_group_cannot_be_rescued(retired):
    plan = exhausted_plan()
    if retired:
        plan.retired.add(0)
    else:
        plan.inflight["g0"] = 1
    assert plan.next_job(budget_left=1) is None
