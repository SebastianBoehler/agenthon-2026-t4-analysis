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


def test_slow_house_relaunches_after_the_first_wave_once_ready():
    """A House whose calls take ~200 s (p75) used to stop after its first wave of workers."""
    import time
    from pathlib import Path
    from baselines.vector_agent.agent import Schedule, forecast
    from baselines.vector_agent.house import Budget
    from baselines.vector_agent.tests.fake_chat import FakeChat
    from baselines.vector_agent.unit import load_unit

    root = Path(__file__).resolve().parents[3] / "units" / "t4-auction-btc-202411-us7"
    unit = load_unit(root / "task.json", root / "corpus")
    f = 1 / 100  # the 600 s schedule at 1/100 scale
    sched = Schedule(**{k: (v * f if isinstance(v, float) and k != "request_timeout" else v)
                        for k, v in Schedule().__dict__.items()})
    sched.request_timeout = 3.0

    class Slow(FakeChat):
        lat = [0.3, 0.4, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]  # two quick replies, then p75 above 2.0 s (200 s scaled)

        def complete(self, system, user, **kw):
            with self.lock:
                i = self.calls
            self.delay = self.lat[i] if i < len(self.lat) else 0.3
            return super().complete(system, user, **kw)

    _, plan = forecast(unit, Slow(), Budget(), time.monotonic(), sched)
    assert plan.trace.launched["unit"] > sched.workers  # a second wave launched


def test_retrieval_prompt_that_times_out_is_halved_until_it_fits():
    """A House that times out every prompt above a size limit leaves no retrieval entity uncovered."""
    import time
    from pathlib import Path
    from baselines.vector_agent.agent import Schedule, forecast
    from baselines.vector_agent.house import Budget
    from baselines.vector_agent.tests.fake_chat import FakeChat
    from baselines.vector_agent.unit import load_unit

    root = Path(__file__).resolve().parents[3] / "units" / "t4-eps-growth-2024Q3-banks"
    unit = load_unit(root / "task.json", root / "corpus")
    sched = Schedule()  # the real deadlines; synthetic replies return at once, so the run takes seconds

    class SizeLimited(FakeChat):
        def complete(self, system, user, **kw):
            if len(user) > 40_000:
                raise TimeoutError("synthetic timeout: prompt too large for the window")
            return super().complete(system, user, **kw)

    results, plan = forecast(unit, SizeLimited(), Budget(), time.monotonic(), sched)
    assert all(plan.samples[eid] for eid in plan.ids), {e: len(s) for e, s in plan.samples.items()}
