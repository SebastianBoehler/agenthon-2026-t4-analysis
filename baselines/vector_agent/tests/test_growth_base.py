"""Reject arithmetic grounded in a different quarter's task-table value."""
import json
import math

import pytest

from baselines.vector_agent.agent import Job, Planner
from baselines.vector_agent.aggregate import combine
from baselines.vector_agent.consistency import ground_growth
from baselines.vector_agent.parse import Sample
from baselines.vector_agent.unit import Unit


def sample(base):
    return Sample("A", 15.0, 5.0, 25.0, None, None, None, "", level=200.0, base=base)


@pytest.mark.parametrize("other_base", [150.0, 120.0, 25.0])
def test_other_quarter_or_growth_number_cannot_be_target_base(other_base):
    row = {"entity_id": "A", "prior_year_q_revenue_musd": 100.0,
           "latest_q_revenue_musd": 150.0, "latest_q_prior_year_revenue_musd": 120.0,
           "latest_q_yoy_growth_pct": 25.0}
    got = ground_growth(sample(other_base), row)
    assert (got.point, got.lo, got.hi) == (15.0, 5.0, 25.0)


def test_declared_target_base_still_recomputes_point_and_shifts_interval():
    row = {"entity_id": "A", "prior_year_q_revenue_musd": 100.0,
           "latest_q_revenue_musd": 150.0}
    got = ground_growth(sample(100.0), row)
    assert (got.point, got.lo, got.hi) == (100.0, 90.0, 110.0)


def test_declared_eps_base_rejects_another_numeric_field():
    row = {"entity_id": "A", "prior_year_q_eps": 100.0, "latest_q_eps": 150.0}
    got = ground_growth(sample(150.0), row)
    assert (got.point, got.lo, got.hi) == (15.0, 5.0, 25.0)


def test_finite_reply_cannot_overflow_during_growth_correction():
    task = {"target": {"name": "eps_yoy_growth_pct", "type": "regression"},
            "entities": [{"entity_id": "A", "prior_year_q_eps": 100.0,
                          "latest_q_eps": 150.0}]}
    plan = Planner(Unit(task, {}), seed=0)
    item = {"entity_id": "A", "point_forecast": -1.7e308, "lo": -1.7e308,
            "hi": 1.7e308, "level_forecast": 200.0, "base_value": 100.0}
    job = Job("group", "g0", ("A",), "", 0.0, 0, 100, 1.0)
    plan.absorb(job, json.dumps({"predictions": [item]}), None)
    got = plan.samples["A"][0]
    assert (got.point, got.lo, got.hi) == (-1.7e308, -1.7e308, 1.7e308)
    pooled = combine([got], "regression")
    assert all(math.isfinite(value) for value in (pooled.point, pooled.lo, pooled.hi))
    assert not plan.trace.failures
