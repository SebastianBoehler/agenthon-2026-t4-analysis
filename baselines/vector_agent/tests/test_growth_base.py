"""Reject arithmetic grounded in a different quarter's task-table value."""
import pytest

from baselines.vector_agent.consistency import ground_growth
from baselines.vector_agent.parse import Sample


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
