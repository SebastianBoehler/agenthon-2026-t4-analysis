"""Regression checks for ambiguous labels and charged upstream rate-limit failures."""
import urllib.error
from unittest.mock import patch

import pytest

from baselines.vector_agent.house import Budget, BudgetExhausted, HouseClient, HouseConfig
from baselines.vector_agent.parse import match_label


def test_negated_or_partial_labels_are_rejected():
    assert match_label("not credit_event", ["credit_event", "no_event"]) is None
    assert match_label("credit", ["credit_event", "no_event"]) is None
    assert match_label("CREDIT-EVENT", ["credit_event", "no_event"]) == "credit_event"


def test_upstream_rate_limits_cannot_exceed_request_cap():
    budget = Budget(2)
    client = HouseClient(HouseConfig("http://local.test", "house", None), budget)
    error = urllib.error.HTTPError("http://local.test", 429, "rate limited", {}, None)
    with patch("urllib.request.urlopen", side_effect=error) as send, patch("time.sleep"):
        with pytest.raises(BudgetExhausted):
            client.complete("system", "user", temperature=0, seed=1, max_tokens=50, timeout=100)
    assert send.call_count == budget.used == 2


@pytest.mark.parametrize("point", [0.0, 1.0])
def test_degenerate_probability_band_stays_in_bounds(point):
    from baselines.vector_agent.aggregate import combine
    from baselines.vector_agent.parse import Sample
    sample = Sample("A", point, point, point, "event", None, None, "")
    forecast = combine([sample], "classification", unit_interval=True)
    assert 0.0 <= forecast.lo <= point <= forecast.hi <= 1.0
    assert forecast.lo < forecast.hi
