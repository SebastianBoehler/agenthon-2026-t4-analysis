"""Honor an invented resolving-report comparative without inventing a future value."""
import json

import pytest

from baselines.vector_agent.agent import Job, Planner
from baselines.vector_agent.consistency import percent_growth_target
from baselines.vector_agent.growth_contract import forecast_method, report_comparative_target
from baselines.vector_agent.prompts import _METHOD, entity_prompt, unit_prompt
from baselines.vector_agent.unit import Unit

REPORT_PROMPT = (
    "Forecast growth = 100 * (current / comparative - 1), using both figures exactly "
    "as printed in that report. Both values resolve in the future report."
)


def unit(prompt):
    return Unit({"prompt": prompt, "target": {"name": "revenue_yoy_growth_pct",
                                             "type": "regression"},
                 "entities": [{"entity_id": "A", "prior_year_q_revenue_musd": 100.0,
                               "latest_q_revenue_musd": 90.0}]}, {})


@pytest.mark.parametrize("base", [100.0, 110.0])
def test_report_comparative_is_not_overwritten_by_task_table(base):
    loaded = unit(REPORT_PROMPT)
    plan = Planner(loaded, seed=0)
    point = (120 / 110 - 1) * 100  # 110 is a hypothetical forecast, never an observed outcome
    item = {"entity_id": "A", "point_forecast": point, "lo": point - 5,
            "hi": point + 5, "level_forecast": 120.0, "base_value": base}
    plan.absorb(Job("group", "g0", ("A",), "", 0, 0, 100, 1),
                json.dumps({"predictions": [item]}), None)
    got = plan.samples["A"][0]
    assert (got.point, got.lo, got.hi) == (point, point - 5, point + 5)
    assert not plan.growth


@pytest.mark.parametrize("prompt", [REPORT_PROMPT, REPORT_PROMPT.upper().replace(" ", "\n  ")])
def test_report_contract_recognizes_case_and_whitespace(prompt):
    assert report_comparative_target(unit(prompt).task)


def test_fixed_table_instructions_and_predicate_remain_unchanged():
    loaded = unit("Forecast growth as a percent relative to the given table comparative.")
    assert percent_growth_target(loaded.task)
    assert forecast_method(_METHOD, loaded.task) == _METHOD
    assert not percent_growth_target(dict(loaded.task, target={"name": "yoy_pct", "type": "classification"}))


@pytest.mark.parametrize("single", [False, True])
def test_both_prompt_paths_use_forecasted_report_comparative(single):
    loaded = unit(REPORT_PROMPT)
    prompt = entity_prompt(loaded, loaded.entities[0], []) if single else unit_prompt(loaded, [])
    assert "forecast both the target-period level and that report's comparative" in prompt
    assert "your forecast of the comparative as printed in the resolving report" in prompt
    assert "using only pre-cutoff evidence" in prompt
    assert "copy that base value from the table row" not in prompt
    assert "base value copied from the table" not in prompt
    assert "the level stays at the table's base value" not in prompt
    assert "target value in the task's stated units" in prompt
