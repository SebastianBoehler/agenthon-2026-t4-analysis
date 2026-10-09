"""Synthetic regressions for target units and disclosed validation recovery."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.vector_agent import cli
from baselines.vector_agent.agent import Schedule
from baselines.vector_agent.degraded import degraded_forecast
from baselines.vector_agent.house import Budget
from baselines.vector_agent.unit import load_unit

from .fake_chat import FakeChat

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name,point,band", [
    ("credit_spread_bps", 125.0, (62.5, 187.5)),
    ("credit_spread_change_bps", 0.0, (-375.0, 375.0)),
])
def test_basis_points_do_not_define_a_change_target(name, point, band):
    unit = SimpleNamespace(target_type="regression", target={"name": name})
    row = {"entity_id": "SYNTHETIC", "latest_credit_spread_bps": 125.0}
    forecast = degraded_forecast(unit, row)
    assert forecast.point == point
    assert (forecast.lo, forecast.hi) == band


def test_validation_replacement_discloses_degraded_rows_and_removed_reasons(tmp_path, monkeypatch):
    unit_dir = ROOT / "units/t4-fomc-curve-20240918"
    unit = load_unit(unit_dir / "task.json", unit_dir / "corpus")
    original_validate = cli.validate
    validation_calls = []

    def fail_first(loaded, answer):
        validation_calls.append(json.loads(json.dumps(answer)))
        if len(validation_calls) == 1:
            return ["synthetic forced validation failure"]
        return original_validate(loaded, answer)

    monkeypatch.setattr(cli, "validate", fail_first)
    lines = []
    output = tmp_path / "answer.json"
    answer = cli.run(unit_dir / "task.json", unit_dir / "corpus", output, FakeChat(), Budget(),
                     Schedule(), time.monotonic(), log=lines.append)
    assert len(validation_calls) == 2
    assert validation_calls[0].get("submitted_reasons")
    assert original_validate(unit, answer) == []
    assert json.loads(output.read_text()) == answer
    assert all(row["point_forecast"] == 0.0 for row in answer["entity_predictions"])
    assert "submitted_reasons" not in answer
    assert answer["notes"] == {"degraded": "self-validation failed"}
    trace = answer["evidence_trace"]
    assert "self-validation failed" in trace
    assert f"all {len(unit.entity_ids)} entities carry the task table forward" in trace
    assert "medians of sampled House replies" not in trace
    assert "reasons removed (self-validation failed)" in trace
    assert "reasons removed (self-validation failed)" in lines[-1]
