"""Synthetic forecasts retain pooled widths; corpus volatility floors still apply.

These CPU controls exercise production forecast and CLI paths without model calls.
"""
import json
import math
import time
from importlib.resources import files

import pytest
from jsonschema.validators import validator_for

from baselines.vector_agent import agent, cli
from baselines.vector_agent.aggregate import combine
from baselines.vector_agent.house import Budget
from baselines.vector_agent.parse import Sample
from baselines.vector_agent.unit import Doc, Unit


SOURCE = "Synthetic source states the historical level of the forecast quantity."
STOPPED = agent.Schedule(launch_stop=-1, hard_stop=0, reasons_stop=-1, workers=1)


def prepared_plan(kind="regression", name="native_level", point=10, lo=8, hi=13):
    task = {"task_id": "synthetic", "schema_version": "3", "cutoff_date": "2020-01-02",
            "target": {"type": kind, "name": name},
            "entities": [{"entity_id": "A", "latest_level": 10}]}
    if kind == "classification":
        task["target"]["labels"] = ["up", "flat", "down"]
    unit = Unit(task, {"D": Doc("D", "2020-01-01", "Synthetic", SOURCE, None)})
    plan = agent.Planner(unit, 0)
    plan.samples["A"] = [Sample("A", point, lo, hi,
        "up" if kind == "classification" else None, None, None, "Synthetic analysis",
        quotes=[("D", SOURCE)])]
    return unit, plan


def finish(unit, plan, monkeypatch, scale=1.0):
    # Synthetic accepted samples; expired launch deadline prevents every model call.
    monkeypatch.setattr(agent, "Planner", lambda unit, seed: plan)
    for name in ("BAND_SCALE", "YIELD_BAND_SCALE", "CLASS_BAND_SCALE"):
        monkeypatch.setattr(agent, name, scale)
    results, _ = agent.forecast(unit, None, Budget(), time.monotonic(), STOPPED)
    return results["A"]


@pytest.mark.parametrize("kind,name", [("regression", "native_level"),
    ("ranking", "native_level"), ("regression", "yield_change_bps"),
    ("classification", "eps_yoy_direction")])
def test_global_scale_widens_numeric_bands_about_the_point(kind, name, monkeypatch):
    unit, plan = prepared_plan(kind, name)
    result = finish(unit, plan, monkeypatch, scale=agent.BAND_SCALE)
    assert agent.BAND_SCALE == agent.YIELD_BAND_SCALE == agent.CLASS_BAND_SCALE == 2.3
    assert result.point == 10
    assert result.lo == pytest.approx(10 - 2.3 * 2) and result.hi == pytest.approx(10 + 2.3 * 3)


@pytest.mark.parametrize("point,lo,hi,probability", [(.7, 0, 1, False), (.7, .5, .9, True)])
def test_global_scale_leaves_placeholder_and_probability_bands(point, lo, hi, probability, monkeypatch):
    unit, plan = prepared_plan("classification", "synthetic_quantity", point, lo, hi)
    if probability:
        unit.task["prompt"] = "Forecast the probability of the event."
    expected = combine(plan.samples["A"], "classification", unit_interval=probability)
    assert finish(unit, plan, monkeypatch, scale=agent.BAND_SCALE) == expected


@pytest.mark.parametrize("kind,name", [("regression", "native_level"),
    ("ranking", "native_level"), ("regression", "yield_change_bps"),
    ("classification", "eps_yoy_direction")])
def test_asymmetric_pooled_interval_has_no_global_rescaling(kind, name, monkeypatch):
    unit, plan = prepared_plan(kind, name)
    expected = combine(plan.samples["A"], kind)
    result = finish(unit, plan, monkeypatch)
    assert (result.point, result.lo, result.hi) == (10, 8, 13)
    assert result == expected


@pytest.mark.parametrize("point,lo,hi,probability", [
    (None, 8, 13, False), (.7, 0, 1, False), (.7, .5, .9, True)])
def test_classification_missingpoint_placeholder_and_probability_stay_valid(
        point, lo, hi, probability, monkeypatch):
    unit, plan = prepared_plan("classification", "synthetic_quantity", point, lo, hi)
    if probability:
        unit.task["prompt"] = "Forecast the probability of the event."
    expected = combine(plan.samples["A"], "classification", unit_interval=probability)
    assert finish(unit, plan, monkeypatch) == expected


def test_percent_yield_history_floor_converts_to_bps_once(monkeypatch):
    unit, plan = prepared_plan(name="yield_change_bps", point=0, lo=-.01, hi=.01)
    unit.task["resolution_date"] = "2020-02-02"
    unit.task["entities"][0].update(series="Y", unit="bps change")
    values = [3 + .01 * (i % 3) for i in range(10)]
    text = "Date | Y\n" + "\n".join(
        f"2019-12-{20+i:02d} | {value:.2f}" for i, value in enumerate(values))
    unit.docs = {"D": Doc("D", "2020-01-01", "Synthetic percent yields", text, None)}
    changes = [b-a for a, b in zip(values, values[1:])]
    mean = sum(changes) / len(changes)
    variance = sum((v-mean)**2 for v in changes) / (len(changes)-1)
    half = 1.6448536269514722 * 100 * math.sqrt(variance * 31)
    result = finish(unit, plan, monkeypatch)
    assert result.point == 0
    assert result.lo == pytest.approx(-half, rel=1e-5)
    assert result.hi == pytest.approx(half, rel=1e-5)
    assert any("1 of 1 bands widened" in note for note in plan.trace.notes)


def test_actual_CLI_keeps_point_threshold_label_and_resolved_claims(tmp_path, monkeypatch):
    unit, plan = prepared_plan("classification", "abnormal_return", 0, -.1, .1)
    unit.task["target"]["label_assertions"] = {"up": "greater than 1%",
        "flat": "between -1% and 1%", "down": "less than -1%"}
    monkeypatch.setattr(agent, "Planner", lambda unit, seed: plan)
    for name in ("BAND_SCALE", "YIELD_BAND_SCALE", "CLASS_BAND_SCALE"):
        monkeypatch.setattr(agent, name, 1.0)
    monkeypatch.setattr(cli, "load_unit", lambda task_path, corpus_dir: unit)
    out = tmp_path / "answer.json"
    answer = cli.run(tmp_path / "task.json", tmp_path / "corpus", out,
                     object(), Budget(), STOPPED, time.monotonic(), log=lambda *args: None)
    row = answer["entity_predictions"][0]
    assert row["point_forecast"] == 0 and row["label"] == "flat"
    assert row["interval"] == {"level": .9, "lo": -.1, "hi": .1}
    assert cli.validate(unit, answer) == []
    assert answer["notes"]["samples_per_entity"] == {"A": 1}
    for claim in row["claims"]:
        assert claim["doc_id"] == "D"
        assert unit.docs["D"].text[claim["span_start"]:claim["span_end"]] == claim["claim"]
    schema = json.loads(files("qfbench2_common").joinpath("schemas/analysis.schema.json").read_text())
    assert list(validator_for(schema)(schema).iter_errors(answer)) == []
    assert json.loads(out.read_text()) == answer
