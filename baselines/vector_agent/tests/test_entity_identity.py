"""Keep explicitly identified replies inside the requested entity roster."""
import json

import pytest

from baselines.vector_agent.agent import Planner
from baselines.vector_agent.parse import parse_predictions, to_sample
from baselines.vector_agent.unit import Doc, Unit


def parse(row, ids={"A"}):
    return parse_predictions(json.dumps(row), target_type="regression", labels=[],
                             entity_ids=ids, weight=1, source="test")


def prediction(**fields):
    return {"point_forecast": 7, "lo": 6, "hi": 8, **fields}


@pytest.mark.parametrize("identity", ["B", None, True, False, "", "  "])
@pytest.mark.parametrize("wrapped", [False, True])
def test_explicit_unresolved_id_is_not_reassigned(identity, wrapped):
    row = prediction(entity_id=identity)
    assert parse({"predictions": [row]} if wrapped else row) == []
    assert to_sample(row, target_type="regression", labels=[], entity_ids={"A"},
                     weight=1, source="test", only="A") is None


def test_foreign_key_is_not_reassigned():
    assert parse({"predictions": {"B": prediction()}}) == []


@pytest.mark.parametrize("identity", ["A", "a", " A "])
def test_valid_singleton_ids_keep_their_values(identity):
    sample, = parse(prediction(entity_id=identity))
    assert (sample.entity_id, sample.point, sample.lo, sample.hi) == ("A", 7, 6, 8)


def test_omitted_singleton_id_is_still_accepted():
    sample, = parse(prediction())
    assert sample.entity_id == "A"
    assert parse({"predictions": [prediction()]}, {"A", "B"}) == []


@pytest.mark.parametrize("identity", [0, 0.0, "0"])
def test_zero_is_an_explicit_entity_id(identity):
    sample, = parse(prediction(entity_id=identity), {"0"})
    assert sample.entity_id == "0"


@pytest.mark.parametrize("identity", ["B", "A"])
def test_actual_singleton_job_keeps_request_identity(identity):
    task = {"prompt": "Forecast the invented quantity.",
            "target": {"type": "regression", "name": "quantity"},
            "entities": [{"entity_id": "A"}, {"entity_id": "B"}]}
    doc = Doc("invented", "2020-01-01", "Invented evidence", "x" * 110001, None)
    plan = Planner(Unit(task, {doc.doc_id: doc}), seed=0)
    job = plan.next_job(budget_left=25)
    assert not plan.full and job.ids == ("A",)
    plan.mark_launched(job)
    plan.absorb(job, json.dumps(prediction(entity_id=identity)), None)
    assert not plan.samples["B"]
    if identity == "B":
        assert not plan.samples["A"] and plan.uncovered()
        assert plan.done[job.key] == plan.trace.succeeded[job.kind] == 0
        assert plan.trace.failures[-1] == f"{job.key}: unusable reply"
    else:
        sample, = plan.samples["A"]
        assert sample.point == 7 and plan.trace.succeeded[job.kind] == 1
