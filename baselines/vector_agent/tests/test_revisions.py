"""Invented vintage tables for the same-stage revision-direction rule."""
from baselines.vector_agent.revisions import revision_labels, stage_majority, vintage_rows
from baselines.vector_agent.unit import Doc, Unit

TABLE = """SERIES X — invented
VINTAGE TABLE — invented values.

reference_month | as_of_2020-02-01 | as_of_2020-03-01 | as_of_2020-04-01 | as_of_2020-05-01
2019-11 | 10.0 | 9.0 | 8.5 | 8.5
2019-12 | | 20.0 | 19.0 | 18.0
2020-01 | | | 30.0 | 31.0
2020-02 | | | | 40.0
"""


def unit(labels=("up", "down"), name="next_estimate_revision_direction", shared=False):
    task = {"task_id": "invented", "schema_version": "3", "cutoff_date": "2020-05-02",
            "target": {"type": "classification", "name": name, "labels": list(labels)},
            "entities": [{"entity_id": "X_2020-02", "series_id": "X", "ref_month": "2020-02"},
                         {"entity_id": "X_2020-01", "series_id": "X", "ref_month": "2020-01"}],
            "prompt": "Predict whether the next estimate will be revised up or down."}
    ids = None if shared else frozenset({"X_2020-02", "X_2020-01"})
    return Unit(task, {"V": Doc("V", "2020-05-01", "Invented vintages", TABLE, ids)})


def test_rows_keep_published_values_in_release_order():
    rows = vintage_rows(TABLE)
    assert rows["2019-12"] == [20.0, 19.0, 18.0] and rows["2020-02"] == [40.0]


def test_stage_majority_reads_same_stage_changes_in_other_months():
    rows = vintage_rows(TABLE)
    assert stage_majority(rows, "2020-02") == -1  # first revisions: 10->9 down, 20->19 down, 30->31 up
    assert stage_majority(rows, "2020-01") == -1  # second revisions: 9->8.5 down, 19->18 down
    assert stage_majority(rows, "2020-03") == 0  # month absent from the table


def test_labels_follow_the_task_vocabulary():
    assert revision_labels(unit()) == {"X_2020-02": "down", "X_2020-01": "down"}
    assert revision_labels(unit(labels=("Revised UP", "Revised DOWN")))["X_2020-02"] == "Revised DOWN"


def test_other_tasks_and_shared_documents_are_left_alone():
    assert revision_labels(unit(labels=("up", "flat", "down"))) == {}
    assert revision_labels(unit(name="eps_yoy_direction")) != {}  # the prompt still names a revision
    task_free = unit(name="eps_yoy_direction")
    task_free.task["prompt"] = "Predict the direction of earnings."
    assert revision_labels(task_free) == {}
    assert revision_labels(unit(shared=True)) == {}
