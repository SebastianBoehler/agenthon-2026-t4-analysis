"""## Executive summary (read this first)

Prompt serialization preserves supplied label meanings without inventing omissions.
Other target types and missing assertion metadata retain their existing prompt.
"""
import json

import pytest

from baselines.strong_rag_baseline.prompts import build_user_prompt

PREFIX = "LABEL ASSERTIONS (from task.json): "


def task(target_type="classification", assertions=None):
    target = {"name": "direction", "type": target_type, "labels": ["up", "down"]}
    if assertions is not None:
        target["label_assertions"] = assertions
    return {"prompt": "Use this full task statement.", "cutoff_date": "2024-01-01",
            "target": target}


def assertions_from(prompt):
    line = next(line for line in prompt.splitlines() if line.startswith(PREFIX))
    return json.loads(line[len(PREFIX):])


def test_supplied_assertions_are_exact_and_before_entity_and_evidence():
    definitions = {"up": 'Value rises beyond the supplied threshold: "higher".',
                   "down": "Value falls; preserve μ and a\nline break."}
    prompt = build_user_prompt(task(assertions=definitions), {"entity_id": "row"}, [])
    assert assertions_from(prompt) == definitions
    assert "TASK: Use this full task statement." in prompt
    assert prompt.index(PREFIX) < prompt.index("\nENTITY:")
    assert prompt.index(PREFIX) < prompt.index("EVIDENCE EXCERPTS")


@pytest.mark.parametrize("definitions", [None, {}], ids=["missing", "empty"])
def test_missing_assertions_do_not_add_guessed_meanings(definitions):
    prompt = build_user_prompt(task(assertions=definitions), {"entity_id": "row"}, [])
    assert PREFIX not in prompt


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_nonclassification_prompt_is_unchanged(target_type):
    plain = build_user_prompt(task(target_type), {"entity_id": "row"}, [])
    supplied = build_user_prompt(task(target_type, {"up": "Supplied meaning."}),
                                 {"entity_id": "row"}, [])
    assert supplied == plain


def test_partial_supplied_mapping_is_not_completed_or_reordered():
    definitions = {"down": "Only this definition was supplied."}
    prompt = build_user_prompt(task(assertions=definitions), {"entity_id": "row"}, [])
    assert assertions_from(prompt) == definitions
    assert list(assertions_from(prompt)) == ["down"]
