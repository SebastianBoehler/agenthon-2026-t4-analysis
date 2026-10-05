"""## Executive summary (read this first)

Manifest admission must govern retrieved evidence and emitted exact quotations.
Shared context remains eligible; absent, malformed and empty admission fails explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.admission import load_admission
from baselines.strong_rag_baseline.cli import _mock_reply, run
from baselines.strong_rag_baseline.client import MockModelClient


def unit(tmp_path: Path, labels: dict) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = []
    for doc_id, label in labels.items():
        path = corpus / f"{doc_id}.json"
        path.write_text(json.dumps({"doc_id": doc_id, "doc_date": "2024-01-01",
                                   "text": f"{doc_id} revenue results and outlook."}))
        entries.append({"path": f"corpus/{path.name}", "role": "corpus",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **label})
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}))
    (tmp_path / "task.json").write_text(json.dumps({
        "task_id": "admission-control", "cutoff_date": "2024-02-01",
        "target": {"name": "result", "type": "regression"},
        "entities": [{"entity_id": "A", "name": "Alpha"}], "interval_level": 0.9}))
    return tmp_path


def test_shared_and_exact_entity_labels(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]},
                        "peer": {"entity_ids": ["B"]},
                        "shared": {"shared": True}, "off_roster": {"entity_ids": []}})
    policy = load_admission(u)
    assert policy["own"] == frozenset({"A"})
    assert policy["shared"] is None
    assert policy["off_roster"] == frozenset()


@pytest.mark.parametrize("label", [{}, {"shared": False}, {"shared": "true"},
    {"entity_ids": "A"}, {"entity_ids": [""]}, {"entity_ids": ["A", "A"]},
    {"entity_ids": ["A"], "shared": True}])
def test_missing_or_malformed_labels_are_errors(tmp_path, label):
    with pytest.raises(ValueError, match="admission"):
        load_admission(unit(tmp_path, {"own": label}))


def test_manifest_digest_and_path_are_enforced(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}})
    (u / "corpus/own.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        load_admission(u)
    manifest = json.loads((u / "manifest.json").read_text())
    manifest["files"][0]["path"] = "corpus/../own.json"
    (u / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        load_admission(u)


def test_unmanifested_and_foreign_documents_never_reach_prompt_or_claims(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]},
                        "peer": {"entity_ids": ["B"]}, "shared": {"shared": True}})
    (u / "corpus/unmanifested.json").write_text(json.dumps({"doc_id": "unmanifested",
        "doc_date": "2024-01-01", "text": "revenue results outlook"}))
    seen = []

    def reply(system, user):
        seen.append(user)
        result = json.loads(_mock_reply(system, user))
        result["evidence"].append({"doc_id": "peer", "quote": "peer revenue",
                                    "claim": "The peer reported revenue."})
        return json.dumps(result)

    answer = run(u / "task.json", u / "corpus", u / "answer.json",
                 MockModelClient(reply=reply), 10)
    assert "doc_id=peer " not in seen[0]
    assert "doc_id=unmanifested " not in seen[0]
    assert "doc_id=shared " in seen[0]
    assert all(c["doc_id"] in {"own", "shared"}
               for c in answer["entity_predictions"][0]["claims"])
    assert answer["notes"]["dropped_evidence_items"] == 1


def test_no_admitted_pre_cutoff_evidence_fails_before_model_call(tmp_path):
    u = unit(tmp_path, {"peer": {"entity_ids": ["B"]}})
    calls = []
    with pytest.raises(ValueError, match="admitted"):
        run(u / "task.json", u / "corpus", u / "answer.json",
            MockModelClient(reply=lambda *args: calls.append(args)), 10)
    assert not calls and not (u / "answer.json").exists()


def test_foreign_only_response_fails_without_fabricating_claim(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}, "peer": {"entity_ids": ["B"]}})
    reply = json.dumps({"point_forecast": 0, "interval": {"lo": -1, "hi": 1},
                       "evidence": [{"doc_id": "peer", "quote": "peer revenue",
                                     "claim": "The peer reported revenue."}]})
    with pytest.raises(ValueError, match="no admitted grounded claims"):
        run(u / "task.json", u / "corpus", u / "answer.json", MockModelClient(reply), 10)
    assert not (u / "answer.json").exists()


def test_post_cutoff_shared_text_cannot_be_cited(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}, "shared": {"shared": True}})
    path = u / "corpus/shared.json"
    data = json.loads(path.read_text()); data["doc_date"] = "2024-03-01"
    path.write_text(json.dumps(data))
    manifest = json.loads((u / "manifest.json").read_text())
    manifest["files"][1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (u / "manifest.json").write_text(json.dumps(manifest))
    seen = []

    def reply(system, user):
        seen.append(user)
        result = json.loads(_mock_reply(system, user))
        result["evidence"].append({"doc_id": "shared", "quote": "shared revenue",
                                    "claim": "Shared revenue was reported."})
        return json.dumps(result)

    answer = run(u / "task.json", u / "corpus", u / "answer.json", MockModelClient(reply), 10)
    assert "doc_id=shared " not in seen[0]
    assert all(c["doc_id"] == "own" for c in answer["entity_predictions"][0]["claims"])


@pytest.mark.parametrize("spoofed_file", ["peer", "unmanifested"])
def test_document_identity_collision_fails_before_model(tmp_path, spoofed_file):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}, "peer": {"entity_ids": ["B"]}})
    path = u / "corpus" / f"{spoofed_file}.json"
    path.write_text(json.dumps({"doc_id": "own", "doc_date": "2024-01-01",
                               "text": "Unverified revenue results outlook."}))
    manifest = json.loads((u / "manifest.json").read_text())
    if spoofed_file == "peer":
        manifest["files"][1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (u / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    with pytest.raises(ValueError, match="document identity"):
        run(u / "task.json", u / "corpus", u / "answer.json",
            MockModelClient(reply=lambda *args: calls.append(args)), 10)
    assert not calls and not (u / "answer.json").exists()


def test_cik_does_not_override_explicit_manifest_admission(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}, "shared": {"shared": True}})
    path = u / "task.json"
    task = json.loads(path.read_text())
    task["entities"][0]["cik"] = "0000000001"
    path.write_text(json.dumps(task))
    answer = run(path, u / "corpus", u / "answer.json", MockModelClient(_mock_reply), 10)
    assert answer["entity_predictions"][0]["claims"]
