"""## Executive summary (read this first)

Legal character boundaries retain exact evidence; oversized items are dropped whole.
These controls separate Unicode characters from bytes and exercise full mixed output.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from baselines.strong_rag_baseline.agent import _ground_claims
from baselines.strong_rag_baseline.cli import _mock_reply, run
from baselines.strong_rag_baseline.client import MockModelClient
from baselines.strong_rag_baseline.indexer import IndexedCorpus
from .test_admission import unit


def corpus(text):
    return IndexedCorpus(chunks=[], doc_texts={"own": text},
                         doc_dates={"own": "2024-01-01"})


@pytest.mark.parametrize("length,accepted", [(7999, True), (8000, True), (8001, False)])
def test_exact_citation_inclusive_code_point_boundary(length, accepted):
    text = "é" * 9000
    quote = text[:length]
    claims, dropped = _ground_claims([
        {"doc_id": "own", "quote": quote, "claim": "Revenue increased."}
    ], corpus(text), {"own"})
    assert bool(claims) is accepted
    assert dropped == int(not accepted)
    if accepted:
        assert claims[0]["span_end"] - claims[0]["span_start"] == length
        assert text[claims[0]["span_start"]:claims[0]["span_end"]] == quote


@pytest.mark.parametrize("length,accepted", [(3999, True), (4000, True), (4001, False)])
def test_claim_inclusive_code_point_boundary(length, accepted):
    claim = "é" * length
    claims, dropped = _ground_claims([
        {"doc_id": "own", "quote": "Revenue increased.", "claim": claim}
    ], corpus("Revenue increased."), {"own"})
    assert bool(claims) is accepted
    assert dropped == int(not accepted)
    if accepted:
        assert claims[0]["claim"] == claim


def test_oversized_nonexact_quote_is_not_substituted():
    text = "revenue increased strongly " * 400
    quote = text + " never stated"
    claims, dropped = _ground_claims([
        {"doc_id": "own", "quote": quote, "claim": "Revenue increased."}
    ], corpus(text), {"own"})
    assert claims == [] and dropped == 1


def test_stripped_legal_quote_and_punctuation_normalization_keep_offsets():
    text = "Management’s revenue increased."
    claims, dropped = _ground_claims([
        {"doc_id": "own", "quote": "  Management's revenue increased.  ",
         "claim": "Revenue increased."}
    ], corpus(text), {"own"})
    assert dropped == 0 and len(claims) == 1
    assert text[claims[0]["span_start"]:claims[0]["span_end"]] == text


@pytest.fixture
def long_admitted_unit(tmp_path):
    u = unit(tmp_path, {"own": {"entity_ids": ["A"]}})
    text = "Revenue increased strongly. " * 500
    path = u / "corpus/own.json"
    doc = json.loads(path.read_text()); doc["text"] = text; path.write_text(json.dumps(doc))
    manifest = json.loads((u / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (u / "manifest.json").write_text(json.dumps(manifest))
    return u, text


def test_mixed_model_output_keeps_valid_claim_without_shortening_bad_items(long_admitted_unit):
    u, text = long_admitted_unit

    def reply(system, user):
        result = json.loads(_mock_reply(system, user))
        result["evidence"].extend([
            {"doc_id": "own", "quote": text[:8100], "claim": "Revenue increased."},
            {"doc_id": "own", "quote": "Revenue increased strongly.",
             "claim": "Revenue increased strongly. " * 200},
        ])
        return json.dumps(result)

    answer = run(u / "task.json", u / "corpus", u / "answer.json", MockModelClient(reply), 10)
    assert len(answer["entity_predictions"][0]["claims"]) == 1
    assert answer["notes"]["dropped_evidence_items"] == 2


@pytest.mark.parametrize("oversized", ["quote", "claim"])
def test_only_over_cap_evidence_fails_without_writing_answer(long_admitted_unit, oversized):
    u, text = long_admitted_unit
    evidence = {"doc_id": "own", "quote": "Revenue increased strongly.",
                "claim": "Revenue increased strongly."}
    evidence[oversized] = text[:8100] if oversized == "quote" else text[:4100]
    reply = json.dumps({"point_forecast": 0, "interval": {"lo": -1, "hi": 1},
                        "evidence": [evidence]})
    with pytest.raises(ValueError, match="no admitted grounded claims"):
        run(u / "task.json", u / "corpus", u / "answer.json", MockModelClient(reply), 10)
    assert not (u / "answer.json").exists()


@pytest.mark.parametrize("claim,accepted", [
    ("e\u0301" * 2001, True),  # raw4002 -> NFC2001: legal
    ("\u0344" * 2001, False),  # raw2001 -> NFC4002: oversized
], ids=["nfc-contracts", "nfc-expands"])
def test_claim_cap_uses_official_nfc_count_and_preserves_original(claim, accepted):
    claims, dropped = _ground_claims([
        {"doc_id": "own", "quote": "Revenue increased.", "claim": claim}
    ], corpus("Revenue increased."), {"own"})
    assert bool(claims) is accepted
    assert dropped == int(not accepted)
    if accepted:
        assert claims[0]["claim"] == claim
