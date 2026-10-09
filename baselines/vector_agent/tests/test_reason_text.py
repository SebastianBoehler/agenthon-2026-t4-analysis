"""Synthetic reason filtering and masked-byte controls; no forecasting or judge claims."""
import json

import pytest

from baselines.vector_agent.claims import _denied, _within_caps, build_reasons
from baselines.vector_agent.reason_text import duplicate_key, fold_for_detection, mask_evidence_uris
from baselines.vector_agent.unit import Doc, Unit


@pytest.mark.parametrize("text", ["u\u200bnits/", "ｕｎｉｔｓ／", "u\u115fnits/",
    "u\u3164nits/", "units\u2215", "units\u2044", "units\u2e4a", "can\u200bary",
    "https\u2236\u2215\u2215example.test", "https：／／example.test"])
def test_disguised_denied_text_and_strict_URI_policy(text):
    assert _denied(text)


def test_detection_does_not_rewrite_the_original_text():
    text = "Revenue remains stable in café operations."
    assert fold_for_detection(text) == text and not _denied(text)


def test_duplicate_normalization_matches_published_unicode_operations():
    assert duplicate_key(" CAFÉ \u200b earnings\tstable ") == duplicate_key("cafe\u0301 earnings stable")
    assert duplicate_key("Ａ") != duplicate_key("A")  # duplicates use NFC, not NFKC


def fixture(texts):
    docs = {f"D{i}": Doc(f"D{i}", "2020-01-01", "Synthetic", text, None)
            for i, text in enumerate(texts)}
    unit = Unit({"task_id": "synthetic", "cutoff_date": "2020-01-02",
                 "target": {"type": "regression", "name": "native_level"},
                 "entities": [{"entity_id": "A"}]}, docs)
    candidates = [(f"R{i+1}", {"doc_id": doc.doc_id, "span_start": 0,
                   "span_end": len(doc.text), "claim": doc.text})
                  for i, doc in enumerate(docs.values())]
    return unit, candidates


def item(key, mechanism="Costs remain stable.", implication="The forecast uses stable costs.", support=None):
    return {"premise": key, "mechanism": mechanism, "answer_implication": implication,
            "support": support or [], "entities": ["A"]}


def test_builder_drops_disguised_path_without_touching_source():
    unit, candidates = fixture(["Company revenue remained stable during the period."])
    reply = {"reasons": [item("R1", mechanism="The u\u200bnits/ path supports revenue.")]}
    assert build_reasons(unit, reply, candidates) == []
    assert unit.docs["D0"].text == candidates[0][1]["claim"]


def test_first_three_distinct_reasons_keep_literal_premises_and_citations():
    texts = ["Company Café revenue remained stable.", "Company Cafe\u0301 revenue remained stable.",
             "Company debt remained stable during the period.", "Company costs remained stable during the period."]
    unit, candidates = fixture(texts)
    built = build_reasons(unit, {"reasons": [item("R1"),
        item("R2", mechanism="COSTS \u200b remain stable.", implication="THE forecast uses stable costs."),
        item("R3"), item("R4")]}, candidates)
    assert len(built) == 3 and [r["premise"] for r in built] == [texts[i] for i in (0, 2, 3)]
    assert [r["reason_id"] for r in built] == ["r1", "r2", "r3"]
    for reason, index in zip(built, (0, 2, 3)):
        assert reason["citations"] == [{"doc_id": f"D{index}", "span_start": 0, "span_end": len(texts[index])}]


def test_evidence_URI_masking_uses_three_UTF8_bytes_per_character():
    text = "Source https://example.test/quoted, remains stable."
    uri = "https://example.test/quoted"
    expected = text.replace(uri, "█" * len(uri))
    assert mask_evidence_uris(text) == expected
    assert len(expected.encode()) == len(text.encode()) + 2 * len(uri)
    assert mask_evidence_uris("www.example.test") == "www.example.test"  # ASCII scheme mask only


def test_masked_evidence_cap_prunes_support_and_retains_distinct_reason():
    texts = [f"Company division {i} revenue remained stable. https://example.test/" + "a" * 1400
             for i in range(12)]
    unit, candidates = fixture(texts)
    for i, (_, candidate) in enumerate(candidates):
        candidate["claim"] = texts[i].split(" https:")[0]
        candidate["span_end"] = len(candidate["claim"])
    built = build_reasons(unit, {"reasons": [item(f"R{start}",
        support=[f"R{start+j}" for j in (1, 2, 3)]) for start in (1, 5, 9)]}, candidates)
    assert len(built) == 3 and len(built[2]["citations"]) < 4
    resolved = {(c["doc_id"], c["span_start"], c["span_end"]):
                unit.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]]
                for r in built for c in r["citations"]}
    assert _within_caps(built, resolved)
    evidence = [{"reason_id": r["reason_id"], **c,
                 "trusted_text": mask_evidence_uris(resolved[(c["doc_id"], c["span_start"], c["span_end"])])}
                for r in built for c in r["citations"]]
    assert len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()) <= 40000
    assert all(r["premise"] == candidates[(0, 4, 8)[i]][1]["claim"] for i, r in enumerate(built))
