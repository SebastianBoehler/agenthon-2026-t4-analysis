"""Invented controls for quotation containment and original-document offsets."""
import json
import unicodedata

import pytest

from baselines.vector_agent.claims import _enclosing, entity_claims, locate
from baselines.vector_agent.parse import parse_predictions
from baselines.vector_agent.unit import Doc, Unit

FACT = "Revenue rose 7 percent while diluted earnings reached 2 dollars per share."


def fixture(text, doc_id="invented", admission=None):
    entity = {"entity_id": "A", "name": "Invented Company"}
    task = {"task_id": "invented", "schema_version": "3", "cutoff_date": "2020-01-02",
            "target": {"type": "regression", "name": "earnings"},
            "entities": [entity], "prompt": "Forecast the invented earnings quantity."}
    doc = Doc(doc_id, "2020-01-01", "Invented source", text, admission)
    return Unit(task, {doc_id: doc}), entity, doc


def samples(quote, doc_id="invented"):
    raw = json.dumps({"entity_id": "A", "point_forecast": 7, "lo": 6, "hi": 8,
                      "evidence": [{"doc_id": doc_id, "quote": quote}]})
    return parse_predictions(raw, target_type="regression", labels=[], entity_ids={"A"},
                             weight=1, source="invented-control")


def folded(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


@pytest.mark.parametrize("boundary", ["word-limit", "newline"])
def test_enclosing_never_discards_the_end_of_a_selected_quote(boundary):
    prefix = "ordinary commentary " * 16
    quote = FACT if boundary == "word-limit" else FACT.replace("while ", "while\n")
    unit, entity, doc = fixture(prefix + quote + " Additional invented commentary continues.")
    hit = locate(doc, quote)
    assert hit is not None
    enclosing = _enclosing(doc, *hit)
    assert enclosing[0] <= hit[0] and enclosing[1] >= hit[1]
    selected = entity_claims(unit, entity, samples(quote), "")
    assert folded(quote) in folded(selected[0]["claim"])
    for claim in selected:
        assert claim["claim"] == doc.text[claim["span_start"]:claim["span_end"]]
        assert 25 <= len(claim["claim"]) <= 399


@pytest.mark.parametrize("prefix,suffix", [("ﬁ ", " e\u0301"), ("e\u0301 ", " ﬁ")])
def test_normalized_offsets_never_point_to_a_different_original_slice(prefix, suffix):
    _, _, doc = fixture(prefix + FACT.replace("7", "７") + suffix)
    assert len(unicodedata.normalize("NFKC", doc.text)) == len(doc.text)
    hit = locate(doc, FACT)
    # Unsupported typography can be rejected; an accepted hit must locate the quotation.
    assert hit is None or folded(doc.text[hit[0]:hit[1]]) == folded(FACT)


@pytest.mark.parametrize("text,quote", [
    (FACT, FACT),
    ("Prefix: " + FACT + " Suffix.", FACT),
    (FACT.replace(" ", "\n"), FACT),
    (FACT.replace(" ", "\t"), FACT),
    (FACT.replace("7", "７"), FACT),
    (FACT.replace(" ", "\u00a0"), FACT),
    (FACT, '"' + FACT + '"'),
])
def test_supported_literal_whitespace_and_typography_offsets(text, quote):
    _, _, doc = fixture(text)
    hit = locate(doc, quote)
    assert hit is not None
    assert folded(doc.text[hit[0]:hit[1]]) == folded(quote.strip('"'))


@pytest.mark.parametrize("quote", ["only three words", "An absent invented quotation has 99 dollars."])
def test_short_and_absent_quotations_remain_rejected(quote):
    _, _, doc = fixture(FACT)
    assert locate(doc, quote) is None


def test_short_segment_crossing_keeps_the_complete_quote():
    quote = "Revenue rose 7 percent\nwhile diluted earnings reached 2 dollars per share."
    unit, entity, doc = fixture(quote)
    hit = locate(doc, quote)
    assert _enclosing(doc, *hit) == hit
    selected = entity_claims(unit, entity, samples(quote), "")
    assert selected[0]["claim"] == quote


def test_single_segment_context_is_preserved():
    text = "Quarterly invented summary, " + FACT
    unit, entity, _ = fixture(text)
    assert entity_claims(unit, entity, samples(FACT), "")[0]["claim"] == text


def test_unsupported_length_changing_normalization_stays_rejected():
    _, _, doc = fixture("ﬁ " + FACT.replace("7", "７"))
    assert locate(doc, FACT) is None


def test_explicit_admitted_document_is_preferred():
    unit, entity, _ = fixture("Prior-quarter accounting heading\n" + FACT, "old")
    unit.docs["current"] = Doc("current", "2020-01-01", "Invented current source",
                               "Current-quarter accounting heading\n" + FACT, frozenset({"A"}))
    claim = entity_claims(unit, entity, samples(FACT, "current"), "")[0]
    assert claim["doc_id"] == "current"


def test_foreign_entity_source_is_not_admitted():
    unit, entity, _ = fixture(FACT, "foreign", frozenset({"B"}))
    claim = entity_claims(unit, entity, samples(FACT, "foreign"), "")[0]
    assert claim["doc_id"] == "task"


def test_sentence_boundary_keeps_the_selected_sentence():
    unit, entity, _ = fixture("Quarterly invented summary: " + FACT)
    assert entity_claims(unit, entity, samples(FACT), "")[0]["claim"] == FACT
