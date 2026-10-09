"""Synthetic controls: entity-name headings do not displace factual claim passages."""
from baselines.vector_agent.claims import claim_is_entity_name, entity_claims
from baselines.vector_agent.parse import Sample
from baselines.vector_agent.unit import Doc, Unit

NAME = "Synthetic Regional Retail and Food Services Control Group"
FACT = "The latest reported sales were 120 million dollars, rising 3 percent during June."


def fixture(text):
    entity = {"entity_id": "SYN-A", "name": NAME, "latest_published_value": 120}
    task = {"target": {"type": "regression"}, "entities": [entity], "prompt": "Forecast retail sales."}
    doc = Doc("SYN-DOC", "2026-01-01", "Synthetic table", text, None)
    return Unit(task, {doc.doc_id: doc}), entity


def test_name_heading_does_not_displace_available_fact():
    unit, entity = fixture(NAME + "\n\n" + FACT)
    selected = entity_claims(unit, entity, [], "")
    assert selected
    assert all(c["claim"] != NAME for c in selected)
    assert any(c["claim"] == FACT for c in selected)
    for claim in selected:
        assert claim["claim"] == unit.docs[claim["doc_id"]].text[claim["span_start"]:claim["span_end"]]


def test_sample_heading_quote_is_rejected_too():
    unit, entity = fixture(NAME + "\n\n" + FACT)
    sample = Sample("SYN-A", 120, 100, 140, None, None, None, "", quotes=[("SYN-DOC", NAME)])
    selected = entity_claims(unit, entity, [sample], "")
    assert [c["claim"] for c in selected] == [FACT]


def test_name_plus_a_factual_assertion_remains_eligible():
    text = NAME + " reported sales of 120 million dollars."
    unit, entity = fixture(text)
    assert not claim_is_entity_name(unit, text)
    assert [c["claim"] for c in entity_claims(unit, entity, [], "")] == [text]


def test_other_roster_name_and_typographic_variants_are_identifiers():
    unit, _ = fixture(FACT)
    unit.task["entities"].append({"entity_id": "SYN-B", "name": "Synthetic Overseas Retail Group"})
    assert claim_is_entity_name(unit, "  SYNTHETIC  OVERSEAS\nRETAIL GROUP. ")
    assert claim_is_entity_name(unit, "Ｓｙｎｔｈｅｔｉｃ Overseas Retail Group")
    assert not claim_is_entity_name(unit, "Synthetic Overseas Retail Group revenue rose 2 percent.")
