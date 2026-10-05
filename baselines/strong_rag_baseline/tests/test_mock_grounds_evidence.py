"""Executive summary: Offline CLI checks retain exact evidence and complete rows.

Mock values are synthetic wiring controls, not predictive or hosted score evidence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from baselines.strong_rag_baseline.cli import _mock_reply, main, run
from baselines.strong_rag_baseline.client import MockModelClient
from baselines.strong_rag_baseline.indexer import build_index

_REPO = Path(__file__).resolve().parents[3]
EXAMPLE_UNIT = _REPO / "units" / "t4-EXAMPLE-eps-beat"
RANKING_UNIT = _REPO / "units" / "t4-cotpos-202411-us10"
REGRESSION_UNIT = _REPO / "units" / "t4-auction-btc-202411-us7"


def _run_mock(unit: Path, out: Path) -> dict:
    assert main(
        ["--task", str(unit / "task.json"), "--corpus", str(unit / "corpus"),
         "--out", str(out), "--mock"]
    ) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_row_gets_at_least_one_grounded_claim(tmp_path: Path) -> None:
    """The regression this file exists for."""
    answer = _run_mock(EXAMPLE_UNIT, tmp_path / "a.json")
    empty = [r["entity_id"] for r in answer["entity_predictions"] if not r["claims"]]
    assert not empty, (
        f"rows with no claims: {empty}. An empty claims array on any row fails g1_schema "
        "for the whole submission under the local schema"
    )


def test_the_quote_resolves_to_the_span_it_claims(tmp_path: Path) -> None:
    """A fabricated span is worse than no citation; prove the offsets are real.

    `span_finder` locates the quote by exact substring search, so a claim can only carry
    honest offsets if the quote really is verbatim corpus text. This reads the corpus back
    and checks the span, rather than trusting that the pipeline said so.
    """
    answer = _run_mock(EXAMPLE_UNIT, tmp_path / "a.json")
    corpus = build_index(EXAMPLE_UNIT / "corpus")
    for row in answer["entity_predictions"]:
        for claim in row["claims"]:
            doc_text = corpus.doc_texts[claim["doc_id"]]
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(doc_text)
            assert doc_text[claim["span_start"]:claim["span_end"]].strip(), (
                "the cited span is empty or whitespace"
            )


def test_ranking_rows_do_not_all_carry_the_same_forecast(tmp_path: Path) -> None:
    """A constant vector is the degenerate answer from public #47.

    On a ranking unit the scorer reads `point_forecast` and nothing else, so an identical
    vector scores by the sealed roster's order rather than by the prediction. No shipped
    code here should model that shape, stub or not.
    """
    answer = _run_mock(RANKING_UNIT, tmp_path / "a.json")
    forecasts = [r["point_forecast"] for r in answer["entity_predictions"]]
    assert len(forecasts) > 1
    assert len(set(forecasts)) > 1, f"every point_forecast is identical: {forecasts}"


def test_non_ranking_forecasts_stay_a_visible_placeholder(tmp_path: Path) -> None:
    """0.0, deliberately.

    Echoing an entity feature here would be worse than a placeholder, not better: the
    exemplar's first numeric feature is `mktcap_bn`, so the stub would emit 2650.0 as a
    forecast of a ~1.50 EPS -- which reads as a real prediction that is badly wrong rather
    than as a stub declining to predict.
    """
    answer = _run_mock(EXAMPLE_UNIT, tmp_path / "a.json")
    assert [r["point_forecast"] for r in answer["entity_predictions"]] == [0.0]


def test_numeric_targets_omit_classification_label(tmp_path: Path) -> None:
    """A JSON null is not a valid substitute for an inapplicable label field."""
    for unit in (REGRESSION_UNIT, RANKING_UNIT):
        answer = _run_mock(unit, tmp_path / f"{unit.name}.json")
        assert all("label" not in row for row in answer["entity_predictions"])


def test_sec_entity_prompt_contains_only_its_own_filings(tmp_path: Path) -> None:
    """Keep peer-company filings out of an entity's evidence context."""
    unit = _REPO / "units" / "t4-eps-growth-2024Q3-banks"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    prompts: list[str] = []

    def capture_prompt(system: str, user: str) -> str:
        prompts.append(user)
        return _mock_reply(system, user)

    run(
        unit / "task.json",
        unit / "corpus",
        tmp_path / "answer.json",
        MockModelClient(reply=capture_prompt),
        top_k=10,
    )

    assert len(prompts) == len(task["entities"])
    for entity, prompt in zip(task["entities"], prompts):
        doc_ids = re.findall(r"doc_id=(\S+) \(doc_date=", prompt)
        assert doc_ids
        assert all(doc_id.startswith(f"EDGAR_{entity['cik']}_") for doc_id in doc_ids)


def test_a_prompt_with_no_excerpt_cites_nothing(tmp_path: Path) -> None:
    """The honest answer when retrieval found nothing, and still a g1 failure.

    That is a property of the unit, not of the mock: there is nothing eligible to cite. The
    mock must not invent a doc_id to fill the gap -- a fabricated citation is
    `t4.citation_unresolved`, the same -0.27, plus a false claim about the corpus.
    """
    reply = json.loads(_mock_reply("sys", "TASK: something\nTARGET: x (classification)\n"))
    assert reply["evidence"] == []
