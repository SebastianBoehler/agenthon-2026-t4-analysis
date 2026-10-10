"""Wiring, fault tolerance and deadline tests with a synthetic House stand-in (no forecasting claims)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from baselines.vector_agent.agent import Schedule
from baselines.vector_agent.cli import run, validate
from baselines.vector_agent.house import Budget
from baselines.vector_agent.parse import extract_json, match_label, number, parse_predictions
from baselines.vector_agent.unit import load_unit

from .fake_chat import FakeChat

ROOT = Path(__file__).resolve().parents[3]
UNITS = sorted(p for p in (ROOT / "units").iterdir() if (p / "task.json").exists())


class BudgetedChat(FakeChat):
    def __init__(self, budget: Budget, **kw):
        super().__init__(**kw)
        self.budget = budget

    def complete(self, system, user, **kw):
        self.budget.take()
        return super().complete(system, user, **kw)


def _run(unit: Path, tmp_path: Path, **kw):
    budget = Budget()
    chat = BudgetedChat(budget, **kw)
    out = tmp_path / unit.name / "answer.json"
    answer = run(unit / "task.json", unit / "corpus", out, chat, budget, Schedule(), time.monotonic(),
                 log=lambda *a: None)
    return answer, chat, budget, out


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_every_unit_yields_a_valid_answer(unit, tmp_path):
    answer, chat, budget, out = _run(unit, tmp_path)
    loaded = load_unit(unit / "task.json", unit / "corpus")
    assert validate(loaded, answer) == []
    assert json.loads(out.read_text(encoding="utf-8")) == answer
    assert budget.used <= 24
    assert answer["target_type"] == loaded.target_type
    for row in answer["entity_predictions"]:
        assert row["interval"]["level"] == 0.9
        assert 1 <= len(row["claims"]) <= 3
        for c in row["claims"]:
            assert "citations" not in c and len(c["claim"]) <= 399
    for reason in answer.get("submitted_reasons", []):
        cite = reason["citations"][0]
        cited = loaded.docs[cite["doc_id"]].text[cite["span_start"]:cite["span_end"]]
        assert reason["premise"] in cited and len(cited) <= 1500 + len(reason["premise"])


@pytest.mark.parametrize("unit_name", ["t4-macrorev-20240930-us6", "t4-credit-event-2023"])
def test_faulty_replies_are_absorbed(unit_name, tmp_path):
    faults = {0: "error", 1: "garbage", 2: "drop_one", 3: "fenced", 4: "think", 5: "case", 6: "error"}
    answer, chat, budget, _ = _run(ROOT / "units" / unit_name, tmp_path, faults=faults)
    assert len(answer["entity_predictions"]) == len(load_unit(
        ROOT / "units" / unit_name / "task.json", ROOT / "units" / unit_name / "corpus").entities)
    assert budget.used <= 24


def test_deadline_stops_waiting_and_writes_a_degraded_answer(tmp_path):
    unit = ROOT / "units" / "t4-fomc-curve-20240918"
    budget = Budget()
    chat = BudgetedChat(budget, faults={i: "slow" for i in range(30)})
    schedule = Schedule(launch_stop=0.5, hard_stop=1.0, reasons_latest=1.0, reasons_stop=1.5,
                        request_timeout=0.6)
    start = time.monotonic()
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "a.json", chat, budget, schedule, start,
                 log=lambda *a: None)
    assert time.monotonic() - start < 3.0
    assert validate(load_unit(unit / "task.json", unit / "corpus"), answer) == []
    assert "degraded" in answer["evidence_trace"] and "submitted_reasons" not in answer
    assert all(r["point_forecast"] == 0.0 for r in answer["entity_predictions"])  # no change carried forward


@pytest.mark.parametrize("unit_name", [p.name for p in UNITS])
def test_without_any_house_every_unit_still_gets_a_valid_answer(unit_name, tmp_path):
    unit = ROOT / "units" / unit_name
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "a.json", None, Budget(), Schedule(),
                 time.monotonic(), log=lambda *a: None)
    assert validate(load_unit(unit / "task.json", unit / "corpus"), answer) == []
    assert (tmp_path / "a.json").exists() and "degraded" in answer["evidence_trace"]


def test_parser_tolerates_common_reply_shapes():
    raw = 'Sure!\n```json\n{"predictions": [{"entity_id": "A", "point_forecast": +0.30, "lo": "-0.1%", "hi": 0.7,}]}\n```'
    got = parse_predictions(raw, target_type="regression", labels=[], entity_ids={"A"}, weight=1, source="t")
    assert len(got) == 1 and got[0].point == pytest.approx(0.30) and got[0].lo == pytest.approx(-0.1)
    assert extract_json("no json here", ("predictions",)) is None
    assert number("1,234.5 bps") == pytest.approx(1234.5)
    assert match_label("No Event", ["credit_event", "no_event"]) == "no_event"
    assert match_label("Positive reaction", ["positive_reaction", "negative_reaction", "flat"]) == "positive_reaction"
    assert match_label("maybe", ["up", "down"]) is None


def test_off_vocabulary_labels_are_not_guessed():
    raw = json.dumps({"entity_id": "A", "label": "sideways", "lo": 0, "hi": 1})
    assert parse_predictions(raw, target_type="classification", labels=["up", "down"],
                             entity_ids={"A"}, weight=1, source="t") == []


def test_truncated_unit_reply_keeps_complete_rows():
    rows = [{"entity_id": e, "point_forecast": 1.0, "lo": 0.0, "hi": 2.0, "evidence": [{"doc_id": "d", "quote": "x"}]}
            for e in ("A", "B", "C")]
    raw = json.dumps({"predictions": rows})[:-60]  # cut inside the last row
    got = parse_predictions(raw, target_type="regression", labels=[], entity_ids={"A", "B", "C"}, weight=1, source="t")
    assert [s.entity_id for s in got] == ["A", "B"]


def _sample(eid, point, lo, hi, label=None, sigma=None):
    from baselines.vector_agent.parse import Sample
    return Sample(eid, point, lo, hi, label, None, None, "", sigma=sigma)


def test_probability_bands_are_clipped_and_outliers_leave_the_band():
    from baselines.vector_agent.aggregate import combine
    fc = combine([_sample("A", 0.9, 0.7, 1.1, "x"), _sample("A", 0.85, 0.6, 1.05, "x")], "classification",
                 unit_interval=True)
    assert 0.0 <= fc.lo <= fc.point <= fc.hi <= 1.0
    fc = combine([_sample("A", 1.0, 0.0, 2.0), _sample("A", 1.2, 0.5, 2.5), _sample("A", 1.5, 0.2, 2.8),
                  _sample("A", 30.0, 20.0, 40.0)], "regression")
    assert fc.point == pytest.approx(1.35) and fc.hi < 5


def test_model_sigma_widens_a_narrower_stated_band():
    raw = json.dumps({"entity_id": "A", "point_forecast": 0.0, "sigma": 10, "lo": -1, "hi": 1})
    s = parse_predictions(raw, target_type="regression", labels=[], entity_ids={"A"}, weight=1, source="t")[0]
    assert s.lo == pytest.approx(-16.45, abs=0.01) and s.hi == pytest.approx(16.45, abs=0.01)


def test_spare_budget_goes_to_disagreeing_entities():
    from baselines.vector_agent.agent import Planner
    unit = load_unit(ROOT / "units" / "t4-postearn-20240201-megacap" / "task.json",
                     ROOT / "units" / "t4-postearn-20240201-megacap" / "corpus")
    plan = Planner(unit, 1)
    for i, eid in enumerate(plan.ids):
        plan.attempts[f"g{i}"] = plan.done[f"g{i}"] = plan.target
        plan.samples[eid] = [_sample(eid, 0.5, -1, 2, "flat") for _ in range(plan.target)]
    plan.attempts["unit"] = plan.done["unit"] = 1
    assert plan.next_job(budget_left=10) is None  # unanimous: no extra calls
    plan.samples["META"][0].label = "positive_reaction"
    job = plan.next_job(budget_left=10)
    assert job is not None and job.ids == ("META",)
    assert plan.next_job(budget_left=2) is None  # reasons reserve plus one spare are kept


def test_one_runaway_width_does_not_set_the_band():
    from baselines.vector_agent.aggregate import combine
    ss = [_sample("A", 710500, 709000, 712000) for _ in range(4)] + [_sample("A", 710800, 450000, 960000)]
    fc = combine(ss, "regression")
    assert 700000 < fc.lo and fc.hi < 720000


def test_reason_premise_key_with_pasted_text_is_resolved():
    from baselines.vector_agent.claims import build_reasons
    unit = load_unit(ROOT / "units" / "t4-fomc-curve-20240918" / "task.json",
                     ROOT / "units" / "t4-fomc-curve-20240918" / "corpus")
    doc = unit.docs["RATES_SNAPSHOT_20240919"]
    cand = {"doc_id": doc.doc_id, "span_start": 0, "span_end": 60, "claim": doc.text[:60]}
    reply = {"reasons": [{"premise": "[R1] " + doc.text[:60], "mechanism": "m", "answer_implication": "i UST2Y",
                          "entities": ["UST2Y"]}]}
    got = build_reasons(unit, reply, [("R1", cand)])
    assert len(got) == 1 and got[0]["premise"] == doc.text[:60]


def _synthetic_unit(tmp_path: Path, source: str, rows: int) -> Path:
    """A copy of a public unit with ``rows`` synthetic roster rows and every document shared."""
    import shutil
    src = ROOT / "units" / source
    dst = tmp_path / f"{source}-x{rows}"
    shutil.copytree(src, dst)
    task = json.loads((dst / "task.json").read_text())
    base = task["entities"]
    task["entities"] = [dict(base[i % len(base)], entity_id=f"{base[i % len(base)]['entity_id']}_{i}")
                        for i in range(rows)]
    (dst / "task.json").write_text(json.dumps(task))
    for path in (dst / "corpus" / "manifest.json", dst / "manifest.json"):
        manifest = json.loads(path.read_text())
        for f in manifest["files"]:
            if f.get("role") == "corpus":
                f.pop("entity_ids", None)
                f["shared"] = True
        path.write_text(json.dumps(manifest))
    return dst


def test_passage_index_is_not_shared_by_two_texts_under_one_doc_id(tmp_path):
    import shutil
    from baselines.vector_agent.evidence import entity_passages
    src = ROOT / "units" / "t4-eps-yoy-2023Q2-mixed"
    dst = tmp_path / "excerpt"
    shutil.copytree(src, dst)
    for path in (dst / "corpus").glob("EDGAR_*.json"):  # same doc ids, shorter excerpts
        doc = json.loads(path.read_text())
        doc["text"] = doc["text"][: len(doc["text"]) // 3]
        path.write_text(json.dumps(doc))
    full = load_unit(src / "task.json", src / "corpus")
    short = load_unit(dst / "task.json", dst / "corpus")
    for unit in (full, short):  # one process, as when a harness runs several units
        for entity in unit.entities:
            for p in entity_passages(unit, entity, 44_000):
                assert p.text == unit.docs[p.doc_id].text[p.start:p.end]


@pytest.mark.parametrize("source", ["t4-cpicomp-202410-us11", "t4-postearn-20240201-megacap"])
def test_thirty_row_rosters_fit_the_request_budget(source, tmp_path):
    unit = _synthetic_unit(tmp_path, source, 30)
    answer, chat, budget, _ = _run(unit, tmp_path)
    assert len(answer["entity_predictions"]) == 30
    assert budget.used <= 24


def test_table_summary_reports_spreads_and_revisions():
    from baselines.vector_agent.tables import summarize
    text = ("x\nmonth | as_of_2024-01-01 | as_of_2024-02-01 | as_of_2024-03-01\n"
            "2023-10 | 100 | 101 | 101\n2023-11 | 102 | 103 | 104\n2023-12 | -- | 105 | 104\n"
            "2024-01 | -- | -- | 107\nNOTES")
    out = summarize(text)
    assert "revisions between adjacent dated columns" in out and "n=4" in out
    assert summarize("no tables here") == ""


def test_band_edge_far_beyond_own_sigma_is_repaired():
    raw = json.dumps({"entity_id": "A", "point_forecast": 289300, "sigma": 300, "lo": 28492, "hi": 290100})
    s = parse_predictions(raw, target_type="regression", labels=[], entity_ids={"A"}, weight=1, source="t")[0]
    assert s.lo == pytest.approx(289300 - 1.6448536269514722 * 300) and s.hi == 290100


def test_growth_is_recomputed_from_level_and_table_base():
    from baselines.vector_agent.consistency import ground_growth, percent_growth_target
    task = json.loads((ROOT / "units" / "t4-eps-growth-2024Q3-banks" / "task.json").read_text())
    assert percent_growth_target(task)
    row = next(e for e in task["entities"] if e["entity_id"] == "GS")
    s = _sample("GS", 150.0, 100.0, 200.0)
    s.level, s.base = 8.5, 5.47
    ground_growth(s, row)
    assert s.point == pytest.approx((8.5 / 5.47 - 1) * 100) and s.hi - s.lo == pytest.approx(100.0)
    t = _sample("GS", 150.0, 100.0, 200.0)
    t.level, t.base = 8.5, 3.08  # a base from the wrong period: the row's only number is used
    assert ground_growth(t, row).point == pytest.approx((8.5 / 5.47 - 1) * 100)
    u = _sample("GS", 150.0, 100.0, 200.0)
    u.level, u.base = 8.5, 3.08
    assert ground_growth(u, dict(row, other_number=1.0)).point == 150.0  # ambiguous: left alone
    other = json.loads((ROOT / "units" / "t4-fomc-curve-20240918" / "task.json").read_text())
    assert not percent_growth_target(other)


def test_probability_of_a_binary_event_gets_the_score_optimal_band():
    from baselines.vector_agent.aggregate import binary_outcome_band, combine
    assert binary_outcome_band(0.3) == (0.0, 1.0)
    assert binary_outcome_band(0.02) == (0.0, 0.02)
    assert binary_outcome_band(0.97) == (0.97, 1.0)
    fc = combine([_sample("A", 0.3, 0.1, 0.5, "no_event")] * 3, "classification", unit_interval=True)
    assert (fc.lo, fc.hi) == (0.0, 1.0)
    fc = combine([_sample("A", 0.01, 0.0, 0.03, "no_event")] * 3, "classification", unit_interval=True)
    assert fc.lo == 0.0 and fc.hi == pytest.approx(0.01)


def test_driver_reasons_cite_support_and_drop_key_references():
    from baselines.vector_agent.claims import build_reasons
    unit = load_unit(ROOT / "units" / "t4-fomc-curve-20240918" / "task.json",
                     ROOT / "units" / "t4-fomc-curve-20240918" / "corpus")
    doc = unit.docs["RATES_SNAPSHOT_20240919"]
    c1 = {"doc_id": doc.doc_id, "span_start": 0, "span_end": 60, "claim": doc.text[:60]}
    c2 = {"doc_id": doc.doc_id, "span_start": 100, "span_end": 160, "claim": doc.text[100:160]}
    reply = {"reasons": [{"driver": "d", "premise": "R1", "support": ["R2", "R9"], "mechanism": "m (R2)",
                          "answer_implication": "i UST2Y", "entities": ["UST2Y"]}]}
    got = build_reasons(unit, reply, [("R1", c1), ("R2", c2)])
    assert len(got) == 1 and got[0]["mechanism"] == "m"
    cited = [doc.text[c["span_start"]:c["span_end"]] for c in got[0]["citations"]]
    assert any(c1["claim"] in t for t in cited) and any(c2["claim"] in t for t in cited)  # same paragraph: one citation


def test_rejected_prompt_is_retried_with_less_evidence():
    import io
    import urllib.error
    from baselines.vector_agent.agent import Job, _call
    from baselines.vector_agent.prompts import shrink_evidence

    class Picky:
        def __init__(self, limit):
            self.lengths = []
            self.limit = limit

        def complete(self, system, user, **kw):
            self.lengths.append(len(user))
            if len(user) > self.limit:
                raise urllib.error.HTTPError("u", 400, "context too long", {}, io.BytesIO(b""))
            return "ok"

    unit = load_unit(ROOT / "units" / "t4-cotpos-202411-us10" / "task.json",
                     ROOT / "units" / "t4-cotpos-202411-us10" / "corpus")
    from baselines.vector_agent import prompts
    text = prompts.unit_prompt(unit, prompts.unit_docs_full(unit))
    assert "How to forecast:" in shrink_evidence(text, 0.5) and len(shrink_evidence(text, 0.5)) < len(text)
    chat = Picky(int(0.4 * len(text)))  # the half-evidence retry is still too long, the quarter fits
    assert _call(chat, Job("unit", "g0", tuple(unit.entity_ids), text, 0.0, 1, 1000, 1.0), 60) == "ok"
    assert len(chat.lengths) >= 2 and chat.lengths[-1] < chat.lengths[0]


def test_pure_label_reply_without_numbers_is_kept():
    raw = json.dumps({"entity_id": "A", "label": "raise", "probability": 0.7, "point_forecast": None,
                      "lo": None, "hi": None})
    got = parse_predictions(raw, target_type="classification", labels=["raise", "cut", "hold"],
                            entity_ids={"A"}, weight=1, source="t")
    assert len(got) == 1 and (got[0].lo, got[0].hi, got[0].point) == (0.0, 1.0, 0.7)


@pytest.mark.parametrize("unit_name", ["t4-macrorev-20240930-us6", "t4-credit-event-2023"])
def test_classification_unit_completes_when_replies_carry_no_numbers(unit_name, tmp_path):
    answer, _, budget, _ = _run(ROOT / "units" / unit_name, tmp_path, no_numbers=True)
    rows = answer["entity_predictions"]
    assert rows and all(r["interval"]["lo"] == 0.0 and r["interval"]["hi"] == 1.0 for r in rows)
    assert budget.used <= 24


def test_spans_only_documents_are_read_but_never_cited(tmp_path):
    import shutil
    src = ROOT / "units" / "t4-auction-btc-202411-us7"
    dst = tmp_path / "u"
    shutil.copytree(src, dst)
    for path in (dst / "corpus").glob("TDIRECT_*.json"):
        doc = json.loads(path.read_text())
        doc["spans"] = [{"text": doc.pop("text")}]
        path.write_text(json.dumps(doc))  # digests now differ; admission reads labels only
    answer, _, _, _ = _run(dst, tmp_path)
    for row in answer["entity_predictions"]:
        assert all(c["doc_id"] == "task" for c in row["claims"])


def test_event_labels_follow_the_cohort_count():
    from baselines.vector_agent.aggregate import Forecast
    from baselines.vector_agent.cohort import event_labels, relabel
    assert event_labels(["credit_event", "no_event"]) == ("credit_event", "no_event")
    assert event_labels(["up", "down"]) is None and event_labels(["beat", "miss", "inline"]) is None
    probs = {"A": 0.85, "B": 0.35, "C": 0.30, "D": 0.05}
    results = {e: Forecast(e, "no_event", p, 0.0, 1.0, "", 3) for e, p in probs.items()}
    samples = {e: [_sample(e, p, 0.0, 1.0, "no_event")] for e, p in probs.items()}
    relabel(results, samples, "credit_event", "no_event", [3, 2, 3], [["A", "C", "B", "D"]], True)
    assert [results[e].label for e in "ABCD"] == ["credit_event", "credit_event", "credit_event", "no_event"]
    relabel(results, samples, "credit_event", "no_event", [0, 0, 1], [], True)
    assert [results[e].label for e in "ABCD"] == ["credit_event", "no_event", "no_event", "no_event"]


def test_credit_unit_runs_cohort_calls_within_budget(tmp_path):
    answer, chat, budget, _ = _run(ROOT / "units" / "t4-credit-event-2023", tmp_path)
    assert sum("COHORT QUESTION" in p for p in chat.prompts) == 3
    assert budget.used <= 24


def test_rate_limit_storm_is_bounded(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from baselines.vector_agent.house import HouseClient, HouseConfig

    calls = []

    def refuse(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "0"}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr("time.sleep", lambda s: None)
    budget = Budget()
    client = HouseClient(HouseConfig("http://h", "m", "t"), budget)
    with pytest.raises(urllib.error.HTTPError):
        client.complete("s", "u", temperature=0, seed=1, max_tokens=10, timeout=100)
    assert len(calls) == HouseClient.retries_429 + 1 and budget.used == len(calls)
    assert client.rate_limited == len(calls)


def test_reply_shapes_that_used_to_lose_an_entity():
    lab = ["inline", "beat", "miss"]
    assert match_label("in-line", lab) == "inline" and match_label("Beat (by 7%)", lab) == "beat"
    assert match_label("up.", ["up", "down"]) == "up" and match_label("maybe", ["up", "down"]) is None
    raw = json.dumps({"predictions": [{"entity_id": 7203, "point_forecast": 1, "lo": 0, "hi": 2},
                                      {"entity_id": "brk-b", "point_forecast": None, "interval": [1, 3]}]})
    got = parse_predictions(raw, target_type="regression", labels=[], entity_ids={"7203", "BRK.B"},
                            weight=1, source="t")
    assert {s.entity_id for s in got} == {"7203", "BRK.B"}
    assert next(s for s in got if s.entity_id == "BRK.B").point == 2
    single = json.dumps({"entity_id": "A ", "point_forecast": 1, "lo": 0, "hi": 2})
    assert parse_predictions(single, target_type="regression", labels=[], entity_ids={"a"},
                             weight=1, source="t")[0].entity_id == "a"
    keyed = json.dumps({"predictions": {"X": {"point_forecast": 1, "lo": 0, "hi": 2, "evidence": 3}}})
    assert parse_predictions(keyed, target_type="regression", labels=[], entity_ids={"X"},
                             weight=1, source="t")[0].entity_id == "X"


def test_hardening_from_the_adversarial_review(tmp_path):
    from baselines.vector_agent.cli import PINNED_SEED
    from baselines.vector_agent.consistency import ground_growth
    from baselines.vector_agent.prompts import shrink_evidence
    assert 0 <= PINNED_SEED < 2**31  # request seeds are a code constant, not QFBENCH_SEED
    s = _sample("A", 5.0, 1.0, 9.0)
    s.level, s.base = 1.3, None
    assert ground_growth(s, {"entity_id": "A", "trailing_growth_pct": 0.2}).point == 5.0
    cohort_like = "TASK\n\nEVIDENCE DOCUMENTS (x)\n\n=== D1 | h ===\n" + "a" * 5000 + "\n\nCOHORT QUESTION\nq"
    assert len(shrink_evidence(cohort_like, 0.25)) < len(cohort_like)
    unit = load_unit(ROOT / "units" / "t4-credit-event-2023" / "task.json", ROOT / "units" / "t4-credit-event-2023" / "corpus")
    unit.task["target"] = dict(unit.task["target"], labels=None)
    assert unit.labels == ["credit_event", "no_event"]
    unit.task["interval_level"] = "90%"
    assert unit.interval_level == 0.9


def test_reasons_reserve_goes_to_an_entity_without_a_forecast(tmp_path):
    unit = ROOT / "units" / "t4-postearn-20240201-megacap"
    budget = Budget(cap=3)
    chat = BudgetedChat(budget)
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "a.json", chat, budget, Schedule(),
                 time.monotonic(), log=lambda *a: None)
    assert len(answer["entity_predictions"]) == 3 and budget.used == 3


def test_reason_candidates_start_with_the_drivers_the_task_names():
    from baselines.vector_agent import prompts
    from baselines.vector_agent.claims import reason_candidates
    from baselines.vector_agent.drivers import task_drivers
    assert task_drivers("Reason from pre-cutoff drivers - net interest income trends, credit provisions, "
                        "and fees. Do not use the corpus.") == ["net interest income trends", "credit provisions", "fees"]
    assert task_drivers("Forecast each row.") == []
    unit = load_unit(ROOT / "units" / "t4-cpicomp-202410-us11" / "task.json",
                     ROOT / "units" / "t4-cpicomp-202410-us11" / "corpus")
    named = task_drivers(unit.task["prompt"])
    got = reason_candidates(unit, {}, named, {})
    tagged = [c for _, c in got if c.get("driver")]
    assert tagged and got[0][1].get("driver") in named
    for _, c in got:
        assert unit.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]] == c["claim"]
    text = prompts.reasons_prompt(unit, [], [("R1", "x", "d", "applies to all entities")], named)
    assert "DRIVERS THE TASK NAMES: " + named[0] in text


def test_bps_change_bands_are_at_least_the_tables_horizon_spread(tmp_path):
    from baselines.vector_agent import prompts
    from baselines.vector_agent.consistency import change_sd_floor
    unit_dir = ROOT / "units" / "t4-fomc-curve-20240918"
    unit = load_unit(unit_dir / "task.json", unit_dir / "corpus")
    floors = change_sd_floor(unit.task, unit.docs.values(), prompts.horizon_days(unit))
    assert set(floors) == set(unit.entity_ids) and all(10 < v < 60 for v in floors.values())
    stats = prompts.with_stats(unit.docs["RATES_SNAPSHOT_20240919"].text, prompts.horizon_days(unit), True)
    assert "bps" in stats.split("[COMPUTED")[1]
    answer, *_ = _run(unit_dir, tmp_path)
    for r in answer["entity_predictions"]:
        half = 1.6448 * floors[r["entity_id"]]
        assert r["interval"]["lo"] <= r["point_forecast"] - half and r["interval"]["hi"] >= r["point_forecast"] + half
    other = ROOT / "units" / "t4-auction-btc-202411-us7"
    unit = load_unit(other / "task.json", other / "corpus")
    assert change_sd_floor(unit.task, unit.docs.values(), prompts.horizon_days(unit)) == {}


def test_reused_premise_falls_back_to_an_unused_support_passage():
    from baselines.vector_agent.claims import build_reasons
    unit = load_unit(ROOT / "units" / "t4-fomc-curve-20240918" / "task.json",
                     ROOT / "units" / "t4-fomc-curve-20240918" / "corpus")
    doc = unit.docs["RATES_SNAPSHOT_20240919"]
    cands = [(f"R{i + 1}", {"doc_id": doc.doc_id, "span_start": 100 * i, "span_end": 100 * i + 60,
                            "claim": doc.text[100 * i:100 * i + 60]}) for i in range(3)]
    reply = {"reasons": [
        {"premise": "R1", "support": ["R2"], "mechanism": "m1", "answer_implication": "UST2Y -5"},
        {"premise": "R1", "support": ["R1", "R3"], "mechanism": "m2", "answer_implication": "UST10Y -5"}]}
    got = build_reasons(unit, reply, cands)
    assert [r["premise"] for r in got] == [cands[0][1]["claim"], cands[2][1]["claim"]]


def test_reasons_cut_off_mid_list_are_salvaged():
    from baselines.vector_agent.parse import salvage_objects
    cut = ('{"reasons": [{"premise": "R1", "support": ["R2"], "mechanism": "m", "answer_implication": "i"}, '
           '{"premise": "R3", "mechanism": "m2", "answer_implication": "i2", "entities": ["A"]')
    assert [r["premise"] for r in salvage_objects(cut, "premise")] == ["R1", "R3"]
    assert [r["premise"] for r in salvage_objects(cut + '}, {"premise": "R4", "mech', "premise")] == ["R1", "R3"]


def test_reasons_still_run_after_the_extras_cutoff_while_time_remains(tmp_path):
    unit = ROOT / "units" / "t4-fomc-curve-20240918"
    budget = Budget()
    chat = BudgetedChat(budget)
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "answer.json", chat, budget,
                 Schedule(reasons_latest=0.0), time.monotonic(), log=lambda *a: None)
    assert answer.get("submitted_reasons")


def test_a_group_whose_calls_time_out_is_split_into_halves(tmp_path):
    class SlowOnLargeReplies(BudgetedChat):
        def complete(self, system, user, **kw):
            reply = super().complete(system, user, **kw)
            if len(json.loads(reply).get("predictions") or []) > 6:  # an 11-row reply outlasts the timeout
                raise TimeoutError("The read operation timed out")
            return reply

    unit = ROOT / "units" / "t4-cpicomp-202410-us11"
    budget = Budget()
    lines: list[str] = []
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "answer.json", SlowOnLargeReplies(budget),
                 budget, Schedule(), time.monotonic(), log=lines.append)
    assert len(answer["entity_predictions"]) == 11
    assert not any("degraded" in line for line in lines)
    assert any("split into halves" in line for line in lines)


def test_a_stalled_reasons_call_is_hedged(tmp_path):
    import threading

    class FirstReasonsStalls(BudgetedChat):
        reason_calls = 0

        def complete(self, system, user, **kw):
            if "CANDIDATE PREMISES" in user:
                with self.lock:
                    self.reason_calls += 1
                    first = self.reason_calls == 1
                if first:  # no reply until the request gives up
                    threading.Event().wait(min(kw["timeout"], 6.0))
                    raise TimeoutError("The read operation timed out")
            return super().complete(system, user, **kw)

    unit = ROOT / "units" / "t4-fomc-curve-20240918"
    budget = Budget()
    t0 = time.monotonic()
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "answer.json", FirstReasonsStalls(budget), budget,
                 Schedule(), t0, log=lambda *a: None)
    assert len(answer.get("submitted_reasons", [])) == 3
    assert time.monotonic() - t0 < 5.0  # the hedge answered; nothing waited for the stalled call


def test_reasons_that_leave_entities_uncovered_are_retried(tmp_path):
    class NarrowFirst(BudgetedChat):
        reason_calls = 0

        def complete(self, system, user, **kw):
            reply = super().complete(system, user, **kw)
            if "CANDIDATE PREMISES" in user:
                self.reason_calls += 1
                if self.reason_calls == 1:  # three reasons, one entity each
                    got = json.loads(reply)["reasons"]
                    return json.dumps({"reasons": [dict(r, entities=r["entities"][i:i + 1])
                                                   for i, r in enumerate(got)]})
            return reply

    unit = ROOT / "units" / "t4-fomc-curve-20240918"
    budget = Budget()
    chat = NarrowFirst(budget)
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "answer.json", chat, budget,
                 Schedule(), time.monotonic(), log=lambda *a: None)
    covered = {e for r in answer["submitted_reasons"] for e in r["scope"]["entities"]}
    assert chat.reason_calls >= 2 and covered == {r["entity_id"] for r in answer["entity_predictions"]}


def test_large_roster_answers_fit_the_reasoning_cap_and_keep_reasons(tmp_path):
    from baselines.vector_agent.agent import ANSWER_BYTES_CAP, _judge_view_bytes
    unit = _synthetic_unit(tmp_path, "t4-macrorev-20240930-us6", 26)
    answer, *_ = _run(unit, tmp_path)
    rows = answer["entity_predictions"]
    assert len(rows) == 26 and _judge_view_bytes(rows, "classification") <= ANSWER_BYTES_CAP
    assert answer.get("submitted_reasons")
    assert all(r["interval"]["lo"] <= r["point_forecast"] <= r["interval"]["hi"] for r in rows)


def test_answer_cap_fit_moves_numbers_by_at_most_one_percent_of_the_band():
    from baselines.vector_agent.agent import ANSWER_BYTES_CAP, _judge_view_bytes, fit_answer_cap
    rows = [{"entity_id": f"E{i:02d}", "point_forecast": 158551.123456 + i,
             "interval": {"level": 0.9, "lo": 158048.987654 + i, "hi": 158961.4321 + i}} for i in range(27)]
    before = json.loads(json.dumps(rows))
    assert _judge_view_bytes(rows, "regression") > ANSWER_BYTES_CAP
    note = fit_answer_cap(rows, "regression")
    assert note.startswith("numbers rounded") and _judge_view_bytes(rows, "regression") <= ANSWER_BYTES_CAP
    for r, b in zip(rows, before):
        width = b["interval"]["hi"] - b["interval"]["lo"]
        assert r["interval"]["lo"] <= b["interval"]["lo"] and r["interval"]["hi"] >= b["interval"]["hi"]
        assert abs(r["point_forecast"] - b["point_forecast"]) <= 0.01 * width
    big = [dict(r, entity_id="X" * 90) for r in before]
    assert fit_answer_cap(big, "regression") == "over cap" and big[0]["point_forecast"] == before[0]["point_forecast"]


def test_reasons_arguing_against_the_submitted_answer_are_dropped():
    from baselines.vector_agent.claims import build_reasons
    unit = load_unit(ROOT / "units" / "t4-fomc-curve-20240918" / "task.json",
                     ROOT / "units" / "t4-fomc-curve-20240918" / "corpus")
    doc = unit.docs["RATES_SNAPSHOT_20240919"]
    cands = [(f"R{i + 1}", {"doc_id": doc.doc_id, "span_start": 300 * i, "span_end": 300 * i + 60,
                            "claim": doc.text[300 * i:300 * i + 60]}) for i in range(3)]
    reply = {"reasons": [
        {"premise": "R1", "mechanism": "m", "answer_implication": "UST2Y -5 bps: this conflicts with easing"},
        {"premise": "R2", "mechanism": "m", "answer_implication": "UST10Y yields rise more than submitted"},
        {"premise": "R3", "mechanism": "m", "answer_implication": "UST30Y +3 bps as term premium rebuilds"}]}
    got = build_reasons(unit, reply, cands)
    assert [r["answer_implication"] for r in got] == ["UST30Y +3 bps as term premium rebuilds"]


def test_a_429_makes_every_worker_wait_out_the_same_backoff(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from baselines.vector_agent.house import HouseClient, HouseConfig

    def refuse(request, timeout):
        raise urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "7"}, io.BytesIO(b""))

    slept = []
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    client = HouseClient(HouseConfig("http://h", "m", "t"), Budget())
    with pytest.raises(urllib.error.HTTPError):
        client.complete("s", "u", temperature=0, seed=1, max_tokens=10, timeout=100)
    assert client._cool_until > time.monotonic() + 5  # a second worker would now wait before its attempt


def test_schedule_follows_the_unit_card_timeout(tmp_path):
    from baselines.vector_agent.cli import card_timeout
    assert card_timeout(ROOT / "units" / "t4-fomc-curve-20240918" / "task.json") == 600.0
    assert Schedule.for_unit_timeout(600.0) == Schedule() and Schedule.for_unit_timeout(None) == Schedule()
    short = Schedule.for_unit_timeout(300.0)
    assert short.reasons_stop == 200.0 and short.hard_stop < short.reasons_stop
    (tmp_path / "task.json").write_text("{}")
    assert card_timeout(tmp_path / "task.json") is None  # no card mounted: the 600 s schedule


def test_threshold_labels_take_the_most_probable_range():
    from baselines.vector_agent.aggregate import range_probabilities
    from baselines.vector_agent.thresholds import label_ranges
    task = json.loads((ROOT / "units" / "t4-postearn-20240201-megacap" / "task.json").read_text())
    ranges = label_ranges(task)
    assert ranges == {"positive_reaction": (1.0, float("inf")), "negative_reaction": (float("-inf"), -1.0),
                      "flat": (-1.0, 1.0)}
    for unit in ("t4-EXAMPLE-eps-beat", "t4-credit-event-2023", "t4-macrorev-20240930-us6"):
        assert label_ranges(json.loads((ROOT / "units" / unit / "task.json").read_text())) is None
    wide = [_sample("AAPL", 0.3, -5.0, 5.6, "flat") for _ in range(3)]  # point inside the flat range
    probs = range_probabilities(wide, ranges)
    assert max(probs, key=probs.get) == "positive_reaction" and probs["flat"] < 0.35
    narrow = [_sample("AAPL", 0.2, -0.8, 1.2, "flat") for _ in range(3)]
    probs = range_probabilities(narrow, ranges)
    assert max(probs, key=probs.get) == "flat"


def test_ready_entities_stop_waiting_for_stragglers_at_reasons_latest(tmp_path):
    # Three fast forecasts cover every entity; the other first-wave calls hang. Collection must
    # end at reasons_latest, not hard_stop, so the reasons call still has its window.
    unit = ROOT / "units" / "t4-cpicomp-202410-us11"
    budget = Budget()
    chat = BudgetedChat(budget, faults={i: "slow" for i in range(3, 30)})
    schedule = Schedule(launch_stop=0.2, extras_stop=0.2, max_launch=0.2, hard_stop=6.0,
                        reasons_latest=1.0, reasons_stop=8.0, request_timeout=5.0)
    from baselines.vector_agent.agent import forecast
    start = time.monotonic()
    results, plan = forecast(load_unit(unit / "task.json", unit / "corpus"), chat, budget, start, schedule)
    elapsed = time.monotonic() - start
    assert 1.0 <= elapsed < 2.5
    assert all(len(plan.samples[eid]) >= 3 for eid in plan.ids) and len(results) == len(plan.ids)
