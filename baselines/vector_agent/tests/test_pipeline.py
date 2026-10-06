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
        assert loaded.docs[cite["doc_id"]].text[cite["span_start"]:cite["span_end"]] == reason["premise"]


@pytest.mark.parametrize("unit_name", ["t4-macrorev-20240930-us6", "t4-credit-event-2023"])
def test_faulty_replies_are_absorbed(unit_name, tmp_path):
    faults = {0: "error", 1: "garbage", 2: "drop_one", 3: "fenced", 4: "think", 5: "case", 6: "error"}
    answer, chat, budget, _ = _run(ROOT / "units" / unit_name, tmp_path, faults=faults)
    assert len(answer["entity_predictions"]) == len(load_unit(
        ROOT / "units" / unit_name / "task.json", ROOT / "units" / unit_name / "corpus").entities)
    assert budget.used <= 24


def test_deadline_stops_waiting_and_fails_without_forecasts(tmp_path):
    unit = ROOT / "units" / "t4-fomc-curve-20240918"
    budget = Budget()
    chat = BudgetedChat(budget, faults={i: "slow" for i in range(30)})
    schedule = Schedule(launch_stop=0.5, hard_stop=1.0, reasons_latest=1.0, reasons_stop=1.5,
                        request_timeout=0.6)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="no usable House forecast"):
        run(unit / "task.json", unit / "corpus", tmp_path / "a.json", chat, budget, schedule, start,
            log=lambda *a: None)
    assert time.monotonic() - start < 3.0
    assert not (tmp_path / "a.json").exists()


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
