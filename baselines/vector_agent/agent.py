"""Orchestrate House calls for one unit and assemble the answer.

Small corpora go to the model whole, with every entity in one prompt, sampled several times.
Large corpora get one prompt per entity with that entity's retrieved excerpts, plus one compact
unit-wide prompt for cross-sectional context. Calls run concurrently under a request cap and a
wall-clock schedule; every forecast in the answer comes from model samples.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace

from . import prompts
from .aggregate import _Z90, Forecast, _round, combine, range_probabilities
from .consistency import bps_change_target, change_sd_floor, ground_growth, percent_growth_target
from .drivers import task_drivers
from .thresholds import label_ranges
from . import cohort
from .claims import build_reasons, entity_claims, reason_candidates
from .evidence import Passage, entity_passages
from .house import Budget, BudgetExhausted, Chat
from .parse import Sample, extract_json, parse_predictions, salvage_objects
from .unit import Unit

FULL_CONTEXT_CHARS = 110_000
ENTITY_EVIDENCE_CHARS = 44_000
QUICK_EVIDENCE_CHARS = 48_000
GROUP_EVIDENCE_CHARS = 80_000
THINKING = False  # Nemotron 3 Super reasons before answering (the House default); costs output tokens
UNIT_SAMPLES = 3 if THINKING else 9
UNIT_SAMPLES_MAX = 20  # with time and budget to spare, more samples steady the medians
EXTRA_ROUNDS = 3  # likewise for retrieval groups
MAX_ENTITY_SAMPLES = 3 if THINKING else 5
RETRIEVAL_GROUPS_SMALL = 5  # rosters of up to 10 rows: at most 5 retrieval groups, so each gets ~5 samples
GROUP_MAX_FULL = 4 if THINKING else 12
MAX_RETRIEVAL_GROUPS = 10
REASONS_CALLS = 1  # planned: the sample targets leave room for one reasons call
REASONS_RESERVE = 2  # held back from launches once covered, so one failed reasons call can be retried
REASONS_ATTEMPTS = 3  # while budget and the reasons window allow; a failed call costs only its request
REASONS_HEDGE = 2  # reasons calls in flight at once
REASONS_MIN_WINDOW = 30.0  # seconds before reasons_stop: a late reasons call still gets a short timeout
# Preserve pooled sample intervals and the corpus-derived floor without learned rescaling.
# These identity values are chosen independently of realized outcomes. The former values
# and their practice/private selection history remain disclosed in ARTIFACT_PROVENANCE.md.
BAND_SCALE = 1.0
YIELD_BAND_SCALE = 1.0
CLASS_BAND_SCALE = 1.0
COVER_MIN_WINDOW = 60.0  # seconds before hard_stop in which a call for an uncovered entity still launches
ANSWER_BYTES_CAP = 2900  # the reasoning grader reads at most 3000 bytes of per-entity answers
MAX_FULL_GROUPS = 16
MAX_PROMPT_CHARS = 120_000  # cohort and unit-wide prompts are cut to this before sending


@dataclass
class Schedule:
    """Seconds after process start. The unit allows 600 s including container creation and any
    image pull, so the schedule ends by reasons_stop and leaves about 100 s for start-up.

    Before any call has returned, new calls launch until launch_stop. Once latencies are known,
    a call launches only if it should finish (1.25 x the 75th-percentile latency) before
    hard_stop, never after max_launch; an extra (variance-reduction) sample launches only if it
    should also finish before reasons_latest, which keeps the reasons step's slot."""
    launch_stop: float = 300.0
    max_launch: float = 400.0
    hard_stop: float = 460.0
    reasons_latest: float = 390.0
    reasons_stop: float = 500.0
    request_timeout: float = 240.0
    workers: int = 4
    extras_stop: float = 200.0  # extras before any latency is observed

    @classmethod
    def for_unit_timeout(cls, timeout_sec: float | None) -> "Schedule":
        """The schedule scaled to a unit time limit other than 600 s: the ~100 s start-up margin
        stays fixed and every deadline scales with the remaining time."""
        base = cls()
        if not timeout_sec or not 120.0 <= timeout_sec <= 7200.0 or abs(timeout_sec - 600.0) < 1.0:
            return base
        f = (timeout_sec - 100.0) / base.reasons_stop
        return cls(**{k: (v * f if isinstance(v, float) and k != "request_timeout" else v)
                      for k, v in base.__dict__.items()})


@dataclass
class Job:
    kind: str  # "unit" | "group" | "entity"
    key: str
    ids: tuple[str, ...]
    prompt: str
    temperature: float
    seed: int
    max_tokens: int
    weight: float


@dataclass
class Trace:
    mode: str = ""
    launched: Counter = field(default_factory=Counter)
    succeeded: Counter = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Planner:
    """Decides the next House call. Work is organised in groups of entities that share one prompt.

    Full-context mode: groups of at most 12 entities over the whole corpus, sampled until every
    entity has UNIT_SAMPLES forecasts. Retrieval mode: at most 5 groups for rosters of up to 10
    rows (one or two entities per prompt, each with its own excerpts), otherwise at most 10 groups,
    sampled in rounds; with single-entity groups, spare budget goes to the entities whose samples
    disagree most.
    """

    def __init__(self, unit: Unit, seed: int):
        self.unit = unit
        self.seed = seed
        self.ids = unit.entity_ids
        self.full = unit.total_chars() <= FULL_CONTEXT_CHARS
        self.samples: dict[str, list[Sample]] = {eid: [] for eid in self.ids}
        self.trace = Trace(mode="full-context" if self.full else "retrieval")
        self.inflight: Counter = Counter()
        self.attempts: Counter = Counter()
        self.done: Counter = Counter()
        self.passages: dict[str, list[Passage]] = {}
        self.entity_prompts: dict[str, str] = {}
        self.unit_prompt: str | None = None
        self.growth = percent_growth_target(unit.task)
        self.rows = {str(e["entity_id"]): e for e in unit.entities}
        self.event = cohort.event_labels(unit.labels) if unit.target_type == "classification" else None
        self.cohort_counts: list[float] = []
        self.cohort_rankings: list[list[str]] = []
        self.cohort_prompt: str | None = None
        self.retired: set[int] = set()  # full-context groups split after a timed-out call
        n = max(1, len(self.ids))
        if self.full:
            self.groups = _chunks(self.ids, max(GROUP_MAX_FULL, -(-n // MAX_FULL_GROUPS)))
            docs = prompts.unit_docs_full(unit)
            self.docs_full = docs
            focus = len(self.groups) > 1
            self.group_prompts = [prompts.unit_prompt(unit, docs, g if focus else None) for g in self.groups]
            self.target = UNIT_SAMPLES
            if self.event:
                self.cohort_prompt = _fit(cohort.question(_head(prompts.unit_prompt(unit, docs)), self.event[0], n))
            return
        for e in unit.entities:
            self.passages[str(e["entity_id"])] = entity_passages(unit, e, ENTITY_EVIDENCE_CHARS)
        n_groups = min(n, RETRIEVAL_GROUPS_SMALL) if n <= 2 * RETRIEVAL_GROUPS_SMALL else MAX_RETRIEVAL_GROUPS
        self.groups = _chunks(self.ids, -(-n // n_groups))
        if len(self.groups[0]) == 1:
            self.group_prompts = [self._entity_prompt(g[0]) for g in self.groups]
            if n <= GROUP_MAX_FULL:  # one compact unit-wide pass for cross-sectional context
                share = max(2500, QUICK_EVIDENCE_CHARS // n)
                quick = {eid: _trim(ps, share) for eid, ps in self.passages.items()}
                self.unit_prompt = _fit(prompts.unit_prompt(unit, prompts.unit_docs_excerpts(unit, quick)))
        else:
            self.group_prompts = []
            for g in self.groups:
                share = max(2500, min(ENTITY_EVIDENCE_CHARS, GROUP_EVIDENCE_CHARS // len(g)))
                excerpts = {eid: _trim(self.passages[eid], share) for eid in g}
                self.group_prompts.append(prompts.unit_prompt(unit, prompts.unit_docs_excerpts(unit, excerpts), g))
        if self.event:
            share = max(1200, QUICK_EVIDENCE_CHARS // n)
            compact = {eid: _trim(ps, share) for eid, ps in self.passages.items()}
            self.cohort_prompt = _fit(cohort.question(
                _head(prompts.unit_prompt(unit, prompts.unit_docs_excerpts(unit, compact))), self.event[0], n))
        spare = (24 - REASONS_CALLS - 2 - (1 if self.unit_prompt else 0)
                 - (cohort.COHORT_CALLS if self.event else 0))
        self.target = max(1, min(MAX_ENTITY_SAMPLES, spare // len(self.groups)))

    def _entity_prompt(self, eid: str) -> str:
        if eid not in self.entity_prompts:
            entity = next(e for e in self.unit.entities if str(e["entity_id"]) == eid)
            if self.full:
                ps = [Passage(d.doc_id, 0, len(d.text), d.text) for d in self.unit.docs_for(eid)]
            else:
                ps = self.passages[eid]
            self.entity_prompts[eid] = prompts.entity_prompt(self.unit, entity, ps)
        return self.entity_prompts[eid]

    def _job(self, kind: str, key: str, ids: list[str], prompt: str, weight: float = 1.0) -> Job:
        k = self.attempts[key]
        if THINKING:
            tokens, temperature = 4000, 0.6
        else:
            tokens = 1500 if len(ids) == 1 else min(4000, 900 + 400 * len(ids))
            temperature = 0.0 if k == 0 else 0.7
        return Job(kind, key, tuple(ids), prompt, temperature,
                   self.seed + 1000 * (_stable(key) % 997) + k, tokens, weight)

    def disagreement(self, eid: str) -> float:
        """How much an entity's samples disagree, relative to their own stated uncertainty."""
        ss = self.samples[eid]
        if len(ss) < 2:
            return 0.0
        if self.unit.target_type == "classification":
            votes = Counter(s.label for s in ss)
            split = 1.0 - max(votes.values()) / len(ss)
            if split > 0:
                return 1.0 + split
        points = sorted(s.point for s in ss if s.point is not None)
        halves = sorted((s.hi - s.lo) / 2 for s in ss if s.hi is not None and s.lo is not None)
        if len(points) < 2 or not halves:
            return 0.0
        half = halves[len(halves) // 2]
        return (points[-1] - points[0]) / half if half > 0 else 0.0

    def _rescue(self) -> Job | None:
        """Rescue uncovered entities before adding a second sample to covered ones."""
        for count in (0, 1):
            for idx, g in enumerate(self.groups):
                if self.inflight[f"g{idx}"] or idx in self.retired:
                    continue
                for eid in g:
                    if len(self.samples[eid]) == count and self.inflight[eid] == 0 and self.attempts[eid] < 3 \
                            and (len(g) > 1 or self.attempts[f"g{idx}"] >= self.target + 2):
                        return self._job("entity", eid, [eid], self._entity_prompt(eid))
        return None

    def next_job(self, budget_left: int, extras: bool = False) -> Job | None:
        """``extras``: early enough in the schedule to spend spare budget on more samples."""
        extras = extras and budget_left > REASONS_RESERVE + 1
        if (self.cohort_prompt and self.attempts["cohort"] < cohort.COHORT_CALLS
                and self.inflight["cohort"] == 0 and all(self.samples[eid] for eid in self.ids)):
            k = self.attempts["cohort"]
            return Job("cohort", "cohort", tuple(self.ids), self.cohort_prompt, 0.7,
                       self.seed + 7000 + k, 900, 1.0)
        if self.full:
            order = sorted((i for i in range(len(self.groups)) if i not in self.retired),
                           key=lambda i: min(len(self.samples[e]) for e in self.groups[i]) + self.inflight[f"g{i}"])
            for idx in order:
                key, g = f"g{idx}", self.groups[idx]
                floor = min(len(self.samples[eid]) for eid in g)
                if floor + self.inflight[key] < self.target and self.attempts[key] < self.target + 4:
                    return self._job("unit", key, g, self.group_prompts[idx])
            rescue = self._rescue()
            if rescue or not extras:
                return rescue
            for idx in order:
                key, g = f"g{idx}", self.groups[idx]
                floor = min(len(self.samples[eid]) for eid in g)
                if floor + self.inflight[key] < UNIT_SAMPLES_MAX and self.attempts[key] < UNIT_SAMPLES_MAX + 4:
                    return self._job("unit", key, g, self.group_prompts[idx])
            return None
        # Retrieval mode: rounds over the groups (plus the unit-wide pass), then rescue, then depth.
        rounds = self.target + (EXTRA_ROUNDS if extras and all(self.done[f"g{i}"] >= self.target
                                                               for i in range(len(self.groups))) else 0)
        for depth in range(1, rounds + 1):
            for idx, g in enumerate(self.groups):
                key = f"g{idx}"
                if self.done[key] + self.inflight[key] >= depth or self.attempts[key] >= rounds + 2:
                    continue
                return self._job("entity" if len(g) == 1 else "group", key, g, self.group_prompts[idx])
            if depth == 1 and self.unit_prompt and self.inflight["unit"] == 0 and self.done["unit"] < 1 \
                    and self.attempts["unit"] < 2:
                return self._job("unit", "unit", self.ids, self.unit_prompt, weight=0.5)
        rescue = self._rescue()
        if rescue or len(self.groups[0]) > 1:
            return rescue
        # Spare budget goes to the entities whose samples disagree most.
        if budget_left <= REASONS_RESERVE + 1 or any(self.inflight.values()):
            return None
        for eid in sorted(self.ids, key=self.disagreement, reverse=True):
            key = f"g{self.ids.index(eid)}"
            if self.disagreement(eid) < 1.0 or self.attempts[key] >= self.target + 3:
                continue
            return self._job("entity", key, [eid], self.group_prompts[self.ids.index(eid)])
        return None

    def _split(self, job: Job) -> None:
        """A full-context call that timed out is replaced by calls on each half of its group: on a
        slow House an 11-entity reply can outlast the request timeout every time, while a half
        reply fits."""
        if not self.full or job.kind != "unit" or len(job.ids) < 2 or not job.key.startswith("g"):
            return
        idx = int(job.key[1:])
        if idx in self.retired or len(self.groups) >= 2 * MAX_FULL_GROUPS:
            return
        self.retired.add(idx)
        for half in _chunks(list(self.groups[idx]), -(-len(self.groups[idx]) // 2)):
            self.groups.append(half)
            self.group_prompts.append(prompts.unit_prompt(self.unit, self.docs_full, half))
        self.trace.notes.append(f"g{idx}: timed out, split into halves")

    def uncovered(self) -> bool:
        return any(not self.samples[eid] for eid in self.ids)

    def mark_launched(self, job: Job) -> None:
        self.attempts[job.key] += 1
        self.inflight[job.key] += 1
        self.trace.launched[job.kind] += 1

    def absorb(self, job: Job, raw: str | None, error: str | None) -> None:
        self.inflight[job.key] -= 1
        if raw is None:
            self.trace.failures.append(f"{job.key}: {error}")
            if "timed out" in str(error).lower() or "timeout" in str(error).lower():
                self._split(job)
            return
        if job.kind == "cohort":
            count, ranking = cohort.parse(raw, set(self.ids))
            if count is None:
                self.trace.failures.append("cohort: unusable reply")
                return
            self.cohort_counts.append(count)
            if ranking:
                self.cohort_rankings.append(ranking)
            self.trace.succeeded["cohort"] += 1
            return
        try:
            got = parse_predictions(raw, target_type=self.unit.target_type, labels=self.unit.labels,
                                    entity_ids=set(job.ids), weight=job.weight, source=job.kind)
            got = [s for s in got if s.lo is not None and s.hi is not None]
            if self.growth:
                got = [ground_growth(s, self.rows[s.entity_id]) for s in got]
        except Exception as exc:  # noqa: BLE001 - a malformed reply is a failed call, never a crash
            self.trace.failures.append(f"{job.key}: unparsable reply ({type(exc).__name__})")
            return
        if not got:
            self.trace.failures.append(f"{job.key}: unusable reply")
            return
        self.done[job.key] += 1
        self.trace.succeeded[job.kind] += 1
        for s in got:
            self.samples[s.entity_id].append(s)


def _chunks(ids: list[str], size: int) -> list[list[str]]:
    """Split ``ids`` into near-equal consecutive groups of at most ``size``."""
    n_groups = max(1, -(-len(ids) // max(1, size)))
    base, extra = divmod(len(ids), n_groups)
    out, i = [], 0
    for k in range(n_groups):
        step = base + (1 if k < extra else 0)
        out.append(ids[i:i + step])
        i += step
    return [g for g in out if g]


def _stable(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text))


def _probability_target(unit: Unit, samples: dict[str, list[Sample]]) -> bool:
    """A classification unit whose numeric quantity is a probability: bands stay inside [0, 1]."""
    if unit.target_type != "classification" or "probabilit" not in str(unit.task.get("prompt", "")).lower():
        return False
    points = [s.point for ss in samples.values() for s in ss if s.point is not None]
    return bool(points) and all(0.0 <= p <= 1.0 for p in points)


def _fit(prompt: str) -> str:
    """Cut a prompt's evidence so the whole prompt stays within MAX_PROMPT_CHARS."""
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    return prompts.shrink_evidence(prompt, max(0.05, MAX_PROMPT_CHARS / len(prompt) * 0.9))


def _head(prompt: str) -> str:
    """The task, table and evidence of a forecasting prompt, without its method and output rules."""
    cut = prompt.find("How to forecast:")
    return prompt[:cut] if cut > 0 else prompt


def _trim(passages, budget: int):
    """Excerpts for a prompt within ``budget`` characters; an oversized first one is cut to fit."""
    out, used = [], 0
    for p in passages:
        if used + len(p.text) > budget:
            if out:
                continue
            p = Passage(p.doc_id, p.start, p.start + budget, p.text[:budget])
        out.append(p)
        used += len(p.text)
    return out


def _call(chat: Chat, job: Job, timeout: float) -> str:
    """One House call; a request the route rejects as malformed (HTTP 400, e.g. a prompt over the
    serving context) is retried with the evidence section cut to a half, then a quarter."""
    deadline = time.monotonic() + timeout
    prompt = job.prompt
    for factor in (None, 0.5, 0.25):
        if factor is not None:
            prompt = prompts.shrink_evidence(job.prompt, factor)
        try:
            return chat.complete(prompts.SYSTEM, prompt, temperature=job.temperature, seed=job.seed,
                                 max_tokens=job.max_tokens, timeout=max(5.0, deadline - time.monotonic()),
                                 thinking=THINKING)
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or factor == 0.25 or deadline - time.monotonic() < 20:
                raise
    raise RuntimeError("unreachable")


def forecast(unit: Unit, chat: Chat, budget: Budget, start: float, schedule: Schedule,
             seed: int = 20261006) -> tuple[dict[str, Forecast], Planner]:
    plan = Planner(unit, seed)
    pool = ThreadPoolExecutor(max_workers=schedule.workers)
    pending: dict[Future, Job] = {}
    launched_at: dict[Future, float] = {}
    durations: list[float] = []
    try:
        while True:
            now = time.monotonic() - start
            workers = 2 if getattr(chat, "rate_limited", 0) else schedule.workers  # back off once throttled
            if durations:
                p75 = sorted(durations)[int(0.75 * (len(durations) - 1))]
                launch_ok = now < schedule.max_launch and now + 1.25 * p75 < schedule.hard_stop
                # Leave the reasons call a slot as long as an observed call: at a slow House, a
                # last forecast wave adds little while losing the reasons costs the whole bonus.
                reasons_slot = max(REASONS_MIN_WINDOW, 1.25 * p75)
                launch_ok = launch_ok and now + 1.25 * p75 + reasons_slot <= schedule.reasons_stop
                extras = launch_ok and now + 1.25 * p75 < schedule.reasons_latest
            else:
                launch_ok, extras = now < schedule.launch_stop, now < schedule.extras_stop
            if plan.uncovered() and schedule.hard_stop - now >= COVER_MIN_WINDOW:
                # Timed-out calls inflate the latency estimate; an entity without any forecast
                # fails the whole unit, so a late attempt is always worth its request.
                launch_ok = True
            # Coverage is required and reasons are optional: the reserve goes to an entity still
            # without a forecast.
            reserve = 0 if plan.uncovered() else REASONS_RESERVE
            while len(pending) < workers and launch_ok and budget.remaining - len(pending) > reserve:
                job = plan.next_job(budget.remaining - len(pending), extras=extras)
                if job is None:
                    break
                plan.mark_launched(job)
                timeout = min(schedule.request_timeout, schedule.hard_stop - now)
                fut = pool.submit(_call, chat, job, timeout)
                pending[fut], launched_at[fut] = job, now
            if not pending:
                break
            done, _ = wait(pending, timeout=max(0.05, schedule.hard_stop - now),
                           return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                job = pending.pop(fut)
                durations.append(time.monotonic() - start - launched_at.pop(fut))
                try:
                    raw, err = fut.result(), None
                except BudgetExhausted as exc:
                    raw, err = None, str(exc)
                except Exception as exc:  # noqa: BLE001 - any call failure is recorded, not fatal
                    raw, err = None, f"{type(exc).__name__}: {exc}"[:200]
                plan.absorb(job, raw, err)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    results = {}
    bounded = _probability_target(unit, plan.samples)
    floors = change_sd_floor(unit.task, unit.docs.values(), prompts.horizon_days(unit))
    bps_target = bps_change_target(unit.task)
    widened = 0
    for eid in plan.ids:
        fc = combine(plan.samples[eid], unit.target_type, unit_interval=bounded)
        if fc is not None and fc.point is not None and eid in floors:
            half = _Z90 * floors[eid]
            lo, hi = min(fc.lo, _round(fc.point - half)), max(fc.hi, _round(fc.point + half))
            widened += (lo, hi) != (fc.lo, fc.hi)
            fc = replace(fc, lo=lo, hi=hi)
        placeholder = fc is not None and fc.lo == 0.0 and fc.hi == 1.0  # a pure-label unit's unscored band
        scale = (YIELD_BAND_SCALE if bps_target
                 else BAND_SCALE if unit.target_type in ("regression", "ranking")
                 else CLASS_BAND_SCALE if not (bounded or placeholder) else 1.0)  # probability bands stay score-optimal
        if fc is not None and fc.point is not None and scale != 1.0:
            fc = replace(fc, lo=_round(fc.point - scale * (fc.point - fc.lo)),
                         hi=_round(fc.point + scale * (fc.hi - fc.point)))
        if fc is not None:
            results[eid] = fc
    if floors:
        plan.trace.notes.append(f"horizon random-walk band floor: {widened} of {len(floors)} bands widened")
    ranges = label_ranges(unit.task)
    if ranges:
        changed = 0
        for eid, fc in list(results.items()):
            probs = range_probabilities(plan.samples[eid], ranges)
            if probs and max(probs.values()) > 0:
                best = max(probs, key=probs.get)
                changed += best != fc.label
                results[eid] = replace(fc, label=best)
        plan.trace.notes.append(f"threshold labels: most probable range chosen, {changed} label(s) changed")
    if plan.event and len(results) == len(plan.ids):
        note = cohort.relabel(results, plan.samples, plan.event[0], plan.event[1], plan.cohort_counts,
                              plan.cohort_rankings, bounded)
        plan.trace.notes.append(note)
    return results, plan


def _judge_view_bytes(rows: list[dict], target_type: str) -> int:
    """Bytes of the answer as the reasoning judge reads it: the fields the unit declares
    (classification: label and, when scored, interval; regression and ranking: point_forecast
    and interval), counting the interval whether or not it is declared."""
    keys = ("entity_id", "label", "interval") if target_type == "classification" else (
        "entity_id", "point_forecast", "interval")
    view = [{k: r[k] for k in keys if k in r} for r in rows]
    return len(json.dumps(view, ensure_ascii=False, separators=(",", ":")).encode())


def _outward(x: float, step: float, direction: int) -> float:
    op = math.floor if direction < 0 else math.ceil if direction > 0 else round
    digits = max(0, -int(math.floor(math.log10(step))))
    return round(op(x / step) * step, digits)


def fit_answer_cap(rows: list[dict], target_type: str) -> str:
    """Large rosters can exceed the reasoning grader's answer cap, which would leave the unit
    with no judged reasons. Numbers are rounded to 1% of each row's band width (bands outward),
    which moves no scored value materially; if that is not enough, nothing is changed and the
    reasons step is skipped."""
    if _judge_view_bytes(rows, target_type) <= ANSWER_BYTES_CAP:
        return "within cap"
    trial = json.loads(json.dumps(rows))
    for r in trial:
        iv = r["interval"]
        width = iv["hi"] - iv["lo"]
        if not (math.isfinite(width) and width > 0):
            continue
        step = 10.0 ** math.floor(math.log10(width / 100.0))
        iv["lo"], iv["hi"] = _outward(iv["lo"], step, -1), _outward(iv["hi"], step, +1)
        if "point_forecast" in r:
            r["point_forecast"] = min(max(_outward(r["point_forecast"], step, 0), iv["lo"]), iv["hi"])
    if _judge_view_bytes(trial, target_type) > ANSWER_BYTES_CAP:
        return "over cap"
    for r, t in zip(rows, trial):
        r["interval"] = t["interval"]
        if "point_forecast" in t:
            r["point_forecast"] = t["point_forecast"]
    return "numbers rounded to 1% of each band's width to fit the answer cap"


def rationales(unit: Unit, rows: list[dict], samples: dict[str, list[Sample]]) -> dict[str, str]:
    """Per entity, the reasoning of a sample that agrees with the submitted answer: the submitted
    label, else the point nearest the submitted point, heavier samples first."""
    out = {}
    for row in rows:
        pool = [s for s in samples.get(row["entity_id"], []) if s.analysis]
        if "label" in row:
            pool = [s for s in pool if s.label == row["label"]] or pool
        if not pool:
            continue
        point = row.get("point_forecast")
        best = min(pool, key=lambda s: (abs(s.point - point) if point is not None and s.point is not None
                                        else 0.0, -s.weight))
        out[row["entity_id"]] = " ".join(best.analysis.split())[:300]
    return out


def _reach(built: list[dict], unit: Unit) -> tuple[bool, int, int]:
    """How completely a reasons set does what the prompt asks: 3 reasons that together cover every
    entity, then the number of reasons, then the number of entities covered. On the private set,
    three single-entity reasons on one driver covered one of five target reasons."""
    covered = {e for r in built for e in (r.get("scope") or {}).get("entities", [])}
    n = len(set(unit.entity_ids) & covered)
    return len(built) >= 3 and n == len(unit.entity_ids), len(built), n


def _probability_row(row: dict) -> bool:
    """A classification row whose number is an event probability (its band lies within [0, 1])."""
    iv = row.get("interval") or {}
    return 0.0 <= iv.get("lo", -1) and iv.get("hi", 2) <= 1.0


def reasons(unit: Unit, chat: Chat, budget: Budget, start: float, schedule: Schedule,
            rows: list[dict], claims_by_entity: dict[str, list[dict]], seed: int,
            passages: dict[str, list[Passage]] | None = None,
            why: dict[str, str] | None = None) -> tuple[list[dict], str]:
    now = time.monotonic() - start
    if schedule.reasons_stop - now < REASONS_MIN_WINDOW or budget.remaining < 1:
        return [], "skipped (time or budget)"
    if _judge_view_bytes(rows, unit.target_type) > ANSWER_BYTES_CAP:
        return [], "skipped (answer over the reasoning grader's cap)"
    named = task_drivers(str(unit.task.get("prompt", "")))
    ranges: dict[str, list[tuple[int, int]]] = {}
    for ps in (passages or {}).values():
        for p in ps:
            ranges.setdefault(p.doc_id, []).append((p.start, p.end))
    candidates = reason_candidates(unit, claims_by_entity, named, ranges)
    if not candidates:
        return [], "skipped (no candidate premises)"
    declared = []
    for row in rows:
        view = {"entity_id": row["entity_id"]}
        if "label" in row:  # a classification answer is read by its label; its probability is not shown
            view["label"] = row["label"]
        elif "point_forecast" in row:  # as the judge reads the answer: rounding here would misquote it
            view["point_forecast"] = float(f"{row['point_forecast']:.6g}")
        if "label" not in row or not _probability_row(row):
            view["interval"] = [float(f"{row['interval']['lo']:.6g}"), float(f"{row['interval']['hi']:.6g}")]
        if (why or {}).get(row["entity_id"]):
            view["rationale"] = why[row["entity_id"]]
        declared.append(view)
    listed = []
    for key, c in candidates:
        doc = unit.docs.get(c["doc_id"])
        applies = "all entities" if doc is None or doc.entity_ids is None else ", ".join(sorted(doc.entity_ids))
        tag = f"driver: {c['driver']}; " if c.get("driver") else ""
        listed.append((key, c["claim"], c["doc_id"], tag + "applies to " + applies))
    text = prompts.reasons_prompt(unit, declared, listed, named)
    note, best = "not attempted", []
    pool = ThreadPoolExecutor(max_workers=REASONS_HEDGE)
    pending: dict[Future, int] = {}
    launched = 0
    try:
        while True:
            now = time.monotonic() - start
            window = schedule.reasons_stop - now  # a slow reply may use all the time left
            # Hedged requests: a second call runs alongside the first, so one stalled reply no
            # longer costs the reasons (on the private set single calls stalled for 245-307 s).
            while (len(pending) < REASONS_HEDGE and launched < REASONS_ATTEMPTS
                   and window >= REASONS_MIN_WINDOW and budget.remaining >= 1):
                fut = pool.submit(chat.complete, prompts.SYSTEM, text, temperature=0.0 if launched == 0 else 0.5,
                                  seed=seed + launched, max_tokens=2000, timeout=window)
                pending[fut] = launched
                launched += 1
            if not pending:
                break
            done, _ = wait(pending, timeout=max(0.05, window + 2), return_when=FIRST_COMPLETED)
            if not done:
                note = "failed (TimeoutError)"
                break
            for fut in done:
                pending.pop(fut)
                try:
                    raw = fut.result()
                except Exception as exc:  # noqa: BLE001 - reasons are optional; never fail the unit for them
                    note = f"failed ({type(exc).__name__})"
                    continue
                reply = extract_json(raw, ("reasons",)) or {"reasons": salvage_objects(raw or "", "premise")}
                built = build_reasons(unit, reply, candidates)
                if _reach(built, unit) > _reach(best, unit):
                    best = built
                note = "unusable reply" if not best else f"{len(best)} reasons, narrow"
            if _reach(best, unit)[0]:
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if best:
        return best, f"{len(best)} reasons"
    return [], note


def assemble(unit: Unit, results: dict[str, Forecast], plan: Planner) -> tuple[list[dict], dict[str, list[dict]]]:
    rows, claims_by_entity = [], {}
    for entity in unit.entities:
        eid = str(entity["entity_id"])
        fc = results[eid]
        claims = entity_claims(unit, entity, plan.samples[eid], fc.analysis,
                               None if plan.full else plan.passages.get(eid))
        claims_by_entity[eid] = claims
        row: dict = {"entity_id": eid}
        if unit.target_type == "classification":
            row["label"] = fc.label
        if fc.point is not None:
            row["point_forecast"] = fc.point
        row["interval"] = {"level": unit.interval_level, "lo": fc.lo, "hi": fc.hi}
        row["claims"] = claims
        rows.append(row)
    return rows, claims_by_entity
