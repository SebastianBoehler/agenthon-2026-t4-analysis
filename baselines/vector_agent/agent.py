"""Orchestrate House calls for one unit and assemble the answer.

Small corpora go to the model whole, with every entity in one prompt, sampled several times.
Large corpora get one prompt per entity with that entity's retrieved excerpts, plus one compact
unit-wide prompt for cross-sectional context. Calls run concurrently under a request cap and a
wall-clock schedule; every forecast in the answer comes from model samples.
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import prompts
from .aggregate import Forecast, combine
from .claims import build_reasons, entity_claims, reason_candidates
from .evidence import Passage, entity_passages
from .house import Budget, BudgetExhausted, Chat
from .parse import Sample, extract_json, parse_predictions
from .unit import Unit

FULL_CONTEXT_CHARS = 110_000
ENTITY_EVIDENCE_CHARS = 36_000
QUICK_EVIDENCE_CHARS = 48_000
UNIT_SAMPLES = 5
MAX_ENTITY_SAMPLES = 4
GROUP_MAX_FULL = 12
MAX_RETRIEVAL_GROUPS = 10
REASONS_RESERVE = 1


@dataclass
class Schedule:
    """Seconds after process start. The unit allows 600 s including container start-up."""
    launch_stop: float = 300.0
    hard_stop: float = 440.0
    reasons_latest: float = 380.0
    reasons_stop: float = 480.0
    request_timeout: float = 240.0
    workers: int = 4


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


class Planner:
    """Decides the next House call. Work is organised in groups of entities that share one prompt.

    Full-context mode: groups of at most 12 entities over the whole corpus, sampled until every
    entity has UNIT_SAMPLES forecasts. Retrieval mode: one group per entity (excerpts for that
    entity) when the roster has at most 10 rows, otherwise at most 10 groups of several entities,
    sampled in rounds; spare budget then goes to the entities whose samples disagree most.
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
        n = max(1, len(self.ids))
        if self.full:
            self.groups = _chunks(self.ids, GROUP_MAX_FULL)
            docs = prompts.unit_docs_full(unit)
            focus = len(self.groups) > 1
            self.group_prompts = [prompts.unit_prompt(unit, docs, g if focus else None) for g in self.groups]
            self.target = UNIT_SAMPLES
            return
        for e in unit.entities:
            self.passages[str(e["entity_id"])] = entity_passages(unit, e, ENTITY_EVIDENCE_CHARS)
        self.groups = _chunks(self.ids, -(-n // MAX_RETRIEVAL_GROUPS))
        if len(self.groups[0]) == 1:
            self.group_prompts = [self._entity_prompt(g[0]) for g in self.groups]
            if n <= GROUP_MAX_FULL:  # one compact unit-wide pass for cross-sectional context
                share = max(2500, QUICK_EVIDENCE_CHARS // n)
                quick = {eid: _trim(ps, share) for eid, ps in self.passages.items()}
                self.unit_prompt = prompts.unit_prompt(unit, prompts.unit_docs_excerpts(unit, quick))
        else:
            self.group_prompts = []
            for g in self.groups:
                share = max(2500, QUICK_EVIDENCE_CHARS // len(g))
                excerpts = {eid: _trim(self.passages[eid], share) for eid in g}
                self.group_prompts.append(prompts.unit_prompt(unit, prompts.unit_docs_excerpts(unit, excerpts), g))
        spare = 24 - REASONS_RESERVE - 2 - (1 if self.unit_prompt else 0)
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
        tokens = 1500 if len(ids) == 1 else min(4000, 700 + 330 * len(ids))
        return Job(kind, key, tuple(ids), prompt, 0.0 if k == 0 else 0.7,
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
        """An entity that group replies keep missing is asked for on its own."""
        for idx, g in enumerate(self.groups):
            if self.inflight[f"g{idx}"]:
                continue
            for eid in g:
                if len(self.samples[eid]) < 2 and self.inflight[eid] == 0 and self.attempts[eid] < 3 \
                        and (len(g) > 1 or self.attempts[f"g{idx}"] >= self.target + 2):
                    return self._job("entity", eid, [eid], self._entity_prompt(eid))
        return None

    def next_job(self, budget_left: int) -> Job | None:
        if self.full:
            order = sorted(range(len(self.groups)), key=lambda i: min(len(self.samples[e]) for e in self.groups[i])
                           + self.inflight[f"g{i}"])
            for idx in order:
                key, g = f"g{idx}", self.groups[idx]
                floor = min(len(self.samples[eid]) for eid in g)
                if floor + self.inflight[key] < self.target and self.attempts[key] < self.target + 4:
                    return self._job("unit", key, g, self.group_prompts[idx])
            return self._rescue()
        # Retrieval mode: rounds over the groups (plus the unit-wide pass), then rescue, then depth.
        for depth in range(1, self.target + 1):
            for idx, g in enumerate(self.groups):
                key = f"g{idx}"
                if self.done[key] + self.inflight[key] >= depth or self.attempts[key] >= self.target + 2:
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

    def mark_launched(self, job: Job) -> None:
        self.attempts[job.key] += 1
        self.inflight[job.key] += 1
        self.trace.launched[job.kind] += 1

    def absorb(self, job: Job, raw: str | None, error: str | None) -> None:
        self.inflight[job.key] -= 1
        if raw is None:
            self.trace.failures.append(f"{job.key}: {error}")
            return
        got = parse_predictions(raw, target_type=self.unit.target_type, labels=self.unit.labels,
                                entity_ids=set(job.ids), weight=job.weight, source=job.kind)
        got = [s for s in got if s.lo is not None and s.hi is not None]
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


def _trim(passages, budget: int):
    out, used = [], 0
    for p in passages:
        if used + len(p.text) > budget and out:
            continue
        out.append(p)
        used += len(p.text)
    return out


def _call(chat: Chat, job: Job, timeout: float) -> str:
    return chat.complete(prompts.SYSTEM, job.prompt, temperature=job.temperature, seed=job.seed,
                         max_tokens=job.max_tokens, timeout=timeout)


def forecast(unit: Unit, chat: Chat, budget: Budget, start: float, schedule: Schedule,
             seed: int = 20261006) -> tuple[dict[str, Forecast], Planner]:
    plan = Planner(unit, seed)
    pool = ThreadPoolExecutor(max_workers=schedule.workers)
    pending: dict[Future, Job] = {}
    try:
        while True:
            now = time.monotonic() - start
            while (len(pending) < schedule.workers and now < schedule.launch_stop
                   and budget.remaining > REASONS_RESERVE):
                job = plan.next_job(budget.remaining)
                if job is None:
                    break
                plan.mark_launched(job)
                timeout = min(schedule.request_timeout, schedule.hard_stop - now)
                pending[pool.submit(_call, chat, job, timeout)] = job
            if not pending:
                break
            done, _ = wait(pending, timeout=max(0.05, schedule.hard_stop - now),
                           return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                job = pending.pop(fut)
                try:
                    plan.absorb(job, fut.result(), None)
                except BudgetExhausted as exc:
                    plan.absorb(job, None, str(exc))
                except Exception as exc:  # noqa: BLE001 - any call failure is recorded, not fatal
                    plan.absorb(job, None, f"{type(exc).__name__}: {exc}"[:200])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    results = {}
    bounded = _probability_target(unit, plan.samples)
    for eid in plan.ids:
        fc = combine(plan.samples[eid], unit.target_type, unit_interval=bounded)
        if fc is not None:
            results[eid] = fc
    return results, plan


def reasons(unit: Unit, chat: Chat, budget: Budget, start: float, schedule: Schedule,
            rows: list[dict], claims_by_entity: dict[str, list[dict]], seed: int) -> tuple[list[dict], str]:
    now = time.monotonic() - start
    if now > schedule.reasons_latest or budget.remaining < 1:
        return [], "skipped (time or budget)"
    candidates = reason_candidates(unit, claims_by_entity)
    if not candidates:
        return [], "skipped (no candidate premises)"
    declared = []
    for row in rows:
        view = {"entity_id": row["entity_id"]}
        for key in ("label", "point_forecast"):
            if key in row:
                view[key] = row[key]
        view["interval"] = [row["interval"]["lo"], row["interval"]["hi"]]
        declared.append(view)
    text = prompts.reasons_prompt(unit, declared, [(k, c["claim"]) for k, c in candidates])
    note = "not attempted"
    for attempt in range(2):
        now = time.monotonic() - start
        timeout = min(120.0, schedule.reasons_stop - now)
        if timeout < 20 or budget.remaining < 1 or (attempt and now > schedule.reasons_latest + 40):
            break
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(chat.complete, prompts.SYSTEM, text, temperature=0.0 if attempt == 0 else 0.5,
                          seed=seed + attempt, max_tokens=1500, timeout=timeout)
        try:
            raw = fut.result(timeout=timeout + 2)
        except Exception as exc:  # noqa: BLE001 - reasons are optional; never fail the unit for them
            note = f"failed ({type(exc).__name__})"
            continue
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        built = build_reasons(unit, extract_json(raw, ("reasons",)), candidates)
        if built:
            return built, f"{len(built)} reasons"
        note = "unusable reply"
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
