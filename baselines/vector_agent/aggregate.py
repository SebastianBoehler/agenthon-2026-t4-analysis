"""Combine sampled House forecasts per entity: weighted medians for numbers, a weighted vote for labels."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .parse import Sample


@dataclass
class Forecast:
    entity_id: str
    label: str | None
    point: float | None
    lo: float
    hi: float
    analysis: str
    samples: int


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= total / 2:
            # Average with the next value on an exact half split, like an ordinary median.
            if abs(acc - total / 2) < 1e-12 and i + 1 < len(pairs):
                return (v + pairs[i + 1][0]) / 2
            return v
    return pairs[-1][0]


_Z90 = 1.6448536269514722  # 95th percentile of the standard normal


def _split_normal_cdf(x: float, centre: float, left: float, right: float) -> float:
    if x <= centre:
        return 0.5 * math.erfc((centre - x) / (left * math.sqrt(2))) if left > 0 else 0.0
    return 1 - 0.5 * math.erfc((x - centre) / (right * math.sqrt(2))) if right > 0 else 1.0


def pooled_interval(samples: list[Sample]) -> tuple[float, float]:
    """5th and 95th percentiles of the weighted mixture of the samples' forecast distributions.

    Each sample's (lo, point, hi) is read as a split normal whose 5% and 95% points are lo and hi,
    so the pool keeps each sample's own band and widens it where samples disagree.
    """
    def centre_of(s: Sample) -> float:
        return s.point if s.point is not None and s.lo <= s.point <= s.hi else (s.lo + s.hi) / 2

    if len(samples) >= 3:
        # Leave out samples centred far outside the consensus band (e.g. a units slip).
        lo_m = weighted_median([(s.lo, s.weight) for s in samples])
        hi_m = weighted_median([(s.hi, s.weight) for s in samples])
        width = max(hi_m - lo_m, 0.0)
        kept = [s for s in samples if lo_m - width <= centre_of(s) <= hi_m + width]
        if 2 * len(kept) > len(samples):
            samples = kept
    comps = []
    for s in samples:
        centre = centre_of(s)
        comps.append((s.weight, centre, (centre - s.lo) / _Z90, (s.hi - centre) / _Z90))
    if len(comps) >= 3:
        # One sample's runaway width (a units slip in its sigma) must not set the pooled tails.
        halves = sorted(h for _, _, a, b in comps for h in (a, b))
        cap = 3.0 * halves[len(halves) // 2]
        if cap > 0:
            comps = [(w, c, min(a, cap), min(b, cap)) for w, c, a, b in comps]
    if len(comps) == 1:
        return samples[0].lo, samples[0].hi
    total = sum(w for w, *_ in comps)

    def cdf(x: float) -> float:
        return sum(w * _split_normal_cdf(x, c, a, b) for w, c, a, b in comps) / total

    def quantile(q: float) -> float:
        a = min(c - 4 * l for _, c, l, _ in comps) - 1e-9
        b = max(c + 4 * r for _, c, _, r in comps) + 1e-9
        for _ in range(80):
            m = (a + b) / 2
            if cdf(m) < q:
                a = m
            else:
                b = m
        return (a + b) / 2

    # A mixture's 5% and 95% points lie within its components' own: clamping only removes
    # bisection noise.
    lo_bound = min(c - _Z90 * l for _, c, l, _ in comps)
    hi_bound = max(c + _Z90 * r for _, c, _, r in comps)
    return max(lo_bound, quantile(0.05)), min(hi_bound, quantile(0.95))


def _round(x: float) -> float:
    return float(f"{x:.6g}")


def binary_outcome_band(p: float, level: float = 0.90) -> tuple[float, float]:
    """Probability-anchored band heuristic under an assumed binary numeric outcome.

    Interior probabilities use [0, 1], the raw interval-score optimum for Bernoulli outcomes.
    Extreme bands retain the forecast probability; this containment choice is not an official
    requirement or the unconstrained optimum. Official predictive gain remains unmeasured.
    """
    p = min(1.0, max(0.0, p))
    edge = (1.0 - level) / 2.0
    if p < edge:
        return 0.0, p
    if p > 1.0 - edge:
        return p, 1.0
    return 0.0, 1.0


def combine(samples: list[Sample], target_type: str, unit_interval: bool = False) -> Forecast | None:
    usable = [s for s in samples if s.lo is not None and s.hi is not None]
    if not usable:
        return None
    label = None
    if target_type == "classification":
        votes: dict[str, float] = defaultdict(float)
        conf: dict[str, list[float]] = defaultdict(list)
        for s in usable:
            votes[s.label] += s.weight
            if s.probability is not None:
                conf[s.label].append(s.probability)
        label = max(votes, key=lambda lab: (votes[lab], sum(conf[lab]) / len(conf[lab]) if conf[lab] else 0.0))
        agreeing = [s for s in usable if s.label == label]
        usable = agreeing or usable
    with_point = [s for s in usable if s.point is not None]
    point = weighted_median([(s.point, s.weight) for s in with_point]) if with_point else None
    lo, hi = pooled_interval(usable)
    if lo > hi:
        lo, hi = hi, lo
    if point is not None:
        lo, hi = min(lo, point), max(hi, point)
    if unit_interval:
        lo, hi = max(0.0, lo), min(1.0, hi)
        if point is not None:
            lo, hi = binary_outcome_band(point)
    if hi <= lo:
        pad = max(abs(lo) * 1e-3, 1e-6)
        lo, hi = lo - pad, hi + pad
        if unit_interval:
            lo, hi = max(0.0, lo), min(1.0, hi)
    best = max(usable, key=lambda s: (s.weight, len(s.analysis)))
    return Forecast(
        entity_id=usable[0].entity_id,
        label=label,
        point=None if point is None else _round(point),
        lo=_round(lo),
        hi=_round(hi),
        analysis=best.analysis,
        samples=len(samples),
    )
