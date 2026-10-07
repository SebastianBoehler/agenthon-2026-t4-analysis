"""Cohort-level decisions for binary event labels (for example credit_event / no_event).

Asked one entity at a time, the model anchors event probabilities low; shown the whole cohort, it
states how many entities it expects to have the event. The event label goes to that many
entities, ranked by their pooled event probability (anything at or above one half always gets
it). Rosters whose evidence shows few events yield a small count, so the rule adapts to the cohort.
"""
from __future__ import annotations

import re
import statistics

from .aggregate import Forecast
from .parse import Sample, extract_json, number

COHORT_CALLS = 3
_NEGATED = re.compile(r"^(?:no|non|not|none)(?:[_\-\s]|$)", re.IGNORECASE)


def event_labels(labels: list[str]) -> tuple[str, str] | None:
    """(event, non_event) when the vocabulary is an event and its negation, else None."""
    if len(labels) != 2:
        return None
    negated = [lab for lab in labels if _NEGATED.match(lab)]
    if len(negated) != 1:
        return None
    event = next(lab for lab in labels if lab != negated[0])
    return event, negated[0]


def question(head: str, event: str, n: int) -> str:
    return (
        head.rstrip()
        + "\n\nCOHORT QUESTION\nConsider all entities above together and the evidence for each. "
        f"Estimate how many of the {n} entities will meet the label '{event}' as defined above, "
        "then rank all entities from most to least likely to meet it.\n"
        'OUTPUT: exactly one JSON object {"expected_count": <number>, "ranking": ["<entity_id>", ...], '
        '"why": "<at most 80 words>"}'
    )


def parse(raw: str, ids: set[str]) -> tuple[float | None, list[str]]:
    obj = extract_json(raw, ("expected_count", "ranking"))
    if obj is None:
        return None, []
    count = number(obj.get("expected_count"))
    ranking = [e for e in obj.get("ranking") or [] if isinstance(e, str) and e in ids]
    return count, list(dict.fromkeys(ranking))


def event_probability(samples: list[Sample], event: str, probability_target: bool) -> float:
    vals = []
    for s in samples:
        if probability_target and s.point is not None and 0.0 <= s.point <= 1.0:
            vals.append(s.point)
        elif s.probability is not None:
            vals.append(s.probability if s.label == event else 1.0 - s.probability)
        else:
            vals.append(1.0 if s.label == event else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def relabel(results: dict[str, Forecast], samples: dict[str, list[Sample]], event: str, non_event: str,
            counts: list[float], rankings: list[list[str]], probability_target: bool) -> str:
    """Give the event label to the k most likely entities; return a short trace note."""
    if not counts:
        return "cohort: no count"
    ids = list(results)
    p = {eid: event_probability(samples[eid], event, probability_target) for eid in ids}
    position = {eid: statistics.mean([r.index(eid) if eid in r else len(ids) for r in rankings] or [len(ids)])
                for eid in ids}
    k = int(round(statistics.median(counts)))
    k = max(0, min(len(ids), k))
    k = max(k, sum(1 for eid in ids if p[eid] >= 0.5))
    order = sorted(ids, key=lambda eid: (-p[eid], position[eid]))
    chosen = set(order[:k])
    for eid in ids:
        results[eid].label = event if eid in chosen else non_event
    return f"cohort: k={k} from counts {counts}"
