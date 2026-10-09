"""Read model replies into per-entity samples.

The parser tolerates fences, prose around the JSON, leading ``+`` signs, trailing commas and
numbers written as strings with units. A sample is kept only when its values are usable for the
unit's target type; nothing missing is invented here.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
_PLUS_NUM = re.compile(r"(?<=[:\[,\s])\+(?=\d|\.\d)")
_TRAILING_COMMA = re.compile(r",\s*(?=[}\]])")
_NUM = re.compile(r"[-+−]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+−]?\.\d+")


@dataclass
class Sample:
    entity_id: str
    point: float | None
    lo: float | None
    hi: float | None
    label: str | None
    probability: float | None
    baseline: float | None
    analysis: str
    sigma: float | None = None
    level: float | None = None
    base: float | None = None
    quotes: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, quote)
    weight: float = 1.0
    source: str = ""


def _repair(text: str) -> str:
    text = _FENCE.sub("", text)
    text = _PLUS_NUM.sub("", text)
    text = _TRAILING_COMMA.sub("", text)
    return text.replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null")


def extract_json(raw: str, wanted: tuple[str, ...]) -> dict | None:
    """Return the first JSON object in ``raw`` that carries one of ``wanted`` keys."""
    for text in (raw, _repair(raw)):
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and any(k in obj for k in wanted):
                return obj
    return None


def salvage_objects(raw: str, field: str) -> list[dict]:
    """The complete objects carrying ``field`` in a reply cut off mid-list; an object cut just
    before its closing brace is closed."""
    text = _repair(raw)
    decoder = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        obj, end = None, i + 1
        for candidate in (text, text[:i] + text[i:].rstrip().rstrip(",") + "}"):
            try:
                obj, end = decoder.raw_decode(candidate, i)
                break
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict) and field in obj:
            out.append(obj)
            i = end
        else:
            i += 1
    return out


def number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        m = _NUM.search(value.replace("−", "-"))
        if not m:
            return None
        try:
            v = float(m.group(0).replace(",", "").replace("−", "-"))
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _norm_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"[\s\-]+", "_", text)


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", text).lower())


def match_label(value, labels: list[str]) -> str | None:
    """Map a model label onto the task vocabulary: case, spacing and punctuation insensitive, and
    the label stated before an annotation ("Up (revised higher)"). Never a guess between labels."""
    if not isinstance(value, str) or not labels:
        return None
    norm = _norm_label(value)
    exact = {_norm_label(lab): lab for lab in labels}
    if norm in exact:
        return exact[norm]
    squashed = {_squash(lab): lab for lab in labels}
    if len(squashed) != len(labels):
        return None
    for candidate in (value, re.split(r"[(\[:;,!?]|\s[-–—]\s", value)[0]):
        key = _squash(candidate)
        if key in squashed:
            return squashed[key]
    return None


def _quotes(item: dict) -> list[tuple[str, str]]:
    out = []
    evidence = item.get("evidence")
    for ev in evidence if isinstance(evidence, list) else []:
        if isinstance(ev, dict) and isinstance(ev.get("quote"), str):
            out.append((str(ev.get("doc_id") or ""), ev["quote"]))
    return out


def _id_resolver(entity_ids: set[str]):
    """Exact ids, then case/punctuation-insensitive ids when that mapping is unambiguous."""
    squashed: dict[str, list[str]] = {}
    for eid in entity_ids:
        squashed.setdefault(_squash(eid), []).append(eid)

    def resolve(value) -> str | None:
        if isinstance(value, bool) or value is None:
            return None
        text = str(value).strip() if not isinstance(value, float) or not value.is_integer() else str(int(value))
        if text in entity_ids:
            return text
        hits = squashed.get(_squash(text), [])
        return hits[0] if len(hits) == 1 else None

    return resolve


def to_sample(item: dict, *, target_type: str, labels: list[str], entity_ids: set[str],
              weight: float, source: str, resolve=None, only: str | None = None) -> Sample | None:
    """``only``: the one entity a single-entity prompt asked for (its echoed id is not required)."""
    eid = only or (resolve or _id_resolver(entity_ids))(item.get("entity_id"))
    if eid is None or eid not in entity_ids:
        return None
    point = number(item.get("point_forecast"))
    interval = item.get("interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        interval = {"lo": interval[0], "hi": interval[1]}
    interval = interval if isinstance(interval, dict) else {}
    lo = number(item.get("lo", interval.get("lo")))
    hi = number(item.get("hi", interval.get("hi")))
    if point is None and lo is not None and hi is not None and target_type in ("regression", "ranking"):
        point = (lo + hi) / 2  # the model's own band, centred
    label = match_label(item.get("label"), labels) if labels else None
    prob = number(item.get("probability"))
    if prob is not None and not 0.0 <= prob <= 1.0:
        prob = prob / 100.0 if 1.0 < prob <= 100.0 else None
    if target_type in ("regression", "ranking") and point is None:
        return None
    if target_type == "classification" and label is None:
        return None
    if target_type == "classification" and (lo is None or hi is None):
        # A pure-label task defines no numeric quantity: the band is required but unscored, so the
        # label's probability stands as the number with the uninformative [0, 1] band.
        if point is None and prob is not None:
            point = prob
        if point is None or 0.0 <= point <= 1.0:
            lo, hi = 0.0, 1.0
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    sigma = number(item.get("sigma"))
    if sigma is not None and sigma > 0 and point is not None:
        # The model's own sigma: its band is at least +/-1.645 sigma, and an edge beyond 5 sigma
        # (a dropped digit, a units slip) is replaced by the sigma-implied edge.
        half, far = 1.6448536269514722 * sigma, 5.0 * sigma
        lo = point - half if lo is None or lo < point - far else min(lo, point - half)
        hi = point + half if hi is None or hi > point + far else max(hi, point + half)
    return Sample(
        entity_id=eid, point=point, lo=lo, hi=hi, label=label, probability=prob,
        baseline=number(item.get("baseline")), analysis=str(item.get("analysis") or "")[:1200],
        quotes=_quotes(item), weight=weight, source=source, sigma=sigma,
        level=number(item.get("level_forecast")), base=number(item.get("base_value")),
    )


def _row_objects(raw: str) -> list[dict]:
    """Every complete object carrying an entity_id, e.g. the rows of a reply cut off mid-array."""
    out: list[dict] = []
    for text in (raw, _repair(raw)):
        decoder = json.JSONDecoder()
        i = 0
        while i < len(text):
            if text[i] == "{":
                try:
                    obj, end = decoder.raw_decode(text, i)
                except json.JSONDecodeError:
                    i += 1
                    continue
                if isinstance(obj, dict) and "entity_id" in obj:
                    out.append(obj)
                    i = end
                    continue
            i += 1
        if out:
            return out
    return out


def parse_predictions(raw: str, *, target_type: str, labels: list[str], entity_ids: set[str],
                      weight: float, source: str) -> list[Sample]:
    obj = extract_json(raw, ("predictions",))
    preds = obj.get("predictions") if obj is not None else None
    if isinstance(preds, list):
        items = preds
    elif isinstance(preds, dict):  # keyed by entity id
        items = [dict(v, entity_id=v.get("entity_id", k)) for k, v in preds.items() if isinstance(v, dict)]
    else:
        items = _row_objects(raw)
        if not items and len(entity_ids) == 1:
            single = extract_json(raw, ("point_forecast", "label", "lo", "interval"))
            items = [single] if single is not None else []
    resolve = _id_resolver(entity_ids)
    only = next(iter(entity_ids)) if len(entity_ids) == 1 and len(items) == 1 else None
    out: list[Sample] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        s = to_sample(item, target_type=target_type, labels=labels, entity_ids=entity_ids,
                      weight=weight, source=source, resolve=resolve, only=only)
        if s is not None and s.entity_id not in seen:
            seen.add(s.entity_id)
            out.append(s)
    return out
