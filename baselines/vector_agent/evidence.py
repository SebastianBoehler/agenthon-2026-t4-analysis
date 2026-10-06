"""Evidence selection: whole small documents, BM25 passages from large ones, exact-offset segments.

Every passage and segment is a slice ``text[start:end]`` of an admitted document, so anything
quoted from it can be cited with exact offsets.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .unit import Doc, Unit

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    """a an and are as at be been by for from has have in into is it its of on or that the this
    to was were will with each which their than then there these those using only your you do not
    predict provide give support every any all per after before between over under via our we
    frozen corpus evidence cutoff citations cite passages documents document filings filing
    available absent design rely post pre forecast forecasts point_forecast interval reason
    table given expressed relative same one year earlier reported market sec most recent
    claims claim based point""".split()
)
# Passages that carry results, guidance, solvency and driver disclosures in financial filings.
_ANCHORS = [
    (re.compile(r"diluted (?:earnings|net income|eps)|earnings per (?:common |diluted )?share|per diluted share", re.I), 2.0),
    (re.compile(r"net income (?:applicable|attributable|was|of|for|increased|decreased)|net (?:loss|income) of \$", re.I), 1.0),
    (re.compile(r"\b(?:guidance|outlook|we expect|we anticipate|expects? (?:to|that)|full[- ]year|second half|next quarter)\b", re.I), 1.5),
    (re.compile(r"going concern|substantial doubt|event of default|forbearance|waiver|chapter 11|bankruptcy|restructuring support|distressed", re.I), 2.0),
    (re.compile(r"liquidity and capital resources|covenant|maturit(?:y|ies) of (?:our )?debt|borrowing capacity|cash and cash equivalents of", re.I), 1.0),
    (re.compile(r"results of operations|executive summary|financial highlights|overview of (?:the )?(?:quarter|results)", re.I), 1.0),
    (re.compile(r"net interest income|provision for credit losses|net charge-offs|investment banking fees|trading revenue", re.I), 1.0),
    (re.compile(r"(?:revenue|net sales|net revenue)s? (?:was|were|increased|decreased|grew|declined)", re.I), 1.0),
]
# Generic financial-analysis vocabulary that marks result, outlook and risk passages.
_BOOST = (
    "diluted earnings per share net income guidance outlook expect expects anticipate "
    "quarter compared increase decrease growth decline revenue margin liquidity going concern "
    "substantial doubt default covenant maturities debt bankruptcy restructuring impairment "
    "provision credit losses net interest income fee trading investment banking expenses"
)

PASSAGE_CHARS = 1600
PASSAGE_OVERLAP = 240
SEGMENT_MAX_CHARS = 380
SMALL_DOC_CHARS = 14000


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class Passage:
    doc_id: str
    start: int
    end: int
    text: str


def split_passages(doc: Doc, size: int = PASSAGE_CHARS, overlap: int = PASSAGE_OVERLAP) -> list[Passage]:
    text = doc.text
    out: list[Passage] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            lower = start + size // 2
            cut = max(text.rfind("\n", lower, end), text.rfind(". ", lower, end))
            if cut < 0:
                cut = text.rfind(" ", lower, end)
            if cut >= lower:
                end = cut + 1
        out.append(Passage(doc.doc_id, start, end, text[start:end]))
        if end >= len(text):
            break
        nxt = max(start + 1, end - overlap)
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        start = nxt
    return out


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.tfs = [Counter(tokens(p.text)) for p in passages]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: Counter) -> list[float]:
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for term, weight in query.items():
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1.0))
                s += weight * self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def anchor_score(text: str) -> float:
    """Saturating credit for each kind of results/guidance/solvency anchor a passage carries."""
    total = 0.0
    for pattern, weight in _ANCHORS:
        hits = len(pattern.findall(text))
        if hits:
            total += weight * min(1.0, 0.5 + 0.25 * hits) * 0.5
    return total


def entity_terms(entity: dict) -> list[str]:
    parts = [str(v) for k, v in entity.items() if isinstance(v, str) and k not in ("corpus_ref",)]
    return tokens(" ".join(parts))


def task_query(unit: Unit, entity: dict | None) -> Counter:
    q: Counter = Counter()
    target = unit.target
    for t in tokens(str(unit.task.get("prompt", ""))):
        q[t] += 1.0
    for t in tokens(str(target.get("name", "")).replace("_", " ")):
        q[t] += 1.5
    for text in (target.get("label_assertions") or {}).values():
        for t in tokens(str(text)):
            q[t] += 0.5
    for t in tokens(_BOOST):
        q[t] += 0.6
    if entity is not None:
        for t in entity_terms(entity):
            q[t] += 1.0
    return q


def entity_passages(unit: Unit, entity: dict, budget_chars: int) -> list[Passage]:
    """Whole small documents first, then the best BM25 passages of large ones, within budget."""
    eid = str(entity["entity_id"])
    docs = unit.docs_for(eid)
    chosen: list[Passage] = []
    used = 0
    small = sorted((d for d in docs if len(d.text) <= SMALL_DOC_CHARS), key=lambda d: d.doc_date, reverse=True)
    large = [d for d in docs if len(d.text) > SMALL_DOC_CHARS]
    # Entity-labelled small documents outrank shared ones.
    small.sort(key=lambda d: d.entity_ids is None)
    for d in small:
        if used + len(d.text) > budget_chars * 0.6 and chosen:
            continue
        chosen.append(Passage(d.doc_id, 0, len(d.text), d.text))
        used += len(d.text)
    if large and used < budget_chars:
        pool = [p for d in large for p in split_passages(d)]
        bm = BM25(pool)
        raw = bm.scores(task_query(unit, entity))
        top = max(raw) if raw and max(raw) > 0 else 1.0
        scores = [r / top + anchor_score(p.text) - 0.8 * len(_BOILER.findall(p.text))
                  - 0.35 * len(_RISK_TALK.findall(p.text)) for r, p in zip(raw, pool)]
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        picked: list[Passage] = []
        for i in order:
            p = pool[i]
            if scores[i] <= 0 or used + len(p.text) > budget_chars:
                continue
            if any(q.doc_id == p.doc_id and q.start < p.end and p.start < q.end for q in picked):
                continue  # skip overlapping windows
            picked.append(p)
            used += len(p.text)
        picked.sort(key=lambda p: (p.doc_id, p.start))
        chosen.extend(picked)
    return chosen


# --------------------------------------------------------------------------- segments
_SENT_END = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\-•])")


def _bounded(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Trim whitespace and cut a range into word-bounded pieces of at most SEGMENT_MAX_CHARS."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    out = []
    while end - start > SEGMENT_MAX_CHARS:
        cut = text.rfind(" ", start + SEGMENT_MAX_CHARS // 2, start + SEGMENT_MAX_CHARS)
        if cut <= start:
            cut = start + SEGMENT_MAX_CHARS
        out.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if end > start:
        out.append((start, end))
    return out


def segments(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Line or sentence segments of ``text[start:end]`` with exact offsets."""
    end = len(text) if end is None else end
    out: list[tuple[int, int]] = []
    chunk = text[start:end]
    if "\n" in chunk:
        pos = start
        for line in chunk.split("\n"):
            out.extend(_bounded(text, pos, pos + len(line)))
            pos += len(line) + 1
    else:
        pos = start
        for m in _SENT_END.finditer(chunk):
            out.extend(_bounded(text, pos, start + m.start()))
            pos = start + m.end()
        out.extend(_bounded(text, pos, end))
    return [(s, e) for s, e in out if e - s >= 25]


_DIGIT = re.compile(r"\d")
_BOILER = re.compile(
    r"securities and exchange commission|/s/|exhibit \d|incorporated by reference|"
    r"pursuant to section|signatures?\b|^source:|should be read in conjunction|forward-looking statements",
    re.IGNORECASE,
)
_RISK_TALK = re.compile(
    r"adversely affect|materially adverse|no assurance|may not be able|could harm|could be harmed|"
    r"there can be no|we cannot assure|could negatively|may be unable",
    re.IGNORECASE,
)


def best_segments(
    doc: Doc, entity: dict, query: Counter, hint: str, k: int,
    ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Rank segments by task relevance, entity mentions, the model's rationale and figures."""
    ent = set(entity_terms(entity))
    hint_terms = set(tokens(hint))
    spans = []
    for s, e in ranges or [(0, len(doc.text))]:
        spans.extend(segments(doc.text, s, e))
    scored = []
    for s, e in spans:
        text = doc.text[s:e]
        seg_terms = set(tokens(text))
        rel = sum(query.get(t, 0.0) for t in seg_terms) / (1.0 + len(seg_terms) ** 0.5)
        score = rel + 2.0 * len(seg_terms & ent) + 1.0 * len(seg_terms & hint_terms)
        if _DIGIT.search(text):
            score += 1.0
        if _BOILER.search(text):
            score -= 6.0
        scored.append((score, s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, e) for _, s, e in scored[:k]]
