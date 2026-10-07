"""Verbatim claims and reasons with exact offsets into admitted pre-cutoff documents.

Every claim text is exactly ``doc.text[span_start:span_end]`` of a document the manifest admits
for the entity, a few hundred characters long. Exact quotation establishes source identity. Official semantic judging and
forecast quality are separate checks; local quote validation does not certify either.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

from .evidence import SMALL_DOC_CHARS, Passage, best_segments, segments, task_query, tokens
from .parse import Sample
from .unit import Doc, Unit, task_table_text

CLAIMS_PER_ENTITY = 2
MAX_CLAIM_CHARS = 399  # judge tokens never exceed NFKC characters, so 400 tokens cannot be reached
DENY = (
    "leaderboard", "canary", "/home/", "units/", "reference/", "outcome.json", "team_id",
    "team name", "participant_id", "participant name", "submission_id", "other submission",
)
REASON_BYTES_CAP = 6000
EVIDENCE_BYTES_CAP = 40000


@lru_cache(maxsize=64)
def _segments_of(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(segments(text))


def _enclosing(doc: Doc, start: int, end: int) -> tuple[int, int]:
    for s, e in _segments_of(doc.text):
        if s <= start < e:
            if end <= e or e - s >= 200:
                return s, e
    return start, end


def locate(doc: Doc, quote: str) -> tuple[int, int] | None:
    """Find ``quote`` in the document allowing only whitespace and typography differences."""
    quote = unicodedata.normalize("NFKC", quote).strip().strip('"').strip()
    words = re.findall(r"\S+", quote)
    if len(words) < 4:
        return None
    i = doc.text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, unicodedata.normalize("NFKC", doc.text)) if len(pattern) < 4000 else None
    if m and len(unicodedata.normalize("NFKC", doc.text)) == len(doc.text):
        return m.start(), m.end()
    return None


def _substantive(text: str) -> bool:
    """A figure, or several content words: never a content-free quotation."""
    return bool(re.search(r"\d", text)) or len(tokens(text)) >= 6


def _claim(doc_id: str, text: str, s: int, e: int) -> dict:
    return {"doc_id": doc_id, "span_start": s, "span_end": e, "claim": text[s:e]}


def entity_claims(unit: Unit, entity: dict, samples: list[Sample], hint: str,
                  passages: list[Passage] | None = None) -> list[dict]:
    """One or two verbatim claims; ``passages`` (retrieved excerpts) bound where large documents are read."""
    eid = str(entity["entity_id"])
    docs = [d for d in unit.docs_for(eid) if d.citable]
    query = task_query(unit, entity)
    ranges: dict[str, list[tuple[int, int]]] = {}
    for p in passages or []:
        ranges.setdefault(p.doc_id, []).append((p.start, p.end))
    claims: list[dict] = []
    seen: set[tuple[str, int, int]] = set()

    def add(doc: Doc, s: int, e: int) -> None:
        key = (doc.doc_id, s, e)
        if key in seen or any(k[0] == doc.doc_id and k[1] < e and s < k[2] for k in seen):
            return
        if not 25 <= e - s <= MAX_CLAIM_CHARS or not _substantive(doc.text[s:e]):
            return
        seen.add(key)
        claims.append(_claim(doc.doc_id, doc.text, s, e))

    # 1. The model's own quotations, when they are verbatim in an admitted document.
    for sample in samples:
        for doc_id, quote in sample.quotes:
            if len(claims) >= 1:
                break
            ordered = sorted(docs, key=lambda d: d.doc_id != doc_id)
            for doc in ordered:
                hit = locate(doc, quote)
                if hit:
                    add(doc, *_enclosing(doc, *hit))
                    break
    # 2. The best-matching segments of the entity's own documents, then shared ones.
    ordered_docs = sorted(docs, key=lambda d: (d.entity_ids is None, d.doc_id not in ranges))
    for doc in ordered_docs:
        if len(claims) >= CLAIMS_PER_ENTITY:
            break
        if passages is not None and len(doc.text) > SMALL_DOC_CHARS and doc.doc_id not in ranges:
            continue  # a large document is read only where retrieval looked
        for s, e in best_segments(doc, entity, query, hint, 2, ranges.get(doc.doc_id)):
            if len(claims) >= CLAIMS_PER_ENTITY:
                break
            add(doc, s, e)
    # 3. No admitted document: quote the entity's own row of the task table.
    if not claims:
        text, ranges = task_table_text(unit.task)
        s, e = ranges[eid]
        claims.append({"doc_id": "task", "span_start": s, "span_end": e, "claim": text[s:e]})
    return claims


REASON_CANDIDATES = 30


def reason_candidates(unit: Unit, claims_by_entity: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    """Premise candidates: each entity's claims, then the passages of the smaller documents that
    best match the drivers the task names (cross-entity premises for broad reasons)."""
    out: list[tuple[str, dict]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(c: dict) -> None:
        key = (c["doc_id"], c["span_start"], c["span_end"])
        if key in seen or any(k[0] == key[0] and k[1] < key[2] and key[1] < k[2] for k in seen):
            return
        seen.add(key)
        out.append((f"R{len(out) + 1}", c))

    for eid in unit.entity_ids:
        for c in claims_by_entity.get(eid, []):
            if c["doc_id"] != "task" and unit.docs[c["doc_id"]].citable and len(out) < REASON_CANDIDATES - 6:
                add(c)
    query = task_query(unit, None)
    for doc in sorted(unit.docs.values(), key=lambda d: (d.entity_ids is not None, d.doc_id)):
        if not doc.citable or len(doc.text) > 3 * SMALL_DOC_CHARS:
            continue
        for st, en in best_segments(doc, {}, query, "", 2):
            if len(out) >= REASON_CANDIDATES:
                return out
            if _substantive(doc.text[st:en]):
                add(_claim(doc.doc_id, doc.text, st, en))
    return out


_KEY_REF = re.compile(r"\s*[\(\[]\s*R\d+(?:\s*[,;/&]\s*R\d+)*\s*[\)\]]")


def _clean(text: object, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    text = _KEY_REF.sub("", text)  # candidate keys mean nothing to the reader
    text = " ".join(text.split())
    return text[:limit].rstrip()


def _denied(text: str) -> bool:
    low = text.lower()
    return any(tok in low for tok in DENY) or "://" in low


MAX_SUPPORT = 3


def _key_of(value: object) -> str:
    m = re.match(r"\s*\[?\s*(R\d+)\b", str(value or ""))
    return m.group(1) if m else ""


def _within_caps(reasons: list[dict], texts: dict[tuple[str, int, int], str]) -> bool:
    judged = [{k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")} for r in reasons]
    evidence = [{"reason_id": r["reason_id"], "doc_id": c["doc_id"], "span_start": c["span_start"],
                 "span_end": c["span_end"], "text": texts[(c["doc_id"], c["span_start"], c["span_end"])]}
                for r in reasons for c in r["citations"]]
    return (len(json.dumps(judged, ensure_ascii=False, separators=(",", ":")).encode()) <= REASON_BYTES_CAP
            and len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()) <= EVIDENCE_BYTES_CAP)


def build_reasons(unit: Unit, reply: dict | None, candidates: list[tuple[str, dict]]) -> list[dict]:
    """Turn the model's driver reasons into schema-valid reasons: a verbatim premise, cited, plus
    up to three supporting passages cited alongside it (the judge reads every cited passage)."""
    if not isinstance(reply, dict) or not isinstance(reply.get("reasons"), list):
        return []
    by_key = dict(candidates)
    roster = set(unit.entity_ids)
    texts = {(c["doc_id"], c["span_start"], c["span_end"]): c["claim"] for _, c in candidates}
    reasons: list[dict] = []
    used: set[str] = set()
    for item in reply["reasons"]:
        if not isinstance(item, dict) or len(reasons) >= 3:
            continue
        key = _key_of(item.get("premise"))
        cand = by_key.get(key)
        mechanism = _clean(item.get("mechanism"), 600)
        implication = _clean(item.get("answer_implication"), 600)
        if cand is None or key in used or not mechanism or not implication:
            continue
        if _denied(mechanism) or _denied(implication):
            continue
        used.add(key)
        citations = [{"doc_id": cand["doc_id"], "span_start": cand["span_start"], "span_end": cand["span_end"]}]
        for extra in (item.get("support") or [])[:MAX_SUPPORT]:
            sup = by_key.get(_key_of(extra))
            if sup is None:
                continue
            cite = {"doc_id": sup["doc_id"], "span_start": sup["span_start"], "span_end": sup["span_end"]}
            if cite not in citations:
                citations.append(cite)
        entities = [e for e in item.get("entities") or [] if isinstance(e, str) and e in roster]
        reason = {
            "reason_id": f"r{len(reasons) + 1}",
            "premise": cand["claim"],
            "mechanism": mechanism,
            "answer_implication": implication,
            "citations": citations,
        }
        if entities:
            reason["scope"] = {"entities": entities}
        while len(reason["citations"]) > 1 and not _within_caps(reasons + [reason], texts):
            reason["citations"].pop()
        if _within_caps(reasons + [reason], texts):
            reasons.append(reason)
    return reasons
