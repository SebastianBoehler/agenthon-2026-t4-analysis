"""Verbatim claims and reasons with exact offsets into admitted pre-cutoff documents.

Every claim text is exactly ``doc.text[span_start:span_end]`` of a document the manifest admits
for the entity, a few hundred characters long. Exact quotation establishes source identity. Official semantic judging and
forecast quality are separate checks; local quote validation does not certify either.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from .evidence import SMALL_DOC_CHARS, Passage, best_segments, scored_segments, segments, task_query, tokens
from .parse import Sample
from .reason_text import fold_for_detection, mask_evidence_uris, reason_key
from .unit import Doc, Unit, task_table_text

CLAIMS_PER_ENTITY = 2
MAX_CLAIM_CHARS = 399  # character heuristic; the 400-judge-token cap needs the judge tokenizer
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
        if s <= start < end <= e:
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
        # Equal total lengths do not guarantee unchanged offsets inside the document.
        original = unicodedata.normalize("NFKC", doc.text[m.start():m.end()])
        if re.fullmatch(pattern, original):
            return m.start(), m.end()
    return None


def _substantive(text: str) -> bool:
    """A figure or several content words; official semantic rules are checked separately."""
    return bool(re.search(r"\d", text)) or len(tokens(text)) >= 6


def claim_is_entity_name(unit: Unit, text: str) -> bool:
    """A roster name alone identifies an entity but states no fact about it."""
    def normalized(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(" .:;!?")
    quote = normalized(text)
    return any(quote == normalized(e["name"]) for e in unit.entities
               if isinstance(e.get("name"), str) and e["name"].strip())


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
        if claim_is_entity_name(unit, doc.text[s:e]):
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
        if e - s > MAX_CLAIM_CHARS:  # keep the quote within the participant character budget
            cut = text.rfind(", ", s, s + MAX_CLAIM_CHARS)
            e = cut if cut > s + 40 else s + MAX_CLAIM_CHARS
        claims.append({"doc_id": "task", "span_start": s, "span_end": e, "claim": text[s:e]})
    return claims


REASON_CANDIDATES = 30


PER_DRIVER = 2


def reason_candidates(unit: Unit, claims_by_entity: dict[str, list[dict]], drivers: list[str] | None = None,
                      ranges: dict[str, list[tuple[int, int]]] | None = None) -> list[tuple[str, dict]]:
    """Premise candidates: the passages that best state each task-named driver (across entities),
    then each entity's claims, then the best task-relevant passages of the smaller documents.
    ``ranges``: where retrieval looked in each large document."""
    out: list[tuple[str, dict]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(c: dict) -> None:
        key = (c["doc_id"], c["span_start"], c["span_end"])
        if key in seen or any(k[0] == key[0] and k[1] < key[2] and key[1] < k[2] for k in seen):
            return
        seen.add(key)
        out.append((f"R{len(out) + 1}", c))

    for driver in drivers or []:
        query = Counter({t: 3.0 for t in tokens(driver)})
        scored = []
        for doc in unit.docs.values():
            if not doc.citable:
                continue
            doc_ranges = (ranges or {}).get(doc.doc_id)
            if len(doc.text) > SMALL_DOC_CHARS and not doc_ranges:
                continue
            for score, st, en in scored_segments(doc, {}, query, driver, doc_ranges)[:2]:
                if _substantive(doc.text[st:en]):
                    scored.append((score, doc, st, en))
        scored.sort(key=lambda x: x[0], reverse=True)
        taken = 0
        for _, doc, st, en in scored:
            if taken >= PER_DRIVER or len(out) >= REASON_CANDIDATES // 2:
                break
            before = len(out)
            add(dict(_claim(doc.doc_id, doc.text, st, en), driver=driver))
            taken += len(out) - before

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
_KEY_IN_PARENS = re.compile(r"\s*[,;]\s*R\d{1,2}(?=\s*[\)\]])|(?<=[\(\[])\s*R\d{1,2}\s*[,;]\s*")


def _clean(text: object, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    text = _KEY_REF.sub("", text)  # candidate keys mean nothing to the reader
    text = _KEY_IN_PARENS.sub("", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for mark in (". ", "; ", ", "):  # end on the last whole sentence or clause that fits
        at = cut.rfind(mark)
        if at >= limit // 2:
            return cut[:at + 1].rstrip(",; ") + ("" if mark == ". " else ".")
    return cut.rsplit(" ", 1)[0]


def _denied(text: str) -> bool:
    low = fold_for_detection(text).lower()
    return any(tok in low for tok in DENY) or "://" in low


MAX_SUPPORT = 3
CITE_CHARS = 1500  # the judge reads cited passages whole; a paragraph grounds figures a sentence omits
_AGAINST = re.compile(
    r"\b(?:conflicts?|contradicts?|inconsistent)\b|\bagainst the submitted\b|\bthan (?:the )?submitted\b"
    r"|\bsubmitted\b[^.;]{0,60}\b(?:too (?:high|low)|overstat|understat|should)", re.IGNORECASE)


def _widen(doc: Doc, start: int, end: int) -> tuple[int, int]:
    """The paragraph around a span, at most CITE_CHARS long and always containing the span."""
    text = doc.text
    a = text.rfind("\n\n", 0, start)
    b = text.find("\n\n", end)
    lo, hi = (a + 2 if a >= 0 else 0), (b if b >= 0 else len(text))
    if hi - lo > CITE_CHARS:
        room = max(0, CITE_CHARS - (end - start))
        lo = max(lo, start - room // 2)
        hi = min(hi, max(end, lo + CITE_CHARS))
    return min(lo, start), max(hi, end)


def _key_of(value: object) -> str:
    m = re.match(r"\s*\[?\s*(R\d+)\b", str(value or ""))
    return m.group(1) if m else ""


def _within_caps(reasons: list[dict], texts: dict[tuple[str, int, int], str]) -> bool:
    judged = [{k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")} for r in reasons]
    evidence = [{"reason_id": r["reason_id"], "doc_id": c["doc_id"], "span_start": c["span_start"],
                 "span_end": c["span_end"],
                 "trusted_text": mask_evidence_uris(texts[(c["doc_id"], c["span_start"], c["span_end"])])}
                for r in reasons for c in r["citations"]]
    # Keep list brackets (+2 bytes) and raw reason text: conservative, not the sealed renderer.
    return (len(json.dumps(judged, ensure_ascii=False, separators=(",", ":")).encode()) <= REASON_BYTES_CAP
            and len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()) <= EVIDENCE_BYTES_CAP)


def build_reasons(unit: Unit, reply: dict | None, candidates: list[tuple[str, dict]]) -> list[dict]:
    """Turn the model's driver reasons into schema-valid reasons: a verbatim premise, cited, plus
    up to three supporting passages cited alongside it (the judge reads every cited passage)."""
    if not isinstance(reply, dict) or not isinstance(reply.get("reasons"), list):
        return []
    by_key = dict(candidates)
    roster = set(unit.entity_ids)
    texts: dict[tuple[str, int, int], str] = {}

    def cite_of(c: dict) -> dict:
        doc = unit.docs.get(c["doc_id"])
        s, e = _widen(doc, c["span_start"], c["span_end"]) if doc is not None else (c["span_start"], c["span_end"])
        texts[(c["doc_id"], s, e)] = doc.text[s:e] if doc is not None else c["claim"]
        return {"doc_id": c["doc_id"], "span_start": s, "span_end": e}

    reasons: list[dict] = []
    used: set[str] = set()
    content_keys: set[tuple[str, str, str]] = set()
    for item in reply["reasons"]:
        if not isinstance(item, dict) or len(reasons) >= 3:
            continue
        key = _key_of(item.get("premise"))
        if key in used:  # a premise already taken: the reason's first unused support passage stands in
            key = next((k for k in map(_key_of, item.get("support") or []) if k in by_key and k not in used), key)
        cand = by_key.get(key)
        mechanism = _clean(item.get("mechanism"), 600)
        implication = _clean(item.get("answer_implication"), 800)  # 3 x (399 + 600 + 800) fits the cap
        if cand is None or key in used or not mechanism or not implication:
            continue
        if _denied(mechanism) or _denied(implication) or _AGAINST.search(implication):
            continue  # an implication arguing against the submitted answer contradicts it
        content_key = reason_key(cand["claim"], mechanism, implication)
        if content_key in content_keys:
            continue
        used.add(key)
        citations = [cite_of(cand)]
        for extra in (item.get("support") or [])[:MAX_SUPPORT + 1]:
            sup = by_key.get(_key_of(extra))
            if sup is None:
                continue
            cite = cite_of(sup)
            if cite not in citations and len(citations) <= MAX_SUPPORT:
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
            content_keys.add(content_key)
    return reasons
