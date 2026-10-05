"""Executive summary: Retrieve admitted evidence and emit model-supplied predictions.

Unmatched quotations are dropped; offsets never substitute a different passage.
Labels and interval bounds are validated rather than fabricated. Exact quotation
matching is this agent's engineering policy, not proof that a claim is true.
"""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass

from .claim_limits import MAX_CLAIM_CHARS, MAX_CITATION_CHARS
from .client import ModelClient
from .indexer import IndexedCorpus
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import BM25Index
from .span_finder import find_span

_MODEL_FIELDS = {"label", "point_forecast", "interval", "evidence", "rank"}

#: Query terms appended to every entity query — steer retrieval toward result
#: and outlook language regardless of family.
_QUERY_SUFFIX = "results revenue earnings guidance outlook growth"


@dataclass
class EntityResult:
    prediction: dict  # entity_predictions[] element
    dropped_claims: int
    model_raw: str


def _parse_model_json(raw: str) -> dict:
    """Extract a prediction object from a possibly reasoning-prefixed reply."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _MODEL_FIELDS.intersection(parsed):
            return parsed
    raise ValueError("model reply contains no prediction JSON object")


def _entity_query(entity: dict) -> str:
    parts = [
        str(entity.get(key, ""))
        for key in ("name", "entity_id", "sector", "series_id", "description")
    ]
    return " ".join(p for p in parts if p) + " " + _QUERY_SUFFIX


def _ground_claims(
    evidence: list[dict],
    corpus: IndexedCorpus,
    admitted_doc_ids: set[str],
) -> tuple[list[dict], int]:
    """Map model evidence to exact-span claims; count what had to be dropped."""
    claims: list[dict] = []
    dropped = 0
    for item in evidence:
        if not isinstance(item, dict):
            dropped += 1
            continue
        doc_id = item.get("doc_id", "")
        quote = str(item.get("quote", ""))
        claim_text = str(item.get("claim", "")).strip()
        doc_text = corpus.doc_texts.get(doc_id)
        if (doc_id not in admitted_doc_ids or doc_text is None or not claim_text
                or len(unicodedata.normalize("NFC", claim_text).strip()) > MAX_CLAIM_CHARS
                or len(quote.strip()) > MAX_CITATION_CHARS):
            dropped += 1
            continue

        span = find_span(doc_text, quote)
        if span is None:
            dropped += 1
            continue

        if span[1] - span[0] > MAX_CITATION_CHARS:
            dropped += 1
            continue

        claims.append(
            {
                "doc_id": doc_id,
                "span_start": span[0],
                "span_end": span[1],
                "claim": claim_text,
            }
        )
    return claims, dropped


def _model_interval(parsed: dict, level: float) -> dict:
    interval = parsed.get("interval")
    if not isinstance(interval, dict):
        raise ValueError("model interval must contain numeric lo and hi")
    lo, hi = interval.get("lo"), interval.get("hi")
    if (any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in (lo, hi)) or lo > hi):
        raise ValueError("model interval must contain finite ordered lo and hi")
    return {"level": level, "lo": float(lo), "hi": float(hi)}


def run_entity(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    client: ModelClient,
    top_k: int,
    admitted_doc_ids: set[str],
) -> EntityResult:
    # Admission labels are authoritative; document-name conventions add no rule.
    doc_ids = {chunk.doc_id for chunk in index.chunks} & admitted_doc_ids
    if not doc_ids:
        raise ValueError(f"no admitted pre-cutoff evidence for entity {entity['entity_id']}")
    retrieved = [
        s.chunk for s in index.search(_entity_query(entity), top_k, doc_ids=doc_ids)
    ]
    if not retrieved:
        raise ValueError(f"no matching admitted evidence for entity {entity['entity_id']}")
    raw = client.complete(SYSTEM_PROMPT, build_user_prompt(task, entity, retrieved))
    parsed = _parse_model_json(raw)

    target = task.get("target", {})
    labels = target.get("labels") or []
    label = parsed.get("label")
    if labels and label not in labels:
        raise ValueError(f"model label {label!r} is outside the task vocabulary")

    point = parsed.get("point_forecast")
    if point is not None and (isinstance(point, bool)
            or not isinstance(point, (int, float)) or not math.isfinite(point)):
        raise ValueError("model point_forecast must be finite numeric or null")
    if target.get("type") in {"regression", "ranking"} and point is None:
        raise ValueError("numeric target requires a model point_forecast")
    point_value = float(point) if point is not None else None
    claims, dropped = _ground_claims(
        parsed.get("evidence") or [], corpus, doc_ids
    )

    if not claims:
        raise ValueError(f"no admitted grounded claims for entity {entity['entity_id']}")

    prediction: dict = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": point_value,
        "interval": _model_interval(parsed, task.get("interval_level", 0.90)),
        "claims": claims,
    }
    if labels:
        prediction["label"] = label
    if target.get("type") == "ranking" and isinstance(parsed.get("rank"), int):
        prediction["rank"] = parsed["rank"]
    return EntityResult(prediction=prediction, dropped_claims=dropped, model_raw=raw)
