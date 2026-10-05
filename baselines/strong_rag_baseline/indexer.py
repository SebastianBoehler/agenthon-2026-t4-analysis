"""Corpus indexer: bounded retrievable excerpts with exact global offsets.

Corpus documents carry their text as a ``spans`` array (or a flat ``text``
field). The scorer's offset convention concatenates span texts with a single
space, so a span that starts at position ``p`` in that concatenation can be
cited directly as ``(doc_id, p, p + len(span_text))``. Indexing at span
granularity gives each excerpt an exact, citation-ready offset pair. Long source
spans are split into overlapping slices before retrieval to bound prompt size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .bounded_spans import spans

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_date: str | None
    span_start: int  # global char offset into the document's joined text
    span_end: int
    text: str


@dataclass(frozen=True)
class IndexedCorpus:
    chunks: list[Chunk]
    doc_texts: dict[str, str]  # doc_id -> full joined text (for span finding)
    doc_dates: dict[str, str | None]


def _iter_span_texts(doc: dict) -> list[str]:
    if isinstance(doc.get("text"), str):
        return [doc["text"]]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return [sp.get("text", "") for sp in spans if isinstance(sp, dict)]
    return []


def build_index(corpus_dir: str | Path) -> IndexedCorpus:
    """Index every corpus document under ``corpus_dir`` (skips the manifest)."""
    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}
    for path in sorted(Path(corpus_dir).glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc_id = doc.get("doc_id", path.stem)
        if doc_id != path.stem:
            raise ValueError(f"corpus document identity disagrees with filename: {path.name}")
        doc_date = doc.get("doc_date")
        offset = 0
        parts: list[str] = []
        for text in _iter_span_texts(doc):
            for start, end in spans(text):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        doc_date=doc_date,
                        span_start=offset + start,
                        span_end=offset + end,
                        text=text[start:end],
                    )
                )
            parts.append(text)
            offset += len(text) + 1  # +1 for the joining space
        doc_texts[doc_id] = " ".join(parts)
        doc_dates[doc_id] = doc_date
    return IndexedCorpus(chunks=chunks, doc_texts=doc_texts, doc_dates=doc_dates)
