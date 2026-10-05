"""## Executive summary (read this first)

Bounded excerpts must preserve complete source coverage and exact code-point offsets.
These controls cover Unicode, long words, empty spans and retrieval-size guarantees.
"""
from __future__ import annotations

import json

from baselines.strong_rag_baseline.bounded_spans import MAX_CHUNK_CHARS, spans
from baselines.strong_rag_baseline.indexer import build_index


def test_unicode_windows_cover_source_with_exact_offsets():
    text = ("München outlook: revenue €42.7 million.\n" * 300).strip()
    windows = list(spans(text))
    assert len(windows) > 1
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    for i, (start, end) in enumerate(windows):
        assert 0 <= start < end <= len(text)
        assert end - start <= MAX_CHUNK_CHARS
        if i:
            previous_start, previous_end = windows[i - 1]
            assert previous_start < start <= previous_end
            assert previous_end < end
            assert text[start - 1].isspace()


def test_unbroken_text_terminates_without_losing_characters():
    text = "x" * (MAX_CHUNK_CHARS * 3 + 7)
    windows = list(spans(text))
    assert "".join(text[start:end] for start, end in windows) == text
    assert all(end - start <= MAX_CHUNK_CHARS for start, end in windows)


def test_small_and_empty_sources():
    assert list(spans("")) == []
    assert list(spans("short excerpt")) == [(0, 13)]


def test_indexer_preserves_joined_span_offsets(tmp_path):
    parts = ["", "α revenue grew. " * 600, "", "Final outlook stays stable."]
    (tmp_path / "source.json").write_text(json.dumps({
        "doc_id": "source", "doc_date": "2024-01-01",
        "spans": [{"text": part} for part in parts],
    }), encoding="utf-8")
    corpus = build_index(tmp_path)
    assert corpus.doc_texts["source"] == " ".join(parts)
    for chunk in corpus.chunks:
        assert chunk.text == corpus.doc_texts["source"][chunk.span_start:chunk.span_end]
        assert len(chunk.text) <= MAX_CHUNK_CHARS
        assert chunk.doc_id == "source" and chunk.doc_date == "2024-01-01"
    assert corpus.chunks[-1].text == parts[-1]


def test_flat_document_is_split_without_changing_source(tmp_path):
    text = "Company earnings and outlook increased. " * 500
    (tmp_path / "source.json").write_text(json.dumps({
        "doc_id": "source", "doc_date": "2024-01-01", "text": text,
    }), encoding="utf-8")
    corpus = build_index(tmp_path)
    assert corpus.doc_texts["source"] == text
    assert len(corpus.chunks) > 1
    assert max(len(chunk.text) for chunk in corpus.chunks) <= MAX_CHUNK_CHARS
