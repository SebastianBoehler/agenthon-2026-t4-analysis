"""## Executive summary (read this first)

Split long evidence into overlapping exact source slices before retrieval.
Each excerpt is bounded, while full document text and citation offsets stay intact.
"""
from __future__ import annotations

from collections.abc import Iterator

MAX_CHUNK_CHARS = 3200
OVERLAP_CHARS = 256


def spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield bounded code-point ranges, preferring sentence and word boundaries."""
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            lower = start + MAX_CHUNK_CHARS // 2
            boundary = max(text.rfind("\n", lower, end), text.rfind(". ", lower, end))
            if boundary < 0:
                boundary = text.rfind(" ", lower, end)
            if boundary >= lower:
                end = boundary + 1
        yield start, end
        if end == len(text):
            break
        # Keep nearby facts together without starting the next excerpt mid-word.
        next_start = max(start + 1, end - OVERLAP_CHARS)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
