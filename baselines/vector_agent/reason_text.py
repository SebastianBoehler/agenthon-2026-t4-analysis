"""Published reason-text normalization and URI evidence masking, using only stdlib.

Detection views never replace the literal submitted premise or cited source text.
"""
from __future__ import annotations

import re
import unicodedata

_SLASH_LOOKALIKES = ("\u2044", "\u2215", "\u29f8", "\u2571", "\u27cb", "\u3033", "\u1735", "\u2cfa",
                     "\ufe68", "\u2e4a", "\u0338")
_COLON_LOOKALIKES = ("\u2236", "\ua789", "\u02d0")
_LOOKALIKES = {**{c: "/" for c in _SLASH_LOOKALIKES}, **{c: ":" for c in _COLON_LOOKALIKES}}
_INVISIBLE_FILLERS = frozenset("\u115f\u1160\u3164\uffa0")
_URI_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s,;()\[\]{}<>\"']*")


def fold_for_detection(text: str) -> str:
    """The published NFKC/Cf/filler/look-alike view used for deny-list detection."""
    return "".join(_LOOKALIKES.get(char) or unicodedata.normalize("NFKC", char)
                   for char in text if char not in _INVISIBLE_FILLERS
                   and unicodedata.category(char) != "Cf")


def duplicate_key(text: str) -> str:
    """NFC, no format characters, casefold and collapsed whitespace, as published."""
    visible = "".join(char for char in unicodedata.normalize("NFC", text)
                      if unicodedata.category(char) != "Cf")
    return " ".join(visible.casefold().split())


def reason_key(premise: str, mechanism: str, implication: str) -> tuple[str, str, str]:
    return duplicate_key(premise), duplicate_key(mechanism), duplicate_key(implication)


def mask_evidence_uris(text: str) -> str:
    """Published ASCII-scheme URI masking: one three-byte block per source character."""
    return _URI_PATTERN.sub(lambda match: "\u2588" * (match.end() - match.start()), text)
