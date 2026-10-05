"""## Executive summary (read this first)

Read entity admission from the trusted unit manifest, never document prose or names.
Shared documents admit every entity. Missing labels and changed bytes fail explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def load_admission(unit_dir: Path) -> dict[str, frozenset[str] | None]:
    """Map manifested document IDs to roster IDs; None denotes explicitly shared text."""
    manifest = json.loads((unit_dir / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("unit manifest admission requires a files array")
    admission: dict[str, frozenset[str] | None] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("unit manifest admission entry must be an object")
        if entry.get("role") != "corpus" or entry.get("path") == "corpus/manifest.json":
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            raise ValueError("corpus admission path must be a string")
        relative = PurePosixPath(path)
        if (len(relative.parts) != 2 or relative.parts[0] != "corpus"
                or relative.suffix != ".json" or not relative.stem
                or str(relative) != path):
            raise ValueError(f"invalid corpus admission path: {path!r}")
        ids, shared = entry.get("entity_ids"), entry.get("shared")
        if ((shared is not None and shared is not True)
                or (ids is not None and (not isinstance(ids, list)
                    or not all(isinstance(i, str) and i for i in ids)
                    or len(ids) != len(set(ids))))
                or (shared is True and ids)
                or (shared is not True and ids is None)):
            raise ValueError(f"missing or malformed corpus admission: {path!r}")
        if relative.stem in admission:
            raise ValueError(f"duplicate corpus admission: {path!r}")
        data = (unit_dir / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise ValueError(f"corpus admission digest mismatch: {path!r}")
        admission[relative.stem] = None if shared is True else frozenset(ids)
    if not admission:
        raise ValueError("unit manifest declares no corpus admission")
    return admission
