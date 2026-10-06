"""Load the task, the admitted pre-cutoff corpus and each document's entity admission.

Admission comes from the manifest that ships with the corpus (``corpus/manifest.json``), falling
back to the task's ``corpus_manifest`` path and the unit-level ``manifest.json``. A document is
citable for an entity only when the manifest labels it with that entity or marks it shared.
Documents without a valid ``doc_date`` on or before the cutoff are never loaded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TASK_DOC_ID = "task"


@dataclass(frozen=True)
class Doc:
    doc_id: str
    doc_date: str
    title: str
    text: str
    entity_ids: frozenset[str] | None  # None = shared with every entity

    def admits(self, entity_id: str) -> bool:
        return self.entity_ids is None or entity_id in self.entity_ids


@dataclass
class Unit:
    task: dict
    docs: dict[str, Doc]
    notes: list[str] = field(default_factory=list)

    @property
    def target(self) -> dict:
        return self.task.get("target") or {}

    @property
    def target_type(self) -> str:
        return str(self.target.get("type", ""))

    @property
    def labels(self) -> list[str]:
        return [str(x) for x in (self.target.get("labels") or [])]

    @property
    def entities(self) -> list[dict]:
        return list(self.task.get("entities") or [])

    @property
    def entity_ids(self) -> list[str]:
        return [str(e["entity_id"]) for e in self.entities]

    @property
    def interval_level(self) -> float:
        return float(self.task.get("interval_level", 0.90))

    def docs_for(self, entity_id: str) -> list[Doc]:
        return [d for d in self.docs.values() if d.admits(entity_id)]

    def total_chars(self) -> int:
        return sum(len(d.text) for d in self.docs.values())


def task_table_text(task: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    """The scorer's citable task table: one compact JSON line per entity row."""
    lines: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for row in task.get("entities") or []:
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        ranges[str(row["entity_id"])] = (offset, offset + len(line))
        lines.append(line)
        offset += len(line) + 1
    return "\n".join(lines), ranges


def _manifest_candidates(task_path: Path, corpus_dir: Path, task: dict) -> list[Path]:
    out = [corpus_dir / "manifest.json"]
    declared = task.get("corpus_manifest")
    if isinstance(declared, str) and declared:
        out.append(task_path.parent / declared)
    out.append(task_path.parent / "manifest.json")
    return out


def _read_admission(paths: list[Path]) -> tuple[dict[str, frozenset[str] | None], str | None]:
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list):
            continue
        admission: dict[str, frozenset[str] | None] = {}
        for entry in files:
            if not isinstance(entry, dict) or entry.get("role") != "corpus":
                continue
            rel = entry.get("path")
            if not isinstance(rel, str) or not rel.endswith(".json"):
                continue
            stem = Path(rel).stem
            if stem == "manifest":
                continue
            ids = entry.get("entity_ids")
            if entry.get("shared") is True:
                admission[stem] = None
            elif isinstance(ids, list) and ids:
                admission[stem] = frozenset(str(i) for i in ids)
        if admission:
            return admission, str(path)
    return {}, None


def _doc_text(doc: dict) -> str:
    if isinstance(doc.get("text"), str):
        return doc["text"]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return " ".join(str(s.get("text", "")) for s in spans if isinstance(s, dict))
    return ""


def load_unit(task_path: Path, corpus_dir: Path) -> Unit:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cutoff = str(task.get("cutoff_date", ""))
    admission, source = _read_admission(_manifest_candidates(task_path, corpus_dir, task))
    unit = Unit(task=task, docs={})
    unit.notes.append(f"admission from {Path(source).name if source else 'none'}")
    for path in sorted(corpus_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unit.notes.append(f"unreadable {path.name}")
            continue
        doc_id = str(doc.get("doc_id") or path.stem)
        doc_date = str(doc.get("doc_date") or "")
        if doc_id == TASK_DOC_ID or doc_id not in admission:
            continue  # only manifest-admitted documents are citable
        if not _ISO_DATE.match(doc_date) or (cutoff and doc_date > cutoff):
            continue  # undated or post-cutoff evidence never enters a prompt
        text = _doc_text(doc)
        if not text.strip():
            continue
        unit.docs[doc_id] = Doc(
            doc_id=doc_id,
            doc_date=doc_date,
            title=str(doc.get("title") or ""),
            text=text,
            entity_ids=admission[doc_id],
        )
    return unit
