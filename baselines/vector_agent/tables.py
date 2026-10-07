"""Arithmetic summaries of the pipe-delimited tables in corpus documents.

Models read tables well but compute spreads badly. For every numeric column of a ``a | b | c``
table this reports the last value, the mean, the spread and the typical step-to-step change;
for vintage tables (columns headed by dates) it also reports the typical revision between
adjacent vintages. The summary is shown next to the document in prompts, labelled as computed,
and is never part of a citable document text.
"""
from __future__ import annotations

import math
import re

_CELL_NUM = re.compile(r"^[+\-−]?\$?\d[\d,]*(?:\.\d+)?%?$")
_DATEISH = re.compile(r"\d{4}-\d{2}")
_MAX_COLUMNS = 14


def _num(cell: str) -> float | None:
    cell = cell.strip()
    if not _CELL_NUM.match(cell):
        return None
    try:
        return float(cell.replace(",", "").replace("$", "").replace("%", "").replace("−", "-"))
    except ValueError:
        return None


def _tables(text: str) -> list[list[list[str]]]:
    out, block = [], []
    for line in text.split("\n"):
        if line.count(" | ") >= 1:
            cells = [c.strip() for c in line.split("|")]
            if block and len(cells) != len(block[0]):
                if len(block) >= 4:
                    out.append(block)
                block = []
            block.append(cells)
        else:
            if len(block) >= 4:
                out.append(block)
            block = []
    if len(block) >= 4:
        out.append(block)
    return out


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _fmt(x: float) -> str:
    return f"{x:.4g}"


_ROW_DATE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def _row_gap_days(rows: list[list[str]]) -> float | None:
    """Mean spacing in days of the first-column dates (YYYY-MM-DD or YYYY-MM), if any."""
    stamps = []
    for r in rows:
        m = _ROW_DATE.match(r[0])
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 15)
        stamps.append(y * 365.25 + (mo - 1) * 30.44 + d)
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return None
    return (stamps[-1] - stamps[0]) / (len(stamps) - 1)  # mean spacing: weekends count for daily data


def summarize(text: str, horizon_days: int | None = None) -> str:
    lines: list[str] = []
    for table in _tables(text):
        header, rows = table[0], table[1:]
        gap = _row_gap_days(rows)
        steps = max(1.0, horizon_days / gap) if gap and horizon_days else None
        numeric_cols = []
        for j in range(1, min(len(header), _MAX_COLUMNS + 1)):
            vals = [_num(r[j]) for r in rows]
            series = [v for v in vals if v is not None]
            if len(series) >= 4 and len(series) >= len(rows) // 2:
                numeric_cols.append((j, vals, series))
        for j, vals, series in numeric_cols:
            changes = [b - a for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
            part = (f"'{header[j]}': n={len(series)}, last={_fmt(series[-1])}, mean={_fmt(sum(series) / len(series))}, "
                    f"sd={_fmt(_sd(series))}, min={_fmt(min(series))}, max={_fmt(max(series))}")
            if len(changes) >= 3:
                mad = sum(abs(c) for c in changes) / len(changes)
                part += (f"; row-to-row change: sd={_fmt(_sd(changes))}, mean abs={_fmt(mad)}, "
                         f"median abs={_fmt(_median([abs(c) for c in changes]))}")
                if steps and steps > 1.5:
                    part += (f"; if moves were a random walk, the change over the {horizon_days}-day horizon "
                             f"(~{steps:.0f} rows) would have sd={_fmt(_sd(changes) * math.sqrt(steps))}")
            lines.append(part)
        dated = [j for j in range(1, min(len(header), _MAX_COLUMNS + 1)) if _DATEISH.search(header[j])
                 and sum(_num(r[j]) is not None for r in rows) >= 2]
        if len(dated) >= 3:
            revisions, per_release, routine = [], [], []
            for a, b in zip(dated, dated[1:]):
                diffs = [_num(r[b]) - _num(r[a]) for r in rows
                         if _num(r[a]) is not None and _num(r[b]) is not None]
                moved = [d for d in diffs if d != 0]
                revisions += moved
                if not (len(diffs) >= 4 and len(moved) == len(diffs)):
                    routine += moved  # a release that revises every value is an annual/benchmark update
                if diffs:
                    mean_abs = sum(abs(d) for d in moved) / len(moved) if moved else 0.0
                    per_release.append(f"{header[a]} -> {header[b]}: {len(moved)} of {len(diffs)} values revised"
                                       + (f", mean abs {_fmt(mean_abs)}, net {'up' if sum(moved) > 0 else 'down'}"
                                          if moved else ""))
            if revisions:
                ups = sum(1 for x in revisions if x > 0)
                med = _median([abs(x) for x in revisions])
                lines.append(f"revisions between adjacent dated columns (nonzero only): n={len(revisions)}, "
                             f"up={ups}, down={len(revisions) - ups}, median abs={_fmt(med)}, "
                             f"mean abs={_fmt(sum(abs(x) for x in revisions) / len(revisions))}; by release "
                             "(a release that revises every month at once is usually an annual or benchmark "
                             "update): " + "; ".join(per_release)
                             + (f"; routine releases only (excluding releases that revised every value): "
                                f"up={sum(1 for x in routine if x > 0)}, down={sum(1 for x in routine if x < 0)}, "
                                f"median abs={_fmt(_median([abs(x) for x in routine]))}"
                                if routine and len(routine) < len(revisions) else ""))
    return "\n".join(lines)
