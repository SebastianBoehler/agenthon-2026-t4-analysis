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
        value = float(cell.replace(",", "").replace("$", "").replace("%", "").replace("−", "-"))
    except ValueError:
        return None
    return value if math.isfinite(value) and abs(value) < 1e100 else None


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


PERCENT_LEVEL = 25.0  # a column whose values all stay below this reads as a rate in percent


def _numeric_columns(rows: list[list[str]], header: list[str]):
    out = []
    for j in range(1, min(len(header), _MAX_COLUMNS + 1)):
        vals = [_num(r[j]) for r in rows]
        series = [v for v in vals if v is not None]
        if len(series) >= 4 and len(series) >= len(rows) // 2:
            out.append((j, vals, series))
    return out


BACKTEST_WINDOW = 4
BACKTEST_MIN = 5  # one-step forecasts needed before the comparison is reported


def backtest(series: list[float], window: int = BACKTEST_WINDOW) -> tuple[float, float, int] | None:
    """Mean absolute one-step error of the latest value and of the mean of the previous ``window``
    values, over the column's own history."""
    last, mean = [], []
    for t in range(window, len(series)):
        last.append(abs(series[t] - series[t - 1]))
        mean.append(abs(series[t] - sum(series[t - window:t]) / window))
    if len(last) < BACKTEST_MIN:
        return None
    return sum(last) / len(last), sum(mean) / len(mean), len(last)


def horizon_sds(text: str, horizon_days: int | None, min_changes: int = 3) -> dict[str, tuple[float, float]]:
    """Per dated-row column with at least ``min_changes`` steps: (random-walk sd of its change over
    the horizon, max |value|)."""
    out: dict[str, tuple[float, float]] = {}
    for table in _tables(text):
        header, rows = table[0], table[1:]
        gap = _row_gap_days(rows)
        steps = max(1.0, horizon_days / gap) if gap and horizon_days else None
        if not steps or steps <= 1.5:
            continue
        for j, vals, series in _numeric_columns(rows, header):
            changes = [b - a for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
            if len(changes) >= min_changes:
                out[header[j]] = (_sd(changes) * math.sqrt(steps), max(abs(v) for v in series))
    return out


def summarize(text: str, horizon_days: int | None = None, bps: bool = False) -> str:
    """``bps``: the target is in basis points, so horizon spreads of percent-level columns are
    restated in basis points too."""
    lines: list[str] = []
    for table in _tables(text):
        header, rows = table[0], table[1:]
        gap = _row_gap_days(rows)
        steps = max(1.0, horizon_days / gap) if gap and horizon_days else None
        numeric_cols = _numeric_columns(rows, header)
        for j, vals, series in numeric_cols:
            changes = [b - a for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
            part = (f"'{header[j]}': n={len(series)}, last={_fmt(series[-1])}, mean={_fmt(sum(series) / len(series))}, "
                    f"sd={_fmt(_sd(series))}, min={_fmt(min(series))}, max={_fmt(max(series))}")
            if len(changes) >= 3:
                mad = sum(abs(c) for c in changes) / len(changes)
                part += (f"; row-to-row change: sd={_fmt(_sd(changes))}, mean abs={_fmt(mad)}, "
                         f"median abs={_fmt(_median([abs(c) for c in changes]))}")
                bt = backtest(series)
                if bt and bt[1] < bt[0]:  # reported only when it overturns the carry-forward default
                    part += (f"; one-step backtest over {bt[2]} rows: the mean of the previous {BACKTEST_WINDOW} "
                             f"rows (now {_fmt(sum(series[-BACKTEST_WINDOW:]) / BACKTEST_WINDOW)}) missed the next "
                             f"row by {_fmt(bt[1])} on average, the latest value by {_fmt(bt[0])}: mean-reverting")
                if steps and steps > 1.5:
                    sd_h = _sd(changes) * math.sqrt(steps)
                    part += (f"; if moves were a random walk, the change over the {horizon_days}-day horizon "
                             f"(~{steps:.0f} rows) would have sd={_fmt(sd_h)}")
                    if bps and max(abs(v) for v in series) < PERCENT_LEVEL:
                        part += f" percentage points = {sd_h * 100:.3g} bps"
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
