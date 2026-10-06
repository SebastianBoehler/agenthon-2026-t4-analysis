"""Local runs against a hosted Nemotron 3 Super (OpenRouter) as a stand-in for the House route.

Not the House: provider, quantization, latency and context limits differ, and every development
unit is memorization-exposed, so these runs check reply format, parsing, timing and plausibility,
not official scores. Reads OPENROUTER_API_KEY from the repo's git-ignored .env.

    python -m baselines.vector_agent.tests.run_local --out /tmp/t4-local [unit names...]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselines.vector_agent.agent import Schedule
from baselines.vector_agent.cli import run
from baselines.vector_agent.house import Budget, HouseClient, HouseConfig

ROOT = Path(__file__).resolve().parents[3]
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY missing from .env")


class LoggedOpenRouter(HouseClient):
    def __init__(self, budget: Budget, log_dir: Path):
        super().__init__(HouseConfig("https://openrouter.ai/api", MODEL, _key()), budget)
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.n = 0

    def payload(self, system, user, **kw):
        body = super().payload(system, user, **kw)
        body["reasoning"] = {"enabled": False}  # House runs with thinking disabled
        return body

    def complete(self, system, user, **kw):
        with self.lock:
            self.n += 1
            idx = self.n
        t0 = time.monotonic()
        err, text = None, None
        try:
            text = super().complete(system, user, **kw)
            return text
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec = {"i": idx, "secs": round(time.monotonic() - t0, 1), "prompt_chars": len(user),
                   "temperature": kw.get("temperature"), "error": err, "reply": text}
            (self.log_dir / f"call-{idx:02d}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def run_unit(unit: Path, out: Path) -> str:
    d = out / unit.name
    d.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    chat = LoggedOpenRouter(budget, d)
    lines: list[str] = []
    t0 = time.monotonic()
    try:
        run(unit / "task.json", unit / "corpus", d / "answer.json", chat, budget, Schedule(), t0,
            log=lines.append)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        status = f"FAIL {type(exc).__name__}: {exc}"
    (d / "log.txt").write_text("\n".join(lines) + f"\n{status}\n", encoding="utf-8")
    return f"{unit.name}: {status} requests={budget.used} {time.monotonic() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("units", nargs="*")
    a = ap.parse_args()
    names = a.units or sorted(p.name for p in (ROOT / "units").iterdir() if (p / "task.json").exists()
                              and "EXAMPLE" not in p.name)
    with ThreadPoolExecutor(a.parallel) as pool:
        for line in pool.map(lambda n: run_unit(ROOT / "units" / n, a.out), names):
            print(line, flush=True)


if __name__ == "__main__":
    main()
