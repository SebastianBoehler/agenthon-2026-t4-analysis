"""Synthetic House stand-in for wiring tests. Its numbers are placeholders, not forecasts."""
from __future__ import annotations

import json
import random
import re
import threading
import time

_ROW = re.compile(r"^\{\"entity_id\": .*\}$", re.MULTILINE)
_FOCUS = re.compile(r"^FOCUS ENTITY: (\S+)$", re.MULTILINE)
_LABELS = re.compile(r"^Allowed labels: (.+)\.$", re.MULTILINE)
_DOC = re.compile(r"^(?:=== |\[E\d+\] )(\S+) \|.*\n(.+)$", re.MULTILINE)


class FakeChat:
    """Answers from the prompt. ``faults`` maps a call index to a fault name."""

    def __init__(self, faults: dict[int, str] | None = None, delay: float = 0.0, seed: int = 0):
        self.faults = faults or {}
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.prompts: list[str] = []

    def complete(self, system, user, *, temperature, seed, max_tokens, timeout):
        with self.lock:
            idx = self.calls
            self.calls += 1
            self.prompts.append(user)
        fault = self.faults.get(idx)
        if self.delay:
            time.sleep(self.delay)
        if fault == "error":
            raise TimeoutError("synthetic timeout")
        if fault == "slow":
            time.sleep(timeout + 1)
            raise TimeoutError("synthetic slow")
        if fault == "garbage":
            return "I cannot answer that."
        if "CANDIDATE PREMISES" in user:
            keys = re.findall(r"^\[(R\d+)\]", user, re.MULTILINE)
            return json.dumps({"reasons": [
                {"premise": k, "mechanism": "Synthetic mechanism text for wiring.",
                 "answer_implication": "Synthetic implication for wiring.", "entities": []}
                for k in keys[:3]]})
        rows = [json.loads(r) for r in _ROW.findall(user)]
        focus = _FOCUS.search(user)
        if focus:
            rows = [r for r in rows if r["entity_id"] == focus.group(1)]
        labels_m = _LABELS.search(user)
        labels = [x.strip() for x in labels_m.group(1).split(",")] if labels_m else []
        docs = _DOC.findall(user)
        preds = []
        for r in rows:
            v = round(self.rng.uniform(-2, 2), 3)
            item = {"entity_id": r["entity_id"], "analysis": "synthetic", "baseline": 0,
                    "point_forecast": v, "lo": v - 1, "hi": v + 1}
            if labels:
                item["label"] = labels[0].upper() if fault == "case" else labels[0]
                item["probability"] = 0.6
            if docs:
                doc_id, line = docs[self.rng.randrange(len(docs))]
                words = line.split()[:12]
                item["evidence"] = [{"doc_id": doc_id, "quote": " ".join(words)}]
            preds.append(item)
        if fault == "drop_one" and len(preds) > 1:
            preds = preds[1:]
        body = json.dumps({"predictions": preds} if not focus else preds[0])
        if fault == "fenced":
            return "Here you go:\n```json\n" + body.replace(": -", ": -").replace(": 0.", ": +0.") + "\n```"
        if fault == "think":
            return "<think>reasoning</think>\n" + body
        return body
