"""House route client: thread-safe request budget, per-call timeouts, thinking disabled.

``POST $MODEL_ENDPOINT/v1/chat/completions`` with ``Authorization: Bearer $MODEL_TOKEN`` and the
injected ``MODEL_NAME``. The route admits 25 requests per unit; every attempt that may have been
admitted is charged against a local cap below that. Proxy variables are honoured by urllib.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

REQUEST_CAP = 24  # one below the House allowance of 25 admitted requests per unit
MAX_OUTPUT_TOKENS = 4000

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class BudgetExhausted(RuntimeError):
    pass


class Chat(Protocol):
    def complete(self, system: str, user: str, *, temperature: float, seed: int,
                 max_tokens: int, timeout: float) -> str: ...


@dataclass
class HouseConfig:
    endpoint: str
    model: str
    token: str | None

    @staticmethod
    def from_env() -> "HouseConfig":
        return HouseConfig(
            endpoint=os.environ.get("MODEL_ENDPOINT", "").rstrip("/"),
            model=os.environ.get("MODEL_NAME") or os.environ.get("MODEL_ID", ""),
            token=os.environ.get("MODEL_TOKEN") or None,
        )


def chat_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


class Budget:
    def __init__(self, cap: int = REQUEST_CAP):
        self.cap = cap
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.cap:
                raise BudgetExhausted(f"request cap {self.cap} reached")
            self.used += 1

    def refund(self) -> None:
        with self._lock:
            self.used = max(0, self.used - 1)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.cap - self.used


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return max(0.0, min(30.0, float(exc.headers.get("Retry-After", ""))))
    except (TypeError, ValueError):
        return None


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text:  # an opening tag swallowed by the template
        text = text.split("</think>", 1)[1]
    return text.strip()


class HouseClient:
    def __init__(self, config: HouseConfig, budget: Budget):
        if not config.endpoint:
            raise RuntimeError("MODEL_ENDPOINT is not set")
        self.config = config
        self.budget = budget

    def payload(self, system: str, user: str, *, temperature: float, seed: int, max_tokens: int) -> dict:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": min(int(max_tokens), MAX_OUTPUT_TOKENS),
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def complete(self, system: str, user: str, *, temperature: float, seed: int,
                 max_tokens: int, timeout: float) -> str:
        payload = self.payload(system, user, temperature=temperature, seed=seed, max_tokens=max_tokens)
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        request = urllib.request.Request(
            chat_url(self.config.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        deadline = time.monotonic() + timeout
        for attempt in range(8):
            self.budget.take()
            try:
                with urllib.request.urlopen(request, timeout=max(5.0, deadline - time.monotonic())) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    self.budget.refund()  # authentication refusal before admission; 429 may be upstream
                left = deadline - time.monotonic()
                if exc.code != 429 or left < 15:
                    raise
                wait = _retry_after(exc) or min(2.0 ** attempt, 12.0)
                time.sleep(min(wait + random.uniform(0, 1.5), left - 10))
        else:
            raise RuntimeError("House kept refusing with HTTP 429")
        message = (body.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("House reply carried no message content")
        return strip_reasoning(content)
