"""OpenAI-compatible chat client for ``$MODEL_ENDPOINT`` (stdlib only).

The eval sandbox's only egress is the organizer-hosted House route. The harness
injects ``MODEL_ENDPOINT`` as the route **origin** (``scheme://host:port``) and the
OpenAI-compatible API is served under ``/v1``, so the request goes to
``$MODEL_ENDPOINT/v1/chat/completions`` with ``Authorization: Bearer $MODEL_TOKEN``
(see the hub's ``docs/HOUSE-MODEL.md``, "Calling the House route"). Locally, any
server speaking that protocol works (ollama, llama.cpp, vLLM) whether its URL is
given with or without the ``/v1`` suffix, and tests inject :class:`MockModelClient`
— same interface, canned replies, no network.

Sampling: temperature and seed are sent; hosted responses need not be deterministic.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import Config

# Conservative local attempt cap; House counts admitted requests, not every failure.
MAX_REQUESTS_PER_UNIT = 25


def chat_completions_url(model_endpoint: str) -> str:
    """The chat-completions URL for an endpoint given with or without ``/v1``.

    The harness injects the route origin (no path); local servers are often
    configured as ``http://host:port/v1``. Both resolve to ``.../v1/chat/completions``.
    """
    base = model_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


class ModelClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant message text for one chat exchange."""
        ...


@dataclass
class HTTPModelClient:
    config: Config
    _attempts: int = field(default=0, init=False)

    def complete(self, system: str, user: str) -> str:
        if not self.config.model_endpoint:
            raise RuntimeError(
                "MODEL_ENDPOINT is not set. In the eval sandbox it is injected "
                "by the harness; locally, point it at an OpenAI-compatible "
                "server or use --mock."
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.model_token:
            headers["Authorization"] = f"Bearer {self.config.model_token}"
        request = urllib.request.Request(
            chat_completions_url(self.config.model_endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            if self._attempts >= MAX_REQUESTS_PER_UNIT:
                raise RuntimeError(
                    f"local attempt cap reached ({MAX_REQUESTS_PER_UNIT} per unit)"
                ) from last_error
            self._attempts += 1
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_s
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model response content is missing or not text")
                return content
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"model call failed after {self.config.max_retries} attempts"
        ) from last_error


@dataclass
class MockModelClient:
    """Test double: returns canned text, or delegates to a callable."""

    reply: str | Callable[[str, str], str]

    def complete(self, system: str, user: str) -> str:
        if callable(self.reply):
            return self.reply(system, user)
        return self.reply
