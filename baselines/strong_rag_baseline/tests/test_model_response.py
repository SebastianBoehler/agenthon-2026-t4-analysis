"""Executive summary: Pin House request controls and tolerate model reasoning prefixes.

The House API allows at most 25 admitted generation requests per unit and enables
thinking by default. These tests verify that the client sends the documented opt-out,
uses the authenticated `/v1` route, enforces a conservative per-unit attempt cap, and
extracts prediction JSON after non-JSON reasoning text.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from baselines.strong_rag_baseline.agent import _parse_model_json
from baselines.strong_rag_baseline.client import HTTPModelClient
from baselines.strong_rag_baseline.config import Config


def _config(max_retries: int = 1) -> Config:
    return Config(
        model_endpoint="http://house.example:8080",
        model_id="house",
        model_token="test-token",
        seed=7,
        top_k=10,
        timeout_s=1,
        max_retries=max_retries,
        temperature=0,
    )


def test_request_uses_house_route_auth_and_disables_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"evidence": []}'}}]}).encode()

    def fake_urlopen(request, timeout: float) -> Response:
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert HTTPModelClient(_config()).complete("sys", "user") == '{"evidence": []}'
    assert seen["url"] == "http://house.example:8080/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-token"
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["payload"]["model"] == "house"


def test_client_never_exceeds_house_unit_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def refused(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("synthetic local refusal")

    monkeypatch.setattr("urllib.request.urlopen", refused)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = HTTPModelClient(_config(max_retries=30))
    with pytest.raises(RuntimeError, match=r"attempt cap reached \(25 per unit\)"):
        client.complete("sys", "user")
    assert calls == 25


@pytest.mark.parametrize(
    "reply",
    [
        'Thinking: the literal {not-json} is irrelevant. Final: {"label":"up","evidence":[]}',
        '```json\n{"point_forecast": 1.25, "evidence": []}\n```',
    ],
)
def test_parser_extracts_prediction_after_reasoning_or_fences(reply: str) -> None:
    parsed = _parse_model_json(reply)
    assert set(parsed).intersection({"label", "point_forecast"})


def test_parser_rejects_reply_without_prediction_object() -> None:
    with pytest.raises(ValueError, match="no prediction JSON object"):
        _parse_model_json("Thinking: no structured answer was produced.")
