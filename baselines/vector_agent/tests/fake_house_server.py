"""Local OpenAI-compatible stand-in for the House route (synthetic replies, configurable latency).

    python -m baselines.vector_agent.tests.fake_house_server --port 18443 --delay 30 --token t
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fake_chat import FakeChat


def serve(port: int, delay: float, token: str, fault_every: int) -> None:
    chat = FakeChat()
    lock = threading.Lock()
    stats = {"requests": 0, "inflight": 0, "max_inflight": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_response(403)
                self.end_headers()
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                stats["requests"] += 1
                stats["inflight"] += 1
                stats["max_inflight"] = max(stats["max_inflight"], stats["inflight"])
                n = stats["requests"]
            try:
                assert body["chat_template_kwargs"] == {"enable_thinking": False}
                assert body["max_tokens"] <= 4000 and "n" not in body
                time.sleep(delay)
                if fault_every and n % fault_every == 0:
                    self.send_response(502)
                    self.end_headers()
                    return
                msgs = body["messages"]
                text = chat.complete(msgs[0]["content"], msgs[1]["content"], temperature=body["temperature"],
                                     seed=body["seed"], max_tokens=body["max_tokens"], timeout=600)
                out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            finally:
                with lock:
                    stats["inflight"] -= 1
                print(json.dumps(stats), flush=True)

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18443)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--token", default="local-test-token")
    ap.add_argument("--fault-every", type=int, default=0)
    a = ap.parse_args()
    serve(a.port, a.delay, a.token, a.fault_every)
