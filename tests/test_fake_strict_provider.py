"""A stricter OpenAI-style provider, simulated by a tiny HTTP server on 127.0.0.1 (real HTTP, real openai client).

It rejects max_tokens, temperature != 1, seed, response_format and reasoning_effort with 400s worded like real services,
makes the first small-budget reply spend everything on hidden "thinking", and then answers with <think> text, prose and a
code fence around the JSON. Every stage must still work: this is what "runs on a non-Groq provider" means in practice.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import openai
import pytest

import memory.llm as llm_mod
from memory.llm import Provider, chat_json


def _make_handler(log: list):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append(sorted(body))
            def bad(msg, param):
                self._send(400, {"error": {"message": msg, "type": "invalid_request_error", "param": param, "code": "unsupported_parameter"}})
            if "max_tokens" in body:
                return bad("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", "max_tokens")
            if body.get("temperature", 1) != 1:
                return bad("Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported.", "temperature")
            if "seed" in body:
                return bad("Unrecognized request argument supplied: seed", "seed")
            if "response_format" in body:
                return bad("'response_format' of type 'json_object' is not supported with this model.", "response_format")
            if "reasoning_effort" in body:
                return bad("Unrecognized request argument supplied: reasoning_effort", "reasoning_effort")
            budget = body.get("max_completion_tokens", 0)
            if budget < 300:   # hidden thinking eats a small budget
                return self._send(200, {"choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": ""}}],
                                        "usage": {"prompt_tokens": 20, "completion_tokens": budget}})
            content = '<think>thinking about {braces} first</think>\nSure! Here is the answer:\n```json\n{"ranked": ["a", "b"], "ok": true}\n```\nLet me know.'
            self._send(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                             "usage": {"prompt_tokens": 20, "completion_tokens": 30}})
    return Handler


@pytest.fixture()
def strict_provider(monkeypatch):
    log: list = []
    server = HTTPServer(("127.0.0.1", 0), _make_handler(log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm_mod, "OpenAI", openai.OpenAI)            # conftest blocks the client by default; this one is loopback only
    p = Provider(name="strict", api_key="fake", base_url=f"http://127.0.0.1:{server.server_port}/v1", model_strong="s", model_fast="f")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ALL_EXHAUSTED_LOGGED", set())
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    yield p, log
    server.shutdown()


def test_every_quirk_is_learned_once_and_the_json_is_recovered(strict_provider, tmp_path):
    p, log = strict_provider
    out = chat_json("Return JSON.", "go", "f", cache_dir=str(tmp_path), max_tokens=100, reasoning_effort="low")
    assert out == {"ranked": ["a", "b"], "ok": True}
    assert {"max_tokens", "temperature", "seed", "response_format"} <= p.unsupported_params
    assert any(k.startswith("reasoning_effort") for k in p.unsupported_params)
    log.clear()
    chat_json("Return JSON.", "go again", "f", cache_dir=str(tmp_path), max_tokens=100, reasoning_effort="low")
    # learned: the second question needs only the truncation retries, never a rejected-parameter round trip
    assert all("max_tokens" not in keys and "seed" not in keys and "response_format" not in keys for keys in log)
    assert len(log) <= 3
