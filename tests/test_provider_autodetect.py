"""A user who sets only one provider key must get every LLM stage working with no other setting (fake server, loopback only)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import openai
import pytest

import memory.llm as llm


def _server(models, chat_status=200):
    calls = {"models": 0, "chat": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls["models"] += 1
            self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in models]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls["chat"] += 1
            if chat_status != 200:
                return self._send(chat_status, {"error": {"message": "Insufficient credits. Add more using https://openrouter.ai/credits", "code": chat_status}})
            self._send(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true}'}}],
                             "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("LLM_PROVIDERS", "LLM_PROVIDERS_STRONG", "LLM_PROVIDERS_FAST", "GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
              "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "CUSTOM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm, "OpenAI", openai.OpenAI)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    yield monkeypatch
    llm.reload_providers()


def test_only_an_openai_key_discovers_models_and_runs(clean_env, tmp_path):
    srv, calls = _server(["text-embedding-3-small", "whisper-1", "gpt-4o-mini", "gpt-4o"])
    clean_env.setenv("OPENAI_API_KEY", "fake")
    clean_env.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    llm.reload_providers()
    assert [p.name for p in llm._PROVIDERS] == ["openai"] and llm.is_available()
    assert calls["models"] == 0                                     # discovery is lazy: nothing at import time
    assert llm.get_model_strong() == "gpt-4o" and llm.get_model_fast() == "gpt-4o-mini"
    assert llm.chat_json("Return JSON.", "go", llm.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert calls["models"] == 1


def test_every_key_present_is_enabled_in_the_documented_order(clean_env):
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        clean_env.setenv(k, "fake")
    llm.reload_providers()
    assert [p.name for p in llm._PROVIDERS] == ["openai", "openrouter", "groq", "gemini"]


def test_explicit_llm_providers_wins(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "fake")
    clean_env.setenv("OPENROUTER_API_KEY", "fake")
    clean_env.setenv("LLM_PROVIDERS", "openrouter")
    llm.reload_providers()
    assert [p.name for p in llm._PROVIDERS] == ["openrouter"]


def test_402_insufficient_credit_fails_over_loudly_to_the_next_provider(clean_env, tmp_path):
    dead, dead_calls = _server(["m-a"], chat_status=402)
    live, live_calls = _server(["m-b"])
    clean_env.setenv("OPENROUTER_API_KEY", "fake")
    clean_env.setenv("OPENROUTER_BASE_URL", f"http://127.0.0.1:{dead.server_port}/v1")
    clean_env.setenv("CUSTOM_API_KEY", "fake")
    clean_env.setenv("CUSTOM_BASE_URL", f"http://127.0.0.1:{live.server_port}/v1")
    llm.reload_providers()
    assert [p.name for p in llm._PROVIDERS] == ["openrouter", "custom"]
    assert llm.chat_json("Return JSON.", "go", llm.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert dead_calls["chat"] == 1 and live_calls["chat"] == 1       # one rejected attempt, no retry loop, then failover
    assert "fast" in llm._PROVIDERS[0].exhausted_roles or llm._PROVIDERS[0].exhausted_models


def test_no_key_prints_the_loud_banner(clean_env):
    llm.reload_providers()
    assert not llm.is_available()
    text = llm.describe_providers()
    assert "NO LLM PROVIDER ENABLED" in text and "OPENAI_API_KEY" in text


def test_pick_models_skips_non_chat_models_and_prefers_open_models():
    assert llm.pick_models(["nomic-embed-text", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]) == ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    assert llm.pick_models(["claude-sonnet-4", "claude-haiku-4"]) == ("claude-sonnet-4", "claude-haiku-4")
    assert llm.pick_models(["whisper-1"]) == ("", "")
