"""scripts/provider_check.py: loud banner and exit status 2 with no key; unchanged path with a key (no network)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "CUSTOM_API_KEY", "LLM_PROVIDERS"]


def _load():
    spec = importlib.util.spec_from_file_location("provider_check_under_test", ROOT / "scripts" / "provider_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_key_prints_banner_and_exits_2(monkeypatch, capsys):
    from memory import llm
    mod = _load()
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(mod, "_load_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["provider_check.py"])
    monkeypatch.setattr(mod, "live_providers", lambda: [])
    monkeypatch.setattr(llm, "reload_providers", lambda: None)
    assert mod.main() == 2
    err = capsys.readouterr().err
    for name in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "CUSTOM_API_KEY"]:
        assert name in err
    assert "falls back" in err and "Quality is lower" in err


def test_with_a_provider_the_check_runs_and_does_not_print_the_banner(monkeypatch, capsys):
    from memory import llm
    mod = _load()
    monkeypatch.setattr(mod, "_load_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["provider_check.py", "--stages", "analysis"])
    monkeypatch.setattr(llm, "reload_providers", lambda: None)

    class FakeProvider:
        name = "fake"
        unsupported_params: set = set()
        api_key = "x"

        def model_for(self, role):
            return "fake-model"

    monkeypatch.setattr(mod, "live_providers", lambda: [FakeProvider()])
    monkeypatch.setattr(llm, "_PROVIDERS", [FakeProvider()])
    monkeypatch.setattr(llm, "_chat_json_one_provider", lambda *a, **k: {"intent": "x", "sub_queries": []})
    rc = mod.main()
    out = capsys.readouterr()
    assert rc == 0 and "NO LLM SERVICE" not in out.err and "PASS" in out.out
