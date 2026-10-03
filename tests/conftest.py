"""Test-suite safety rails.

1. Usage ledger: fake-provider calls never reach outputs/usage_ledger.jsonl
   (env set at import AND per test).
2. No network, ever: the real OpenAI client class is replaced by one that
   raises, so a test that forgets to inject a fake client fails loudly
   instead of spending quota.
3. LLM_FROZEN_ROLES is cleared per test (tests drive fake clients); tests of
   the freeze itself set it explicitly.
"""
import os
import tempfile

import pytest

os.environ["USAGE_LEDGER_PATH"] = os.path.join(tempfile.mkdtemp(prefix="candor-test-ledger-"), "usage_ledger.jsonl")


def _no_network(*_a, **_k):
    raise RuntimeError("tests must not create a real OpenAI client; inject a fake client instead")


@pytest.fixture(autouse=True)
def _safety_rails(tmp_path, monkeypatch):
    import memory.llm as llm_mod
    monkeypatch.setenv("USAGE_LEDGER_PATH", str(tmp_path / "usage_ledger.jsonl"))
    monkeypatch.delenv("LLM_FROZEN_ROLES", raising=False)
    monkeypatch.setattr(llm_mod, "OpenAI", _no_network)
