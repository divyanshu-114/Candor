"""Quota-aware roles: per-role provider order, same-id strong/fast, premium
model never used by default, actions planner role selection. Fake providers,
no network."""
from unittest.mock import MagicMock

import pytest

import memory.llm as llm_mod
from memory.llm import Provider, chat_json
from tests.test_llm_providers import _fake_openai_response, _raw_response


def _prov(name, strong, fast, premium=""):
    return Provider(name=name, api_key=f"fake-{name}", base_url=f"https://{name}.invalid",
                    model_strong=strong, model_fast=fast, model_premium=premium)


def _client(p, monkeypatch, content='{"ok": true}'):
    c = MagicMock()
    c.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response(content))
    monkeypatch.setattr(p, "_client", c)
    return c


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setattr(llm_mod, "_TOKEN_BUDGETS", {})  # per-name buckets are process-wide; don't inherit a drained one
    monkeypatch.setattr(llm_mod, "_ALL_EXHAUSTED_LOGGED", set())
    monkeypatch.setattr(llm_mod, "_RESOLVED_MODELS", {})


def test_role_orders_differ(monkeypatch, tmp_path):
    g, m = _prov("groq", "g-big", "g-small"), _prov("gem", "lite", "lite")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [g, m])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": ["groq", "gem"], "fast": ["gem", "groq"]})
    cg, cm = _client(g, monkeypatch), _client(m, monkeypatch)

    assert llm_mod.get_model_strong() == "g-big"
    assert llm_mod.get_model_fast() == "lite"
    chat_json("s", "u1", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    cm.chat.completions.with_raw_response.create.assert_called_once()
    cg.chat.completions.with_raw_response.create.assert_not_called()
    chat_json("s", "u2", llm_mod.get_model_strong(), cache_dir=str(tmp_path))
    cg.chat.completions.with_raw_response.create.assert_called_once()


def test_same_model_id_for_both_roles_keeps_role(monkeypatch, tmp_path):
    """gem's strong and fast ids are identical; a FAST call must still follow
    the FAST order (gem first), not be mistaken for strong (groq first)."""
    g, m = _prov("groq", "g-big", "g-small"), _prov("gem", "lite", "lite")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [g, m])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": ["groq", "gem"], "fast": ["gem", "groq"]})
    cg, cm = _client(g, monkeypatch), _client(m, monkeypatch)
    assert llm_mod.get_model_fast().role == "fast"
    role, attempts = llm_mod._provider_attempts(llm_mod.get_model_fast())
    assert role == "fast" and [p.name for p in attempts] == ["gem", "groq"]


def test_strong_fails_over_in_strong_order_only(monkeypatch, tmp_path):
    g, m = _prov("groq", "g-big", "g-small"), _prov("gem", "lite", "lite")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [g, m])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": ["groq", "gem"], "fast": ["gem", "groq"]})
    g.exhausted_roles.add("strong")
    assert llm_mod.get_model_strong() == "lite"
    assert llm_mod.get_model_fast() == "lite"  # fast unaffected, gem still first


def test_unset_role_order_falls_back_to_global(monkeypatch):
    g, m = _prov("groq", "g-big", "g-small"), _prov("gem", "lite", "lite")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [g, m])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    assert [p.name for p in llm_mod._role_providers("fast")] == ["groq", "gem"]


def test_env_role_order_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.setenv("LLM_PROVIDERS_FAST", "gemini, groq")
    monkeypatch.delenv("LLM_PROVIDERS_STRONG", raising=False)
    assert llm_mod._build_role_order() == {"strong": None, "fast": ["gemini", "groq"]}


def test_premium_model_is_opt_in_and_never_default(monkeypatch):
    m = _prov("gem", "lite", "lite", premium="big-20-per-day")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [m])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    assert llm_mod.get_model_strong() == "lite" and llm_mod.get_model_fast() == "lite"
    assert llm_mod.get_model_premium() == "big-20-per-day"
    m.model_premium = ""
    assert llm_mod.get_model_premium() is None


def _planner_models(monkeypatch, step2_role):
    """Which role each planner call uses: the plan call is always fast; the
    follow-up uses ACTIONS_STEP2_ROLE."""
    import actions.planner as planner
    seen = []
    monkeypatch.setattr(planner.config, "ACTIONS_STEP2_ROLE", step2_role)
    monkeypatch.setattr(planner.llm_module, "get_model_strong", lambda: llm_mod.RoleModel("S", "strong"))
    monkeypatch.setattr(planner.llm_module, "get_model_fast", lambda: llm_mod.RoleModel("F", "fast"))
    monkeypatch.setattr(planner.llm_module, "chat_json",
                        lambda system, user, model, **kw: seen.append((kw.get("stage"), str(model))) or {"actions": []})
    planner._call(planner.PLAN_SYSTEM, "u", "fast", 500, "actions_plan")
    planner._call(planner.FOLLOWUP_SYSTEM, "u", planner.config.ACTIONS_STEP2_ROLE, 300, "actions_followup")
    return seen


def test_actions_use_fast_for_both_calls_by_default(monkeypatch):
    assert _planner_models(monkeypatch, "fast") == [("actions_plan", "F"), ("actions_followup", "F")]


def test_actions_followup_strong_opt_in(monkeypatch):
    assert _planner_models(monkeypatch, "strong") == [("actions_plan", "F"), ("actions_followup", "S")]


# --- a rejected key must fail over, not silently degrade -------------------

def test_auth_error_fails_over_and_disables_provider(monkeypatch, tmp_path):
    from tests.test_llm_providers import _status_error
    bad, good = _prov("gem", "lite", "lite"), _prov("groq", "g-big", "g-small")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [bad, good])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": ["groq", "gem"], "fast": ["gem", "groq"]})
    cbad = MagicMock()
    cbad.chat.completions.with_raw_response.create.side_effect = _status_error(
        "Please pass a valid API key", status=400)
    monkeypatch.setattr(bad, "_client", cbad)
    cgood = _client(good, monkeypatch)

    r1 = chat_json("s", "u1", llm_mod.get_model_fast(), cache_dir=str(tmp_path), reasoning_effort="low")
    assert r1 == {"ok": True}
    assert cbad.chat.completions.with_raw_response.create.call_count == 1  # no wasted param-retry
    assert {"strong", "fast"} <= bad.exhausted_roles

    r2 = chat_json("s", "u2", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    assert r2 == {"ok": True}
    assert cbad.chat.completions.with_raw_response.create.call_count == 1  # not retried on later calls
    assert cgood.chat.completions.with_raw_response.create.call_count == 2


def test_persistent_5xx_fails_over_to_next_provider(monkeypatch, tmp_path):
    from tests.test_llm_providers import _status_error
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    flaky, good = _prov("gem", "lite", "lite"), _prov("groq", "g-big", "g-small")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [flaky, good])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": ["gem", "groq"]})
    cf = MagicMock()
    cf.chat.completions.with_raw_response.create.side_effect = _status_error("boom", status=503)
    monkeypatch.setattr(flaky, "_client", cf)
    _client(good, monkeypatch)
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert "fast" not in flaky.exhausted_roles  # transient: not disabled, just skipped for this call


# --- generic OpenAI-compatible providers ("custom", "openai") ---------------

def _clear_provider_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith(("CUSTOM_", "OPENAI_", "GEMINI_", "LLM_PROVIDERS")):
            monkeypatch.delenv(k, raising=False)


def test_custom_provider_built_from_env_and_is_only_one(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(llm_mod, "GROQ_API_KEY", None)
    monkeypatch.setenv("LLM_PROVIDERS", "custom")
    monkeypatch.setenv("CUSTOM_API_KEY", "fake-custom-key")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("CUSTOM_MODEL_STRONG", "big-1")
    monkeypatch.setenv("CUSTOM_MODEL_FAST", "small-1")
    llm_mod.reload_providers()
    try:
        assert [p.name for p in llm_mod._PROVIDERS] == ["custom"]
        c = _client(llm_mod._PROVIDERS[0], monkeypatch)
        assert llm_mod.get_model_strong() == "big-1" and llm_mod.get_model_fast() == "small-1"
        assert chat_json("s", "u", llm_mod.get_model_strong(), cache_dir=str(tmp_path)) == {"ok": True}
        assert c.chat.completions.with_raw_response.create.call_args.kwargs["model"] == "big-1"
    finally:
        monkeypatch.undo()
        llm_mod.reload_providers()


def test_openai_provider_defaults_and_single_model(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDERS", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENAI_MODEL_FAST", "gpt-mini")
    try:
        built = llm_mod._build_providers()
        assert built[0].base_url == "https://api.openai.com/v1"
        assert built[0].model_strong == built[0].model_fast == "gpt-mini"
    finally:
        monkeypatch.undo()


def test_custom_without_base_url_is_skipped(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDERS", "custom")
    monkeypatch.setenv("CUSTOM_API_KEY", "fake")
    monkeypatch.setenv("CUSTOM_MODEL_FAST", "m")
    assert llm_mod._build_providers() == []


def test_failover_from_groq_to_custom_on_daily_quota(monkeypatch, tmp_path):
    from tests.test_llm_providers import _rate_limit_error
    g, c = _prov("groq", "g-big", "g-small"), _prov("custom", "c-big", "c-small")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [g, c])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    cg = MagicMock()
    cg.chat.completions.with_raw_response.create.side_effect = _rate_limit_error(
        "Rate limit reached ... tokens per day (TPD): Limit 100000. Please try again in 30m.")
    monkeypatch.setattr(g, "_client", cg)
    cc = _client(c, monkeypatch)
    assert chat_json("s", "u", llm_mod.get_model_strong(), cache_dir=str(tmp_path)) == {"ok": True}
    assert "strong" in g.exhausted_roles
    assert cc.chat.completions.with_raw_response.create.call_args.kwargs["model"] == "c-big"


def test_response_format_dropped_and_json_extracted_from_text(monkeypatch, tmp_path):
    from tests.test_llm_providers import _status_error
    p = _prov("custom", "m", "m")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    seen = []

    def create(**kw):
        seen.append(dict(kw))
        if "response_format" in kw:
            raise _status_error("Invalid parameter: 'response_format' of type 'json_object' is not supported", status=400)
        return _raw_response(_fake_openai_response('Sure! Here it is:\n```json\n{"ok": true}\n```'))

    c = MagicMock()
    c.chat.completions.with_raw_response.create.side_effect = create
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("sys", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert "response_format" in p.unsupported_params
    assert "JSON" in seen[-1]["messages"][0]["content"]
    # remembered: the next call doesn't even try it
    chat_json("sys", "u2", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    assert "response_format" not in seen[-1]


def test_all_three_optional_params_dropped_in_one_call(monkeypatch, tmp_path):
    """seed, reasoning_effort and response_format all rejected one after the
    other must still end in a success, not exhaust the retry budget."""
    from tests.test_llm_providers import _status_error
    p = _prov("custom", "m", "m")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})

    def create(**kw):
        if "seed" in kw:
            raise _status_error("Unknown field: seed", status=400)
        if "reasoning_effort" in kw:
            raise _status_error("reasoning_effort is not supported", status=400)
        if "response_format" in kw:
            raise _status_error("response_format unsupported", status=400)
        return _raw_response(_fake_openai_response())

    c = MagicMock()
    c.chat.completions.with_raw_response.create.side_effect = create
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path), reasoning_effort="low") == {"ok": True}
    assert p.unsupported_params >= {"seed", "response_format"}


def test_fake_provider_call_never_touches_real_ledger(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    real = Path(llm_mod.DEFAULT_LEDGER_PATH)
    before = real.read_bytes() if real.exists() else None
    p = _prov("alpha", "a", "a")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    _client(p, monkeypatch)
    chat_json("s", "u-ledger", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    assert (real.read_bytes() if real.exists() else None) == before
    redirected = Path(os.environ["USAGE_LEDGER_PATH"])
    assert redirected != real and redirected.exists()
    row = __import__("json").loads(redirected.read_text().splitlines()[-1])
    assert row["provider"] == "alpha" and row["base_url"] == "https://alpha.invalid" and row["model"] == "a"


def test_llm_cache_dir_env_is_used_when_no_explicit_dir(monkeypatch, tmp_path):
    p = _prov("alpha", "a", "a")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    c = _client(p, monkeypatch)
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cold"))
    chat_json("s", "u", llm_mod.get_model_fast())
    assert list((tmp_path / "cold").glob("*.json"))
    chat_json("s", "u", llm_mod.get_model_fast())  # second call is a cache hit
    assert c.chat.completions.with_raw_response.create.call_count == 1


def test_empty_result_is_not_cached_and_old_empty_entries_are_misses(monkeypatch, tmp_path):
    p = _prov("alpha", "a", "a")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    c = MagicMock()
    c.chat.completions.with_raw_response.create.side_effect = [
        _raw_response(_fake_openai_response('{"ok": tru')),   # truncated -> {}
        _raw_response(_fake_openai_response('{"ok": true}')),
    ]
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {}
    assert not list(tmp_path.glob("*.json"))            # failure not cached
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert c.chat.completions.with_raw_response.create.call_count == 2
    # a legacy cached {} is ignored too
    for f in tmp_path.glob("*.json"):
        f.write_text("{}")
    c.chat.completions.with_raw_response.create.side_effect = [_raw_response(_fake_openai_response('{"ok": 1}'))]
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": 1}


# --- same-provider model fallback on daily quota ---------------------------

def _two_model_provider(monkeypatch, name="groq"):
    p = Provider(name=name, api_key="k", base_url="https://x.invalid", model_strong="big", model_fast="small",
                 fallbacks_fast=["big"], fallbacks_strong=["small"])
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    return p


def _daily():
    from tests.test_llm_providers import _rate_limit_error
    return _rate_limit_error("Rate limit reached ... tokens per day (TPD): Limit 200000. Please try again in 30m.")


def test_daily_quota_switches_to_fallback_model_on_same_provider(monkeypatch, tmp_path):
    p = _two_model_provider(monkeypatch)
    used = []

    def create(**kw):
        used.append(kw["model"])
        if kw["model"] == "small":
            raise _daily()
        return _raw_response(_fake_openai_response())

    c = MagicMock()
    c.chat.completions.with_raw_response.create.side_effect = create
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("s", "u1", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    assert used == ["small", "big"]
    assert "small" in p.exhausted_models and "fast" not in p.exhausted_roles
    # exhausted for the whole process: the next call goes straight to the fallback
    assert llm_mod.get_model_fast() == "big"
    chat_json("s", "u2", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    assert used == ["small", "big", "big"]


def test_fallback_result_is_cached_under_the_model_actually_used(monkeypatch, tmp_path):
    import hashlib, json
    p = _two_model_provider(monkeypatch)
    p.exhausted_models.add("small")
    c = MagicMock()
    c.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response())
    monkeypatch.setattr(p, "_client", c)
    chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path))
    assert c.chat.completions.with_raw_response.create.call_args.kwargs["model"] == "big"
    key = hashlib.sha256(json.dumps(["groq", "s", "u", "big", llm_mod.config.LLM_SEED, 500, None]).encode()).hexdigest()
    assert (tmp_path / f"{key}.json").exists()


def test_all_models_exhausted_degrades_and_cache_still_serves(monkeypatch, tmp_path):
    p = _two_model_provider(monkeypatch)
    # warm one entry while healthy
    c = MagicMock()
    c.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response())
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("s", "warm", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}
    c2 = MagicMock()
    c2.chat.completions.with_raw_response.create.side_effect = lambda **kw: (_ for _ in ()).throw(_daily())
    monkeypatch.setattr(p, "_client", c2)
    assert chat_json("s", "cold", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) is None
    assert {"small", "big"} <= p.exhausted_models and {"strong", "fast"} <= p.exhausted_roles
    calls = c2.chat.completions.with_raw_response.create.call_count
    assert calls == 2                                           # one per model, then nothing more
    assert chat_json("s", "cold2", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) is None
    assert c2.chat.completions.with_raw_response.create.call_count == calls  # no further requests


def test_fallback_env_parsing_and_groq_defaults(monkeypatch):
    for k in ("GROQ_MODEL_FAST_FALLBACKS", "GROQ_MODEL_STRONG_FALLBACKS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.delenv("LLM_PROVIDERS_STRONG", raising=False)
    monkeypatch.delenv("LLM_PROVIDERS_FAST", raising=False)
    monkeypatch.setattr(llm_mod, "GROQ_API_KEY", "k")
    g = llm_mod._build_providers()[0]
    assert g.chain("fast") == [g.model_fast, g.model_strong] and g.chain("strong") == [g.model_strong, g.model_fast]
    monkeypatch.setenv("GROQ_MODEL_FAST_FALLBACKS", "")
    monkeypatch.setenv("GROQ_MODEL_STRONG_FALLBACKS", "x-model, y-model")
    g = llm_mod._build_providers()[0]
    assert g.chain("fast") == [g.model_fast] and g.chain("strong") == [g.model_strong, "x-model", "y-model"]


def test_frozen_strong_blocks_fast_to_strong_fallback(monkeypatch, tmp_path):
    p = _two_model_provider(monkeypatch)
    p.exhausted_models.add("small")
    c = MagicMock()
    c.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response())
    monkeypatch.setattr(p, "_client", c)
    monkeypatch.setenv("LLM_FROZEN_ROLES", "strong")
    assert chat_json("s", "u", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) is None
    c.chat.completions.with_raw_response.create.assert_not_called()
