"""Step 0: multi-provider failover fault-injection tests (fake HTTP errors,
no network). Covers: TPM 429 then success, TPD 429 then failover, all
providers exhausted, 400 on an unsupported parameter, model_not_found.
"""
import time
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError, RateLimitError

import memory.llm as llm_mod
from memory.llm import Provider, chat_json


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/chat/completions"))


def _rate_limit_error(message: str, status: int = 429) -> RateLimitError:
    return RateLimitError(message, response=_response(status), body=None)


def _status_error(message: str, status: int) -> APIStatusError:
    return APIStatusError(message, response=_response(status), body=None)


def _fake_openai_response(content: str = '{"ok": true}'):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _raw_response(parsed, headers=None):
    raw = MagicMock()
    raw.headers = headers or {}
    raw.parse.return_value = parsed
    return raw


def _provider(name: str, model_strong="strong-model", model_fast="fast-model") -> Provider:
    return Provider(name=name, api_key=f"fake-{name}-key", base_url=f"https://{name}.invalid",
                     model_strong=model_strong, model_fast=model_fast)


@pytest.fixture(autouse=True)
def _reset_providers(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, "_ALL_EXHAUSTED_LOGGED", set())
    monkeypatch.setattr(llm_mod, "_RESOLVED_MODELS", {})
    yield


# (a) TPM 429 (short Retry-After) then success, same provider, no failover.
def test_tpm_429_then_success_same_provider(monkeypatch, tmp_path):
    p = _provider("alpha")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])

    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _rate_limit_error("Rate limit reached ... tokens per minute (TPM). Please try again in 0.5s.")
        return _raw_response(_fake_openai_response())

    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.side_effect = side_effect
    monkeypatch.setattr(p, "_client", fake_client)

    started = time.monotonic()
    result = chat_json("sys", "user", p.model_fast, cache_dir=str(tmp_path))
    elapsed = time.monotonic() - started

    assert result == {"ok": True}
    assert call_count["n"] == 2
    assert elapsed < 2.0, "a short TPM 429 must not block for long"
    assert "fast" not in p.exhausted_roles


# (b) TPD 429 (daily quota) -> immediate failover to the next provider.
def test_tpd_429_fails_over_to_next_provider(monkeypatch, tmp_path):
    p1, p2 = _provider("alpha"), _provider("beta")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p1, p2])

    client1 = MagicMock()
    client1.chat.completions.with_raw_response.create.side_effect = _rate_limit_error(
        "Rate limit reached ... tokens per day (TPD): Limit 200000, Used 199999. Please try again in 22m.")
    monkeypatch.setattr(p1, "_client", client1)

    client2 = MagicMock()
    client2.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response())
    monkeypatch.setattr(p2, "_client", client2)

    result = chat_json("sys", "user", p1.model_fast, cache_dir=str(tmp_path))

    assert result == {"ok": True}
    assert "fast" in p1.exhausted_roles
    assert "fast" not in p2.exhausted_roles
    client1.chat.completions.with_raw_response.create.assert_called_once()
    client2.chat.completions.with_raw_response.create.assert_called_once()


# all providers exhausted -> None, logged once (not per call).
def test_all_providers_exhausted_returns_none_and_logs_once(monkeypatch, tmp_path, caplog):
    p1, p2 = _provider("alpha"), _provider("beta")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p1, p2])

    def exhausted(**kwargs):
        raise _rate_limit_error("Rate limit reached ... tokens per day (TPD). Please try again in 30m.")

    for p in (p1, p2):
        client = MagicMock()
        client.chat.completions.with_raw_response.create.side_effect = exhausted
        monkeypatch.setattr(p, "_client", client)

    import logging
    with caplog.at_level(logging.ERROR, logger="memory.llm"):
        r1 = chat_json("sys", "user1", p1.model_fast, cache_dir=str(tmp_path))
        r2 = chat_json("sys", "user2", p1.model_fast, cache_dir=str(tmp_path))

    assert r1 is None and r2 is None
    assert "fast" in p1.exhausted_roles and "fast" in p2.exhausted_roles
    all_exhausted_logs = [rec for rec in caplog.records if "All configured providers exhausted" in rec.message]
    assert len(all_exhausted_logs) == 1, "must log 'all exhausted' once, not once per question"


# (c) 400 caused by an unsupported parameter (reasoning_effort) -> dropped, retried once, succeeds.
def test_unsupported_param_dropped_and_retried(monkeypatch, tmp_path):
    p = _provider("alpha")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])

    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if "reasoning_effort" in kwargs:
            raise _status_error("Invalid parameter: 'reasoning_effort' is not supported", status=400)
        return _raw_response(_fake_openai_response())

    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = side_effect
    monkeypatch.setattr(p, "_client", client)

    result = chat_json("sys", "user", p.model_fast, cache_dir=str(tmp_path), reasoning_effort="low")

    assert result == {"ok": True}
    assert call_count["n"] == 2
    assert "reasoning_effort:fast" in p.unsupported_params


# Rejecting reasoning_effort on one role (e.g. a "lite" fast model with no
# thinking mode) must not disable it for the other role too -- observed live
# against Gemini, where the strong model needs reasoning_effort to avoid
# burning its whole max_tokens budget on hidden "thinking" tokens.
def test_reasoning_effort_rejection_is_scoped_per_role(monkeypatch, tmp_path):
    p = _provider("alpha")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])

    def side_effect(**kwargs):
        if kwargs["model"] == p.model_fast and "reasoning_effort" in kwargs:
            raise _status_error("Invalid parameter: 'reasoning_effort' is not supported", status=400)
        return _raw_response(_fake_openai_response())

    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = side_effect
    monkeypatch.setattr(p, "_client", client)

    fast_result = chat_json("sys", "user", p.model_fast, cache_dir=str(tmp_path), reasoning_effort="low")
    assert fast_result == {"ok": True}
    assert "reasoning_effort:fast" in p.unsupported_params

    strong_kwargs_seen = []
    real_create = client.chat.completions.with_raw_response.create.side_effect

    def recording_side_effect(**kwargs):
        if kwargs["model"] == p.model_strong:
            strong_kwargs_seen.append(kwargs)
        return real_create(**kwargs)

    client.chat.completions.with_raw_response.create.side_effect = recording_side_effect
    strong_result = chat_json("sys2", "user2", p.model_strong, cache_dir=str(tmp_path), reasoning_effort="medium")

    assert strong_result == {"ok": True}
    assert "reasoning_effort" in strong_kwargs_seen[0], "strong role must still get reasoning_effort after fast role's rejection"
    assert "reasoning_effort:strong" not in p.unsupported_params


# A provider (e.g. Gemini's OpenAI-compatible endpoint) that rejects `seed`
# with a real HTTP 400 (not a client-side TypeError) must have `seed`
# dropped and the request retried, not fail the whole call over.
def test_seed_rejected_as_http_400_dropped_and_retried(monkeypatch, tmp_path):
    p = _provider("alpha")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])

    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if "seed" in kwargs:
            raise _status_error('Invalid JSON payload received. Unknown name "seed": Cannot find field.', status=400)
        return _raw_response(_fake_openai_response())

    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = side_effect
    monkeypatch.setattr(p, "_client", client)

    result = chat_json("sys", "user", p.model_fast, cache_dir=str(tmp_path))

    assert result == {"ok": True}
    assert call_count["n"] == 2
    assert "seed" in p.unsupported_params


# (c) model_not_found -> discovery runs, retries once with the resolved model.
def test_model_not_found_runs_discovery_and_retries(monkeypatch, tmp_path):
    p = _provider("alpha", model_fast="old-fast-model")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])

    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if kwargs["model"] == "old-fast-model":
            raise _status_error("The model `old-fast-model` does not exist", status=404)
        return _raw_response(_fake_openai_response())

    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = side_effect
    client.models.list.return_value = MagicMock(data=[MagicMock(id="new-20b-model")])
    monkeypatch.setattr(p, "_client", client)
    monkeypatch.setattr("memory.config.MODEL_PREFERENCE_FAST", ["20b"])

    result = chat_json("sys", "user", "old-fast-model", cache_dir=str(tmp_path))

    assert result == {"ok": True}
    assert call_count["n"] == 2
    assert p.model_fast == "new-20b-model"
