"""Provider quirks (Phase 6): parameter names/support differ across OpenAI-compatible services, and reply formats vary.
Fake clients only; no network."""
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError

import memory.llm as llm_mod
from memory.llm import Provider, _extract_json, chat_json


def _status_error(message: str, status: int = 400) -> APIStatusError:
    return APIStatusError(message, response=httpx.Response(status, request=httpx.Request("POST", "https://x.invalid/c")), body=None)


def _reply(content="", **extra):
    msg = MagicMock(content=content)
    for k, v in extra.items():
        setattr(msg, k, v)
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=7, completion_tokens=3)
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = resp
    return raw


def _provider(monkeypatch, handler):
    p = Provider(name="quirky", api_key="fake", base_url="https://quirky.invalid", model_strong="m-strong", model_fast="m-fast")
    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = handler
    monkeypatch.setattr(p, "_client", client)
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ALL_EXHAUSTED_LOGGED", set())
    return p


def test_max_tokens_rejected_switches_to_max_completion_tokens(monkeypatch, tmp_path):
    seen = []

    def handler(**kw):
        seen.append(set(kw))
        if "max_tokens" in kw:
            raise _status_error("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
        return _reply('{"ok": true}')

    p = _provider(monkeypatch, handler)
    assert chat_json("s", "u", "m-fast", cache_dir=str(tmp_path), max_tokens=50) == {"ok": True}
    assert "max_tokens" in p.unsupported_params and "max_completion_tokens" in seen[-1]
    seen.clear()
    chat_json("s", "u2", "m-fast", cache_dir=str(tmp_path), max_tokens=50)          # remembered: no second failed attempt
    assert len(seen) == 1 and "max_tokens" not in seen[0]


def test_temperature_rejected_is_dropped(monkeypatch, tmp_path):
    def handler(**kw):
        if "temperature" in kw:
            raise _status_error("Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) is supported.")
        return _reply('{"ok": 1}')

    p = _provider(monkeypatch, handler)
    assert chat_json("s", "u", "m-fast", cache_dir=str(tmp_path)) == {"ok": 1}
    assert "temperature" in p.unsupported_params


def test_answer_in_reasoning_field_when_content_is_empty(monkeypatch, tmp_path):
    _provider(monkeypatch, lambda **kw: _reply("", reasoning_content='thinking... {"ranked": ["a", "b"]} done'))
    assert chat_json("s", "u", "m-fast", cache_dir=str(tmp_path)) == {"ranked": ["a", "b"]}


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go:\n```\n{"a": {"b": 2}}\n```\nHope that helps.', {"a": {"b": 2}}),
    ('<think>maybe {"wrong": true} first</think>\n{"right": true}', {"right": True}),
    ('The set {x, y} is irrelevant. Answer: {"ok": [1, 2]} (done)', {"ok": [1, 2]}),
    ('no json here', {}),
    ('{"truncated": ', {}),
])
def test_extract_json_handles_provider_habits(text, expected):
    assert _extract_json(text) == expected


def test_reasoning_model_that_exhausts_its_budget_is_retried_with_more_tokens(monkeypatch, tmp_path):
    budgets = []

    def handler(**kw):
        budgets.append(kw.get("max_tokens") or kw.get("max_completion_tokens"))
        if budgets[-1] < 400:
            raw = _reply("", reasoning="long hidden thinking")
            raw.parse.return_value.choices[0].finish_reason = "length"
            return raw
        raw = _reply('{"ranked": ["a"]}')
        raw.parse.return_value.choices[0].finish_reason = "stop"
        return raw

    _provider(monkeypatch, handler)
    assert chat_json("s", "u", "m-strong", cache_dir=str(tmp_path), max_tokens=100) == {"ranked": ["a"]}
    assert budgets == [100, 200, 400]        # doubled twice, then enough


def test_truncation_retry_is_bounded(monkeypatch, tmp_path):
    calls = []

    def handler(**kw):
        calls.append(1)
        raw = _reply("")
        raw.parse.return_value.choices[0].finish_reason = "length"
        return raw

    _provider(monkeypatch, handler)
    assert chat_json("s", "u", "m-strong", cache_dir=str(tmp_path), max_tokens=100) in (None, {})
    assert len(calls) <= 4
