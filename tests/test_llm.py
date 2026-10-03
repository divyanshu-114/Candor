"""Tests for LLM utilities and robust JSON extraction."""
import time
from unittest.mock import MagicMock, patch

import httpx
from openai import RateLimitError

from memory.llm import _extract_json, _is_daily_quota_exhausted, _parse_duration, chat_json

def test_extract_json_clean():
    text = '{"ranked": ["a", "b"]}'
    assert _extract_json(text) == {"ranked": ["a", "b"]}

def def_extract_json_markdown():
    text = '''```json
{
    "ranked": ["x"]
}
```'''
    assert _extract_json(text) == {"ranked": ["x"]}

def test_extract_json_garbage_around():
    text = 'Here is your json: {"ranked": [1, 2]} Hope this helps!'
    assert _extract_json(text) == {"ranked": [1, 2]}

def test_extract_json_invalid():
    text = "Sorry, I can't do that."
    assert _extract_json(text) == {}

def test_parse_duration_formats():
    assert _parse_duration("622ms") == 0.622
    assert _parse_duration("7.66s") == 7.66
    assert abs(_parse_duration("1h9m7.2s") - 4147.2) < 1e-6

def _rate_limit_error(message: str) -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return RateLimitError(message, response=response, body=None)

def test_is_daily_quota_exhausted_detects_tpd_message():
    tpd = _rate_limit_error("Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199999")
    tpm = _rate_limit_error("Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7900")
    assert _is_daily_quota_exhausted(tpd) is True
    assert _is_daily_quota_exhausted(tpm) is False

def test_chat_json_degrades_immediately_on_daily_quota_without_sleeping(tmp_path, monkeypatch):
    """A TPD 429 must return None fast, not sleep out a multi-minute Retry-After."""
    import memory.llm as llm_mod
    provider = llm_mod.Provider(name="fake-provider", api_key="fake-key", base_url="https://example.invalid",
                                 model_strong="fake-strong", model_fast="fake-fast")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [provider])
    monkeypatch.setattr(llm_mod, "_ALL_EXHAUSTED_LOGGED", set())

    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.side_effect = _rate_limit_error(
        "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 199999. "
        "Please try again in 22m5.376s."
    )
    monkeypatch.setattr(provider, "_client", fake_client)

    started = time.monotonic()
    result = chat_json("sys", "user", provider.model_fast, cache_dir=str(tmp_path))
    elapsed = time.monotonic() - started
    assert result is None
    assert elapsed < 2.0, f"daily-quota 429 should fail fast, took {elapsed:.1f}s"
    fake_client.chat.completions.with_raw_response.create.assert_called_once()
    assert "fast" in provider.exhausted_roles
