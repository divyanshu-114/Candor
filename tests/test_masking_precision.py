"""Precise masking (USE_PRECISE_MASKING): real credentials still masked, ordinary prose and code left alone.

Examples are synthetic; none is taken from the data or from an eval file.
"""
import pytest

from memory import config
from memory.safety import mask_secrets

MASK = "[REDACTED-SECRET]"


@pytest.fixture(autouse=True)
def _precise(monkeypatch):
    monkeypatch.setattr(config, "USE_PRECISE_MASKING", True)


@pytest.mark.parametrize("text", [
    "Today it's email and password with two-factor. But SSO is on our Q4 roadmap.",
    "rows = sorted(items, key=lambda r: r.score)",
    "Foreign-key constraints:\n    \"orders_customer_fkey\" FOREIGN KEY (customer_id) REFERENCES customers(id)",
    "the token is expired, please refresh it",
    "ask them what the login is for",
])
def test_prose_and_code_are_not_masked(text):
    assert MASK not in mask_secrets(text)


@pytest.mark.parametrize("text", [
    "the password is sample-word",
    "the key is s k dash sample value",
    "api key: sk-test-not-a-real-secret-value",
    "token = Zx9fQ2LmPq81",
    "password: Hunter2Hunter2",
    "secret is correct-horse-battery",
])
def test_credentials_are_still_masked(text):
    assert MASK in mask_secrets(text)


def test_flag_off_restores_v1_behaviour(monkeypatch):
    monkeypatch.setattr(config, "USE_PRECISE_MASKING", False)
    assert MASK in mask_secrets("Today it's email and password with two-factor. But SSO is on our Q4 roadmap.")
