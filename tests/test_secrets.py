"""Tests for memory.safety secret masking — project rule 4.

We verify GENERAL pattern matching (not hard-coded IDs or values), and confirm
that ingestion masks real secrets before they land in the store.
"""
from __future__ import annotations

import json

from memory.safety import mask_secrets, iter_secret_matches


def test_api_key_prefix_masked() -> None:
    """Prefix-based rule: any sk-… key must be REDACTED."""
    assert "[REDACTED-SECRET]" in mask_secrets("api key: sk-test-not-a-real-secret-value")


def test_password_phrase_masked() -> None:
    """Keyword-in-context rule: 'password is <word>' must be REDACTED."""
    assert "[REDACTED-SECRET]" in mask_secrets("the password is sample-word")


def test_spoken_key_masked() -> None:
    """Spoken-secret rule: 'key is s k dash …' must be REDACTED."""
    assert "[REDACTED-SECRET]" in mask_secrets("the key is s k dash sample value")


def test_file_paths_not_masked() -> None:
    """Ordinary file paths must NOT be masked (false positive prevention)."""
    text = "token: /usr/local/bin/python3"
    # The path /usr/... should NOT be redacted
    result = mask_secrets(text)
    assert "/usr/local/bin/python3" in result


def test_hex_hash_not_masked() -> None:
    """Long hex strings (hashes) must NOT be masked."""
    import re
    text = "token: " + "a" * 40
    result = mask_secrets(text)
    # The 40-char hex hash should not be replaced
    assert "[REDACTED-SECRET]" not in result


def test_real_detected_secret_values_are_absent_after_ingestion() -> None:
    """Inspect values only in memory; never emit the raw value into test output.

    This test verifies that any sk-… prefix found in raw Slack messages is
    absent from every unit in the store — proving mask_secrets() ran at ingest.
    """
    from memory.store import MemoryStore

    raw = [
        json.loads(line)
        for line in open("data/connectors/slack/messages.jsonl")
        if line.strip()
    ]
    values = [
        value
        for record in raw
        for _rule, value in iter_secret_matches(record.get("text", ""))
    ]
    # There must be at least one match (the known sk-… key in SL-DM-AB-0915-2)
    assert values, "expected at least one secret match in raw Slack data"

    store = MemoryStore("./data")
    corpus = "\n".join(unit.text for unit in store.units)
    # None of the raw secret values should appear verbatim in the store corpus
    for value in values:
        assert value not in corpus, f"raw secret value found in store corpus"
