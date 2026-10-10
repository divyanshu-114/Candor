"""Real credentials in the raw data stay masked everywhere the system can read them (checked without printing any value)."""
import json
import re
from pathlib import Path

from memory.safety import iter_secret_matches
from memory.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
KEY_SHAPES = re.compile(r"gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{16,}|xox[bpoa]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}")


def _raw_texts():
    d = ROOT / "data"
    for f in (d / "native/meetings").glob("*.json"):
        for s in json.loads(f.read_text())["segments"]:
            yield s["text"]
    for line in (d / "native/dictation/dictations.jsonl").read_text().splitlines():
        r = json.loads(line)
        yield r.get("raw_transcript", "")
        yield r.get("cleaned_text", "")
    for p in ("connectors/slack/messages.jsonl", "connectors/gmail/messages.jsonl", "connectors/google_calendar/events.jsonl"):
        for line in (d / p).read_text().splitlines():
            r = json.loads(line)
            yield r.get("text") or r.get("body") or r.get("description") or ""
    for f in (d / "connectors/codex/sessions").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            yield (r.get("content") or "") + (r.get("input") or "") + (r.get("output") or "")
    for c in json.loads((d / "connectors/chatgpt/conversations.json").read_text()):
        for m in c["messages"]:
            yield m["content"]


def test_every_credential_found_in_the_raw_data_is_absent_from_every_unit():
    store = MemoryStore(str(ROOT / "data"))
    corpus = "\n".join(u.text for u in store.units)
    found = [v for t in _raw_texts() for rule, v in iter_secret_matches(t) if rule != "connection-string"]
    assert found, "the data contains at least one real credential (a pasted staging key)"
    assert all(v not in corpus for v in found)


def test_the_pasted_key_is_still_detected_by_the_precise_rules(monkeypatch):
    from memory import config
    monkeypatch.setattr(config, "USE_PRECISE_MASKING", True)
    rules = [rule for t in _raw_texts() for rule, _v in iter_secret_matches(t)]
    assert "openai-key" in rules


def test_no_key_shaped_string_survives_ingestion_anywhere():
    store = MemoryStore(str(ROOT / "data"))
    leaked = [u.id for u in store.units if KEY_SHAPES.search(u.text) or KEY_SHAPES.search(json.dumps(u.meta, default=str))]
    assert leaked == []
