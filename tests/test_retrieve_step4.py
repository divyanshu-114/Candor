"""Tests for the Step 4 retrieval upgrades (synonyms, HyDE sketch, date
agenda, date resolver), each gated behind its own config flag.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from memory import config
from memory.retrieve import _change_reason_query, _date_anchored_candidates, retrieve
from memory.store import MemoryStore


def test_change_reason_query_adds_synonyms_when_trigger_present():
    expanded = _change_reason_query("Why did the launch get delayed?")
    assert "regression" in expanded
    assert "blocker" in expanded
    # original question still present (and at higher weight via duplication)
    assert expanded.count("Why did the launch get delayed?") == 2


def test_change_reason_query_is_a_no_op_without_a_trigger_word():
    q = "What is the pricing for the starter tier?"
    assert _change_reason_query(q) == q


def _build_dataset(d: Path) -> None:
    (d / "connectors/slack").mkdir(parents=True)
    (d / "connectors/google_calendar").mkdir(parents=True)
    (d / "connectors/gmail").mkdir(parents=True)
    (d / "native/dictation").mkdir(parents=True)
    with open(d / "connectors/slack/messages.jsonl", "w") as f:
        # Deliberately on the target date too -- step 4's date agenda spec
        # only names calendar/email/dictation/meeting, not Slack, so this
        # must NOT show up in the result despite matching by date.
        f.write(json.dumps({"id": "SL-1", "user": "U1", "ts": "2026-09-23T08:00:00-07:00",
                             "text": "Heading to the airport now."}) + "\n")
    with open(d / "native/dictation/dictations.jsonl", "w") as f:
        f.write(json.dumps({"id": "DCT-1", "timestamp": "2026-09-23T07:00:00-07:00", "mode": "note_to_self",
                             "target_app": "Notes", "target_context": "", "raw_transcript": "don't forget board prep",
                             "cleaned_text": "Don't forget board prep.", "delivery_state": "saved_note"}) + "\n")
    with open(d / "connectors/google_calendar/events.jsonl", "w") as f:
        f.write(json.dumps({"id": "CAL-BOARD", "summary": "Board meeting",
                             "start": {"dateTime": "2026-09-23T09:00:00-07:00"},
                             "end": {"dateTime": "2026-09-23T12:00:00-07:00"},
                             "organizer": "alex@example.com", "attendees": [], "status": "confirmed",
                             "created": "2026-09-01T00:00:00-07:00", "updated": "2026-09-17T00:00:00-07:00"}) + "\n")
        f.write(json.dumps({"id": "CAL-OTHERDAY", "summary": "Unrelated meeting",
                             "start": {"dateTime": "2026-09-20T09:00:00-07:00"},
                             "end": {"dateTime": "2026-09-20T10:00:00-07:00"},
                             "organizer": "alex@example.com", "attendees": [], "status": "confirmed",
                             "created": "2026-09-01T00:00:00-07:00", "updated": "2026-09-01T00:00:00-07:00"}) + "\n")
    with open(d / "connectors/gmail/messages.jsonl", "w") as f:
        f.write(json.dumps({"id": "EM-FLIGHT", "thread_id": "TH-1", "date": "2026-09-15T10:00:00-07:00",
                             "from": "united@example.com", "to": ["alex@example.com"], "cc": [],
                             "subject": "Your flight confirmation", "body": "Flight on Sep 23.",
                             "labels": []}) + "\n")


def test_date_anchored_candidates_matches_calendar_by_overlap_and_others_by_delivery_date():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _build_dataset(d)
        store = MemoryStore(str(d))
        visible = store.visible("2026-09-24T00:00:00-07:00")
        ids = _date_anchored_candidates(["2026-09-23"], visible, cap=12)
        assert "CAL-BOARD" in ids
        assert "CAL-OTHERDAY" not in ids
        assert "DCT-1" in ids  # dictation delivered on the target date
        assert "SL-1" not in ids  # spec names calendar/email/dictation/meeting, not Slack
        assert "EM-FLIGHT" not in ids  # delivered Sep 15, not the target date


def test_date_anchored_candidates_empty_for_unparseable_dates():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _build_dataset(d)
        store = MemoryStore(str(d))
        visible = store.visible("2026-09-24T00:00:00-07:00")
        assert _date_anchored_candidates(["not-a-date"], visible, cap=12) == []


def test_retrieve_with_all_step4_flags_on_still_returns_only_visible_ids():
    """End-to-end smoke test: flipping every Step 4 flag on, with a mocked
    LLM, must never leak a forbidden id and must not raise.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _build_dataset(d)

        def fake_analysis(*_a, **_k):
            return {
                "intent": "schedule", "wants_latest": True, "wants_history": False,
                "sub_queries": ["board meeting", "flight"], "entities": [], "dates": ["2026-09-23"],
                "ambiguous_person": None, "needs_followup": True,
                "answer_sketch": "The board meeting is at 9am and the flight leaves in the evening.",
            }

        def fake_date_resolver(*_a, **_k):
            return {"resolved_dates": ["2026-09-23"], "extra_queries": ["flight evening"]}

        old = {name: getattr(config, name) for name in
               ("USE_ANALYSIS", "USE_MULTIQUERY", "USE_NEIGHBORS", "USE_HOP2", "USE_RERANK",
                "USE_CHANGE_REASON_SYNONYMS", "USE_HYDE_ANSWER_SKETCH", "USE_DATE_AGENDA", "USE_DATE_RESOLVER")}
        config.USE_ANALYSIS = config.USE_MULTIQUERY = config.USE_NEIGHBORS = True
        config.USE_HOP2 = config.USE_RERANK = True
        config.USE_CHANGE_REASON_SYNONYMS = config.USE_HYDE_ANSWER_SKETCH = True
        config.USE_DATE_AGENDA = config.USE_DATE_RESOLVER = True
        try:
            with patch("memory.query.chat_json", side_effect=fake_analysis), \
                 patch("memory.retrieve.chat_json", side_effect=fake_date_resolver):
                ranked = retrieve("What's on my calendar the day I fly?", "2026-09-24T00:00:00-07:00",
                                   data_dir=str(d), cache_dir=str(d / ".cache"))
        finally:
            for name, value in old.items():
                setattr(config, name, value)

        visible_ids = {u.id for u in MemoryStore(str(d)).visible("2026-09-24T00:00:00-07:00")}
        ids = [uid for uid, _score in ranked]
        assert ids, "Step 4 flags combined should still return some results"
        assert all(uid in visible_ids for uid in ids)
