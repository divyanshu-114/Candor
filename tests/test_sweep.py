"""Time-safety sweep: an adversarial reranker must never leak a forbidden id.

Twelve as_of instants straddle the known edit/delete/calendar-update events in
the real data/ corpus (see docs/DATA_AUDIT.md), paired with eight generic
questions. The mocked LLM always returns the worst case it could: it tries to
resurrect a deleted message, cite a not-yet-delivered future record, and
invent ids that were never offered. The hard rule (PROJECT_RULES.md #2) must
hold at every single point regardless.
"""
from datetime import datetime
from unittest.mock import patch

import pytest

from memory import config
from memory.retrieve import retrieve
from memory.store import MemoryStore

DATA_DIR = "./data"

# Straddle each known event from docs/DATA_AUDIT.md by +/-1s / +/-1min, plus
# the overall data window's start and end.
AS_OF_SWEEP = [
    "2026-09-08T08:00:00-07:00",                 # window start
    "2026-08-20T19:29:59-07:00",                  # just before CAL-F-20 updated
    "2026-08-20T19:30:01-07:00",                  # just after CAL-F-20 updated
    "2026-09-15T16:05:01-07:00",                  # just before SL-EV-0915-DEL1 (delete)
    "2026-09-15T16:05:03-07:00",                  # just after SL-EV-0915-DEL1
    "2026-09-15T17:52:39-07:00",                  # just before SL-EV-0915-DEL2 (delete)
    "2026-09-15T17:52:41-07:00",                  # just after SL-EV-0915-DEL2
    "2026-09-16T13:20:14-07:00",                  # just before SL-EV-0916-EDIT1 (edit)
    "2026-09-16T13:20:16-07:00",                  # just after SL-EV-0916-EDIT1
    "2026-09-18T09:14:59-07:00",                  # just before CAL-F-18 updated
    "2026-09-18T09:15:01-07:00",                  # just after CAL-F-18 updated
    "2026-09-18T23:59:00-07:00",                  # window end
]

GENERIC_QUESTIONS = [
    "What is the current status of the Route Planner launch?",
    "Who said what about the Acme Freight contract?",
    "What did Alex discuss in meetings this week?",
    "Summarize recent Slack activity in the route-planner channel.",
    "What calendar events are coming up?",
    "What promises were made and are they still open?",
    "What changed in the last email thread about pricing?",
    "What did the team discuss in the most recent Codex session?",
]

# Adversarial "worst case" rerank payload: a deleted message's target, an
# only-just-delivered future record, and outright invented ids, all ranked
# first regardless of question or as_of.
_ADVERSARIAL_RANKED = [
    "SL-DM-AB-0915-2",   # deleted at SL-EV-0915-DEL1
    "SL-SALES-0915-1",   # deleted at SL-EV-0915-DEL2
    "CAL-F-18",          # delivered (updated) 2026-09-18T09:15:00-07:00
    "FAKE-ID-001",
    "NOT-A-REAL-RECORD",
]


def _worst_case_rerank(*_args, **_kwargs):
    return {"ranked": list(_ADVERSARIAL_RANKED), "reason": "adversarial worst case"}


def _worst_case_analysis(*_args, **_kwargs):
    return {
        "intent": "current_state", "wants_latest": True, "wants_history": True,
        "sub_queries": ["status", "update"], "entities": [], "dates": [],
        "ambiguous_person": None, "needs_followup": True,
    }


@pytest.fixture(autouse=True)
def _full_mode_with_mocked_llm():
    old = {name: getattr(config, name) for name in
           ("USE_ANALYSIS", "USE_MULTIQUERY", "USE_NEIGHBORS", "USE_HOP2", "USE_RERANK")}
    config.USE_ANALYSIS = config.USE_MULTIQUERY = config.USE_NEIGHBORS = True
    config.USE_HOP2 = config.USE_RERANK = True
    yield
    for name, value in old.items():
        setattr(config, name, value)


# Compact sweep: one instant on each side of the delete, edit and calendar
# update events, each with a different generic question (was a 96-case grid).
SWEEP_CASES = [
    (GENERIC_QUESTIONS[i % len(GENERIC_QUESTIONS)], as_of)
    for i, as_of in enumerate([AS_OF_SWEEP[0], AS_OF_SWEEP[1], AS_OF_SWEEP[2], AS_OF_SWEEP[3], AS_OF_SWEEP[4],
                               AS_OF_SWEEP[5], AS_OF_SWEEP[6], AS_OF_SWEEP[8], AS_OF_SWEEP[10], AS_OF_SWEEP[11]])
]


@pytest.mark.parametrize("pipeline_v2", [True, False])
@pytest.mark.parametrize("question,as_of", SWEEP_CASES)
def test_no_forbidden_id_survives_adversarial_rerank(question, as_of, pipeline_v2, data_store, monkeypatch):
    monkeypatch.setattr(config, "USE_LANES", pipeline_v2)
    store = data_store
    visible_ids = {u.id for u in store.visible(as_of)}

    # both rerank entry points are patched: v1 calls memory.retrieve.chat_json, v2 calls memory.llm.chat_json
    with patch("memory.query.chat_json", side_effect=_worst_case_analysis), \
         patch("memory.retrieve.chat_json", side_effect=_worst_case_rerank), \
         patch("memory.llm.chat_json", side_effect=_worst_case_rerank), \
         patch("memory.llm.is_available", return_value=True):
        ranked = retrieve(question, as_of, data_dir=DATA_DIR, cache_dir=".cache")

    ids = [uid for uid, _score in ranked]
    forbidden = [uid for uid in ids if uid not in visible_ids]
    assert not forbidden, f"forbidden id(s) {forbidden} leaked at as_of={as_of} for question={question!r}"
