"""Contracts of the v2 retrieval parts, on a tiny synthetic corpus (no network, no model downloads)."""
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from memory import anchor, chains, config, ledger, lanes, people
from memory.queries import build_variants, entity_query
from memory.store import MemoryStore

def test_fuse_is_deterministic_and_rewards_agreement():
    a, b = ["x", "y", "z"], ["y", "x", "w"]
    assert lanes.fuse([a, b]) == lanes.fuse([a, b])
    assert [u for u, _ in lanes.fuse([a, b])][:2] == ["x", "y"] or [u for u, _ in lanes.fuse([a, b])][:2] == ["y", "x"]


def test_lane_floor_keeps_a_rare_source_in_the_pool(monkeypatch):
    monkeypatch.setattr(config, "LANE_FLOOR", 1)
    source_of = {f"m{i}": "meeting" for i in range(30)} | {"c1": "codex"}
    bm25 = {f"m{i}": 10 - i * 0.1 for i in range(30)} | {"c1": 0.01}
    pool, lane_rank = lanes.build_pool([(bm25, {})], source_of, lane_k=30, pool_size=5)
    assert "c1" in [u for u, _ in pool], "the single codex hit must survive a pool that 30 meeting segments would fill"
    assert lane_rank["c1"]["codex"] == 1


def test_entity_query_keeps_names_numbers_and_drops_scaffolding():
    q = entity_query("What did Priya say about the 61 passing cases in Toronto?")
    assert "Priya" in q and "61" in q and "Toronto" in q
    assert "what" not in q.lower().split() and "did" not in q.lower().split()
    assert build_variants("plain question here", None)[0] == ("plain question here", 1.0)


def test_people_resolution_uses_the_rest_of_the_question(corpus):
    vis = {u.id for u in corpus.units}
    idf = {"pricing": 5.0, "proposal": 5.0, "geocoder": 5.0, "staging": 4.0, "fix": 3.0}
    assert people.ambiguous_first_names(corpus) == {"sarah": ["Sarah Kim", "Sarah Patel"]}
    assert people.resolve("Has Sarah sent the pricing proposal?", vis, corpus, idf)["resolved"] == "Sarah Patel"
    assert people.resolve("Did Sarah verify the geocoder fix on staging?", vis, corpus, idf)["resolved"] == "Sarah Kim"
    assert people.resolve("What did Sarah Kim say?", vis, corpus, idf)["resolved"] == "Sarah Kim"
    assert people.resolve("What happened yesterday?", vis, corpus, idf) == {}


def test_anchor_relation_parsing():
    assert anchor.parse_relation("What happened right after the kickoff call?") == {"relation": "after", "anchor_query": "the kickoff call"}
    assert anchor.parse_relation("What is planned the day before the offsite?")["relation"] == "day_before"
    assert anchor.parse_relation("Which events fall on the day I travel to Austin?")["relation"] == "same_day"
    assert anchor.parse_relation("When is the budget review?") is None


def test_anchor_window_only_returns_visible_records_after_the_anchor(corpus):
    visible = {u.id for u in corpus.visible("2026-09-02T23:00:00-07:00")}
    got = anchor.window_candidates("after", ["SL-2"], visible, corpus)
    assert "SL-3" not in got or "SL-3" in visible          # SL-3 is Sep 2 10:00: visible, different day -> allowed
    assert "SL-4" not in got and "SL-5" not in got         # after as_of: never returned
    assert all(i in visible for i in got)


def test_chain_picks_up_a_correction_that_shares_few_words(corpus):
    visible = {u.id for u in corpus.visible("2026-09-04T00:00:00-07:00")}
    idf = {"launch": 4.0, "sep": 3.0}
    new, chain = chains.expand(["SL-2"], "When is the launch?", visible, corpus, idf)
    assert "SL-3" in new, "the later 'Correction: ... moved' message must be pulled in"
    assert chain == sorted(chain, key=lambda i: (corpus.get(i).time, i))   # chain is oldest-first


def test_ledger_links_delivery_and_respects_as_of(corpus):
    led = ledger.get_ledger(corpus)
    promise = [c for c in led if "pricing plan" in c.action.lower()]
    assert promise, "the 'I'll send the pricing plan by Friday' promise must be extracted"
    c = promise[0]
    before = c.view(datetime(2026, 9, 2, 12, tzinfo=timezone.utc))
    after = c.view(datetime(2026, 9, 9, 12, tzinfo=timezone.utc))
    assert before["status"] == "open" and "SL-4" not in before["source_ids"]
    assert after["status"] == "done" and "SL-4" in after["source_ids"]


def test_due_date_parsing_is_relative_to_message_time():
    at = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)   # a Wednesday
    assert ledger.parse_due("I'll have it by Friday", at) == "2026-09-11"
    assert ledger.parse_due("sending it tomorrow", at) == "2026-09-10"
    assert ledger.parse_due("no deadline here", at) is None


def test_hard_rule_v2_never_returns_invisible_ids_even_if_the_reranker_tries(corpus, corpus_dir, monkeypatch):
    from memory.retrieve import retrieve
    for flag in ("USE_LANES", "USE_CHAINS", "USE_ANCHOR", "USE_PEOPLE", "USE_LEDGER", "USE_RERANK", "USE_ANALYSIS"):
        monkeypatch.setattr(config, flag, True)
    monkeypatch.setattr(config, "USE_CROSS_ENCODER", False)
    monkeypatch.setattr(config, "USE_DENSE", False)
    evil = {"ranked": ["SL-5", "SL-4", "NOT-A-REAL-ID", "SL-1"], "reason": "x"}
    analysis = {"intent": "other", "wants_latest": True, "wants_history": False, "sub_queries": ["launch"], "entities": [],
                "dates": [], "ambiguous_person": None, "needs_followup": False, "answer_sketch": "", "relation": "", "anchor_query": ""}
    as_of = "2026-09-02T23:00:00-07:00"
    visible = {u.id for u in corpus.visible(as_of)}
    with patch("memory.query.chat_json", return_value=analysis), patch("memory.llm.chat_json", return_value=evil):
        import memory.retrieve as r
        r._resources.cache_clear()
        got = [u for u, _ in retrieve("When is the launch after the pricing proposal?", as_of, data_dir=str(corpus_dir), cache_dir=str(corpus_dir / ".c"))]
    assert got and set(got) <= visible
    assert "SL-4" not in got and "SL-5" not in got and "NOT-A-REAL-ID" not in got



def test_anchor_ignores_clauses_and_marks_gap_questions():
    assert anchor.parse_relation("What did she say after reviewing the contract?") is None            # gerund clause, not an event
    assert anchor.parse_relation("What did he say after they signed the form?") is None               # pronoun clause
    assert anchor.parse_relation("Notes on following up with the vendor") is None                      # "following up" is a verb phrase
    assert anchor.parse_relation("How many days after the kickoff call did the budget land?")["relation"] == "gap"


def test_context_neighbors_returns_adjacent_transcript_segments(tmp_path):
    import json as _json
    from memory.chains import context_neighbors
    d = tmp_path
    (d / "native/meetings").mkdir(parents=True)
    segs = [{"seg_id": f"MTG-X#{i:04d}", "start_s": i, "end_s": i + 1, "speaker_label": "S", "speaker_name": "Pat Doe",
             "speaker_confidence": 0.9, "channel": "room", "text": f"segment number {i} says something useful here"} for i in range(1, 8)]
    (d / "native/meetings/m.json").write_text(_json.dumps({"id": "MTG-X", "title": "t", "type": "in_person", "start": "2026-09-01T10:00:00-07:00",
                                                           "end": "2026-09-01T10:30:00-07:00", "segments": segs}))
    store = MemoryStore(str(d))
    visible = {u.id for u in store.units}
    got = context_neighbors(["MTG-X#0004"], visible, store, span=2)
    assert set(got) == {"MTG-X#0003", "MTG-X#0005", "MTG-X#0002", "MTG-X#0006"}
    assert context_neighbors(["MTG-X#0004"], {"MTG-X#0004", "MTG-X#0005"}, store, span=2) == ["MTG-X#0005"]   # invisible ones never appear
