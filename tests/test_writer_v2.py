"""Writer v2 (config.WRITER_V2): soft quote matching, partial answers, version-chain hint, arithmetic in code.
End to end through answer_question with a fake model; retrieval forced to the plain baseline path."""
import json

import pytest

from memory import answer as ans
from memory import config


@pytest.fixture()
def run(corpus_dir, monkeypatch):
    monkeypatch.setattr("memory.llm.GROQ_API_KEY", "fake")
    for flag in ("USE_LANES", "USE_ANALYSIS", "USE_MULTIQUERY", "USE_NEIGHBORS", "USE_HOP2", "USE_RERANK", "USE_DENSE"):
        monkeypatch.setattr(config, flag, False)
    monkeypatch.setattr(config, "WRITER_V2", True)
    seen = {}

    def go(reply: dict, question="When is the launch?", as_of="2026-10-30T00:00:00-07:00"):
        def fake_chat(system, user, model, **kw):
            seen["system"], seen["user"] = system, user
            return reply
        monkeypatch.setattr(ans, "chat_json", fake_chat)
        import memory.retrieve as r
        r._resources.cache_clear()
        return ans.answer_question("Q", question, as_of, data_dir=str(corpus_dir), cache_dir=str(corpus_dir / ".c")), seen
    return go


def test_dropped_filler_word_no_longer_turns_a_correct_answer_into_an_abstention(run):
    # SL-3 says: "Correction: the launch moved to Oct 14, sorry."  The quote drops "the" and "sorry" (harmless drift).
    row, _ = run({"answerable": True, "answer": "The launch moved to Oct 14.", "used_ids": ["SL-3"],
                  "support": [{"id": "SL-3", "quote": "Correction: launch moved to Oct 14"}]})
    assert row["abstained"] is False and row["sources"] == ["SL-3"]


def test_a_changed_figure_in_the_quote_still_abstains(run):
    row, _ = run({"answerable": True, "answer": "Oct 21.", "used_ids": ["SL-3"],
                  "support": [{"id": "SL-3", "quote": "Correction: the launch moved to Oct 21"}]})
    assert row["abstained"] is True and row["answer"].startswith("I don't know")


def test_partial_answer_is_returned_not_abstained_and_names_what_is_missing(run):
    row, _ = run({"answerable": "partial", "answer": "The launch moved to Oct 14. I couldn't find who approved it.", "missing": "approver",
                  "used_ids": ["SL-3"], "support": [{"id": "SL-3", "quote": "the launch moved to Oct 14"}]})
    assert row["abstained"] is False and "couldn't find" in row["answer"] and row["sources"] == ["SL-3"]


def test_related_records_do_not_justify_an_answer_when_the_model_says_no(run):
    row, _ = run({"answerable": False, "answer": "The records mention the launch but not the budget.", "used_ids": [], "support": []},
                 question="What is the launch budget?")
    assert row["abstained"] is True and row["sources"] == []


def test_days_are_counted_by_code_not_by_the_model(run):
    row, _ = run({"answerable": True, "answer": "That was {{result}} days later.", "compute": {"op": "days_between", "a": "2026-09-01", "b": "2026-09-02"},
                  "used_ids": ["SL-3"], "support": [{"id": "SL-3", "quote": "the launch moved to Oct 14"}]})
    assert "1 days later" in row["answer"] and "{{" not in row["answer"]


def test_sources_are_only_the_records_whose_quotes_verified(run):
    row, _ = run({"answerable": True, "answer": "Moved to Oct 14 and the plan was sent.", "used_ids": ["SL-3", "SL-4"],
                  "support": [{"id": "SL-3", "quote": "launch moved to Oct 14"}, {"id": "SL-4", "quote": "completely invented words about tacos"}]},
                 as_of="2026-10-30T00:00:00-07:00")
    assert row["sources"] == ["SL-3"]


def test_version_chain_hint_reaches_the_prompt_in_time_order(run, monkeypatch):
    import memory.retrieve as r
    real = r.retrieve
    def with_chain(*a, **k):
        ranked, meta = real(*a, **k)
        meta["chain"] = ["SL-2", "SL-3"]
        return ranked, meta
    monkeypatch.setattr(ans, "retrieve", with_chain)
    _, seen = run({"answerable": True, "answer": "Oct 14.", "used_ids": ["SL-3"], "support": [{"id": "SL-3", "quote": "launch moved to Oct 14"}]})
    assert "<chain>SL-2 -> SL-3</chain>" in seen["user"] and "VERSIONS" in seen["system"] and "PARTIAL" in seen["system"]
