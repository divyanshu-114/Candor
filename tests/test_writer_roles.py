"""Writer role split, bigger writer budget, 400-misattribution fix, cleaner
email extractive fallback. Fake providers / synthetic text only."""
from datetime import datetime
from unittest.mock import MagicMock

import memory.llm as llm_mod
from memory import answer as ans
from memory import config
from memory.llm import Provider, chat_json
from memory.models import Unit
from tests.test_llm_providers import _fake_openai_response, _raw_response, _status_error


def test_writer_budget_and_default_role():
    assert config.MAX_TOKENS_WRITER >= 800
    assert config.WRITER_ROLE == "fast"


def test_writer_uses_fast_model_by_default_and_strong_when_asked(monkeypatch):
    monkeypatch.setattr(ans.llm_module, "get_model_strong", lambda: llm_mod.RoleModel("S", "strong"))
    monkeypatch.setattr(ans.llm_module, "get_model_fast", lambda: llm_mod.RoleModel("F", "fast"))
    monkeypatch.setattr(config, "WRITER_ROLE", "fast")
    assert ans._writer_model() == "F" and ans._writer_model().role == "fast"
    monkeypatch.setattr(config, "WRITER_ROLE", "strong")
    assert ans._writer_model() == "S"


def test_run_writer_calls_the_writer_role_with_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "WRITER_ROLE", "fast")
    monkeypatch.setattr(ans.llm_module, "get_model_fast", lambda: llm_mod.RoleModel("F", "fast"))
    monkeypatch.setattr(ans.llm_module, "get_model_strong", lambda: llm_mod.RoleModel("S", "strong"))
    monkeypatch.setattr(ans, "chat_json", lambda s, u, m, **kw: seen.update(model=str(m), **kw) or {"answerable": False})
    ans._run_writer("<evidence></evidence>", "q", "2026-09-18T00:00:00-07:00", "other")
    assert seen["model"] == "F" and seen["max_tokens"] >= 800 and seen["stage"] == "writer"


def test_writer_schema_has_no_reasoning_field_and_answer_precedes_ids():
    text = ans.WRITER_SYSTEM
    assert '"reasoning"' not in text
    assert text.index('"answer"') < text.index('"used_ids"') < text.index('"support"')


def test_failed_generation_400_is_not_blamed_on_reasoning_effort(monkeypatch, tmp_path):
    p = Provider(name="alpha", api_key="k", base_url="https://alpha.invalid", model_strong="m", model_fast="m")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    monkeypatch.setattr(llm_mod, "_TOKEN_BUDGETS", {})
    c = MagicMock()
    c.chat.completions.with_raw_response.create.side_effect = _status_error(
        "Failed to generate JSON. code: json_validate_failed, failed_generation: {...", status=400)
    monkeypatch.setattr(p, "_client", c)
    assert chat_json("s", "u", "m", cache_dir=str(tmp_path), reasoning_effort="low") is None
    assert not any(k.startswith("reasoning_effort") for k in p.unsupported_params)
    assert "response_format" not in p.unsupported_params
    assert c.chat.completions.with_raw_response.create.call_count == 1  # no pointless param-drop retries


def _email(text):
    return Unit(id="EM-X", record_id="EM-X", source="email", time=datetime(2026, 9, 1), text=text,
                title="Re: Subject line words", speaker="Pat")


EMAIL = ("From: Pat <pat@x.example>\nTo: Lee <lee@x.example>\nSubject: Re: Subject line words\n\n"
         "Hi Lee,\n\nThe vendor contract renews on March 3 after legal signs off. Everything else is unchanged.\n\n"
         "Thanks,\nPat\n\nOn Mon, Sep 1, 2026 at 9:00 AM Lee wrote:\n> Earlier quoted question about pricing\n")


def test_email_extractive_is_one_body_sentence_without_headers_chain_or_signature():
    out = ans._email_body_sentence(EMAIL, "When does the vendor contract renew?", 35)
    assert out.startswith("The vendor contract renews on March 3")
    for bad in ("From:", "Subject", "Thanks", "Earlier quoted", "wrote:", "pat@x.example"):
        assert bad not in out
    assert len(out.split()) <= 35


def test_email_extractive_caps_words():
    long_body = "From: a\nTo: b\nSubject: s\n\n" + " ".join(["word"] * 80) + "."
    assert len(ans._email_body_sentence(long_body, "word", 35).split()) == 35


def test_extractive_answer_uses_email_cleaner(monkeypatch):
    store = MagicMock()
    store.visible.return_value = [_email(EMAIL)]
    monkeypatch.setattr(ans, "_coverage_says_abstain", lambda *a: False)
    out = ans._extractive_answer(["EM-X"], "When does the vendor contract renew?", "2026-09-18T00:00:00-07:00", store)
    assert out["sources"] == ["EM-X"] and out["answer"].startswith("The vendor contract renews")


def test_frozen_role_serves_cache_hits_but_never_calls_on_a_miss(monkeypatch, tmp_path):
    p = Provider(name="alpha", api_key="k", base_url="https://alpha.invalid", model_strong="big", model_fast="small")
    monkeypatch.setattr(llm_mod, "_PROVIDERS", [p])
    monkeypatch.setattr(llm_mod, "_ROLE_ORDER", {"strong": None, "fast": None})
    monkeypatch.setattr(llm_mod, "_TOKEN_BUDGETS", {})
    c = MagicMock()
    c.chat.completions.with_raw_response.create.return_value = _raw_response(_fake_openai_response())
    monkeypatch.setattr(p, "_client", c)
    # warm the cache while unfrozen
    assert chat_json("s", "u-cached", llm_mod.get_model_strong(), cache_dir=str(tmp_path)) == {"ok": True}
    assert c.chat.completions.with_raw_response.create.call_count == 1
    monkeypatch.setenv("LLM_FROZEN_ROLES", "strong")
    assert chat_json("s", "u-cached", llm_mod.get_model_strong(), cache_dir=str(tmp_path)) == {"ok": True}  # hit
    assert chat_json("s", "u-new", llm_mod.get_model_strong(), cache_dir=str(tmp_path)) is None           # miss
    assert chat_json("s", "u-new", llm_mod.get_model_fast(), cache_dir=str(tmp_path)) == {"ok": True}      # fast unaffected
    assert c.chat.completions.with_raw_response.create.call_count == 2


# --- writer rules (prompt + code check) ------------------------------------

def test_writer_prompt_contains_the_three_rules():
    t = ans.WRITER_SYSTEM
    assert "NEVER ASSERT ABSENCE" in t and "answerable=false" in t and "may be incomplete" in t
    assert "REPORTED SPEECH" in t and "second-hand" in t and "first-hand" in t and "final decision" in t
    flat = " ".join(t.split())
    assert "COMPLETENESS" in flat and "61 of 64" in flat and "who agreed to a change" in flat


def test_absence_detector():
    for yes in ("You have no calendar events scheduled for that day.", "Nothing was scheduled.",
                "There were no records of that.", "The meeting did not happen.", "It was not scheduled."):
        assert ans._asserts_absence(yes), yes
    for no in ("No, John did not agree to cut dark mode.", "The board meeting is at 9am.", "61 of 64 cases passed."):
        assert not ans._asserts_absence(no), no


def test_absence_needs_a_supporting_quote():
    sup = [{"id": "A", "quote": "board run-through at HQ"}]
    assert ans._absence_unsupported("You have no events that day.", sup, ["A"])
    sup2 = [{"id": "A", "quote": "no events are scheduled for Sept 23"}]
    assert not ans._absence_unsupported("You have no events that day.", sup2, ["A"])
    assert not ans._absence_unsupported("The board meeting is at 9am.", sup, ["A"])
    # a quote that was NOT verified cannot support the claim
    assert ans._absence_unsupported("You have no events that day.", sup2, [])


def _run_answer(monkeypatch, written):
    u = Unit(id="CAL-X", record_id="CAL-X", source="calendar", time=datetime(2026, 9, 1),
             text="Offsite planning at the lodge from 9am", title="Offsite", speaker="Pat", speaker_known=True)
    store = MagicMock()
    store.visible.return_value = [u]
    monkeypatch.setattr(ans, "retrieve", lambda *a, **k: ([("CAL-X", 1.0)], {"analysis": {"intent": "schedule"}}))
    monkeypatch.setattr(ans, "_resources", lambda *a, **k: (store, MagicMock()))
    monkeypatch.setattr(ans.llm_module, "GROQ_API_KEY", "fake")
    monkeypatch.setattr(ans, "_run_writer", lambda *a, **k: written)
    return ans.answer_question("Q", "What is on my calendar on Sep 23?", "2026-09-18T09:00:00-07:00", data_dir="x")


def test_bare_no_events_claim_with_unrelated_support_is_rejected_into_abstain(monkeypatch):
    row = _run_answer(monkeypatch, {"answerable": True, "answer": "You have no events scheduled that day.",
                                    "used_ids": ["CAL-X"], "support": [{"id": "CAL-X", "quote": "Offsite planning at the lodge"}]})
    assert row["abstained"] is True and row["sources"] == []


def test_normal_answer_with_verified_support_still_passes(monkeypatch):
    row = _run_answer(monkeypatch, {"answerable": True, "answer": "Offsite planning at the lodge starts at 9am.",
                                    "used_ids": ["CAL-X"], "support": [{"id": "CAL-X", "quote": "Offsite planning at the lodge"}]})
    assert row["abstained"] is False and row["sources"] == ["CAL-X"]


# --- verification text must equal what the writer was shown ------------------

def test_display_text_uses_retrieval_rank_budgets_not_chronological_position():
    """A long record that is retrieval rank 0 but chronologically LAST must be
    verified against the same 200-word trim the writer saw (not a 70-word one)."""
    words = [f"w{i}" for i in range(190)] + ["decisive", "phrase", "here"]
    old = Unit(id="A-OLD", record_id="A", source="slack", time=datetime(2026, 9, 1), text="short note", speaker="x", speaker_known=True)
    new = Unit(id="B-NEW", record_id="B", source="slack", time=datetime(2026, 9, 5), text=" ".join(words), speaker="x", speaker_known=True)
    store = MagicMock()
    store.visible.return_value = [old, new]
    # rank order: B-NEW first (best), then A-OLD; chronological order: A-OLD first
    pkg, chosen, display = ans.build_evidence_package(["B-NEW", "A-OLD"], "decisive phrase", "2026-09-18T00:00:00-07:00",
                                                       store, return_display=True)
    assert chosen == ["A-OLD", "B-NEW"]                        # chronological in the prompt
    assert "decisive phrase here" in display["B-NEW"]         # kept at the rank-0 budget (>= 190 words)
    assert ans._verify_support([{"id": "B-NEW", "quote": "decisive phrase here"}], display) == ["B-NEW"]
    assert display["B-NEW"] in pkg.replace("&quot;", '"')


def test_two_value_call_still_returns_pair():
    store = MagicMock()
    store.visible.return_value = []
    assert len(ans.build_evidence_package([], "q", "2026-09-18T00:00:00-07:00", store)) == 2
