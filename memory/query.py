"""Query analysis using LLM."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from memory import config
from memory import llm as llm_module
from memory.llm import chat_json

LOG = logging.getLogger(__name__)

# Kept short on purpose (step 2c): fewer few-shots = fewer prompt tokens on
# every single question, and this stage runs on every question regardless
# of outcome.
SYSTEM_PROMPT = """You are an IR engineer. Analyze a user's question and produce a JSON search plan.

Keys:
- "intent": one of [current_state, historical_state, who_said, commitment_status, ownership, arithmetic_or_dates, fact_lookup, preference, schedule, other]
- "wants_latest": true if the user wants the most recent status/update
- "wants_history": true if the user asks for a timeline or how things changed
- "sub_queries": 1-4 short keyword-style strings, each targeting a DIFFERENT piece of evidence (not natural-language questions)
- "entities": people/organizations/products mentioned
- "dates": ISO date strings (YYYY-MM-DD) explicit or implied (e.g. "tomorrow"), resolved relative to 'as_of'
- "ambiguous_person": null, or a first name that might refer to multiple people
- "needs_followup": true if the question needs a lookup first (e.g. "the day I fly", "the person who presented")
- "answer_sketch": two short invented sentences, in the style of a Slack message or meeting remark, that WOULD answer
  the question -- using concrete words a real record might use (invented names/numbers are fine here, this is only
  used to help a search engine match vocabulary, never shown to the user)

Output ONLY JSON, no conversational text. 'as_of' is "today" for the user.

EXAMPLES (made-up topics, not the real dataset; as_of = 2026-10-15):
User: "Who took over the widget inventory audit after the retreat?"
{"intent": "ownership", "wants_latest": true, "wants_history": true, "sub_queries": ["widget inventory audit owner", "retreat widget audit"], "entities": ["widget inventory audit"], "dates": [], "ambiguous_person": null, "needs_followup": true, "answer_sketch": "Priya is taking over the widget inventory audit starting Monday. She'll have the first count done by Friday."}

User: "How many times did we change the pricing for the starter tier?"
{"intent": "arithmetic_or_dates", "wants_latest": false, "wants_history": true, "sub_queries": ["starter tier pricing", "pricing change starter"], "entities": ["starter tier"], "dates": [], "ambiguous_person": null, "needs_followup": false, "answer_sketch": "We've changed the starter tier price twice: once in March to $12/mo, then again in July to $15/mo."}
"""

def fallback_analyze(question: str, as_of: str, degraded: bool = False) -> dict:
    """Fallback when LLM is unavailable or disabled by MODE.

    `degraded` distinguishes "MODE=baseline chose this on purpose" (False)
    from "the LLM call failed and we're falling back" (True), so callers can
    log/measure real degradation without false positives on a legitimate
    heuristic-only run.
    """
    # Basic date regex extractor (very naive)
    dates = []
    # If there's an ISO date in the question
    for m in re.finditer(r"\d{4}-\d{2}-\d{2}", question):
        dates.append(m.group(0))

    return {
        "intent": "other",
        "wants_latest": True,
        "wants_history": False,
        "sub_queries": [question],
        "entities": [],
        "dates": dates,
        "ambiguous_person": None,
        "needs_followup": False,
        "answer_sketch": "",
        "_degraded": degraded,
    }

def analyze_question(question: str, as_of: str) -> dict:
    if not config.USE_ANALYSIS:
        return fallback_analyze(question, as_of, degraded=False)

    user_prompt = f"as_of: {as_of}\nUser: {question}"
    try:
        res = chat_json(SYSTEM_PROMPT, user_prompt, llm_module.get_model_fast(),
                         max_tokens=config.MAX_TOKENS_ANALYSIS, reasoning_effort=config.REASONING_EFFORT_LOW,
                         stage="analysis")
        if res:
            # Ensure all keys exist
            return {
                "intent": res.get("intent", "other"),
                "wants_latest": bool(res.get("wants_latest", True)),
                "wants_history": bool(res.get("wants_history", False)),
                "sub_queries": res.get("sub_queries", [question]) or [question],
                "entities": res.get("entities", []),
                "dates": res.get("dates", []),
                "ambiguous_person": res.get("ambiguous_person", None),
                "needs_followup": bool(res.get("needs_followup", False)),
                "answer_sketch": str(res.get("answer_sketch", "") or ""),
                "_degraded": False,
            }
    except Exception as e:
        LOG.warning("LLM analysis failed, using fallback. Error: %s", e)

    return fallback_analyze(question, as_of, degraded=True)

def get_date_expansions(iso_date: str) -> str:
    """Expand 'YYYY-MM-DD' into forms like '2026-09-10 Sep 10 September 10 Thursday'."""
    try:
        dt = datetime.strptime(iso_date[:10], "%Y-%m-%d")
        return f"{dt:%Y-%m-%d} {dt:%b} {dt.day} {dt:%B} {dt.day} {dt:%A}"
    except ValueError:
        return iso_date
