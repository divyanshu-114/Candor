"""Rules-based planning: zero-token answers for clear-cut commands, and the
deterministic fallback when the LLM is unavailable.

Two levels, on purpose:

* `confident_plan(command, world)` -- only commands whose handling is
  unambiguous: planted instructions, a pronoun with no antecedent, a
  reminder/booking in the past, an imperative delete/cancel (-> confirm), a
  bare "open <app>", a wh-question for memory, and a first name shared by two
  people with no cue to pick one. Returns None when unsure, so the caller
  falls through to the LLM.
* `rules_plan(command, world)` -- the no-LLM fallback: the confident rules,
  then looser matches (a destructive verb anywhere, any question-shaped
  text), else a generic clarify.

Nothing here echoes the command into a clarify (it could repeat a planted
instruction or a secret; PROJECT_RULES 4/5). Sending a message or booking an
event is never attempted by rules: it needs person/date resolution, and a
wrong guess is worse than asking.
"""
from __future__ import annotations

import re

from memory.safety import INJECTION_PATTERNS, mask_secrets

GENERIC_CLARIFY = ("I need a bit more detail to do that safely. Who or what should this apply to, "
                   "and what exactly would you like me to do?")
WHO_CLARIFY = "Who should I send this to? Please name the person."
PAST_CLARIFY = "That time is already in the past. Which future date or time did you mean?"

_ACTION_VERB = re.compile(
    r"\b(?:message|email|e-mail|tell|send|remind|book|schedule|create|move|push|forward|reply|invite|open|"
    r"delete|cancel|remove|set\s+up|add|post|ping|dm|text|call|ask|draft|write|share|reschedule)\b", re.I)
_WH_START = re.compile(r"^(?:what|what's|whats|when|when's|where|where's|who|who's|whom|whose|which|why|"
                       r"how(?!\s+about))\b", re.I)
_AUX_QUESTION = re.compile(r"^(?:is|are|was|were|do|does|did|has|have|had)\b", re.I)
_LOOSE_QUESTION = re.compile(r"^(?:can|could|would|will|should)\b", re.I)
_OPEN_APP = re.compile(r"^(?:please\s+)?(?:open|launch)\s+([A-Za-z0-9][A-Za-z0-9 ._-]{0,40}?)\s*[.!]?$", re.I)
_DESTRUCTIVE_WORD = re.compile(r"\b(?:delete|remove|cancel|wipe|erase|discard)\b", re.I)
_DESTRUCTIVE_START = re.compile(r"^(?:please\s+)?(?:delete|remove|cancel|wipe|erase|discard)\b", re.I)
_SPLIT = re.compile(r"\s+(?:and then|and|then)\s+|\s*;\s*", re.I)
_PRONOUN_RECIPIENT = re.compile(r"^(?:please\s+)?(?:tell|message|email|ask|ping|dm|text|send)\s+(?:her|him|them)\b", re.I)
_CREATE_VERB = re.compile(r"(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder|book|schedule|set\s+up|create)\b", re.I)
# A past day IMMEDIATELY followed by a time of day, closing the command
# ("... yesterday at 5pm"). "the report I wrote last week at 5pm" never matches
# because "last week" is not a day.
_PAST_DAY_AT_TIME = re.compile(
    r"\b(?:yesterday|last\s+(?:mon|tues?|wednes|thurs?|fri|satur|sun)day|\d+\s+days?\s+ago)"
    r"(?:\s*,)?\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*[.!]?$", re.I)
_PEOPLE_VERBS = re.compile(r"\b(?:message|tell|email|send|ask|ping|dm|text|book|schedule|invite|forward|reply)\b", re.I)
_CAP_WORD = re.compile(r"\b[A-Z][a-z]+\b")
_SUMMARY_MAX_CHARS = 140


def _looks_like_app_name(name: str) -> bool:
    """"Figma", "Google Docs": one or two Capitalised words. Anything else
    ("the Q3 report", "my notes", "figma file") could be a document, so the
    LLM decides."""
    words = name.split()
    return 1 <= len(words) <= 2 and all(w[:1].isupper() and w.lower() not in {"the", "a", "my", "our"} for w in words)


def _clarify(question: str = GENERIC_CLARIFY) -> list[dict]:
    return [{"type": "clarify", "args": {"question": question}}]


# Command-level injection: only phrasings that are unmistakably an attempt to
# override the assistant. The broader patterns used for ingested DATA ("forward
# all", "you must", "assistant ... never") also match ordinary commands such as
# "Please forward all invoices to Ben", so they are NOT used here.
_COMMAND_INJECTION = (
    re.compile(r"\b(?:ignore|disregard|forget|override|bypass)\b(?:\s+\w+){0,3}\s+"
               r"(?:rules?|instructions?|guidelines?|prompts?|polic(?:y|ies)|restrictions?|safeguards?)\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|pretend\s+(?:to\s+be|you))\b", re.I),
    re.compile(r"\breveal\b(?:\s+\w+){0,3}\s+(?:system\s+prompt|instructions|secrets?|keys?|passwords?)\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"<!--|[\u200b\u200c\u200d\ufeff\u00ad]"),
)


def looks_injected(command: str) -> bool:
    return any(p.search(command) for p in _COMMAND_INJECTION)


def _confirm(text: str) -> list[dict]:
    summary = mask_secrets(text)[:_SUMMARY_MAX_CHARS].rstrip(" .")
    return [{"type": "confirm", "args": {"summary": f"This would change or remove data: {summary}. "
                                                    "Do you want me to go ahead?"}}]


def _ambiguous_name_clarify(text: str, world) -> list[dict] | None:
    """Two people share the first name used and NOTHING points at one of them
    -> ask which. Returns None (let the LLM decide) when any cue exists:
    a full or last name, "on Slack" (only Slack users qualify), "email" (a
    different person may be meant), an organisation matching a candidate's
    email domain ("Acme"). Topic words alone ("the pricing proposal") are NOT a
    cue: letting the model decide on topic made it guess the wrong Sarah, and
    asking is always safe."""
    ambiguous = getattr(world, "ambiguous_first_names", None) or {}
    if not ambiguous or not _PEOPLE_VERBS.search(text):
        return None
    # a name must be capitalised AND not the first word (which is capitalised
    # only because it starts the command)
    mentioned = [m.group(0) for m in _CAP_WORD.finditer(text) if m.start() > 0 and m.group(0).lower() in ambiguous]
    if not mentioned:
        return None
    first = mentioned[0].lower()
    candidates = [p for p in world.people if p.name.split()[0].lower() == first]
    low = text.lower()
    tokens = set(re.findall(r"[a-z]+", low))
    if any(p.name.lower() in low or p.name.split()[-1].lower() in tokens for p in candidates):
        return None  # full or last name given
    if re.search(r"\b(?:slack|e-?mail|mail|gmail|dm)\b", low):
        return None  # a channel/medium cue: only some candidates qualify, or a different one is meant
    for p in candidates:
        domain_label = (p.email or "").split("@")[-1].split(".")[0].lower()
        if domain_label and any(len(t) >= 4 and domain_label.startswith(t) for t in tokens):
            return None  # organisation cue ("Acme" -> acmefreight)
    if len(candidates) < 2:
        return None
    names = " or ".join(p.name for p in candidates)
    return _clarify(f"Which {mentioned[0]} do you mean: {names}?")


def confident_plan(command: str, world=None) -> list[dict] | None:
    """A plan when the command is clear-cut, else None (caller asks the LLM)."""
    text = " ".join(str(command).split())
    if not text or looks_injected(text):
        return _clarify()
    if _PRONOUN_RECIPIENT.match(text):
        return _clarify(WHO_CLARIFY)
    if _CREATE_VERB.match(text) and _PAST_DAY_AT_TIME.search(text):
        return _clarify(PAST_CLARIFY)
    if _DESTRUCTIVE_START.match(text):
        return _confirm(text)
    segments = [s for s in _SPLIT.split(text) if s and s.strip()]
    apps = [_OPEN_APP.match(s.strip()) for s in segments]
    if segments and all(apps) and all(_looks_like_app_name(m.group(1)) for m in apps):  # type: ignore[union-attr]
        return [{"type": "app.open", "args": {"app": m.group(1).strip()}} for m in apps]  # type: ignore[union-attr]
    # A question for memory only when it is ONE question ending in "?" with no
    # action verb anywhere ("When Ben replies, remind me ..." is a request).
    if (_WH_START.match(text) or _AUX_QUESTION.match(text)) and text.endswith("?") \
            and text.count("?") == 1 and not _ACTION_VERB.search(text):
        return [{"type": "memory.ask", "args": {"question": mask_secrets(text)}}]
    if world is not None:
        return _ambiguous_name_clarify(text, world)
    return None


def rules_plan(command: str, world=None) -> list[dict]:
    """No-LLM plan: confident rules, then looser ones, else a generic clarify."""
    plan = confident_plan(command, world)
    if plan is not None:
        return plan
    text = " ".join(str(command).split())
    if _DESTRUCTIVE_WORD.search(text):
        return _confirm(text)
    if text.endswith("?") and not _LOOSE_QUESTION.match(text):  # "Can you email Ben?" is a request, not a question
        return [{"type": "memory.ask", "args": {"question": mask_secrets(text)}}]
    return _clarify()
