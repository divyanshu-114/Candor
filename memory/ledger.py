"""Commitments ledger: who promised what to whom, by when, and what became of it.

WHY: broad questions ("what do I still owe?", "what did I promise Dave?", "what's still open?") have no single
answer record. The answer is scattered: the promise in one message, a deadline extension in another, the
delivery in a third. A ledger built at ingest links those records so one lookup returns the promise, any
extension and the fulfilment -- each as a REAL record id (never a derived one) -- filtered by `as_of` so a
later fulfilment cannot leak into an earlier moment.

Extraction is rule-based (works with no key): first-person promise patterns, requests ("can you ..."), and
deadline phrases, resolved to dates relative to the message time. Later records are linked to a commitment
when they share its distinctive words in the same thread / conversation partner and carry a fulfilment,
extension or cancellation cue. An optional LLM pass over only the flagged units can refine owner/due/status
(see scripts/ledger_llm.py); the default path costs nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache

from memory.index import tokenize
from memory.queries import QUESTION_WORDS
from memory.store import MemoryStore

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
PROMISE = re.compile(
    r"\b(?:i'?ll|i will|i'?m going to|i am going to|i can have|we'?ll|we will|let me|i promise|i promised|i'?d like to|"
    r"i'?ll have|i can get|i'?ll get|i'?ll need to|will send|will do|i should)\b", re.I)
REQUEST = re.compile(r"\b(?:can you|could you|would you|please|need you to|can we|do you have time to|remind me)\b", re.I)
OWE = re.compile(r"\b(?:action item|todo|to-do|follow[- ]?up|promised|committed|owe|deadline|due)\b", re.I)
# A promise needs a concrete task verb; conversational "I'll stop sharing" / "let me recap" / "can you hear me" are not commitments.
ACTION = re.compile(
    r"\b(?:send|write|put|set up|setup|create|update|fix|ping|ask|find|get|let \w+ know|schedule|review|share|post|bring|draft|"
    r"check|follow up|confirm|prepare|book|call|email|loop|pull|run|talk to|reach out|spin|line up|add|submit|test|rerun|"
    r"look at|go through|send over|make sure|move|build|ship|publish|clone|backfill|rotate|tell|introduce|make|swap|darken|mock|"
    r"do a|handle|take|own|keep|bring|work on|reach|sync|investigate|document|record|track|owe)\b", re.I)
NOT_A_TASK = re.compile(
    r"\b(?:hear|see (?:my|the|that)|stop (?:sharing|the recording)|recap|take that back|find it|be honest|i promise\b|i think|i guess|"
    r"let me (?:just )?(?:be|say|think|stop|take|see|recap|find it)|by popular demand|didn'?t|haven'?t)\b", re.I)
ACCEPT = re.compile(r"^(?:yes|yep|yeah|sure|ok|okay|on it|will do|got it|done|agreed|works|sounds good)\b", re.I)
DEADLINE = re.compile(
    r"\bby (?:eod|end of (?:the )?(?:day|week)|(?:next )?(?:mon|tues?|wednes|thurs?|fri|satur|sun)day|tomorrow|tonight|"
    r"the \d{1,2}(?:st|nd|rd|th)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2})\b|"
    r"\buntil (?:(?:next )?(?:mon|tues?|wednes|thurs?|fri|satur|sun)day|the \d{1,2}(?:st|nd|rd|th))\b|"
    r"\b(?:next|this) (?:week|(?:mon|tues?|wednes|thurs?|fri|satur|sun)day)\b|\btomorrow\b|\btonight\b|"
    r"\bbefore (?:the )?(?:launch|release|deadline|meeting|call|event)\b", re.I)
DONE = re.compile(r"\b(?:done|sent|attached|here'?s|went out|just sent|posted|merged|shipped|ready|finished|completed|landed|"
                  r"is up|are up|is in|are in|submitted|created|went through|as promised|is live|all set)\b", re.I)
EXTEND = re.compile(r"\b(?:push|pushed|extend|extension|until|more time|can i have|slip|later|moved? (?:it|to)|delay\w*)\b", re.I)
CANCEL = re.compile(r"\b(?:scratch|cancel\w*|no need|never ?mind|not needed|don'?t need|won'?t need|pushed (?:it|the \w+|them) to|dead|off the table)\b", re.I)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
GENERIC = QUESTION_WORDS | set(_WEEKDAYS) | {"done", "today", "tomorrow", "week", "next", "morning", "tonight", "monday", "meeting",
                                              "call", "thing", "stuff", "time", "day", "thanks", "ready", "good", "great", "nice", "okay"} | {"will", "ll", "let", "can", "could", "would", "please", "send", "get", "have", "need", "just", "now",
                            "going", "also", "make", "sure", "thanks", "thank", "yes", "yep", "okay"}


@dataclass
class Commitment:
    id: str
    owner: str
    action: str
    to_whom: str
    due: str | None
    t0: datetime
    events: list[tuple[str, str, datetime]] = field(default_factory=list)   # (kind, unit id, time), time-ordered; first is the promise
    tokens: frozenset[str] = frozenset()

    def view(self, as_of: datetime) -> dict:
        """Status as of `as_of` using only events that already existed then."""
        ev = [(k, i, t) for k, i, t in self.events if t <= as_of]
        status = "open"
        due = self.due
        for kind, _i, _t in ev[1:]:
            if kind == "done":
                status = "done"
            elif kind == "cancel" and status != "done":
                status = "cancelled"
            elif kind == "extend" and status not in ("done", "cancelled"):
                status = "extended"
        return {"id": self.id, "owner": self.owner, "to_whom": self.to_whom, "action": self.action, "due": due,
                "status": status, "source_ids": [i for _k, i, _t in ev], "events": [(k, i) for k, i, _t in ev]}


def _next_weekday(base: date, name: str, strictly_after: bool = True) -> date:
    target = _WEEKDAYS.index(name)
    delta = (target - base.weekday()) % 7
    if delta == 0 and strictly_after:
        delta = 7
    return base + timedelta(days=delta)


def parse_due(text: str, at: datetime) -> str | None:
    """Best-effort ISO date for a deadline phrase in `text`, relative to the message time `at`; None if there is none."""
    m = DEADLINE.search(text)
    if not m:
        return None
    phrase = m.group(0).lower()
    base = at.date()
    if "tomorrow" in phrase:
        return (base + timedelta(days=1)).isoformat()
    if "tonight" in phrase or "eod" in phrase or "end of day" in phrase or "end of the day" in phrase:
        return base.isoformat()
    if "end of week" in phrase or "end of the week" in phrase:
        return _next_weekday(base, "friday", strictly_after=False).isoformat()
    if "next week" in phrase:
        return (base + timedelta(days=7)).isoformat()
    for day in _WEEKDAYS:
        if day[:3] in phrase and (day in phrase or phrase.endswith(day[:3])):
            return _next_weekday(base, day).isoformat()
    mm = re.search(r"the (\d{1,2})(?:st|nd|rd|th)", phrase)
    if mm:
        d = int(mm.group(1))
        try:
            cand = base.replace(day=d)
            return (cand if cand >= base else (cand.replace(month=cand.month % 12 + 1))).isoformat()
        except ValueError:
            return None
    mm = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{1,2})", phrase)
    if mm:
        try:
            return date(base.year, _MONTHS[mm.group(1)], int(mm.group(2))).isoformat()
        except ValueError:
            return None
    return None


_CLAUSE_SPLIT = re.compile(r"[;,—–]| but | and then |\bso\b")


def _promise_clause(sent: str) -> str:
    """The clause that carries the promise ("Launch is Sep 30, I'll send the pricing plan by Friday" -> the second half), so the
    promise is matched on ITS words, not on the first clause's."""
    m = PROMISE.search(sent) or REQUEST.search(sent)
    if not m:
        return sent
    start = 0
    for cut in _CLAUSE_SPLIT.finditer(sent):
        if cut.end() <= m.start():
            start = cut.end()
    end = len(sent)
    for cut in _CLAUSE_SPLIT.finditer(sent):
        if cut.start() >= m.end():
            end = cut.start()
            break
    return sent[start:end].strip() or sent


def _content(text: str) -> frozenset[str]:
    return frozenset(t for t in tokenize(text) if t not in GENERIC and len(t) > 2)


def _counterpart(u, store: MemoryStore) -> str:
    """Who the speaker is talking to: the other member of a DM, the email recipient, or the channel."""
    if u.source == "email":
        to = u.meta.get("to") or []
        return ", ".join(re.sub(r"\s*<.*?>", "", x).strip() for x in to[:2])
    if u.source == "dictation":
        ctx = str(u.meta.get("target_context") or "")
        return ctx.replace("Compose – to ", "").replace("Reply – Re: ", "")[:60]
    return u.title or ""


@lru_cache(maxsize=4)
def self_name(store_id: int, n: int, store: MemoryStore) -> str:
    """Whose memory this is: the most frequent recipient of the mailbox (derived from the data, never hard-coded)."""
    counts: dict[str, int] = {}
    names = {}
    for u in store.units:
        if u.source != "email":
            continue
        names = u.meta.get("email_names") or names
        for addr in u.meta.get("to") or []:
            m = re.search(r"<([^>]+)>", addr)
            a = (m.group(1) if m else addr).strip().lower()
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return ""
    return names.get(max(counts, key=lambda a: (counts[a], a)), "")


def _owner(u, me: str) -> str:
    if u.source in ("email",):
        return re.sub(r"\s*<.*?>", "", str(u.meta.get("from") or "")).strip()
    if u.source in ("dictation", "chatgpt", "codex"):
        return me
    return u.speaker or ""


@lru_cache(maxsize=4)
def _build(store_id: int, n: int, store: MemoryStore) -> tuple[Commitment, ...]:
    commitments: list[Commitment] = []
    pending_request: dict[str, tuple[str, str, datetime, str]] = {}   # thread -> (requester, text, time, id)
    me = self_name(id(store), len(store.units), store)
    for u in store.units:                                              # time order
        if u.source not in ("slack", "email", "dictation", "meeting") or "deleted_target" in u.meta or "edit_target" in u.meta:
            continue
        if u.source == "slack" and (u.meta.get("subtype") == "bot_message" or (u.speaker or "") in ("Linear", "GitHub")):
            continue
        if u.source == "email" and any(x in str(u.meta.get("from", "")).lower() for x in ("newsletter", "digest", "no-reply", "notifications@", "news@")):
            continue
        text = u.text.split("\n\n", 1)[-1] if u.source == "email" else u.text
        text = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(">"))
        for sent in (s.strip() for s in _SENT.split(text)):
            if len(sent.split()) < 4 or len(sent.split()) > 60:
                continue
            owner = _owner(u, me)
            key = u.thread_key or u.id
            # a reply that accepts the previous request is the addressee's promise
            if ACCEPT.match(sent) and key in pending_request and (u.time - pending_request[key][2]) < timedelta(hours=6) \
                    and pending_request[key][0] != owner:
                req_owner, req_text, req_t, req_id = pending_request.pop(key)
                c = Commitment(f"C-{len(commitments) + 1}", owner, req_text, req_owner, parse_due(req_text + " " + sent, req_t), req_t,
                               [("promise", req_id, req_t), ("promise", u.id, u.time)], _content(req_text))
                commitments.append(c)
                continue
            task = bool(ACTION.search(sent)) and not NOT_A_TASK.search(sent)
            is_promise = bool(PROMISE.search(sent)) and not sent.rstrip().endswith("?") and task
            is_request = bool(REQUEST.search(sent)) and task
            if is_request:
                pending_request[key] = (owner, sent, u.time, u.id)
            if is_promise or (OWE.search(sent) and DEADLINE.search(sent) and task):
                toks = _content(_promise_clause(sent))
                if len(toks) < 2:
                    continue
                commitments.append(Commitment(f"C-{len(commitments) + 1}", owner, sent, _counterpart(u, store),
                                              parse_due(sent, u.time), u.time, [("promise", u.id, u.time)], toks))
    _link(commitments, store)
    return tuple(commitments)


def _link(commitments: list[Commitment], store: MemoryStore) -> None:
    """Attach later extension / fulfilment / cancellation records to the commitments they refer to.

    A record links only if it shares DISTINCTIVE words with the promise: rare words (idf-weighted) carrying at
    least 45% of the promise's weight and at least two of them -- otherwise a generic "done" would close
    every open item."""
    units = [u for u in store.units if u.source in ("slack", "email", "dictation", "meeting") and "deleted_target" not in u.meta
             and "edit_target" not in u.meta]
    df: dict[str, int] = {}
    toks_of: dict[str, frozenset[str]] = {}
    for u in units:
        t = _content(u.text.split("\n\n", 1)[-1] if u.source == "email" else u.text)
        toks_of[u.id] = t
        for w in t:
            df[w] = df.get(w, 0) + 1
    n = len(units)
    weight = {w: __import__("math").log(1 + n / c) for w, c in df.items()}
    for c in commitments:
        own = {i for _k, i, _t in c.events}
        total = sum(weight.get(w, 1.0) for w in c.tokens) or 1.0
        for u in units:
            if u.time <= c.t0 or u.id in own:
                continue
            overlap = [w for w in c.tokens & toks_of[u.id] if df.get(w, 0) <= max(3, 0.15 * n)]   # rare words only
            if len(overlap) < 2 or sum(weight[w] for w in overlap) < 0.45 * total:
                continue
            text = u.text.split("\n\n", 1)[-1] if u.source == "email" else u.text
            kind = "cancel" if CANCEL.search(text) else "done" if DONE.search(text) else "extend" if EXTEND.search(text) else None
            if kind:
                c.events.append((kind, u.id, u.time))
                own.add(u.id)
        c.events.sort(key=lambda e: (e[2], e[1]))


def get_ledger(store: MemoryStore) -> tuple[Commitment, ...]:
    return _build(id(store), len(store.units), store)


LEDGER_INTENT = re.compile(
    r"\b(?:still (?:open|pending|outstanding|owe)|owe|promis\w*|commit\w*|follow[- ]?ups?|to-?dos?|action items?|pending|outstanding|"
    r"agreed to|supposed to|have i (?:done|sent|set up|finished)|did i (?:send|set up|finish|submit|schedule|reply|follow)|is it done|done yet)\b", re.I)


def wants_ledger(question: str) -> bool:
    return bool(LEDGER_INTENT.search(question))


_FIRST_PERSON = re.compile(r"\b(?:i|my|me|mine|i'?ve|i'?d)\b", re.I)


def lookup(question: str, as_of: datetime, store: MemoryStore, idf: dict[str, float], limit: int = 6) -> list[dict]:
    """Commitments (as of `as_of`) whose words overlap the question, best first. A first-person question
    ("what do I owe", "my promises") favours the memory owner's own commitments."""
    q = [t for t in dict.fromkeys(tokenize(question)) if t not in GENERIC]
    me = self_name(id(store), len(store.units), store) if _FIRST_PERSON.search(question) else ""
    scored = []
    for c in get_ledger(store):
        v = c.view(as_of)
        if not v["source_ids"] or c.t0 > as_of:
            continue
        pool = set(c.tokens) | set(tokenize(c.owner + " " + c.to_whom))
        score = sum(idf.get(t, 1.0) for t in q if t in pool)
        if score > 0 and me and c.owner == me:
            score *= 1.25
        if score > 0:
            scored.append((-score, c.t0, v))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [v for _s, _t, v in scored[:limit]]
