"""Anchor-then-window retrieval for relative-time questions.

WHY: "what did I do right after the planning meeting?", "what's on my calendar the day before the board
meeting?", "the day I fly" cannot be answered by similarity alone -- the question words describe the
ANSWER, while the time comes from a different record (the anchor). We (1) parse the relation and the anchor
phrase (regex here; the analysis model supplies the same two fields when a key exists), (2) find the anchor
record, (3) turn it into a time or a calendar date, (4) fetch the visible records in the window after /
before it (same day first, then neighbours in the same thread) or on the target date, and (5) rank those
against the rest of the question. Only visible records are ever considered.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from memory.corpus import get_info
from memory.store import MemoryStore

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("day_before", re.compile(r"\bthe day before (?P<a>[^?.,;]+)", re.I)),
    ("day_after", re.compile(r"\bthe day after (?P<a>[^?.,;]+)", re.I)),
    ("same_day", re.compile(r"\b(?:the day of|the same day (?:as|that)|on the day (?:of|that|when)|the day (?=I |we |my ))(?P<a>[^?.,;]+)", re.I)),
    ("after", re.compile(r"\b(?:right after|just after|shortly after|immediately after|straight after|after|following(?! up\b)|since|once)\s+(?P<a>[^?.,;]+)", re.I)),
    ("before", re.compile(r"\b(?:right before|just before|shortly before|before|prior to|ahead of|until)\s+(?P<a>[^?.,;]+)", re.I)),
]
_ANCHOR_STOP = re.compile(r"\b(?:about|did|do|does|was|were|is|are|have|has|had|which|that|what|who|how|when|where|and then|,)\b.*$", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", re.I)


_GAP_QUESTION = re.compile(r"^\s*how (?:many|long|much|far|soon)\b", re.I)
_NOT_AN_EVENT = re.compile(r"^(?:(?:i|we|you|he|she|they|it|there|this|that)\b|\w+ing\b)", re.I)   # clauses, not event records


def parse_relation(question: str) -> dict | None:
    """{"relation", "anchor_query"} from wording alone, or None. The first matching pattern wins (most specific first).

    "after/before" only counts when the phrase after it names an EVENT ("the planning meeting"), not a clause ("they grow past
    500", "reviewing our proposal", "turning on notifications"). A gap question ("how many days after A did B") needs both
    records rather than a window, so its relation is "gap" (the caller fetches the anchor only)."""
    for relation, pattern in _PATTERNS:
        m = pattern.search(question)
        if m:
            anchor = _ANCHOR_STOP.sub("", m.group("a")).strip(" ?.,;")
            if not anchor or anchor.lower() in {"i", "we", "it", "that", "this"}:
                continue
            if relation in ("after", "before") and _NOT_AN_EVENT.match(anchor):
                continue
            if relation in ("after", "before") and _GAP_QUESTION.match(question):
                relation = "gap"
            return {"relation": relation, "anchor_query": anchor}
    return None


def mentioned_dates(text: str, default_year: int) -> list[date]:
    out = []
    for m in _DATE_RE.finditer(text):
        month = _MONTHS[m.group(1).lower()[:3]]
        try:
            out.append(date(int(m.group(3)) if m.group(3) else default_year, month, int(m.group(2))))
        except ValueError:
            continue
    return out


def event_time(uid: str, store: MemoryStore) -> datetime:
    """When the anchor HAPPENS (not when it was delivered): a calendar event's start; a meeting's start; else delivery time."""
    u = store.get(uid)
    if u.source == "calendar" and u.meta.get("start"):
        try:
            return datetime.fromisoformat(str(u.meta["start"]).replace("Z", "+00:00")).replace(tzinfo=u.time.tzinfo) \
                if "T" not in str(u.meta["start"]) else datetime.fromisoformat(str(u.meta["start"]))
        except ValueError:
            pass
    if u.source == "meeting":
        first = get_info(store).record_units[u.record_id][0]
        return store.get(first).time
    return u.time


def event_dates(uid: str, store: MemoryStore) -> list[date]:
    """Calendar dates this record is ABOUT: a calendar event's span; for others the dates written in the text that fall
    on or after delivery (a flight confirmation delivered Sep 12 about Sep 23), else the delivery date."""
    u = store.get(uid)
    if u.source == "calendar" and u.meta.get("start"):
        start = event_time(uid, store).date()
        try:
            end_raw = str(u.meta.get("end") or "")
            end = datetime.fromisoformat(end_raw).date() if end_raw else start
        except ValueError:
            end = start
        n = max(0, min((end - start).days, 6))
        return [start + timedelta(days=i) for i in range(n + 1)]
    mentioned = [d for d in mentioned_dates(u.text, u.time.year) if d >= u.time.date()]
    return mentioned[:2] or [u.time.date()]


def window_candidates(relation: str, anchor_ids: list[str], visible: set[str], store: MemoryStore, same_day_cap: int = 60) -> list[str]:
    """Visible units in the window after/before the first anchor: same local day first, then neighbours in its thread."""
    if not anchor_ids:
        return []
    anchor = anchor_ids[0]
    t = event_time(anchor, store)
    a_rec = store.get(anchor).record_id
    day = t.date()
    out: list[str] = []
    for u in store.units:
        if u.id not in visible or u.record_id == a_rec or "deleted_target" in u.meta:
            continue
        same_day = u.time.date() == day
        if relation == "after" and same_day and u.time >= t:
            out.append(u.id)
        elif relation == "before" and same_day and u.time <= t:
            out.append(u.id)
    out.sort(key=lambda i: store.get(i).time, reverse=(relation == "before"))
    out = out[:same_day_cap]
    info = get_info(store)
    key = store.get(anchor).thread_key
    if key:
        thread = [i for i in info.thread_units.get(key, []) if i in visible]
        later = [i for i in thread if (store.get(i).time > t) == (relation == "after") and store.get(i).record_id != a_rec]
        later.sort(key=lambda i: store.get(i).time, reverse=(relation == "before"))
        out += [i for i in later[:12] if i not in out]
    return out
