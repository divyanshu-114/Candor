"""Deterministic command resolver: act on the best-supported interpretation, ask only when something is truly missing.

POLICY (dry run, so a wrong guess is cheap but a needless question is a failure to help):
  * a person / channel / event that is uniquely identifiable from the command and the data -> ACT
  * two or more equally plausible candidates after every cue (a surname, "on Slack", an organisation such as "at Acme",
    channel membership, a recent interaction), or a REQUIRED value that cannot be derived (the time of a meeting, a
    recipient) -> ASK, with a question that names the candidates / the missing value
  * a command that needs a fact from memory ("the corrected NRR", "what we decided") -> not handled here (returns None)
    so the language model + memory lookup can do it
  * destructive commands, planted instructions, past times: unchanged (actions/rules.py)

Everything is a pure function of (command, world); same input, same plan; no key and no tokens needed. A command the
resolver cannot fully resolve returns None and falls through to the model (or to a specific clarify without a key).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from actions import rules as R
from actions.context import TZ, Person, World
from actions.timeparse import WEEKDAYS, When, parse_duration, parse_relative_to_event, parse_when

Plan = list[dict]
_STOP = {"the", "a", "an", "my", "our", "your", "of", "to", "with", "and", "meeting", "call", "event", "that", "this", "me"}
_SOFT_STOP = {"meeting", "call"}
# A message that DESCRIBES a fact instead of stating it ("the corrected NRR", "the new launch date", "what we decided"): the value lives in
# memory. A modifier alone is not enough ("the updated pricing summary" is a document), so it must precede a fact-like noun.
_FACT_NOUN = r"(?:launch\s+date|launch|nrr|arr|mrr|churn|retention|numbers?|figures?|metrics?|dates?|deadline|price|prices|total|count|amount|rate|status|decision|estimate|forecast|headcount|revenue|budget|timeline|schedule)(?:\s+date)?"
_MEMORY_REF = re.compile(
    rf"\b(?:corrected|correct|new|updated|latest|current|final|revised|real|actual)\s+(?:\w+\s+){{0,2}}?{_FACT_NOUN}\b|"
    r"\bwhat\s+we\s+(?:decided|agreed|said)\b|\bthe\s+(?:numbers|figures?|status|decision)\b(?!\s+(?:are|is)\s+ready)", re.I)
_MSG_VERBS = r"(?:tell|message|ping|dm|text|slack|let)"
_MAIL_VERBS = r"(?:email|e-mail|mail)"
_WHEN_FRAGMENT = re.compile(r"\b(?:tomorrow|today|tonight|yesterday|day after tomorrow|next \w+|on the \d+\w*|by \w+day|at \d+)\b", re.I)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


_NAME_LIKE = re.compile(r"^[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*){0,2}$")


def _ask(question: str) -> Plan:
    return [{"type": "clarify", "args": {"question": question}}]


def _safe_name(phrase: str) -> str:
    """A name phrase may be echoed in a question only if it looks like a name (never planted text or a secret)."""
    p = phrase.strip()
    return p if _NAME_LIKE.match(p) and not any(c.isdigit() for c in p) else "that person"


# ------------------------------------------------------------------ people
def _channel(world: World, phrase: str) -> dict | None:
    """"the route planner channel", "#sales", "sales" -> channel dict (non-DM)."""
    p = phrase.lower().replace("#", " ")
    p = re.sub(r"\b(?:the|channel|in)\b", " ", p)
    words = _tokens(p)
    if not words:
        return None
    for c in world.channels:
        if c.get("is_dm"):
            continue
        name = _tokens(c["name"])
        if words == name or " ".join(words) == " ".join(name):
            return c
    return None


def _candidates(world: World, name_phrase: str) -> list[Person]:
    """People whose full name, or first name when the phrase is a single word, matches."""
    low = " ".join(_tokens(name_phrase))
    if not low:
        return []
    others = [p for p in world.people if p.name != world.self_name]   # you never message yourself
    full = [p for p in others if " ".join(_tokens(p.name)) == low]
    if full:
        return full
    words = low.split()
    if len(words) == 1:
        first = [p for p in others if p.name.split()[0].lower() == words[0]]
        if first:
            return first
        return [p for p in others if words[0] == _tokens(p.name)[-1]]
    return [p for p in others if all(w in _tokens(p.name) for w in words)]


def _apply_cues(cands: list[Person], text: str, world: World) -> tuple[list[Person], str]:
    """Narrow candidates by the cues in the whole command. Returns (remaining, cue name)."""
    low = text.lower()
    if len(cands) < 2:
        return cands, ""
    if re.search(r"\b(?:on slack|via slack|slack dm|slack message)\b", low):
        keep = [p for p in cands if p.slack_id]
        if keep and len(keep) < len(cands):
            return keep, "slack"
    tokens = set(_tokens(text))
    org = [p for p in cands if p.email and any(len(t) >= 4 and p.email.split("@")[-1].split(".")[0].lower().startswith(t) for t in tokens)]
    if org and len(org) < len(cands):
        return org, "organisation"
    for c in world.channels:
        if not c.get("is_dm") and re.search(rf"\b{re.escape(c['name'].replace('-', ' '))}\b", low.replace("-", " ")):
            keep = [p for p in cands if p.slack_id in c.get("members", [])]
            if keep and len(keep) < len(cands):
                return keep, "channel"
    seen = {p.name: world.last_seen.get(p.name) for p in cands}
    ranked = sorted(((t, n) for n, t in seen.items() if t), reverse=True)
    if ranked:
        newest_t, newest = ranked[0]
        recent = (world.as_of - newest_t) <= timedelta(hours=3)
        runner = ranked[1][0] if len(ranked) > 1 else None
        quiet_others = all(seen.get(p.name) is None or (world.as_of - seen[p.name]) > timedelta(hours=24)
                           for p in cands if p.name != newest)
        if recent and (runner is None or quiet_others):
            return [p for p in cands if p.name == newest], "recent interaction"
    return cands, ""


def _person(world: World, phrase: str, whole_command: str) -> tuple[Person | None, Plan | None]:
    """(person, None) when resolved; (None, clarify plan) when ambiguous or unknown."""
    cands = _candidates(world, phrase)
    if not cands:
        return None, _ask(f"I couldn't find anyone called {_safe_name(phrase)} in your contacts. Who did you mean?")
    cands, _cue = _apply_cues(cands, whole_command, world)
    if len(cands) > 1:
        return None, _ask(f"Which {_safe_name(phrase).split()[0]} do you mean: {' or '.join(p.name for p in cands)}?")
    return cands[0], None


# ------------------------------------------------------------------ events
def _rrule(rule: list[str] | None) -> dict:
    out: dict = {}
    for r in rule or []:
        if r.startswith("RRULE:"):
            for part in r[6:].split(";"):
                k, _, v = part.partition("=")
                out[k] = v
    return out


def _instance(event: dict, today: date) -> tuple[datetime, datetime] | None:
    """The occurrence of `event` on/after today (a recurring event's stored start is its FIRST occurrence)."""
    try:
        start = datetime.fromisoformat(str(event["start"]))
        end = datetime.fromisoformat(str(event["end"]))
    except (KeyError, ValueError, TypeError):
        return None
    if start.tzinfo is None:
        start, end = start.replace(tzinfo=TZ), end.replace(tzinfo=TZ)
    rule = _rrule(event.get("recurrence"))
    if rule.get("FREQ") != "WEEKLY":
        return start, end
    interval = int(rule.get("INTERVAL", 1))
    days = [("MO", "TU", "WE", "TH", "FR", "SA", "SU").index(d) for d in rule.get("BYDAY", "MO,TU,WE,TH,FR,SA,SU").split(",")]
    first_monday = start.date() - timedelta(days=start.weekday())
    for k in range(0, 200):
        week = first_monday + timedelta(weeks=k * interval)
        for d in sorted(days):
            day = week + timedelta(days=d)
            if day >= start.date() and day >= today:
                s = datetime.combine(day, start.timetz())
                return s, s + (end - start)
    return start, end


def _find_event(world: World, phrase: str) -> tuple[dict | None, tuple[datetime, datetime] | None, Plan | None]:
    """The calendar event a phrase names. Several events with the same title (a series, or a repeated all-hands): the one
    nearest to now, upcoming first. A tie between DIFFERENT titles -> ask."""
    today = world.as_of.astimezone(TZ).date()
    scored = []
    # first with the soft words ("meeting", "call": "the board meeting" is not "board deck prep"), then without them
    for want in ([t for t in _tokens(phrase) if t not in _STOP - _SOFT_STOP], [t for t in _tokens(phrase) if t not in _STOP]):
        if not want:
            continue
        for e in world.calendar_events:
            title = set(_tokens((e.get("title") or "") + " " + " ".join(a.split("@")[0] for a in e.get("attendees", []))))
            if all(w in title or any(t.startswith(w) or w.startswith(t) for t in title if len(t) > 3 and len(w) > 3) for w in want):
                inst = _instance(e, today)
                if inst is None:
                    continue
                upcoming = inst[1] >= world.as_of
                scored.append((0 if upcoming else 1, abs((inst[0] - world.as_of).total_seconds()), e["id"], e, inst))
        if scored:
            break
    if not scored:
        return None, None, None
    scored.sort(key=lambda t: t[:3])
    best = scored[0]
    titles = {t[3].get("title") for t in scored if t[0] == best[0]}
    if len(titles) > 1:
        return None, None, _ask("Which event do you mean: " + " or ".join(sorted(str(t) for t in titles)[:3]) + "?")
    return best[3], best[4], None


# ------------------------------------------------------------------ building blocks
def _sentence(text: str) -> str:
    text = " ".join(text.split()).strip(" ,.;:")
    return (text[:1].upper() + text[1:] + ".") if text else text


def _strip_cues(rest: str) -> str:
    """Remove routing cues from the part of the command that follows a person's name ("on Slack", "at Acme")."""
    rest = re.sub(r"^\s*(?:on|via|over|by)\s+(?:slack|e-?mail|mail|dm)\b\s*", "", rest, flags=re.I)
    rest = re.sub(r"^\s*(?:at|from|in)\s+[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*\s+(?=(?:the|that|a|about|to)\b)", "", rest)
    return rest


def _split_after_name(rest: str) -> str:
    rest = _strip_cues(rest)
    rest = re.sub(r"^\s*[,:]\s*", "", rest)
    rest = re.sub(r"^\s*(?:that|to say|saying|about|:)\s+", "", rest, flags=re.I)
    return rest.strip()


def _split_person_phrase(rest: str, world: World) -> tuple[str, str]:
    """Longest known name at the start of `rest` -> (name phrase, remainder). Falls back to the first word."""
    words = rest.split()
    for n in (3, 2):
        phrase = " ".join(words[:n])
        if len(words) >= n and _candidates(world, phrase):
            return phrase, " ".join(words[n:])
    w = re.sub(r"[^A-Za-z.'-]", "", words[0]) if words else ""
    # "Alex Johnson": a capitalised word right after the first name is a surname we did not find -> keep both words so the
    # lookup fails loudly ("couldn't find Alex Johnson") instead of quietly picking the first name's owner
    if len(words) > 1 and re.fullmatch(r"[A-Z][a-z]+", words[1]) and not _candidates(world, w + " " + words[1]) \
            and words[1].lower() not in {"the", "that", "about", "to"}:
        return f"{w} {words[1]}", " ".join(words[2:])
    return w, " ".join(words[1:])


def _message(seg: str, world: World, whole: str, defer: bool = False, lookup=None) -> Plan | None:
    lead = re.match(r"^in\s+(?:the\s+)?(?P<chan>#?[\w-]+(?:\s+[\w-]+)?)\s+channel,?\s+(?:please\s+)?(?:tell|message|ping|let)\s+(?P<who>[A-Z][\w.'-]*)\s+(?:that\s+)?(?P<text>.+)$", seg, re.I)
    if lead:
        chan = _channel(world, lead.group("chan") + " channel")
        if chan:
            return [{"type": "slack.send_message", "args": {"to": chan["id"], "text": f"{lead.group('who')}, {lead.group('text').strip(' .')}."}}]
    m = re.match(rf"^(?:please\s+)?(?P<verb>{_MSG_VERBS}|{_MAIL_VERBS})\s+(?P<rest>.+)$", seg, re.I)
    if not m:
        return None
    verb, rest = m.group("verb").lower(), m.group("rest").strip()
    wants_mail = bool(re.match(_MAIL_VERBS, verb)) or bool(re.search(r"\b(?:by|via|over)\s+e-?mail\b", whole, re.I))
    if verb == "let":
        k = re.match(r"^(?P<who>.+?)\s+know\s+(?:that\s+)?(?P<text>.+)$", rest, re.I)
        if not k:
            return None
        who, text = k.group("who"), k.group("text")
    else:
        # a channel? ("the sales channel ...", "#sales ...")
        cm = re.match(r"^(?P<chan>#[\w-]+|(?:the\s+)?[\w-]+(?:\s+[\w-]+)?\s+channel)\s*(?P<text>.*)$", rest, re.I)
        if cm and not wants_mail:
            chan = _channel(world, cm.group("chan"))
            if chan:
                return [{"type": "slack.send_message", "args": {"to": chan["id"], "text": _sentence(_split_after_name(cm.group("text")))}}]
        who, tail = _split_person_phrase(rest, world)
        text = tail
    if not who:
        return None
    person, clar = _person(world, who, whole)
    if clar:
        return clar
    cc: list[Person] = []
    cc_m = re.search(r"\b(?:and\s+)?cc\s+(?P<cc>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)?)", text)
    if cc_m:
        cc_person, clar = _person(world, cc_m.group("cc"), whole)
        if clar:
            return clar
        cc = [cc_person]
        text = (text[:cc_m.start()] + text[cc_m.end():]).strip()
    topic_only = bool(re.match(r"^\s*about\b", _strip_cues(text), re.I))
    text = _split_after_name(text)
    if not text:
        return None
    if _MEMORY_REF.search(text) and defer:
        return None                                  # needs a fact from memory ("the corrected NRR"): the model looks it up
    if topic_only and defer:
        return None                                  # "email X about Y": the body is a fact from memory, let the model look it up
    text = re.sub(r"\band\s+ask\s+if\b", "and ask whether", text, flags=re.I)
    body = _sentence(text)
    if lookup is not None and not defer:
        # no model: fill a described fact from memory with the most relevant sentence (extractive, no invention)
        queries = memory_lookups(whole) or ([text] if topic_only else [])
        facts = [f for q in queries if (f := lookup(q))]
        if facts:
            body = body + " " + " ".join(facts)
    if wants_mail or not person.slack_id:
        if not person.email:
            return _ask(f"I don't have an email address or Slack account for {person.name}. How should I reach them?")
        args = {"to": [person.email], "subject": " ".join(text.split()[:8]).strip(" ,.;:").capitalize() or "Message from Alex", "body": body}
        if cc and cc[0].email:
            args["cc"] = [cc[0].email]
        return [{"type": "gmail.send", "args": args}]
    return [{"type": "slack.send_message", "args": {"to": person.slack_id, "text": body}}]


def _thank(seg: str, world: World, whole: str) -> Plan | None:
    m = re.match(r"^(?:please\s+)?thank\s+(?P<who>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)?)(?P<rest>.*)$", seg)
    if not m:
        return None
    who, rest = _split_person_phrase(m.group("who") + " " + m.group("rest"), world)
    person, clar = _person(world, who, whole)
    if clar:
        return clar
    why = re.search(r"\bfor\s+(?P<why>.+)$", rest, re.I)
    text = "Thank you" + (f" for {why.group('why').strip(' .')}" if why else "") + "!"
    if person.slack_id and not re.search(r"\be-?mail\b", seg, re.I):
        return [{"type": "slack.send_message", "args": {"to": person.slack_id, "text": text}}]
    if person.email:
        return [{"type": "gmail.send", "args": {"to": [person.email], "subject": "Thank you", "body": text}}]
    return None


def _clean_reminder_text(text: str, spans: list[tuple[int, int]]) -> str:
    out = list(text)
    for a, b in spans:
        for i in range(a, min(b, len(out))):
            out[i] = " "
    t = "".join(out)
    t = re.sub(r"\b(?:on the|by|at|on|for|in the|this|next|morning|afternoon|evening|tomorrow|today|tonight)\b(?=\s*(?:$|[.,;]))", " ", t, flags=re.I)
    t = re.sub(r"\s+(?:by|at|on|for)\s*$", "", " ".join(t.split()), flags=re.I)
    return " ".join(t.split()).strip(" ,.;:")


def _reminder(seg: str, world: World, whole: str) -> Plan | None:
    m = re.match(r"^(?:please\s+)?(?:remind me|set (?:a )?reminder)\s+(?:for\s+|about\s+|to\s+)?(?P<body>.+)$", seg, re.I)
    if not m:
        return None
    body = m.group("body").strip()
    rel_m = re.search(r"(?P<delta>(?:\d+(?:\.\d+)?|an?|half an?|one|two|three|ten|fifteen|twenty|thirty)\s*(?:and a half\s*)?(?:minutes?|mins?|hours?|hrs?))\s+"
                      r"(?P<dir>before|after)\s+(?P<ref>.+?)(?:\s+to\s+(?P<todo>.+))?$", body, re.I)
    if rel_m:
        delta = parse_duration(rel_m.group("delta"))
        if delta is None:
            return None
        event, inst, clar = _find_event(world, rel_m.group("ref").strip())
        if clar:
            return clar
        if event is None:
            return None                              # the anchor is not on the calendar (a flight, a meeting in memory): model + memory
        due = inst[0] + (-1 if rel_m.group("dir").lower() == "before" else 1) * delta
        todo = (rel_m.group("todo") or "").strip()
        if due < world.as_of:
            return _ask(R.PAST_CLARIFY)
        text = (todo[:1].upper() + todo[1:]) if todo else f"{event.get('title')}"
        return [{"type": "reminder.create", "args": {"text": text, "due": due.isoformat()}}]
    when = parse_when(body, world.as_of)
    if when.tod is None and when.day is None:
        return None
    if when.past_word:
        return _ask(R.PAST_CLARIFY)
    # a reminder "for the <event> at 5:30": the event names the day when no day is given
    event_ref = re.search(r"(?:for|about)\s+(?:the\s+)?(?P<ev>[A-Za-z0-9 /-]+?)\s+(?:at|on|by)\b", seg, re.I)
    day = when.day
    if day is None and event_ref:
        ev, inst, _ = _find_event(world, event_ref.group("ev"))
        if ev:
            day = inst[0].date()
    if when.tod is None:
        return _ask("What time should I set the reminder for?")
    due = datetime.combine(day or world.as_of.astimezone(TZ).date(), when.tod, tzinfo=TZ)
    if due < world.as_of:
        if when.day is None and day is None:
            due += timedelta(days=1)                 # "at 9" said at 4pm means tomorrow
        else:
            return _ask(R.PAST_CLARIFY)
    text = _clean_reminder_text(body, when.spans or [])
    text = re.sub(r"^(?:to\s+|about\s+|for\s+(?:the\s+)?)", "", text, flags=re.I).strip()
    if not text:
        return None
    return [{"type": "reminder.create", "args": {"text": text[:1].upper() + text[1:] if text.islower() else text, "due": due.isoformat()}}]


def _create_event(seg: str, world: World, whole: str) -> Plan | None:
    m = re.match(r"^(?:please\s+)?(?:book|schedule|set up|create|add|arrange)\s+(?:a\s+|an\s+)?(?P<rest>.+)$", seg, re.I)
    if not m:
        return None
    rest = m.group("rest")
    wm = re.search(r"\bwith\s+(?P<who>.+)$", rest, re.I)
    if not wm:
        return None
    dur = parse_duration(rest.split(" with ")[0]) or parse_duration(rest) or timedelta(minutes=30)
    after = wm.group("who")
    when = parse_when(after, world.as_of)
    topic = None
    tm = re.search(r"\babout\s+(?P<t>.+)$", after, re.I)
    if tm:
        topic = tm.group("t").strip(" .")
        after = after[:tm.start()]
    who_text = re.split(r"\b(?:at|on|tomorrow|today|tonight|next|last|yesterday|this|the day|in \d+|by)\b", after, maxsplit=1, flags=re.I)[0].strip(" ,")
    if not who_text:
        return None
    names = [n.strip() for n in re.split(r"\s*(?:,|\band\b)\s*", who_text) if n.strip()]
    people = []
    for n in names:
        p, clar = _person(world, " ".join(n.split()[:3]), whole)
        if clar:
            return clar
        people.append(p)
    if when.past_word:
        return _ask(R.PAST_CLARIFY)
    if when.tod is None or when.day is None:
        return _ask("When should I schedule it? Please give a day and a time.")
    start = datetime.combine(when.day, when.tod, tzinfo=TZ)
    if start < world.as_of:
        return _ask(R.PAST_CLARIFY)
    emails = [p.email for p in people if p.email]
    if len(emails) != len(people):
        return None
    title = topic[:1].upper() + topic[1:] if topic else "Meeting with " + ", ".join(p.name for p in people)
    return [{"type": "calendar.create_event", "args": {"title": title, "start": start.isoformat(), "end": (start + dur).isoformat(),
                                                       "attendees": emails}}]


def _move_event(seg: str, world: World, whole: str) -> Plan | None:
    m = re.match(r"^(?:please\s+)?(?P<verb>move|push|shift|reschedule|bump|delay)\s+(?:my\s+)?(?P<rest>.+)$", seg, re.I)
    if not m:
        return None
    rest = m.group("rest")
    shift = re.search(r"\b(?P<dir>back|forward|earlier|later|ahead)\b(?:\s+by)?\s+(?P<d>.+)$", rest, re.I)
    to = re.search(r"\s+to\s+(?P<to>.+)$", rest, re.I)
    if shift and parse_duration(shift.group("d")):
        phrase = rest[:shift.start()].strip()
        event, inst, clar = _find_event(world, phrase)
        if clar:
            return clar
        if event is None:
            return None
        delta = parse_duration(shift.group("d"))
        sign = 1 if shift.group("dir").lower() in ("back", "later") else -1   # "push back" = later
        start = inst[0] + sign * delta
        return [{"type": "calendar.update_event", "args": {"event_id": event["id"], "start": start.isoformat(), "end": (start + (inst[1] - inst[0])).isoformat()}}]
    if to:
        phrase = rest[:to.start()].strip()
        event, inst, clar = _find_event(world, phrase)
        if clar:
            return clar
        if event is None:
            return None
        when = parse_when(to.group("to"), world.as_of)
        if when.tod is None and when.day is None:
            return None
        base_day = inst[0].date()
        day = when.day or base_day
        tod = when.tod or inst[0].timetz().replace(tzinfo=None)
        start = datetime.combine(day, tod, tzinfo=TZ)
        if start < world.as_of:
            return _ask(R.PAST_CLARIFY)
        return [{"type": "calendar.update_event", "args": {"event_id": event["id"], "start": start.isoformat(), "end": (start + (inst[1] - inst[0])).isoformat()}}]
    return None


_SEGMENT_STARTS = re.compile(r"\s+(?:and|then|and then)\s+(?=(?:remind|open|launch|message|tell|email|e-mail|mail|ping|dm|book|schedule|set up|create|move|push|let|thank)\b)", re.I)


def resolve_plan(command: str, world: World, defer_to_model: bool = False, lookup=None) -> Plan | None:
    """A plan when EVERY step is resolved, a specific clarify when something is truly missing/ambiguous, else None.

    `defer_to_model=True` (a model is available): commands whose content is a fact from memory ("the corrected NRR", "about the
    NRR fix") return None so the model can look the fact up; without a model the phrase itself becomes the message."""
    text = " ".join(str(command).split())
    if not text:
        return None
    segments = [s.strip() for s in _SEGMENT_STARTS.split(text) if s.strip()]
    plan: Plan = []
    for seg in segments:
        app = re.match(r"^(?:please\s+)?(?i:open|launch)\s+([A-Z][A-Za-z0-9]*(?:\s[A-Z][A-Za-z0-9]*)?)\s*[.!]?$", seg)
        step = ([{"type": "app.open", "args": {"app": app.group(1)}}] if app else
                _reminder(seg, world, text) or _create_event(seg, world, text) or _move_event(seg, world, text) or _thank(seg, world, text)
                or _message(seg, world, text, defer_to_model, lookup))
        if step is None:
            return None
        if any(a["type"] == "clarify" for a in step):
            return step                              # one specific question beats a partial plan
        plan += step
    return plan or None


def memory_lookups(command: str) -> list[str]:
    """Questions for memory when the command DESCRIBES a fact it does not state ("email John the corrected NRR" ->
    "What is the corrected NRR?", "tell Dana what we decided about dark mode" -> "What did we decide about dark mode?").
    A safety net for models that forget to ask for the lookup themselves."""
    out: list[str] = []
    for m in re.finditer(r"\bwhat\s+we\s+(?:decided|agreed|said)(?:\s+(?:about|on|regarding)\s+(?P<topic>[^,.;]+?))?(?=\s+(?:and|then)\b|[,.;]|$)", command, re.I):
        topic = (m.group("topic") or "").strip()
        out.append(f"What did we decide about {topic}?" if topic else "What did we decide?")
    for m in re.finditer(rf"\b(?:corrected|correct|new|updated|latest|current|final|revised|real|actual)\s+(?:\w+\s+){{0,2}}?{_FACT_NOUN}\b", command, re.I):
        phrase = m.group(0).strip()
        out.append(f"What is the {phrase}?")
    return list(dict.fromkeys(out))[:2]
