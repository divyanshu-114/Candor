"""Date / time / duration phrases in a command -> concrete values, relative to the user's "now" (`as_of`).

WHY a deterministic parser: "tomorrow at 2", "the 25th at 9am", "back half an hour" are cheap to resolve exactly and
a language model occasionally gets them off by a day or an hour. Everything here is a pure function of the text and
`as_of`, so the same command always gives the same answer, and it works without a key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from actions.context import TZ

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WD = "|".join(w[:3] + r"[a-z]*" for w in WEEKDAYS)          # mon, monday, tues, tuesday, ...
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
              "thirty": 30, "forty-five": 45, "forty five": 45, "sixty": 60}

DURATION_RE = re.compile(
    r"\b(?P<n>\d+(?:\.\d+)?|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|forty[- ]five|sixty|half an?|an? half)"
    r"(?:[- ](?P<half>and a half))?\s*[- ]?(?P<unit>minutes?|mins?|hours?|hrs?|h)\b", re.I)
HALF_HOUR_RE = re.compile(r"\bhalf an hour\b|\b30[- ]?min", re.I)


def parse_duration(text: str) -> timedelta | None:
    """"30 minutes", "1 hour", "half an hour", "an hour and a half", "45 min" -> timedelta."""
    low = text.lower()
    if re.search(r"\bhour and a half\b", low):
        return timedelta(minutes=90)
    if HALF_HOUR_RE.search(low):
        return timedelta(minutes=30)
    m = DURATION_RE.search(low)
    if not m:
        return None
    raw = m.group("n").lower()
    if raw.startswith("half") or raw.endswith("half"):
        n = 0.5
    elif raw in _NUM_WORDS:
        n = _NUM_WORDS[raw]
    else:
        n = float(raw)
    unit = m.group("unit")
    return timedelta(hours=n) if unit.startswith(("h",)) else timedelta(minutes=n)


@dataclass
class When:
    day: date | None = None
    tod: time | None = None
    past_word: bool = False          # "yesterday", "last Monday", "3 days ago"
    spans: list[tuple[int, int]] = None  # character spans consumed in the original text

    def at(self, default_day: date) -> datetime | None:
        if self.tod is None:
            return None
        return datetime.combine(self.day or default_day, self.tod, tzinfo=TZ)


def _weekday_index(token: str) -> int:
    t = token.lower()[:3]
    return next(i for i, w in enumerate(WEEKDAYS) if w.startswith(t))


def _next_weekday(base: date, idx: int, strictly_after: bool) -> date:
    delta = (idx - base.weekday()) % 7
    if delta == 0 and strictly_after:
        delta = 7
    return base + timedelta(days=delta)


def _clock(hour: int, minute: int, meridiem: str | None, morning_hint: bool, explicit_at: bool) -> time | None:
    if hour > 23 or minute > 59:
        return None
    if meridiem:
        m = meridiem.lower().replace(".", "")
        if m == "pm" and hour < 12:
            hour += 12
        elif m == "am" and hour == 12:
            hour = 0
    elif hour <= 12 and not morning_hint:
        # no am/pm: working-hours convention -- 1..6 means afternoon, 7..11 morning, 12 noon
        if hour <= 6:
            hour += 12
    elif hour < 12 and morning_hint:
        pass
    return time(hour, minute)


_TIME_PATTERNS = [
    re.compile(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ap>a\.?m\.?|p\.?m\.?)?(?!\d)", re.I),                  # 3:30pm, 15:30
    re.compile(r"\b(?P<h>\d{1,2})\s*(?P<ap>a\.?m\.?|p\.?m\.?)\b", re.I),                                       # 3pm
    re.compile(r"\bat\s+(?P<h>\d{1,2})\b(?!\s*(?:minutes?|mins?|hours?|hrs?|%|:\d))", re.I),                 # at 9
]


def parse_when(text: str, as_of: datetime) -> When:
    """Find the day and the time of day in `text` (either may be missing). Never invents a missing part."""
    low = text.lower()
    today = as_of.astimezone(TZ).date()
    out = When(spans=[])
    morning_hint = bool(re.search(r"\b(morning|a\.?m\.?)\b", low))
    evening_hint = bool(re.search(r"\b(evening|afternoon|tonight)\b", low))

    # --- time of day ---
    if re.search(r"\bnoon\b", low):
        out.tod = time(12, 0)
        m = re.search(r"\bnoon\b", low)
        out.spans.append(m.span())
    elif re.search(r"\bmidnight\b", low):
        out.tod = time(0, 0)
    else:
        for pat in _TIME_PATTERNS:
            for m in pat.finditer(text):
                tail = low[m.end(): m.end() + 12]
                if re.match(r"\s*(?:st|nd|rd|th)\b", tail) or re.match(r"\s*(?:minutes?|mins?|hours?|hrs?)\b", tail):
                    continue                       # "the 25th", "30 minutes"
                if re.match(r"\d:\d$", m.group(0)) or (m.group("h") == "1" and m.groupdict().get("m") == "1"):
                    continue                       # "1:1"
                hour = int(m.group("h"))
                minute = int(m.groupdict().get("m") or 0)
                ap = m.groupdict().get("ap")
                t = _clock(hour, minute, ap, morning_hint, pat is _TIME_PATTERNS[2])
                if t and (ap or ":" in m.group(0) or pat is _TIME_PATTERNS[2]):
                    if evening_hint and not ap and t.hour < 12 and hour <= 11:
                        t = time(hour + 12 if hour < 12 else hour, minute)
                    out.tod = t
                    out.spans.append(m.span())
                    break
            if out.tod:
                break

    # --- day ---
    def mark(m: re.Match) -> None:
        out.spans.append(m.span())

    if m := re.search(r"\bday after tomorrow\b", low):
        out.day = today + timedelta(days=2); mark(m)
    elif m := re.search(r"\btomorrow\b", low):
        out.day = today + timedelta(days=1); mark(m)
    elif m := re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", low):
        out.day = today; mark(m)
    elif m := re.search(r"\byesterday\b", low):
        out.day = today - timedelta(days=1); out.past_word = True; mark(m)
    elif m := re.search(rf"\blast\s+({_WD})\b", low):
        idx = _weekday_index(m.group(1))
        out.day = today - timedelta(days=((today.weekday() - idx) % 7) or 7); out.past_word = True; mark(m)
    elif m := re.search(r"\b(\d+)\s+days?\s+ago\b", low):
        out.day = today - timedelta(days=int(m.group(1))); out.past_word = True; mark(m)
    elif m := re.search(rf"\b(?:next|this|on)\s+({_WD})\b", low) or re.search(rf"\b({_WD})\b", low):
        idx = _weekday_index(m.group(1))
        has_next = bool(re.search(r"\bnext\b", m.group(0)))
        out.day = _next_weekday(today, idx, strictly_after=has_next); mark(m)
    elif m := re.search(r"\b(?:on |by |for )?the (\d{1,2})(?:st|nd|rd|th)\b", low):
        dom = int(m.group(1))
        cur = today
        for _ in range(62):
            if cur.day == dom:
                out.day = cur
                break
            cur += timedelta(days=1)
        mark(m)
    elif m := re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", low):
        try:
            d = date(today.year, _MONTHS[m.group(1)], int(m.group(2)))
            out.day = d if d >= today else d.replace(year=today.year + 1)
        except ValueError:
            pass
        mark(m)
    elif m := re.search(r"\bin (\d+) days?\b", low):
        out.day = today + timedelta(days=int(m.group(1))); mark(m)
    elif m := re.search(r"\bnext week\b", low):
        out.day = today + timedelta(days=7); mark(m)
    return out


_REL_RE = re.compile(
    r"(?P<delta>(?:\d+(?:\.\d+)?|a|an|one|half an?|an? half|two|three|ten|fifteen|twenty|thirty)\s*(?:and a half\s*)?"
    r"(?:minutes?|mins?|hours?|hrs?))\s+(?P<dir>before|after|ahead of|prior to)\s+(?P<ref>.+)$", re.I)


def parse_relative_to_event(text: str) -> tuple[timedelta, int, str] | None:
    """"an hour before the board run-through" -> (1h, -1, "the board run-through"); sign is -1 for before, +1 for after."""
    m = _REL_RE.search(text.strip().rstrip(".!?"))
    if not m:
        return None
    delta = parse_duration(m.group("delta"))
    if delta is None:
        return None
    sign = -1 if m.group("dir").lower() in ("before", "ahead of", "prior to") else 1
    return delta, sign, m.group("ref").strip()
