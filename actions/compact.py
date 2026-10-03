"""Compact, line-oriented rendering of the planner's world.

WHY: the JSON world cost ~3.6k tokens per call and was sent twice per
command. This renders only what planning needs, one record per line:
people, channels, the date table for the next 8 days, and calendar events
within a window around as_of, with attendees shortened to local parts for the
user's own domain. Pure formatting: no LLM, deterministic.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from actions.context import TZ, World

EVENT_WINDOW_BEFORE_DAYS = 2
EVENT_WINDOW_AFTER_DAYS = 16
MAX_ATTENDEES = 6
DATE_DAYS = 8


def _offset(dt: datetime) -> str:
    raw = dt.astimezone(TZ).strftime("%z")
    return f"{raw[:3]}:{raw[3:]}"


def _event_dt(value: str | None) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ)


def _home_domain(world: World) -> str:
    alex = next((p for p in world.people if p.internal and p.email), None)
    return alex.email.split("@", 1)[1] if alex else ""


def _short_email(email: str, domain: str) -> str:
    e = email.strip()
    return e.split("@", 1)[0] if domain and e.lower().endswith("@" + domain.lower()) else e


def events_in_window(world: World) -> list[dict]:
    lo = world.as_of - timedelta(days=EVENT_WINDOW_BEFORE_DAYS)
    hi = world.as_of + timedelta(days=EVENT_WINDOW_AFTER_DAYS)
    keep = []
    for e in world.calendar_events:
        start = _event_dt(e.get("start"))
        if start is not None and lo <= start <= hi:
            keep.append(e)
    return keep


def compact_world(world: World) -> str:
    domain = _home_domain(world)
    off = _offset(world.as_of)
    days = ", ".join((world.as_of + timedelta(days=i)).strftime("%a %Y-%m-%d") for i in range(DATE_DAYS))
    lines = [
        f"today: {world.today} {world.weekday}; timezone America/Los_Angeles, UTC offset {off} "
        f"(write every time as ISO 8601 with {off})",
        f"next days: {days}",
        "PEOPLE (name | slack_id | dm_id | email | int/ext):",
    ]
    for p in world.people:
        lines.append(f"{p.name} | {p.slack_id or '-'} | {p.dm_channel or '-'} | {p.email or '-'} | "
                     f"{'int' if p.internal else 'ext'}")
    if world.ambiguous_first_names:
        lines.append("AMBIGUOUS FIRST NAMES: " + "; ".join(
            f"{first}: {', '.join(names)}" for first, names in sorted(world.ambiguous_first_names.items())))
    lines.append("CHANNELS (id name):")
    lines += [f"{c['id']} {c['name']}" for c in world.channels]
    lines.append(f"EVENTS (id | title | start..end | attendees; bare attendee names are @{domain}):")
    for e in events_in_window(world):
        start, end = _event_dt(e.get("start")), _event_dt(e.get("end"))
        when = (f"{start.strftime('%Y-%m-%d %H:%M')}..{end.strftime('%H:%M')}" if start and end
                else str(e.get("start")))
        att = ", ".join(_short_email(a, domain) for a in (e.get("attendees") or [])[:MAX_ATTENDEES])
        lines.append(f"{e['id']} | {e.get('title')} | {when} | {att}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough chars/4 estimate (the budget check in tests; not billing)."""
    return (len(text) + 3) // 4
