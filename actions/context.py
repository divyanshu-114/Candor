"""Builds a compact, as-of-aware "world" snapshot for the planner: a people
directory (Slack + external, with first-name ambiguity flagged), channels,
calendar events visible at as_of, and a date-resolution table. Read-only --
no side effects, no LLM calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from memory.store import MemoryStore

from actions import config

TZ = ZoneInfo(config.TIMEZONE)

# Senders that are automated, not people -- never add these to the people
# directory even though they show up as email addresses in the data.
_AUTO_SENDER_PREFIXES = (
    "no-reply", "noreply", "notifications", "notification", "digest", "news",
    "events", "partners", "calendar-notification", "reminders", "hello",
)
_NAME_EMAIL_RE = re.compile(r"^\s*([^<]+?)\s*<([^>]+)>\s*$")


def _is_auto_sender(email: str) -> bool:
    local = email.split("@", 1)[0].lower()
    return any(local.startswith(p) for p in _AUTO_SENDER_PREFIXES)


def _parse_as_of(as_of: str | datetime) -> datetime:
    if isinstance(as_of, datetime):
        return as_of
    dt = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ)


@dataclass
class Person:
    name: str
    email: str | None = None
    slack_id: str | None = None
    dm_channel: str | None = None
    internal: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email, "slack_id": self.slack_id,
                "dm_channel": self.dm_channel, "internal": self.internal}


@dataclass
class World:
    as_of: datetime
    people: list[Person]
    ambiguous_first_names: dict[str, list[str]]  # first name -> full names
    channels: list[dict]
    calendar_events: list[dict]
    today: str
    weekday: str
    date_table: list[dict]  # [{"date": "YYYY-MM-DD", "weekday": "Monday"}, ...]

    def find_by_first_name(self, first_name: str) -> list[Person]:
        fn = first_name.strip().lower()
        return [p for p in self.people if p.name.split()[0].lower() == fn]

    def find_by_full_name(self, name: str) -> Person | None:
        n = name.strip().lower()
        for p in self.people:
            if p.name.lower() == n:
                return p
        return None

    def resolve_date_phrase(self, day_of_month: int) -> str | None:
        """'the 25th' -> the next date on/after as_of whose day-of-month is 25."""
        for row in self.date_table:
            if datetime.strptime(row["date"], "%Y-%m-%d").day == day_of_month:
                return row["date"]
        return None

    def to_prompt_dict(self) -> dict:
        return {
            "today": self.today,
            "weekday": self.weekday,
            "date_table": self.date_table,
            "people": [p.to_dict() for p in self.people],
            "ambiguous_first_names": self.ambiguous_first_names,
            "channels": self.channels,
            "calendar_events": self.calendar_events,
        }


def _load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _slack_people(data_dir: Path, channels: list[dict]) -> list[Person]:
    users = _load_json(data_dir / "connectors/slack/users.json")
    dm_by_user: dict[str, str] = {}
    for c in channels:
        if c.get("is_dm") and len(c.get("members", [])) == 2:
            for uid in c["members"]:
                dm_by_user[uid] = c["id"]

    people = []
    for u in users:
        if u.get("is_bot"):
            continue
        people.append(Person(
            name=u["real_name"], email=u.get("email"), slack_id=u["id"],
            dm_channel=dm_by_user.get(u["id"]), internal=True,
        ))
    return people


def _guess_name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    parts = re.split(r"[._-]+", local)
    return " ".join(p.capitalize() for p in parts if p)


def _external_people(data_dir: Path, internal_emails: set[str]) -> list[Person]:
    found: dict[str, str] = {}  # email -> best-known name
    gmail_path = data_dir / "connectors/gmail/messages.jsonl"
    if gmail_path.exists():
        for line in gmail_path.read_text().splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            headers = [msg.get("from", "")] + msg.get("to", []) + msg.get("cc", [])
            for h in headers:
                m = _NAME_EMAIL_RE.match(h)
                if not m:
                    continue
                name, email = m.group(1).strip(), m.group(2).strip().lower()
                if email in internal_emails or _is_auto_sender(email):
                    continue
                if email not in found or len(name) > len(found[email]):
                    found[email] = name

    cal_path = data_dir / "connectors/google_calendar/events.jsonl"
    if cal_path.exists():
        for line in cal_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            emails = [event.get("organizer", "")] + [a.get("email", "") for a in event.get("attendees", [])]
            for email in emails:
                email = email.strip().lower()
                if not email or email in internal_emails or _is_auto_sender(email):
                    continue
                found.setdefault(email, _guess_name_from_email(email))

    return [Person(name=name, email=email, slack_id=None, dm_channel=None, internal=False)
            for email, name in sorted(found.items())]


def _ambiguous_first_names(people: list[Person]) -> dict[str, list[str]]:
    by_first: dict[str, list[str]] = {}
    for p in people:
        first = p.name.split()[0].lower()
        by_first.setdefault(first, []).append(p.name)
    return {first: names for first, names in by_first.items() if len(names) > 1}


def _calendar_events(data_dir: str, as_of: datetime) -> list[dict]:
    store = MemoryStore(data_dir)
    visible = [u for u in store.visible(as_of) if u.source == "calendar"]
    # Current-state-per-id: a later delivered record for the same event id
    # supersedes an earlier one (matches data/README's "current state" model).
    latest_by_id: dict[str, object] = {}
    for u in visible:
        prior = latest_by_id.get(u.record_id)
        if prior is None or u.time > prior.time:
            latest_by_id[u.record_id] = u

    events = []
    for u in latest_by_id.values():
        meta = u.meta
        if meta.get("status") == "cancelled":
            continue
        events.append({
            "id": u.record_id,
            "title": u.title,
            "start": meta.get("start"),
            "end": meta.get("end"),
            "attendees": meta.get("attendees", []),
            "status": meta.get("status"),
            "location": meta.get("location", ""),
        })
    events.sort(key=lambda e: e["start"] or "")
    return events


def build_world(as_of: str | datetime, data_dir: str | None = None) -> World:
    data_dir = data_dir or config.DATA_DIR
    d = Path(data_dir)
    as_of_dt = _parse_as_of(as_of)

    channels_raw = _load_json(d / "connectors/slack/channels.json")
    channels = [{"id": c["id"], "name": c.get("name", ""), "is_dm": c.get("is_dm", False)}
                for c in channels_raw]

    slack_people = _slack_people(d, channels_raw)
    internal_emails = {p.email.lower() for p in slack_people if p.email}
    people = slack_people + _external_people(d, internal_emails)

    date_table = []
    for i in range(config.DATE_TABLE_DAYS):
        day = as_of_dt + timedelta(days=i)
        date_table.append({"date": day.strftime("%Y-%m-%d"), "weekday": day.strftime("%A")})

    return World(
        as_of=as_of_dt,
        people=people,
        ambiguous_first_names=_ambiguous_first_names(people),
        channels=[c for c in channels if not c["is_dm"]],
        calendar_events=_calendar_events(data_dir, as_of_dt),
        today=as_of_dt.strftime("%Y-%m-%d"),
        weekday=as_of_dt.strftime("%A"),
        date_table=date_table,
    )
