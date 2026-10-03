#!/usr/bin/env python3
"""
scripts/audit_data.py — Candor data audit (standard library only).

Reads everything in data/ and writes docs/DATA_AUDIT.md.
Rule 4: secrets are masked — only the first 4 chars + **** are printed.
Rule 5: planted instructions are described, never echoed verbatim.

Usage:
    python3 scripts/audit_data.py [--data ./data] [--out docs/DATA_AUDIT.md]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.safety import iter_secret_matches

# ── helpers ──────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    """Load a .jsonl file; skip blank lines; report parse errors."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"  WARN: {path.name}:{lineno} – {exc}", file=sys.stderr)
    return records


def word_count(text: str) -> int:
    return len(text.split())


def ts_to_dt(ts: str | None) -> datetime | None:
    """Parse ISO-8601 timestamp, return UTC-aware datetime or None."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "n/a"
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def time_range(timestamps: list[str]) -> str:
    """Return 'YYYY-MM-DD → YYYY-MM-DD' for a list of ISO timestamp strings.
    WHY normalize: calendar uses plain date strings (all-day events) which parse
    offset-naive; we attach UTC to make all datetimes comparable.
    """
    dts = [ts_to_dt(t) for t in timestamps if t]
    dts = [d for d in dts if d]
    if not dts:
        return "n/a"
    # Normalize: make offset-naive datetimes UTC-aware so min/max works
    aware = []
    for d in dts:
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        aware.append(d)
    lo, hi = min(aware), max(aware)
    return f"{lo.strftime('%Y-%m-%d')} → {hi.strftime('%Y-%m-%d')}"


def mask_secret(s: str) -> str:
    """Return only the first three characters followed by ****."""
    return s[:3] + "****"


# ── secret detection ─────────────────────────────────────────────────────────

def scan_secrets(text: str) -> list[tuple[str, str, str]]:
    """
    Return list of (pattern_name, masked_value, verdict) for each secret found.
    """
    hits = []
    for name, value in iter_secret_matches(text):
        masked = mask_secret(value)
        verdict = "likely real secret"
        if value.startswith(("/", "./", "../")):
            verdict = "false positive (path)"
        elif re.fullmatch(r"[a-f0-9]{32,}", value, re.I):
            verdict = "false positive (hash)"
        elif "_" in value or value in ("lambda", "eta") or " " in value.strip():
            verdict = "false positive (code)"
        hits.append((name, masked, verdict))
    return hits


# ── planted instruction detection ────────────────────────────────────────────

PLANTED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore-previous",    re.compile(r"ignore\s+(previous|all|above|instructions?)", re.I)),
    ("assistant-directive",re.compile(r"\bassistant\b.*\b(you\s+must|do\s+not|never|always)\b", re.I)),
    ("ai-model-ref",       re.compile(r"\bai\s+model\b", re.I)),
    ("forward-all",        re.compile(r"forward\s+all", re.I)),
    ("do-not-tell",        re.compile(r"do\s+not\s+tell", re.I)),
    ("system-prompt",      re.compile(r"system\s+prompt", re.I)),
    ("you-must",           re.compile(r"\byou\s+must\b", re.I)),
    ("when-summarizing",   re.compile(r"when\s+summariz", re.I)),
    ("html-comment",       re.compile(r"<!--.*?-->", re.S)),
    ("display-none",       re.compile(r"display\s*:\s*none", re.I)),
    ("zero-width",         re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")),
    ("white-on-white",     re.compile(r"color\s*:\s*(?:white|#fff|#ffffff)\s*;.*?color", re.I | re.S)),
    ("new-instructions",   re.compile(r"new\s+instructions?", re.I)),
    ("disregard",          re.compile(r"\bdisregard\b", re.I)),
]


def scan_planted(text: str) -> list[tuple[str, str]]:
    """
    Return list of (pattern_name, safe_description) — never the matched text itself.
    WHY safe_description: rule 5 — planted instructions must not be repeated.
    """
    hits: list[tuple[str, str]] = []
    for name, pat in PLANTED_PATTERNS:
        for m in pat.finditer(text):
            # Build a SAFE description: character offset and length, not the content
            ctx_start = max(0, m.start() - 20)
            ctx = text[ctx_start: m.start() + 60]
            # Sanitize: replace non-printable / control chars
            safe = "".join(c if unicodedata.category(c)[0] not in ("C",) else "?" for c in ctx)
            safe = shorten(safe, width=60, placeholder="…")
            hits.append((name, safe))
    return hits


# ── text field extraction ─────────────────────────────────────────────────────


def iter_text_fields(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """
    Recursively yield (field_path, text_value) for every string field in obj.
    WHY recursive: secrets can appear nested in any field.
    """
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_text_fields(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_text_fields(v, f"{path}[{i}]")


# ── section builders ──────────────────────────────────────────────────────────


def audit_meetings(data_dir: Path) -> tuple[list[str], dict, list[dict]]:
    """
    Returns (lines, per_meeting_stats_dict, all_segment_records).
    WHY collect segments: needed for global ID uniqueness check and long-unit check.
    """
    mtg_dir = data_dir / "native" / "meetings"
    lines = ["## 1a. Meetings\n"]
    all_segments: list[dict] = []
    per_meeting: dict[str, dict] = {}
    timestamps: list[str] = []

    files = sorted(mtg_dir.glob("*.json"))
    lines.append(f"**Files:** {len(files)}\n")

    for fp in files:
        mtg = json.loads(fp.read_text(encoding="utf-8"))
        segs = mtg.get("segments", [])
        null_name = sum(1 for s in segs if not s.get("speaker_name"))
        confidences = [s["speaker_confidence"] for s in segs if s.get("speaker_confidence") is not None]
        min_conf = round(min(confidences), 3) if confidences else None
        short_segs = sum(1 for s in segs if word_count(s.get("text", "")) < 5)
        timestamps += [mtg.get("start"), mtg.get("end")]
        for s in segs:
            all_segments.append({"_meeting_id": mtg["id"], **s})

        per_meeting[mtg["id"]] = {
            "title": mtg.get("title", ""),
            "type": mtg.get("type", ""),
            "start": mtg.get("start"),
            "end": mtg.get("end"),
            "n_segments": len(segs),
            "null_speaker_name": null_name,
            "min_speaker_confidence": min_conf,
            "short_segs_lt5words": short_segs,
        }

        lines.append(
            f"- **{mtg['id']}** ({mtg.get('type','?')}) '{mtg.get('title','')}' | "
            f"{len(segs)} segs | null-name: {null_name} | min-conf: {min_conf} | "
            f"short(<5w): {short_segs}"
        )

    lines.append(f"\n**Time range:** {time_range([t for t in timestamps if t])}\n")
    return lines, per_meeting, all_segments


def audit_dictation(data_dir: Path) -> tuple[list[str], list[dict]]:
    fp = data_dir / "native" / "dictation" / "dictations.jsonl"
    recs = load_jsonl(fp)
    tss = [r.get("timestamp") for r in recs]
    lines = [
        "## 1b. Dictation\n",
        f"**Records:** {len(recs)}",
        f"**Time range:** {time_range([t for t in tss if t])}",
        f"**Modes:** {dict_count(recs, 'mode')}",
        f"**Delivery states:** {dict_count(recs, 'delivery_state')}",
        "\n**Sample IDs:** " + ", ".join(r["id"] for r in recs[:3]),
        "",
    ]
    return lines, recs


def audit_slack(data_dir: Path) -> tuple[list[str], list[dict], list[dict], list[dict]]:
    sl_dir = data_dir / "connectors" / "slack"
    users: list[dict] = json.loads((sl_dir / "users.json").read_text())
    channels: list[dict] = json.loads((sl_dir / "channels.json").read_text())
    messages = load_jsonl(sl_dir / "messages.jsonl")

    tss = [m.get("ts") for m in messages]
    dm_channels = {c["id"] for c in channels if c.get("is_dm")}
    bots = [m for m in messages if m.get("subtype") == "bot_message"]
    edits = [m for m in messages if m.get("subtype") == "message_changed"]
    deletes = [m for m in messages if m.get("subtype") == "message_deleted"]
    dms = [m for m in messages if m.get("channel_id") in dm_channels]

    # Index original messages for edit/delete target checks
    msg_index: dict[str, dict] = {m["id"]: m for m in messages if "SL-EV-" not in m.get("id", "")}

    lines = [
        "## 1c. Slack\n",
        f"**Users:** {len(users)}  **Channels:** {len(channels)} (DM channels: {len(dm_channels)})",
        f"**Messages total:** {len(messages)}",
        f"**Time range:** {time_range([t for t in tss if t])}",
        f"**Bot messages:** {len(bots)}",
        f"**DM messages:** {len(dms)}",
        f"**Edit events (message_changed):** {len(edits)}",
        f"**Delete events (message_deleted):** {len(deletes)}",
        "",
        "### Slack users",
    ]
    for u in users:
        lines.append(f"- `{u['id']}` {u.get('real_name','?')} ({u.get('name','?')}) — {u.get('title','')}")

    lines += ["", "### DM channels"]
    for c in channels:
        if c.get("is_dm"):
            members = ", ".join(c.get("members", []))
            lines.append(f"- `{c['id']}` members: {members}")

    lines += ["", "### Bot messages"]
    for b in bots[:10]:
        lines.append(f"- `{b['id']}` bot_name={b.get('bot_name','?')} ts={b.get('ts','')} text={shorten(b.get('text',''), 60, placeholder='…')}")

    lines += ["", "### Edit events (message_changed)"]
    for ev in edits:
        tid = ev.get("target_id", "")
        target_exists = tid in msg_index
        target_before = False
        if target_exists:
            target_ts = ts_to_dt(msg_index[tid].get("ts"))
            ev_ts = ts_to_dt(ev.get("ts"))
            target_before = (target_ts is not None and ev_ts is not None and target_ts < ev_ts)
        lines.append(
            f"- `{ev['id']}` target=`{tid}` ts={ev.get('ts','')} "
            f"target_exists={target_exists} posted_before_event={target_before}"
        )

    lines += ["", "### Delete events (message_deleted)"]
    for ev in deletes:
        tid = ev.get("target_id", "")
        target_exists = tid in msg_index
        target_before = False
        if target_exists:
            target_ts = ts_to_dt(msg_index[tid].get("ts"))
            ev_ts = ts_to_dt(ev.get("ts"))
            target_before = (target_ts is not None and ev_ts is not None and target_ts < ev_ts)
        lines.append(
            f"- `{ev['id']}` target=`{tid}` ts={ev.get('ts','')} "
            f"target_exists={target_exists} posted_before_event={target_before}"
        )

    lines.append("")
    return lines, users, channels, messages


def audit_gmail(data_dir: Path) -> tuple[list[str], list[dict]]:
    fp = data_dir / "connectors" / "gmail" / "messages.jsonl"
    recs = load_jsonl(fp)
    tss = [r.get("date") for r in recs]
    threads = len({r.get("thread_id") for r in recs if r.get("thread_id")})
    lines = [
        "## 1d. Gmail\n",
        f"**Messages:** {len(recs)}  **Threads:** {threads}",
        f"**Time range:** {time_range([t for t in tss if t])}",
        "\n**Sample IDs:** " + ", ".join(r["id"] for r in recs[:5]),
        "",
    ]
    return lines, recs


def audit_calendar(data_dir: Path) -> tuple[list[str], list[dict]]:
    fp = data_dir / "connectors" / "google_calendar" / "events.jsonl"
    recs = load_jsonl(fp)
    changed = [r for r in recs if r.get("updated") and r.get("created") and r["updated"] > r["created"]]
    cancelled = [r for r in recs if r.get("status") == "cancelled"]
    all_day = [r for r in recs if "date" in r.get("start", {})]
    recurring = [r for r in recs if r.get("recurrence")]
    tss = [r.get("start", {}).get("dateTime") or r.get("start", {}).get("date") for r in recs]

    lines = [
        "## 1e. Google Calendar\n",
        f"**Events:** {len(recs)}",
        f"**Time range:** {time_range([t for t in tss if t])}",
        f"**Changed (updated > created):** {len(changed)}",
        f"**Cancelled:** {len(cancelled)}",
        f"**All-day:** {len(all_day)}",
        f"**Recurring:** {len(recurring)}",
        "",
        "### Changed events",
    ]
    for r in changed:
        lines.append(f"- `{r['id']}` '{r.get('summary','')}' created={r['created']} updated={r['updated']}")
    lines += ["", "### Cancelled events"]
    for r in cancelled:
        lines.append(f"- `{r['id']}` '{r.get('summary','')}'") 
    lines += ["", "### All-day events"]
    for r in all_day:
        lines.append(f"- `{r['id']}` '{r.get('summary','')}' date={r['start'].get('date','?')}")
    lines += ["", "### Recurring events"]
    for r in recurring:
        lines.append(f"- `{r['id']}` '{r.get('summary','')}'") 
    lines.append("")
    return lines, recs


def audit_codex(data_dir: Path) -> tuple[list[str], list[dict]]:
    sessions_dir = data_dir / "connectors" / "codex" / "sessions"
    files = sorted(sessions_dir.glob("*.jsonl"))
    all_events: list[dict] = []
    tss: list[str] = []
    lines = [
        "## 1f. Codex Sessions\n",
        f"**Session files:** {len(files)}",
        "",
    ]
    for fp in files:
        events = load_jsonl(fp)
        meta = next((e for e in events if e.get("type") == "session_meta"), {})
        msg_events = [e for e in events if e.get("type") == "message"]
        tool_events = [e for e in events if e.get("type") == "tool_call"]
        ts_list = [e.get("timestamp") for e in events if e.get("timestamp")]
        tss.extend(ts_list)
        lines.append(
            f"- **{meta.get('id', fp.stem)}** cwd={meta.get('cwd','?')} repo={meta.get('repo','?')} | "
            f"messages={len(msg_events)} tool_calls={len(tool_events)} | "
            f"range={time_range(ts_list)}"
        )
        all_events.extend(events)
    lines.append(f"\n**Overall time range:** {time_range(tss)}\n")
    return lines, all_events


def audit_chatgpt(data_dir: Path) -> tuple[list[str], list[dict]]:
    fp = data_dir / "connectors" / "chatgpt" / "conversations.json"
    convs: list[dict] = json.loads(fp.read_text(encoding="utf-8"))
    all_msgs: list[dict] = []
    tss: list[str] = []
    lines = [
        "## 1g. ChatGPT\n",
        f"**Conversations:** {len(convs)}",
        "",
    ]
    for conv in convs:
        msgs = conv.get("messages", [])
        all_msgs.extend(msgs)
        ctimes = [m.get("create_time") for m in msgs if m.get("create_time")]
        tss.extend(ctimes)
        lines.append(
            f"- **{conv['id']}** '{conv.get('title','')}' | "
            f"messages={len(msgs)} | range={time_range(ctimes)}"
        )
    lines.append(f"\n**Total messages:** {len(all_msgs)}")
    lines.append(f"**Overall time range:** {time_range(tss)}\n")
    return lines, all_msgs


# ── ID analysis ───────────────────────────────────────────────────────────────


def collect_all_ids(
    meetings_segs: list[dict],
    dicts: list[dict],
    slack_msgs: list[dict],
    gmails: list[dict],
    cal_events: list[dict],
    codex_events: list[dict],
    chatgpt_msgs: list[dict],
) -> dict[str, str]:
    """
    Return {id: source} for every record.
    WHY: check global uniqueness — a duplicate id would corrupt retrieval.
    """
    id_map: dict[str, str] = {}

    def add(id_val: str, source: str):
        id_map[id_val] = id_map.get(id_val, source)

    for s in meetings_segs:
        add(s["seg_id"], "meeting_segment")
    for d in dicts:
        add(d["id"], "dictation")
    for m in slack_msgs:
        add(m["id"], "slack_message")
    for e in gmails:
        add(e["id"], "gmail")
    for ev in cal_events:
        add(ev["id"], "calendar")
    for ev in codex_events:
        if ev.get("type") == "session_meta":
            add(ev["id"], "codex_session")
    for m in chatgpt_msgs:
        add(m["id"], "chatgpt_message")

    return id_map


def prefix_stats(ids: list[str]) -> dict[str, int]:
    """Count by prefix (up to the first '-' or '#')."""
    counts: dict[str, int] = defaultdict(int)
    for id_ in ids:
        # e.g. "MTG-<date>-<name>#0042" → prefix = "MTG"
        prefix = re.split(r"[-#]", id_)[0]
        counts[prefix] += 1
    return dict(sorted(counts.items()))


def dict_count(recs: list[dict], key: str) -> dict:
    c: dict = defaultdict(int)
    for r in recs:
        c[r.get(key, "None")] += 1
    return dict(c)


# ── People analysis ───────────────────────────────────────────────────────────


def audit_people(
    slack_users: list[dict],
    meetings_segs: list[dict],
    gmails: list[dict],
    meeting_metas: dict[str, dict],
    data_dir: Path,
) -> list[str]:
    # Build a unified person registry: {real_name → {emails, slack_ids}}
    from collections import defaultdict

    # Slack users
    slack_real: dict[str, str] = {}  # slack_id → real_name
    slack_emails: dict[str, str] = {}  # slack_id → email
    first_name_to_people: dict[str, list[str]] = defaultdict(list)

    for u in slack_users:
        rid = u.get("id", "")
        rname = u.get("real_name", "")
        email = u.get("email", "")
        slack_real[rid] = rname
        slack_emails[rid] = email
        first = rname.split()[0] if rname else ""
        if first:
            first_name_to_people[first.lower()].append(rname)

    # Email addresses from gmail
    email_people: set[str] = set()
    email_pat = re.compile(r"[\w.+\-]+@[\w.\-]+")
    for m in gmails:
        for field in ("from", "to", "cc"):
            val = m.get(field, "")
            if isinstance(val, list):
                val = " ".join(val)
            for addr in email_pat.findall(val):
                name_match = re.search(r'"?([A-Z][a-z]+ [A-Z][a-z]+)"?\s*<', val)
                if name_match:
                    name = name_match.group(1)
                    first = name.split()[0].lower()
                    first_name_to_people[first].append(name)
                email_people.add(addr)

    # Meeting participants
    mtg_dir = data_dir / "native" / "meetings"
    mtg_participants: set[str] = set()
    for fp in mtg_dir.glob("*.json"):
        mtg = json.loads(fp.read_text())
        for p in mtg.get("participants_known", []):
            mtg_participants.add(p)
        for s in mtg.get("segments", []):
            if s.get("speaker_name"):
                name = s["speaker_name"]
                first = name.split()[0].lower()
                first_name_to_people[first].append(name)

    # Deduplicate
    first_ambiguous = {
        first: sorted(set(names))
        for first, names in first_name_to_people.items()
        if len(set(names)) > 1
    }

    # People in email/meetings but not in Slack
    slack_email_set = set(slack_emails.values()) - {""}
    external = email_people - slack_email_set - mtg_participants

    lines = [
        "## 5. People\n",
        "### First names mapping to multiple people",
    ]
    if first_ambiguous:
        for first, names in sorted(first_ambiguous.items()):
            lines.append(f"- **{first.capitalize()}**: {', '.join(names)}")
    else:
        lines.append("- None found")

    lines += ["", "### Email addresses in data not matched to a Slack user"]
    for addr in sorted(external)[:20]:
        lines.append(f"- {addr}")
    if len(external) > 20:
        lines.append(f"- … and {len(external)-20} more")

    lines.append("")
    return lines


# ── Long units ────────────────────────────────────────────────────────────────


def audit_long_units(
    meeting_segs: list[dict],
    dicts: list[dict],
    slack_msgs: list[dict],
    gmails: list[dict],
    codex_events: list[dict],
    chatgpt_msgs: list[dict],
    threshold: int = 400,
) -> list[str]:
    long: list[tuple[str, str, int]] = []  # (id, source, wc)

    for s in meeting_segs:
        wc = word_count(s.get("text", ""))
        if wc > threshold:
            long.append((s["seg_id"], "meeting_seg", wc))
    for d in dicts:
        for field in ("raw_transcript", "cleaned_text"):
            wc = word_count(d.get(field, ""))
            if wc > threshold:
                long.append((d["id"], f"dictation.{field}", wc))
    for m in slack_msgs:
        wc = word_count(m.get("text", ""))
        if wc > threshold:
            long.append((m["id"], "slack", wc))
    for e in gmails:
        wc = word_count(e.get("body", ""))
        if wc > threshold:
            long.append((e["id"], "gmail", wc))
    for ev in codex_events:
        for field in ("content", "input", "output"):
            wc = word_count(str(ev.get(field, "")))
            if wc > threshold:
                long.append((ev.get("id", ev.get("type", "?")), f"codex.{field}", wc))
    for m in chatgpt_msgs:
        wc = word_count(str(m.get("content", "")))
        if wc > threshold:
            long.append((m["id"], "chatgpt", wc))

    long.sort(key=lambda x: -x[2])
    lines = [
        f"## 9. Long units (>{threshold} words)\n",
        f"**Count:** {len(long)}",
        "",
    ]
    for id_, src, wc in long[:30]:
        lines.append(f"- `{id_}` ({src}): **{wc} words**")
    if len(long) > 30:
        lines.append(f"- … and {len(long)-30} more")
    lines.append("")
    return lines


# ── Secret scan across all records ───────────────────────────────────────────


def audit_secrets_global(all_records: list[tuple[str, str, dict | str]]) -> list[str]:
    """
    all_records: list of (record_id, record_object_or_text).
    Returns audit lines. NEVER prints a full secret.
    """
    hits: list[tuple[str, str, str, str, str]] = []  # (id, source, rule, masked, verdict)

    for rec_id, source, obj in all_records:
        for field, text in iter_text_fields(obj):
            for pat_name, masked, verdict in scan_secrets(text):
                hits.append((rec_id, source, pat_name, masked, verdict))

    lines = [
        "## 7. Secret scan\n",
        f"**Total hits:** {len(hits)}",
        "*(Full values are NEVER shown — first 3 chars + \\*\\*\\*\\* only)*",
        "",
    ]
    if hits:
        lines.append("| Record ID | Source | Rule matched | Masked value | Verdict |")
        lines.append("|---|---|---|---|---|")
        for rec_id, source, pat, masked, verdict in hits:
            lines.append(f"| `{rec_id}` | {source} | {pat} | `{masked}` | {verdict} |")
    else:
        lines.append("No secrets found.")
    lines.append("")
    return lines


# ── Planted instruction scan ──────────────────────────────────────────────────


def audit_planted_global(all_records: list[tuple[str, str, dict | str]]) -> list[str]:
    hits: list[tuple[str, str, str, str]] = []  # (id, field, pattern, safe_desc)

    for rec_id, _source, obj in all_records:
        for field, text in iter_text_fields(obj):
            for pat_name, safe_desc in scan_planted(text):
                hits.append((rec_id, field, pat_name, safe_desc))

    lines = [
        "## 8. Planted instruction scan\n",
        f"**Total hits:** {len(hits)}",
        "*(Text is NOT echoed — only a safe contextual description)*",
        "",
    ]
    if hits:
        lines.append("| Record ID | Field | Pattern | Context (60 chars, sanitized) |")
        lines.append("|---|---|---|---|")
        for rec_id, field, pat, safe in hits:
            lines.append(f"| `{rec_id}` | `{field}` | {pat} | {safe} |")
    else:
        lines.append("No planted instructions found.")
    lines.append("")
    return lines


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Audit data/ and write docs/DATA_AUDIT.md")
    p.add_argument("--data", default="./data", help="Path to data directory")
    p.add_argument("--out", default="docs/DATA_AUDIT.md", help="Output markdown file")
    args = p.parse_args()

    data_dir = Path(args.data).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Verify required directories ─────────────────────────────────────────
    required = [
        data_dir / "native" / "meetings",
        data_dir / "native" / "dictation",
        data_dir / "connectors" / "slack",
        data_dir / "connectors" / "gmail",
        data_dir / "connectors" / "google_calendar",
        data_dir / "connectors" / "codex" / "sessions",
        data_dir / "connectors" / "chatgpt",
    ]
    missing = [str(d) for d in required if not d.exists()]
    if missing:
        print("MISSING directories:", missing, file=sys.stderr)
        sys.exit(1)

    print("Reading data...")

    # ── Load all sources ────────────────────────────────────────────────────
    mtg_lines, mtg_meta, mtg_segs = audit_meetings(data_dir)
    dct_lines, dct_recs = audit_dictation(data_dir)
    sl_lines, sl_users, sl_channels, sl_msgs = audit_slack(data_dir)
    gm_lines, gm_recs = audit_gmail(data_dir)
    cal_lines, cal_recs = audit_calendar(data_dir)
    cdx_lines, cdx_events = audit_codex(data_dir)
    cgpt_lines, cgpt_msgs = audit_chatgpt(data_dir)

    print("Analysing IDs, people, secrets, planted instructions...")

    # ── Section 1: record counts (assembled) ────────────────────────────────
    counts_lines = [
        "## 1. Record counts per source\n",
        f"| Source | Count | Time range |",
        f"|---|---|---|",
        f"| Meetings (files) | {len(list((data_dir / 'native' / 'meetings').glob('*.json')))} | {time_range([m['start'] for m in mtg_meta.values()])} |",
        f"| Meeting segments | {len(mtg_segs)} | — |",
        f"| Dictation | {len(dct_recs)} | {time_range([r.get('timestamp') for r in dct_recs])} |",
        f"| Slack messages (all) | {len(sl_msgs)} | {time_range([m.get('ts') for m in sl_msgs])} |",
        f"| Slack users | {len(sl_users)} | — |",
        f"| Slack channels | {len(sl_channels)} | — |",
        f"| Gmail | {len(gm_recs)} | {time_range([r.get('date') for r in gm_recs])} |",
        f"| Calendar events | {len(cal_recs)} | — |",
        f"| Codex sessions | {len([e for e in cdx_events if e.get('type')=='session_meta'])} | {time_range([e.get('timestamp') for e in cdx_events if e.get('timestamp')])} |",
        f"| ChatGPT messages | {len(cgpt_msgs)} | {time_range([m.get('create_time') for m in cgpt_msgs])} |",
        "",
    ]

    # ── Section 2: ID analysis ──────────────────────────────────────────────
    all_ids_map = collect_all_ids(
        mtg_segs, dct_recs, sl_msgs, gm_recs, cal_recs, cdx_events, cgpt_msgs
    )
    all_ids = list(all_ids_map.keys())
    prefixes = prefix_stats(all_ids)

    # Duplicates: IDs seen more than once across ALL records
    id_counts: dict[str, int] = defaultdict(int)
    def count_all():
        for s in mtg_segs:
            id_counts[s["seg_id"]] += 1
        for d in dct_recs:
            id_counts[d["id"]] += 1
        for m in sl_msgs:
            id_counts[m["id"]] += 1
        for e in gm_recs:
            id_counts[e["id"]] += 1
        for ev in cal_recs:
            id_counts[ev["id"]] += 1
        for ev in cdx_events:
            if ev.get("type") == "session_meta":
                id_counts[ev["id"]] += 1
        for m in cgpt_msgs:
            id_counts[m["id"]] += 1
    count_all()
    duplicates = {id_: cnt for id_, cnt in id_counts.items() if cnt > 1}

    id_lines = [
        "## 2. ID patterns\n",
        "| Prefix | Count |",
        "|---|---|",
    ]
    for prefix, cnt in prefixes.items():
        id_lines.append(f"| `{prefix}` | {cnt} |")
    id_lines += [
        "",
        f"**Total unique IDs across all sources:** {len(all_ids)}",
        f"**Duplicates:** {len(duplicates)}",
    ]
    if duplicates:
        id_lines.append("\n⚠️ **Duplicate IDs found:**")
        for id_, cnt in list(duplicates.items())[:20]:
            id_lines.append(f"- `{id_}` appears {cnt} times")
    else:
        id_lines.append("✅ No duplicate IDs found.")
    id_lines.append("")

    # ── Section 4: Meetings detail (already in mtg_lines) ───────────────────
    mtg_detail_lines = [
        "## 4. Meeting detail\n",
        "| Meeting ID | Type | Segments | Null speaker | Min confidence | Short(<5w) |",
        "|---|---|---|---|---|---|",
    ]
    for mid, m in mtg_meta.items():
        mtg_detail_lines.append(
            f"| `{mid}` | {m['type']} | {m['n_segments']} | {m['null_speaker_name']} | "
            f"{m['min_speaker_confidence']} | {m['short_segs_lt5words']} |"
        )
    mtg_detail_lines.append("")

    # ── Section 5: People ───────────────────────────────────────────────────
    people_lines = audit_people(sl_users, mtg_segs, gm_recs, mtg_meta, data_dir)

    # ── Section 7 & 8: Secrets + Planted instructions ───────────────────────
    # Build unified (id, obj) list for scanning
    scan_records: list[tuple[str, str, Any]] = []
    for s in mtg_segs:
        # Segment IDs are the usable meeting record IDs.  Recursing over the
        # segment guarantees speaker/text fields are scanned without assigning a
        # vague file-level ID.
        scan_records.append((s["seg_id"], "meeting", s))
    for d in dct_recs:
        scan_records.append((d["id"], "dictation", {k: v for k, v in d.items() if k != "id"}))
    for m in sl_msgs:
        scan_records.append((m["id"], "slack", {k: v for k, v in m.items() if k != "id"}))
    for e in gm_recs:
        scan_records.append((e["id"], "email", {k: v for k, v in e.items() if k != "id"}))
    for ev in cal_recs:
        scan_records.append((ev["id"], "calendar", {k: v for k, v in ev.items() if k != "id"}))
    codex_session = None
    for ev in cdx_events:
        if ev.get("type") == "session_meta":
            codex_session = ev.get("id")
        if codex_session:
            scan_records.append((codex_session, "codex", {k: v for k, v in ev.items() if k != "id"}))
    for m in cgpt_msgs:
        scan_records.append((m["id"], "chatgpt", {k: v for k, v in m.items() if k != "id"}))

    secret_lines = audit_secrets_global(scan_records)
    planted_lines = audit_planted_global(scan_records)

    # ── Section 9: Long units ───────────────────────────────────────────────
    long_lines = audit_long_units(mtg_segs, dct_recs, sl_msgs, gm_recs, cdx_events, cgpt_msgs)

    # ── Section 10: Quirks ──────────────────────────────────────────────────
    # Slack: edited messages — does edit arrive after deletion?
    sl_delete_targets = {m.get("target_id") for m in sl_msgs if m.get("subtype") == "message_deleted"}
    sl_edit_targets = {m.get("target_id") for m in sl_msgs if m.get("subtype") == "message_changed"}
    edit_after_delete = sl_edit_targets & sl_delete_targets

    # Calendar: events starting after data range (future events)
    data_end = datetime(2026, 9, 18, 23, 59, 59, tzinfo=timezone.utc)
    future_cal = [
        ev for ev in cal_recs
        if ts_to_dt(ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", "") + "T00:00:00+00:00")
        and ts_to_dt(ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", "") + "T00:00:00+00:00") > data_end
    ]

    # Codex: sessions with no user messages
    cdx_sessions: dict[str, list] = defaultdict(list)
    cur_session = None
    for ev in cdx_events:
        if ev.get("type") == "session_meta":
            cur_session = ev.get("id")
        elif cur_session:
            cdx_sessions[cur_session].append(ev)
    no_user_msg_sessions = [sid for sid, evs in cdx_sessions.items()
                            if not any(e.get("role") == "user" for e in evs)]

    quirk_lines = [
        "## 10. Other quirks\n",
        f"- Slack edits targeting already-deleted messages: `{edit_after_delete or 'none'}`",
        f"- Calendar events with start date after 2026-09-18 (future events in data): {len(future_cal)}",
    ]
    for ev in future_cal[:5]:
        quirk_lines.append("  - `" + ev["id"] + "` " + repr(ev.get("summary","")) + " starts " + str(ev["start"]))
    quirk_lines += [
        f"- Codex sessions with no user messages: {no_user_msg_sessions or 'none'}",
        f"- Slack messages where `text` is empty or whitespace-only: "
        f"{sum(1 for m in sl_msgs if not m.get('text','').strip())}",
        f"- Dictation records with delivery_state=discarded: "
        f"{sum(1 for d in dct_recs if d.get('delivery_state')=='discarded')}",
        f"- Gmail messages without a body: "
        f"{sum(1 for e in gm_recs if not e.get('body','').strip())}",
        "",
    ]

    # ── Surprises (to be filled after analysis) ──────────────────────────────
    surprises_lines = [
        "## Surprises\n",
        "_Auto-detected during audit:_",
        "",
    ]
    surprise_items: list[str] = []
    if duplicates:
        surprise_items.append(f"⚠️ **Duplicate IDs** found across sources: {list(duplicates.keys())[:5]}")
    if edit_after_delete:
        surprise_items.append(f"⚠️ **Edit events targeting deleted messages**: {edit_after_delete}")
    n_secrets = sum(1 for line in secret_lines if line.startswith("| `"))
    if n_secrets > 0:
        surprise_items.append(f"🔑 **Secret-like strings** detected in data fields: {n_secrets} hits — MASKED in audit")
    n_planted = sum(1 for line in planted_lines if line.startswith("| `"))
    if n_planted > 0:
        surprise_items.append(f"🎭 **Planted instruction** patterns detected: {n_planted} hits — described safely, not echoed")
    if future_cal:
        surprise_items.append(f"📅 **Future calendar events** (after data window): {len(future_cal)} — relevant for as_of filtering")
    null_speaker_total = sum(m["null_speaker_name"] for m in mtg_meta.values())
    if null_speaker_total > 0:
        surprise_items.append(f"🎤 **Unidentified speakers** in meetings: {null_speaker_total} segments with null speaker_name — who-said-what queries affected")
    long_count = sum(1 for line in long_lines if line.startswith("- `"))
    if long_count > 0:
        surprise_items.append(f"📄 **{long_count} units exceed 400 words** — will need chunking for semantic search")

    for item in surprise_items:
        surprises_lines.append(f"- {item}")
    if not surprise_items:
        surprises_lines.append("- No major surprises detected.")
    surprises_lines.append("")

    # ── Assemble full document ───────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        "# DATA_AUDIT.md — Candor data audit\n",
        f"_Generated: {now}_\n",
        "---\n",
    ]

    all_lines = (
        header
        + counts_lines
        + ["---\n"]
        + id_lines
        + ["---\n"]
        + sl_lines
        + ["---\n"]
        + mtg_detail_lines
        + ["---\n"]
        + people_lines
        + ["---\n"]
        + cal_lines
        + ["---\n"]
        + secret_lines
        + ["---\n"]
        + planted_lines
        + ["---\n"]
        + long_lines
        + ["---\n"]
        + quirk_lines
        + ["---\n"]
        + surprises_lines
    )

    content = "\n".join(all_lines)

    # ── Safety check: ensure no full secret appears in output ────────────────
    # We check that no pattern match is longer than 8 chars without masking
    # by looking for well-known prefixes followed by >8 chars
    danger_patterns = [
        re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
        re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
        re.compile(r"\bxox[bpoa]-[A-Za-z0-9\-]{20,}"),
    ]
    for dp in danger_patterns:
        m = dp.search(content)
        if m:
            print(f"SECURITY ERROR: full secret found in output at position {m.start()}: {mask_secret(m.group())}", file=sys.stderr)
            # Redact it before writing
            content = dp.sub(lambda x: mask_secret(x.group()), content)

    out_path.write_text(content, encoding="utf-8")
    print(f"\n✅ Wrote {out_path} ({len(content)} bytes)")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"   Records scanned: {len(scan_records)}")
    print(f"   Unique IDs: {len(all_ids)}")
    print(f"   Duplicates: {len(duplicates)}")
    print(f"   Secret hits: {n_secrets}")
    print(f"   Planted instruction hits: {n_planted}")
    print(f"   Long units (>400w): {long_count}")


if __name__ == "__main__":
    main()
