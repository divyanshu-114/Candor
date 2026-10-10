import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

from memory.models import Unit
from memory.safety import mask_secrets, neutralize_injection

def _dt(s: str) -> datetime:
    if not s:
        return datetime.min # fallback, shouldn't happen with valid data
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))

def _clean_text(text: str) -> Tuple[str, bool]:
    text = mask_secrets(text)
    text, has_injection = neutralize_injection(text)
    return text, has_injection


def _email_name_map(data_dir: Path, slack_names: Dict[str, str]) -> Dict[str, str]:
    """Resolve identities only for search metadata; display text stays untouched."""
    result: Dict[str, str] = {}
    # Slack's users.json carries each member's email: use it (the old code guessed first.last@<our domain>).
    users_path = data_dir / "connectors/slack/users.json"
    if users_path.exists():
        for u in json.loads(users_path.read_text()):
            if u.get("email") and (u.get("real_name") or u.get("name")):
                result.setdefault(u["email"].lower(), u.get("real_name") or u.get("name"))
    for path, fields in (
        (data_dir / "connectors/gmail/messages.jsonl", ("from", "to", "cc")),
        (data_dir / "connectors/google_calendar/events.jsonl", ("organizer",)),
    ):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            values = []
            for field in fields:
                value = record.get(field, [])
                values.extend(value if isinstance(value, list) else [value])
            for value in values:
                match = re.search(r"([^<]+?)\s*<([^>]+)>", str(value))
                if match:
                    result[match.group(2).lower()] = match.group(1).strip()
    return result

def load_all(data_dir: str) -> Tuple[List[Unit], Dict[str, datetime], Dict[str, List[Tuple[datetime, str]]]]:
    d = Path(data_dir)
    units = []
    deletions: Dict[str, datetime] = {}
    edits: Dict[str, List[Tuple[datetime, str]]] = {}

    # 1. Meetings
    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text())
        start = _dt(m["start"])
        title = m.get("title")
        for i, s in enumerate(m.get("segments", [])):
            text, has_injection = _clean_text(s.get("text", ""))
            
            speaker_name = s.get("speaker_name")
            if speaker_name:
                speaker = speaker_name
                speaker_known = True
            else:
                speaker = None
                speaker_known = False

            meta = {
                "type": m.get("type"),
                "participants": m.get("participants_known", []),
                "channel": s.get("channel"),
                "has_injection": has_injection
            }
            units.append(Unit(
                id=s["seg_id"],
                record_id=m["id"],
                source="meeting",
                time=start + timedelta(seconds=s.get("end_s", 0)),
                text=text,
                title=title,
                speaker=speaker,
                speaker_label=s.get("speaker_label"),
                speaker_confidence=s.get("speaker_confidence"),
                speaker_known=speaker_known,
                thread_key=m["id"],
                seq=i,
                meta=meta
            ))

    # 2. Dictation
    dictations_file = d / "native/dictation/dictations.jsonl"
    if dictations_file.exists():
        for i, line in enumerate(open(dictations_file)):
            if not line.strip(): continue
            x = json.loads(line)
            raw_clean, raw_injection = _clean_text(x.get("raw_transcript", ""))
            cleaned, cleaned_injection = _clean_text(x.get("cleaned_text", ""))
            text = cleaned or raw_clean
            has_injection = raw_injection or cleaned_injection
            meta = {
                "mode": x.get("mode"),
                "target_app": x.get("target_app"),
                "target_context": x.get("target_context"),
                "delivery_state": x.get("delivery_state"),
                "raw_transcript": raw_clean,
                "has_injection": has_injection
            }
            units.append(Unit(
                id=x["id"],
                record_id=x["id"],
                source="dictation",
                time=_dt(x["timestamp"]),
                text=text,
                title=f"Dictation into {x.get('target_app')}",
                meta=meta
            ))

    # 3. Slack
    users_file = d / "connectors/slack/users.json"
    names = {}
    if users_file.exists():
        names = {u["id"]: u.get("real_name") or u.get("name") for u in json.loads(users_file.read_text())}
    email_names = _email_name_map(d, names)
    
    channels_file = d / "connectors/slack/channels.json"
    chans = {}
    if channels_file.exists():
        chans = {c["id"]: c.get("name") for c in json.loads(channels_file.read_text())}

    slack_file = d / "connectors/slack/messages.jsonl"
    if slack_file.exists():
        for i, line in enumerate(open(slack_file)):
            if not line.strip(): continue
            x = json.loads(line)
            t = _dt(x["ts"])
            
            subtype = x.get("subtype")
            if subtype == "message_deleted":
                deletions[x["target_id"]] = t
                units.append(Unit(
                    id=x["id"], record_id=x["id"], source="slack", time=t, text="",
                    meta={"deleted_target": x["target_id"]}
                ))
                continue
            
            text, has_injection = _clean_text(x.get("text", ""))
            
            if subtype == "message_changed":
                edits.setdefault(x["target_id"], []).append((t, text))
                units.append(Unit(
                    id=x["id"], record_id=x["id"], source="slack", time=t, text=text,
                    meta={"edit_target": x["target_id"], "has_injection": has_injection}
                ))
                continue

            user_id = x.get("user")
            who = names.get(user_id) if user_id else (x.get("bot_name") or user_id)
            channel_name = chans.get(x.get("channel_id"), x.get("channel_id"))
            
            meta = {
                "subtype": subtype,
                "reactions": x.get("reactions", []),
                "slack_user_names": names,
                "email_names": email_names,
                "has_injection": has_injection
            }
            units.append(Unit(
                id=x["id"],
                record_id=x["id"],
                source="slack",
                time=t,
                text=text,
                title=channel_name,
                speaker=who,
                speaker_known=bool(who),
                thread_key=x.get("thread_parent_id") or x.get("channel_id"),
                meta=meta
            ))

    # 4. Gmail
    gmail_file = d / "connectors/gmail/messages.jsonl"
    if gmail_file.exists():
        for i, line in enumerate(open(gmail_file)):
            if not line.strip(): continue
            x = json.loads(line)
            body, has_injection = _clean_text(x.get("body", ""))
            
            # format text as instructed
            to_str = ", ".join(x.get("to", []))
            cc_str = ", ".join(x.get("cc", []))
            text = f"From: {x.get('from')}\nTo: {to_str}\n"
            if cc_str:
                text += f"Cc: {cc_str}\n"
            text += f"Subject: {x.get('subject')}\n\n{body}"

            meta = {
                "labels": x.get("labels", []),
                "from": x.get("from", ""), "to": x.get("to", []), "cc": x.get("cc", []),
                "email_names": email_names,
                "has_injection": has_injection
            }
            units.append(Unit(
                id=x["id"],
                record_id=x["id"],
                source="email",
                time=_dt(x["date"]),
                text=text,
                title=x.get("subject"),
                thread_key=x.get("thread_id"),
                meta=meta
            ))

    # 5. Calendar
    cal_file = d / "connectors/google_calendar/events.jsonl"
    if cal_file.exists():
        for i, line in enumerate(open(cal_file)):
            if not line.strip(): continue
            x = json.loads(line)
            desc, has_injection = _clean_text(x.get("description", ""))
            
            st = x.get("start", {})
            en = x.get("end", {})
            when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
            att = ", ".join(a.get("email", "") for a in x.get("attendees", []))
            
            text = f"Summary: {x.get('summary')}\nTime: {when}\nLocation: {x.get('location') or ''}\nAttendees: {att}\nStatus: {x.get('status')}"
            if desc:
                text += f"\nDescription: {desc}"
            
            meta = {
                "status": x.get("status"),
                "recurrence": x.get("recurrence"),
                "start": st.get("dateTime") or st.get("date"),
                "end": en.get("dateTime") or en.get("date"),
                "location": x.get("location") or "",
                "attendees": [a.get("email", "") for a in x.get("attendees", [])],
                "organizer": x.get("organizer", ""),
                "email_names": email_names,
                "has_injection": has_injection
            }
            units.append(Unit(
                id=x["id"],
                record_id=x["id"],
                source="calendar",
                time=_dt(x.get("updated") or x.get("created")),
                text=text,
                title=x.get("summary"),
                meta=meta
            ))

    # 6. Codex
    codex_dir = d / "connectors/codex/sessions"
    if codex_dir.exists():
        for f in sorted(codex_dir.glob("*.jsonl")):
            events = [json.loads(l) for l in open(f) if l.strip()]
            if not events: continue
            meta_event = events[0]
            body_events = events[1:]
            
            lines = []
            has_inj = False
            for e in body_events:
                role = e.get("role", e.get("tool", e.get("type")))
                content = e.get("content") or e.get("input") or ""
                out = str(e.get("output", ""))
                if out:
                    # truncate tool output to 500 chars
                    if len(out) > 500:
                        out = out[:500] + "... [truncated]"
                    content += "\nOutput: " + out
                
                cln, inj = _clean_text(content)
                if inj: has_inj = True
                lines.append(f"{role}: {cln}")
            
            text = "\n".join(lines)
            last_ts = body_events[-1].get("timestamp") if body_events else meta_event.get("started_at")
            
            meta = {
                "cwd": meta_event.get("cwd"),
                "repo": meta_event.get("repo"),
                "has_injection": has_inj
            }
            units.append(Unit(
                id=meta_event["id"],
                record_id=meta_event["id"],
                source="codex",
                time=_dt(last_ts),
                text=text,
                title=meta_event.get("repo"),
                meta=meta
            ))

    # 7. ChatGPT
    chatgpt_file = d / "connectors/chatgpt/conversations.json"
    if chatgpt_file.exists():
        for c in json.loads(chatgpt_file.read_text()):
            for i, m in enumerate(c.get("messages", [])):
                text, has_injection = _clean_text(m.get("content", ""))
                meta = {"has_injection": has_injection}
                units.append(Unit(
                    id=m["id"],
                    record_id=c["id"],
                    source="chatgpt",
                    time=_dt(m["create_time"]),
                    text=text,
                    title=c.get("title"),
                    speaker=m.get("role"),
                    speaker_known=True,
                    thread_key=c["id"],
                    seq=i,
                    meta=meta
                ))

    return units, deletions, edits
