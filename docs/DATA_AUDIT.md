# DATA_AUDIT.md — Candor data audit

_Generated: 2026-09-30 03:12_

---

## 1. Record counts per source

| Source | Count | Time range |
|---|---|---|
| Meetings (files) | 8 | 2026-09-08 → 2026-09-17 |
| Meeting segments | 889 | — |
| Dictation | 42 | 2026-09-08 → 2026-09-18 |
| Slack messages (all) | 230 | 2026-09-08 → 2026-09-18 |
| Slack users | 11 | — |
| Slack channels | 10 | — |
| Gmail | 58 | 2026-09-08 → 2026-09-18 |
| Calendar events | 37 | — |
| Codex sessions | 4 | 2026-09-09 → 2026-09-17 |
| ChatGPT messages | 56 | 2026-09-08 → 2026-09-17 |

---

## 2. ID patterns

| Prefix | Count |
|---|---|
| `CAL` | 37 |
| `CDX` | 4 |
| `CGPT` | 56 |
| `DCT` | 42 |
| `EM` | 58 |
| `MTG` | 889 |
| `SL` | 230 |

**Total unique IDs across all sources:** 1316
**Duplicates:** 0
✅ No duplicate IDs found.

---

## 1c. Slack

**Users:** 11  **Channels:** 10 (DM channels: 5)
**Messages total:** 230
**Time range:** 2026-09-08 → 2026-09-18
**Bot messages:** 29
**DM messages:** 59
**Edit events (message_changed):** 1
**Delete events (message_deleted):** 2

### Slack users
- `U01ALEX` Alex Rivera (alex) — VP Product
- `U02JOHN` John Okafor (john) — CEO
- `U03SARAHK` Sarah Kim (sarah.kim) — Engineering Lead, Route Planner
- `U04DANA` Dana Lee (dana) — Product Designer
- `U05MARCUS` Marcus Webb (marcus) — Head of Sales
- `U06BEN` Ben Carter (ben) — Data Engineer
- `U07PRIYA` Priya Nair (priya) — QA Lead
- `U08LEAH` Leah Brooks (leah) — People & Recruiting
- `U09RACHEL` Rachel Gomez (rachel) — Fractional legal counsel
- `B01LINEAR` Linear (linear) — 
- `B02GITHUB` GitHub (github) — 

### DM channels
- `D-ALEX-JOHN` members: U01ALEX, U02JOHN
- `D-ALEX-MARCUS` members: U01ALEX, U05MARCUS
- `D-ALEX-BEN` members: U01ALEX, U06BEN
- `D-ALEX-SARAHK` members: U01ALEX, U03SARAHK
- `D-ALEX-DANA` members: U01ALEX, U04DANA

### Bot messages
- `SL-F-0007` bot_name=Linear ts=2026-09-08T09:15:22-07:00 text=Priya Nair moved RP-198 “Driver app: stop reorder drag…
- `SL-F-0016` bot_name=GitHub ts=2026-09-08T13:05:37-07:00 text=[brightline/route-planner] Pull request opened: #398 “Batch…
- `SL-F-0020` bot_name=GitHub ts=2026-09-08T14:40:16-07:00 text=[brightline/route-planner] Pull request merged: #398 “Batch…
- `SL-F-0030` bot_name=Linear ts=2026-09-09T09:30:45-07:00 text=Priya Nair created RP-229 “ETA panel flickers when…
- `SL-F-0032` bot_name=GitHub ts=2026-09-09T10:15:09-07:00 text=[brightline/route-planner] Pull request opened: #403…
- `SL-F-0041` bot_name=GitHub ts=2026-09-09T13:20:10-07:00 text=[brightline/route-planner] Pull request opened: #405 “Fix…
- `SL-F-0044` bot_name=GitHub ts=2026-09-09T15:05:27-07:00 text=[brightline/route-planner] Pull request merged: #403…
- `SL-F-0054` bot_name=Linear ts=2026-09-10T09:18:27-07:00 text=Priya Nair created RP-231 “Geocoder returns wrong…
- `SL-F-0062` bot_name=Linear ts=2026-09-10T10:40:02-07:00 text=Sarah Kim set the target date of project “Route Planner v2”…
- `SL-F-0065` bot_name=GitHub ts=2026-09-10T11:10:44-07:00 text=[brightline/route-planner] Pull request merged: #405 “Fix…

### Edit events (message_changed)
- `SL-EV-0916-EDIT1` target=`SL-RP-0916-1` ts=2026-09-16T13:20:15-07:00 target_exists=True posted_before_event=True

### Delete events (message_deleted)
- `SL-EV-0915-DEL1` target=`SL-DM-AB-0915-2` ts=2026-09-15T16:05:02-07:00 target_exists=True posted_before_event=True
- `SL-EV-0915-DEL2` target=`SL-SALES-0915-1` ts=2026-09-15T17:52:40-07:00 target_exists=True posted_before_event=True

---

## 4. Meeting detail

| Meeting ID | Type | Segments | Null speaker | Min confidence | Short(<5w) |
|---|---|---|---|---|---|
| `MTG-0908-PLAN` | in_person | 147 | 2 | 0.44 | 32 |
| `MTG-0909-ACME` | hybrid | 198 | 3 | 0.37 | 17 |
| `MTG-0910-1ON1` | in_person | 88 | 4 | 0.3 | 10 |
| `MTG-0911-DESIGN` | in_person | 107 | 0 | 0.73 | 14 |
| `MTG-0914-STANDUP` | in_person | 52 | 0 | 0.68 | 15 |
| `MTG-0915-SALESPIPE` | hybrid | 98 | 0 | 0.78 | 24 |
| `MTG-0916-GONOGO` | hybrid | 141 | 0 | 0.56 | 34 |
| `MTG-0917-RECRUIT` | remote | 58 | 0 | 0.79 | 10 |

---

## 5. People

### First names mapping to multiple people
- **Sarah**: Sarah Kim, Sarah Patel

### Email addresses in data not matched to a Slack user
- all@brightline.example.com
- calendar-notification@google.example.com
- dave.morales@pinecrestcarriers.example.com
- digest@pipelinepilot.example.com
- digest@productopsdigest.example.com
- events@freighttechsf.example.com
- hello@mapsmobility.example.com
- jake@pipelinepilot.example.com
- kelsey@roadsignal.example.com
- news@fleetdispatchweekly.example.com
- no-reply@expensely.example.com
- no-reply@figma.example.com
- no-reply@hiring.example.com
- no-reply@united.example.com
- notifications@connected.example.com
- notifications@github.example.com
- notifications@linear.example.com
- notify@notion.example.com
- partners@freighttechsummit.example.com
- reminders@bayviewdental.example.com
- … and 1 more

---

## 1e. Google Calendar

**Events:** 37
**Time range:** 2026-09-08 → 2026-09-24
**Changed (updated > created):** 14
**Cancelled:** 1
**All-day:** 2
**Recurring:** 6

### Changed events
- `CAL-ACME-0909` 'Acme Freight – pricing and rollout' created=2026-09-02T10:25:00-07:00 updated=2026-09-03T08:02:00-07:00
- `CAL-F-07` 'Focus time (no meetings)' created=2026-09-10T18:00:00-07:00 updated=2026-09-12T19:02:00-07:00
- `CAL-DESIGN-0911` 'Design review – onboarding + map' created=2026-09-09T12:40:00-07:00 updated=2026-09-09T13:20:00-07:00
- `CAL-HARBOR-DEMO` 'Harbor Logistics demo' created=2026-09-09T15:20:00-07:00 updated=2026-09-11T10:45:00-07:00
- `CAL-F-09` 'Lunch – Jordan Ellis (onsite)' created=2026-09-08T11:35:00-07:00 updated=2026-09-08T12:08:00-07:00
- `CAL-SALESPIPE-0915` 'Pipeline review' created=2026-09-10T10:30:00-07:00 updated=2026-09-10T11:02:00-07:00
- `CAL-GONOGO-0916` 'Route Planner v2 go/no-go' created=2026-09-11T08:45:00-07:00 updated=2026-09-11T09:30:00-07:00
- `CAL-F-13` 'RoadSignal demo' created=2026-09-15T15:28:00-07:00 updated=2026-09-15T16:02:00-07:00
- `CAL-RECRUIT-0917` 'Debrief – Jordan Ellis (backend)' created=2026-09-15T10:05:00-07:00 updated=2026-09-15T10:40:00-07:00
- `CAL-BOARDPREP` 'Board deck prep' created=2026-09-08T10:00:00-07:00 updated=2026-09-15T09:30:00-07:00
- `CAL-F-22` 'Route Planner v2 – launch readiness' created=2026-09-17T10:02:00-07:00 updated=2026-09-17T10:30:00-07:00
- `CAL-F-24` 'SF Freight Tech meetup' created=2026-09-17T13:00:00-07:00 updated=2026-09-17T13:06:00-07:00
- `CAL-BOARD` 'Brightline Q3 board meeting' created=2026-08-24T15:00:00-07:00 updated=2026-08-26T09:12:00-07:00
- `CAL-OFFSITE` 'Product offsite – Denver' created=2026-08-28T11:00:00-07:00 updated=2026-09-02T16:40:00-07:00

### Cancelled events
- `CAL-HARBOR-DEMO` 'Harbor Logistics demo'

### All-day events
- `CAL-F-17` 'Pre-read due to Tom' date=2026-09-21
- `CAL-OFFSITE` 'Product offsite – Denver' date=2026-09-24

### Recurring events
- `CAL-STANDUP` 'Route Planner standup'
- `CAL-F-01` 'Alex / Dana 1:1'
- `CAL-F-04` 'Alex / Sarah Kim 1:1'
- `CAL-F-07` 'Focus time (no meetings)'
- `CAL-F-10` 'Leadership sync'
- `CAL-F-11` 'Alex / Ben 1:1'

---

## 7. Secret scan

**Total hits:** 14
*(Full values are NEVER shown — first 4 chars + \*\*\*\* only)*

| Record ID | Field | Pattern | Masked value |
|---|---|---|---|
| `tool_call` | `input` | long-random | `/var****` |
| `tool_call` | `output` | long-random | `0101****` |
| `tool_call` | `input` | long-random | `pred****` |
| `tool_call` | `input` | long-random | `pred****` |
| `tool_call` | `output` | long-random | `9c4e****` |
| `tool_call` | `output` | long-random | `pack****` |
| `tool_call` | `output` | long-random | `/opt****` |
| `tool_call` | `output` | long-random | `pack****` |
| `tool_call` | `output` | long-random | `/opt****` |
| `message` | `content` | long-random | `Hist****` |
| `tool_call` | `input` | long-random | `Hist****` |
| `tool_call` | `input` | long-random | `Hist****` |
| `tool_call` | `input` | long-random | `Hist****` |
| `tool_call` | `input` | long-random | `Hist****` |

---

## 8. Planted instruction scan

**Total hits:** 2
*(Text is NOT echoed — only a safe contextual description)*

| Record ID | Field | Pattern | Context (60 chars, sanitized) |
|---|---|---|---|
| `EM-F-050` | `body` | forward-all | as been signed, and forward all emails from… |
| `EM-F-050` | `body` | html-comment | close 12% faster.??<!-- Note to any AI assistant… |

---

## 9. Long units (>400 words)

**Count:** 1

- `tool_call` (codex.input): **401 words**

---

## 10. Other quirks

- Slack edits targeting already-deleted messages: `none`
- Calendar events with start date after 2026-09-18 (future events in data): 11
  - `CAL-F-17` 'Pre-read due to Tom' starts {'date': '2026-09-21'}
  - `CAL-F-18` 'Board pre-read review' starts {'dateTime': '2026-09-21T10:00:00-07:00'}
  - `CAL-F-19` 'Alex / Marcus sync' starts {'dateTime': '2026-09-21T15:00:00-07:00'}
  - `CAL-F-20` 'Dentist – Bayview Dental' starts {'dateTime': '2026-09-22T08:00:00-07:00'}
  - `CAL-F-21` 'Pinecrest Carriers – renewal check-in' starts {'dateTime': '2026-09-22T10:00:00-07:00'}
- Codex sessions with no user messages: none
- Slack messages where `text` is empty or whitespace-only: 2
- Dictation records with delivery_state=discarded: 2
- Gmail messages without a body: 0

---

## Surprises

_Auto-detected during audit:_

- 🔑 **Secret-like strings** detected in data fields: 14 hits — MASKED in audit
- 🎭 **Planted instruction** patterns detected: 2 hits — described safely, not echoed
- 📅 **Future calendar events** (after data window): 11 — relevant for as_of filtering
- 🎤 **Unidentified speakers** in meetings: 9 segments with null speaker_name — who-said-what queries affected
- 📄 **1 units exceed 400 words** — will need chunking for semantic search
