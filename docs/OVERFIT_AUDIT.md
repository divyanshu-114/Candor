# Overfit audit of the v2 rules (2026-10-10)

Method: list every rule / regex / list / threshold added in v2 (plus v1 prompt examples), state the general behaviour it implements, delete or
generalise anything whose only justification was one eval item, and add automatic checks (`tests/test_no_leakage.py`): (1) ids, question text,
gold text and needed ids of every eval file may not appear in code; (2) every person / organisation / product name taken from the data may not
appear in code strings or identifiers (comments and docstrings excluded); (3) six-word runs from any question or command may not appear in code.

## Removed or generalised
| Where | What it was | Why it was a problem | Now |
|---|---|---|---|
| memory/ingest.py, memory/people.py | the home email domain was written into the code ("first.last@<our domain>", `startswith("<our org>")`) | data-specific constant (v1 and v2) | emails come from Slack's `users.json`; the own domain is derived from them |
| actions/resolve.py `_FACT_NOUN` | listed `nrr`, `arr`, `mrr` | NRR came from a train command | a modifier ("corrected", "latest") before ANY all-caps token, or a generic fact noun (price, date, total ...) |
| memory/ledger.py | action verbs `walk`, `walkthrough`; cancel cue `pushed the demo`; deadline `before (launch\|the board ...)` | each was added after one dev item | removed / generalised to "pushed (it\|the X) to", "before the launch/release/deadline/meeting/call/event" |
| prompts: memory/query.py, retrieve.py, answer.py, actions/planner.py | examples that were train questions or data names ("the day I fly", "flight to Denver", "Dana said that John ...", "Priya", "at Acme", "the day of the offsite") | train text in prompts (rule 7) | invented examples ("the day I travel", "flight to Paris", "Pat / Lee", "Casey", "at <Company>") |
Note: changing the v1 prompt examples means `PIPELINE=v1` is now "v1 behaviour with sanitised prompts"; its saved LLM replies no longer replay.

## Kept, with the general behaviour and where it fires on unseen phrasings
| Rule | General behaviour | Unseen phrasing it covers |
|---|---|---|
| lanes + RRF (retrieve_v2) | each source gets its own ranking so one chatty source cannot crowd out the rest | any question |
| cross-encoder cut, CE_WEIGHT=2 | rank-fuse a local relevance model with the first-stage rank | any question; weight picked from {0.5,1,2,4} on train+v2_dev (noisy, documented) |
| `anchor.parse_relation` | "after / before / the day before / the day of <event>" -> anchor, then window or target date; clause anchors (gerund, pronoun) and "how many days after A did B" excluded | "what happened right after the kickoff call", "the day before the conference" |
| `anchor.question_dates`, agenda | an explicit date in the question -> that day's calendar | "what's on the 14th", "on Oct 3" |
| chains `CHANGE_CUES` | records with change vocabulary (moved, actually, correction, instead, cancelled ...) that share rare words with the question | any changed-fact question |
| neighbours | adjacent transcript turns of the best hits (question/answer pairs) | any meeting question |
| ledger regexes | first-person promises with a task verb, "can you ..." requests, deadline phrases; later records linked only if they share rare words (<=15% df) and carry a done/extend/cancel cue | "what do I owe", "what did I promise X" |
| people resolution (MIN_EVIDENCE 2.5, WIN_RATIO 2, STRONG 0.85) | discriminative word-rate fit of the question against each same-first-name candidate's records; ties stay ambiguous. The three numbers were set a priori and nudged once after seeing a "What did Sarah say?" case: treat as weakly tuned | any shared first name |
| writer v2 prompt | versions / partial answers / arithmetic in code | any answer |
| quotes.py (0.85, exact digits and names) | soft match on content words in order | any quote |
| actions: `timeparse` | day words, weekdays, "the 25th", times with/without am/pm, working-hours convention (bare 1-6 = pm), durations | any command |
| actions: person cues (slack / organisation / channel / recent interaction within 3 h and the other >24 h quiet) | narrow a shared first name before asking | any shared first name; 3 h / 24 h are a priori |
| actions: event match (all words of the phrase in the title, upcoming first, series -> next instance) | resolve "the all-hands", "my 1:1 with X" | any calendar |
| actions: `memory_lookups` safety net | "the corrected/latest <metric>" and "what we decided about X" -> a memory question | any described fact |
| actions: default meeting length 30 min | a required arg with a conventional default | any "set up a call" without a length |
| `injection_clarify` | name the action category from a fixed list, never repeat the text | any override attempt |
| evidence gates (off) | co-occurrence / cross-encoder gates; thresholds tuned by looking at train+dev (documented as not helpful) | off by default |

## How did the no-key actions score jump from 4/12, 13/30, 2/15 to 12/12, 30/30, 14/15?
The v1 no-key planner never attempted a send, booking or move (it only recognised open / delete / question / injection and otherwise asked a generic
question), so the baseline was deliberately crippled, not a fair "rules" planner. The v2 resolver is a deterministic parser written AFTER reading the 57
commands in these three files, and the commands are templated ("Tell X ...", "Remind me ... at ...", "Move X to ...", "Email X ..."). The number is therefore
a development-set number and is inflated by that. After the audit above it is 55/57 (96.5%). The honest estimate is the fresh `actions_holdout2.jsonl`.
