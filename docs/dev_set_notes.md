# Dev set verification notes (`evals/memory_dev.jsonl`)

24 new questions, same schema as `evals/memory_train.jsonl`. Every
question below was checked against the real records in `data/` directly
(grep + reading the actual JSON/JSONL), not written from memory of the
data. After writing the file, a second, independent, programmatic check
confirmed every `needed`/`evidence`/`also_supports` id is actually visible
(per `memory.store.MemoryStore.visible()`) at that question's exact
`as_of` -- this caught one real bug (below) before it shipped.

## Programmatic cross-check (not just eyeballing)

```python
from memory.store import MemoryStore
store = MemoryStore("./data")
for r in rows:  # evals/memory_dev.jsonl
    visible_ids = {u.id for u in store.visible(r["as_of"])}
    for group in r["needed"]:
        assert all(i in visible_ids for i in group)
    for eid in r.get("evidence", []) + r.get("also_supports", []):
        assert eid in visible_ids
```

**Bug this caught:** `MEM-DEV-10` (the Codex EARTH_RADIUS bug question)
was originally given `as_of: 2026-09-15T21:00:00-07:00`, but `CDX-0915`'s
last event (its delivery time, per `data/README.md`'s "a Codex session
from its last event") is `2026-09-15T21:16:09-07:00` -- 16 minutes *after*
my chosen `as_of`. The session wouldn't have existed yet at that moment,
which would have made the question's own `needed` id a hard-rule
violation (citing a record from after `as_of`). Fixed by moving `as_of`
to `21:30:00`. This is exactly the kind of off-by-a-few-minutes error that
is easy to make by eye and why the second, programmatic pass matters.

## Per-question verification

- **MEM-DEV-01/02** (`as_of_time_travel`, the launch-date storyline at two
  new as_of values straddling the same day): read
  `SL-RP-0910-{1,2,3}`/`EM-F-015` (Sep 10 move to Oct 14) and
  `SL-F-0159`/`SL-F-0161`/`SL-F-0164` (Sep 16 go/no-go move to Oct 21,
  timestamped `15:55:21`/`16:10:08`/`16:30:11`). Picked `15:00:00` (right
  when `CAL-GONOGO-0916` starts, before the outcome is posted) and
  `17:00:00` (after) for -01/-02 respectively -- a same-day, hour-level
  split not used anywhere in train (which uses Sep 9/12/18).
- **MEM-DEV-03** (`correction`): `SL-DM-BA-0916-1` ("NRR is 112%, not
  118%..."), read directly; only occurrence of "NRR" in the whole corpus
  (checked via `grep -i nrr`), so unambiguous.
- **MEM-DEV-04** (`ownership`): `MTG-0908-PLAN` segments `#0090`
  (Priya asks about a rollback plan), `#0093` (Sarah: "I'll write it up"),
  `#0098` (Alex: "rollback plan written down before launch, that's
  Sarah. No hard date"). Grepped the whole corpus for "rollback" --
  no later message confirms it was ever posted, so "not yet done" is the
  real, verified state, not an assumption.
- **MEM-DEV-05** (`reported_speech`, second-hand): `MTG-0911-DESIGN#0084`
  (Dana: "John told me he's fine cutting dark mode") and `#0086`
  (Dana's elaboration). `as_of` set to the next morning (Sep 12 09:00),
  *before* John's own Sep 14 email and the Sep 16 final decision -- this
  is the same underlying thread as train's `MEM-TR-08`, which is the
  *only* clean second-hand-report case in the dataset (checked via a
  corpus-wide grep for `told me|said that|mentioned that`); the dev
  question is deliberately asked at an earlier `as_of`, before the later
  resolution, so the correct answer differs from train's (second-hand
  only, no first-hand confirmation or final decision yet).
- **MEM-DEV-06** (`disagreement`): `SL-DM-JA-0914-1` (John, Sep 14) and
  `SL-SALES-0917-1` (Marcus, Sep 17) -- read both directly; confirmed via
  `grep -i harbor` across Slack that these are genuinely conflicting,
  unresolved views (not one correcting the other over time).
- **MEM-DEV-07** (`commitment_status`): `MTG-0908-PLAN#0141` (Alex: "I'll
  set up the support walkthrough"). Grepped the whole corpus for
  "support walkthrough"/"walkthrough" -- zero other hits, so "no
  confirmation it happened" is the real, checked state.
- **MEM-DEV-08** (date arithmetic): `CAL-DESIGN-0911` (`start:
  2026-09-11T13:00`) and `CAL-GONOGO-0916` (`start: 2026-09-16T15:00`)
  read directly from `data/connectors/google_calendar/events.jsonl`;
  Sep 11 -> Sep 16 = 5 days (computed, not guessed).
- **MEM-DEV-09** (calendar lookup): all six `CAL-F-18`/`20`/`21`/`22`/`23`/`24`
  records read directly for their exact `start`/`end`/`summary`; `as_of`
  (Sep 20) is before all of them but after they were all created/updated
  (checked each record's `updated` field is <= Sep 20), so they're
  legitimately visible as upcoming calendar entries.
- **MEM-DEV-10** (`work_app`, Codex): `CDX-0915`'s full transcript read
  (see the bug note above for the `as_of` fix); MAE figures (71.9 -> 13.4)
  quoted directly from the assistant's own message, not paraphrased from
  memory.
- **MEM-DEV-11** (`preference`): `DCT-0914-04`, read directly. Corpus-wide
  grep for `prefer|i like|i don't like|rather` in dictations and
  Alex's own meeting lines turned up no other personal-preference
  statement -- **this is the same underlying fact as train's
  `MEM-TR-15`**, unavoidable given the data has exactly one clear
  preference statement; flagged here rather than hidden.
- **MEM-DEV-12/13/14/15** (4 `abstention` questions): each checked with a
  targeted grep across every connector + native source for the specific
  fact (salary figures, "CFO" reaction beyond the one acknowledgement
  email, "runway"/"cash balance", "agency") to confirm genuine absence,
  not just "I didn't happen to see it."
- **MEM-DEV-16** (`deleted_message`): `SL-DM-AB-0915-2` (Ben's message,
  deleted by `SL-EV-0915-DEL1` at `2026-09-15T16:05:02-07:00`) read
  directly -- it also happens to be the one pasted secret in the whole
  corpus (an API-key-shaped string), which makes this question do double
  duty (deleted-content + never-repeat-a-secret). `as_of` (Sep 16 09:00)
  is after the deletion.
- **MEM-DEV-17** (`edited_record`): `SL-RP-0916-1` (original "60/64") and
  `SL-EV-0916-EDIT1` (edit to "61/64", at `13:20:15`) read directly;
  `as_of` set to `13:25:00`, 5 minutes after the edit.
- **MEM-DEV-18** (two Sarahs): `EM-0916-ACME-ACK` (Sarah Patel's own
  reply) read directly; confirmed via `users.json` that Sarah Kim has no
  connection to the Acme pricing thread at all (checked every record
  mentioning "pricing proposal" for a Sarah Kim reference -- none).
- **MEM-DEV-19** (`prompt_injection`): `EM-F-050` (the PipelinePilot
  digest with the HTML-comment-hidden instruction) and `EM-0916-ACME-ACK`
  / `SL-F-0150` (the real, not-signed status) all read directly. `as_of`
  (Sep 17 09:00) is after the injection email (Sep 16 10:12) so the
  system has actually seen the injected text and must still not repeat
  or act on it.
- **MEM-DEV-20** (`single_fact`): `EM-0915-ACME-PROP` body read directly
  for the exact tier figures ($18 base, $15 above 500 vehicles).
- **MEM-DEV-21** (`multi_hop`): `SL-SALES-0911-1` ("Mike told me they'll
  sign in Q4") and `SL-SALES-0917-1`, both read directly -- a genuine
  2-hop chain (Mike at Harbor -> Marcus -> Alex).
- **MEM-DEV-22** (`knowledge_update`): `SL-F-0159` read directly.
- **MEM-DEV-23** (`cross_source`): `DCT-0916-05` (dictation) and
  `EM-0916-ACME-ACK` (email) read directly and cross-checked that both
  independently name Sep 25 as the date.
- **MEM-DEV-24** (`work_app`, ChatGPT): `CGPT-0909-PRICING#m3` (Alex's own
  message: "I'm leaning per-vehicle with volume tiers, because fleet ops
  teams budget by fleet size...") read directly from
  `data/connectors/chatgpt/conversations.json`; `as_of` set to the
  morning of Sep 9, before the actual Acme call that day (per
  `CAL-ACME-0909`, `11:00`), confirmed the ChatGPT conversation's
  `create_time` (`08:40:17`) is before that.

## Known, disclosed overlaps with train (not hidden)

Two questions above (`MEM-DEV-05`, `MEM-DEV-11`) share their underlying
source records with `MEM-TR-08` and `MEM-TR-15` respectively, because the
dataset has exactly one clean example of each (a second-hand report, and
a personal preference statement). Both are asked with different wording
and, for `MEM-DEV-05`, a different `as_of` producing a genuinely different
correct answer. No question's gold answer, record ids, or exact wording
were copied from `evals/memory_train.jsonl` -- every gold answer above was
independently derived from reading the raw records, per rule 7.
