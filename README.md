# Candor memory

Candor reads two weeks of one person's work life (Slack, email, calendar, meetings, dictation, Codex sessions, ChatGPT chats) and answers questions about it. Every answer names its source records, uses only what was known at the moment of the question, and says "I don't know" when memory has no answer. A second, smaller program turns a command such as "Move board deck prep to 3pm" into a checked, dry-run action plan.

## The 60-second version

```bash
git clone <REPO_URL> candor && cd candor      # TODO(user): replace <REPO_URL> with your public GitHub URL
cp .env.example .env
./run.sh memory evals/memory_train.jsonl outputs/answers.jsonl
```

It works with no API key (skip the `cp`, or delete the two placeholder key lines in `.env`); a key only improves answer quality.

## What you need

| Need | Detail |
|---|---|
| Python | 3.10 or newer. Everything here was run on Python 3.14 on macOS; older versions were not tested. |
| System | macOS or Linux, with `git`. |
| Disk | About 0.3 GB: 212 MB environment, 64 MB embedding model, 4 MB search index. |
| Time | **133 seconds** for the first run in a fresh clone (setup, install, model download, 27 questions); later runs are much faster. |
| Internet | Only for the first-run downloads (packages and the small local embedding model `BAAI/bge-small-en-v1.5`) and for model calls if you add a key. |

## Run it

**a) Memory questions.** One JSON object per line in, one per line out.

```json
{"id": "MEM-TR-01", "question": "When is Route Planner v2 launching?", "as_of": "2026-09-18T18:00:00-07:00"}
```
```json
{"id": "MEM-TR-01", "answer": "October 21, 2026. ...", "sources": ["MTG-0916-GONOGO#0077"], "retrieved": ["MTG-0916-GONOGO#0077", "SL-F-0164", "SL-RP-0910-1"], "abstained": false}
```

`retrieved` is the ranked list of record ids looked at (best first, up to 20): the main score. `sources` are the ids the answer relies on. `abstained: true` means "not in memory".

**Try it yourself.** One question asked as of two moments gives different answers (no key needed; output shortened):

```bash
cat > my_questions.jsonl <<'EOF'
{"id": "early", "question": "When is board deck prep?", "as_of": "2026-09-10T12:00:00-07:00"}
{"id": "late",  "question": "When is board deck prep?", "as_of": "2026-09-18T18:00:00-07:00"}
EOF
./run.sh memory my_questions.jsonl my_answers.jsonl
```
```text
early -> Board deck prep When Thursday Sep 17, 2026 2pm - 3pm (Pacific Time - Los Angeles) Where Brightline HQ ...   sources: ['EM-F-003']
late  -> Moved board deck prep to Fri 10am, Thursday got messy.                                                sources: ['SL-F-0128']
```

On Sep 10 only the calendar invite exists; by Sep 18 a Slack message has moved the meeting.

**b) Actions (dry run).** Nothing is ever sent; you get the plan.

```json
{"id": "ACT-TR-11", "command": "Delete all my emails from Marcus", "as_of": "2026-09-18T09:00:00-07:00"}
```
```json
{"id": "ACT-TR-11", "actions": [{"type": "confirm", "args": {"summary": "This would change or remove data: Delete all my emails from Marcus. Do you want me to go ahead?"}}]}
```

Run with `./run.sh actions commands.jsonl plans.jsonl`. Deletions become a confirmation request.

**c) Interactive assistant (dry run).** Use the virtual environment's Python:

```bash
./.venv/bin/python -m actions.repl --as-of 2026-09-18T09:00:00-07:00
> Open Figma
1. app.open
     app: Figma
run it? (y/n) n
```

**d) Useful flags**

| Flag or variable | What it does |
|---|---|
| `--resume` | Skip ids already in the output file. Output is written one line at a time in input order, so a killed run leaves a valid file. |
| `--ids A,B` | Recompute only those ids; the rest of the output file is untouched. |
| `--retrieval-only` | Memory only: write the ranked `retrieved` ids and skip the answer. |
| `MODE=baseline\|no_rerank\|full` | How much model help is used (default `full`). |
| `WORKERS=N` | Parallel workers (default 4). |
| `DATA_DIR=./data` | Where the data lives. |
| `LLM_FROZEN_ROLES=strong,fast` | Never send a model request for those roles; saved replies are still used. Costs nothing. |

## Check it works

```bash
./.venv/bin/python -m pytest -q          # 363 tests, about 11 seconds, no network, no key
python3 eval_harness/score_retrieval.py --gold evals/memory_train.jsonl --answers outputs/answers.jsonl
python3 eval_harness/score_memory.py --gold evals/memory_train.jsonl --answers outputs/answers.jsonl --judge none
python3 eval_harness/score_actions.py --gold evals/actions_train.jsonl --predictions outputs/plans.jsonl
./run.sh score-memory outputs/answers.jsonl
./scripts/release_check.sh
```

(Make the plans file first: `./run.sh actions evals/actions_train.jsonl outputs/plans.jsonl`.) With no key you should see retrieval **64.0%** (95% CI 44-100%) and strict answers **25.9%** (10-50%), and `./run.sh score-memory` prints the same two scores. The actions scorer should show **33.3%** (arguments 50.0%) on the training commands and 43.3% (62.2%) on `evals/actions_dev.jsonl`. The scorers write `results_*.json` files here; delete them before the release check (it needs a clean tree). `release_check.sh` clones the repo fresh, runs the tests and a key-less run, scans code and history for secrets, and checks that no hidden record is ever returned. It took 210 seconds and ends with `RELEASE CHECK: PASSED`. Run it with no API keys exported.

## API keys (optional)

| | No key | With a key |
|---|---|---|
| Search | Keyword + local embeddings | Same, plus a larger model re-ranks candidates |
| Understanding | Heuristics | A small model reads dates and sub-questions |
| Answers | Best matching sentence; refuses if most question words are absent from memory | A model writes it; each claim needs a verified quote |
| Actions | Rules only | Rules first, then one or two model calls |

Add a key in `.env`: `GROQ_API_KEY=...` (free at console.groq.com/keys). Any OpenAI-compatible service also works: set `CUSTOM_API_KEY`, `CUSTOM_BASE_URL`, `CUSTOM_MODEL_STRONG`, `CUSTOM_MODEL_FAST` (or the plain `OPENAI_*` names) and add `custom` or `openai` to `LLM_PROVIDERS`; `.env.example` explains each. If a model hits its daily limit or a key is rejected, calls move to the next model, then the next provider.

Free tiers have small daily token limits: wait or add a provider, then `--resume`. Every model reply is saved in `.cache/llm`, so repeating a run is free and byte-identical.

## Results

Official scorers; intervals where the scorer prints one. All dates are 2026; "20b"/"120b" are Groq `gpt-oss-20b`/`gpt-oss-120b`. Train questions were visible while building, so dev rows are the fairer guide.

| What | Mode | Score | 95% CI | n | Models | Date |
|---|---|---|---|---|---|---|
| Retrieval, train | replayed from cache | **80.0%** | 68-94% | 27 | gpt-oss-20b + 120b | 10-02/03 |
| Answers, train, rule scorer | live | **74.1%** strict | 60-88% | 27 | writer 20b | 10-03 |
| Answers, train, model judge | live | **66.7%** strict | 54-82% | 27 | judge 20b | 10-03 |
| Actions, train | live | **100.0%** (12 of 12) | not printed | 12 | planner 20b | 10-03 |
| Actions, dev | live | **90.0%** (27 of 30) | not printed | 30 | planner **120b** | 10-03 |
| Dev memory (12 of 24): retrieval | live | **60.0%** | 30-90% | 12 | 20b + 120b | 10-03 |
| Dev memory (12 of 24): answers | live | **58.3%** strict | 33-83% | 12 | writer 120b | 10-03 |
| Retrieval, train / dev | no key | 64.0% / 57.9% | 44-100% / 35-83% | 27 / 24 | none | 10-03 |
| Answers strict, train / dev | no key | 25.9% / 20.8% | 10-50% / 4-39% | 27 / 24 | none | 10-03 |
| Actions, train / dev | no key (rules) | 25.0% / 43.3% | not printed | 12 / 30 | none | 10-03 |

*Note on the last row:* the 25.0% was measured one change earlier. On the final code in a fresh clone it is **33.3%** (arguments 50.0%), because the same-name rule now answers one more training command. Dev (43.3%) did not change.

**Not measured** (free-tier daily token limits): the other 12 dev questions, a model-judged dev score, and dev actions with the default small planner.

Reading it: train actions is partly tuned (a rule fixed one command after it failed; the first live pass was 11 of 12). Dev memory is 12 questions, so intervals are wide, and its writer was the larger model. The judge is a small model, stricter than the rule scorer. Live answers can vary slightly between runs.

## How it works

```
 data/ -> load: hide secrets, remove planted instructions, give every record a stable id
       -> time filter: drop anything not yet delivered, or already deleted, at "as_of"
       -> search: keyword (BM25) + local embeddings, merged into one ranking
       -> small model: understand the question; larger model: reorder the candidates
       -> writer: answer plus a short quote for each claim
       -> code check: each quote must appear in the evidence, else abstain
       -> {"id", "answer", "sources", "retrieved", "abstained"}
```

- **Specific ids and delivery times** for every segment, message, email, dictation, event and chat message.
- **Visibility is enforced in code.** Records from after `as_of` and deleted Slack messages are removed before any search or model call.
- **Secrets and planted instructions** are masked or removed at load time, so they never reach a prompt or an answer.
- **Hybrid search**: keyword plus a local embedding model, merged by rank.
- **The model only reorders** a fixed candidate list; it cannot add ids.
- **Abstain unless a verified quote supports the answer**, checked against the text the writer saw.
- **Fallbacks.** If a model call fails or there is no key, the plain search ranking is used, so a failure never makes results worse than the baseline.

**Time and `as_of`.** Each record's delivery time follows the data's rule for its source (a Codex session is delivered at its last event). `as_of` is the only gate, so "now" and "last Tuesday" use the same code. Edits replace text from when they happened; deletions remove the message entirely.

**Who said what.** Speakers come from the data. When a record reports what someone else said ("Dana said John told her..."), the writer must give who reported it, what the person said themselves, and any final decision. Unidentified speakers are never guessed. **What failed:** one training question (MEM-TR-08, did John agree to cut dark mode) still loses the second-hand claim: the writer reports only John's own words.

**The two Sarahs.** Sarah Kim is internal (Slack); Sarah Patel is external (Acme Freight). The code finds this collision in the data and, for commands, asks "Which Sarah?" unless the command says "on Slack", "email", Acme, or a surname. **Verified:** that rule and its tests. **Measured:** "Message Sarah about the pricing proposal" is now a clarifying question (before, the planner guessed Sarah Kim). **Failed:** a dev question ("Has Sarah replied about the pricing proposal yet?") was answered "Yes" without noticing two Sarahs.

**Abstention.** "I don't know" whenever no verified quote supports the answer. Both "not in memory" training questions abstain, and a bare "there are no events" is rejected unless a record says so.

## Known limits and what didn't work

- **Search misses cause most wrong answers**: 3 of the 5 wrong dev answers and 4 of the 7 non-correct training answers had the needed record outside the top ten.
- **"The day I fly to Denver"** (find a date, then look it up) still fails (MEM-TR-25). A date-lookup stage exists but showed no gain under limited model access, so it is off.
- **Quotes must be exact.** A correct answer (MEM-TR-14) became "I don't know" because its quote dropped one word.
- **Free-tier limits** (8,000 tokens a minute, daily caps) forced the partial measurement above.
- **Invented recipient.** On one dev command the planner wrote an email address that exists nowhere. A new check rejects unknown addresses (a safe question instead), but the score did not rise.

## Repo map

```text
memory/        the memory system
  store.py       time-safe records (as_of, edits, deletes)
  retrieve.py    search + reranking
  answer.py      writer, quote check, abstention
  llm.py         every model call: cache, retries, fallbacks
  safety.py      secret masking, planted-instruction removal
actions/       the action planner (planner.py, rules.py)
eval_harness/  the official scorers (unchanged)
evals/ examples/ data/   questions and commands, format examples, the records
scripts/       release_check.sh and helpers
tests/         363 tests
docs/          DEVLOG.md: raw development log, for the curious
run.sh         one-command entry point
BRIEF.md, PROJECT_RULES.md   the task brief and working guidelines
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` or the virtual environment fails | Run `python3 -m venv .venv` then `python3 -m ensurepip --upgrade`, or install `uv` (`run.sh` uses it if present). |
| Embedding model cannot download (offline) | `run.sh` warns "answering BM25-only" and continues with keyword search. |
| 429 or "daily quota" messages | Wait, add a second provider, or rerun with `--resume`. |
| Many "I don't know" answers, `Rerank failed` lines | Expected with no key; they are warnings. |
| Slow first run | Packages, model and search index are built once (133 s here). |
| Odd model errors | A stale key may be exported: `env \| grep API_KEY`, then unset it. |
| `release_check.sh` says the tree is dirty | Delete `results_*.json` and other generated files. |

## Tools, models and cost

- **Building:** Antigravity (models: TODO(user): list the models shown in Antigravity); later quota, caching and planner work and this README with Claude Code.
- **Runtime models:** Groq `openai/gpt-oss-120b` (ranking; also the writer in the dev check) and `openai/gpt-oss-20b` (understanding, writer, planner, judge). The Gemini key was rejected for generation and not used.
- **Embeddings:** `BAAI/bge-small-en-v1.5` via fastembed, running locally.
- **Spend:** 0rs (only free tiers were used).
- **Demo video:** not included

## Development history

Older measurements, not claims about the current code.

| When | Commit | Model | Number |
|---|---|---|---|
| 2026-09-30 | `63f1f11` | llama-3.3-70b-versatile | retrieval 92.0% (train, 45 candidates) |
| 2026-10-01 | `98486bf` | gpt-oss-120b | retrieval 76.0%; answers 63.0% strict (rules) and 59.3% (judged) |
| 2026-10-01/02 | `4a48683` | gpt-oss-120b | retrieval 84.0% (retrieval-only) |
