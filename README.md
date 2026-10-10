# Candor memory

Candor reads two weeks of one person's work life (Slack, email, calendar, meetings, dictation, Codex sessions, ChatGPT chats) and answers questions about it. Every answer names its source records, uses only what was known at the moment of the question, and says "I don't know" when memory has no answer. A second program turns a command such as "Move board deck prep to 3pm" into a checked, dry-run action plan.

## Run it (60 seconds, no key needed)

```bash
git clone https://github.com/divyanshu-114/Candor candor && cd candor
./run.sh memory evals/memory_train.jsonl outputs/answers.jsonl
./run.sh actions evals/actions_train.jsonl outputs/plans.jsonl
```

**Try it yourself: the same question at two times (no key).** Memory only uses what was known at `as_of`:

```bash
cat > /tmp/two.jsonl <<'EOF'
{"id": "A", "question": "When is Route Planner v2 launching?", "as_of": "2026-09-09T09:00:00-07:00"}
{"id": "B", "question": "When is Route Planner v2 launching?", "as_of": "2026-09-18T18:00:00-07:00"}
EOF
./run.sh memory /tmp/two.jsonl /tmp/two_out.jsonl && cat /tmp/two_out.jsonl
```
Real output of a no-key run in a fresh clone (shortened; the `retrieved` lists are cut):

```text
{"id": "A", "answer": "Launch is September 30 for Route Planner v2.", "sources": ["MTG-0908-PLAN#0140"], "retrieved": ["MTG-0908-PLAN#0140", "DCT-F-003", ...]}
{"id": "B", "answer": "FYI for anyone talking to customers: Route Planner v2 launch is now Oct 14 (we're fixing a geocoding issue with Canada/Mexico addresses).", "sources": ["SL-F-0063"], "retrieved": ["SL-F-0063", "DCT-F-003", ...]}
```

The answer changes with the time, as it should. It is also an honest weakness: by the 18th the launch had moved again (to Oct 21) and the no-key answer still says Oct 14; with a model this question is answered correctly.

The first run builds a virtual environment and downloads two small local models (search embeddings and a relevance re-ranker). On a fresh clone here that took **6.5 minutes** and **0.73 GB** (224 MB packages + 504 MB models). Later runs start in seconds.

**Which AI service?** None is required. If you add one key to `.env` the program finds it by itself and picks a strong and a small model: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, or your own `CUSTOM_*` (see `.env.example`). Several keys are used in that order, each taking over when the one before runs out of credit or quota. With no key a banner says so and you still get search, extractive answers and rule-based actions. Every run ends with a banner listing each stage that fell back; `--strict` makes the run exit non-zero if anything fell back.

## What changed in v2 (all numbers from the final commit)

Two scores matter. **Retrieval** (main score): the right records are in the top 10 and nothing from the future or deleted is. **Answers**: the official rule scorer, strict. n is small (19-35 scored questions per set), so one question is worth 3-5 points.

| Set | What it is | Used for tuning? |
|---|---|---|
| train (27), dev (24) | the given and earlier questions | yes |
| v2_dev (40) | new, written for v2 | yes |
| v2_holdout (25) | new; only its totals were ever looked at, at the end of each phase; it decided whether the new answer writer stays on | partly (one decision) |
| **v2_holdout2 (25) and actions_holdout2 (20)** | written after the audit, then frozen; each run once at the end | **no** |

**What "v1" means in the tables.** "v1" is the **v1 pipeline** (`PIPELINE=v1`: original components: bge-small embeddings, no cross-encoder, original masking), run on today's code with the same OpenRouter models. The original first submission is commit `4c41059` (tag `v1`); On the fresh set with no key it scores **68.4% retrieval and 12.0% answers**, the same as `PIPELINE=v1` today (22 of 25 retrieved lists identical, all answers and sources identical). Machine-readable scorer output for every table row is in `docs/results/`.

**Retrieval, main score**

| Set | v1 pipeline, no key | v2, no key | v2, with a model |
|---|---|---|---|
| train | 64.0% | 92.0% | 96.0% |
| dev (24) | 57.9% | 78.9% | not run |
| v2_dev | 68.6% | 82.9% | 88.6% |
| v2_holdout | 75.0% | 75.0% | 80.0% |
| **v2_holdout2 (fresh)** | **68.4%** | **84.2%** | **78.9%** |

v1 pipeline with a model (Groq, measured live on 10-02/03): train 80.0%, v2_dev 82.9%. **Like for like on the fresh set** (`PIPELINE=v1` with the same OpenRouter models): v2_holdout2 retrieval **73.7%**, answers 60.0%, 0 of 6 false answers, versus v2 78.9% / 68.0% / 0 of 6.

**The 84.2% (no key) to 78.9% (with a model) step on the fresh set.** The saved diagnostics of the with-model run show **0 of 25 questions with any degraded stage**, so this is model behaviour or run-to-run variation on 19 scored questions (one question is 5 points), not failed calls.

**Answers, strict rule score (with a model = OpenRouter, gpt-oss-120b ranks, gpt-oss-20b writes)**

| Set | v1 pipeline, no key | v2, no key | v2, with a model | False answers on unanswerable questions, with a model |
|---|---|---|---|---|
| train | 25.9% | 40.7% | 77.8% (85.2% on a cold-cache re-run) | 0 of 2 |
| v2_dev | 20.0% | 37.5% | 67.5% | 0 of 5 |
| v2_holdout | 20.0% | 40.0% | 56.0% | 2 of 6 |
| **v2_holdout2 (fresh)** | 12.0% | 8.0% | **68.0%** | **0 of 6** |

v1 pipeline with a model: train 74.1% (Groq, 10-03). Run-to-run noise with a model is about +/-2 questions (the same set gave 77.8% and 85.2% on train).

**Actions (dry-run plans)**

| Set | v1 pipeline, no key | v2, no key | v1 pipeline, with a model | v2, with a model |
|---|---|---|---|---|
| train (12) | 4/12 | 12/12 | 12/12 | 11/12 |
| dev (30) | 13/30 | 30/30 | 27/30 | 29/30 |
| new dev (15) | 2/15 | 14/15 | 9/15 | 15/15 |
| **actions_holdout2 (20, fresh)** | not run | **18/20 (90%)** | not run | **16/20 (80%)** |

The no-key jump from 4/12 to 12/12 is mostly a weak baseline: v1 without a model never tried to send, book or move anything. The v2 resolver was written after reading train/dev/new-dev, so those are development numbers; the fresh 90% is the honest one. On the fresh set the model path (80%) did *worse* than the rules alone (90%).

### What the changes are
- **Search**: each source (meetings, Slack, email, calendar, dictation, Codex, ChatGPT) gets its own ranking so no source drowns out another; a local relevance model (cross-encoder) re-orders the candidates, so the no-key mode is much better than v1's. A better embedding model (`gte-base`).
- **Changing facts, relative time, broad questions**: later corrections are pulled in next to the first statement; "after / before / the day before X" finds X first and then the window around it; an explicit date shows that day's calendar; a commitments list (who promised what, extended, done) feeds "what do I owe / what is open".
- **Answers**: a quote check that forgives small drift ("the", "sorry") but still requires every number and name to match; answers can be partial ("I couldn't find who approved it"); date gaps are computed in code, not by the model.
- **Actions**: acts when the person/event/time can be worked out (a person with no Slack is emailed; a series uses its next occurrence; "on Slack", "at <Company>", a channel or a very recent conversation picks between two people with the same first name), and asks one specific question only when two candidates stay equally plausible or a required value is missing.
- **Providers and failures**: any OpenAI-compatible service works, models are picked automatically, a payment failure (HTTP 402) or exhausted quota moves on to the next service loudly, and parameter differences between services (`max_completion_tokens`, `temperature`, `seed`, reasoning models that run out of tokens) are handled.
- **Secrets**: the over-eager masking that also hid "SSO is on our Q4 roadmap" now masks real credentials only (the pasted key in the data is still masked; tests check this without printing it).

**Corrections to earlier numbers** (everything above is re-scored on the final commit): actions with a model on train is 11/12 (not 12/12) and on dev 29/30 (not 30/30), both older pre-audit Groq numbers; no-key dev is 30/30 (not 29/30); fresh-set v2 with a model is 78.9% / 68.0% / 0 of 6 (earlier run: 84.2% / 64.0% / 1 of 6). The v1 pipeline without a key was 63.2% on the fresh set in an earlier run and is 68.4% now: the earlier run happened before the dense index was built (BM25 only); four earlier commits give 63.2% without the index and 68.4% with it, equal to the original v1 code. `run.sh` always builds the index first.

**Optional model pass over the commitments list** (`USE_LEDGER_LLM`, about 15k tokens): it judged 94 items, dropped none, changed no owner; scores were identical on and off, so it stays **off**.

**No-key answer sentence**: the extractive answer is the best sentence (under 40 words, no headers or signatures) among the top 3 records, chosen by the cross-encoder. No-key answers: train 37.0 to 40.7, v2_dev 22.5 to 37.5, v2_holdout 44.0 to 40.0 (one question worse): a small tuned gain, not a clean one.

## What still does not work (with numbers)
- **Broad "what is still open / what do I owe" questions**: 0 of 2 retrieved on each of the two newest sets, with or without a model. The commitments list helps on easy cases but misses promises spread across many records.
- **No key means weak answers**: extractive answers score 8-40% and **every unanswerable question gets an answer** (6 of 6 on both holdouts). Two no-key "is this in memory?" checks (words meeting in one record; cross-encoder score) caught 1 of 7 unanswerable questions on the tuning sets and 0 of 6 on the holdout, so they stay off.
- **The model path for actions can be worse than the rules** (80% vs 90% on the fresh set).
- Holdout retrieval without a model did not move from v1 to v2 (75% both), so part of the train/dev gain is tuning.
- With a model, one question's answer can change between runs by a point or two; a few percent of model calls fail on this service and that question falls back to an extractive answer.
- Some questions need records scattered across many items ("which vendors sent cold emails"): not solved.
- A possible error in an older dev question (`MEM-DEV-09` lists an event that is not on the asked day) was left as is.

## Spend
v1 (the first submission) used only free tiers (Groq). v2 used OpenRouter: total token spend **$0.0814** (about **2.56 million tokens**: gpt-oss-120b 0.65M for ranking, gpt-oss-20b 1.91M for analysis, answers and actions), counted from `outputs/usage_ledger.jsonl` by `scripts/spend.py`. I bought $0 of credit in advance. The fresh-set run costs about 909 tokens per question. No-key runs cost nothing. All work lives on `main`; there is no separate v2 branch.

## What was tuned on what, and the cost
- Tuned on **train, dev, v2_dev** (weights for the relevance model, bonus sizes, which features stay on). `v2_holdout` totals decided one thing (the new answer writer is on). `v2_holdout2` and `actions_holdout2` were written after the audit in `docs/OVERFIT_AUDIT.md`, frozen, and run once.
- **Models used**: OpenRouter `openai/gpt-oss-120b` ($0.037 in / $0.17 out per million tokens) to rank candidates, `openai/gpt-oss-20b` ($0.018 / $0.09) for question analysis, answers and actions. Earlier v1 numbers used the same two models on Groq's free tier (no cost).
- **Cost**: a cold-cache run of the 27 train questions costs about 284k tokens (about 1.2 US cents); the whole v2 effort cost **$0.0814** (see Spend). The commitments list is built by rules (0 tokens). No-key runs cost nothing, about 10 s per question; with a model 8-50 s.

## Check it works
```bash
./.venv/bin/python -m pytest -q                 # 430 fast tests, about 15 s, no network, no key
./.venv/bin/python -m pytest -q -m slow         # 12 integration tests, about 20 s
./scripts/release_check.sh                      # clean clone, both test suites, no-key run (92.0% on train), secret scan, forbidden-id check: PASSED on the v2 commit
                                                # (if your default python3 cannot create a venv, run it as PYTHON=/path/to/python3.13 ./scripts/release_check.sh)
.venv/bin/python scripts/eval_report.py --questions evals/v2_dev.jsonl           # per-category scores, abstention causes, tokens
.venv/bin/python scripts/provider_check.py                                       # every model stage on every configured service
PIPELINE=v1 ./run.sh memory ...                 # the v1 behaviour (with the prompt examples cleaned up)
```

## Repo map
```text
memory/        retrieve.py (v1 path), retrieve_v2.py (lanes, cross-encoder, extras), lanes.py, crossenc.py, anchor.py, chains.py, ledger.py,
               people.py, quotes.py, arith.py, answer.py (writer, quote check), llm.py (services, cache, failover), degraded.py (loud banner)
actions/       planner.py, resolve.py (deterministic act-vs-ask), timeparse.py, rules.py
evals/         train/dev/v2 sets, frozen holdouts; scripts/ eval_report, ablate, provider_check, verify_eval_set, spend, gate_calibrate
docs/          DEVLOG.md (everything tried, incl. failures), OVERFIT_AUDIT.md, V2_PLAN.md, dev_set_notes.md, baselines/
```

## Development history

Older measurements, not claims about the current code.

| When | Commit | Model | Number |
|---|---|---|---|
| 2026-09-30 | `63f1f11` | llama-3.3-70b-versatile | retrieval 92.0% (train, 45 candidates) |
| 2026-10-01 | `98486bf` | gpt-oss-120b | retrieval 76.0%; answers 63.0% strict (rules) and 59.3% (judged) |
| 2026-10-01/02 | `4a48683` | gpt-oss-120b | retrieval 84.0% (retrieval-only) |
