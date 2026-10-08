# V2 plan and progress (branch `v2`, from main @ 8f1a418)

Resume rule: if context resets, read this file, then `docs/DEVLOG.md` (tail), then continue at the first unchecked phase.
v1 hidden-test result (reviewer): retrieval 79%, answers 50%, actions 10/13. Feedback: too many "I don't know"; retrieval/answers can improve; how does it run without Groq / on a non-Groq provider.

Hard rules: no train/dev/holdout text, answers or ids in code or prompts; never print a key; never delete `.cache/llm` (backup first: `scripts/backup_llm_cache.sh`); never push; never touch main.

| Phase | Status | Key numbers |
|---|---|---|
| 0 Setup | done | branch v2 created; cache backed up (113 files); OPENROUTER_* added to gitignored .env |
| 1 Fast tests | done | 363 tests 20.7 s (cold) / 10.5 s warm -> 269 fast in 3.4 s + 10 slow in 1.1 s; socket guard on |
| 2 Eval sets + diagnostics | done | v2_dev 40 + v2_holdout 25 (verified, 0 problems); eval_report/ablate/verify scripts; v1 baselines in DEVLOG + docs/baselines |
| 3 Retrieval v2 (3a-3f) | todo | |
| 4 Answers v2 | todo | |
| 5 Actions ask vs act | done | with key: train 12/12, dev 30/30 (v1 27/30), new 15/15 (v1 9/15); no key: 12/12, 30/30, 14/15 (v1 4/12, 13/30, 2/15); 0 unnecessary clarifies |
| 6 Providers + loud failures | mostly done | banner + --strict + diagnostics; max_completion_tokens/temperature/truncation/JSON-habit handling; provider_check: groq 5/5, openrouter(free models) 5/5; fake strict server test |
| 7 Finish (README, tag v2) | todo | |

## Files expected to change
memory/{config,retrieve,index,query,answer,llm,cli,diagnostics,ingest}.py, new memory/{lanes,chains,anchor,people,ledger}.py,
actions/{planner,rules,context}.py, scripts/{eval_report,provider_check,release_check}.{py,sh}, tests/*, evals/v2_{dev,holdout}.jsonl,
docs/{DEVLOG,V2_PLAN,dev_set_notes}.md, README.md, pytest.ini.
