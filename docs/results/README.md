# Scorer output behind the README tables

Each file is the output of an official scorer in `eval_harness/` on the final commit, with no model calls.
Modes: `nokey` = no key; `v2model` = v2 with a model, replayed from the saved response cache (set `LLM_FROZEN_ROLES=strong,fast` and any placeholder key so cache hits are served and a miss never calls a service); `v1nokey` / `v1model` = `PIPELINE=v1`.

| File pattern | Produced by |
|---|---|
| `retrieval_<set>_<mode>.json` | `python eval_harness/score_retrieval.py --gold evals/<set>.jsonl --answers <answers.jsonl> --quiet --out <file>` |
| `answers_<set>_<mode>.json` | `python eval_harness/score_memory.py --gold evals/<set>.jsonl --answers <answers.jsonl> --judge none --quiet --out <file>` (rule score, no judge model) |
| `actions_<set>_<mode>.json` | `python eval_harness/score_actions.py --gold evals/actions_<set>.jsonl --predictions <plans.jsonl> --out <file>` |

The answers file for a row comes from `python scripts/eval_report.py --questions evals/<set>.jsonl --out <answers.jsonl>` (memory) or `python -m actions.cli --commands evals/actions_<set>.jsonl --out <plans.jsonl>` (actions), with `PIPELINE=v1` for the v1 files.
Not regenerated here: the v1-pipeline rows on train / v2_dev (Groq, measured on 10-02/03) and v1 actions with a model; they are older measurements and are labelled as such in the README.
