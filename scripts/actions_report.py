"""Run the action planner on an eval file and print pass rate, argument rate, clarify review and token use.

  scripts/with_keys.sh .venv/bin/python scripts/actions_report.py evals/actions_train.jsonl [more files...]
  LLM_FROZEN_ROLES=strong,fast ... (replay cache only)       .venv/bin/python scripts/actions_report.py FILE   (no key: rules only)
For every command whose predicted plan contains a `clarify` it prints the command, the question asked and the expected
plan, so each clarify can be judged justified / unnecessary by hand. Uses the official scorer unmodified.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_harness"))

import score_actions  # noqa: E402

from actions.cli import answer  # noqa: E402


def main(paths: list[str]) -> None:
    total = passed = 0
    for path in paths:
        name = Path(path).stem
        out = ROOT / "outputs" / "scratch" / f"{name}.plans.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        answer(Path(path), out, diagnostics_path=ROOT / "outputs" / "scratch" / f"{name}.actions_diag.jsonl")
        gold = {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}
        preds = {json.loads(l)["id"]: json.loads(l) for l in open(out) if l.strip()}
        n_ok = n_args = n_args_tot = unnecessary = 0
        print(f"\n=== {name}: {len(gold)} commands")
        for cid, g in gold.items():
            pred = preds.get(cid, {"actions": []})["actions"]
            ok = False
            for cand in [g["expected"]] + g.get("alternatives", []):
                if score_actions.list_ok(cand, pred)[0]:
                    ok = True
                    break
            n_ok += ok
            clar = [a for a in pred if a["type"] == "clarify"]
            expects_clarify = any(a["type"] == "clarify" for a in g["expected"])
            if clar:
                just = "expected" if expects_clarify else ("accepted alternative" if ok else "UNNECESSARY?")
                unnecessary += (not expects_clarify and not ok)
                print(f"  clarify [{just}] {cid}: {g['command']!r} -> {clar[0]['args'].get('question')!r}")
            if not ok:
                print(f"  FAIL {cid}: {g['command']!r}\n       pred={json.dumps(pred)[:260]}")
        total += len(gold)
        passed += n_ok
        print(f"  pass {n_ok}/{len(gold)} = {n_ok / len(gold):.1%}; unnecessary clarifies: {unnecessary}")
    print(f"\nALL: {passed}/{total} = {passed / total:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
