"""Run eval_report.py (retrieval-only) for several flag sets and print a markdown table.

  .venv/bin/python scripts/ablate.py --files memory_train v2_dev --configs v1 "lanes=USE_LANES=true" ...
Each config is NAME=ENV1=val,ENV2=val (or just NAME for no env). The holdout is refused: tune on train/dev only.
Add --key to run with scripts/with_keys.sh (live model calls!) and --frozen to replay the cache only.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, env_str: str, qfile: str, key: bool, frozen: bool, retrieval_only: bool = True) -> dict:
    env = dict(os.environ)
    env.pop("GROQ_API_KEY", None) if not key else None
    for kv in filter(None, env_str.split(",")):
        k, v = kv.split("=", 1)
        env[k] = v
    if frozen:
        env["LLM_FROZEN_ROLES"] = "strong,fast"
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/eval_report.py"), "--questions", str(ROOT / f"evals/{qfile}.jsonl"),
           "--out", str(ROOT / f"outputs/scratch/abl_{qfile}_{name}.jsonl")]
    if retrieval_only:
        cmd.append("--retrieval-only")
    if key:
        cmd = [str(ROOT / "scripts/with_keys.sh")] + cmd
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"RETRIEVAL score ([\d.]+)\s+complete@5/10/20 = ([\d./]+)\s+MRR ([\d.]+)", out)
    t = re.search(r"tokens/question (\d+)\s+seconds/question ([\d.]+)", out)
    a = re.search(r"ANSWERS strict\(rules\) ([\d.]+).*?FALSE-ANSWER rate on unanswerable (\d+)/(\d+)", out, re.S)
    return {"score": float(m.group(1)) if m else None, "complete": m.group(2) if m else "-", "mrr": float(m.group(3)) if m else None,
            "tokens": int(t.group(1)) if t else None, "secs": float(t.group(2)) if t else None,
            "answers": float(a.group(1)) if a else None, "false_ans": f"{a.group(2)}/{a.group(3)}" if a else "-", "raw": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["memory_train", "v2_dev"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--key", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--answers", action="store_true", help="also run the answer writer (needs a key for real answers)")
    a = ap.parse_args()
    if any("holdout" in f for f in a.files):
        sys.exit("refusing: do not tune on the holdout (use scripts/eval_report.py once per phase)")
    print("| config | " + " | ".join(f"{f} score / c@5,10,20 / MRR" + (" / answers" if a.answers else "") for f in a.files) + " | tokens/q | s/q |")
    print("|---|" + "---|" * (len(a.files) + 2))
    for cfg in a.configs:
        name, _, env = cfg.partition("=")
        rows = [run(name, env, f, a.key, a.frozen, not a.answers) for f in a.files]
        cells = " | ".join(f"{r['score']} / {r['complete']} / {r['mrr']}" + (f" / {r['answers']} (false-ans {r['false_ans']})" if a.answers else "") for r in rows)
        print(f"| {name} | {cells} | {rows[-1]['tokens']} | {rows[-1]['secs']} |", flush=True)
