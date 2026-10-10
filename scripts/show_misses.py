"""List the questions whose needed groups are not all in the top-10 of an answers file (tuning aid; refuses the holdout)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval_harness"))
import records  # noqa: E402
import score_retrieval as sr  # noqa: E402

gold_path, ans_path = sys.argv[1], sys.argv[2]
if "holdout" in gold_path:
    sys.exit("refusing: individual holdout failures must not be inspected")
gold = [json.loads(l) for l in open(gold_path) if l.strip()]
answers = {json.loads(l)["id"]: json.loads(l) for l in open(ans_path) if l.strip()}
ctx = records.context(str(ROOT / "data"))
rec = lambda c: ctx["record_of"].get(c, c)
for item in gold:
    if not item["answerable"] or not item.get("needed"):
        continue
    ranked = [x for x in answers[item["id"]]["retrieved"]]
    missing = []
    for g in item["needed"]:
        ranks = [i + 1 for i, c in enumerate(ranked) if c in g or (rec(c) in g and rec(c) not in ctx["record_of"])]
        if not ranks or min(ranks) > 10:
            missing.append((g[:3], min(ranks) if ranks else None))
    if missing:
        print(item["id"], item["category"], "|", item["question"][:70], "| missing:", missing)
