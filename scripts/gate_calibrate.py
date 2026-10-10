"""Calibrate ONE global no-key abstention threshold on the cross-encoder relevance of the top evidence (train + v2_dev only).

Features per question (final v2 ranking, no key): ce1 = CE score of the top-ranked record, ce_gap = ce1 - ce2, ce_max3.
Reports, per feature, the highest threshold that refuses ZERO answerable training questions and how many unanswerable ones it catches,
then (with --holdout) applies that single threshold to the holdout and prints AGGREGATE counts only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from unittest.mock import patch  # noqa: E402

from memory.retrieve import retrieve  # noqa: E402


def feats(q):
    with patch("memory.llm.GROQ_API_KEY", None), patch("memory.llm.is_available", return_value=False):
        ranked, meta = retrieve(q["question"], q["as_of"], return_meta=True)
    ids = [u for u, _ in ranked]
    ce = meta.get("ce_scores", {})
    s = [ce.get(i, -99.0) for i in ids[:3]] + [-99.0] * 3
    return {"ce1": s[0], "ce_gap": s[0] - s[1], "ce_max3": max(s[:3])}


def load(names):
    return [json.loads(l) for n in names for l in (ROOT / "evals" / f"{n}.jsonl").read_text().splitlines() if l.strip()]


def main():
    tune = [dict(q, f=feats(q)) for q in load(["memory_train", "v2_dev"])]
    ans = [q for q in tune if q["answerable"]]
    un = [q for q in tune if not q["answerable"]]
    print(f"tune set: {len(ans)} answerable, {len(un)} unanswerable")
    chosen = {}
    for name in ("ce1", "ce_gap", "ce_max3"):
        floor = min(q["f"][name] for q in ans)
        caught = sum(q["f"][name] < floor for q in un)
        print(f"  {name:8s}: answerable min = {floor:7.2f}; a threshold at/below it refuses 0 answerable and catches {caught}/{len(un)} unanswerable")
        chosen[name] = floor
        vals = sorted(q["f"][name] for q in ans)
        for pct in (0.02, 0.05, 0.10):
            t = vals[int(len(vals) * pct)]
            print(f"      threshold {t:7.2f} (refuses {sum(q['f'][name] < t for q in ans)} answerable) catches {sum(q['f'][name] < t for q in un)}/{len(un)} unanswerable")
    if "--holdout" in sys.argv:
        ho = [dict(q, f=feats(q)) for q in load(["v2_holdout"])]
        for name, t in chosen.items():
            a = [q for q in ho if q["answerable"]]
            u = [q for q in ho if not q["answerable"]]
            print(f"  HOLDOUT aggregate with threshold {t:.2f} on {name}: refuses {sum(q['f'][name] < t for q in a)}/{len(a)} answerable, catches {sum(q['f'][name] < t for q in u)}/{len(u)} unanswerable")


if __name__ == "__main__":
    main()
