"""Study the no-key "is this answer in memory?" signals on train / dev / v2_dev (never the holdout).

Prints, per question: label (answerable?), the top cross-encoder score, the co-occurrence coverage of the top records,
and the v1 coverage numbers, so a general threshold can be chosen and checked. Output is a TSV in outputs/scratch/gate_study.tsv.
  USE_LANES=true USE_CROSS_ENCODER=true .venv/bin/python scripts/gate_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import coverage  # noqa: E402
from memory.index import search_text, tokenize  # noqa: E402
from memory.retrieve import _resources, retrieve  # noqa: E402
import memory.llm as llm  # noqa: E402

llm.GROQ_API_KEY = None


def cooccurrence(question: str, ids: list[str], visible, store, top_n: int = 5) -> float:
    """Best idf-weighted share of the question's content terms found in ONE record + its adjacent meeting segments."""
    terms = [t for t in dict.fromkeys(tokenize(question)) if t not in coverage.FILLER]
    if not terms:
        return 1.0
    toks = {u.id: set(tokenize(search_text(u))) for u in visible}
    import math
    from collections import Counter
    df: Counter = Counter()
    for t in toks.values():
        df.update(t)
    n = len(toks)
    idf = {t: math.log(1 + (n - df[t] + .5) / (df[t] + .5)) for t in terms}
    total = sum(idf.values()) or 1.0
    best = 0.0
    by_record: dict[str, list[str]] = {}
    for u in visible:
        by_record.setdefault(u.record_id, []).append(u.id)
    for rid in ids[:top_n]:
        u = store.get(rid)
        group = {rid}
        if u.source == "meeting":
            seq = by_record[u.record_id]
            i = seq.index(rid)
            group |= set(seq[max(0, i - 1): i + 2])
        have = set().union(*(toks.get(g, set()) for g in group))
        best = max(best, sum(idf[t] for t in terms if coverage._record_has(t, have)) / total)
    return best


def main() -> None:
    store, index = _resources("./data", ".cache")
    rows = []
    for f in ("memory_train", "memory_dev", "v2_dev"):
        for line in (ROOT / "evals" / f"{f}.jsonl").read_text().splitlines():
            q = json.loads(line)
            ranked, meta = retrieve(q["question"], q["as_of"], return_meta=True)
            ids = [u for u, _ in ranked]
            visible = store.visible(q["as_of"])
            ce = meta.get("ce_scores", {})
            stats = coverage.coverage_stats(q["question"], visible, ids)
            top_ce = max((ce.get(i, -99) for i in ids[:3]), default=-99)
            rows.append((q["id"], int(q["answerable"]), round(top_ce, 2), round(cooccurrence(q["question"], ids, visible, store), 3),
                         round(stats["best_coverage"], 3), round(stats["missing_share"], 3)))
            print(*rows[-1], sep="\t", flush=True)
    out = ROOT / "outputs" / "scratch" / "gate_study.tsv"
    out.write_text("id\tanswerable\ttop_ce\tcooccur\tbest_cov\tmissing\n" + "\n".join("\t".join(map(str, r)) for r in rows))


if __name__ == "__main__":
    main()
