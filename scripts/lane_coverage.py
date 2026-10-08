"""Per-lane coverage of the first-stage pool (no LLM, no cross-encoder): for every needed record, which source lane
(meeting / slack / email / calendar / dictation / codex / chatgpt) put it in the pool, and at what rank inside that lane?
Also lists every question whose needed records live in the Codex or ChatGPT sources and whether the pool has them.
  .venv/bin/python scripts/lane_coverage.py [files...]      (default: train, dev, v2_dev; refuses the holdout)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memory.query import fallback_analyze  # noqa: E402
from memory.retrieve import _resources  # noqa: E402
from memory.retrieve_v2 import first_stage  # noqa: E402
from memory.corpus import get_info  # noqa: E402

files = sys.argv[1:] or ["memory_train", "memory_dev", "v2_dev"]
if any("holdout" in f for f in files):
    sys.exit("refusing: the holdout is not for diagnosis")
store, index = _resources("./data", ".cache")
source_of = get_info(store).source_of
found_by_lane: Counter = Counter()
missing = 0
total = 0
pool_sizes = []
special = defaultdict(list)
for f in files:
    for line in (ROOT / "evals" / f"{f}.jsonl").read_text().splitlines():
        q = json.loads(line)
        if not q["answerable"] or not q.get("needed"):
            continue
        visible = {u.id for u in store.visible(q["as_of"])}
        pool, lane_rank, _ = first_stage(q["question"], fallback_analyze(q["question"], q["as_of"]), visible, store, index)
        pool_ids = [u for u, _ in pool]
        for group in q["needed"]:
            total += 1
            members = [m for m in group if m in pool_ids or any(p.startswith(m + "#") for p in pool_ids)]
            if not members:
                missing += 1
                hit = None
            else:
                m = min(members, key=lambda m: pool_ids.index(m) if m in pool_ids else 10_000)
                lane = source_of.get(m, source_of.get(next((p for p in pool_ids if p.startswith(m + "#")), ""), "?"))
                found_by_lane[lane] += 1
                hit = (m, lane, lane_rank.get(m, {}).get(lane))
            srcs = {source_of.get(m, "meeting" if "#" in m else "?") for m in group}
            if srcs & {"codex", "chatgpt"}:
                special[q["id"]].append((group[0], hit))
        pool_sizes.append(len(pool_ids))
print(f"needed groups: {total}; in the first-stage pool: {total - missing} ({(total - missing) / total:.1%}); pool size mean {sum(pool_sizes) / len(pool_sizes):.0f}")
print("first lane to deliver the needed record:", dict(found_by_lane.most_common()))
print("Codex / ChatGPT questions:")
for qid, items in special.items():
    print("  ", qid, [(g, h) for g, h in items])
