#!/usr/bin/env python3
"""Inspect retrieval coverage for one labelled development question."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.retrieve import retrieve
from memory.store import MemoryStore


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question_id")
    parser.add_argument("--gold", default=str(ROOT / "evals/memory_train.jsonl"))
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    item = next((x for x in rows(Path(args.gold)) if x["id"] == args.question_id), None)
    if item is None:
        raise SystemExit(f"Question not found: {args.question_id}")
    data_dir = args.data or str(ROOT / "data")
    store = MemoryStore(data_dir)
    as_of = datetime.fromisoformat(item["as_of"])
    visible = {u.id: u for u in store.visible(as_of)}
    all_units = store.id_to_unit
    needed = item.get("needed") or [[x] for x in item.get("evidence", [])]
    also = set(item.get("also_supports", []))
    ranked, meta = retrieve(item["question"], item["as_of"], data_dir=data_dir, return_meta=True)
    print(f"{item['id']}: {item['question']}  (as of {item['as_of']})")
    if "analysis" in meta:
        print(f"Intent: {meta['analysis'].get('intent')} | sub_queries: {meta['analysis'].get('sub_queries')}")
    if "rerank_reason" in meta and meta["rerank_reason"] != "N/A":
        print(f"Rerank Reason: {meta['rerank_reason']}")
    print("rank  unit id                    source     speaker              time                       snippet                                                                                              origin/flags")
    
    origins = meta.get("origins", {})
    for rank, (unit_id, _score) in enumerate(ranked, 1):
        unit = all_units.get(unit_id)
        if unit is None:
            continue
        flags: list[str] = []
        if any(unit_id in group for group in needed):
            flags.append("NEEDED")
        if unit_id in also:
            flags.append("ALSO")
        if unit_id not in visible:
            flags.append("FORBIDDEN")
        snippet = " ".join(unit.text.split())[:100]
        origin = origins.get(unit_id, "unknown")
        print(f"{rank:>4}  {unit_id:<25} {unit.source:<10} {(unit.speaker or '-'):18.18} {unit.time.isoformat():<26} {snippet:<100} [{origin}] {' '.join(flags)}")
    print("\nNeeded groups:")
    rank_of = {unit_id: rank for rank, (unit_id, _score) in enumerate(ranked, 1)}
    missing = []
    for number, group in enumerate(needed, 1):
        ranks = [rank_of[member] for member in group if member in rank_of]
        best = min(ranks) if ranks else None
        if best is None:
            missing.append(number)
        print(f"  group {number}: {'best rank ' + str(best) if best else 'MISSING'}")
    print("Missing needed groups: " + (", ".join(map(str, missing)) if missing else "none"))


if __name__ == "__main__":
    main()
