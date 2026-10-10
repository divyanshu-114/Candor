"""Verify an eval question file against the real records, independently of our own code.

WHY: a gold answer or `needed` id written from memory is the easiest way to poison an
evaluation. This script re-derives everything from the official harness loader
(eval_harness/records.py, raw text, no masking), not from memory/store.py, and checks:
  1. ids unique, required fields present, as_of parses
  2. every needed / evidence / also_supports id exists (as a unit or a whole record)
  3. each is VISIBLE at as_of (delivered by then, not deleted by then)
  4. for answerable questions: every key_terms group matches the text of the cited records
  5. stale / future_terms never appear in the visible needed records (the as_of is consistent)
  6. for unanswerable questions: lists visible units containing ALL `absence_probe` words, for a
     human to confirm the asked fact really is absent
Usage: python3 scripts/verify_eval_set.py evals/v2_dev.jsonl [--data data] [--show-probes]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval_harness"))
import records  # noqa: E402

REQUIRED = ("id", "question", "as_of", "category", "answerable", "gold_answer", "rubric", "needed")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def verify(path: str, data_dir: str = "data", show_probes: bool = False) -> list[str]:
    ctx = records.context(data_dir)
    avail, record_of, deleted = ctx["avail"], ctx["record_of"], ctx["deleted"]
    units_of: dict[str, list[str]] = {}
    for uid, rid in record_of.items():
        units_of.setdefault(rid, []).append(uid)
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("id", "?")

        def err(msg: str) -> None:
            errors.append(f"{qid}: {msg}")

        if qid in seen_ids:
            err("duplicate id")
        seen_ids.add(qid)
        for field in REQUIRED:
            if field not in row:
                err(f"missing field {field}")
        try:
            as_of = datetime.fromisoformat(row["as_of"])
        except Exception:
            err("as_of does not parse")
            continue
        visible = {u.id: u for u in records.visible(data_dir, as_of)}
        ids = [i for g in row.get("needed", []) for i in g] + row.get("evidence", []) + row.get("also_supports", [])
        cited_text = []
        for cid in dict.fromkeys(ids):
            if cid not in avail:
                err(f"unknown id {cid}")
                continue
            members = units_of.get(cid, [cid]) if cid in units_of and cid not in record_of else [cid]
            if cid in record_of:
                members = [cid]
            if avail[cid] > as_of:
                err(f"{cid} not delivered at as_of (available {avail[cid].isoformat()})")
            if cid in deleted and deleted[cid] <= as_of:
                err(f"{cid} deleted at/before as_of")
            for m in members:
                if m in visible:
                    cited_text.append(visible[m].text)
        if row.get("answerable") and not row.get("needed"):
            err("answerable question without needed groups")
        blob = _norm(" ".join(cited_text))
        if row.get("answerable") and not row.get("computed"):  # computed answers (date gaps, counts) are not literal text
            for group in row.get("key_terms", []):
                if not any(_norm(t) in blob for t in group):
                    err(f"key_terms group {group} not found in cited records")
        needed_blob = " ".join(_norm(visible[i].text) for g in row.get("needed", []) for i in g if i in visible)
        leaked = [x for x in (row.get("future_terms") or []) if _norm(x) in needed_blob]
        if leaked:
            err(f"future term(s) {leaked} already visible in needed records at as_of")
        probe = row.get("absence_probe")
        if probe:
            hits = [u.id for u in visible.values() if all(_norm(w) in _norm(u.text) for w in probe)]
            print(f"{qid}: probe {probe} -> {len(hits)} visible unit(s) contain all words" + (f": {hits[:6]}" if show_probes or hits else ""))
    print(f"{path}: {len(seen_ids)} questions, {len(errors)} problem(s)")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--data", default="data")
    ap.add_argument("--show-probes", action="store_true")
    a = ap.parse_args()
    errs = verify(a.file, a.data, a.show_probes)
    for e in errs:
        print("  PROBLEM", e)
    sys.exit(1 if errs else 0)
