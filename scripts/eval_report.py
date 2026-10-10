"""One command: run the memory system on a question file and print a scoreboard.

  scripts/with_keys.sh .venv/bin/python scripts/eval_report.py --questions evals/v2_dev.jsonl
  .venv/bin/python scripts/eval_report.py --questions evals/v2_dev.jsonl --retrieval-only
  LLM_FROZEN_ROLES=strong,fast scripts/with_keys.sh .venv/bin/python scripts/eval_report.py --questions F   # replay cache, 0 tokens
  .venv/bin/python scripts/eval_report.py --questions F --reuse outputs/scratch/F.answers.jsonl           # re-score only

Prints, per category: retrieval score (all needed groups in the top 10 and nothing forbidden), answer
score (official rule scorer, strict: unverified counts as wrong), abstentions on answerable questions,
and the false-answer rate on unanswerable ones. Plus complete@5/10/20, MRR, tokens and seconds per
question, and the ABSTENTION DIAGNOSTIC: why each answerable-but-abstained question was withheld:
  a  needed record not in the evidence package the writer saw
  b  record present, the writer said "not answerable"
  c  writer answered, quote check failed, but a normalised-word match would have passed
  d  writer answered, quote check failed and the quote is really not in the evidence
  e  abstained without a writer verdict (no key / coverage gate / crash / absence guard)
The official scorers are imported unmodified; nothing here changes the system under test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_harness"))

import records  # noqa: E402  (official harness loader)
import score_memory  # noqa: E402
import score_retrieval  # noqa: E402

from memory.quotes import soft_quote_match  # noqa: E402


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def classify_abstention(item: dict, answer: dict, diag: dict, display_by_id: dict[str, str]) -> str:
    """One letter a/b/c/d/e for an answerable question that was abstained on (see module doc)."""
    details = (diag or {}).get("details", {})
    writer = details.get("writer")
    evidence_ids = set(details.get("evidence_ids") or [])
    groups = score_retrieval.needed_groups(item)
    record_of = {}
    covered = bool(groups) and all(any(m in evidence_ids for m in g) or
                                    any(i.startswith(m + "#") for m in g for i in evidence_ids) for g in groups)
    if writer is None:
        return "e"
    if not covered:
        return "a"
    if not writer.get("answerable"):
        return "b"
    quotes = writer.get("support") or []
    if any(soft_quote_match(q["quote"], display_by_id.get(q["id"], "")) for q in quotes):
        return "c"
    return "d"


def _display_texts(answer: dict, item: dict, data_dir: str) -> dict[str, str]:
    from memory.answer import build_evidence_package
    from memory.retrieve import _resources
    store, _ = _resources(data_dir, ".cache")
    _pkg, _ids, display = build_evidence_package(answer.get("retrieved", []), item["question"], item["as_of"], store,
                                                  return_display=True)
    return display


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default=None, help="answers file (default outputs/scratch/<name>.answers.jsonl)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--reuse", default=None, help="score an existing answers file instead of running")
    ap.add_argument("--ids", default=None, help="comma-separated ids to (re)compute")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None, help="also write the summary as JSON here")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="never print question ids or reasons (automatic for any file named *holdout*)")
    args = ap.parse_args()
    agg_only = args.aggregate_only or "holdout" in Path(args.questions).name  # holdout: aggregates only, never failures

    qpath = Path(args.questions)
    gold = _jsonl(qpath)
    scratch = ROOT / "outputs" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(args.reuse or args.out or scratch / f"{qpath.stem}.answers.jsonl")
    diag_path = scratch / f"{qpath.stem}.diag.jsonl"
    stats_path = scratch / f"{qpath.stem}.stats.json"
    t0 = time.monotonic()
    if not args.reuse:
        from memory.cli import answer as run_answer
        ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
        run_answer(qpath, out, data_dir=args.data_dir, retrieval_only=args.retrieval_only, workers=args.workers,
                   stats_path=stats_path, diagnostics_path=diag_path, ids=ids)
    wall = time.monotonic() - t0
    answers = {r["id"]: r for r in _jsonl(out)}
    diags = {d["id"]: d for d in _jsonl(diag_path)} if diag_path.exists() else {}
    ctx = records.context(args.data_dir)

    ret = score_retrieval.score(gold, answers, ctx)
    ret_by_id = {r["id"]: r for r in ret["items"]}
    s = ret["summary"]

    rows = []
    for item in gold:
        ans = answers.get(item["id"], {})
        r = {"id": item["id"], "category": item["category"], "answerable": item["answerable"],
             "ret": ret_by_id[item["id"]]["score"], "scored": ret_by_id[item["id"]]["scored"]}
        if not args.retrieval_only:
            verdict, reason, unverified = score_memory.deterministic(item, ans, ctx)
            r.update(verdict=verdict, unverified=unverified, reason=reason, abstained=bool(ans.get("abstained")),
                     strict=1.0 if verdict == "correct" and not unverified else 0.0)
        rows.append(r)

    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)

    print(f"\n=== {qpath.name} {args.label}| n={len(gold)} | {'retrieval-only' if args.retrieval_only else 'with answers'} "
          f"| frozen={os.environ.get('LLM_FROZEN_ROLES') or '-'} | key={'yes' if __import__('memory.llm', fromlist=['x']).is_available() else 'no'} ===")
    hdr = f"{'category':<22}{'n':>3}{'retrieval':>11}" + ("" if args.retrieval_only else f"{'answer':>8}{'abst/ans':>10}")
    print(hdr)
    for c in sorted(cats):
        v = cats[c]
        line = f"{c:<22}{len(v):>3}{sum(x['ret'] for x in v) / len(v):>11.2f}"
        if not args.retrieval_only:
            ans_n = [x for x in v if x["answerable"]]
            line += f"{sum(x['strict'] for x in v) / len(v):>8.2f}{sum(x['abstained'] for x in ans_n):>6}/{len(ans_n):<3}"
        print(line)
    score = s["score"]
    print(f"\nRETRIEVAL score {score['mean'] if isinstance(score, dict) and 'mean' in score else score}  "
          f"complete@5/10/20 = {s['complete_unit@5']}/{s['complete_unit@10']}/{s['complete_unit@20']}  MRR {s['mrr']}  "
          f"forbidden@10 questions {s['questions_with_forbidden@10']}  forbidden@20 total {s['forbidden_retrieved@20']}")
    if not args.retrieval_only:
        ansr = [r for r in rows if r["answerable"]]
        unans = [r for r in rows if not r["answerable"]]
        strict = sum(r["strict"] for r in rows) / len(rows)
        false_abst = [r for r in ansr if r["abstained"]]
        false_ans = [r for r in unans if r["verdict"] == "incorrect"]
        print(f"ANSWERS strict(rules) {strict:.3f} ({int(sum(r['strict'] for r in rows))}/{len(rows)})  "
              f"abstained on answerable {len(false_abst)}/{len(ansr)}  "
              f"FALSE-ANSWER rate on unanswerable {len(false_ans)}/{len(unans)}"
              + (f" = {len(false_ans) / len(unans):.2f}" if unans else ""))
        hard = [r["id"] for r in rows if r["reason"].startswith("hard fail")]
        print(f"hard failures: {len(hard)}" + ("" if agg_only else f" {hard[:6]}"))
        counts = Counter()
        detail = []
        for item in gold:
            ans = answers.get(item["id"], {})
            if item["answerable"] and ans.get("abstained"):
                disp = _display_texts(ans, item, args.data_dir) if diags.get(item["id"], {}).get("details", {}).get("writer") else {}
                k = classify_abstention(item, ans, diags.get(item["id"], {}), disp)
                counts[k] += 1
                detail.append((item["id"], k))
        print(f"ABSTENTION causes on answerable questions: a={counts['a']} b={counts['b']} c={counts['c']} d={counts['d']} e={counts['e']}  "
              f"(total {sum(counts.values())})" + ("" if agg_only else f" {detail}"))
    toks = sum(d.get("prompt_tokens", 0) + d.get("completion_tokens", 0) for d in diags.values())
    secs = sum(d.get("seconds", 0) for d in diags.values())
    n = max(len(diags), 1)
    degraded = Counter(st for d in diags.values() for st in d.get("degraded_stages", []))
    print(f"tokens/question {toks / n:.0f}  seconds/question {secs / n:.2f} (wall {wall:.1f}s)  degraded stages {dict(degraded) or 'none'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": s, "rows": rows, "tokens_per_q": toks / n, "sec_per_q": secs / n}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
