"""CLI for retrieval-driven answers.

Robustness contract (see docs/DEVLOG.md "hardening" entry):
- one question raising ANY exception never aborts the run
- output is written incrementally, one flushed line per question
- `--resume` skips questions already present in an existing output file
- output line order always matches input line order, even with WORKERS>1
- total LLM token usage for the run is logged to outputs/run_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory import config
from memory import diagnostics
from memory.retrieve import baseline_retrieve, retrieve, warmup
from memory.answer import answer_question
from memory.llm import get_usage, reset_usage
from memory import degraded, ordered_io

LOG = logging.getLogger(__name__)

DATA_DIR_DEFAULT = os.environ.get("DATA_DIR", "./data")
RUN_STATS_PATH = Path("outputs/run_stats.json")
DIAGNOSTICS_PATH = Path("outputs/diagnostics.jsonl")


def _read_lines(path: Path) -> list[dict]:
    items = []
    with path.open() as source:
        for line in source:
            if line.strip():
                items.append(json.loads(line))
    return items


def _existing_answers(out: Path) -> dict[str, str]:
    """id -> raw output line, for --resume."""
    if not out.exists():
        return {}
    existing = {}
    with out.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row:
                existing[row["id"]] = line if line.endswith("\n") else line + "\n"
    return existing


def _process_one(item: dict, data_dir: str, cache_dir: str, retrieval_only: bool) -> tuple[str, dict]:
    """Compute one answer line (+ its diagnostics record). Never raises --
    any failure becomes a valid abstained line citing the best-effort
    baseline retrieval, per the hard rule that a single bad question must
    not crash the run.
    """
    qid = item["id"]
    question = item["question"]
    as_of = item["as_of"]
    diagnostics.begin()
    try:
        if retrieval_only:
            ranked = retrieve(question, as_of, data_dir=data_dir, cache_dir=cache_dir)
            ids = [uid for uid, _score in ranked]
            row = {"id": qid, "answer": "", "sources": [], "retrieved": ids[:20], "abstained": False}
        else:
            row = answer_question(qid, question, as_of, data_dir=data_dir, cache_dir=cache_dir)
    except Exception:
        LOG.exception("processing raised for question id=%s; falling back to baseline-only retrieval", qid)
        try:
            fallback_ids = [uid for uid, _score in baseline_retrieve(question, as_of, data_dir=data_dir, cache_dir=cache_dir)]
        except Exception:
            LOG.exception("baseline-only retrieval also failed for question id=%s; returning empty result", qid)
            fallback_ids = []
        row = {
            "id": qid,
            "answer": "I don't know. (internal error)",
            "sources": [],
            "retrieved": fallback_ids[:20],
            "abstained": True,
        }
        diagnostics.mark_degraded("crash")
    diag = diagnostics.finish(qid)
    return json.dumps(row) + "\n", diag


def answer(questions: Path, out: Path, data_dir: str | None = None, cache_dir: str = ".cache",
           resume: bool = False, workers: int | None = None, stats_path: Path | None = None,
           retrieval_only: bool = False, diagnostics_path: Path | None = None,
           ids: set[str] | None = None) -> dict[str, int]:
    """Run all questions; returns {degraded stage: number of questions it affected} (empty when nothing degraded)."""
    data_dir = data_dir or DATA_DIR_DEFAULT
    workers = workers if workers is not None else config.WORKERS
    out.parent.mkdir(parents=True, exist_ok=True)

    items = _read_lines(questions)
    started = time.monotonic()
    reset_usage()

    # `ids`: recompute only this subset (e.g. re-running a handful of
    # questions that degraded, without paying for the whole set again).
    # Every other id already in --out is kept untouched, regardless of
    # --resume. This targeted mode composes the final file at the end (it is
    # small by design). The normal mode below writes each result as soon as
    # it is ready, in input order (see memory/ordered_io.py).
    diagnostics_path = diagnostics_path or DIAGNOSTICS_PATH
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diag_by_id: dict[str, dict] = {}

    def _compute(item: dict) -> tuple[str, dict]:
        return _process_one(item, data_dir, cache_dir, retrieval_only)

    if ids:
        todo = [item for item in items if item["id"] in ids]
        existing = {qid: line for qid, line in _existing_answers(out).items() if qid not in ids}
        computed: dict[str, str] = {}
        ordered_io.run_in_order(todo, _compute, lambda it, r: (computed.__setitem__(it["id"], r[0]),
                                                                  diag_by_id.__setitem__(it["id"], r[1])), workers)
        with out.open("w") as target:
            for item in items:
                line = existing.get(item["id"]) or computed.get(item["id"])
                if line is not None:
                    target.write(line)
                    target.flush()
                    os.fsync(target.fileno())
        with diagnostics_path.open("w") as diag_file:
            for item in items:
                if item["id"] in diag_by_id:
                    diag_file.write(json.dumps(diag_by_id[item["id"]]) + "\n")
    else:
        existing = ordered_io.start_output(out, resume)
        existing = {qid: line for qid, line in existing.items() if qid in {i["id"] for i in items}}
        todo = [item for item in items if item["id"] not in existing]
        if not resume:
            diagnostics_path.write_text("")

        def _sink(item: dict, result: tuple[str, dict]) -> None:
            line, diag = result
            diag_by_id[item["id"]] = diag
            ordered_io.append_line(out, line)
            ordered_io.append_line(diagnostics_path, json.dumps(diag) + "\n")

        ordered_io.run_in_order(todo, _compute, _sink, workers)
        ordered_io.finalize_order(out, [i["id"] for i in items])

    degraded_count = sum(1 for d in diag_by_id.values() if d.get("degraded_stages"))
    print(f"[diagnostics] {degraded_count}/{len(todo)} computed questions had at least one degraded stage "
          f"(see {diagnostics_path})")
    stage_counts, seen = degraded.summarize(diag_by_id.values())
    print(degraded.banner(stage_counts, seen, str(diagnostics_path)))

    # Per-stage average tokens/question (step 1, token budget task): sum
    # each stage's tokens across all computed questions, divide by question
    # count (not call count) so this reads as "tokens this stage added to
    # the per-question total", matching the <=6000/question budget goal.
    stage_totals: dict[str, dict[str, int]] = {}
    n_questions = max(len(diag_by_id), 1)
    for d in diag_by_id.values():
        for stage, s in (d.get("tokens_by_stage") or {}).items():
            entry = stage_totals.setdefault(stage, {"prompt_tokens": 0, "completion_tokens": 0})
            entry["prompt_tokens"] += s["prompt_tokens"]
            entry["completion_tokens"] += s["completion_tokens"]
    if stage_totals:
        print("[diagnostics] average tokens/question by stage (prompt+completion):")
        grand_total = 0.0
        for stage in sorted(stage_totals):
            s = stage_totals[stage]
            avg_prompt = s["prompt_tokens"] / n_questions
            avg_completion = s["completion_tokens"] / n_questions
            grand_total += avg_prompt + avg_completion
            print(f"    {stage:>10}: {avg_prompt:7.1f} prompt + {avg_completion:6.1f} completion "
                  f"= {avg_prompt + avg_completion:7.1f}")
        print(f"    {'TOTAL':>10}: {grand_total:7.1f} tokens/question")

    elapsed = time.monotonic() - started
    usage = get_usage()
    stats_path = stats_path or RUN_STATS_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "questions": len(items),
        "computed": len(todo),
        "resumed_from_existing": len(existing),
        "wall_seconds": round(elapsed, 3),
        "mode": config.MODE,
        "workers": workers,
        "retrieval_only": retrieval_only,
        "degraded_questions": degraded_count,
        "avg_tokens_by_stage": {s: {"prompt_tokens": round(v["prompt_tokens"] / n_questions, 1),
                                     "completion_tokens": round(v["completion_tokens"] / n_questions, 1)}
                                 for s, v in stage_totals.items()},
        **usage,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return dict(stage_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("answer")
    command.add_argument("--questions", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--data-dir", default=None)
    command.add_argument("--cache-dir", default=".cache")
    command.add_argument("--resume", action="store_true", help="skip questions already answered in --out")
    command.add_argument("--workers", type=int, default=None)
    command.add_argument("--retrieval-only", action="store_true",
                          help="skip the answer writer; write retrieved ids only (answer='', sources=[])")
    command.add_argument("--ids", default=None,
                          help="comma-separated question ids; recompute only this subset, "
                               "leaving every other id already in --out untouched")
    command.add_argument("--strict", action="store_true",
                          help="exit with status 3 if any stage degraded (no key, quota, provider failure, missing local model)")
    sub.add_parser("warmup", help="download/load embeddings and cache every unit")
    args = parser.parse_args()
    if args.command == "answer":
        ids = set(x.strip() for x in args.ids.split(",") if x.strip()) if args.ids else None
        summary = answer(Path(args.questions), Path(args.out), data_dir=args.data_dir, cache_dir=args.cache_dir,
                         resume=args.resume, workers=args.workers, retrieval_only=args.retrieval_only, ids=ids)
        if args.strict and summary:
            raise SystemExit(3)
    elif args.command == "warmup":
        dimension, count, seconds = warmup()
        print(f"Warmup complete: dimension={dimension}, vectors={count}, seconds={seconds:.2f}")


if __name__ == "__main__":
    main()
