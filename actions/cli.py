"""CLI for the TextOS action planner (dry run).

  python -m actions.cli --commands IN.jsonl --out OUT.jsonl

Same robustness contract as memory.cli: one command raising ANY exception
never aborts the run (falls back to a `clarify` action), output is written
in input order, `--resume` skips commands already in --out.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from actions import config
from memory import diagnostics, ordered_io
from actions.planner import plan
from actions.rules import GENERIC_CLARIFY

LOG = logging.getLogger(__name__)

DATA_DIR_DEFAULT = os.environ.get("DATA_DIR", config.DATA_DIR)


def _read_lines(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _existing(out: Path) -> dict[str, str]:
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


class _TokenCounter:
    """Per-run token accounting (thread-safe): tokens and LLM calls spent by
    planning, so a run can print what each command really cost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.commands = 0
        self.tokens = 0
        self.calls = 0          # real (non-cached) calls
        self.cached_calls = 0
        self.zero_llm = 0       # commands answered with no LLM call at all

    def add(self, diag: dict) -> None:
        calls = diag.get("llm_calls") or []
        with self._lock:
            self.commands += 1
            self.tokens += int(diag.get("prompt_tokens", 0)) + int(diag.get("completion_tokens", 0))
            self.calls += sum(1 for c in calls if not c.get("cache_hit"))
            self.cached_calls += sum(1 for c in calls if c.get("cache_hit"))
            self.zero_llm += 0 if calls else 1

    def summary(self) -> str:
        n = max(self.commands, 1)
        return (f"[actions] tokens: {self.tokens} total, {self.tokens / n:.0f}/command avg; "
                f"LLM calls: {self.calls / n:.2f}/command real + {self.cached_calls / n:.2f} cached; "
                f"{self.zero_llm}/{self.commands} commands needed no LLM call")


TOKENS = _TokenCounter()


def _process_one(item: dict, data_dir: str, cache_dir: str) -> str:
    cid = item["id"]
    command = item["command"]
    as_of = item["as_of"]
    diagnostics.begin()
    try:
        actions = plan(command, as_of, data_dir=data_dir, cache_dir=cache_dir)
    except Exception:
        LOG.exception("plan() raised for command id=%s; falling back to clarify", cid)
        actions = [{"type": "clarify", "args": {"question": GENERIC_CLARIFY}}]  # never echo the command
    TOKENS.add(diagnostics.finish(cid))
    return json.dumps({"id": cid, "actions": actions}) + "\n"


def answer(commands: Path, out: Path, data_dir: str | None = None, cache_dir: str = ".cache",
           resume: bool = False, workers: int | None = None, ids: set[str] | None = None) -> None:
    data_dir = data_dir or DATA_DIR_DEFAULT
    workers = workers if workers is not None else config.WORKERS
    out.parent.mkdir(parents=True, exist_ok=True)

    items = _read_lines(commands)
    # See memory.cli.answer: --ids recomputes only this subset, leaving
    # every other id already in --out untouched, regardless of --resume.
    started = time.monotonic()
    TOKENS.reset()

    def _compute(item: dict) -> str:
        return _process_one(item, data_dir, cache_dir)

    if ids:
        todo = [item for item in items if item["id"] in ids]
        existing = {cid: line for cid, line in _existing(out).items() if cid not in ids}
        computed: dict[str, str] = {}
        ordered_io.run_in_order(todo, _compute, lambda it, line: computed.__setitem__(it["id"], line), workers)
        with out.open("w") as target:
            for item in items:
                line = existing.get(item["id"]) or computed.get(item["id"])
                if line is not None:
                    target.write(line)
                    target.flush()
                    os.fsync(target.fileno())
    else:
        # Each plan is flushed + fsynced to --out as soon as it (and every
        # earlier command) is done, so a killed run keeps its finished prefix.
        known = {i["id"] for i in items}
        existing = {c: l for c, l in ordered_io.start_output(out, resume).items() if c in known}
        todo = [item for item in items if item["id"] not in existing]
        ordered_io.run_in_order(todo, _compute, lambda it, line: ordered_io.append_line(out, line), workers)
        ordered_io.finalize_order(out, [i["id"] for i in items])

    elapsed = time.monotonic() - started
    print(TOKENS.summary())
    print(f"[actions] {len(items)} commands ({len(todo)} computed, {len(existing)} resumed) in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ids", default=None,
                         help="comma-separated command ids; recompute only this subset, "
                              "leaving every other id already in --out untouched")
    args = parser.parse_args()
    ids = set(x.strip() for x in args.ids.split(",") if x.strip()) if args.ids else None
    answer(Path(args.commands), Path(args.out), data_dir=args.data_dir, cache_dir=args.cache_dir,
           resume=args.resume, workers=args.workers, ids=ids)


if __name__ == "__main__":
    main()
