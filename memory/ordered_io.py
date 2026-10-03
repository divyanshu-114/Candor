"""Crash-safe, input-ordered JSONL output shared by both CLIs.

WHY: both CLIs used to write --out only after every worker finished, so a
killed run (quota stop, ctrl-C, laptop sleep) lost everything. Here each
result is flushed + fsynced as soon as it is complete, but ONLY in input
order: a result is held back until every earlier id has been written, so the
file is always a valid prefix of the final output and --resume can simply
continue after it. Deterministic: final order == input order, regardless of
worker timing.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def load_valid(path: Path) -> dict[str, str]:
    """id -> raw line for every well-formed line (first occurrence wins).
    A truncated last line from a killed run is silently dropped."""
    found: dict[str, str] = {}
    if not path.exists():
        return found
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "id" in row and row["id"] not in found:
                found[row["id"]] = line if line.endswith("\n") else line + "\n"
    return found


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for line in lines:
            f.write(line)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def start_output(path: Path, resume: bool) -> dict[str, str]:
    """Prepare --out for appending. resume: keep (and clean) the valid lines
    already there; otherwise start empty. Returns the kept id -> line map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = load_valid(path) if resume else {}
    _atomic_write(path, kept.values())
    return kept


def append_line(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def finalize_order(path: Path, input_ids: list[str]) -> None:
    """If a resumed file isn't in input order, rewrite it atomically so it is."""
    lines = load_valid(path)
    wanted = [i for i in input_ids if i in lines]
    if list(lines) != wanted:
        _atomic_write(path, (lines[i] for i in wanted))


def run_in_order(todo: list[T], compute: Callable[[T], R], sink: Callable[[T, R], None], workers: int) -> None:
    """Run `compute` on a thread pool; call `sink(item, result)` in `todo`
    order as soon as each result (and every earlier one) is ready. If anything
    raises (including KeyboardInterrupt in a worker), queued work is cancelled
    and the exception propagates; everything sunk so far is already on disk."""
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(compute, item) for item in todo]
        for item, fut in zip(todo, futures):
            sink(item, fut.result())
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
