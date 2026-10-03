"""Per-question diagnostics collection.

WHY a separate module instead of threading extra return values through
retrieve()/answer_question(): those functions are called from several
places (tests, scripts, the CLI) and changing their signatures would ripple
everywhere. A thread-local recorder lets any stage (analysis, hop2, rerank,
writer, chat_json itself) report what happened without changing a single
call site's return type. Each worker thread in the CLI's ThreadPoolExecutor
handles one question at a time synchronously, so thread-local state here is
naturally scoped to "the question currently being processed by this thread"
with no cross-question bleed.
"""
from __future__ import annotations

import threading
import time
from typing import Any

_LOCAL = threading.local()


class _Recorder:
    def __init__(self) -> None:
        self.degraded: set[str] = set()
        self.calls: list[dict[str, Any]] = []
        self.pool_size: int | None = None
        self.rerank_valid: bool | None = None
        self.started = time.monotonic()


def begin() -> None:
    _LOCAL.rec = _Recorder()


def _current() -> _Recorder | None:
    return getattr(_LOCAL, "rec", None)


def mark_degraded(stage: str) -> None:
    rec = _current()
    if rec is not None:
        rec.degraded.add(stage)


def record_call(model: str, prompt_tokens: int, completion_tokens: int, retries: int,
                 cache_hit: bool, seconds: float, stage: str = "") -> None:
    rec = _current()
    if rec is not None:
        rec.calls.append({
            "model": model, "stage": stage, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "retries": retries, "cache_hit": cache_hit,
            "seconds": round(seconds, 3),
        })


def set_pool_size(n: int) -> None:
    rec = _current()
    if rec is not None:
        rec.pool_size = n


def set_rerank_valid(valid: bool) -> None:
    rec = _current()
    if rec is not None:
        rec.rerank_valid = valid


def finish(qid: str) -> dict[str, Any]:
    rec = _current()
    if rec is None:
        return {"id": qid}
    elapsed = time.monotonic() - rec.started
    total_retries = sum(c["retries"] for c in rec.calls)
    by_stage: dict[str, dict[str, int]] = {}
    for c in rec.calls:
        stage = c.get("stage") or "unlabeled"
        entry = by_stage.setdefault(stage, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        entry["prompt_tokens"] += c["prompt_tokens"]
        entry["completion_tokens"] += c["completion_tokens"]
        entry["calls"] += 1
    _LOCAL.rec = None
    return {
        "id": qid,
        "degraded_stages": sorted(rec.degraded),
        "llm_calls": rec.calls,
        "llm_retries_total": total_retries,
        "prompt_tokens": sum(c["prompt_tokens"] for c in rec.calls),
        "completion_tokens": sum(c["completion_tokens"] for c in rec.calls),
        "tokens_by_stage": by_stage,
        "seconds": round(elapsed, 3),
        "candidate_pool_size": rec.pool_size,
        "rerank_valid": rec.rerank_valid,
    }
