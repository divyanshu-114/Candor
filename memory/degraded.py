"""Loud failure reporting: which stages fell back, for how many questions, and why it matters.

WHY: v1 degraded quietly (a log line per question). A reviewer running without a key, or on a provider that
rejected a stage, saw plausible-looking output with no hint that half the pipeline had been skipped. A run now ends
with a banner naming every degraded stage and how many questions it touched; `--strict` turns any degradation into a
non-zero exit code. Default behaviour is unchanged otherwise: the run still completes.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

STAGE_HELP = {
    "analysis": "query analysis fell back to heuristics (no key, quota exhausted, or an invalid reply)",
    "hop2": "second-hop search skipped",
    "rerank": "LLM rerank skipped; the local (BM25 + dense + cross-encoder) order was used",
    "writer": "answer writer unavailable; an extractive answer or an abstention was used instead",
    "cross_encoder": "local cross-encoder unavailable (model missing offline); the fused first-stage order was used",
    "frozen_cache_miss": "a model request was not cached and its role is frozen (LLM_FROZEN_ROLES); it was not sent",
    "llm_unavailable": "every configured provider failed or was exhausted for this request",
    "crash": "the question raised an internal error; a safe baseline result was returned",
    "actions_plan": "action planning call unavailable; deterministic rules decided",
    "actions_followup": "action follow-up call unavailable; deterministic rules decided",
    "actions_validation": "the planned actions failed validation twice; a safe clarify was returned",
}


def summarize(diags: Iterable[dict]) -> tuple[Counter, int]:
    """(stage -> number of questions that degraded in it, number of questions seen)."""
    counts: Counter = Counter()
    n = 0
    for d in diags:
        n += 1
        for stage in set(d.get("degraded_stages") or []):
            counts[stage] += 1
    return counts, n


def banner(counts: Counter, total: int, diagnostics_path: str, label: str = "questions") -> str:
    if not counts:
        return f"[degraded] none: every stage ran for all {total} {label}."
    width = max(len(s) for s in counts)
    lines = ["", "=" * 78, f"DEGRADED STAGES ({total} {label} in this run)  -- results are valid but weaker than a full run", "=" * 78]
    for stage, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {stage:<{width}}  {n:>4}/{total} {label}  {STAGE_HELP.get(stage, 'stage fell back')}")
    lines.append(f"  per-{label[:-1]} detail: {diagnostics_path}   (use --strict to fail the run on any degradation)")
    lines.append("=" * 78)
    return "\n".join(lines)
