"""Arithmetic the writer must NOT do itself.

WHY: language models miscount days ("Sep 10 to Sep 16" -> 5 or 7). The model only extracts the two dates and
names the operation; Python computes the number and substitutes it for the {{result}} placeholder.
"""
from __future__ import annotations

from datetime import date

PLACEHOLDER = "{{result}}"


def _d(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def compute(spec: dict | None) -> int | None:
    """days_between(a, b) = |b - a| in whole days; count(items) = len(items). None when the spec is unusable."""
    if not isinstance(spec, dict):
        return None
    op = spec.get("op")
    if op == "days_between":
        a, b = _d(spec.get("a", "")), _d(spec.get("b", ""))
        return abs((b - a).days) if a and b else None
    if op == "count" and isinstance(spec.get("items"), list):
        return len({str(i).strip().lower() for i in spec["items"] if str(i).strip()})
    return None


def apply(answer: str, spec: dict | None) -> str:
    """Replace {{result}} in `answer` with the computed number (left untouched if the spec is unusable)."""
    value = compute(spec)
    if value is None:
        return answer.replace(PLACEHOLDER, "").replace("  ", " ").strip() if PLACEHOLDER in answer else answer
    return answer.replace(PLACEHOLDER, str(value))
