"""No-LLM lexical coverage gate for the no-key and extractive-fallback paths.

WHY: without an LLM there is no writer to say "this isn't in memory", and a
BM25 score alone can't tell "found something" from "found something that
doesn't answer this". Two cheap, general signals can:

1. *Unmatched mass*: the share of the question's informative vocabulary
   (idf-weighted) that appears nowhere in the records visible at as_of
   (morphological prefix matching, so "dictate" matches "dictation").
2. *Best single-record coverage*: how much of the question's idf mass the
   best of the top retrieved records contains.

We abstain only when BOTH say "not here": a lot of the question is
unmatched anywhere AND no top record covers most of it. Requiring both keeps
paraphrased-but-answerable questions (one abstract word missing, good
record found) from being wrongly refused. Pure function of (question,
visible units, retrieved ids): deterministic, no network, no question text
or ids in code. Thresholds live in memory/config.py.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from memory import config
from memory.index import search_text, tokenize
from memory.models import Unit

# Words that carry no topic: question scaffolding, pronouns, auxiliaries.
# Generic English only -- never domain words, names or dates.
FILLER = frozenset("""
say said tell told current currently now about have has had any many much how why whose whom we our us my me
i you your he she they there this that these those can could will shall should last next new get got give gave
what's does did do is are was were am be been being it its if so than then also just only very really still
again already ever yet who which where when
""".split())

_MIN_PREFIX = 4  # prefix matching never applies to tokens shorter than this


def _equiv(a: str, b: str) -> bool:
    """Same word modulo a short suffix (schedule/scheduled, dictate/dictation).
    Tokens under 4 chars must match exactly so "soc" never matches "social"."""
    if a == b:
        return True
    if len(a) < _MIN_PREFIX or len(b) < _MIN_PREFIX:
        return False
    return a.startswith(b[:max(_MIN_PREFIX, len(b) - 2)]) or b.startswith(a[:max(_MIN_PREFIX, len(a) - 2)])


def _present(term: str, vocab: set[str], by_prefix: dict[str, list[str]]) -> bool:
    if term in vocab:
        return True
    if len(term) < _MIN_PREFIX:
        return False
    return any(_equiv(term, v) for v in by_prefix.get(term[:_MIN_PREFIX], ()))


def _record_has(term: str, rec_tokens: set[str]) -> bool:
    return term in rec_tokens or (len(term) >= _MIN_PREFIX and any(_equiv(term, t) for t in rec_tokens))


def coverage_stats(question: str, visible: Sequence[Unit], retrieved_ids: Iterable[str],
                   top_n: int = 10) -> dict:
    """Return {"best_coverage", "missing_share", "missing_terms", "terms"}."""
    terms = [t for t in dict.fromkeys(tokenize(question)) if t not in FILLER]
    if not terms:
        return {"best_coverage": 1.0, "missing_share": 0.0, "missing_terms": [], "terms": []}

    toks = {u.id: set(tokenize(search_text(u))) for u in visible}
    df: Counter = Counter()
    for t in toks.values():
        df.update(t)
    vocab = set(df)
    by_prefix: dict[str, list[str]] = {}
    for v in vocab:
        if len(v) >= _MIN_PREFIX:
            by_prefix.setdefault(v[:_MIN_PREFIX], []).append(v)

    n = len(toks)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    total = sum(idf.values()) or 1.0
    missing = [t for t in terms if not _present(t, vocab, by_prefix)]

    best = 0.0
    for rid in list(retrieved_ids)[:top_n]:
        rec = toks.get(rid)
        if rec is None:
            continue
        best = max(best, sum(idf[t] for t in terms if _record_has(t, rec)) / total)
    return {"best_coverage": best, "missing_share": sum(idf[t] for t in missing) / total,
            "missing_terms": missing, "terms": terms}


def should_abstain(question: str, visible: Sequence[Unit], retrieved_ids: Iterable[str]) -> bool:
    s = coverage_stats(question, visible, retrieved_ids)
    return (s["missing_share"] >= config.COVERAGE_MISSING_SHARE_MIN
            and s["best_coverage"] < config.COVERAGE_BEST_MAX)
