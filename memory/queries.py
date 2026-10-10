"""Query variants for the wide first-stage pool.

WHY several variants: the question's own wording often differs from the record's ("Did John agree to cut
dark mode?" vs "fine cutting it"). One variant is the question itself; one is an ENTITY-ONLY keyword query
(names, numbers, rare words) that survives paraphrase; with a key the analysis model adds sub-queries and
an answer sketch. Everything here works without a key.
"""
from __future__ import annotations

import re

# Question scaffolding that carries no retrieval signal.
QUESTION_WORDS = frozenset("""what whats when where which who whom whose why how did does do is are was were
will would should could can i me my mine we our us you your he she they them their it its the a an and or but
of to in on at for from with by about as that this these those there here has have had having been be being
not no yes any some all many much more most still yet already just also then than so if""".split())
_WORD = re.compile(r"[A-Za-z0-9$%][A-Za-z0-9$%'\-./:]*")


def entity_query(question: str, idf: dict[str, float] | None = None, keep_fraction: float = 0.6) -> str:
    """Names, numbers, quoted phrases and the rarest remaining words of `question`, joined by spaces.

    `idf` (token -> idf from the index) lets rare words win when provided; without it we keep every
    non-scaffolding word. Order is preserved, duplicates dropped, so the result is deterministic.
    """
    tokens = _WORD.findall(question)
    keep: list[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower().strip("'.-:")
        if not low or low in QUESTION_WORDS:
            continue
        proper = tok[0].isupper() and i > 0
        number = any(c.isdigit() for c in tok)
        keep.append((tok, low, proper or number))
    if idf and len(keep) > 3:
        scored = sorted((idf.get(low, 0.0) for _t, low, flag in keep if not flag))
        if scored:
            cut = scored[int(len(scored) * (1 - keep_fraction))] if keep_fraction < 1 else 0.0
            keep = [(t, l, f) for t, l, f in keep if f or idf.get(l, 0.0) >= cut]
    seen, out = set(), []
    for tok, low, _f in keep:
        if low not in seen:
            seen.add(low)
            out.append(tok)
    return " ".join(out)


def build_variants(question: str, analysis: dict | None, idf: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """[(query text, weight)] -- the question first (weight 1.0), then the entity query, then LLM material."""
    variants: list[tuple[str, float]] = [(question, 1.0)]
    ent = entity_query(question, idf)
    if ent and ent.lower() != question.lower():
        variants.append((ent, 0.8))
    if analysis:
        extra_ents = " ".join(str(e) for e in (analysis.get("entities") or []) if e)
        if extra_ents.strip() and extra_ents.lower() not in (ent.lower(), question.lower()):
            variants.append((extra_ents, 0.6))
        for sq in (analysis.get("sub_queries") or [])[:3]:
            if sq and sq.strip().lower() != question.strip().lower() and all(sq != v for v, _ in variants):
                variants.append((str(sq), 0.7))
        sketch = (analysis.get("answer_sketch") or "").strip()
        if sketch:
            variants.append((sketch, 0.5))
    return variants[:6]
