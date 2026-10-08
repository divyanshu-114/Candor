"""Soft quote matching: is a writer's support quote really in the evidence?

WHY: v1 required the quote to be a verbatim substring after punctuation
normalisation. One dropped filler word ("let's") turned a correct answer into
"I don't know". This matcher keeps the safety property that matters (the facts
in the quote must really be in the record) while forgiving harmless drift:

  * order-preserving match on CONTENT words (fillers ignored) with a threshold
    (default 85% of the quote's content words)
  * every number, date, amount and capitalised name in the quote must appear
    EXACTLY in the record, so a changed figure can never pass.

Pure functions, no I/O, deterministic.
"""
from __future__ import annotations

import re

QUOTE_THRESHOLD = 0.85

# Small filler words that carry no fact. Kept deliberately short: domain words
# (negations, modals) must still be matched.
FILLER = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "that", "this", "it", "is",
    "are", "was", "were", "be", "been", "so", "just", "like", "um", "uh", "oh", "well", "okay", "ok", "yeah",
    "i", "we", "you", "let", "lets", "s", "ll", "re", "ve", "d", "my", "our", "your", "as", "by", "from",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def words(text: str) -> list[str]:
    """Lowercased tokens; keeps numbers like 61, 1.8, 187,406 whole and splits apostrophes."""
    text = text.lower().replace("’", "'").replace("'", " ")
    return _TOKEN_RE.findall(text)


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in FILLER]


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (order-preserving, not contiguous)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def exact_facts(quote: str) -> list[str]:
    """Tokens that must appear verbatim in the record: anything with a digit, and
    capitalised words that are not the first word of the quote (names)."""
    facts = [w for w in words(quote) if any(c.isdigit() for c in w)]
    raw = re.findall(r"[A-Za-z][A-Za-z'’-]*", quote)
    for i, tok in enumerate(raw):
        if i > 0 and tok[0].isupper() and not tok.isupper() and tok.lower() not in FILLER:
            facts.append(tok.lower().replace("'", " ").split()[0])
    return facts


def soft_quote_match(quote: str, record_text: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    q = content_words(quote)
    if not q:
        return False
    record_words = words(record_text)
    record_set = set(record_words)
    if any(f not in record_set for f in exact_facts(quote)):
        return False
    return _lcs_len(q, [w for w in record_words if w not in FILLER]) / len(q) >= threshold


def exact_quote_match(quote: str, record_text: str) -> bool:
    """v1 rule: normalised substring."""
    nq = " ".join(words(quote))
    return bool(nq) and nq in " ".join(words(record_text))
