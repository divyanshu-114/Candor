"""Safety transformations shared by ingestion and audit tooling."""
from __future__ import annotations

import re
from typing import Iterator, Tuple

# Match credential structure and keyword/value context, never generic long strings.
PREFIX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-key", re.compile(r"\b(?:sk-|sk_live)[A-Za-z0-9_-]{12,}\b", re.I)),
    ("public-key", re.compile(r"\bpk_[A-Za-z0-9_-]{12,}\b", re.I)),
    ("aws-access-key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[po]_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[bp]-[A-Za-z0-9-]{12,}\b", re.I)),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----")),
    ("connection-string", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:([^\s@]+)@", re.I)),
)
KEYWORD_VALUE = re.compile(
    r"\b(?:api[ _-]?key|apikey|key|secret|token|password|passcode|passwd|credential|bearer|login)\b"
    r"[^\n\r]{0,30}?(?::|=|\bis\b)\s*[\"']?([^\s\"';,\n\r]{3,})",
    re.I,
)
SPOKEN_VALUE = re.compile(
    # Spoken aloud: "the key is s k dash sample value" – all tokens ≤15 chars each,
    # separated by spaces only, 2–8 tokens, no newlines.
    r"\b(?:api[ _-]?key|apikey|key|secret|token|password|passcode|passwd|credential)\b[^\n\r]{0,30}?\bis\b\s+"
    r"((?:[a-zA-Z0-9]{1,15}[ -]){1,7}[a-zA-Z0-9]{1,15})(?=\s|$|[^\w-])",
    re.I,
)


def iter_secret_matches(text: str) -> Iterator[tuple[str, str]]:
    """Yield (general-rule-name, value) without printing or persisting values."""
    for name, pattern in PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            yield name, match.group(1) if name == "connection-string" else match.group(0)
    for match in KEYWORD_VALUE.finditer(text):
        value = match.group(1).strip()
        if not value.startswith(("/", "./", "../")) and not re.fullmatch(r"[a-f0-9]{32,}", value, re.I):
            yield "keyword-value", value
    for match in SPOKEN_VALUE.finditer(text):
        yield "spoken-secret", match.group(1).strip()


def mask_secrets(text: str) -> str:
    """Replace detected values while retaining surrounding useful prose.

    WHY we filter: file paths and hex hashes are common in code/logs and
    must not be masked even when preceded by a keyword like 'token:'.
    """
    spans: list[tuple[int, int]] = []
    for name, pattern in PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(1), match.end(1)) if name == "connection-string" else match.span())
    for pattern in (KEYWORD_VALUE, SPOKEN_VALUE):
        for match in pattern.finditer(text):
            value = match.group(1)
            if value.startswith(("/", "./", "../")) or re.fullmatch(r"[a-f0-9]{32,}", value, re.I):
                continue  # path or hash – not a secret
            spans.append(match.span(1))
    # Nested patterns must become one replacement.  Applying overlapping original
    # offsets would otherwise corrupt text after the first replacement.
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + "[REDACTED-SECRET]" + text[end:]
    return text


INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(previous|all|above|instructions?)", re.I), re.compile(r"\bassistant\b.*\b(you\s+must|do\s+not|never|always)\b", re.I),
    re.compile(r"\bai\s+model\b", re.I), re.compile(r"forward\s+all", re.I), re.compile(r"do\s+not\s+tell", re.I), re.compile(r"system\s+prompt", re.I),
    re.compile(r"\byou\s+must\b", re.I), re.compile(r"when\s+summariz", re.I), re.compile(r"<!--.*?-->", re.S),
    re.compile(r"display\s*:\s*none", re.I), re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]"), re.compile(r"\bdisregard\b", re.I),
)


def neutralize_injection(text: str) -> Tuple[str, bool]:
    found = False
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            found = True
            text = pattern.sub("[suspected planted instruction removed]", text)
    return text, found
