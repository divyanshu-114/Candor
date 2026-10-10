"""Version chains for facts that change ("moved", "actually", "correction").

WHY: the answer to "when is X?" is the NEWEST decisive statement, but a later correction often shares few
words with the question ("Sorry, 61 not 60", "Moved to Fri 10am"). Retrieval by similarity to the question
finds the original statement and misses the fix. After the first pass we therefore expand the top
candidates with (a) records in the same container (email thread, Slack thread / neighbouring messages in the
same channel, neighbouring meeting segments) and (b) records that share the question's rare entities AND
carry a change cue. The result is a time-ordered chain of every version visible at `as_of`; the writer
(memory/answer.py) is told to use the latest and mention earlier values only as history.

Pure ranking logic over visible ids -- nothing after `as_of` can enter.
"""
from __future__ import annotations

import re
from datetime import timedelta

from memory import config
from memory.corpus import get_info
from memory.index import tokenize
from memory.queries import QUESTION_WORDS
from memory.store import MemoryStore

CHANGE_CUES = re.compile(
    r"\b(moved?|moving|push(?:ed|ing)?|slip(?:ped|s|ping)?|delay(?:ed|s)?|resched\w*|actually|correct(?:ion|ed|s)?|"
    r"updated?|instead|revis\w*|cancel\w*|postpon\w*|bump(?:ed)?|scratch\w*|changed?|no longer|never ?mind|"
    r"my bad|sorry|(?:is|are) now|now (?:targets?|is|are)|not \w+ after all|ignore (?:that|the)|deleted|wait,)\b", re.I)
SAME_CHANNEL_WINDOW = timedelta(minutes=90)
SAME_CHANNEL_SPAN = 3


def container_neighbors(seed: str, visible: set[str], store: MemoryStore) -> list[str]:
    """Visible units that live next to `seed`: same email thread / Slack thread / neighbouring segments or messages."""
    info = get_info(store)
    u = store.get(seed)
    out: list[str] = []
    if u.source == "email" and u.thread_key:
        out = [i for i in info.thread_units.get(u.thread_key, []) if i in visible]
    elif u.source == "meeting":
        ids = info.record_units[u.record_id]
        pos = info.position[seed]
        out = [i for i in ids[max(0, pos - 3): pos + 4] if i in visible]
    elif u.source == "slack" and u.thread_key:
        thread = [i for i in info.thread_units.get(u.thread_key, []) if i in visible]
        if u.thread_key.startswith("SL-"):          # a real thread: take it whole
            out = thread
        else:                                       # a channel: only the few messages around the seed in time
            near = [i for i in thread if abs(store.get(i).time - u.time) <= SAME_CHANNEL_WINDOW]
            pos = near.index(seed) if seed in near else 0
            out = near[max(0, pos - SAME_CHANNEL_SPAN): pos + SAME_CHANNEL_SPAN + 1]
    elif u.source == "chatgpt":
        ids = info.record_units[u.record_id]
        pos = info.position[seed]
        out = [i for i in ids[max(0, pos - 1): pos + 2] if i in visible]
    return [i for i in out if i != seed]


def entity_tokens(question: str, idf: dict[str, float], min_idf: float = 2.0) -> list[str]:
    """Rare content tokens of the question (stemmed like the index), best-first."""
    toks = [t for t in dict.fromkeys(tokenize(question)) if t not in QUESTION_WORDS and idf.get(t, 0) >= min_idf]
    return sorted(toks, key=lambda t: (-idf[t], t))


def change_records(question: str, visible: set[str], store: MemoryStore, idf: dict[str, float], limit: int) -> list[str]:
    """Visible records that mention >= 2 of the question's rare tokens (>= 1 if it has only one) and carry a change cue."""
    ents = entity_tokens(question, idf)[:8]
    if not ents:
        return []
    need = 1 if len(ents) == 1 else 2
    scored = []
    for uid in visible:
        u = store.get(uid)
        if u.source in ("meeting",) and len(u.text.split()) < 6:
            continue
        if not CHANGE_CUES.search(u.text):
            continue
        toks = set(tokenize(u.text + " " + (u.title or "")))
        hit = [e for e in ents if e in toks]
        if len(hit) >= need:
            scored.append((-sum(idf[e] for e in hit), -u.time.timestamp(), uid))
    scored.sort()
    return [uid for _s, _t, uid in scored[:limit]]


def expand(seeds: list[str], question: str, visible: set[str], store: MemoryStore, idf: dict[str, float],
           max_new: int = 8) -> tuple[list[str], list[str]]:
    """(new candidate ids newest-first, the whole chain oldest-first including the seeds)."""
    seen = set(seeds)
    new: list[str] = []
    for seed in seeds[: config.CHAIN_SEEDS]:
        for uid in container_neighbors(seed, visible, store):
            if uid not in seen and CHANGE_CUES.search(store.get(uid).text or ""):
                seen.add(uid)
                new.append(uid)
    for uid in change_records(question, visible, store, idf, limit=max_new):
        if uid not in seen:
            seen.add(uid)
            new.append(uid)
    new = sorted(new, key=lambda i: (store.get(i).time, i), reverse=True)[:max_new]
    chain = sorted(set(seeds[: config.CHAIN_SEEDS]) | set(new), key=lambda i: (store.get(i).time, i))
    return new, chain


def context_neighbors(seeds: list[str], visible: set[str], store: MemoryStore, span: int = 2, max_new: int = 16) -> list[str]:
    """Records right next to the best candidates, with NO change-cue filter: the segments just before/after a meeting hit,
    the messages around a Slack hit, the turns around a ChatGPT hit.

    WHY: in a transcript the question and its answer are often adjacent segments ("What are the three?" / "One is the CSV
    export ...") and only one of them matches the question's words. Order: closest to the best seed first."""
    info = get_info(store)
    out: list[str] = []
    seen = set(seeds)
    for seed in seeds:
        u = store.get(seed)
        if u.source not in ("meeting", "chatgpt"):
            continue
        ids = info.record_units[u.record_id]
        pos = info.position[seed]
        for off in range(1, span + 1):
            for j in (pos + off, pos - off):
                if 0 <= j < len(ids) and ids[j] in visible and ids[j] not in seen and len(store.get(ids[j]).text.split()) >= 4:
                    seen.add(ids[j])
                    out.append(ids[j])
        if len(out) >= max_new:
            break
    return out[:max_new]
