"""Static lookup tables over the corpus, built once per store (not per question)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from memory.store import MemoryStore


@dataclass
class CorpusInfo:
    source_of: dict[str, str]
    record_units: dict[str, list[str]] = field(default_factory=dict)   # record id -> unit ids, in seq/time order
    thread_units: dict[str, list[str]] = field(default_factory=dict)   # thread key -> unit ids, in time order
    position: dict[str, int] = field(default_factory=dict)             # unit id -> index inside its record


@lru_cache(maxsize=4)
def _build(store_id: int, units_len: int, store: MemoryStore) -> CorpusInfo:
    info = CorpusInfo(source_of={u.id: u.source for u in store.units})
    for u in store.units:  # store.units is time-sorted
        info.record_units.setdefault(u.record_id, []).append(u.id)
        if u.thread_key:
            info.thread_units.setdefault(u.thread_key, []).append(u.id)
    for rid, ids in info.record_units.items():
        if store.get(ids[0]).source == "meeting":
            ids.sort(key=lambda i: store.get(i).seq or 0)
        for pos, uid in enumerate(ids):
            info.position[uid] = pos
    return info


def get_info(store: MemoryStore) -> CorpusInfo:
    return _build(id(store), len(store.units), store)
