"""Local cross-encoder reranker (fastembed ONNX): works with no API key.

WHY: BM25 and the embedding model score the query and a record separately; a cross-encoder reads them
together and is much better at "does this passage answer this question". It runs locally in a few seconds for
~150 short passages, so it can cut the wide pool down before any paid model sees it.

Scores are cached on disk (sqlite, keyed by model + query + passage): repeating a question, or re-tuning a
downstream weight, costs nothing, and results stay identical run to run. If the model cannot be loaded
(offline first run) we return None and the caller keeps the fused order and marks the stage degraded -- loudly,
never silently.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from memory import config

LOG = logging.getLogger(__name__)
# ONNX already uses every core for one batch; letting several worker threads score at once only oversubscribes the CPU.
_LOCK = threading.Lock()
_DB: sqlite3.Connection | None = None
_DB_PATH = ""


@lru_cache(maxsize=2)
def _load(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=model_name)


def _db() -> sqlite3.Connection | None:
    global _DB, _DB_PATH
    path = str(Path(config.CE_CACHE_PATH))
    if _DB is None or _DB_PATH != path:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            _DB = sqlite3.connect(path, check_same_thread=False)
            _DB.execute("CREATE TABLE IF NOT EXISTS ce (k TEXT PRIMARY KEY, v REAL)")
            _DB_PATH = path
        except sqlite3.Error as exc:  # a broken cache must never break retrieval
            LOG.warning("cross-encoder cache disabled: %s", exc)
            _DB = None
    return _DB


def _key(model: str, query: str, doc: str) -> str:
    return hashlib.sha1(f"{model}\x1f{query}\x1f{doc}".encode()).hexdigest()


def available() -> bool:
    try:
        _load(config.CE_MODEL)
        return True
    except Exception as exc:  # pragma: no cover - depends on network on first run
        LOG.warning("Cross-encoder %s unavailable: %s", config.CE_MODEL, exc)
        return False


def score(query: str, docs: list[str]) -> list[float] | None:
    """Relevance score per doc (higher is better), or None if the model is unavailable."""
    if not docs:
        return []
    model_name = config.CE_MODEL
    keys = [_key(model_name, query, d) for d in docs]
    with _LOCK:
        db = _db()
        cached: dict[str, float] = {}
        if db is not None:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                rows = db.execute(f"SELECT k, v FROM ce WHERE k IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                cached.update(rows)
        todo = [i for i, k in enumerate(keys) if k not in cached]
        if todo:
            try:
                got = [float(s) for s in _load(model_name).rerank(query, [docs[i] for i in todo])]
            except Exception as exc:  # pragma: no cover
                LOG.warning("Cross-encoder scoring failed: %s", exc)
                return None
            for i, s in zip(todo, got):
                cached[keys[i]] = s
            if db is not None:
                db.executemany("INSERT OR REPLACE INTO ce (k, v) VALUES (?, ?)", [(keys[i], s) for i, s in zip(todo, got)])
                db.commit()
    return [cached[k] for k in keys]
