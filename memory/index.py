"""Cached BM25 and fastembed index, built once over all safely ingested units."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from memory import config
from memory.models import Unit

LOG = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\$?\d+(?:\.\d+)?|[a-z0-9]+", re.I)
MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)>")


def stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"): return token[:-3]
    if len(token) > 4 and token.endswith("ed"): return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"): return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t.lower()) for t in TOKEN_RE.findall(text) if t.lower() not in config.STOPWORDS]


def _date_forms(value: datetime | str | None) -> str:
    if not value: return ""
    if isinstance(value, str):
        try: value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError: return value
    return f"{value:%Y-%m-%d} {value:%b} {value.day} {value:%B} {value.day} {value:%A}"


def _names(value: str | list[str], email_names: dict[str, str] | None = None) -> str:
    entries = [value] if isinstance(value, str) else value
    result = []
    for entry in entries:
        found = re.match(r"([^<]+)\s*<([^>]+)>", entry)
        if found: result.append(f"{found.group(1).strip()} {found.group(2)}")
        else:
            name = (email_names or {}).get(entry.lower())
            result.append(f"{name} {entry}" if name else entry)
    return " ".join(result)


def _expand_identities(text: str, user_names: dict[str, str], email_names: dict[str, str]) -> str:
    """Add human-readable aliases to index text without changing source content."""
    text = MENTION_RE.sub(lambda m: f"{m.group(0)} {user_names.get(m.group(1), m.group(1))}", text)
    for address, name in email_names.items():
        if address.lower() in text.lower():
            text += f" {name}"
    return text


def search_text(unit: Unit, old_text: bool = False) -> str:
    """Metadata-rich retrieval representation; ``Unit.text`` remains display-safe."""
    base = " ".join(x for x in (unit.title or "", unit.speaker or "unidentified speaker", unit.text) if x)
    if old_text:
        return base
    meta = unit.meta
    base = _expand_identities(base, meta.get("slack_user_names", {}), meta.get("email_names", {}))
    source = {"slack": "Slack", "email": "email", "calendar": "calendar", "meeting": "meeting",
              "dictation": "dictation", "codex": "Codex", "chatgpt": "ChatGPT"}.get(unit.source, unit.source)
    extras = [source, _date_forms(unit.time)]
    if unit.source == "dictation":
        extras += [str(meta.get("target_app", "")), str(meta.get("target_context", "")), str(meta.get("delivery_state", ""))]
    elif unit.source == "email":
        names = meta.get("email_names", {})
        extras += [_names(meta.get("from", ""), names), _names(meta.get("to", []), names), _names(meta.get("cc", []), names), unit.title or ""]
    elif unit.source == "calendar":
        names = meta.get("email_names", {})
        # resolve attendee emails to real names where possible
        start_val = meta.get("start", "")
        extras += [str(meta.get("location", "")), _names(meta.get("organizer", ""), names),
                   _names(meta.get("attendees", []), names), str(meta.get("status", "")),
                   str(start_val), str(meta.get("end", "")), _date_forms(start_val)]
    elif unit.source == "meeting":
        if not unit.speaker:
            extras.append("unidentified speaker")
    elif unit.source == "codex":
        extras += [str(meta.get("repo", "")), str(meta.get("cwd", ""))]
    elif unit.source == "slack":
        # channel name is unit.title in slack; also redundant but helps channeled queries
        if unit.title:
            extras.append(unit.title)
    return " ".join(str(x) for x in extras + [base] if x)


@dataclass(frozen=True)
class IndexedChunk:
    parent_id: str
    bm25_text: str
    dense_text: str
    tokens: tuple[str, ...]
    short_meeting: bool


class MemoryIndex:
    def __init__(self, units: Sequence[Unit], cache_dir: str | Path = ".cache", dense_model: str | None = None) -> None:
        self.units, self.cache_dir = list(units), Path(cache_dir)
        self.dense_model = dense_model or config.DENSE_MODEL
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.content_hash = self._hash()
        self.docs = self._load_or_build_docs()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, doc in enumerate(self.docs):
            self.lengths.append(len(doc.tokens))
            for token, count in Counter(doc.tokens).items(): self.postings[token].append((i, count))
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self._model = None
        self._vectors = None

    def _hash(self) -> str:
        digest = hashlib.sha256(f"enriched={config.USE_ENRICHED_TEXT};v5".encode())
        for u in self.units:
            digest.update("\x1f".join((u.id, search_text(u, not config.USE_ENRICHED_TEXT))).encode())
        return digest.hexdigest()[:20]

    def _load_or_build_docs(self) -> list[IndexedChunk]:
        path = self.cache_dir / f"index-{self.content_hash}.json"
        if path.exists():
            return [IndexedChunk(x["parent_id"], x["bm25_text"], x["dense_text"], tuple(x["tokens"]), x["short_meeting"])
                    for x in json.loads(path.read_text())]
        docs: list[IndexedChunk] = []
        by_record: dict[str, list[Unit]] = defaultdict(list)
        for unit in self.units: by_record[unit.record_id].append(unit)
        for unit in self.units:
            base = search_text(unit, not config.USE_ENRICHED_TEXT)
            neighbors = by_record[unit.record_id]
            pos = neighbors.index(unit) if unit.source == "meeting" else -1
            context = ""
            if pos >= 0:
                context = " ".join(
                    search_text(x, not config.USE_ENRICHED_TEXT)
                    for x in neighbors[max(0, pos - 1):pos] + neighbors[pos + 1:pos + 2]
                )
            words = base.split()
            chunks = [words] if len(words) <= config.LONG_UNIT_WORDS else [words[i:i + config.CHUNK_WORDS] for i in range(0, len(words), config.CHUNK_WORDS - config.CHUNK_OVERLAP_WORDS)]
            for words_chunk in chunks:
                lexical = " ".join(words_chunk)
                dense = f"{context} {lexical}".strip() if unit.source == "meeting" else lexical
                docs.append(IndexedChunk(unit.id, lexical, dense, tuple(tokenize(lexical)),
                                         unit.source == "meeting" and len(unit.text.split()) < config.SHORT_MEETING_WORDS))
        path.write_text(json.dumps([d.__dict__ | {"tokens": list(d.tokens)} for d in docs], separators=(",", ":")))
        return docs

    @property
    def vector_path(self) -> Path:
        # The default model keeps its v1 file name so an existing cache stays valid; any other
        # model gets its own file (vectors from different models must never be mixed).
        if self.dense_model == "BAAI/bge-small-en-v1.5":
            return self.cache_dir / f"vectors-{self.content_hash}.npy"
        return self.cache_dir / f"vectors-{self.content_hash}-{self.dense_model.replace('/', '__')}.npy"

    def warmup(self) -> tuple[int, int, float]:
        """Download/load model and cache all vectors; intentionally no short timeout."""
        import numpy as np
        started = time.monotonic()
        if self.vector_path.exists():
            vectors = np.load(self.vector_path)
            self._vectors = vectors
            return vectors.shape[1], vectors.shape[0], time.monotonic() - started
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=self.dense_model)
        print(f"Embedding {len(self.docs)} chunks with {self.dense_model}...", flush=True)
        doc_prefix = config.DENSE_DOC_PREFIX.get(self.dense_model, "")
        batches = []
        batch_size = 64
        for start in range(0, len(self.docs), batch_size):
            texts = [doc_prefix + doc.dense_text for doc in self.docs[start:start + batch_size]]
            batches.extend(self._model.embed(texts, batch_size=batch_size))
            print(f"  embedded {min(start + batch_size, len(self.docs))}/{len(self.docs)}", flush=True)
        vectors = np.asarray(batches, dtype="float32")
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        np.save(self.vector_path, vectors)
        self._vectors = vectors
        return vectors.shape[1], vectors.shape[0], time.monotonic() - started

    def bm25_scores(self, question: str, visible_ids: set[str]) -> dict[str, float]:
        """BM25 score of every visible record that shares a token with the query (unsorted, unlimited)."""
        scores: dict[str, float] = {}
        for token in dict.fromkeys(tokenize(question)):
            posting = self.postings.get(token, [])
            if not posting: continue
            idf = math.log(1 + (len(self.docs) - len(posting) + .5) / (len(posting) + .5))
            for doc_i, tf in posting:
                doc = self.docs[doc_i]
                if doc.parent_id not in visible_ids: continue
                if doc.short_meeting and doc.bm25_text.lower() not in question.lower(): continue
                dl = self.lengths[doc_i]
                score = idf * tf * (config.BM25_K1 + 1) / (tf + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * dl / self.avgdl))
                scores[doc.parent_id] = scores.get(doc.parent_id, 0) + score
        return scores

    def bm25(self, question: str, visible_ids: set[str], limit: int) -> list[tuple[str, float]]:
        scores = self.bm25_scores(question, visible_ids)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:limit]

    def _load_dense(self) -> None:
        import numpy as np
        if self._vectors is None:
            if not self.vector_path.exists(): raise RuntimeError("embedding cache missing; run `python -m memory.cli warmup`")
            self._vectors = np.load(self.vector_path)
            self._parents = [d.parent_id for d in self.docs]
            self._skip = np.array([d.short_meeting for d in self.docs])
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.dense_model)

    def dense_scores(self, question: str, visible_ids: set[str]) -> dict[str, float]:
        """Best cosine similarity per visible record (all of them, unsorted). Returns {} when dense is
        unavailable so callers degrade to BM25 only."""
        if not config.USE_DENSE: return {}
        try:
            import numpy as np
            self._load_dense()
            query = np.asarray(next(self._model.embed([config.DENSE_QUERY_PREFIX.get(self.dense_model, "") + question])), dtype="float32")
            query /= max(float(np.linalg.norm(query)), 1e-12)
            sims = self._vectors @ query
            best: dict[str, float] = {}
            for i in range(len(sims)):
                if self._skip[i]: continue
                pid = self._parents[i]
                if pid in visible_ids and sims[i] > best.get(pid, -2.0): best[pid] = float(sims[i])
            return best
        except Exception as exc:
            LOG.warning("Dense retrieval unavailable; continuing with BM25 only: %s", exc)
            return {}

    def dense(self, question: str, visible_ids: set[str], limit: int) -> list[tuple[str, float]]:
        return sorted(self.dense_scores(question, visible_ids).items(), key=lambda x: (-x[1], x[0]))[:limit]
