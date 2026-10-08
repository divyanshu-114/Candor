"""Compare local embedding models and cross-encoder rerankers on our eval files (no LLM, no key).

  .venv/bin/python scripts/model_bench.py embed BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 ...
  .venv/bin/python scripts/model_bench.py rerank Xenova/ms-marco-MiniLM-L-6-v2 jinaai/jina-reranker-v1-turbo-en

For embeddings it embeds every index chunk once (cached in .cache/bench), then scores DENSE-ONLY and HYBRID
(BM25 + dense, RRF) retrieval: share of answerable questions whose every needed group is in the top 10 / 20,
and the share of needed groups found at 20. For rerankers it takes the hybrid top-60 of each question
and measures the same numbers after reordering the 60 with the cross-encoder.
Reports download size (from fastembed's catalogue) and seconds on this machine. Eval files used: train, dev,
v2_dev (tuning) -- never the holdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import config  # noqa: E402
from memory.retrieve import _resources  # noqa: E402

FILES = ["memory_train.jsonl", "memory_dev.jsonl", "v2_dev.jsonl"]
QUERY_PREFIX = {
    "BAAI/bge-base-en-v1.5": "", "BAAI/bge-small-en-v1.5": "",
    "snowflake/snowflake-arctic-embed-m": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1.5-Q": "search_query: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
}
DOC_PREFIX = {"nomic-ai/nomic-embed-text-v1.5-Q": "search_document: ", "nomic-ai/nomic-embed-text-v1.5": "search_document: "}


def load_questions() -> list[dict]:
    rows = []
    for f in FILES:
        rows += [json.loads(l) for l in (ROOT / "evals" / f).read_text().splitlines() if l.strip() and json.loads(l)["answerable"] and json.loads(l).get("needed")]
    return rows


def group_hits(ids: list[str], groups: list[list[str]], k: int) -> list[bool]:
    top = set(ids[:k])
    tops_rec = {i.split("#")[0] for i in top}
    return [any(m in top or m in tops_rec for m in g) for g in groups]


def summarize(name: str, ranked: dict[str, list[str]], rows: list[dict]) -> dict:
    out = {}
    for k in (10, 20):
        hits = [group_hits(ranked[r["id"]], r["needed"], k) for r in rows]
        out[f"complete@{k}"] = sum(all(h) for h in hits) / len(rows)
        out[f"groups@{k}"] = sum(sum(h) for h in hits) / sum(len(h) for h in hits)
    print(f"  {name:<58} " + "  ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def embed_docs(model_name: str, texts: list[str]) -> tuple[np.ndarray, float]:
    cache = ROOT / ".cache" / "bench"
    cache.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "__")
    path = cache / f"{slug}.npy"
    if path.exists():
        return np.load(path), 0.0
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=model_name)
    prefix = DOC_PREFIX.get(model_name, "")
    t0 = time.monotonic()
    vecs = np.asarray(list(model.embed([prefix + t for t in texts], batch_size=64)), dtype="float32")
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    np.save(path, vecs)
    return vecs, time.monotonic() - t0


def bench_embed(models: list[str]) -> None:
    from fastembed import TextEmbedding
    sizes = {m["model"]: m.get("size_in_GB", 0) for m in TextEmbedding.list_supported_models()}
    store, index = _resources("./data", ".cache")
    rows = load_questions()
    visible = {r["id"]: {u.id for u in store.visible(r["as_of"])} for r in rows}
    parents = [d.parent_id for d in index.docs]
    short = [d.short_meeting for d in index.docs]
    texts = [d.dense_text for d in index.docs]
    print(f"{len(rows)} answerable questions with needed groups; {len(texts)} chunks")
    lex = {r["id"]: index.bm25(r["question"], visible[r["id"]], 60) for r in rows}
    for name in models:
        vecs, secs = embed_docs(name, texts)
        from fastembed import TextEmbedding as TE
        model = TE(model_name=name)
        qp = QUERY_PREFIX.get(name, "")
        dense, hybrid = {}, {}
        for r in rows:
            q = np.asarray(next(model.embed([qp + r["question"]])), dtype="float32")
            q /= max(float(np.linalg.norm(q)), 1e-12)
            sims = vecs @ q
            best: dict[str, float] = {}
            for i in np.argsort(-sims):
                if parents[i] in visible[r["id"]] and not short[i]:
                    best[parents[i]] = max(best.get(parents[i], -1), float(sims[i]))
                if len(best) >= 60:
                    break
            d = sorted(best.items(), key=lambda x: (-x[1], x[0]))
            dense[r["id"]] = [i for i, _ in d]
            fused: dict[str, float] = {}
            for ranking in (lex[r["id"]], d):
                for rank, (uid, _s) in enumerate(ranking, 1):
                    fused[uid] = fused.get(uid, 0) + 1 / (config.RRF_K + rank)
            hybrid[r["id"]] = [i for i, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]
        print(f"\n{name}: download {sizes.get(name, '?')} GB, embedding {secs:.0f}s for {len(texts)} chunks, dim {vecs.shape[1]}")
        summarize("dense only", dense, rows)
        summarize("hybrid (BM25 + dense, RRF)", hybrid, rows)


def bench_rerank(models: list[str]) -> None:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    sizes = {m["model"]: m.get("size_in_GB", 0) for m in TextCrossEncoder.list_supported_models()}
    store, index = _resources("./data", ".cache")
    rows = load_questions()
    from memory.retrieve import hybrid_search
    pools = {}
    for r in rows:
        vis = {u.id for u in store.visible(r["as_of"])}
        pools[r["id"]] = [u for u, _ in index_hybrid(index, r["question"], vis)]
    print(f"{len(rows)} questions; pool = hybrid top-60")
    summarize("hybrid order (no rerank)", pools, rows)
    for name in models:
        t0 = time.monotonic()
        enc = TextCrossEncoder(model_name=name)
        load_s = time.monotonic() - t0
        ranked, t_score = {}, 0.0
        for r in rows:
            docs = [(" ".join(store.get(u).text.split()[:180])) for u in pools[r["id"]]]
            t1 = time.monotonic()
            scores = list(enc.rerank(r["question"], docs))
            t_score += time.monotonic() - t1
            order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
            ranked[r["id"]] = [pools[r["id"]][i] for i in order]
        print(f"\n{name}: download {sizes.get(name, '?')} GB, load {load_s:.1f}s, {t_score / len(rows):.2f}s per question (60 docs)")
        summarize("after cross-encoder", ranked, rows)


def index_hybrid(index, question: str, visible: set[str]) -> list[tuple[str, float]]:
    lex = index.bm25(question, visible, 60)
    dense = index.dense(question, visible, 60)
    fused: dict[str, float] = {}
    for ranking in (lex, dense):
        for rank, (uid, _s) in enumerate(ranking, 1):
            fused[uid] = fused.get(uid, 0) + 1 / (config.RRF_K + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:60]


if __name__ == "__main__":
    mode, models = sys.argv[1], sys.argv[2:]
    (bench_embed if mode == "embed" else bench_rerank)(models)
