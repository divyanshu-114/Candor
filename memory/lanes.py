"""Per-source retrieval lanes and reciprocal-rank fusion.

WHY: v1 fused BM25 and dense over ALL sources and kept the top ~30, so one chatty source (870 meeting
segments) crowded out the single Codex session or ChatGPT message that held the answer. A lane is a
source-restricted ranking: the best meeting segments, the best Slack messages, the best emails ... each
get their own top-k from BM25 and from dense search, for several query variants. Fusing ALL lane
rankings with reciprocal-rank fusion gives every source an equal chance (rank 1 in any lane scores the
same) and builds a wide, diverse pool (~150) for the cross-encoder to cut.

Pure functions over precomputed score maps; no I/O, deterministic (ties broken by id).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from memory import config

SOURCES = ("meeting", "slack", "email", "calendar", "dictation", "codex", "chatgpt")


def top_k(scores: Mapping[str, float], ids: Iterable[str], k: int) -> list[str]:
    """Best `k` ids from `ids` by score (id order breaks ties)."""
    ranked = sorted(((scores[i], i) for i in ids if i in scores), key=lambda t: (-t[0], t[1]))
    return [i for _s, i in ranked[:k]]


def lane_rankings(bm25: Mapping[str, float], dense: Mapping[str, float], source_of: Mapping[str, str],
                  k: int) -> dict[str, list[list[str]]]:
    """{lane: [bm25 top-k, dense top-k]} for one query. A lane with no hits has empty rankings."""
    members: dict[str, list[str]] = defaultdict(list)
    for uid in set(bm25) | set(dense):
        members[source_of.get(uid, "other")].append(uid)
    out: dict[str, list[list[str]]] = {}
    for lane in SOURCES:
        ids = members.get(lane, [])
        out[lane] = [top_k(bm25, ids, k), top_k(dense, ids, k)]
    return out


def fuse(rankings: Iterable[list[str]], rrf_k: int = config.RRF_K, weights: Iterable[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of ranked id lists. Sorted by score desc, then id."""
    fused: dict[str, float] = {}
    w = list(weights) if weights is not None else None
    for n, ranking in enumerate(rankings):
        weight = w[n] if w is not None else 1.0
        for rank, uid in enumerate(ranking, 1):
            fused[uid] = fused.get(uid, 0.0) + weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda t: (-t[1], t[0]))


def build_pool(per_query: list[tuple[dict[str, float], dict[str, float]]], source_of: Mapping[str, str],
               lane_k: int, pool_size: int, query_weights: list[float] | None = None
               ) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    """Fuse every (query variant x lane x method) ranking into one pool.

    Returns (pool sorted best-first and cut to `pool_size`, lane_rank) where lane_rank[uid] = {lane: best
    rank of uid inside that lane's rankings} for coverage reporting. The pool keeps at least a few
    candidates from every lane that produced any (so a rare source is never fully cut off).
    """
    rankings: list[list[str]] = []
    weights: list[float] = []
    lane_rank: dict[str, dict[str, int]] = defaultdict(dict)
    for qi, (bm25, dense) in enumerate(per_query):
        qw = query_weights[qi] if query_weights else 1.0
        for lane, rks in lane_rankings(bm25, dense, source_of, lane_k).items():
            for ranking in rks:
                rankings.append(ranking)
                weights.append(qw)
                for r, uid in enumerate(ranking, 1):
                    prev = lane_rank[uid].get(lane)
                    if prev is None or r < prev:
                        lane_rank[uid][lane] = r
    fused = fuse(rankings, weights=weights)
    pool = fused[:pool_size]
    chosen = {uid for uid, _ in pool}
    # lane floor: top candidates of each lane (by fused score) are always kept
    floor = config.LANE_FLOOR
    per_lane: dict[str, int] = defaultdict(int)
    extra = []
    for uid, score in fused:
        lane = source_of.get(uid, "other")
        if per_lane[lane] < floor:
            per_lane[lane] += 1
            if uid not in chosen:
                extra.append((uid, score))
                chosen.add(uid)
    return pool + extra, lane_rank
