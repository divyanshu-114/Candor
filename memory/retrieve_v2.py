"""Retrieval v2: wide per-source pool -> local cross-encoder cut -> optional LLM rerank.

Stages (each behind a flag in memory/config.py so it can be ablated, see docs/DEVLOG.md):
  1. analysis (LLM if a key exists, heuristics otherwise)
  2. query variants: question, entity-only keywords, LLM sub-queries / answer sketch
  3. LANES: per-source BM25 + dense rankings for every variant, fused by RRF into a pool of ~150
  4. extras (version chains, anchor window, person resolution, commitments ledger): guaranteed candidates
  5. local cross-encoder cut to ~40 with a per-lane quota
  6. LLM rerank of the survivors (strong model) -- or the combined local order when there is no key

Visibility is applied before ANY of this: every candidate comes from `visible_ids` (hard rule #2).
"""
from __future__ import annotations

import logging
import math

from memory import config, diagnostics
from memory import crossenc, lanes
from memory import llm as llm_module
from memory.corpus import get_info
from memory.index import MemoryIndex, tokenize
from memory.queries import build_variants
from memory.store import MemoryStore

LOG = logging.getLogger(__name__)


def _idf(index: MemoryIndex) -> dict[str, float]:
    cached = getattr(index, "_idf_cache", None)
    if cached is None:
        n = len(index.docs)
        cached = {t: math.log(1 + (n - len(p) + .5) / (len(p) + .5)) for t, p in index.postings.items()}
        index._idf_cache = cached  # type: ignore[attr-defined]
    return cached


def _entity_idf(index: MemoryIndex) -> dict[str, float]:
    """idf keyed by the lowercase surface word (what queries.entity_query looks up)."""
    return _idf(index)


def doc_text(uid: str, store: MemoryStore, words: int) -> str:
    """Passage shown to the cross-encoder: speaker + text, with the neighbouring meeting segments as context
    (a segment like "Okay, great." is meaningless alone)."""
    u = store.get(uid)
    info = get_info(store)
    head = f"{u.source} {u.title or ''} {u.speaker or ''}:".strip()
    body = u.text
    if u.source == "meeting":
        ids = info.record_units[u.record_id]
        pos = info.position[uid]
        prev_t = store.get(ids[pos - 1]).text if pos > 0 else ""
        next_t = store.get(ids[pos + 1]).text if pos + 1 < len(ids) else ""
        body = f"{prev_t} {u.text} {next_t}"
    return f"{head} {' '.join(body.split()[:words])}"


def first_stage(question: str, analysis: dict, visible_ids: set[str], store: MemoryStore, index: MemoryIndex,
                extra_variants: list[tuple[str, float]] | None = None
                ) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]], list[tuple[str, float]]]:
    """Wide fused pool. Returns (pool, lane_rank, variants)."""
    info = get_info(store)
    variants = build_variants(question, analysis, _entity_idf(index)) + list(extra_variants or [])
    per_query = []
    for text, _w in variants:
        per_query.append((index.bm25_scores(text, visible_ids), index.dense_scores(text, visible_ids)))
    pool, lane_rank = lanes.build_pool(per_query, info.source_of, config.LANE_K, config.POOL_SIZE,
                                       [w for _t, w in variants])
    return pool, lane_rank, variants


class LocalScorer:
    """Cross-encoder scores for one question, computed lazily and cached (extras found later are scored on demand)."""

    def __init__(self, question: str, store: MemoryStore, meta: dict) -> None:
        self.question, self.store, self.meta = question, store, meta
        self.scores: dict[str, float] = {}
        self.failed = not config.USE_CROSS_ENCODER

    def ensure(self, ids: list[str]) -> None:
        todo = [i for i in ids if i not in self.scores]
        if not todo or self.failed:
            return
        got = crossenc.score(self.question, [doc_text(i, self.store, config.CE_DOC_WORDS) for i in todo])
        if got is None:
            self.failed = True
            self.meta["degraded"].append("cross_encoder")
            diagnostics.mark_degraded("cross_encoder")
            return
        self.scores.update(zip(todo, got))

    def combine(self, ids: list[str], fused_rank: dict[str, int], bonus: dict[str, float]) -> dict[str, float]:
        """Rank-fusion of: cross-encoder rank over `ids`, first-stage fused rank, and a small bonus for guaranteed extras."""
        ce_rank: dict[str, int] = {}
        if not self.failed:
            order = sorted(ids, key=lambda i: (-self.scores.get(i, -1e9), i))
            ce_rank = {u: r for r, u in enumerate(order, 1)}
        out = {}
        for uid in ids:
            v = bonus.get(uid, 0.0)
            if uid in fused_rank:
                v += 1.0 / (config.RRF_K + fused_rank[uid])
            if uid in ce_rank:
                v += config.CE_WEIGHT / (config.RRF_K + ce_rank[uid])
            out[uid] = v
        return out


def pick_survivors(ids: list[str], combined: dict[str, float], store: MemoryStore) -> list[str]:
    """Best CE_KEEP by combined score, plus the best CE_LANE_QUOTA of every source so no source is crowded out."""
    ordered = sorted(ids, key=lambda u: (-combined[u], u))
    source_of = get_info(store).source_of
    keep = ordered[: config.CE_KEEP]
    chosen = set(keep)
    per_lane: dict[str, int] = {}
    for uid in ordered:
        lane = source_of.get(uid, "other")
        if per_lane.get(lane, 0) < config.CE_LANE_QUOTA:
            per_lane[lane] = per_lane.get(lane, 0) + 1
            if uid not in chosen:
                keep.append(uid)
                chosen.add(uid)
    return sorted(keep, key=lambda u: (-combined[u], u))


def _llm_rerank(question: str, as_of: str, analysis: dict, cand_ids: list[str], store: MemoryStore, meta: dict
                ) -> list[str] | None:
    """Same prompt and JSON contract as v1's rerank (memory/prompts.py), over the survivors of the local cut."""
    from memory.llm import chat_json
    snippets = []
    for uid in cand_ids:
        u = store.get(uid)
        if u:
            text = " ".join(u.text.split()[:config.RERANK_V2_SNIPPET_WORDS])
            snippets.append(f"ID: {uid} | Source: {u.source} | Time: {u.time} | Speaker: {u.speaker} | Text: {text}")
    if not snippets:
        return None
    from memory.prompts import RERANK_SYSTEM
    user_prompt = (f"Question: {question}\nAs of: {as_of}\nIntent: {analysis['intent']}\n"
                   f"Sub-queries: {analysis['sub_queries']}\nwants_latest: {analysis['wants_latest']}\n\nCandidates:\n"
                   + "\n".join(snippets))
    try:
        res = chat_json(RERANK_SYSTEM, user_prompt, llm_module.get_model_strong(),
                        max_tokens=config.MAX_TOKENS_RERANK, reasoning_effort=config.REASONING_EFFORT_LOW, stage="rerank")
    except Exception as exc:  # a stage must degrade, never raise
        LOG.warning("rerank raised (treated as degraded): %s", exc)
        res = None
    diagnostics.set_rerank_valid(bool(res and isinstance(res.get("ranked"), list)))
    if not (res and isinstance(res.get("ranked"), list)):
        return None
    meta["rerank_reason"] = str(res.get("reason", ""))
    seen, ordered = set(), []
    for uid in res["ranked"]:
        if uid in cand_ids and uid not in seen:
            ordered.append(uid)
            seen.add(uid)
    ordered += [uid for uid in cand_ids if uid not in seen]  # the model dropped some: keep local order for them
    return ordered


def _ranked_subset(query: str, ids: set[str], index: MemoryIndex, limit: int) -> list[str]:
    """Hybrid (BM25 + dense, RRF) ranking of `query` restricted to `ids`."""
    if not ids:
        return []
    bm, de = index.bm25_scores(query, ids), index.dense_scores(query, ids)
    fused = lanes.fuse([lanes.top_k(bm, ids, limit), lanes.top_k(de, ids, limit)])
    return [uid for uid, _ in fused[:limit]]


def _strip_anchor(question: str, anchor_query: str) -> str:
    out = question.replace(anchor_query, " ")
    return " ".join(out.split()) or question


def anchor_extras(question: str, analysis: dict, visible_ids: set[str], visible_units, store: MemoryStore,
                  index: MemoryIndex, meta: dict) -> list[str]:
    """3d: find the anchor event, then the records in the window / on the target date."""
    from memory import anchor as anchor_mod
    from memory.retrieve import _date_anchored_candidates
    rel = None
    if analysis.get("relation") in ("after", "before", "same_day", "day_before", "day_after") and analysis.get("anchor_query"):
        rel = {"relation": analysis["relation"], "anchor_query": str(analysis["anchor_query"])}
    rel = rel or anchor_mod.parse_relation(question)
    if not rel:
        return []
    anchors = _ranked_subset(rel["anchor_query"], visible_ids, index, 5)
    if rel["relation"] in ("same_day", "day_before", "day_after"):
        # "the day I fly": the anchor is the record that NAMES a date (a booking, an invitation), not just any record that
        # mentions the topic; move such records first. Dates in the text only count when they differ from the delivery date.
        dated = [a for a in anchors if anchor_mod.event_dates(a, store) != [store.get(a).time.date()]]
        anchors = dated + [a for a in anchors if a not in dated]
    anchors = anchors[:3]
    meta["anchor"] = {"relation": rel["relation"], "query": rel["anchor_query"], "ids": anchors}
    if not anchors:
        return []
    out: list[str] = list(anchors[:2])
    if rel["relation"] == "gap":
        return []                                         # both ends of a gap question come from the normal pool
    elif rel["relation"] in ("after", "before"):
        window = set(anchor_mod.window_candidates(rel["relation"], anchors, visible_ids, store))
        rest = _strip_anchor(question, rel["anchor_query"])
        out += _ranked_subset(rest, window, index, 8)
    else:
        offset = {"day_before": -1, "day_after": 1, "same_day": 0}[rel["relation"]]
        dates = anchor_mod.event_dates(anchors[0], store)
        from datetime import timedelta
        targets = [(d + timedelta(days=offset)).isoformat() for d in dates[:1]]
        meta["anchor"]["target_dates"] = targets
        out += _date_anchored_candidates(targets, visible_units, 12)
    return [u for u in dict.fromkeys(out) if u in visible_ids]


def people_variants(question: str, resolution: dict) -> list[tuple[str, float]]:
    """Full-name versions of the question: the bare first name becomes each plausible full name."""
    if not resolution or resolution.get("by") == "name":      # the question already says who: nothing to add
        return []
    names = [resolution["resolved"]] if resolution.get("resolved") else [n for n, _s in resolution["candidates"]]
    out = []
    for name in names:
        q = re.sub(rf"\b{re.escape(resolution['first'])}\b", name, question, count=1, flags=re.I)
        out.append((q, 0.9 if resolution.get("resolved") else 0.6))
    return out


def run_v2(question: str, as_of: str, k: int, store: MemoryStore, index: MemoryIndex, visible_units, visible_ids: set[str],
           analysis: dict, meta: dict, return_meta: bool):
    idf = _idf(index)
    extras: dict[str, list[str]] = {}        # source name -> ids (anchor / people / chain / ledger)
    meta["extras"] = extras
    resolution: dict = {}
    if config.USE_PEOPLE:
        from memory import people as people_mod
        resolution = people_mod.resolve(question, visible_ids, store, idf)
        meta["people"] = resolution
    pool, lane_rank, variants = first_stage(question, analysis, visible_ids, store, index,
                                            people_variants(question, resolution) if config.PEOPLE_EXTRAS else None)
    meta["variants"] = [v for v, _w in variants]
    meta["pool_size"] = len(pool)
    meta["lane_rank"] = {uid: lane_rank[uid] for uid, _ in pool}
    pool_ids = [uid for uid, _ in pool]
    fused_rank = {uid: r for r, uid in enumerate(pool_ids, 1)}
    if config.USE_ANCHOR:
        extras["anchor"] = anchor_extras(question, analysis, visible_ids, visible_units, store, index, meta)
    if resolution and resolution.get("by") != "name" and config.PEOPLE_EXTRAS:
        from memory import people as people_mod
        names = [resolution["resolved"]] if resolution.get("resolved") else [n for n, _s in resolution["candidates"]]
        got: list[str] = []
        for name in names:
            got += _ranked_subset(question, set(people_mod.person_ids(name, visible_ids, store)), index, 4)
        extras["people"] = list(dict.fromkeys(got))
    if config.USE_LEDGER:
        from memory import ledger as ledger_mod
        if ledger_mod.wants_ledger(question):
            from datetime import datetime
            hits = ledger_mod.lookup(question, datetime.fromisoformat(as_of.replace("Z", "+00:00")), store, idf)
            meta["ledger"] = hits
            extras["ledger"] = list(dict.fromkeys(i for h in hits for i in h["source_ids"]))

    scorer = LocalScorer(question, store, meta)
    scorer.ensure(pool_ids + [i for ids in extras.values() for i in ids])
    if config.USE_CHAINS:
        from memory import chains
        base = scorer.combine(pool_ids, fused_rank, {})
        seeds = sorted(pool_ids, key=lambda u: (-base[u], u))
        new, chain = chains.expand(seeds, question, visible_ids, store, idf)
        extras["chain"] = new
        meta["chain"] = chain
        scorer.ensure(new)
    tail: list[str] = []                       # adjacent transcript turns: offered to the LLM reranker, ranked after the top 10 locally
    if config.USE_NEIGHBORS_V2:
        from memory import chains
        base = scorer.combine(pool_ids, fused_rank, {})
        top = sorted(pool_ids, key=lambda u: (-base[u], u))[: config.NEIGHBOR_SEEDS]
        tail = [u for u in chains.context_neighbors(top, visible_ids, store, config.NEIGHBOR_SPAN) if u not in fused_rank]
        extras["neighbors"] = tail
    bonus: dict[str, float] = {}
    for source, ids in extras.items():
        if source == "neighbors":
            continue
        for uid in ids[: config.EXTRAS_MAX]:
            bonus[uid] = max(bonus.get(uid, 0.0), config.EXTRA_BONUS[source])
    all_ids = list(dict.fromkeys(pool_ids + [u for u in bonus if u in visible_ids]))
    combined = scorer.combine(all_ids, fused_rank, bonus)
    survivors = [u for u in pick_survivors(all_ids, combined, store) if u in visible_ids]
    local = survivors[:10] + [u for u in tail if u in visible_ids and u not in survivors[:10]] + survivors[10:]
    cand = list(dict.fromkeys(survivors[: config.RERANK_V2_MAX - len(tail)] + [u for u in tail if u in visible_ids]))[: config.RERANK_V2_MAX]
    diagnostics.set_pool_size(len(cand))
    meta["origins"] = {}
    meta["rerank_reason"] = "N/A"

    final: list[str] | None = None
    if config.USE_RERANK:
        final = _llm_rerank(question, as_of, analysis, cand, store, meta)
        if final is None:
            meta["degraded"].append("rerank")
            diagnostics.mark_degraded("rerank")
            LOG.warning("Rerank failed or invalid JSON for question: %r", question)
    if final is None:
        final = list(dict.fromkeys(local))
    final = [uid for uid in final if uid in visible_ids]  # hard rule: nothing outside the visible set, ever
    ranked = [(uid, 1.0 - i * 0.01) for i, uid in enumerate(final[:k])]
    return (ranked, meta) if return_meta else ranked
