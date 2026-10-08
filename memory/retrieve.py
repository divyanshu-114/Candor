"""Time-safe hybrid retrieval entry point with LLM stages."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache

from memory import config
from memory import diagnostics
from memory.index import MemoryIndex
from memory.store import MemoryStore
from memory.query import analyze_question, get_date_expansions
from memory import llm as llm_module
from memory.llm import chat_json
from memory.prompts import RERANK_SYSTEM

LOG = logging.getLogger(__name__)

# step 4a: a small generic map of change/reason vocabulary. If the question
# contains any of these, the rest are added as extra OR terms (at roughly
# half weight -- see _change_reason_query) to the lexical (BM25) query only.
_CHANGE_REASON_TERMS = (
    "moved", "pushed", "slipped", "delayed", "postponed", "rescheduled", "shifted", "bumped",
    "cancelled", "canceled", "dropped", "because", "reason", "why", "due", "regression", "bug",
    "blocker", "issue",
)


def _change_reason_query(question: str) -> str:
    """Add change/reason synonyms as extra lexical terms when the question
    uses any of them. WHY half weight via duplication: this custom BM25
    scorer takes a single query string, not per-token weights -- repeating
    the original question doubles its term frequency relative to a
    single appended copy of the synonym set, approximating "original at
    full weight, synonyms at half".
    """
    q_lower = question.lower()
    if not any(t in q_lower for t in _CHANGE_REASON_TERMS):
        return question
    q_words = set(q_lower.split())
    extra = " ".join(t for t in _CHANGE_REASON_TERMS if t not in q_words)
    return f"{question} {question} {extra}"


def _date_anchored_candidates(dates: list[str], visible_units, cap: int) -> list[str]:
    """step 4c: structured (not semantic) lookup of every visible record
    whose own date overlaps one of `dates` -- calendar events by their
    start/end, everything else by its delivery time's date.
    """
    target_dates = set()
    for d in dates:
        try:
            target_dates.add(datetime.strptime(str(d)[:10], "%Y-%m-%d").date())
        except ValueError:
            continue
    if not target_dates:
        return []

    matches = []
    for u in visible_units:
        if u.source == "calendar":
            start_raw = u.meta.get("start")
            end_raw = u.meta.get("end") or start_raw
            try:
                start_d = datetime.fromisoformat(str(start_raw)).date() if start_raw else None
            except ValueError:
                start_d = None
            try:
                end_d = datetime.fromisoformat(str(end_raw)).date() if end_raw else start_d
            except ValueError:
                end_d = start_d
            if start_d and any(start_d <= td <= (end_d or start_d) for td in target_dates):
                matches.append(u)
        elif u.source in ("email", "dictation", "meeting"):
            if u.time.date() in target_dates:
                matches.append(u)

    matches.sort(key=lambda u: u.time)
    return [u.id for u in matches[:cap]]


def _safe_chat_json(*args, **kwargs):
    """chat_json already swallows provider errors into None, but defend here
    too: this stage must degrade, not propagate, even if a mocked/future
    client implementation raises instead of returning None.
    """
    try:
        return chat_json(*args, **kwargs)
    except Exception as e:
        LOG.warning("chat_json raised unexpectedly (treating as degraded stage): %s", e)
        return None


@lru_cache(maxsize=4)
def _resources(data_dir: str, cache_dir: str) -> tuple[MemoryStore, MemoryIndex]:
    store = MemoryStore(data_dir)
    return store, MemoryIndex(store.units, cache_dir)


def hybrid_search(question: str, visible_ids: set[str], index: MemoryIndex,
                   bm25_query: str | None = None) -> list[tuple[str, float]]:
    """`bm25_query` lets a caller send a lexically-expanded query (step 4a)
    to BM25 only, while dense search still sees the original question.
    """
    lexical = index.bm25(bm25_query or question, visible_ids, config.BM25_TOP_K)
    dense = index.dense(question, visible_ids, config.DENSE_TOP_K)
    fused: dict[str, float] = {}
    for ranking in (lexical, dense):
        for rank, (unit_id, _score) in enumerate(ranking, 1):
            fused[unit_id] = fused.get(unit_id, 0.0) + 1.0 / (config.RRF_K + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:30]


def baseline_retrieve(question: str, as_of: str, k: int = config.RESULT_K,
                       data_dir: str | None = None, cache_dir: str = ".cache") -> list[tuple[str, float]]:
    """Plain fused BM25+dense search, no LLM stages, no config.USE_* branching.

    WHY it exists as its own function: it never touches the mutable config
    flags, so it is safe to call from a crash handler even while other
    threads are mid-retrieve() with different stages enabled -- there is no
    shared state to race on beyond the read-only, lru_cache'd index.
    """
    data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
    store, index = _resources(data_dir, cache_dir)
    visible_ids = {u.id for u in store.visible(as_of)}
    return hybrid_search(question, visible_ids, index)[:k]


def _fallback_order(question: str, visible_ids: set[str], index: MemoryIndex,
                     pool: list[tuple[str, float, str]], k: int,
                     priority_ids: list[str] | None = None) -> list[str]:
    """Baseline-first fallback used whenever rerank is off/unavailable.

    WHY: an earlier version fell back to the raw multi-query pool order,
    which the D2 ablation (docs/DEVLOG.md) showed destroys ranking quality
    (MRR 0.52 -> 0.22) relative to plain hybrid search on the original
    question. The contract here is: the first 6 ranks are always exactly
    what MODE=baseline would have returned, so a degraded run can never be
    worse than the baseline at the ranks that matter most; everything from
    sub-queries/neighbors/hop2 is interleaved in from rank 7 on, which is
    where it was shown to help recall@20 without hurting MRR.

    `priority_ids` (step 4c/d's date-anchored/resolved candidates) get
    slotted in right after the baseline head, ahead of the generic
    sub-query pool -- a structurally-certain match (a calendar event that
    literally overlaps a resolved date) earned at the pool-construction
    stage, not through ranking, shouldn't then get buried behind ~50
    ordinary candidates by the generic interleaving below. The first
    ablation measurement caught exactly this: priority items survived
    truncation but still never reached the top 40 because interleaving
    processes the rest of the pool in its original (unprioritized) order.
    """
    baseline = [uid for uid, _score in hybrid_search(question, visible_ids, index)]
    extra = [uid for uid, _score, _origin in pool if uid not in set(baseline)]

    head = [uid for uid in baseline[:6] if uid in visible_ids]
    seen = set(head)

    priority_tail = []
    for uid in (priority_ids or []):
        if uid in visible_ids and uid not in seen:
            priority_tail.append(uid)
            seen.add(uid)

    tail: list[str] = []
    rest_baseline = [uid for uid in baseline[6:] if uid in visible_ids]
    budget = k - len(head) - len(priority_tail)
    i = j = 0
    while (i < len(rest_baseline) or j < len(extra)) and len(tail) < budget:
        if i < len(rest_baseline):
            uid = rest_baseline[i]
            i += 1
            if uid in visible_ids and uid not in seen:
                tail.append(uid)
                seen.add(uid)
        if len(tail) >= budget:
            break
        if j < len(extra):
            uid = extra[j]
            j += 1
            if uid in visible_ids and uid not in seen:
                tail.append(uid)
                seen.add(uid)

    return (head + priority_tail + tail)[:k]


def retrieve(question: str, as_of: str, k: int = config.RESULT_K,
             data_dir: str | None = None, cache_dir: str = ".cache",
             return_meta: bool = False):
    """Return relevance-ranked, unique visible unit IDs; never datesorted.

    If return_meta=True, returns (ranked, metadata).
    """
    data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
    store, index = _resources(data_dir, cache_dir)
    visible_units = store.visible(as_of)
    visible_ids = {u.id for u in visible_units}

    meta: dict = {"degraded": []}

    # STAGE A: Analysis
    analysis = analyze_question(question, as_of)
    if analysis.pop("_degraded", False):
        meta["degraded"].append("analysis")
        diagnostics.mark_degraded("analysis")
        LOG.warning("Analysis stage degraded to heuristic fallback for question: %r", question)
    meta["analysis"] = analysis
    if config.USE_LANES:
        from memory.retrieve_v2 import run_v2  # lazy: retrieve_v2 builds on this module's helpers
        return run_v2(question, as_of, k, store, index, visible_units, visible_ids, analysis, meta, return_meta)
    sub_queries = analysis["sub_queries"] if config.USE_MULTIQUERY else [question]

    date_exp = " ".join(get_date_expansions(d) for d in analysis["dates"])
    expanded_queries = []
    if config.USE_MULTIQUERY and question not in sub_queries:
        sub_queries.insert(0, question)

    # step 4b: HyDE-style hypothetical-answer sub-query, used alongside the
    # real sub-queries so its vocabulary can match a real record even when
    # the question's own wording doesn't. Token-budget task (step 1): only
    # spend the extra sub-query slot when analysis itself signals the
    # question needs history/causal reasoning or a follow-up lookup -- a
    # plain current-state lookup doesn't need a hypothetical-answer probe.
    hyde_needed = bool(analysis.get("wants_history") or analysis.get("needs_followup"))
    if config.USE_HYDE_ANSWER_SKETCH and config.USE_MULTIQUERY and hyde_needed and analysis.get("answer_sketch"):
        sub_queries.append(analysis["answer_sketch"])

    for sq in sub_queries:
        expanded_queries.append(f"{sq} {date_exp}".strip() if date_exp else sq)

    # STAGE B: Multi-query
    pool = []  # list of (id, score, origin)
    seen = set()
    origins = {}

    if not config.USE_MULTIQUERY:
        # Fallback to single hybrid
        bm25_q = _change_reason_query(expanded_queries[0]) if config.USE_CHANGE_REASON_SYNONYMS else None
        res = hybrid_search(expanded_queries[0], visible_ids, index, bm25_query=bm25_q)
        for uid, score in res:
            pool.append((uid, score, "hybrid"))
            seen.add(uid)
            origins[uid] = "hybrid"
    else:
        results_by_q = {}
        for i, sq in enumerate(expanded_queries):
            bm25_q = _change_reason_query(sq) if config.USE_CHANGE_REASON_SYNONYMS else None
            results_by_q[i] = hybrid_search(sq, visible_ids, index, bm25_query=bm25_q)

        max_len = max((len(r) for r in results_by_q.values()), default=0)
        for i in range(max_len):
            for q_idx in range(len(expanded_queries)):
                if i < len(results_by_q[q_idx]):
                    uid, score = results_by_q[q_idx][i]
                    if uid not in seen:
                        origin = expanded_queries[q_idx]
                        pool.append((uid, score, origin))
                        seen.add(uid)
                        origins[uid] = origin
            if len(pool) >= 60:
                break
        pool = pool[:60]
        
    # Neighbors
    if config.USE_NEIGHBORS:
        new_neighbors = []
        for uid, score, origin in pool[:15]:
            u = store.get(uid)
            if u and u.source == "meeting":
                # Find adjacent
                segs = [x for x in visible_units if x.record_id == u.record_id]
                segs.sort(key=lambda x: x.seq)
                try:
                    idx = segs.index(u)
                    start = max(0, idx - 2)
                    end = min(len(segs), idx + 3)
                    for neighbor in segs[start:end]:
                        if neighbor.id not in seen:
                            new_neighbors.append((neighbor.id, score * 0.8, f"neighbor of {uid}"))
                            seen.add(neighbor.id)
                            origins[neighbor.id] = f"neighbor of {uid}"
                except ValueError:
                    pass
        pool.extend(new_neighbors)

    # step 4c/4d candidates are guaranteed to survive the pool[:RERANK_CANDIDATES]
    # cut below -- they come from a small, deliberately-capped structured
    # lookup (config.DATE_AGENDA_CAP), not a semantic ranking, so truncating
    # them away before rerank/fallback ever saw them would defeat the point
    # (this is exactly the bug the first ablation measurement caught: they
    # were being appended to an already-60-item pool and silently cut).
    guaranteed: list[tuple[str, float, str]] = []

    # step 4c: date-anchored agenda -- structured (not semantic) lookup of
    # every visible record whose own date overlaps a date already named in
    # the question (not one that still needs resolving -- that's 4d below).
    if config.USE_DATE_AGENDA and analysis["dates"]:
        for uid in _date_anchored_candidates(analysis["dates"], visible_units, config.DATE_AGENDA_CAP):
            if uid not in seen:
                guaranteed.append((uid, 0.5, "date_agenda"))
                seen.add(uid)
                origins[uid] = "date_agenda"

    # Hop 2, OR (step 4d) the date resolver that replaces it when the
    # question needs a lookup first (e.g. "the day I fly") and the flag is on.
    if config.USE_DATE_RESOLVER and analysis["needs_followup"]:
        snippets = []
        for uid, _, _ in pool[:12]:
            u = store.get(uid)
            if u:
                text = " ".join(u.text.split()[:80])
                snippets.append(f"ID: {uid} | Time: {u.time} | Text: {text}")

        if snippets:
            prompt = (f"Question: {question}\nAs of: {as_of}\nTop snippets:\n" +
                      "\n".join(snippets) +
                      "\n\nResolve any date this question needs (e.g. 'the day I fly' -> an actual date found "
                      "in the snippets) and suggest follow-up search terms.\n"
                      "Return JSON: {\"resolved_dates\": [\"YYYY-MM-DD\", ...], \"extra_queries\": [\"...\", ...]}.")
            res = _safe_chat_json("IR assistant resolving a date needed for a follow-up search.", prompt,
                                   llm_module.get_model_fast(), max_tokens=config.MAX_TOKENS_HOP2,
                                   reasoning_effort=config.REASONING_EFFORT_LOW, stage="resolver")
            if res and (isinstance(res.get("resolved_dates"), list) or isinstance(res.get("extra_queries"), list)):
                for nq in (res.get("extra_queries") or [])[:2]:
                    hop_res = hybrid_search(nq, visible_ids, index)
                    for uid, score in hop_res[:15]:
                        if uid not in seen:
                            pool.append((uid, score, f"date_resolver: {nq}"))
                            seen.add(uid)
                            origins[uid] = f"date_resolver: {nq}"
                resolved_dates = [str(x) for x in (res.get("resolved_dates") or [])]
                if resolved_dates and config.USE_DATE_AGENDA:
                    for uid in _date_anchored_candidates(resolved_dates, visible_units, config.DATE_AGENDA_CAP):
                        if uid not in seen:
                            guaranteed.append((uid, 0.5, "date_agenda(resolved)"))
                            seen.add(uid)
                            origins[uid] = "date_agenda(resolved)"
            else:
                meta["degraded"].append("hop2")
                diagnostics.mark_degraded("hop2")
                LOG.warning("Date resolver degraded for question: %r", question)
    elif config.USE_HOP2 and analysis["needs_followup"]:
        snippets = []
        for uid, _, _ in pool[:12]:
            u = store.get(uid)
            if u:
                text = " ".join(u.text.split()[:40])
                snippets.append(f"ID: {uid} | Time: {u.time} | Speaker: {u.speaker} | Text: {text}")

        if snippets:
            prompt = (f"Question: {question}\nAs of: {as_of}\nTop snippets:\n" +
                      "\n".join(snippets) +
                      "\n\nThe question needs a follow-up search based on these snippets. "
                      "Provide a JSON with 'new_queries': a list of up to 2 new short keyword search strings.")
            res = _safe_chat_json("IR assistant. Return JSON with 'new_queries' (list of strings).", prompt,
                                   llm_module.get_model_fast(), max_tokens=config.MAX_TOKENS_HOP2,
                                   reasoning_effort=config.REASONING_EFFORT_LOW, stage="hop2")
            if res and isinstance(res.get("new_queries"), list):
                for nq in res["new_queries"][:2]:
                    hop_res = hybrid_search(nq, visible_ids, index)
                    for uid, score in hop_res[:15]:
                        if uid not in seen:
                            pool.append((uid, score, f"hop2: {nq}"))
                            seen.add(uid)
                            origins[uid] = f"hop2: {nq}"
            else:
                meta["degraded"].append("hop2")
                diagnostics.mark_degraded("hop2")
                LOG.warning("Hop2 stage degraded (LLM call failed or returned invalid JSON) for question: %r", question)

    # Recency
    if analysis["wants_latest"] and config.USE_MULTIQUERY:
        top_40 = pool[:40]
        # sort by time descending
        top_40_sorted = sorted(top_40, key=lambda x: store.get(x[0]).time if store.get(x[0]) else store.units[0].time, reverse=True)
        latest_5 = top_40_sorted[:5]
        new_pool = latest_5.copy()
        for p in pool:
            if p not in latest_5:
                new_pool.append(p)
        pool = new_pool

    pool = pool[:config.RERANK_CANDIDATES] + guaranteed
    meta["origins"] = origins
    meta["rerank_reason"] = "N/A"
    diagnostics.set_pool_size(len(pool))

    # STAGE C: Rerank
    if config.USE_RERANK:
        snippets = []
        for uid, _, _ in pool:
            u = store.get(uid)
            if u:
                text = " ".join(u.text.split()[:config.RERANK_SNIPPET_WORDS])
                snippets.append(f"ID: {uid} | Source: {u.source} | Time: {u.time} | Speaker: {u.speaker} | Text: {text}")
        
        if snippets:
            sys_prompt = RERANK_SYSTEM
            user_prompt = f"Question: {question}\nAs of: {as_of}\nIntent: {analysis['intent']}\nSub-queries: {analysis['sub_queries']}\nwants_latest: {analysis['wants_latest']}\n\nCandidates:\n" + "\n".join(snippets)

            res = _safe_chat_json(sys_prompt, user_prompt, llm_module.get_model_strong(),
                                   max_tokens=config.MAX_TOKENS_RERANK, reasoning_effort=config.REASONING_EFFORT_LOW,
                                   stage="rerank")
            diagnostics.set_rerank_valid(bool(res and isinstance(res.get("ranked"), list)))
            if res and isinstance(res.get("ranked"), list):
                ranked_ids = res["ranked"]
                meta["rerank_reason"] = str(res.get("reason", ""))
                
                pool_ids = [p[0] for p in pool]
                valid_ranked = []
                seen_final = set()
                for uid in ranked_ids:
                    if uid in pool_ids and uid not in seen_final:
                        valid_ranked.append(uid)
                        seen_final.add(uid)
                        
                # fill missing
                for uid in pool_ids:
                    if uid not in seen_final:
                        valid_ranked.append(uid)
                        seen_final.add(uid)
                
                final_ids = [uid for uid in valid_ranked if uid in visible_ids]
                for uid in valid_ranked:
                    if uid not in visible_ids:
                        LOG.warning("Dropped forbidden id from LLM rerank: %s", uid)
                        
                ranked = [(uid, 1.0 - (i*0.01)) for i, uid in enumerate(final_ids[:k])]
                return (ranked, meta) if return_meta else ranked
            else:
                meta["degraded"].append("rerank")
                diagnostics.mark_degraded("rerank")
                LOG.warning("Rerank failed or invalid JSON for question: %r", question)

    # Fallback: baseline-first order (see _fallback_order docstring). Used for
    # MODE=baseline/no_rerank by design, and for MODE=full whenever the
    # rerank call degraded above -- in both cases this is never worse than
    # plain hybrid search on the original question for the ranks that matter
    # most (top 6).
    final_ids = _fallback_order(question, visible_ids, index, pool, k,
                                 priority_ids=[uid for uid, _score, _origin in guaranteed])
    ranked = [(uid, 1.0 - (i*0.01)) for i, uid in enumerate(final_ids[:k])]
    return (ranked, meta) if return_meta else ranked


def warmup(data_dir: str | None = None, cache_dir: str = ".cache") -> tuple[int, int, float]:
    """Explicit embedding warmup used by the CLI and deployment bootstrap."""
    data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
    _store, index = _resources(data_dir, cache_dir)
    return index.warmup()
