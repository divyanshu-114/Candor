"""Central, deterministic retrieval settings.

Keeping retrieval knobs here makes offline evaluation reproducible and avoids
question-specific tuning in the retrieval implementation.
"""
import os

BM25_K1 = 1.35
BM25_B = 0.72
BM25_TOP_K = 60
DENSE_TOP_K = 60
RRF_K = 60
RESULT_K = 20
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
LONG_UNIT_WORDS = 250
SHORT_MEETING_WORDS = 4
DENSE_MODEL = os.environ.get("DENSE_MODEL", "thenlper/gte-base" if os.environ.get("PIPELINE", "v2").lower() != "v1" else "BAAI/bge-small-en-v1.5")
# Models whose training expects a text prefix on queries / documents (fastembed does not add it).
DENSE_QUERY_PREFIX = {
    "snowflake/snowflake-arctic-embed-m": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1.5-Q": "search_query: ", "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
}
DENSE_DOC_PREFIX = {"nomic-ai/nomic-embed-text-v1.5-Q": "search_document: ", "nomic-ai/nomic-embed-text-v1.5": "search_document: "}
USE_DENSE = os.environ.get("USE_DENSE", "true").lower() not in {"0", "false", "no"}
USE_ENRICHED_TEXT = os.environ.get("USE_ENRICHED_TEXT", "true").lower() not in {"0", "false", "no"}

# Candidates/snippet size fed to the strong-model reranker. This is the one
# stage that uses the expensive model, so its input size drives most of the
# per-question cost and latency -- but see docs/DEVLOG.md "regression
# diagnosis": trimming this too far can cost retrieval quality. Overridable
# via env for the A/B measurement in that entry.
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "30"))
RERANK_SNIPPET_WORDS = int(os.environ.get("RERANK_SNIPPET_WORDS", "30"))

# --- MODE: single top-level lever for "how much LLM machinery runs" ---------
# baseline   -> plain fused BM25+dense on the raw question only (no LLM calls)
# no_rerank  -> analysis + multiquery + neighbors + hop2, but skip the strong
#               reranker (cheaper, still LLM-assisted recall)
# full       -> everything, including the strong-model rerank
# Individual USE_* env vars still override a MODE's default for ad-hoc
# ablation studies (see docs/DEVLOG.md), but MODE is what operators should set.
MODE = os.environ.get("MODE", "full").lower()
if MODE not in {"baseline", "no_rerank", "full"}:
    MODE = "full"

_MODE_DEFAULTS = {
    "baseline": dict(analysis=False, multiquery=False, neighbors=False, hop2=False, rerank=False),
    "no_rerank": dict(analysis=True, multiquery=True, neighbors=True, hop2=True, rerank=False),
    "full": dict(analysis=True, multiquery=True, neighbors=True, hop2=True, rerank=True),
}[MODE]


def _flag(name: str, mode_default: bool) -> bool:
    env = os.environ.get(name)
    if env is not None:
        return env.lower() not in {"0", "false", "no"}
    return mode_default


# LLM Stages
USE_ANALYSIS = _flag("USE_ANALYSIS", _MODE_DEFAULTS["analysis"])
USE_MULTIQUERY = _flag("USE_MULTIQUERY", _MODE_DEFAULTS["multiquery"])
USE_NEIGHBORS = _flag("USE_NEIGHBORS", _MODE_DEFAULTS["neighbors"])
USE_HOP2 = _flag("USE_HOP2", _MODE_DEFAULTS["hop2"])
USE_RERANK = _flag("USE_RERANK", _MODE_DEFAULTS["rerank"])

# --- v2 retrieval (each stage independently switchable; see docs/DEVLOG.md "v2 Phase 3") ---------
# PIPELINE=v1 restores the v1 behaviour exactly (every v2 stage off, v1 embedding model); PIPELINE=v2 (default) turns on the
# stages that were measured to help (docs/DEVLOG.md "v2 Phase 3"). Any single flag can still be overridden by its own env var.
PIPELINE = os.environ.get("PIPELINE", "v2").lower()
_V2 = PIPELINE != "v1"


def _bool(name: str, default: str) -> bool:
    """Env override wins; otherwise `default` only applies when the v2 pipeline is selected ("true" means "v2 on")."""
    env = os.environ.get(name)
    if env is not None:
        return env.lower() not in {"0", "false", "no"}
    return default.lower() == "true" and _V2

USE_LANES = _bool("USE_LANES", "true")                  # per-source lanes + query variants + RRF pool (3b)
POOL_SIZE = int(os.environ.get("POOL_SIZE", "150"))      # fused first-stage pool
LANE_K = int(os.environ.get("LANE_K", "25"))             # top-k per lane, per method, per query variant
LANE_FLOOR = int(os.environ.get("LANE_FLOOR", "3"))      # a lane's best N fused candidates always stay in the pool
USE_CROSS_ENCODER = _bool("USE_CROSS_ENCODER", "true")  # local cross-encoder cut (3a)
CE_MODEL = os.environ.get("CE_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
CE_CACHE_PATH = os.environ.get("CE_CACHE_PATH", ".cache/ce_scores.sqlite")
CE_KEEP = int(os.environ.get("CE_KEEP", "40"))           # candidates that survive the cross-encoder cut
CE_LANE_QUOTA = int(os.environ.get("CE_LANE_QUOTA", "2"))  # best-N per lane survive the cut regardless of score
CE_DOC_WORDS = int(os.environ.get("CE_DOC_WORDS", "110"))
CE_WEIGHT = float(os.environ.get("CE_WEIGHT", "2.0"))    # weight of the cross-encoder rank vs the fused rank
RERANK_V2_SNIPPET_WORDS = int(os.environ.get("RERANK_V2_SNIPPET_WORDS", "22"))
RERANK_V2_MAX = int(os.environ.get("RERANK_V2_MAX", "46"))  # most candidates the LLM reranker sees
USE_CHAINS = _bool("USE_CHAINS", "true")                # version chains for changing facts (3c)
CHAIN_SEEDS = int(os.environ.get("CHAIN_SEEDS", "6"))      # top candidates whose neighbourhood is searched for versions
# Extras are scored like any candidate; a source's bonus is added to its combined rank score. 1/(60+r) is 0.0164 at r=1, so
# 0.012 lifts an extra to roughly the level of a top-5 first-stage hit without overriding a clearly better candidate.
USE_NEIGHBORS_V2 = _bool("USE_NEIGHBORS_V2", "true")    # records adjacent to the best candidates (transcript question/answer pairs)
NEIGHBOR_SEEDS = int(os.environ.get("NEIGHBOR_SEEDS", "6"))
NEIGHBOR_SPAN = int(os.environ.get("NEIGHBOR_SPAN", "2"))
EXTRA_BONUS = {"agenda": float(os.environ.get("BONUS_AGENDA", "0.012")), "neighbors": float(os.environ.get("BONUS_NEIGHBORS", "0.006")), "anchor": float(os.environ.get("BONUS_ANCHOR", "0.012")), "chain": float(os.environ.get("BONUS_CHAIN", "0.008")),
               "people": float(os.environ.get("BONUS_PEOPLE", "0.008")), "ledger": float(os.environ.get("BONUS_LEDGER", "0.012"))}
EXTRAS_MAX = int(os.environ.get("EXTRAS_MAX", "14"))      # most guaranteed extra candidates (chains / anchor / people / ledger)
USE_ANCHOR = _bool("USE_ANCHOR", "true")                # anchor-then-window for relative time (3d)
USE_LEDGER_LLM = _bool("USE_LEDGER_LLM", "false")        # optional model pass over the commitments (owner / due / is-it-real); ~17k tokens once
EXTRACTIVE_CE = _bool("EXTRACTIVE_CE", "true")            # no-key answer sentence chosen by the local cross-encoder over the top 3 records
EXTRACTIVE_RANK_PENALTY = float(os.environ.get("EXTRACTIVE_RANK_PENALTY", "7.0"))   # cross-encoder points subtracted from sentences of records ranked 2-3
USE_AGENDA = _bool("USE_AGENDA", "true")                # explicit dates in the question ("on September 22nd") -> that day's calendar and records
AGENDA_CAP = int(os.environ.get("AGENDA_CAP", "14"))
USE_PEOPLE = _bool("USE_PEOPLE", "true")                # person resolution metadata (writer is told about ambiguity)
PEOPLE_EXTRAS = _bool("PEOPLE_EXTRAS", "false")          # also add full-name query variants and the person's records to the pool (measured: no gain)                # person resolution for shared first names (3e)
USE_LEDGER = _bool("USE_LEDGER", "true")                # commitments ledger lane (3f)
USE_PRECISE_MASKING = _bool("USE_PRECISE_MASKING", "true")  # fewer false-positive secret masks (3g)

# --- Concurrency / rate limiting / safety -----------------------------------
WORKERS = int(os.environ.get("WORKERS", "4"))
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "25"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "40"))
LLM_SEED = int(os.environ.get("LLM_SEED", "0"))
# Fallback TPM budget used only when the provider's rate-limit response
# headers aren't available yet (first call of the process, or a provider
# that doesn't send them). Groq's real on-demand-tier limit observed on this
# key is ~8000 TPM; kept a little under that on purpose.
LLM_TPM = int(os.environ.get("LLM_TPM", "7000"))

# Per-stage max_tokens and reasoning_effort (step 2b). Analysis/hop2 use the
# fast model with low effort and a small budget; rerank/writer use the
# strong model with medium effort and a bit more room.
MAX_TOKENS_ANALYSIS = int(os.environ.get("MAX_TOKENS_ANALYSIS", "350"))
MAX_TOKENS_HOP2 = int(os.environ.get("MAX_TOKENS_HOP2", "350"))
# 900, not 500: this provider's reasoning models (gpt-oss) reject
# reasoning_effort outright (see docs/DEVLOG.md step 1), so there is no way
# to suppress their chain-of-thought -- at 500 they were observed failing
# json_validate_failed on ~90% of rerank calls (truncated mid-reasoning,
# never closing valid JSON), which wastes more tokens on Groq's side (a
# failed generation still bills prompt+partial-completion) than the extra
# headroom here costs on the calls that would have succeeded anyway.
MAX_TOKENS_RERANK = int(os.environ.get("MAX_TOKENS_RERANK", "900"))
# 800, not 450: observed live that 3 of 4 writer calls on gpt-oss hit exactly
# 450 completion tokens (reasoning used the budget) and returned truncated JSON.
MAX_TOKENS_WRITER = int(os.environ.get("MAX_TOKENS_WRITER", "800"))
# Which role writes the answer. "fast" by default so the (quota-limited)
# strong role is spent on rerank only. Set WRITER_ROLE=strong to change.
WRITER_ROLE = "strong" if os.environ.get("WRITER_ROLE", "fast").strip().lower() == "strong" else "fast"
# Lower effort for the writer only (its prompt is new either way). Analysis and
# rerank keep their original effort values on purpose: the effort is part of the
# LLM cache key, and changing it would turn every cached call into a paid one.
REASONING_EFFORT_WRITER = os.environ.get("REASONING_EFFORT_WRITER", "low")
REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_MEDIUM = "medium"

# Model fallback preference order (step 2d): substrings matched, in order,
# against the provider's GET /models list when the configured model 404s.
# Automatic model choice for a provider that has a key but no model ids (GET /models; first matching substring wins).
# WHY these: open instruction-following models first (cheap, reproducible), then the usual flagship / small tiers.
AUTO_PREFERENCE_STRONG = ["gpt-oss-120b", "gpt-5", "gpt-4.1", "gpt-4o", "claude-sonnet", "claude-opus", "gemini-2.5-pro", "gemini-1.5-pro", "70b", "large"]
AUTO_PREFERENCE_FAST = ["gpt-oss-20b", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini", "claude-haiku", "gemini-2.5-flash", "flash", "mini", "8b", "small"]
NON_CHAT_MARKERS = ("embed", "tts", "whisper", "image", "dall", "moderation", "audio", "realtime", "transcribe", "rerank", "vision-preview", "guard", "safeguard")
MODEL_PREFERENCE_STRONG = ["120b", "70b"]
MODEL_PREFERENCE_FAST = ["20b", "8b"]

# --- Step 4 retrieval upgrades, each independently gated so they can be
# ablated one at a time (see docs/DEVLOG.md "Step 4"). All default off;
# the measured-best combination becomes the new default once decided.
USE_CHANGE_REASON_SYNONYMS = os.environ.get("USE_CHANGE_REASON_SYNONYMS", "false").lower() not in {"0", "false", "no"}
USE_HYDE_ANSWER_SKETCH = os.environ.get("USE_HYDE_ANSWER_SKETCH", "false").lower() not in {"0", "false", "no"}
USE_DATE_AGENDA = os.environ.get("USE_DATE_AGENDA", "false").lower() not in {"0", "false", "no"}
USE_DATE_RESOLVER = os.environ.get("USE_DATE_RESOLVER", "false").lower() not in {"0", "false", "no"}
DATE_AGENDA_CAP = int(os.environ.get("DATE_AGENDA_CAP", "12"))

# --- Writer v2 (Phase 4): soft quote match, partial answers, version-chain hint, computed arithmetic ------------------
WRITER_V2 = _bool("WRITER_V2", "true")
QUOTE_SOFT_THRESHOLD = float(os.environ.get("QUOTE_SOFT_THRESHOLD", "0.85"))

# --- Answer writer -----------------------------------------------------------
EVIDENCE_UNITS = 14
EVIDENCE_RECORD_MAX_WORDS = 120  # legacy fallback for callers not using the tiered budget below
EVIDENCE_TOKENS = 3000  # rough char/4 budget for the whole evidence package (token-budget task, step 1)
EVIDENCE_TOP_N = 5
EVIDENCE_TOP_N_WORDS = 200
EVIDENCE_REST_WORDS = 70
ANSWER_MAX_WORDS = 100
EXTRACTIVE_MAX_WORDS = 40
EXTRACTIVE_EMAIL_MAX_WORDS = 35
ANSWER_SOURCES_MAX = 6
SUPPORT_QUOTE_MAX_WORDS = 15
# Degraded/no-key mode only: BM25 scores on this corpus for the 27 train
# questions (all answerable) range from ~6.4 to ~60 (see docs/DEVLOG.md).
# This threshold is set far below that floor -- it exists only to catch
# "BM25 found essentially nothing" (near-zero signal), not as a quality gate;
# Gate 1/2 below do the real abstention work whenever a key is available.
NO_KEY_ABSTAIN_BM25_MIN = float(os.environ.get("NO_KEY_ABSTAIN_BM25_MIN", "2.0"))

# Evidence gate for the no-key path (memory/coverage.py evidence_gate). "v1" = only the original coverage gate.
NO_KEY_GATE = os.environ.get("NO_KEY_GATE", "v1" if os.environ.get("PIPELINE", "v2").lower() == "v1" else "v1")
GATE_COOCCUR_MIN = float(os.environ.get("GATE_COOCCUR_MIN", "0.45"))
GATE_CE_MIN = float(os.environ.get("GATE_CE_MIN", "0.0"))

# No-LLM coverage gate (memory/coverage.py): abstain only when at least this
# share of the question's idf mass is unmatched anywhere in visible memory AND
# no top-10 record covers this much of it. Generic thresholds, not per-question.
COVERAGE_MISSING_SHARE_MIN = float(os.environ.get("COVERAGE_MISSING_SHARE_MIN", "0.30"))
COVERAGE_BEST_MAX = float(os.environ.get("COVERAGE_BEST_MAX", "0.60"))

# Deliberately small: domain words and dates must remain searchable.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "with", "would", "did", "do", "does",
})
