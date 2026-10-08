"""LLM interaction: multi-provider failover, caching, token-aware rate
limiting, model fallback, and JSON extraction.

Public entry point (unchanged signature, so existing call sites/tests keep
working): `chat_json(system, user, model, cache_dir=..., max_tokens=...,
reasoning_effort=...)`. Internally, if `model` matches a configured
provider's role model ("strong"/"fast"), failures that exhaust that
provider's quota transparently fail over to the next configured provider's
model for the same role -- callers never see this, they just get a result
or None.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from memory import config
from memory import diagnostics

LOG = logging.getLogger(__name__)

# Legacy single-provider names, kept working per PROJECT_RULES rule 8 /
# this task's instructions. These are also provider "groq"'s defaults when
# GROQ_BASE_URL/GROQ_MODEL_STRONG/GROQ_MODEL_FAST aren't set.
LLM_MODEL_STRONG = os.environ.get("LLM_MODEL_STRONG", "openai/gpt-oss-120b")
LLM_MODEL_FAST = os.environ.get("LLM_MODEL_FAST", "openai/gpt-oss-20b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Providers: LLM_PROVIDERS=groq,gemini (ordered). Each NAME needs
# <NAME>_API_KEY to be included at all; a provider with no key is skipped.
# No provider at all -> degraded no-key mode (GROQ_API_KEY stays the
# backward-compat signal other modules check, e.g. memory/answer.py).
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    name: str
    api_key: str
    base_url: str
    model_strong: str
    model_fast: str
    model_premium: str = ""
    # Same-provider fallback models per role, tried (on daily-quota
    # exhaustion only) before failing over to the next provider. The daily
    # cap is per MODEL, so another model on the same key still has quota.
    fallbacks_strong: list[str] = field(default_factory=list)
    fallbacks_fast: list[str] = field(default_factory=list)
    exhausted_models: set[str] = field(default_factory=set)
    exhausted_roles: set[str] = field(default_factory=set)
    exhausted_logged_roles: set[str] = field(default_factory=set)
    unsupported_params: set[str] = field(default_factory=set)
    _client: OpenAI | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def model_for(self, role: str) -> str:
        return self.model_strong if role == "strong" else self.model_fast

    def chain(self, role: str) -> list[str]:
        """Primary model for `role`, then its fallbacks (de-duplicated, in order)."""
        extra = self.fallbacks_strong if role == "strong" else self.fallbacks_fast
        out: list[str] = []
        for m in [self.model_for(role), *extra]:
            if m and m not in out:
                out.append(m)
        return out

    def live_model(self, role: str) -> str | None:
        """First model in the role's chain not yet exhausted for the day."""
        return next((m for m in self.chain(role) if m not in self.exhausted_models), None)

    def mark_model_exhausted(self, model: str) -> bool:
        """Record a model as out for the day (whole process). Returns True the
        first time, so the caller can log once. Also flags every role whose
        whole chain is now exhausted."""
        first = model not in self.exhausted_models
        self.exhausted_models.add(model)
        for role in ("strong", "fast"):
            if self.live_model(role) is None:
                self.exhausted_roles.add(role)
        return first


def _env_names(var: str) -> list[str] | None:
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return None
    return [n.strip().lower() for n in raw.split(",") if n.strip()]


def _build_role_order() -> dict[str, list[str] | None]:
    """Per-role provider order. LLM_PROVIDERS_STRONG / LLM_PROVIDERS_FAST
    override LLM_PROVIDERS for that role only; None means "use the global
    order". WHY per role: the quota-limited providers differ per role (a
    capped strong model must not serve routine fast calls and vice versa).
    """
    return {"strong": _env_names("LLM_PROVIDERS_STRONG"), "fast": _env_names("LLM_PROVIDERS_FAST")}


def _build_providers() -> list[Provider]:
    names: list[str] = []
    for group in (_env_names("LLM_PROVIDERS") or ["groq"], _env_names("LLM_PROVIDERS_STRONG") or [],
                  _env_names("LLM_PROVIDERS_FAST") or []):
        for n in group:
            if n not in names:
                names.append(n)
    providers = []
    for name in names:
        upper = name.upper()
        if name == "groq":
            # legacy GROQ_API_KEY/LLM_BASE_URL/LLM_MODEL_* still work.
            api_key = os.environ.get(f"{upper}_API_KEY") or GROQ_API_KEY
            base_url = os.environ.get(f"{upper}_BASE_URL") or LLM_BASE_URL
            model_strong = os.environ.get(f"{upper}_MODEL_STRONG") or LLM_MODEL_STRONG
            model_fast = os.environ.get(f"{upper}_MODEL_FAST") or LLM_MODEL_FAST
        else:
            api_key = os.environ.get(f"{upper}_API_KEY")
            base_url = os.environ.get(f"{upper}_BASE_URL", "")
            if name == "openai" and not base_url:
                base_url = "https://api.openai.com/v1"
            model_strong = os.environ.get(f"{upper}_MODEL_STRONG", "")
            model_fast = os.environ.get(f"{upper}_MODEL_FAST", "")
            # An OpenAI-compatible endpoint with a single model id serves both roles.
            model_strong, model_fast = model_strong or model_fast, model_fast or model_strong
        if not api_key:
            continue  # provider without a key is skipped
        if not base_url or not (model_strong and model_fast):
            LOG.warning("Provider %r has a key but no base URL / model ids; skipping it.", name)
            continue
        def _fallbacks(role_var: str, default: list[str]) -> list[str]:
            raw = os.environ.get(f"{upper}_MODEL_{role_var}_FALLBACKS")
            return default if raw is None else [m.strip() for m in raw.split(",") if m.strip()]

        # Groq defaults: each role falls back to the OTHER role's model (the
        # daily cap is per model, so the other one usually still has quota).
        groq = name == "groq"
        providers.append(Provider(name=name, api_key=api_key, base_url=base_url,
                                   model_strong=model_strong, model_fast=model_fast,
                                   model_premium=os.environ.get(f"{upper}_MODEL_PREMIUM", ""),
                                   fallbacks_fast=_fallbacks("FAST", [model_strong] if groq else []),
                                   fallbacks_strong=_fallbacks("STRONG", [model_fast] if groq else [])))
    return providers


_PROVIDERS: list[Provider] = _build_providers()
_ROLE_ORDER: dict[str, list[str] | None] = _build_role_order()
_ALL_EXHAUSTED_LOGGED: set[str] = set()  # role names already logged as "all providers exhausted"


def is_available() -> bool:
    """True when at least one configured provider has a key and has not been taken out for the process."""
    return any(p.api_key and _provider_is_live(p) for p in _PROVIDERS)


def reload_providers() -> None:
    """Rebuild the provider list from the current environment. Only meant
    for tests that monkeypatch env vars; production never needs this.
    """
    global _PROVIDERS, _ALL_EXHAUSTED_LOGGED, _ROLE_ORDER
    _PROVIDERS = _build_providers()
    _ROLE_ORDER = _build_role_order()
    _ALL_EXHAUSTED_LOGGED = set()


def _provider_is_live(p: Provider) -> bool:
    """`_PROVIDERS` is built once at import time, but PROJECT_RULES'
    backward-compat contract (and a lot of existing tests) treat
    `memory.llm.GROQ_API_KEY` as a live, patchable "is there a key" signal
    for the legacy single-provider case. Re-check it dynamically here so
    `patch("memory.llm.GROQ_API_KEY", None)` still correctly disables the
    groq provider even though the module global is only read once at
    `_build_providers()` time otherwise.
    """
    if p.name == "groq" and not GROQ_API_KEY:
        return False
    return True


class RoleModel(str):
    """A model id that remembers which role asked for it.

    WHY: with quota-aware roles the strong and fast role can legitimately
    resolve to the *same* model id (e.g. one lite model for both), so the
    role can no longer be inferred from the id string. Being a `str`
    subclass keeps every existing call site and test working unchanged.
    """
    role: str | None = None

    def __new__(cls, value: str, role: str | None = None) -> "RoleModel":
        obj = super().__new__(cls, value)
        obj.role = role
        return obj


def _role_providers(role: str) -> list[Provider]:
    """All configured providers for a role, in that role's priority order
    (LLM_PROVIDERS_<ROLE>, falling back to LLM_PROVIDERS / `_PROVIDERS`)."""
    names = _ROLE_ORDER.get(role)
    if names is None:
        return list(_PROVIDERS)
    by_name = {p.name: p for p in _PROVIDERS}
    return [by_name[n] for n in names if n in by_name]


def _available_providers(role: str) -> list[Provider]:
    return [p for p in _role_providers(role) if role not in p.exhausted_roles and _provider_is_live(p)]


def get_model_strong() -> RoleModel:
    for p in _available_providers("strong"):
        return RoleModel(p.live_model("strong") or p.model_strong, "strong")
    roster = _role_providers("strong")
    return RoleModel(roster[0].model_strong if roster else LLM_MODEL_STRONG, "strong")


def get_model_fast() -> RoleModel:
    for p in _available_providers("fast"):
        return RoleModel(p.live_model("fast") or p.model_fast, "fast")
    roster = _role_providers("fast")
    return RoleModel(roster[0].model_fast if roster else LLM_MODEL_FAST, "fast")


def get_model_premium() -> str | None:
    """Optional capped/expensive model (e.g. a 20-requests/day tier). Only
    returned when `<PROVIDER>_MODEL_PREMIUM` is set; nothing in the pipeline
    calls it by default, so it can never burn quota on routine work."""
    for p in _PROVIDERS:
        if p.model_premium and _provider_is_live(p):
            return p.model_premium
    return None


def _provider_attempts(model: str) -> tuple[str | None, list[Provider]]:
    """Resolve which role `model` stands for and the ordered, not-yet-
    exhausted providers to try for it. A `RoleModel` carries its role
    explicitly and gets that role's full provider order. A plain string
    (tests, ad-hoc callers) is matched against provider models as before,
    then continues along the inferred role's order from that provider.
    """
    role = getattr(model, "role", None)
    if role in ("strong", "fast"):
        return role, [p for p in _role_providers(role) if role not in p.exhausted_roles and _provider_is_live(p)]
    start_name = None
    for p in _PROVIDERS:
        if model == p.model_strong:
            role, start_name = "strong", p.name
            break
        if model == p.model_fast:
            role, start_name = "fast", p.name
            break
    if role is None:
        return None, [p for p in _PROVIDERS if p.api_key and _provider_is_live(p)][:1]
    roster = _role_providers(role)
    names = [p.name for p in roster]
    first = names.index(start_name) if start_name in names else 0
    return role, [p for p in roster[first:] if role not in p.exhausted_roles and _provider_is_live(p)]


# ---------------------------------------------------------------------------
# Model-not-found fallback: if a configured model 404s, ask that provider
# what's actually available and pick the closest match once.
# ---------------------------------------------------------------------------

_MODEL_RESOLVE_LOCK = threading.Lock()
_RESOLVED_MODELS: dict[tuple[str, str], str] = {}  # (provider.name, role) -> working model id


def _fetch_available_models(provider: Provider) -> list[str]:
    try:
        resp = provider.client.models.list()
        return [m.id for m in resp.data]
    except Exception as e:
        LOG.error("Could not fetch %s/models for fallback: %s", provider.base_url, e)
        return []


def _resolve_fallback_model(provider: Provider, role: str) -> str | None:
    key = (provider.name, role)
    with _MODEL_RESOLVE_LOCK:
        if key in _RESOLVED_MODELS:
            return _RESOLVED_MODELS[key]
        prefs = config.MODEL_PREFERENCE_STRONG if role == "strong" else config.MODEL_PREFERENCE_FAST
        available = _fetch_available_models(provider)
        chosen = None
        for pref in prefs:
            for model_id in available:
                if pref in model_id:
                    chosen = model_id
                    break
            if chosen:
                break
        if chosen is None and available:
            chosen = available[0]
        if chosen is None:
            return None
        _RESOLVED_MODELS[key] = chosen
        if role == "strong":
            provider.model_strong = chosen
        else:
            provider.model_fast = chosen
        LOG.warning("Model not available on provider %r; falling back to %r for the rest of this process "
                    "(role=%s, available=%s)", provider.name, chosen, role, available)
        return chosen


def _is_model_not_found(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    message = str(exc).lower()
    return "model_not_found" in message or "does not exist" in message


def _is_auth_error(exc: Exception) -> bool:
    """401/403, or the 400 some OpenAI-compatible endpoints (Gemini) return
    for a bad key ("Please pass a valid API key")."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return status == 400 and "api key" in str(exc).lower()


def _names_response_format(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(t in message for t in ("response_format", "json_object", "json mode", "json_schema"))


def _names_param(exc: Exception, name: str) -> bool:
    """Does the provider's error message name this request parameter ("Unsupported parameter: 'max_tokens'", ...)?"""
    return name in str(exc).lower()


def _is_failed_generation(exc: Exception) -> bool:
    """Groq's 400 for output that didn't close as valid JSON (usually a
    reasoning model that ran out of max_tokens). It says nothing about which
    request parameters are supported."""
    message = str(exc).lower()
    return "json_validate_failed" in message or "failed_generation" in message


def _is_unsupported_param_error(exc: Exception) -> bool:
    """A 400 that is plausibly "this parameter isn't accepted". WHY the
    exclusion: a failed generation is also a 400, and before this check it
    was misread as "reasoning_effort rejected", silently disabling the
    parameter for the whole process (Groq's gpt-oss accepts it: low/medium/high)."""
    return (isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 400
            and not _is_failed_generation(exc))


def _is_seed_unsupported_error(exc: Exception) -> bool:
    """Some providers (e.g. Gemini's OpenAI-compatible endpoint) reject an
    unknown field with a real HTTP 400 instead of the SDK raising a
    client-side TypeError -- that path only covered the latter. Without
    this, a 400 caused by `seed` was silently treated as the reasoning_effort
    check instead (which doesn't match), then fell through to "permanently
    failed", failing the whole call over to the next provider forever,
    instead of just dropping the one unsupported field."""
    return (isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 400
            and "seed" in str(exc).lower())


# ---------------------------------------------------------------------------
# Token-aware rate limiting. Groq (and presumably other providers) enforce
# TOKENS per minute, not just a request count -- track the provider's own
# x-ratelimit-remaining-tokens/reset-tokens headers when present, with a
# plain per-minute bucket (config.LLM_TPM) fallback otherwise. One budget
# per provider, since different providers have different limits.
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?$")


def _parse_duration(value: str) -> float | None:
    value = value.strip()
    m = _DURATION_RE.match(value)
    if not m or not any(m.groups()):
        return None
    hours, minutes, seconds, millis = m.groups()
    total = 0.0
    if hours:
        total += int(hours) * 3600
    if minutes:
        total += int(minutes) * 60
    if seconds:
        total += float(seconds)
    if millis:
        total += int(millis) / 1000.0
    return total


class _TokenBudget:
    def __init__(self, fallback_tpm: int) -> None:
        self.fallback_tpm = fallback_tpm
        self._lock = threading.Lock()
        self.remaining_tokens: int | None = None
        self.reset_at: float | None = None
        self._local_budget = float(fallback_tpm)
        self._window_start = time.monotonic()

    def wait_time(self, estimated_tokens: int) -> float:
        with self._lock:
            now = time.monotonic()
            if self.remaining_tokens is not None and self.reset_at is not None:
                if now >= self.reset_at:
                    self.remaining_tokens = None
                elif self.remaining_tokens < estimated_tokens:
                    return max(min(self.reset_at - now, 60.0), 0.05)
                else:
                    self.remaining_tokens -= estimated_tokens
                    return 0.0
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._local_budget = float(self.fallback_tpm)
            if self._local_budget < estimated_tokens:
                return max(min(60.0 - (now - self._window_start), 60.0), 0.05)
            self._local_budget -= estimated_tokens
            return 0.0

    def update_from_headers(self, headers: dict[str, str] | None) -> None:
        if not headers:
            return
        with self._lock:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset = headers.get("x-ratelimit-reset-tokens")
            if remaining is not None:
                try:
                    self.remaining_tokens = int(remaining)
                except ValueError:
                    pass
            if reset is not None:
                secs = _parse_duration(reset)
                if secs is not None:
                    self.reset_at = time.monotonic() + secs


_TOKEN_BUDGETS: dict[str, _TokenBudget] = {}
_TOKEN_BUDGETS_LOCK = threading.Lock()


def _budget_for(provider: Provider) -> _TokenBudget:
    with _TOKEN_BUDGETS_LOCK:
        if provider.name not in _TOKEN_BUDGETS:
            _TOKEN_BUDGETS[provider.name] = _TokenBudget(config.LLM_TPM)
        return _TOKEN_BUDGETS[provider.name]


def _estimate_tokens(system: str, user: str, max_tokens: int) -> int:
    return (len(system) + len(user)) // 4 + max_tokens


MAX_TOKENS_CEILING = 4096  # a truncated reasoning reply is retried with a larger budget, never above this
MAX_SINGLE_WAIT_S = 60.0  # never sleep longer than this for any single wait


def _acquire_token_budget(provider: Provider, estimated_tokens: int) -> None:
    budget = _budget_for(provider)
    while True:
        wait = min(budget.wait_time(estimated_tokens), MAX_SINGLE_WAIT_S)
        if wait <= 0:
            return
        time.sleep(wait)


# Kept for compatibility with any ad-hoc request-count limiting on top of
# the token budget; not used by chat_json itself (the per-provider token
# budget above supersedes it for TPM-style limits).
class _RateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.max_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60.0 - (now - self._calls[0])
            time.sleep(max(sleep_for, 0.01))


_RATE_LIMITER = _RateLimiter(config.LLM_MAX_RPM)

# Thread-safe running total of tokens spent this process, for outputs/run_stats.json.
_USAGE_LOCK = threading.Lock()
_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cache_hits": 0, "rate_limit_hits": 0}


def get_usage() -> dict[str, int]:
    with _USAGE_LOCK:
        return dict(_USAGE)


def reset_usage() -> None:
    with _USAGE_LOCK:
        for k in _USAGE:
            _USAGE[k] = 0


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += prompt_tokens
        _USAGE["completion_tokens"] += completion_tokens
        _USAGE["calls"] += 1


_LEDGER_LOCK = threading.Lock()
DEFAULT_LEDGER_PATH = "outputs/usage_ledger.jsonl"


def ledger_path() -> Path:
    """Read at call time (not import time) so tests can redirect it with the
    USAGE_LEDGER_PATH env var no matter when `memory.llm` was imported."""
    return Path(os.environ.get("USAGE_LEDGER_PATH") or DEFAULT_LEDGER_PATH)


def _ledger(provider: Provider, model: str, role: str | None, prompt_tokens: int, completion_tokens: int) -> None:
    """Append one line per real (non-cached) call so daily quota use survives
    across runs (diagnostics.jsonl is rewritten each run). Numbers and model
    ids only -- never a key, prompt or response. Best effort: a read-only
    disk must not break answering."""
    try:
        with _LEDGER_LOCK:
            path = ledger_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": provider.name,
                                    "base_url": provider.base_url, "model": model,
                                    "role": role, "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens}) + "\n")
    except OSError:
        pass


def _record_cache_hit() -> None:
    with _USAGE_LOCK:
        _USAGE["cache_hits"] += 1


def _record_rate_limit_hit() -> None:
    with _USAGE_LOCK:
        _USAGE["rate_limit_hits"] += 1


_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> dict[str, Any]:
    """Robust JSON extraction from LLM output.

    Handles the habits seen across providers: <think>...</think> blocks before the answer, markdown code fences
    (anywhere, not only at the start), prose before/after the object, and braces inside that prose. Tries every
    '{' as the start of a JSON value and returns the first object that parses; {} (logged) when none does."""
    text = _THINK_RE.sub(" ", text or "").strip()
    fenced = _FENCE_RE.findall(text)
    candidates = [f.strip() for f in fenced if "{" in f] + [text]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for m in re.finditer(r"\{", cand):
            try:
                value, _end = decoder.raw_decode(cand[m.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    LOG.warning("Failed to decode JSON from a model reply (%d chars)", len(text))
    return {}


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """(b) daily/quota exhausted: message mentions "per day"/"TPD"/"quota"/
    "daily", OR the Retry-After is over 120s. Resets on a rolling window,
    not a per-minute one -- retrying it like a TPM 429 can block a call for
    an hour. Fail over to the next provider immediately instead.
    """
    message = str(exc).lower()
    if any(t in message for t in ("per day", "tpd", "quota", "daily")):
        return True
    retry_after = _retry_after_seconds(exc)
    return retry_after is not None and retry_after > 120


def _is_short_rate_limit(exc: Exception) -> bool:
    """(a) per-minute limit: 429 with a short Retry-After (<60s) -- worth
    waiting out on the SAME provider.
    """
    retry_after = _retry_after_seconds(exc)
    return retry_after is None or retry_after < 60


def chat_json(system: str, user: str, model: str, cache_dir: str | None = None,
              max_tokens: int = 500, reasoning_effort: str | None = None, stage: str = "") -> dict[str, Any] | None:
    """Send a chat request, enforce JSON response, with caching, a
    token-aware rate limiter, cross-provider failover, and backoff.

    Returns None if no provider is configured or every attempted provider
    fails (quota exhausted, down, etc.) -- callers must treat None as "this
    stage degraded" and fall back to the last good (non-LLM) output, never
    raise.
    """
    # Explicit argument > LLM_CACHE_DIR env > default. The env var lets a cold
    # measurement use a fresh directory instead of deleting the warm cache.
    cache_dir = cache_dir or os.environ.get("LLM_CACHE_DIR") or ".cache/llm"
    role, attempts = _provider_attempts(model)
    if not attempts:
        LOG.debug("No provider available for model %r; skipping LLM call.", model)
        return None

    os.makedirs(cache_dir, exist_ok=True)
    call_started = time.monotonic()

    for provider in attempts:
        # same-provider model chain first (daily caps are per model), then
        # the next provider
        for provider_model in (provider.chain(role) if role else [model]):
            result = _chat_json_one_provider(provider, provider_model, role, system, user, cache_dir,
                                              max_tokens, reasoning_effort, call_started, stage)
            if result is not _DAILY_EXHAUSTED:
                return result

    if role and role not in _ALL_EXHAUSTED_LOGGED:
        LOG.error("All configured providers exhausted for role=%s; degrading this stage for the "
                  "rest of the process (logged once, not per question).", role)
        _ALL_EXHAUSTED_LOGGED.add(role)
    diagnostics.mark_degraded("llm_unavailable")
    return None


_DAILY_EXHAUSTED = object()  # sentinel: this provider is out, try the next one


def _chat_json_one_provider(provider: Provider, model: str, role: str | None, system: str, user: str,
                             cache_dir: str, max_tokens: int, reasoning_effort: str | None,
                             call_started: float, stage: str = "") -> dict[str, Any] | None | object:
    req_hash = hashlib.sha256(
        json.dumps([provider.name, system, user, model, config.LLM_SEED, max_tokens, reasoning_effort]).encode()
    ).hexdigest()
    cache_path = Path(cache_dir) / f"{req_hash}.json"

    if cache_path.exists():
        try:
            result = json.loads(cache_path.read_text())
            if not result:
                raise ValueError("cached empty result = an earlier failed/truncated generation; treat as a miss")
            _record_cache_hit()
            diagnostics.record_call(model, 0, 0, retries=0, cache_hit=True, seconds=time.monotonic() - call_started, stage=stage)
            return result
        except Exception:
            pass

    # LLM_FROZEN_ROLES=strong: cache hits still work, but a cache MISS for a
    # frozen role is never sent -- it degrades loudly instead of spending quota
    # that is reserved elsewhere.
    frozen = {r.strip() for r in os.environ.get("LLM_FROZEN_ROLES", "").lower().split(",") if r.strip()}
    # A fallback model that is another frozen role's primary is frozen too
    # (otherwise falling back fast -> strong model would spend frozen quota).
    frozen_models = {provider.model_for(r) for r in frozen if r in ("strong", "fast")}
    if role in frozen or model in frozen_models:
        LOG.error("Role %r is frozen (LLM_FROZEN_ROLES) and this request is not cached; not calling the provider.", role)
        diagnostics.mark_degraded("frozen_cache_miss")
        diagnostics.record_call(model, 0, 0, retries=0, cache_hit=False,
                                 seconds=time.monotonic() - call_started, stage=stage)
        return None

    if model in provider.exhausted_models:
        return _DAILY_EXHAUSTED  # out for the day; a cache hit above was still served

    max_retries = 3  # (d) 5xx/timeouts: at most 3 retries, then fail over
    base_delay = 1.0
    supports_seed = True
    # Scoped by role, not just provider: observed live that a provider can
    # support reasoning_effort on one model/role and reject it on the other
    # (e.g. a "lite" fast model with no thinking mode at all) -- a single
    # provider-wide flag meant rejecting it on one role silently disabled it
    # for the other role too, which for a model that defaults to a "thinking"
    # mode (see .env.example) can silently burn the whole max_tokens budget
    # on hidden reasoning instead of a real answer.
    reasoning_effort_key = f"reasoning_effort:{role or model}"
    try_reasoning_effort = (reasoning_effort is not None and reasoning_effort_key not in provider.unsupported_params)
    retries_used = 0
    drops = 0  # parameter drops don't consume the backoff-retry budget (3 droppable params)
    truncations = 0
    use_response_format = "response_format" not in provider.unsupported_params

    for loop_i in range(max_retries + 8):
        attempt = loop_i - drops  # backoff-retry counter
        _acquire_token_budget(provider, _estimate_tokens(system, user, max_tokens))
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=config.LLM_TIMEOUT_S,
            )
            # Providers disagree on these: newer OpenAI models want max_completion_tokens and some reasoning
            # models accept only the default temperature. Both are learned from a 400 and remembered per provider.
            kwargs["max_completion_tokens" if "max_tokens" in provider.unsupported_params else "max_tokens"] = max_tokens
            if "temperature" not in provider.unsupported_params:
                kwargs["temperature"] = 0.0
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                # No JSON mode on this provider: ask in words, and rely on
                # _extract_json() to pull the object out of the text.
                kwargs["messages"][0]["content"] = system + "\n\nRespond with ONLY a single valid JSON object, no prose."
            if supports_seed and "seed" not in provider.unsupported_params:
                kwargs["seed"] = config.LLM_SEED
            if try_reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            raw = provider.client.chat.completions.with_raw_response.create(**kwargs)
            _budget_for(provider).update_from_headers(dict(raw.headers))
            response = raw.parse()

            message = response.choices[0].message
            content = message.content or ""
            if not content.strip():
                # some reasoning models leave `content` empty and put the answer in a reasoning field
                for attr in ("reasoning_content", "reasoning"):
                    alt = getattr(message, attr, None)
                    if isinstance(alt, str) and "{" in alt:
                        content = alt
                        break
            result = _extract_json(content or "{}")

            # A reasoning model can spend the whole token budget on hidden thinking and stop with finish_reason="length"
            # and nothing usable. Retry with a doubled budget (twice at most) instead of failing the stage.
            finish = getattr(response.choices[0], "finish_reason", None)
            if not result and finish == "length" and truncations < 2 and max_tokens < MAX_TOKENS_CEILING:
                truncations += 1
                new_budget = min(max_tokens * 2, MAX_TOKENS_CEILING)
                LOG.warning("Provider %r model %s ran out of tokens (%d) with no usable reply; retrying with %d",
                            provider.name, model, max_tokens, new_budget)
                _record_usage(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                              getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0)
                max_tokens = new_budget
                drops += 1
                continue

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            _record_usage(prompt_tokens, completion_tokens)
            _ledger(provider, model, role, prompt_tokens, completion_tokens)
            diagnostics.record_call(model, prompt_tokens, completion_tokens, retries=retries_used,
                                     cache_hit=False, seconds=time.monotonic() - call_started, stage=stage)

            if result:  # never cache a failed/truncated generation: it would repeat forever
                cache_path.write_text(json.dumps(result))
            return result

        except TypeError:
            # SDK/model combination rejects a kwarg at the Python level (not
            # even reaching the HTTP layer) -- drop seed and retry immediately.
            supports_seed = False
            provider.unsupported_params.add("seed")
            drops += 1
            continue

        except (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError) as e:
            # (c) unsupported parameter, HTTP-level: a provider that rejects
            # `seed` with a real 400 (not a client-side TypeError) -- drop it
            # for this provider for the rest of the process and retry once.
            if supports_seed and "seed" not in provider.unsupported_params and _is_seed_unsupported_error(e) and not _is_auth_error(e):
                LOG.warning("Provider %r rejected seed; dropping it for the rest of this process", provider.name)
                provider.unsupported_params.add("seed")
                drops += 1
                continue

            # (c) token-limit parameter name / temperature: adapt to the provider, once each, not counted as a backoff.
            if _is_unsupported_param_error(e) and not _is_auth_error(e):
                if "max_tokens" not in provider.unsupported_params and _names_param(e, "max_tokens"):
                    LOG.warning("Provider %r rejected max_tokens; using max_completion_tokens from now on", provider.name)
                    provider.unsupported_params.add("max_tokens")
                    drops += 1
                    continue
                if "temperature" not in provider.unsupported_params and _names_param(e, "temperature"):
                    LOG.warning("Provider %r rejected temperature; using its default from now on", provider.name)
                    provider.unsupported_params.add("temperature")
                    drops += 1
                    continue

            # (c) a 400 that names response_format / JSON mode: drop it,
            # rely on text extraction (checked before the generic
            # reasoning_effort branch so it isn't blamed for this).
            if use_response_format and _is_unsupported_param_error(e) and not _is_auth_error(e) \
                    and _names_response_format(e):
                LOG.warning("Provider %r rejected response_format; dropping it for the rest of this process",
                            provider.name)
                provider.unsupported_params.add("response_format")
                use_response_format = False
                drops += 1
                continue

            # (c) unsupported parameter: a 400 caused by reasoning_effort (or
            # any other optional param) -- drop it for this provider for the
            # rest of the process and retry once, not counted as a backoff.
            if try_reasoning_effort and _is_unsupported_param_error(e) and not _is_auth_error(e):
                LOG.warning("Provider %r rejected reasoning_effort for role=%s; dropping it for that role "
                            "for the rest of this process", provider.name, role or model)
                provider.unsupported_params.add(reasoning_effort_key)
                try_reasoning_effort = False
                drops += 1
                continue

            # (c) an unexplained 400 with nothing else left to drop: the only
            # remaining optional parameter is response_format.
            if use_response_format and _is_unsupported_param_error(e) and not _is_auth_error(e) \
                    and not try_reasoning_effort and not _is_model_not_found(e):
                LOG.warning("Provider %r returned 400 with JSON mode on; retrying without response_format",
                            provider.name)
                provider.unsupported_params.add("response_format")
                use_response_format = False
                drops += 1
                continue

            # (c) model_not_found: discover what's available, retry once.
            if _is_model_not_found(e):
                fallback = _resolve_fallback_model(provider, role) if role else None
                if fallback and fallback != model:
                    model = fallback
                    continue

            if isinstance(e, RateLimitError) or getattr(e, "status_code", None) == 429:
                _record_rate_limit_hit()
                headers = getattr(getattr(e, "response", None), "headers", None)
                _budget_for(provider).update_from_headers(dict(headers) if headers else None)

                # (b) daily/quota exhausted -> this MODEL is out for the day
                # (the cap is per model). Mark it for the whole process, log
                # once, and let the caller try the next model in the chain,
                # then the next provider.
                if _is_daily_quota_exhausted(e):
                    if provider.mark_model_exhausted(model):
                        nxt = provider.live_model(role) if role else None
                        LOG.error("Daily/quota limit hit for provider=%r model=%s (role=%s); %s: %s",
                                  provider.name, model, role,
                                  f"falling back to model {nxt}" if nxt else "failing over to the next provider", e)
                    diagnostics.record_call(model, 0, 0, retries=retries_used, cache_hit=False,
                                             seconds=time.monotonic() - call_started, stage=stage)
                    return _DAILY_EXHAUSTED

                # (a) short per-minute limit -- wait it out on this provider,
                # never more than MAX_SINGLE_WAIT_S for any single wait.
                if _is_short_rate_limit(e) and attempt < max_retries - 1:
                    retries_used += 1
                    retry_after = _retry_after_seconds(e)
                    delay = min(retry_after, MAX_SINGLE_WAIT_S) if retry_after is not None else base_delay * (2 ** attempt)
                    LOG.warning("Provider %r rate-limited (retry %d/%d) sleeping %.1fs: %s",
                                provider.name, attempt + 1, max_retries, delay, e)
                    time.sleep(min(delay, MAX_SINGLE_WAIT_S))
                    continue

            # (d) 5xx/connection/timeout: backoff, at most max_retries, then fail over.
            is_retryable = isinstance(e, (APIConnectionError, APITimeoutError)) or (
                hasattr(e, "status_code") and e.status_code is not None and e.status_code >= 500
            )
            if is_retryable and attempt < max_retries - 1:
                retries_used += 1
                delay = min(base_delay * (2 ** attempt), MAX_SINGLE_WAIT_S)
                LOG.warning("Provider %r call failed (retry %d/%d) sleeping %.1fs: %s",
                            provider.name, attempt + 1, max_retries, delay, e)
                time.sleep(delay)
                continue

            diagnostics.record_call(model, 0, 0, retries=retries_used, cache_hit=False,
                                     seconds=time.monotonic() - call_started, stage=stage)
            if _is_auth_error(e):
                # A rejected key will be rejected on every call: take the
                # provider out for both roles so later calls don't each pay a
                # failed round trip, and let the next provider serve this one.
                provider.exhausted_roles.update({"strong", "fast"})
                if "auth" not in provider.exhausted_logged_roles:
                    LOG.error("Provider %r rejected its API key (HTTP %s); disabling it for this process "
                              "and failing over.", provider.name, getattr(e, "status_code", "?"))
                    provider.exhausted_logged_roles.add("auth")
                return _DAILY_EXHAUSTED
            if is_retryable:
                LOG.error("Provider %r unavailable after %d attempts, failing over: %s",
                          provider.name, max_retries, e)
                return _DAILY_EXHAUSTED
            LOG.error("Provider %r call permanently failed: %s", provider.name, e)
            return None

        except Exception as e:
            LOG.error("Unexpected LLM error on provider %r: %s", provider.name, e)
            diagnostics.record_call(model, 0, 0, retries=retries_used, cache_hit=False,
                                     seconds=time.monotonic() - call_started, stage=stage)
            return None

    diagnostics.record_call(model, 0, 0, retries=retries_used, cache_hit=False,
                             seconds=time.monotonic() - call_started, stage=stage)
    return None
