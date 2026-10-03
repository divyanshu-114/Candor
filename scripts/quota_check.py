"""One minimal call per (provider, role); print rate-limit headers and the
token usage recorded for today. Run it once, by hand -- never in a loop.

Prints only numbers and model ids: no keys, prompts or response text.
Usage: .venv/bin/python scripts/quota_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()  # .env wins over a stale exported variable


_load_env()
from openai import APIStatusError  # noqa: E402

from memory import llm  # noqa: E402

_SHOW = ("remaining-requests", "remaining-tokens", "limit-requests", "limit-tokens",
         "reset-requests", "reset-tokens", "retry-after")


def _numbers(headers) -> dict[str, str]:
    out = {}
    for k, v in dict(headers).items():
        kl = k.lower()
        if ("ratelimit" in kl or "rate-limit" in kl or kl == "retry-after") and any(s in kl for s in _SHOW):
            out[kl] = str(v)[:24]
    return out


def _probe(provider, model: str) -> tuple[str, dict[str, str]]:
    try:
        raw = provider.client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=24, temperature=0.0, timeout=30)
        return f"HTTP {raw.status_code}", _numbers(raw.headers)
    except APIStatusError as e:
        return f"HTTP {e.status_code}", _numbers(e.response.headers) if e.response is not None else {}
    except Exception as e:  # network etc.: type name only, never the message
        return type(e).__name__, {}


def _ledger_today() -> dict[str, tuple[int, int, int]]:
    path = Path(os.environ.get("USAGE_LEDGER_PATH") or ROOT / "outputs/usage_ledger.jsonl")
    today = time.strftime("%Y-%m-%d")
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("ts", "")).startswith(today):
                a = agg[f"{r['provider']}/{r['model']}"]
                a[0] += 1
                a[1] += int(r.get("prompt_tokens", 0))
                a[2] += int(r.get("completion_tokens", 0))
    return {k: tuple(v) for k, v in agg.items()}


def _diagnostics_today() -> dict[str, tuple[int, int, int]]:
    path = ROOT / "outputs/diagnostics.jsonl"
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    if path.exists() and time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime)) == time.strftime("%Y-%m-%d"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for c in row.get("llm_calls", []):
                if not c.get("cache_hit"):
                    a = agg[c.get("model", "?")]
                    a[0] += 1
                    a[1] += int(c.get("prompt_tokens", 0))
                    a[2] += int(c.get("completion_tokens", 0))
    return {k: tuple(v) for k, v in agg.items()}


def main() -> None:
    if not llm._PROVIDERS:
        print("No provider has a key configured.")
        return
    probed: dict[tuple[str, str], tuple[str, dict]] = {}
    for role in ("strong", "fast"):
        order = [p.name for p in llm._role_providers(role)]
        print(f"[{role}] provider order: {','.join(order) or '-'}")
        for p in llm._role_providers(role):
            model = p.model_for(role)
            key = (p.name, model)
            if key not in probed:  # strong/fast sharing one model id: probe once
                probed[key] = _probe(p, model)
            status, nums = probed[key]
            print(f"  {p.name:<7} {role:<6} model={model} -> {status}")
            for k in sorted(nums):
                print(f"      {k} = {nums[k]}")
    print("\nToken usage recorded today (calls, prompt, completion):")
    for title, data in (("usage_ledger (all runs today)", _ledger_today()),
                        ("diagnostics.jsonl (last run, if from today)", _diagnostics_today())):
        print(f"  {title}")
        if not data:
            print("      none")
        for model, (c, pt, ct) in sorted(data.items()):
            print(f"      {model}: calls={c} prompt={pt} completion={ct} total={pt + ct}")


if __name__ == "__main__":
    main()
