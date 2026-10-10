"""Tokens and money spent per provider/model, from outputs/usage_ledger.jsonl (numbers only).
  .venv/bin/python scripts/spend.py [--since 2026-10-10T00:00]
Prices are USD per million tokens (prompt, completion), taken from OpenRouter's public /models list on 2026-10-10; Groq is free tier (0).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PRICES = {("openrouter", "openai/gpt-oss-120b"): (0.037, 0.17), ("openrouter", "openai/gpt-oss-20b"): (0.018, 0.09)}
path = Path(__file__).resolve().parent.parent / "outputs" / "usage_ledger.jsonl"
since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else "0000"
agg = defaultdict(lambda: [0, 0, 0])
for line in path.read_text().splitlines():
    r = json.loads(line)
    if r.get("ts", "") >= since:
        a = agg[(r["provider"], r["model"])]
        a[0] += 1; a[1] += r["prompt_tokens"]; a[2] += r["completion_tokens"]
total = 0.0
for (prov, model), (n, p, c) in sorted(agg.items()):
    pp, cp = PRICES.get((prov, model), (0.0, 0.0))
    usd = p * pp / 1e6 + c * cp / 1e6
    total += usd
    print(f"{prov:11s}{model:42s} calls={n:5d} prompt={p:9d} completion={c:8d} usd={usd:.4f}")
print(f"TOTAL usd={total:.4f}")
