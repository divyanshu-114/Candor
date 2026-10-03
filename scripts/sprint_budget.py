"""Print today's real token use from outputs/usage_ledger.jsonl, per model,
plus the strong-budget remainder. Numbers only. Usage:
  .venv/bin/python scripts/sprint_budget.py [--since ISO_TS]"""
import json, os, sys, time
from pathlib import Path
BUDGET = int(os.environ.get("STRONG_BUDGET_TOKENS", "110000"))
path = Path(os.environ.get("USAGE_LEDGER_PATH") or "outputs/usage_ledger.jsonl")
since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else time.strftime("%Y-%m-%d")
agg = {}
if path.exists():
    for line in path.read_text().splitlines():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("ts", "") >= since:
            a = agg.setdefault(r["model"], [0, 0, 0]); a[0] += 1; a[1] += r["prompt_tokens"]; a[2] += r["completion_tokens"]
strong = sum(a[1] + a[2] for m, a in agg.items() if "120b" in m)
for m, a in sorted(agg.items()): print(f"{m}: calls={a[0]} prompt={a[1]} completion={a[2]} total={a[1]+a[2]}")
print(f"STRONG spent={strong} budget={BUDGET} remaining={BUDGET-strong}")
