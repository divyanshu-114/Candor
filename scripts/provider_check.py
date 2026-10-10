"""Run EVERY LLM stage once, per configured provider, on a tiny synthetic input and print a PASS/FAIL table.

  .venv/bin/python scripts/provider_check.py [--providers groq,openrouter] [--roles strong,fast] [--stages analysis,rerank,...]

Stages: analysis (query understanding), rerank, writer, planner (actions), ledger (commitments extraction).
Columns: JSON validity, whether the reply has the keys the pipeline needs, parameters the provider rejected and the
pipeline dropped (max_tokens/seed/temperature/response_format/reasoning_effort), tokens, seconds. No caching, no
failover: each cell is a single provider/role/stage call. Never prints keys or prompts. Cost: ~10 small calls per provider.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')


def stages() -> dict:
    from actions.planner import PLAN_SYSTEM
    from memory import answer as answer_mod
    from memory.ledger_llm import LEDGER_SYSTEM
    from memory.prompts import RERANK_SYSTEM
    from memory.query import SYSTEM_PROMPT
    evidence = ('<evidence>\n<record id="SL-1" source="slack" time="2026-09-10T09:42:00-07:00" speaker="Sam Lee" speaker_known="true" edited="false">'
                'We moved the launch to Oct 14 because QA found a geocoding bug.</record>\n</evidence>')
    return {
        "analysis": (SYSTEM_PROMPT, "as_of: 2026-09-18T18:00:00-07:00\nUser: When is the launch?", {"intent", "sub_queries"}, "fast", 350),
        "rerank": (RERANK_SYSTEM, "Question: When is the launch?\nAs of: 2026-09-18T18:00:00-07:00\nIntent: current_state\nSub-queries: ['launch date']\n"
                   "wants_latest: True\n\nCandidates:\nID: SL-1 | Source: slack | Time: 2026-09-10 | Speaker: Sam Lee | Text: moved the launch to Oct 14\n"
                   "ID: SL-2 | Source: slack | Time: 2026-09-11 | Speaker: Pat | Text: lunch is tacos today", {"ranked"}, "strong", 400),
        "writer": (answer_mod.WRITER_SYSTEM_V2, f"{evidence}\n\nToday (as_of): 2026-09-18T18:00:00-07:00\nQuestion intent: current_state "
                   "(checklist: state the current value first)\nQuestion: When is the launch?", {"answerable", "answer", "support"}, "fast", 500),
        "planner": (PLAN_SYSTEM, "today: 2026-09-18 Friday; timezone America/Los_Angeles, UTC offset -07:00 (write every time as ISO 8601 with -07:00)\n"
                    "PEOPLE (name | slack_id | dm_id | email | int/ext):\nSam Lee | U1 | D1 | sam@example.com | int\nCHANNELS (id name):\nC1 general\n"
                    "EVENTS (id | title | start..end | attendees):\n\nCOMMAND: Open Figma", {"actions"}, "fast", 400),
        "ledger": (LEDGER_SYSTEM, json.dumps([{"id": "C-1", "owner": "Sam", "action": "I'll send the deck by Friday", "due": None,
                   "snippets": [{"kind": "promise", "text": "[09-10 10:00] Sam: I'll send the deck by Friday"}, {"kind": "done", "text": "[09-11 15:00] Sam: deck sent"}]}]),
                   {"items"}, "fast", 400),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default=os.environ.get("LLM_PROVIDERS", "groq,openrouter"))
    ap.add_argument("--roles", default="strong,fast")
    ap.add_argument("--stages", default="analysis,rerank,writer,planner,ledger")
    a = ap.parse_args()
    _load_env()
    os.environ["LLM_PROVIDERS"] = a.providers
    from memory import llm
    llm.reload_providers()
    wanted = {s.strip() for s in a.stages.split(",")}
    roles = [r.strip() for r in a.roles.split(",")]
    table, failures = [], 0

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    capture = _Capture()
    logging.getLogger("memory.llm").addHandler(capture)
    logging.getLogger("memory.llm").setLevel(logging.WARNING)
    for provider in llm._PROVIDERS:
        for stage, (system, user, keys, default_role, max_tokens) in stages().items():
            if stage not in wanted:
                continue
            for role in roles if len(roles) == 1 else [default_role]:
                model = provider.model_for(role)
                before = set(provider.unsupported_params)
                capture.messages.clear()
                t0 = time.monotonic()
                with tempfile.TemporaryDirectory() as td:
                    res = llm._chat_json_one_provider(provider, model, role, system, user, td, max_tokens, "low", t0, stage)
                secs = time.monotonic() - t0
                ok_json = isinstance(res, dict) and bool(res)
                ok_keys = ok_json and keys <= set(res)
                dropped = sorted(set(provider.unsupported_params) - before)
                limited = not ok_json and any("429" in m or "rate" in m.lower() for m in capture.messages)
                status = "PASS" if ok_keys else ("PARTIAL" if ok_json else ("LIMITED" if limited else "FAIL"))
                failures += status != "PASS"
                table.append((provider.name, role, model[:34], stage, status, "yes" if ok_json else "NO", "yes" if ok_keys else "NO",
                              ",".join(dropped) or "-", secs))
    usage = llm.get_usage()
    hdr = f"{'provider':<11}{'role':<7}{'model':<35}{'stage':<10}{'result':<9}{'json':<5}{'keys':<5}{'dropped params':<28}{'sec':>6}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in table:
        print(f"{r[0]:<11}{r[1]:<7}{r[2]:<35}{r[3]:<10}{r[4]:<9}{r[5]:<5}{r[6]:<5}{r[7]:<28}{r[8]:>6.1f}")
    if any(r[4] == "LIMITED" for r in table):
        print("\nLIMITED = the provider answered 429 (rate limit / shared free pool) after retries; not a format problem. Re-run later.")
    print(f"\ntokens used by this check: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion; "
          f"{failures} of {len(table)} cells not PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
