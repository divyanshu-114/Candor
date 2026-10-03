"""Run eval_harness/score_memory.py (unmodified) with a custom User-Agent.

WHY: the official judge uses urllib's default "Python-urllib/3.x" agent, which
Cloudflare blocks (HTTP 403, error 1010) on Groq's API. This wrapper installs
an opener with a normal User-Agent and then calls the real scorer's main().

Usage (judge = the fast Groq model; the key is read from .env, never printed):
  scripts/with_keys.sh .venv/bin/python scripts/score_memory_judged.py --gold G --answers A [--out O]
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval_harness"))

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "candor-eval/1.0 (+judge)")]
urllib.request.install_opener(opener)

# Point the scorer's OpenAI-compatible judge at Groq's fast model.
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("GROQ_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
model = os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")

import score_memory  # noqa: E402  (the unmodified official scorer)

sys.argv = [sys.argv[0], "--judge", "openai", "--model", model, *sys.argv[1:]]
score_memory.main()
