#!/usr/bin/env bash
# run.sh — entry point for the Candor memory system
# Usage:
#   ./run.sh memory  <questions.jsonl> <answers.jsonl>
#   ./run.sh actions <commands.jsonl>  <predictions.jsonl>
#   ./run.sh score-memory <answers.jsonl>
#
# DATA_DIR: override the data directory (default: ./data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"

# ── Bootstrap: create venv and install deps on first run ──────────────────────
if [ ! -f "$VENV/bin/python" ]; then
  echo "[run.sh] Creating virtual environment at $VENV ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

# Some system Python builds create a venv without a pip launcher.  Repair it
# here so a fresh checkout and an existing partial venv behave identically.
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "[run.sh] pip missing; trying python3 -m ensurepip --upgrade ..." >&2
  "$VENV/bin/python" -m ensurepip --upgrade
fi

# Always ensure requirements are up-to-date (cheap if already installed)
"$VENV/bin/python" -m pip install --quiet --upgrade pip || {
  echo "[run.sh] pip failed; trying ensurepip repair ..." >&2
  "$VENV/bin/python" -m ensurepip --upgrade || {
    echo "[run.sh] Could not repair pip. Install pip for python3, then retry." >&2
    exit 1
  }
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
}
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  "$VENV/bin/python" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

# Load .env if it exists (non-exported vars only, safe to re-run)
if [ -f "$SCRIPT_DIR/.env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^\s*#' "$SCRIPT_DIR/.env" | grep -v '^\s*$' | xargs)
fi

export DATA_DIR

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in
  memory)
    # ./run.sh memory <questions.jsonl> <answers.jsonl>
    if [ $# -lt 2 ]; then
      echo "Usage: ./run.sh memory <questions.jsonl> <answers.jsonl>" >&2
      exit 1
    fi
    if [ "${USE_DENSE:-true}" != "0" ] && [ "${USE_DENSE:-true}" != "false" ]; then
      "$VENV/bin/python" -m memory.cli warmup || {
        echo "[run.sh] WARNING: embedding warmup failed; answering BM25-only." >&2
        USE_DENSE=false "$VENV/bin/python" -m memory.cli answer --questions "$1" --out "$2"
        exit 0
      }
    fi
    "$VENV/bin/python" -m memory.cli answer --questions "$1" --out "$2"
    ;;

  actions)
    # ./run.sh actions <commands.jsonl> <predictions.jsonl>
    if [ $# -lt 2 ]; then
      echo "Usage: ./run.sh actions <commands.jsonl> <predictions.jsonl>" >&2
      exit 1
    fi
    if ! "$VENV/bin/python" -c "import actions.cli" 2>/dev/null; then
      echo "[run.sh] actions module does not exist yet. Skipping." >&2
      exit 1
    fi
    "$VENV/bin/python" -m actions.cli --commands "$1" --out "$2"
    ;;

  score-memory)
    # ./run.sh score-memory <answers.jsonl>
    if [ $# -lt 1 ]; then
      echo "Usage: ./run.sh score-memory <answers.jsonl>" >&2
      exit 1
    fi
    ANSWERS="$1"
    GOLD="$SCRIPT_DIR/evals/memory_train.jsonl"
    echo "=== Retrieval score ==="
    "$VENV/bin/python" "$SCRIPT_DIR/eval_harness/score_retrieval.py" \
      --gold "$GOLD" --answers "$ANSWERS" --data "$DATA_DIR"
    echo ""
    echo "=== Answer score (rules only, no judge) ==="
    "$VENV/bin/python" "$SCRIPT_DIR/eval_harness/score_memory.py" \
      --gold "$GOLD" --answers "$ANSWERS" --judge none
    ;;

  *)
    echo "Usage: ./run.sh {memory|actions|score-memory} [args...]" >&2
    echo ""
    echo "  memory  <questions.jsonl> <answers.jsonl>   — run the memory system"
    echo "  actions <commands.jsonl> <predictions.jsonl> — run the actions system (bonus)"
    echo "  score-memory <answers.jsonl>                 — score memory answers on train set"
    exit 1
    ;;
esac
