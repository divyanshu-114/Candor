#!/usr/bin/env bash
# scripts/release_check.sh — pre-submission gate. Runs on macOS and Linux.
#
# Checks, in order (any failure aborts with a non-zero exit code):
#   (a) working tree is clean (no uncommitted changes)
#   (b) no secret-shaped key is tracked by git (files or history), ignoring
#       the detector regexes themselves in scripts/audit_data.py / memory/safety.py
#   (c) a fresh clone + fresh venv + requirements install passes the unit tests
#   (d) `./run.sh memory` on evals/memory_train.jsonl with NO API key produces
#       27 valid lines (all fields present, <=20 unique retrieved ids), scored
#       by score_retrieval.py
#   (e) no forbidden id appears in any question's `retrieved` at its as_of
#   (f) same clone's `./run.sh actions` on evals/actions_train.jsonl produces
#       valid output
#   (g) prints the git commit hash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0
step() { echo ""; echo "=== $* ==="; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }

# ── (a) clean working tree ───────────────────────────────────────────────
step "(a) working tree clean"
if [ -n "$(git status --porcelain)" ]; then
  fail "working tree is dirty:"
  git status --porcelain >&2
else
  pass "working tree is clean"
fi

# ── (b) no secret-shaped key tracked, in files or history ───────────────
step "(b) no tracked secret-shaped keys"
# Key-pattern substrings to look for. These are intentionally looser than
# the real detector regexes (memory/safety.py, scripts/audit_data.py) --
# this check's job is "does a real-looking key sit in the repo", not to
# re-validate the detectors. We therefore explicitly exclude those two
# files' own lines (they legitimately contain the pattern literals), plus
# tests/ (which legitimately constructs synthetic secret-shaped fixture
# strings to verify the masking logic itself -- same rationale as the
# detector files: this is code ABOUT secrets, not a leaked one), from
# every scan below. (This file and this comment are themselves excluded
# below for the same reason -- it names the patterns it looks for.)
KEY_PATTERNS='gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|xox[bpoa]-[A-Za-z0-9-]{10,}'
DETECTOR_FILES="scripts/audit_data.py memory/safety.py"
# One documented, obviously synthetic fixture (sequential digits) that the
# docs quote when explaining the masking tests. Lines containing exactly this
# literal are ignored; any other key-shaped string still fails the check.
SYNTHETIC_FIXTURE="sk-12345678901234567890"

if git ls-files | grep -qx '\.env$'; then
  fail ".env is tracked by git"
else
  pass ".env is not tracked"
fi

TRACKED_HIT=0
for f in $(git ls-files); do
  case " $DETECTOR_FILES " in *" $f "*) continue;; esac
  case "$f" in tests/*) continue;; esac
  [ -f "$f" ] || continue
  if grep -IE "$KEY_PATTERNS" "$f" 2>/dev/null | grep -vqF "$SYNTHETIC_FIXTURE"; then
    echo "  possible key-shaped string in tracked file: $f" >&2
    TRACKED_HIT=1
  fi
done
if [ "$TRACKED_HIT" -eq 1 ]; then
  fail "key-shaped string(s) found in tracked files (see above)"
else
  pass "no key-shaped string in any tracked file"
fi

# History scan: every blob ever committed, minus the two detector files'
# current+historical content (approximated by filtering out lines that are
# themselves the detector regex literals -- a real key has no word
# boundary making it look like a regex source line).
# Only scan actual ADDED file-content lines ("+..." diff lines, not "+++"
# file headers) -- this deliberately excludes commit message text (which
# git log -p prints indented, with no leading "+") so a commit message
# that merely *discusses* a fixture string (as this script's own history
# now does) can't trip a false positive the way a real committed secret
# would.
HISTORY_HIT=0
HISTORY_MATCHES="$(git log -p --all -- . ':!scripts/audit_data.py' ':!memory/safety.py' ':!tests/*' 2>/dev/null \
  | grep -E '^\+[^+]' | grep -E "$KEY_PATTERNS" | grep -vF "$SYNTHETIC_FIXTURE" | grep -v '^\+[[:space:]]*#' || true)"
if [ -n "$HISTORY_MATCHES" ]; then
  echo "  possible key-shaped string(s) in git history:" >&2
  echo "$HISTORY_MATCHES" | head -20 >&2
  HISTORY_HIT=1
fi
if [ "$HISTORY_HIT" -eq 1 ]; then
  fail "key-shaped string(s) found in git history (see above)"
else
  pass "no key-shaped string in git history (outside the detector files)"
fi

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "Aborting before the expensive clone/test steps -- fix the secret-hygiene failure(s) above first." >&2
  exit 1
fi

# ── (c) fresh clone, fresh venv, install, unit tests ─────────────────────
step "(c) fresh clone + venv + unit tests"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
CLONE_DIR="$TMP_DIR/clone"

git clone --quiet "$REPO_ROOT" "$CLONE_DIR"
"${PYTHON:-python3}" -m venv "$CLONE_DIR/.venv"
"$CLONE_DIR/.venv/bin/python" -m pip install --quiet --upgrade pip
"$CLONE_DIR/.venv/bin/python" -m pip install --quiet -r "$CLONE_DIR/requirements.txt"

T0=$(date +%s)
if (cd "$CLONE_DIR" && PYTHONPATH=. "$CLONE_DIR/.venv/bin/python" -m pytest -q); then
  pass "fast unit tests pass in the fresh clone ($(( $(date +%s) - T0 ))s)"
else
  fail "fast unit tests failed in the fresh clone"
fi
T1=$(date +%s)
if (cd "$CLONE_DIR" && PYTHONPATH=. "$CLONE_DIR/.venv/bin/python" -m pytest -q -m slow); then
  pass "slow/integration tests pass in the fresh clone ($(( $(date +%s) - T1 ))s)"
else
  fail "slow/integration tests failed in the fresh clone"
fi

# ── (d) memory, no key, 27 valid lines ───────────────────────────────────
step "(d) memory run, no API key"
MEM_OUT="$TMP_DIR/a.jsonl"
(
  cd "$CLONE_DIR"
  rm -f .env
  env -u GROQ_API_KEY -u LLM_PROVIDERS -u GEMINI_API_KEY -u LLM_BASE_URL -u LLM_MODEL_STRONG -u LLM_MODEL_FAST \
    ./run.sh memory evals/memory_train.jsonl "$MEM_OUT"
)
MEM_RUN_STATUS=$?

if [ "$MEM_RUN_STATUS" -ne 0 ] || [ ! -f "$MEM_OUT" ]; then
  fail "memory run did not produce an output file"
else
  LINE_COUNT=$(wc -l < "$MEM_OUT" | tr -d ' ')
  if [ "$LINE_COUNT" != "27" ]; then
    fail "expected 27 lines in memory output, got $LINE_COUNT"
  else
    pass "27 lines produced"
  fi

  "$CLONE_DIR/.venv/bin/python" - "$MEM_OUT" <<'PYEOF'
import json, sys
path = sys.argv[1]
required = {"id", "answer", "sources", "retrieved", "abstained"}
bad = []
for i, line in enumerate(open(path), 1):
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        bad.append(f"line {i}: invalid JSON ({e})")
        continue
    missing = required - row.keys()
    if missing:
        bad.append(f"line {i} ({row.get('id')}): missing fields {missing}")
    retrieved = row.get("retrieved") or []
    if len(retrieved) > 20:
        bad.append(f"line {i} ({row.get('id')}): retrieved has {len(retrieved)} > 20 ids")
    ids = [r.get("id") if isinstance(r, dict) else r for r in retrieved]
    if len(set(ids)) != len(ids):
        bad.append(f"line {i} ({row.get('id')}): retrieved has duplicate ids")
if bad:
    print("FIELD CHECK FAILED:")
    for b in bad:
        print(" -", b)
    sys.exit(1)
print("FIELD CHECK OK: every line has all required fields, retrieved <=20 unique ids")
PYEOF
  if [ $? -eq 0 ]; then
    pass "memory output field check"
  else
    fail "memory output field check"
  fi

  echo ""
  echo "--- retrieval scorer summary (no-key run) ---"
  (cd "$CLONE_DIR" && "$CLONE_DIR/.venv/bin/python" eval_harness/score_retrieval.py \
    --gold evals/memory_train.jsonl --answers "$MEM_OUT" --quiet) || fail "score_retrieval.py errored"
fi

# ── (e) no forbidden id in any question's retrieved, at its own as_of ───
step "(e) no forbidden id at any question's as_of"
(cd "$CLONE_DIR" && PYTHONPATH="$CLONE_DIR/eval_harness" "$CLONE_DIR/.venv/bin/python" - "$MEM_OUT" <<'PYEOF'
import json, sys
from datetime import datetime
import records

answers_path = sys.argv[1]
gold = [json.loads(l) for l in open("evals/memory_train.jsonl") if l.strip()]
answers = {}
for line in open(answers_path):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    answers[row["id"]] = row

ctx = records.context("data")
avail, deleted = ctx["avail"], ctx["deleted"]

violations = []
for item in gold:
    ans = answers.get(item["id"])
    if not ans:
        continue
    as_of = datetime.fromisoformat(item["as_of"])
    for cid in (ans.get("retrieved") or [])[:20]:
        cid = cid.get("id") if isinstance(cid, dict) else cid
        t = avail.get(cid)
        if t and t > as_of:
            violations.append(f"{item['id']}: retrieved {cid}, not delivered until {t} (as_of {as_of})")
        elif cid in deleted and deleted[cid] <= as_of:
            violations.append(f"{item['id']}: retrieved {cid}, deleted at {deleted[cid]} (as_of {as_of})")

if violations:
    print("FORBIDDEN ID CHECK FAILED:")
    for v in violations:
        print(" -", v)
    sys.exit(1)
print(f"FORBIDDEN ID CHECK OK: 0 violations across {len(gold)} questions")
PYEOF
)
if [ $? -eq 0 ]; then
  pass "no forbidden id leaked"
else
  fail "forbidden id check failed"
fi

# ── (f) actions, no key ──────────────────────────────────────────────────
step "(f) actions run, no API key"
ACT_OUT="$TMP_DIR/actions_out.jsonl"
(
  cd "$CLONE_DIR"
  env -u GROQ_API_KEY -u LLM_PROVIDERS -u GEMINI_API_KEY -u LLM_BASE_URL -u LLM_MODEL_STRONG -u LLM_MODEL_FAST \
    ./run.sh actions evals/actions_train.jsonl "$ACT_OUT"
)
if [ $? -ne 0 ] || [ ! -f "$ACT_OUT" ]; then
  fail "actions run did not produce an output file"
else
  ACT_LINES=$(wc -l < "$ACT_OUT" | tr -d ' ')
  GOLD_LINES=$(wc -l < "$CLONE_DIR/evals/actions_train.jsonl" | tr -d ' ')
  if [ "$ACT_LINES" != "$GOLD_LINES" ]; then
    fail "expected $GOLD_LINES action lines, got $ACT_LINES"
  else
    pass "$ACT_LINES action lines produced"
  fi
  "$CLONE_DIR/.venv/bin/python" - "$ACT_OUT" <<'PYEOF'
import json, sys
bad = []
for i, line in enumerate(open(sys.argv[1]), 1):
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        bad.append(f"line {i}: invalid JSON ({e})")
        continue
    if "id" not in row or "actions" not in row:
        bad.append(f"line {i}: missing 'id' or 'actions'")
if bad:
    print("ACTIONS FIELD CHECK FAILED:")
    for b in bad:
        print(" -", b)
    sys.exit(1)
print("ACTIONS FIELD CHECK OK")
PYEOF
  if [ $? -eq 0 ]; then
    pass "actions output field check"
  else
    fail "actions output field check"
  fi
fi

# ── (g) commit hash ──────────────────────────────────────────────────────
step "(g) commit hash"
echo "$(git -C "$REPO_ROOT" rev-parse HEAD)"

echo ""
if [ "$FAIL" -ne 0 ]; then
  echo "=== RELEASE CHECK: FAILED ==="
  exit 1
else
  echo "=== RELEASE CHECK: PASSED ==="
  exit 0
fi
