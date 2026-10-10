"""Guards against eval-set leakage into the implementation.

WHY: the hidden test set reuses this code unchanged. If a prompt, rule or
constant was tuned to a specific train/dev/holdout question, answer or record
id, it would look good on our own files but be a question-specific hack that
the brief forbids (PROJECT_RULES.md #7). Every memory eval file we ship is
checked against all implementation code (memory/, actions/, scripts/*.py that
feed prompts). Add new eval files to EVAL_FILES.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILES = ["memory_train.jsonl", "memory_dev.jsonl", "v2_dev.jsonl", "v2_holdout.jsonl", "v2_dev_half.jsonl", "v2_holdout2.jsonl"]
CODE_FILES = sorted(list((ROOT / "memory").glob("*.py")) + list((ROOT / "actions").glob("*.py")))
# Ids that are generic structural vocabulary rather than distinctive facts.
GENERIC_IDS: set[str] = set()


def _flatten(value) -> str:
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value) if value else ""


def _load(name: str):
    rows = [json.loads(line) for line in (ROOT / "evals" / name).open() if line.strip()]
    needed_ids: set[str] = set()
    for r in rows:
        for group in r.get("needed", []):
            needed_ids.update(group)
        needed_ids.update(r.get("evidence", []))
    return rows, needed_ids


def _source_blob() -> str:
    return "\n".join(f.read_text() for f in CODE_FILES)


@pytest.mark.parametrize("name", EVAL_FILES)
def test_no_question_ids_in_code(name):
    rows, _ = _load(name)
    blob = _source_blob()
    hits = [r["id"] for r in rows if r["id"] in blob]
    assert not hits, f"{name}: question ids referenced in code: {hits}"


@pytest.mark.parametrize("name", EVAL_FILES)
def test_no_question_sentences_in_code(name):
    rows, _ = _load(name)
    blob = _source_blob()
    hits = [r["question"] for r in rows if r["question"] in blob]
    assert not hits, f"{name}: question sentences copied into code: {hits}"


@pytest.mark.parametrize("name", EVAL_FILES)
def test_no_gold_answers_in_code(name):
    rows, _ = _load(name)
    blob = _source_blob()
    # Only substrings long enough to be distinctive (avoid flagging common words).
    hits = [_flatten(r.get("gold_answer", "")) for r in rows
            if len(_flatten(r.get("gold_answer", ""))) > 15 and _flatten(r.get("gold_answer", "")) in blob]
    assert not hits, f"{name}: gold answer text copied into code: {hits}"


@pytest.mark.parametrize("name", EVAL_FILES)
def test_no_needed_record_ids_in_code(name):
    _, needed_ids = _load(name)
    blob = _source_blob()
    hits = sorted(i for i in needed_ids if i not in GENERIC_IDS and i in blob)
    assert not hits, f"{name}: needed/evidence record ids referenced in code: {hits}"


def test_eval_files_exist_and_are_nonempty():
    for name in EVAL_FILES:
        assert (ROOT / "evals" / name).stat().st_size > 0, name


# --- data-derived distinctive terms must not appear in implementation code (comments / docstrings excluded) ------------------------
import ast
import re

_GENERIC_OK = {"linear", "github", "figma", "notion", "slack", "gmail", "google", "meet", "zoom", "calendar", "chatgpt", "codex", "email",
               "digest", "example", "weekly", "pipeline", "planner", "daily", "engineer"}  # app / product names a planner legitimately knows, and the generic source names


def _code_words(path: Path) -> str:
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)
                  and isinstance(n.body[0].value.value, str)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            out.append(n.value)
        elif isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
    return " ".join(out).lower()


def _data_terms() -> set[str]:
    """People, organisations, products, places and projects named in the data (capitalised words of names / titles / summaries)."""
    data = ROOT / "data"
    terms: set[str] = set()
    for u in json.loads((data / "connectors/slack/users.json").read_text()):
        terms.update(re.findall(r"[A-Za-z]{4,}", u.get("real_name") or ""))
    for line in (data / "connectors/google_calendar/events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        terms.update(re.findall(r"\b[A-Z][a-z]{3,}\b", ev.get("summary", "")))
        for a in ev.get("attendees", []):
            dom = a.get("email", "").split("@")[-1].split(".")[0]
            if len(dom) > 4:
                terms.add(dom)
    for line in (data / "connectors/gmail/messages.jsonl").read_text().splitlines():
        m = json.loads(line)
        for hdr in [m.get("from", "")] + m.get("to", []):
            terms.update(re.findall(r"\b[A-Z][a-z]{3,}\b", hdr.split("<")[0]))
    for f in (data / "native/meetings").glob("*.json"):
        terms.update(re.findall(r"\b[A-Z][a-z]{3,}\b", json.loads(f.read_text()).get("title", "")))
    return {t.lower() for t in terms} - _GENERIC_OK - {"alex"}   # the memory owner's first name is allowed in no code either; see below


def test_no_data_specific_names_in_implementation_code():
    terms = _data_terms()
    hits = {}
    for path in CODE_FILES:
        words = set(re.findall(r"[a-z]{4,}", _code_words(path)))
        for t in terms & words:
            hits.setdefault(t, []).append(path.name)
    assert not hits, f"names from the data appear in code (strings/identifiers): {hits}"


def test_no_eval_file_distinctive_phrases_in_code():
    """Six-word runs from questions / commands must not appear verbatim in code strings."""
    blob = " ".join(_code_words(p) for p in CODE_FILES)
    bad = []
    for name in EVAL_FILES + ["actions_train.jsonl", "actions_dev.jsonl", "actions_v2_dev.jsonl", "v2_holdout2.jsonl", "actions_holdout2.jsonl"]:
        path = ROOT / "evals" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            for key in ("question", "command"):
                words = re.findall(r"[a-z0-9]+", r.get(key, "").lower())
                for i in range(len(words) - 5):
                    if " ".join(words[i:i + 6]) in " ".join(re.findall(r"[a-z0-9]+", blob)):
                        bad.append((name, " ".join(words[i:i + 6])))
    assert not bad, f"6-word phrases from eval files found in code: {bad[:5]}"
