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
EVAL_FILES = ["memory_train.jsonl", "memory_dev.jsonl", "v2_dev.jsonl", "v2_holdout.jsonl"]
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
