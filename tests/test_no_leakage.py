"""Guards against train-set leakage into the implementation.

WHY: the hidden test set reuses this code unchanged. If a prompt, rule or
constant was tuned to a specific train question/answer/id, it would look
good on evals/memory_train.jsonl but be a question-specific hack that the
brief explicitly forbids (PROJECT_RULES.md #7).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "evals" / "memory_train.jsonl"
CODE_FILES = sorted((ROOT / "memory").glob("*.py"))


def _flatten(value) -> str:
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value) if value else ""


def _load_train():
    rows = [json.loads(line) for line in TRAIN_FILE.open() if line.strip()]
    questions = [r["question"] for r in rows]
    gold_answers = [_flatten(r.get("gold_answer", "")) for r in rows]
    train_ids = [r["id"] for r in rows]
    needed_ids = set()
    for r in rows:
        for group in r.get("needed", []):
            needed_ids.update(group)
    return questions, gold_answers, train_ids, needed_ids


def _source_blob() -> str:
    return "\n".join(f.read_text() for f in CODE_FILES)


def test_no_train_question_ids_in_code():
    _, _, train_ids, _ = _load_train()
    blob = _source_blob()
    hits = [tid for tid in train_ids if tid in blob]
    assert not hits, f"train question ids referenced in memory/*.py: {hits}"


def test_no_train_question_sentences_in_code():
    questions, _, _, _ = _load_train()
    blob = _source_blob()
    hits = [q for q in questions if q in blob]
    assert not hits, f"train question sentences copied into memory/*.py: {hits}"


def test_no_train_gold_answers_in_code():
    _, gold_answers, _, _ = _load_train()
    blob = _source_blob()
    # Only check substrings long enough to be distinctive (avoid flagging
    # single common words that legitimately appear in prose/docstrings).
    hits = [g for g in gold_answers if len(g) > 15 and g in blob]
    assert not hits, f"train gold answer text copied into memory/*.py: {hits}"


def test_no_train_needed_record_ids_in_code():
    _, _, _, needed_ids = _load_train()
    blob = _source_blob()
    hits = [nid for nid in needed_ids if nid in blob]
    assert not hits, f"train 'needed' record ids referenced in memory/*.py: {hits}"
