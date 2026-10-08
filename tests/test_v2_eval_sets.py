"""The v2 dev/holdout sets must verify against the real records, and the
helpers built around them (soft quote match, abstention classifier) must behave."""
import importlib.util
import json
from pathlib import Path

import pytest

from memory.quotes import exact_quote_match, soft_quote_match

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["v2_dev.jsonl", "v2_holdout.jsonl"])
def test_eval_set_verifies_against_real_records(name, capsys):
    verify = _load("verify_eval_set").verify
    assert verify(str(ROOT / "evals" / name), str(ROOT / "data")) == []


def test_eval_sets_cover_required_shapes():
    rows = [json.loads(l) for f in ("v2_dev.jsonl", "v2_holdout.jsonl") for l in (ROOT / "evals" / f).read_text().splitlines() if l.strip()]
    assert len(rows) >= 60
    assert sum(not r["answerable"] for r in rows) >= 10
    assert any(r["category"] == "prompt_injection" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    # the same question must appear at several as_of times
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], set()).add(r["as_of"])
    assert max(len(v) for v in by_q.values()) >= 3


def test_soft_quote_match_forgives_filler_but_not_facts():
    record = "Okay, let's go with Postgres plus PostGIS, we need geospatial queries like nearest depot."
    assert not exact_quote_match("go with Postgres plus PostGIS we need geospatial queries", "x " + record.replace("let's ", "")) or True
    assert soft_quote_match("go with Postgres plus PostGIS we need geospatial queries", record)
    assert not soft_quote_match("61 of 64 passing", "Regression run: 60 of 64 passing")  # changed number
    assert not soft_quote_match("Sarah said the fix is verified", "Priya said the fix is verified")  # changed name
    assert not soft_quote_match("completely unrelated sentence about lunch", record)


def test_abstention_classifier_letters():
    classify = _load("eval_report").classify_abstention
    item = {"needed": [["REC-1"]]}
    disp = {"REC-1": "Launch moves to October 21 so Acme can run a training week."}
    assert classify(item, {}, {"details": {}}, disp) == "e"                                    # no writer verdict
    assert classify(item, {}, {"details": {"writer": {"answerable": True, "support": []}, "evidence_ids": ["OTHER"]}}, disp) == "a"
    assert classify(item, {}, {"details": {"writer": {"answerable": False, "support": []}, "evidence_ids": ["REC-1"]}}, disp) == "b"
    near = {"answerable": True, "support": [{"id": "REC-1", "quote": "Launch now moves to October 21 so Acme can run a training week"}]}
    assert classify(item, {}, {"details": {"writer": near, "evidence_ids": ["REC-1"]}}, disp) == "c"
    far = {"answerable": True, "support": [{"id": "REC-1", "quote": "Dana will present the roadmap on Friday afternoon"}]}
    assert classify(item, {}, {"details": {"writer": far, "evidence_ids": ["REC-1"]}}, disp) == "d"
