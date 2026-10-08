"""Loud failures: a run without a model must say so, and --strict must fail it (Phase 6)."""
import json
import sys

import pytest

from memory import cli, config, degraded


@pytest.fixture()
def questions(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"id": "Q1", "question": "When is the launch?", "as_of": "2026-09-04T00:00:00-07:00"}) + "\n")
    return q


def _run(corpus_dir, questions, tmp_path, monkeypatch, capsys, extra=()):
    monkeypatch.setattr("memory.llm._PROVIDERS", [])                       # no provider at all = no key
    monkeypatch.setattr("memory.llm.GROQ_API_KEY", None)
    monkeypatch.setattr(config, "USE_DENSE", False)
    monkeypatch.setattr(config, "USE_LANES", False)
    monkeypatch.setenv("DATA_DIR", str(corpus_dir))
    argv = ["memory.cli", "answer", "--questions", str(questions), "--out", str(tmp_path / "a.jsonl"),
            "--data-dir", str(corpus_dir), "--cache-dir", str(tmp_path / ".c"), "--workers", "1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "RUN_STATS_PATH", tmp_path / "stats.json")
    monkeypatch.setattr(cli, "DIAGNOSTICS_PATH", tmp_path / "diag.jsonl")
    cli.main()
    return capsys.readouterr().out


def test_banner_names_each_degraded_stage_and_count(corpus_dir, questions, tmp_path, monkeypatch, capsys):
    out = _run(corpus_dir, questions, tmp_path, monkeypatch, capsys)
    assert "DEGRADED STAGES" in out and "analysis" in out and "1/1" in out
    diag = [json.loads(l) for l in (tmp_path / "diag.jsonl").read_text().splitlines()]
    assert "analysis" in diag[0]["degraded_stages"]                       # per-question record on disk


def test_strict_exits_nonzero_when_anything_degraded(corpus_dir, questions, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(corpus_dir, questions, tmp_path, monkeypatch, capsys, extra=("--strict",))
    assert exc.value.code == 3


def test_banner_for_clean_run_says_so():
    counts, n = degraded.summarize([{"degraded_stages": []}, {"degraded_stages": []}])
    assert "none" in degraded.banner(counts, n, "x.jsonl")
    counts, n = degraded.summarize([{"degraded_stages": ["rerank"]}, {"degraded_stages": ["rerank", "writer"]}])
    text = degraded.banner(counts, n, "x.jsonl")
    assert "rerank" in text and "2/2" in text and "writer" in text and "1/2" in text
