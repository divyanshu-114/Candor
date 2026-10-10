"""Both CLIs write each result as soon as it (and every earlier one) is
complete, in input order; a killed run keeps its finished prefix and
--resume completes the rest with no duplicates. Fake workers, no LLM."""
import json
import threading
import time
from pathlib import Path

import pytest

import actions.cli as acli
import memory.cli as mcli
from memory import ordered_io


def _write_items(path: Path, n: int, key: str):
    path.write_text("".join(json.dumps({"id": f"Q{i}", key: f"text {i}", "as_of": "2026-09-18T09:00:00-07:00"}) + "\n"
                            for i in range(n)))


def _ids(path: Path):
    return [json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()]


def _fake_memory(kill_at=None, delay_first=0.0):
    def fake(item, data_dir, cache_dir, retrieval_only):
        if item["id"] == "Q0" and delay_first:
            time.sleep(delay_first)  # an early item that finishes AFTER later ones
        if kill_at is not None and item["id"] == kill_at:
            raise KeyboardInterrupt
        row = {"id": item["id"], "answer": "a", "sources": [], "retrieved": [], "abstained": False}
        return json.dumps(row) + "\n", {"id": item["id"], "degraded_stages": []}
    return fake


def _fake_actions(kill_at=None, delay_first=0.0):
    def fake(item, data_dir, cache_dir):
        if item["id"] == "Q0" and delay_first:
            time.sleep(delay_first)
        if kill_at is not None and item["id"] == kill_at:
            raise KeyboardInterrupt
        return json.dumps({"id": item["id"], "actions": []}) + "\n"
    return fake


def _run_memory(tmp_path, **kw):
    mcli.answer(tmp_path / "q.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=kw.pop("workers", 1),
                stats_path=tmp_path / "stats.json", diagnostics_path=tmp_path / "diag.jsonl", **kw)


def test_memory_killed_run_keeps_prefix_and_resume_finishes(tmp_path, monkeypatch):
    _write_items(tmp_path / "q.jsonl", 6, "question")
    monkeypatch.setattr(mcli, "_process_one", _fake_memory(kill_at="Q3"))
    with pytest.raises(KeyboardInterrupt):
        _run_memory(tmp_path)
    assert _ids(tmp_path / "out.jsonl") == ["Q0", "Q1", "Q2"]           # valid prefix survived
    assert len((tmp_path / "diag.jsonl").read_text().splitlines()) == 3

    monkeypatch.setattr(mcli, "_process_one", _fake_memory())
    _run_memory(tmp_path, resume=True)
    assert _ids(tmp_path / "out.jsonl") == [f"Q{i}" for i in range(6)]   # complete, ordered, no duplicates


def test_actions_killed_run_keeps_prefix_and_resume_finishes(tmp_path, monkeypatch):
    _write_items(tmp_path / "c.jsonl", 6, "command")
    monkeypatch.setattr(acli, "_process_one", _fake_actions(kill_at="Q2"))
    with pytest.raises(KeyboardInterrupt):
        acli.answer(tmp_path / "c.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=1)
    assert _ids(tmp_path / "out.jsonl") == ["Q0", "Q1"]
    monkeypatch.setattr(acli, "_process_one", _fake_actions())
    acli.answer(tmp_path / "c.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=1, resume=True)
    assert _ids(tmp_path / "out.jsonl") == [f"Q{i}" for i in range(6)]


def test_results_are_written_in_input_order_even_if_early_item_is_slow(tmp_path, monkeypatch):
    _write_items(tmp_path / "c.jsonl", 5, "command")
    monkeypatch.setattr(acli, "_process_one", _fake_actions(delay_first=0.3))
    acli.answer(tmp_path / "c.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=4)
    assert _ids(tmp_path / "out.jsonl") == [f"Q{i}" for i in range(5)]


def test_a_line_is_on_disk_before_the_run_ends(tmp_path, monkeypatch):
    """Incremental, not end-of-run: while the last item is still working, the
    earlier lines are already in --out."""
    _write_items(tmp_path / "c.jsonl", 3, "command")
    seen = {}
    release = threading.Event()

    def fake(item, data_dir, cache_dir):
        if item["id"] == "Q2":
            deadline = time.time() + 5
            while time.time() < deadline and len(_ids(tmp_path / "out.jsonl")) < 2:
                time.sleep(0.02)
            seen["during"] = _ids(tmp_path / "out.jsonl")
        return json.dumps({"id": item["id"], "actions": []}) + "\n"

    monkeypatch.setattr(acli, "_process_one", fake)
    acli.answer(tmp_path / "c.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=3)
    assert seen["during"][:2] == ["Q0", "Q1"]


def test_resume_drops_a_truncated_last_line_and_reorders(tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text(json.dumps({"id": "Q1"}) + "\n" + json.dumps({"id": "Q0"}) + "\n" + '{"id": "Q2", "acti')
    kept = ordered_io.start_output(out, resume=True)
    assert set(kept) == {"Q0", "Q1"}
    ordered_io.finalize_order(out, ["Q0", "Q1", "Q2"])
    assert _ids(out) == ["Q0", "Q1"]


def test_resume_does_not_duplicate_existing_ids(tmp_path, monkeypatch):
    _write_items(tmp_path / "c.jsonl", 3, "command")
    (tmp_path / "out.jsonl").write_text(json.dumps({"id": "Q0", "actions": [{"type": "x"}]}) + "\n")
    monkeypatch.setattr(acli, "_process_one", _fake_actions())
    acli.answer(tmp_path / "c.jsonl", tmp_path / "out.jsonl", data_dir="x", workers=1, resume=True)
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["Q0", "Q1", "Q2"]
    assert rows[0]["actions"] == [{"type": "x"}]   # kept as-is, not recomputed


pytestmark = pytest.mark.slow
