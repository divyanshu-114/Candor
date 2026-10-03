"""Concurrency must never reorder output, and --resume must never recompute."""
import json
import random
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from memory import answer as answer_module
from memory import cli


def _slack_dataset(d: Path, n: int) -> list[dict]:
    (d / "connectors/slack").mkdir(parents=True)
    items = []
    with open(d / "connectors/slack/messages.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"SL-{i}", "user": "U1",
                                 "ts": f"2026-09-08T{10 + i % 10:02d}:00:00-07:00",
                                 "text": f"message number {i} about widgets"}) + "\n")
    for i in range(n):
        items.append({"id": f"Q-{i}", "question": f"widget message {i}", "as_of": "2026-09-09T00:00:00-07:00"})
    return items


def test_output_order_matches_input_order_under_concurrency():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        items = _slack_dataset(d, 12)
        questions_path = d / "q.jsonl"
        out_path = d / "a.jsonl"
        with questions_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        # Randomize how long each "retrieve" takes so completion order != input order.
        real_retrieve = answer_module.retrieve

        def jittered_retrieve(question, as_of, **kwargs):
            time.sleep(random.uniform(0, 0.02))
            return real_retrieve(question, as_of, **kwargs)

        # Not about answer-writing quality; force no-key (extractive) mode so
        # this test is fast and deterministic regardless of the host's env.
        with patch("memory.answer.retrieve", side_effect=jittered_retrieve), \
             patch("memory.llm.GROQ_API_KEY", None):
            cli.answer(questions_path, out_path, data_dir=str(d), cache_dir=str(d / ".cache"), workers=6, stats_path=d / "run_stats.json", diagnostics_path=d / "diagnostics.jsonl")

        ids = [json.loads(line)["id"] for line in out_path.read_text().splitlines()]
        assert ids == [item["id"] for item in items]


def test_resume_does_not_recompute_existing_answers():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        items = _slack_dataset(d, 5)
        questions_path = d / "q.jsonl"
        out_path = d / "a.jsonl"
        with questions_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        cli.answer(questions_path, out_path, data_dir=str(d), cache_dir=str(d / ".cache"), stats_path=d / "run_stats.json", diagnostics_path=d / "diagnostics.jsonl")
        first_pass = out_path.read_text()

        calls = {"n": 0}
        real_process = cli._process_one

        def counting_process(item, data_dir, cache_dir, retrieval_only=False):
            calls["n"] += 1
            return real_process(item, data_dir, cache_dir, retrieval_only)

        with patch("memory.cli._process_one", side_effect=counting_process):
            cli.answer(questions_path, out_path, data_dir=str(d), cache_dir=str(d / ".cache"), resume=True, stats_path=d / "run_stats.json", diagnostics_path=d / "diagnostics.jsonl")

        assert calls["n"] == 0, "resume recomputed answers that already existed in --out"
        assert out_path.read_text() == first_pass


def test_ids_recomputes_only_the_requested_subset():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        items = _slack_dataset(d, 5)
        questions_path = d / "q.jsonl"
        out_path = d / "a.jsonl"
        with questions_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        cli.answer(questions_path, out_path, data_dir=str(d), cache_dir=str(d / ".cache"),
                   stats_path=d / "run_stats.json", diagnostics_path=d / "diagnostics.jsonl")
        first_pass = {json.loads(l)["id"]: l for l in out_path.read_text().splitlines()}

        calls = {"ids": []}
        real_process = cli._process_one

        def recording_process(item, data_dir, cache_dir, retrieval_only=False):
            calls["ids"].append(item["id"])
            return real_process(item, data_dir, cache_dir, retrieval_only)

        with patch("memory.cli._process_one", side_effect=recording_process):
            cli.answer(questions_path, out_path, data_dir=str(d), cache_dir=str(d / ".cache"),
                       stats_path=d / "run_stats.json", diagnostics_path=d / "diagnostics.jsonl",
                       ids={"Q-1", "Q-3"})

        assert calls["ids"] == ["Q-1", "Q-3"], "--ids computed something outside the requested subset"
        second_pass = {json.loads(l)["id"]: l for l in out_path.read_text().splitlines()}
        assert set(second_pass) == set(first_pass), "--ids run dropped ids that were already in --out"
        for qid in ("Q-0", "Q-2", "Q-4"):
            assert second_pass[qid] == first_pass[qid], f"{qid} was untouched by --ids but changed anyway"
