"""Our own dev eval files must work with the official scorers, unmodified.

(a) schema checks against what eval_harness/score_*.py actually read;
(b) round trip: build concrete predictions from each actions_dev matcher and
feed them through the real score_actions logic -- must be 100% pass.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEV_ACTIONS = ROOT / "evals" / "actions_dev.jsonl"
DEV_MEMORY = ROOT / "evals" / "memory_dev.jsonl"
V2_ACTIONS = ROOT / "evals" / "actions_v2_dev.jsonl"

_spec = importlib.util.spec_from_file_location("score_actions", ROOT / "eval_harness" / "score_actions.py")
score_actions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_actions)

MATCHER_KEYS = {"eq", "in", "contains_any", "contains_all", "set_eq", "includes", "datetime"}
ACTION_TYPES = {"slack.send_message", "gmail.send", "calendar.create_event", "calendar.update_event",
                "reminder.create", "memory.ask", "app.open", "clarify", "confirm"}


def _load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _check_action(a, where):
    assert isinstance(a, dict), f"{where}: action must be an object"
    assert a.get("type") in ACTION_TYPES, f"{where}: unknown type {a.get('type')!r}"
    assert isinstance(a.get("args"), dict), f"{where}: args must be an object"
    for k, m in a["args"].items():
        assert isinstance(m, dict) and (set(m) & MATCHER_KEYS), f"{where}.{k}: no known matcher in {m!r}"
        assert set(m) <= MATCHER_KEYS | {"tolerance_min"}, f"{where}.{k}: unknown matcher key in {m!r}"


@pytest.mark.parametrize("item", _load(DEV_ACTIONS) + _load(V2_ACTIONS), ids=lambda i: i["id"])
def test_actions_dev_schema(item):
    assert {"id", "command", "as_of", "expected", "alternatives"} <= set(item)
    assert isinstance(item["expected"], list) and item["expected"]
    for a in item["expected"]:
        _check_action(a, item["id"] + ".expected")
    assert isinstance(item["alternatives"], list)
    for alt in item["alternatives"]:
        assert isinstance(alt, list) and alt, f"{item['id']}: each alternative must be a non-empty list of actions"
        for a in alt:
            _check_action(a, item["id"] + ".alternative")


def test_ids_unique_and_nonempty():
    for path in (DEV_ACTIONS, DEV_MEMORY, V2_ACTIONS):
        ids = [i["id"] for i in _load(path)]
        assert ids and len(ids) == len(set(ids))


@pytest.mark.parametrize("item", _load(DEV_MEMORY), ids=lambda i: i["id"])
def test_memory_dev_schema(item):
    for k in ("id", "question", "as_of", "category", "answerable", "gold_answer", "key_terms", "needed"):
        assert k in item, f"{item['id']}: missing {k}"
    assert isinstance(item["answerable"], bool)
    for group in item["needed"]:
        assert isinstance(group, list) and group and all(isinstance(x, str) for x in group)
    if item["answerable"]:
        assert item["needed"], "an answerable question needs at least one needed group"
    for group in item["key_terms"] + item.get("stale", []):
        assert isinstance(group, list) and all(isinstance(x, str) for x in group)


def _concrete(matcher):
    if "eq" in matcher:
        return matcher["eq"]
    if "in" in matcher:
        return matcher["in"][0]
    if "contains_any" in matcher:
        return matcher["contains_any"][0]
    if "contains_all" in matcher:
        return " ".join(matcher["contains_all"])
    if "set_eq" in matcher:
        return list(matcher["set_eq"])
    if "includes" in matcher:
        return list(matcher["includes"])
    if "datetime" in matcher:
        return matcher["datetime"]
    raise AssertionError(f"unhandled matcher {matcher}")


def _predict(expected):
    return [{"type": a["type"], "args": {k: _concrete(m) for k, m in a["args"].items()}} for a in expected]


def test_actions_dev_round_trip_scores_100_percent():
    for item in _load(DEV_ACTIONS):
        ok, hits, total = score_actions.list_ok(item["expected"], _predict(item["expected"]))
        assert ok and hits == total, f"{item['id']} does not pass against its own expected list"
        for alt in item["alternatives"]:
            ok, _, _ = score_actions.list_ok(alt, _predict(alt))
            assert ok, f"{item['id']} alternative does not pass against itself"


def test_score_actions_cli_runs_on_dev_file(tmp_path):
    """The official script itself (not a copy) must accept the dev file."""
    import subprocess, sys
    preds = tmp_path / "p.jsonl"
    preds.write_text("\n".join(json.dumps({"id": i["id"], "actions": _predict(i["expected"])})
                               for i in _load(DEV_ACTIONS)))
    out = subprocess.run([sys.executable, str(ROOT / "eval_harness" / "score_actions.py"), "--gold", str(DEV_ACTIONS),
                          "--predictions", str(preds), "--out", str(tmp_path / "r.json")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "pass rate 100.0%" in out.stdout
