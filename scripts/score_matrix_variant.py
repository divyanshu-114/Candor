"""Score one matrix variant's retrieval-only output, plus report whether
every question was REAL (no degraded stage) or some were DEGRADED.

Usage: python3 scripts/score_matrix_variant.py outputs/matrix/C_full_off.jsonl
"""
import json
import subprocess
import sys


def main() -> None:
    path = sys.argv[1]
    diag_path = path.replace(".jsonl", "_diag.jsonl")
    stats_path = path.replace(".jsonl", "_stats.json")

    out = subprocess.run(
        ["python3", "eval_harness/score_retrieval.py", "--gold", "evals/memory_train.jsonl",
         "--answers", path, "--quiet"],
        capture_output=True, text=True,
    )
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)

    try:
        rows = [json.loads(l) for l in open(diag_path)]
        degraded_ids = [r["id"] for r in rows if r.get("degraded_stages")]
        print(f"REAL/DEGRADED: {'REAL (0 degraded)' if not degraded_ids else f'DEGRADED ({len(degraded_ids)}/{len(rows)}): {degraded_ids}'}")
    except FileNotFoundError:
        print("(no diagnostics file found)")

    try:
        stats = json.load(open(stats_path))
        print(f"wall_seconds={stats.get('wall_seconds')} calls={stats.get('calls')} "
              f"prompt_tokens={stats.get('prompt_tokens')} completion_tokens={stats.get('completion_tokens')}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
