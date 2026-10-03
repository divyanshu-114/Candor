"""Interactive TextOS: a terminal loop. Type a command, see the planned
actions as a readable card, then "run it? (y/n)".

Default is dry-run (prints what would happen). With --execute, only safe
local executors run for real: app.open via `open -a` on macOS. Slack,
Gmail and Calendar always print "(not executed: needs OAuth)" since this
is a take-home dry-run system, not a connected one.

Optional voice: --voice records 5 seconds and transcribes with
faster-whisper + sounddevice if both are installed; otherwise prints an
install hint and falls back to typed input.

Usage:
  python -m actions.repl                 # dry run
  python -m actions.repl --execute       # app.open actually opens apps
  python -m actions.repl --voice         # speak commands instead of typing
  python -m actions.repl --as-of ISO     # pin "now" for reproducible demos
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys

from actions.planner import plan


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _print_card(actions: list[dict]) -> None:
    print("\n--- planned actions " + "-" * 40)
    for i, a in enumerate(actions, 1):
        print(f"{i}. {a.get('type')}")
        for k, v in (a.get("args") or {}).items():
            v_str = ", ".join(v) if isinstance(v, list) else str(v)
            print(f"     {k}: {v_str}")
    print("-" * 60)


def _execute(action: dict) -> str:
    t = action.get("type")
    args = action.get("args") or {}
    if t == "app.open":
        app = args.get("app", "")
        if platform.system() == "Darwin":
            try:
                subprocess.run(["open", "-a", app], check=True)
                return f"opened {app!r}"
            except Exception as e:
                return f"failed to open {app!r}: {e}"
        return f"(not executed: app.open only supported on macOS in this skill)"
    if t in ("slack.send_message", "gmail.send", "calendar.create_event", "calendar.update_event", "reminder.create"):
        return "(not executed: needs OAuth)"
    if t in ("memory.ask", "clarify", "confirm"):
        return "(no execution needed for this action type)"
    return "(unknown action type, not executed)"


def _record_and_transcribe(seconds: int = 5) -> str | None:
    try:
        import sounddevice as sd  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        print("Voice input needs `faster-whisper` and `sounddevice`:\n"
              "  pip install faster-whisper sounddevice\n"
              "Falling back to typed input.")
        return None

    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    print(f"Recording {seconds}s... speak now.")
    sample_rate = 16000
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    model = WhisperModel("base")
    segments, _info = model.transcribe(np.squeeze(recording), language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"Heard: {text!r}")
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive TextOS")
    parser.add_argument("--execute", action="store_true", help="actually run safe local actions (app.open)")
    parser.add_argument("--voice", action="store_true", help="speak commands instead of typing")
    parser.add_argument("--as-of", default=None, help="pin the current time (ISO 8601 with offset)")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    print("TextOS (dry run by default; Ctrl-D or 'quit' to exit)")
    while True:
        if args.voice:
            command = _record_and_transcribe()
            if command is None:
                try:
                    command = input("> ").strip()
                except EOFError:
                    break
        else:
            try:
                command = input("> ").strip()
            except EOFError:
                break
        if not command or command.lower() in ("quit", "exit"):
            break

        as_of = args.as_of or _now_iso()
        try:
            actions = plan(command, as_of, data_dir=args.data_dir)
        except Exception as e:
            print(f"(internal error planning this command: {e})")
            continue

        _print_card(actions)
        try:
            reply = input("run it? (y/n) ").strip().lower()
        except EOFError:
            break
        if reply != "y":
            print("(not run)")
            continue
        for a in actions:
            if args.execute:
                print(f"  -> {_execute(a)}")
            else:
                print(f"  -> (dry run, would: {a.get('type')} {a.get('args')})")


if __name__ == "__main__":
    main()
