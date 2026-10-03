"""actions — TextOS: turns a natural-language command into a list of dry-run
actions (slack.send_message, gmail.send, calendar.create_event,
calendar.update_event, reminder.create, memory.ask, app.open, clarify,
confirm).

Sub-modules:
  context   builds a compact, as-of-aware "world" snapshot (people,
            channels, calendar events, date table) -- no side effects
  planner   plan(command, as_of) -> list of actions
  cli       command-line entry point (python -m actions.cli --commands ...)
  repl      interactive terminal loop (python -m actions.repl)

Only imports from the memory package: memory.store, memory.answer,
memory.llm. Never modifies memory/*.
"""
