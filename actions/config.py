"""Self-contained config for actions/ -- deliberately does not import
memory.config, to keep actions/ decoupled from memory/* internals beyond
the three modules the brief allows (store, answer, llm).
"""
import os

DATA_DIR = os.environ.get("DATA_DIR", "./data")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
TIMEZONE = "America/Los_Angeles"

DATE_TABLE_DAYS = 14

# Both planner steps use the FAST role by default so the strong-model quota
# is saved for memory retrieval/answers. Set ACTIONS_STEP2_ROLE=strong to
# put step 2 (final action JSON) back on the strong model.
ACTIONS_STEP2_ROLE = "strong" if os.environ.get("ACTIONS_STEP2_ROLE", "fast").strip().lower() == "strong" else "fast"
MAX_TOKENS_PLAN = 500       # the single planning call
MAX_TOKENS_FOLLOWUP = 300   # the optional follow-up (lookup evidence or repair)
LOOKUP_SNIPPETS = 5         # top-N retrieved records per lookup (plain hybrid search)
LOOKUP_SNIPPET_WORDS = 60
REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_MEDIUM = "medium"

WORKERS = int(os.environ.get("WORKERS", "4"))

VALID_ACTION_TYPES = {
    "slack.send_message", "gmail.send", "calendar.create_event", "calendar.update_event",
    "reminder.create", "memory.ask", "app.open", "clarify", "confirm",
}
REQUIRED_ARGS = {
    "slack.send_message": {"to", "text"},
    "gmail.send": {"to", "subject", "body"},
    "calendar.create_event": {"title", "start", "end", "attendees"},
    "calendar.update_event": {"event_id"},
    "reminder.create": {"text", "due"},
    "memory.ask": {"question"},
    "app.open": {"app"},
    "clarify": {"question"},
    "confirm": {"summary"},
}
DESTRUCTIVE_WORDS = ("delete", "cancel all", "remove", "wipe")

# Deterministic resolver (actions/resolve.py): act on the best-supported interpretation without a model call.
USE_RESOLVER = os.environ.get("USE_RESOLVER", "true").lower() not in {"0", "false", "no"}
