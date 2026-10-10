"""Cross-check our visibility rule against the official harness (eval_harness/records.py)."""
from datetime import datetime

from eval_harness.records import visible as harness_visible

TIMES = [
    "2026-09-08T12:00:00-07:00",
    "2026-09-10T09:00:00-07:00",
    "2026-09-14T00:00:00-07:00",
    "2026-09-16T18:00:00-07:00",
    "2026-09-18T23:59:59-07:00",
]


def test_crosscheck_eval_harness(data_store):
    """Ours may differ from the harness only by dropping message_deleted event
    records (we never return those); everything else must be identical."""
    for t in TIMES:
        as_of = datetime.fromisoformat(t)
        mine = {u.id for u in data_store.visible(as_of)}
        theirs = {u.id for u in harness_visible("./data", as_of)}
        assert mine <= theirs, f"we return ids the harness hides at {t}: {sorted(mine - theirs)[:5]}"
        extra = theirs - mine
        assert all(i.startswith("SL-EV-") for i in extra), f"unexpected harness-only ids at {t}: {sorted(extra)[:5]}"
