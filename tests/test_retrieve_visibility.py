from datetime import timedelta
from unittest.mock import patch

from memory.retrieve import retrieve
from memory.store import MemoryStore


def test_retrieval_never_returns_future_or_deleted_units_for_many_times():
    """The indexed full corpus must be constrained by the query-time visibility mask.

    Forces no-key (baseline) mode: this test is about visibility filtering,
    not LLM-stage quality, and must never make a real network call just
    because a real key happens to be present in the host's .env -- each of
    the 20 as_of values would otherwise cost real tokens/quota for no reason.
    """
    with patch("memory.llm.GROQ_API_KEY", None):
        store = MemoryStore("./data")
        start, end = store.units[0].time, store.units[-1].time
        span = end - start
        question = "Route Planner launch customer pricing engineering hiring update"
        for step in range(20):
            as_of = start + span * step / 19
            visible = {unit.id for unit in store.visible(as_of)}
            retrieved = retrieve(question, as_of.isoformat(), data_dir="./data")
            assert all(unit_id in visible for unit_id, _score in retrieved)
