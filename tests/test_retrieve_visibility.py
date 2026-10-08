from unittest.mock import patch

from memory.retrieve import retrieve


def test_retrieval_never_returns_future_or_deleted_units_for_many_times(data_store):
    """The indexed full corpus must be constrained by the query-time visibility mask.

    Forces no-key (baseline) mode: this test is about visibility filtering,
    not LLM-stage quality. Six instants spread over the whole window are
    enough (the adversarial sweep in test_sweep.py covers the edit/delete edges).
    """
    with patch("memory.llm.GROQ_API_KEY", None):
        start, end = data_store.units[0].time, data_store.units[-1].time
        span = end - start
        question = "Route Planner launch customer pricing engineering hiring update"
        for step in range(6):
            as_of = start + span * step / 5
            visible = {unit.id for unit in data_store.visible(as_of)}
            retrieved = retrieve(question, as_of.isoformat(), data_dir="./data")
            assert all(unit_id in visible for unit_id, _score in retrieved)
