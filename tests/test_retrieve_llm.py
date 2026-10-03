"""Test LLM retrieval stages including ID validation and fallback."""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from memory import config
from memory.retrieve import retrieve
from memory.store import MemoryStore


@patch("memory.retrieve.chat_json")
@patch("memory.query.chat_json")
def test_llm_rerank_validation(mock_query_chat, mock_retrieve_chat):
    """Test that LLM reranking drops forbidden IDs and fills missing ones."""
    # Temporarily enable all features
    old_ana = config.USE_ANALYSIS
    old_mq = config.USE_MULTIQUERY
    old_nb = config.USE_NEIGHBORS
    old_h2 = config.USE_HOP2
    old_rr = config.USE_RERANK
    config.USE_ANALYSIS = True
    config.USE_MULTIQUERY = True
    config.USE_NEIGHBORS = False
    config.USE_HOP2 = False
    config.USE_RERANK = True

    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "connectors/slack").mkdir(parents=True)
            
            # 1 valid, 1 future
            s1 = {"id": "SL-1", "user": "U1", "ts": "2026-09-08T10:00:00-07:00", "text": "Valid message"}
            s2_fut = {"id": "SL-2", "user": "U1", "ts": "2026-09-10T10:00:00-07:00", "text": "Future message"}
            
            with open(d / "connectors/slack/messages.jsonl", "w") as f:
                f.write(json.dumps(s1) + "\n")
                f.write(json.dumps(s2_fut) + "\n")

            # Mock analysis
            mock_query_chat.return_value = {
                "intent": "fact_lookup",
                "wants_latest": False,
                "wants_history": False,
                "sub_queries": ["Valid"],
                "entities": [],
                "dates": [],
                "ambiguous_person": None,
                "needs_followup": False
            }

            # Mock rerank with a valid ID, an invented ID, and a future ID
            mock_retrieve_chat.return_value = {
                "ranked": ["INVENTED-1", "SL-2", "SL-1"],
                "reason": "Because I said so"
            }

            ranked = retrieve("test query", "2026-09-09T00:00:00-07:00", data_dir=str(d), cache_dir=str(d / ".cache"))
            
            ids = [x[0] for x in ranked]
            
            # SL-1 should be the only one present. INVENTED-1 and SL-2 are forbidden.
            assert "INVENTED-1" not in ids
            assert "SL-2" not in ids
            assert "SL-1" in ids
    finally:
        config.USE_ANALYSIS = old_ana
        config.USE_MULTIQUERY = old_mq
        config.USE_NEIGHBORS = old_nb
        config.USE_HOP2 = old_h2
        config.USE_RERANK = old_rr
