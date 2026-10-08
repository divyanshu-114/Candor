"""Prompt text shared by retrieval paths (kept in one place so v1 and v2 rerank prompts cannot drift)."""

RERANK_SYSTEM = (
    "Rank candidate records by usefulness for answering the question.\n"
    "RULES:\n"
    "- first 10 must include the best 2 records for EACH sub-query\n"
    "- if wants_latest: newest decisive record first, but keep earlier records showing WHAT CHANGED in ranks 4-10\n"
    "- for who-said questions keep second-hand reports and the person's own statement separate\n"
    "- never invent ids; only use ids from the list; drop chit-chat\n\n"
    "Return JSON: {\"ranked\": [\"id1\", \"id2\", ...], \"reason\": \"short explanation\"}"
)
