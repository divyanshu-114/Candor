"""
memory package — Candor memory system.

Sub-modules:
  cli      command-line entry point (python -m memory.cli answer ...)
  ingest   load and filter data/, build the searchable corpus
  search   retrieval: BM25 + dense re-rank
  llm      single LLM-call function with caching and retries
  answer   generate answers from retrieved records
"""
