"""memlab — agent memory you can run and inspect (07.6, the lab).

A SQLite store with FTS5 and vectors (and a pgvector twin), a memory service and an agent, prefix-cache
hits per memory layout, consolidation as a scheduled job, a planted-facts eval and a deletion checked on
disk. Concepts: ``../PRIMER.md``. Everything runs offline at T0; set ``MEMLAB_LLM_URL`` /
``MEMLAB_EMBED_URL`` to measure a real model or embedder (T1).
"""
__version__ = "0.1.0"
