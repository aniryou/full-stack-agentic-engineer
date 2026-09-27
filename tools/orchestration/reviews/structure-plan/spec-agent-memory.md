# 07 · agent-memory — the SPEC §6b block and its build inputs (draft, 2026-09-26)

For package `c5-memory` in `tools/orchestration/reviews/2026-09-26-structure-plan.md`. The block goes into `tools/orchestration/SPEC.md`
§6b after the sandboxed-execution block, unchanged (56 lines, the size of the sandboxed-execution block). The sections after it are the
other inputs of `build_topic.js`: `researchBrief` ("Research brief"), `existing` ("Existing material (paths)"), then the curriculum module
and what the integrator changes elsewhere. Environment assumed: the scratch `FACTS.md` section "Environment for the 2026-09-26
structure-plan builds" (one shared venv, no torch, no Terraform, `$SP/ref/` empty until the researcher clones).

---

### 07 · agent-memory — "Agent memory: what an agent remembers, how it is written, retrieved, consolidated and forgotten"
Primer sections: 1 What an agent remembers (working memory is the context window and its budget; episodic (what happened, timestamped), semantic (distilled
facts) and procedural (how-to notes, tool preferences) memory as one typed record — kind, scope (tenant / user / session / agent), source (turn, tool output,
human, consolidation), provenance, confidence, importance, validity, TTL, deletion key; link 07.1 notebook 03 (`history`), 07.2 notebook 03 §1, durable primer
§3.5, lra-gcp primer §3.10, vector-databases primer §17) · 2 The write path: extraction, provenance and write policy (extraction after the turn vs an explicit
`remember`; extract → compare → ADD/UPDATE/DELETE/NOOP as mem0 does (verify); a write policy in code — kinds per source, a confidence floor, screening before
persistence (07.2 notebook 11 §4, §6); merge; idempotent writes, durable primer §3.2) · 3 Retrieval: similarity, recency and importance (scope as a partition
— vector-databases primer §9, §11; the generative-agents score with tunable weights (Park et al. 2023), whose paper and reference code differ — both (verify)
— worked on three memories; top-k in a token budget; as-of filters; hybrid search stays 07.4's (`ragkit.reference`); `minifaiss`'s HNSW (M0 = 2M) pays only at
tenant scale) · 4 Measuring memory: planted facts across sessions (facts planted by a seeded generator and asked later, in LongMemEval's and LoCoMo's task
shapes — extraction, preference, multi-session, temporal, knowledge update, abstention, adversarial (verify both, and licences; nothing downloaded); recall
within the budget, accuracy, stale answers, abstention, tokens and calls per turn, Wilson intervals (07.2 notebook 08 §4); a paraphrase subset the hashing
embedder misses by design) · 5 The context budget: tokens, the prefix cache and cost per turn (recall vs tokens injected per turn, and the knee; memory
re-retrieved each turn and placed before the history changes the prefix, so every later block misses — vLLM's rules as `minengine.kv` and the serving lab's
`expected_cached_tokens` apply them (serving-engine PRIMER §5, mini-engine-core notebook 03 exercise 3.4, 07.2 notebook 04 §1–§2); memory before the history
vs a profile pinned per session vs memory at the tail; TTFT lost, by `minengine.perf.step_cost`'s model; $ per turn with and without memory at (verify)
prices, extraction amortised (scaling primer §3.4); budgets in code (durable primer §3.4)) · 6 Memory as tools, or memory before every turn (`remember` /
`recall` / `forget` with 07.1 contracts — an idempotent `remember`, a confirm-gated `forget`: the model chooses when and misses what it did not ask for;
implicit retrieval pays tokens every turn and cannot fetch mid-plan; the hybrid (a pinned profile plus `recall`); ADK's `load_memory` vs `PreloadMemoryTool`,
LangMem, Letta (verify); compared on the harness) · 7 Consolidation, forgetting and deletion (episodic → semantic per (user, window) on a schedule: newer
supersedes older, kept with `valid_to` (bi-temporal, as in Graphiti, verify), precedence human > user > tool > inferred, a weaker contradiction flagged;
reflection in brief — an importance-sum trigger, insights citing evidence (generative agents, verify), lra-gcp primer §3.8's exits; the job as a durable
scheduled run (lra-gcp primer §3.3, §3.13); recall after consolidation vs raw episodes at a fixed budget; forgetting = decay + TTL + the per-turn budget + a
cap per scope; deletion must reach the record, vector and index entry (vector-databases primer §8), derived facts (provenance), full-text indexes, prompt
caches, logs, eval sets, backups — a checklist with counts; embeddings are the data (embeddings primer §15)) · 8 Tenancy, trust and memory poisoning (tool
output or a web page (sandboxed-execution PRIMER §1) written to memory replays in later sessions — ASI06, identity primer §2; the 06/07 split: 06 owns scope
keyed by the verified principal (§8), reads under delegated identity (§3.5) and an audit event per memory read, write and forget (§9); 07 owns provenance and
trust by source (tool output quarantined, never promoted unreviewed), screening before persistence, memory fenced as data (07.2 notebook 11 §3), an injection
golden case (07.2 notebook 11 §7); leaks via shared prefix caches (`cache_salt`, vllm-internals §4.3); prompt injection's home stays 06.6) · 9 Where to run it
(T0: all of it — SQLite, a scripted model, a hashing embedder; T0 + Docker: Postgres + pgvector; T1: a real embedder or a 0.5–1.5B model via the 04 serving
lab's `deploy/any-gpu/`; T3: a Cloud Run job on Cloud Scheduler; stores and memory services, verify-marked; `COMPUTE.md`).
Core `memcore` (standard library + numpy, no torch; a scripted "LLM" — template extractor and answerer — and a hashing embedder keep every number offline;
tokens = `len(text) // 4`, as 07.2 counts): `records.py` (the record, scopes, source trust), `store.py` (per-(tenant, user) partitions; `ragkit`'s crc32
hashing embedder, re-implemented; a flat index with the delete `minifaiss` lacks), `write.py` (extraction, `WritePolicy`, ADD/UPDATE/NOOP, merge, idempotency
keys), `retrieve.py` (`score()` in both forms, filters, budget packing), `budget.py` (layouts; `expected_cached_tokens` and a block-hash cache with vLLM's
rules; `hits_per_turn()`, `prefill_seconds()`, `turn_cost()` with a dated price table; `Budget`), `consolidate.py` (the §7 rules, reflection, run id + lease +
checkpoints), `forget.py` (decay, TTL, caps; `propagate()` → a `DeletionReport`), `harness.py` (the §4 generator and answerer, metrics, `recall_vs_budget()`,
`compare_modes()`), `agent.py` (agent-core's loop with the three tools, implicit and pinned modes, `AuditEvent`-named audit, a poisoned tool result). Tests
pin each worked number; `tests/test_repo_numbers.py` reproduces `ragkit`'s vectors, `expected_cached_tokens`, `minengine.kv` and `minengine.perf`,
`capacity.ttft_s`, `scalelab.capacity.cost_per_call` (≈ $0.0070, scaling primer §3.4) and `wilson_interval` (cross-checked by path import where present).
Core notebooks: `01_records_and_the_write_path`, `02_retrieval_and_the_planted_facts_harness`, `03_the_context_budget_and_the_prefix_cache`,
`04_consolidation_forgetting_and_deletion`, `05_memory_tools_and_memory_poisoning`.
Lab `memlab`: `store/` (`sqlite.py` — the schema, float32 BLOB vectors scored in numpy, FTS5 `bm25()`, RRF (k = 60), and a deletion that removes the bytes —
`secure_delete`, FTS5 `secure-delete` or `optimize`, WAL checkpoint, `VACUUM` — checked by searching the file; `pgvector.py` — the same on Postgres +
pgvector, psycopg lazy, SQL checked offline), `embedders.py` (hashing; an OpenAI-compatible `/v1/embeddings` client), `llm.py` (an OpenAI-compatible chat
client with tool calls; the scripted model), `service.py` (aiohttp: write with `Idempotency-Key`, search, forget by deletion key; scope from a verified
stand-in token, never the body; audit JSON lines), `agent.py` (07.1-style; three modes), `consolidate.py` (a lease row, checkpoints, a crash hook),
`fakeserver.py` (chat + embeddings; cached tokens and vLLM-named metrics simulated by block hashing), `cachebench.py`, `harness.py`, `report.py`, a CLI.
Deploy: `deploy/local/` (SQLite; optional Postgres + pgvector in compose; `up.sh` prints commands without Docker), `deploy/any-gpu/` (README: the 04 serving
lab's `serve.sh`: `EXTRA_ARGS` for tool calls and `--enable-prompt-tokens-details`; an embedder on vLLM's pooling runner (verify)), `deploy/gcp/` (README, no
Terraform: consolidation as a Cloud Run job on Cloud Scheduler; the model on the 04 serving lab's Cloud Run GPU; a verify-marked table of managed stores).
Lab notebooks: `01_a_memory_store_on_sqlite` (T0; T0 + Docker for pgvector), `02_a_memory_service_and_an_agent` (T0; T1 a real small model with tool calls),
`03_memory_layouts_and_the_prefix_cache` (T0 simulated; T1 vLLM, measured), `04_consolidation_as_a_scheduled_job` (T0 with a crash and a resume; T3 printed),
`05_evaluate_forget_and_audit` (T0: the harness end to end, a deletion checked on disk; T1: a real embedder on the paraphrase subset).
Must cite/reuse: the sections named above and their code — `agentcore`, `agentlab` (`ContextBuilder`, `token_cost`, `wilson_interval`), `ragkit`, `minifaiss`,
`agentsec/audit/log.py`, `scalelab/capacity.py`, `minengine.kv`/`perf`, `capacity.py`, vllm-serving-lab notebook 04 and `deploy/`.

---

## Research brief

`$SP/ref/` is empty in this session: clone what you read (`flock $SP/ref/.lock git clone --depth 1 <url> $SP/ref/<name>`; for large repos add
`--filter=blob:none --sparse` and `git sparse-checkout set <dirs>`). arxiv.org and huggingface.co are blocked, so a paper's formula is checked
against the authors' code or README, and anything only the paper states is written (unverified) with the reason. Write the sheet to
`$SP/facts-agent-memory.md` (the integrator files it as `tools/orchestration/facts/agent-memory.md`).

1. **Generative-agents retrieval and reflection** — `joonspk-research/generative_agents`. Quote `reverie/backend_server/persona/cognitive_modules/retrieve.py`
   (`new_retrieve`, `extract_recency`, `extract_importance`, `extract_relevance`, `normalize_dict_floats`, the weight vector `gw`),
   `persona/memory_structures/scratch.py` (`recency_w`, `relevance_w`, `importance_w`, `recency_decay`, `importance_trigger_max`,
   `importance_ele_n`), `persona/memory_structures/associative_memory.py` (`ConceptNode`: `created`, `last_accessed`, `expiration`, `poignancy`)
   and `cognitive_modules/reflect.py` (trigger, focal-point questions, insights with evidence ids). Settle: is recency `recency_decay ** i` over
   nodes sorted by `last_accessed`, and does that sort give the oldest or the newest node the largest value? The defaults (decay 0.99? weights
   `[0.5, 3, 2]`? trigger 150?). The paper's form, from memory — all weights 1; recency 0.995 per sandbox hour since last retrieval; each term
   min-max normalised to [0, 1]; importance 1–10 rated by the model; cosine relevance; reflection when recent importance sums past 150; the 100
   most recent memories → 3 questions → 5 insights citing evidence — is (unverified) unless the repo states it. Licence. The core implements the
   paper's form by default and the code's as an option; both get a hand-computed test.
2. **LongMemEval** — `xiaowu0162/LongMemEval`. The `question_type` values (expected `single-session-user`, `single-session-assistant`,
   `single-session-preference`, `multi-session`, `knowledge-update`, `temporal-reasoning`) and how abstention is marked (an `_abs` id suffix?);
   the five abilities; the item schema (`question_id`, `question`, `answer`, `question_date`, `haystack_sessions`, `haystack_dates`,
   `answer_session_ids`, …); the S / M / oracle variants with sessions and tokens per item; how answers are judged (per-type judge prompts);
   code and data licences; the citation (Wu et al., ICLR 2025 (verify)). Nothing is downloaded: the harness copies task shapes, not data.
3. **LoCoMo** — `snap-research/locomo`. Conversations released (`data/locomo10.json`), sessions, turns and tokens per conversation; the QA
   categories and the numeric `category` → name mapping the JSON and its evaluation code use (reported inconsistently elsewhere: quote the code);
   the metric; the **data licence** (CC BY-NC 4.0? if non-commercial, the repo bundles none of it); the citation (Maharana et al., ACL 2024 (verify)).
4. **mem0** — `mem0ai/mem0`. `Memory.add(...)` parameters (`user_id`, `agent_id`, `run_id`, `metadata`, `infer`), `search`, `get_all`, `update`,
   `delete`, `delete_all`, `history`; the fact-extraction prompt and the update prompt with its events (`ADD`, `UPDATE`, `DELETE`, `NONE`?) in
   `mem0/configs/prompts.py`; how many existing memories each new fact is compared with; the SQLite history store (`mem0/memory/storage.py`);
   what `delete` removes (the vector? the history rows?); licence.
5. **Letta (MemGPT)** — `letta-ai/letta`. Memory blocks (`Block`: `label`, `value`, `limit` and its default); the self-editing tools as named today
   (`core_memory_append` / `core_memory_replace`, or `memory_insert` / `memory_replace` / `memory_rethink`); archival memory
   (`archival_memory_insert`, `archival_memory_search`); `conversation_search`; what happens when the window overflows (summariser, thresholds);
   sleep-time agents (the flag); licence. MemGPT's paper numbers (a warning near 70% of the window, eviction at 100%) are (unverified) unless in code.
6. **Graphiti** — `getzep/graphiti`. `EntityEdge`'s temporal fields (`valid_at`, `invalid_at`, `created_at`, `expired_at`); the function that
   invalidates a contradicted edge (under `graphiti_core/utils/maintenance/`); episodes (`add_episode`); `group_id` as the partition; search
   recipes and rerankers (RRF, MMR, node distance, episode mentions, cross-encoder) under `graphiti_core/search/`; licence; one line on whether
   `getzep/zep` still ships open-source server code.
7. **LangMem** — `langchain-ai/langmem`. `create_manage_memory_tool`, `create_search_memory_tool` (signatures, namespace templates),
   `create_memory_manager`, `create_memory_store_manager`, `ReflectionExecutor` (background, delayed); the docs' hot-path vs background split and
   semantic / episodic / procedural types; licence. Optional: `langchain-ai/langgraph` (sparse `libs/checkpoint/langgraph/store/`) for `BaseStore`
   (`put`, `get`, `search`, `delete`, namespaces, TTL and index config).
8. **ADK memory and Vertex AI Agent Engine Memory Bank** — `google/adk-python` (sparse `src/google/adk/memory`, `src/google/adk/tools`).
   `BaseMemoryService.add_session_to_memory` and `search_memory` signatures; what `InMemoryMemoryService` matches on; what
   `VertexAiMemoryBankService` calls (generate and retrieve, `scope`, similarity top-k, TTL); `load_memory` (the model calls it) vs
   `PreloadMemoryTool` (runs before every request — on which query?): the explicit/implicit pair of PRIMER §6. The ADK version at HEAD; licence.
   Managed-service facts (regions, quotas, prices) go in the Verify list only.
9. **cognee** — `topoteretes/cognee` (one table row): `add`, `cognify`, `search` (search types), `memify` if present, how data is deleted; licence.
10. **pgvector** — `pgvector/pgvector` (README, `sql/`) and `pgvector/pgvector-python` (psycopg 3 registration). Current version; `vector(n)`
    storage and index dimension limits (`halfvec`); operators `<->`, `<#>`, `<=>`, `<+>`; HNSW options and defaults (`m`, `ef_construction`,
    `hnsw.ef_search`); iterative index scans for filtered queries (`hnsw.iterative_scan` values, `hnsw.max_scan_tuples`, the version that added
    them); what `DELETE` and `VACUUM` do to HNSW entries and the README's advice on slow vacuums; Docker tags (`pgvector/pgvector:pg17`,
    `:0.8.x-pg17`, pg18); licence.
11. **vLLM v0.30.0** — `vllm-project/vllm` (tag `v0.30.0` if present, else `main`; sparse `docs/`, `vllm/entrypoints/openai/`, `vllm/config/`,
    `vllm/v1/core/`). `--enable-prompt-tokens-details` → `usage.prompt_tokens_details.cached_tokens`; the `cache_salt` request field (its name,
    the docs' guidance on length and secrecy, and that it enters the first block's hash, as vllm-internals §4.3 says); tool calling for Qwen2.5
    (`--enable-auto-tool-choice --tool-call-parser hermes`?); serving an embedding model (`--runner pooling` vs `--convert embed` vs the older
    `--task embed`; `/v1/embeddings`; whether `BAAI/bge-small-en-v1.5` and `Qwen/Qwen3-Embedding-0.6B` are detected as pooling models); the
    default block size (16).
12. **SQLite (standard library)** — measured in this environment on 2026-09-26 (Python 3.11.15, SQLite 3.45.1, compile options include
    `SECURE_DELETE` and `ENABLE_FTS5`): `PRAGMA secure_delete` reads 1; FTS5 works and `bm25()` is negative (lower is better). After `DELETE`
    from a table and from a regular FTS5 table, the deleted term still occurs twice in the database file, and still after `VACUUM`. With FTS5's
    `secure-delete` option set before the inserts (`INSERT INTO fts(fts, rank) VALUES('secure-delete', 1)`) it occurs 0 times. The FTS5
    command `INSERT INTO fts(fts) VALUES('optimize')` (regular table) or `'rebuild'` (external content), then `VACUUM`, also removes it. In WAL
    mode the term occurs 5 times right after the `DELETE` (old page images in the `-wal` file) and twice after
    `PRAGMA wal_checkpoint(TRUNCATE)` (the FTS5 index); with FTS5 `secure-delete` on, 3 times after the `DELETE` (all in the `-wal` file) and 0
    after the checkpoint. Verify from the SQLite sources (the `sqlite/sqlite` mirror: `ext/fts5/`, `src/`): the release that added FTS5
    `secure-delete` (3.42?), the upstream default of `secure_delete` (0 unless built with `SQLITE_SECURE_DELETE`?), and what Colab's Python
    links (Ubuntu 22.04's 3.37.2? — then `secure-delete` is missing and the lab falls back to `optimize`). The lab feature-detects.
13. **Cloud Run jobs on Cloud Scheduler** (for `deploy/gcp/README.md`; no Terraform) — `gcloud run jobs deploy` flags (`--tasks`,
    `--max-retries`, `--task-timeout`, `--set-env-vars`) and how Cloud Scheduler runs a job: an HTTP target
    `https://run.googleapis.com/v2/projects/<p>/locations/<r>/jobs/<job>:run` with `--oauth-service-account-email` (OAuth rather than OIDC for a
    googleapis.com target?) and the role its service account needs. Sources: `GoogleCloudPlatform/cloud-run-samples` or
    `GoogleCloudPlatform/python-docs-samples` (grep `jobs/` and `:run`); the repo's own scheduler pattern is lra-gcp's
    `google_cloud_scheduler_job.reaper` and `lra-core/gcp/deploy.sh` (OIDC to its own service). No new prices; link `COMPUTE.md`.
14. **Managed memory services** (PRIMER §9's verify table; one dated row each): Vertex AI Agent Engine Memory Bank (from item 8), Amazon Bedrock
    AgentCore Memory (`aws/bedrock-agentcore-sdk-python`: strategies — semantic, summary, user preference; short- vs long-term), Anthropic's memory
    tool (`anthropics/anthropic-sdk-python`: the tool type string, `memory_20250818`?, and its commands, `view` / `create` / `str_replace` /
    `insert` / `delete` / `rename`?, storage on the client), mem0 platform, Zep, Letta Cloud. No prices unless dated and (verify).
15. **OpenTelemetry GenAI** — `open-telemetry/semantic-conventions` (sparse `docs/gen-ai/`, `model/gen-ai/`): is there a memory or retrieval
    operation or attribute (a `gen_ai.operation.name` value, `gen_ai.data_source.id`, anything named `memory`)? The lab reuses 07.2's names
    (`gen_ai.usage.input_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.tool.name`) and keeps memory fields in its own namespace unless
    a convention exists. The llm-gateway researcher (c4) reads the same repo; share the clone.
16. **Memory poisoning** (PRIMER §8 Sources): AgentPoison (Chen et al., NeurIPS 2024; `BillChan226/AgentPoison`), MINJA (memory injection through
    queries alone, 2025), PoisonedRAG (Zou et al., 2024; already cited by the embeddings primer §15) — title, year, one-line finding each;
    (unverified) where memory is the only source.
17. **Model and tool ids for T1**: a 1–2B chat model that follows tool calls in vLLM v0.30.0 (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-1.7B`)
    and an embedder (`BAAI/bge-small-en-v1.5`, 384-d; `sentence-transformers/all-MiniLM-L6-v2`, 384-d, rag-from-scratch's default at ~90 MB
    (verify)): sizes, licences, whether vLLM serves each. Nothing is loaded at T0.

**Repo numbers the core reproduces** (record them with their paths; all computed here on 2026-09-26 unless marked): `ragkit.embed.HashingEmbedder`
(dim 1024, crc32 of each `[a-z0-9]+` token mod dim, L2-normalised); the serving lab's `expected_cached_tokens` — defined in its notebook 04
source, not in `servelab`, so restate its three rules and asserts (identical 64-token prompts → 48; a 40-token `prev` → 32; divergence inside
block 3 with B = 8 → 16); `minengine.kv.block_hashes` / `KVCacheManager.lookup` (full blocks only, hit ≤ `(len − 1) // B`);
`minengine.perf.step_cost` for one 2,000-token prefill, cold vs 1,800 cached + 200 new: 78.7 vs 15.1 ms (L4, `qwen2.5-1.5b`), 401.0 vs 65.6 ms
(L4, `llama-3.1-8b`), 11.4 vs 3.2 ms (H100-SXM, `qwen2.5-1.5b`), 50.8 vs 7.7 ms (H100-SXM, `llama-3.1-8b`) — a roofline model, labelled
simulated; `capacity.ttft_s(24, 2000, GPUS["H100"])` = 0.0970 s; `scalelab.capacity.cost_per_call("gemini-3.5-flash", 5000, 350, 2700)` =
$0.007005 (the scaling primer §3.4 call, prices of 5 Sep 2026 (verify)); `agentlab.estimation.calc.token_cost(5000, 350,
PRICES["gemini-3-flash"], cached_share=0.54)` = $0.002335 (`calc.py` and `llm/types.py` load standalone by path; `evals/gate.py` needs
pydantic, absent here, so restate `wilson_interval` and pin `wilson_interval(45, 50)` = (0.7864, 0.9565), `wilson_interval(0, 20)` = (0.0, 0.1611));
`count_tokens(text)` = `max(1, len(text) // 4)`, 0 for empty text.

**Pitfalls to carry into the sheet**: the generative-agents recency form and sort direction; LoCoMo's category numbering and licence; "deleted"
text surviving in FTS5 indexes and WAL files; `bm25()`'s sign; `cache_salt` must be secret and per tenant; `PreloadMemoryTool` runs every turn;
OAuth vs OIDC for Cloud Scheduler targets; the §6b opening list says torch is installed, but this build has none; no weights at T0.

## Existing material (paths)

Repo-relative paths; from the topic dir prefix `../../`, from a core or lab dir `../../../`. Three structure-plan packages run beside this
one, so check each path against `main` before linking and cite section titles as well as numbers: `c1-durable` may move
`long-running-durable/00_primer.md` → `long-running-durable/PRIMER.md`, `lra/lra-gcp/` → `lra-gcp/` and `lra-core/lra-core/` → `lra-core/`, and
remove `long-running-agents-core/` and `long-running-agentic/`; `c2-mistral` folds `mistral-agent-core` into `agent-core` (a new module; `Agent`,
`tool` and `ToolError` stay); `c3-nbdirs` (after c1 and c2) moves notebook directories in the 06 identity labs and `embeddings-lab`.

**07.1 — `07-application-agent-framework/agent-fundamentals/agent-core/`**
- `agentcore/agent.py`: `Agent(llm, tools=None, instruction=..., max_steps=6).run(user_message, history=None, on_confirm=None)` →
  `Result(text, messages, steps, done)`. The loop `memcore.agent` and `memlab.agent` follow (re-implemented, no import).
- `agentcore/tools.py`: `@tool`, `@tool(confirm=True)`, `ToolError(message, kind="tool_error", hint=None)`; results `{"ok": True, "data": ...}` or
  `{"ok": False, "error": <kind>, "message": ..., "hint": ...}`; `NotImplementedError` propagates (blank exercises). `agentcore/fake_llm.py`:
  `FakeLLM(responses | policy)`, `text()`, `call()`, `calls()`.
- Notebook `03_state_and_control`, "Multi-turn memory": memory as the whole transcript passed back as `history` — the baseline this topic replaces.
- `pyproject.toml`, `Makefile`, `tools/build_notebooks.py`, `tools/run_notebooks.py`: the packaging and notebook tooling to copy.

**07.2 — `07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/`**
- Notebook `03_state_sessions_checkpoints` §1 "The event log is the source of truth" (with "What the model sees is derived, not stored" and
  "Scoped state keys": `user:` / `app:` / `temp:`, "a promise about lifetime"); `agentlab/agents/state.py` (`Session`, `Event`, `set_state`,
  `clear_temp_state`, `messages()`, `InMemorySessionStore`, `JsonFileSessionStore`, `VersionConflict`).
- Notebook `04_context_engineering_and_caching` §1 "The layout" (stable → volatile), §2 "Caching, measured" (a timestamp first vs a stable
  prefix; illustrative $0.30 / $0.075 per M), §4 "Compaction: old turns become a summary", Exercise 6.5 "a memory provider" (at most three
  facts, sorted, byte-identical across turns). `agentlab/agents/context.py`: `ContextBuilder(instruction, static_context, max_recent_turns=12,
  max_tool_result_chars=1500, max_input_tokens=None, summarizer=naive_summarizer, memory_provider=None)`, layout `[system + static] [memory]
  [summary] [recent turns] [current turn]`, `memory_messages()` ("# Known about this user"), `cacheable_prefix_tokens()`, `context_report()`.
  `agentlab/llm/fake.py` `PrefixCache(min_prefix_tokens=32, ttl_s=3600.0)`; `agentlab/llm/types.py` `count_tokens`.
- Notebook `08_evals_trajectory_judge_gates` §4 (Exercise 4.1, the Wilson interval), §8 "An injection suite", §9 "Growing the golden set from
  production" (a deletion surface); `agentlab/evals/gate.py` `wilson_interval(passes, n, z=1.96)`; `agentlab/evals/golden.py`
  (`GoldenCase`, `GoldenSet`, `from_transcripts`).
- Notebook `11_security_prompt_injection` §2 "Indirect injection, end to end", §3 "Data blocks: provenance, and why delimiters must be escaped",
  §4 "Screening: injection phrases, secrets, PII", §6 "Redact before you log", §7 "An injection golden case for the eval harness";
  `agentlab/security/injection.py` (`DataBlock`, `render_context`, `escape_delimiters`, `screen`, `redact`, `sanitize_tool_output`,
  `POISONED_TICKET`).
- Notebook `12_resource_estimation` §1 (Exercise 1.1, `token_cost` with a cached share), §6 "Sizing a vector store" (`vector_store_bytes`);
  `agentlab/estimation/calc.py` (`Price`, `PRICES` — illustrative, verify — `token_cost(in_tokens, out_tokens, price, cached_share=0.0,
  batch=False)`).
- `agentlab/observability/tracing.py`: `gen_ai.usage.input_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.tool.name`.
- `docs/LAB_TO_ADK.md`: "Long-term memory | `ContextBuilder.memory_provider` | `MemoryService` (in-memory, Agent Runtime Memory Bank)".

**07.3 — `07-application-agent-framework/long-running-durable/`**
- `00_primer.md` §3.2 "Idempotency — effectively-once, not exactly-once", §3.3 "Exclusivity — leases, not locks", §3.4 "Boundedness — budgets
  are code", §3.5 "Context hygiene — every fact has a shelf life" (the lifetime table, including "long-term memory (Memory Bank / RAG) |
  indefinite, curated | explicit `remember()` calls"; "summaries paraphrase"; "stale context is worse than missing context"), P5 "Scheduled /
  heartbeat agent", P6 "Reflection / evaluator–optimizer".
- `lra/lra-gcp/docs/primer.md` §3.3 "Leases and the reaper (crash recovery)", §3.8 "Reflection (evaluator–optimizer) as durable steps" (three
  exits: threshold met, max iterations, budget exhausted), §3.9 "Budgets, deadlines, cancellation (fail closed)", §3.10 "Memory and context
  management" (run state, working memory, episodic log, long-term memory), §3.13 "Scheduled and event-triggered runs" (a deterministic
  `run_id` such as `weekly-review-2026-W37`).
- `lra/lra-gcp/infra/terraform/main.tf` (`google_cloud_scheduler_job.reaper`); `lra-core/lra-core/gcp/deploy.sh` (`gcloud scheduler jobs create
  http lra-reap --schedule="*/2 * * * *" ... --oidc-service-account-email`); `lra-core/lra-core/core.py` (`Store.effect_once`,
  `Engine(store, queue, steps, clock, lease_ttl=60, max_attempts=3)`, `Engine.reap`); `lra/lra-gcp/src/lra/patterns/reflection.py`.

**07.4 — `07-application-agent-framework/retrieval-rag/`**
- `vector-databases-primer.md` §8 "What the database layer adds" (deletes are tombstones; "a collection carrying 30% tombstones has worse recall
  and latency"; vacuum), §9 "Metadata filtering — the hardest 'easy' problem" ("if a filter is always present and highly selective … make it a
  partition or namespace, not a filter"), §11 "Scale-out: multi-tenancy, sharding, replication, storage tiers" (four tenancy patterns),
  §17 "Use-case patterns" (the "Agent memory" bullet), §18 "Operating a vector database in production" (embeddings are sensitive; recall decays
  with deletes), §19 pitfalls 5 and 12.
- `embeddings-lab/docs/primer.md` §15 "Retrieval pipelines for RAG and agents": "Embeddings elsewhere in agent systems" (the "Memory" bullet,
  Park et al. 2023) and "Security and governance" (inversion, ACLs at retrieval time, PoisonedRAG, extraction); `embeddings-lab/src/04_vector_search.py`
  (`post_filter` vs `filtered_gold`: the filtered-search trap).
- `rag-from-scratch/ragkit/embed.py`: `HashingEmbedder(dim=1024)`, `FALLBACK_LABEL = "hashing embedder (T0 fallback; not semantic)"`,
  `get_embedder()`, `RAGKIT_EMBEDDER=hashing`; `ragkit/corpus.py` `tokenize`; `ragkit/reference.py`: `BM25(corpus_tokens, k1=1.5, b=0.75)`,
  `reciprocal_rank_fusion(rankings, k=60)` (score `1 / (k + rank + 1)`, rank from 0), `mmr(query_vec, cand_vecs, cand_ids, lambda_=0.7, k=5)`,
  `recall_at_k`, `mrr`, the chunkers. Notebooks `02_chunking`, `03_hybrid_search`, `05_evaluation`.
- `vector_stores/minifaiss/hnsw.py`: `IndexHNSWFlat(d, M=16, metric_type=METRIC_L2, seed=1234)`, `M0 = 2 * M`; `minifaiss/base.py` `Index` has
  `add`, `search`, `reset`, `reconstruct` and no delete or update (the review's gap, which `memcore.store` closes for memory).

**07.5 — `07-application-agent-framework/sandboxed-execution/`**
- `PRIMER.md` §1 "Why a sandbox, and the threat model" (untrusted content; the risk → control table), §8 "Observability, audit and abuse detection".
- `sandbox-core/sandboxcore/audit.py` (a standalone mirror of `AuditEvent` and `args_digest`: copy the pattern for memory events);
  `sandbox-core/sandboxcore/agent.py` (`ScriptedLLM`, `SandboxAgent`, `injection_scenarios()`: the shape for `memcore.agent`).

**06 — `06-gateway/`**
- `identity-security/agentic-identity-gcp-lab/docs/primer.md` §2 "Threat model" (ASI06 Memory & Context Poisoning: "Provenance tags on stored
  content; per-user/per-tenant memory isolation; write-gating to memory | Screening before persistence"), §3.5 "Delegation mechanics" (RFC 8693
  token exchange), §6.1 "The untrusted-content boundary", §8 "Data boundaries, network, and tenancy" ("Sessions and Memory Bank must be keyed by
  user/tenant and never searchable across tenants"), §9 "Observability, audit, and governance" (the minimum audit event).
- `identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py`: `AuditEvent(event_type, agent, authority, user=None, tool=None,
  decision=None, reasons=[], args_hash=None, args_redacted=None, result_hash=None, approver=None, provenance=[], trace_id=None,
  invocation_id=None, session_id=None, latency_ms=None, ts, id, extra={})`, `args_digest` (sha256 of canonical JSON, first 16 hex). Memory
  events add `event_type` values `memory.write`, `memory.read`, `memory.forget`.
- `identity-security/agentic-identity-core/agentsec_core.py`: `Issuer.mint` / `exchange` / `verify` (RS256 via PyJWT; the lab's stand-in token is
  HMAC and says so), `screen`, `fence(text, source)`. Lab notebooks `05_prompt_injection_and_guardrails` (module 06.6, prompt injection's home in
  CURRICULUM §3.4) and `08_audit_and_governance`.
- `scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md` §1.4 "Cost scales with context, and context grows" (cached input
  billed at 10%), §3.4 "Cost per conversation" (the $0.0070 call), §5.5 "Context engineering for scale" (compaction: 385 k vs 184 k input tokens
  over 20 turns; minimum cacheable prefix 4,096 tokens on Gemini 3.x (verify)), §5.9 "State stores"; `scalelab/capacity.py`
  `cost_per_call(model, input_tokens, output_tokens, cached_tokens=0)`, `PRICES` (checked 5 Sep 2026, verify).

**04 — `04-inference-engine/`**
- `serving-engine/PRIMER.md` §5 "Prefix caching" ("Naming blocks", "Lookup, adopt, publish", the hit cap `(len(prompt) − 1) // B`, `cache_salt`,
  "Designing agent prompts for hits"; the 60-request simulation: 84% hit rate, TTFT p50 13 ms vs 74 ms, SIMULATED), §11 "Measuring an engine".
- `serving-engine/mini-engine-core/minengine/kv.py`: `hash_block(parent, tokens, extra=None)`, `block_hashes(tokens, block_size, extra=None,
  hash_fn=hash_block)`, `KVCacheManager(num_blocks, block_size=16, ...)`, `.lookup(tokens, extra=None)`; `minengine/perf.py`: `GPU`, `LLM`,
  `GPUS`, `LLMS`, `step_cost(gpu, llm, chunks, flop_eff=0.6, bw_eff=0.8, overhead_s=0.002)`; notebook `03_prefix_caching` exercise 3.4 ("lay
  out an agent prompt for the cache": `clock_first` 0% vs `append_only` 81–86% per turn).
- `serving-engine/vllm-serving-lab/notebooks_src/04_prefix_caching_for_agents.py`: exercise 4.1 `expected_cached_tokens(prev, new,
  block_size=16)`, 4.3 `predicted_hit_rate` (three layouts), 4.5 `ttft_estimate`, "In a design review" drill 2 (`cache_salt`);
  `vllm-serving-lab/deploy/any-gpu/serve.sh` (`MODEL`, `EXTRA_ARGS`, `DRY_RUN`, vLLM v0.30.0) and its README (`--enable-prompt-tokens-details`);
  `vllm-serving-lab/deploy/gcp/cloud-run/` (Terraform and `deploy.sh`).
- `vllm-internals/vllm-internals-primer.md` §4.1 "Object model" (extra keys: `cache_salt` on the first block only), §4.3 "Block hashes: a chain
  over the prefix".

**00 and root**
- `00-foundations/gpu-capacity-planning/PRIMER.md` "The formulas (all of `capacity.py` in one page)", item 4 "Prefill — compute-bound" ("RAG and
  agents are prefill-dominated, so prefix caching … is the biggest single win"); `capacity.py` `ttft_s(active_b, prompt_tokens, gpu,
  dtype="fp8", mfu=0.5, model=None)`, `GPUS`.
- `CURRICULUM.md` §3.4 "One home per cross-layer concept" (prompt injection → 06.6; prefix caching → 04.3; Little's law → 07.5.5), the 07 row of §2,
  the "Agent memory" row of §6; `COMPUTE.md` (tiers, the §6 per-lab tables).

## Curriculum

**Module 07.6 Agent memory** — `07-application-agent-framework/agent-memory/`, the next free 07 module (07.5 is sandboxed execution).

§4 subsection, in 07.5's shape: `#### 07.6 Agent memory — [agent-memory](07-application-agent-framework/agent-memory/README.md)`, then
"Remember what matters, for whom and for how long — and prove you forgot." Primer: `PRIMER.md`. Core: `memory-core` (package `memcore`,
standard library + numpy: typed records, the write path, generative-agents retrieval, the context budget and the prefix cache, consolidation and
deletion, a planted-facts harness). Lab: `memory-lab` (package `memlab`: SQLite with FTS5 and vectors and a pgvector twin, a memory service and an
agent, prefix hits measured on vLLM, consolidation as a scheduled job, a deletion checked on disk). No GPU needed; T1 swaps in a real model or
embedder.

One-line description ("You can …", as in the 07 table): *decide what an agent writes to memory, from which source and for whom; retrieve by
similarity, recency and importance inside a per-turn token budget; lay memory out so the prefix cache survives, and price a turn with and without
it; consolidate episodes into facts on a schedule; forget by decay, TTL and budget; carry a deletion to every copy; keep another tenant's memory,
and an attacker's, out of the prompt.* **13.5 h**, T0 (T1 optional; T3 as a printed walkthrough).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **07.6.1 Records and the write path** | name the four kinds of memory and their lifetimes; write a typed record with scope, source, provenance, confidence and a deletion key; gate writes by source and confidence, screen before persistence, merge duplicates, make a retried turn write once; store records in SQLite with FTS5 and vectors | §1 · §2 | `01_records_and_the_write_path` | `01_a_memory_store_on_sqlite` | 2.5 | T0 (+ Docker for pgvector) |
| **07.6.2 Retrieval, and measuring it** | score memories by similarity, recency and importance and say where the paper and its code differ; pack the top-k into a token budget; build a planted-facts benchmark in LongMemEval's and LoCoMo's shapes and read recall, stale answers and abstention with Wilson intervals | §3 · §4 | `02_retrieval_and_the_planted_facts_harness` | `05_evaluate_forget_and_audit` (the harness) | 3 | T0 (T1 embedder) |
| **07.6.3 The context budget and the prefix cache** | find the recall-vs-tokens knee; predict the prefix-cache hit rate of three memory layouts from vLLM's block rules, then measure it; turn lost hits into TTFT and dollars per turn | §5 | `03_the_context_budget_and_the_prefix_cache` | `03_memory_layouts_and_the_prefix_cache` | 2.5 | T0 → T1 |
| **07.6.4 Consolidation, forgetting and deletion** | consolidate episodes into facts with supersede and precedence rules, as a durable scheduled job that survives a crash; compare recall after consolidation with raw episodes at a fixed budget; forget by decay, TTL and caps; follow a deletion through records, vectors, indexes, derived facts, caches and the database file | §7 | `04_consolidation_forgetting_and_deletion` | `04_consolidation_as_a_scheduled_job`, `05_evaluate_forget_and_audit` (the deletion) | 3 | T0 (T3 printed) |
| **07.6.5 Memory tools, tenancy and poisoning** | compare `remember` / `recall` / `forget` tools with retrieval before every turn on the harness; key scope to the verified principal; stop a poisoned tool result from becoming a standing instruction; say which controls belong to the gateway (06) and which to the agent (07) | §6 · §8 · §9 | `05_memory_tools_and_memory_poisoning` | `02_a_memory_service_and_an_agent` | 2.5 | T0 → T1 |

Elsewhere in `CURRICULUM.md` (the integrator's edits):
- §3.2: the last step, after 07.4 (step 29 today; 30 once the llm-gateway step, which `spec-llm-gateway.md` inserts after step 24, lands):
  `| 30 | 07 | 07.6 | agent-memory PRIMER; memory-core 01–05; memory-lab 01–05 | 13.5 | T0 | T0; lab 02–03 at T1; lab 04's GCP part at T3 |`.
  It comes after retrieval (07.4) and durable execution (07.3), whose embedder, fusion, metrics and scheduled jobs it reuses, and after 04.3,
  whose block rules price its context budget; say so in §3.1's paragraph on the newer topics. Totals: about 282.5 h with this module alone
  (269 + 13.5), about 299 h with the llm-gateway's 16.5; the newer topics then account for 98.5 h (68.5 + 16.5 + 13.5).
- §3.3 "Agent builder": add 07.6.3 (core 03, ~1.5 h) after 04.3.
- §3.4: "Prompt injection" → also applied in 07.6.5 (memory poisoning: provenance and trust on the write path, fenced data on the read path);
  "Prefix caching" → also applied in 07.6.3 (memory layout).
- §2, 07 row: add `agent-memory/` to "What is here"; drop "agent memory, episodic and semantic (a paragraph each, no lab)" and "per-turn cost
  budgets tied to 04.3's prefix-cache numbers"; "reflection loops" becomes "reflection loops beyond 07.6's brief"; "where prompt-injection
  defence splits between 06 and 07" stays for tools and retrieval (07.6.5 settles it for memory content); keep "`minifaiss` has no delete or
  update" for ANN, noting that `memory-core`'s flat index and `memory-lab`'s SQLite and pgvector stores delete.
- §5.1: add the agent-memory primer to the "newer primers" row. §5.2: two cross-layer drills, numbered after the llm-gateway's 16–18 —
  19 | *After long-term memory was added, TTFT p50 tripled and input cost rose 40% for a small gain in recall.* | Memory is re-retrieved every turn
  and injected above the history, so the prefix changes each turn and every history block after it misses the prefix cache: the conversation
  re-prefills on every turn. Pin a per-session profile or append per-turn memory at the tail; cap injected tokens at the recall-vs-budget knee;
  watch `prompt_tokens_details.cached_tokens` or `vllm:prefix_cache_hits` / `queries` per turn. | 07.6.3, 04.3, 07.2
  20 | *A user asked the assistant to forget their address; a week later it quoted it back.* | Deleting the record did not delete the fact: it
  lived on in a consolidated summary derived from it, in the full-text index (FTS5 keeps deleted terms until a merge or secure-delete), in a WAL
  file, a cached prompt prefix, a log or an eval set. Give every record a deletion key and provenance, propagate to what was derived, vacuum the
  indexes, evict the caches, and test deletion by searching the bytes on disk. | 07.6.4, 04.3, 06.6
- §6: move "Agent memory" from the table to "Built from this list so far", as `agent-memory` (07.6).
- §1.1 / the one-minute version: the count of main topics (eleven with 06.7 and 07.6) and the total hours (coordinate with the llm-gateway package).

## Notes for the integrator

- SPEC §2 table: `| 07 | 07-application-agent-framework/agent-memory | memory-core (memcore) | memory-lab (memlab) |`.
- SPEC §6b: "Four more topics" becomes six with llm-gateway and agent-memory; its opening list's torch and Terraform bullets do not hold for these
  two (scratch FACTS "structure-plan builds"), which is why the block says "no torch" and "no Terraform" itself.
- Facts: `$SP/facts-agent-memory.md` → `tools/orchestration/facts/agent-memory.md`, plus a line in `FACTS.md`'s per-topic list.
- `tools/ci/labs.json`: `memory-core` and `memory-lab` in the sandbox labs' shape (`python -m pip install -e ".[dev]"`, `python -m pytest -q`,
  solutions and `--expect-fail`); check with `tools/ci/run_local.sh --check`.
- `07-application-agent-framework/README.md`: a row for `agent-memory/`, its Run-it lines, "modules 07.1–07.6", and the caveat that still says
  "not covered yet: agent memory beyond a paragraph".
- `COMPUTE.md` §6: a `### 07 · memory-lab` table (five notebooks plus deploy rows: T0, T0 + Docker, T1, T3 printed) and §9 dated items
  (pgvector image, vLLM pooling and tool-call flags, Memory Bank).
- `CLAUDE.md` (c6-final): the percent-source list gains `memory-core` and `memory-lab`; a decisions-log entry; the notebook baseline + 10.
- Regenerate, never hand-edit: `python3 tools/gen_colab_index.py`, `python3 tools/site/build_site_content.py`.
