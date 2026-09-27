# memory-lab — put an agent's memory on real storage, measure what it costs, and prove it forgot

After this lab you can run an agent's long-term memory as a service — SQLite with FTS5 and vectors (or
Postgres + pgvector), scope taken from a verified token, idempotent writes, a pinned profile or per-turn
retrieval — measure what each memory layout does to the prefix cache and the bill, consolidate episodes into
facts as a scheduled job that survives a crash, grade the whole thing on a planted-facts benchmark, and carry
a forget to every copy and check it on disk.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (30 min): what an agent remembers, and the write path.
2. `python3 -m pip install -e ".[dev]" && python3 -m memlab demo` — under a minute: writes two memories to a
   SQLite file, recalls one, forgets the address and counts its bytes on disk (0 after the purge).
3. Open [`notebooks/01_a_memory_store_on_sqlite.ipynb`](notebooks/01_a_memory_store_on_sqlite.ipynb) (T0): the
   store, hybrid search, `bm25()` by hand, and why `DELETE` is not a forget.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T0 + Docker = the local compose stack; T1 = one small GPU (Colab/Kaggle
T4 or a rented card); T3 = the Google Cloud deployment, optional and printed here.* Each notebook opens with
*the one-minute version*, works examples against the library, then 4–5 exercises (implement the key function,
predict a number, pick a setting) each followed by a check that prints ✅, and closes with *in a design review*.
Answers are in [`solutions/`](solutions/). About 6.5 hours in all.

| # | Notebook | Tier | You will be able to… | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`a_memory_store_on_sqlite`](notebooks/01_a_memory_store_on_sqlite.ipynb) | T0 (+Docker for pgvector) | store typed records with float32 vectors and FTS5 in one file, partitioned by tenant and user; fuse cosine and `bm25()` with RRF (k = 60); reproduce `bm25()` exactly, sign and IDF floor included; size a memory; count the copies a `DELETE` leaves and purge them | §1, §3, §7 | ~1.5 h |
| 02 | [`a_memory_service_and_an_agent`](notebooks/02_a_memory_service_and_an_agent.ipynb) | T0 (T1: a real model with tool calls) | serve memory over HTTP with scope from a verified token and 400 for a body that names one; make a retried turn write once; run the 07.1 loop with memory as tools, before every turn, or as a pinned profile; quarantine a poisoned page's "remember"; decline a page's "forget" | §2, §6, §8 | ~1.5 h |
| 03 | [`memory_layouts_and_the_prefix_cache`](notebooks/03_memory_layouts_and_the_prefix_cache.ipynb) | T0 simulated (T1: vLLM, cached tokens measured) | predict and measure `cached_tokens` per turn for memory before the history, a pinned profile and memory at the tail; find where the prefix breaks; turn misses into TTFT (roofline) and into dollars on a hosted API with a caching minimum; give each tenant its own `cache_salt` | §5, §8 | ~1.5 h |
| 04 | [`consolidation_as_a_scheduled_job`](notebooks/04_consolidation_as_a_scheduled_job.ipynb) | T0 (T3 printed) | consolidate episodes into facts with supersede and precedence rules; run it as a durable job — a weekly run id and a weekly schedule to match, a one-statement lease, checkpoints — kill it mid-write and resume to the same state; forget episodes by TTL; print the Cloud Run job and Cloud Scheduler commands | §7 | ~1 h |
| 05 | [`evaluate_forget_and_audit`](notebooks/05_evaluate_forget_and_audit.ipynb) | T0 (T1: a real embedder) | grade five memory designs on 84 planted-fact questions in LongMemEval's and LoCoMo's shapes with Wilson intervals; pick the budget at the recall knee; forget an address everywhere — records, derived facts, idempotency rows, the prefix cache, the eval set — and prove it on disk; read the audit trail | §4, §5, §7, §8 | ~1 h |

Everything runs on a laptop first. The model is **scripted** (a template extractor and answerer, so every
accuracy number is reproducible and attributable to the memory system), the embedder is the **hashing
embedder** of 07.4's `ragkit` (lexical, not semantic — the paraphrase subset measures what that costs), and
the inference server in notebook 03 is a **fake** whose prefix-cache hits follow vLLM's block rules and whose
TTFTs come from a roofline model — its numbers are labelled *simulated*. A real model, embedder or vLLM behind
`MEMLAB_LLM_URL` / `MEMLAB_EMBED_URL` turns the answers, the embeddings and the cached-token counts into
measurements; prefill times stay a roofline estimate and dollars a price table, and every table says which column
is which. Missing Docker, Postgres or GCP
prints the exact commands and, where a container would answer, shows sample output in the documented format
(illustrative).

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the SQLite store, the service, the agent, the fake server, the job, the harness | every notebook, all tests |
| **T0 + Docker** | a laptop with Docker | the service, the fake model server and Postgres + pgvector in compose | [`deploy/local/`](deploy/local/), notebook 01's last section |
| **T1** | one GPU: Colab/Kaggle T4 (free) or a 24 GB card | Qwen2.5-1.5B-Instruct with tool calls and `cached_tokens`; `bge-small-en-v1.5` on vLLM's pooling runner | [`deploy/any-gpu/`](deploy/any-gpu/), notebooks 02, 03, 05 |
| **T3** | GCP | consolidation as a Cloud Run job on Cloud Scheduler; the model on the 04 serving lab's Cloud Run GPU | [`deploy/gcp/`](deploy/gcp/) (printed; no Terraform) |

## Run it

```bash
cd memory-lab
python3 -m pip install -e ".[dev]"                 # numpy + aiohttp; dev: pytest, jupyter, pyyaml, pglast
python3 -m pytest -q                               # 175 tests, ~20 s, offline, no GPU (1 skip by design: the Colab injector)
python3 -m memlab env                              # which tier this machine can measure
python3 -m memlab demo                             # write, recall, forget, count the bytes on disk
python3 -m memlab cachebench                       # the three memory layouts against the fake server (simulated)
python3 -m memlab harness                          # five memory designs on the planted-facts benchmark
python3 -m memlab gcp-commands --project my-project   # the T3 job and schedule, printed
python3 -m jupyterlab notebooks                    # the exercises; answers in solutions/
```

T1: start vLLM with the flags in [`deploy/any-gpu/`](deploy/any-gpu/), then
`export MEMLAB_LLM_URL=http://127.0.0.1:8000 MEMLAB_EMBED_URL=http://127.0.0.1:8001` and re-run notebooks 02, 03
and 05 (or `python3 -m memlab cachebench --url $MEMLAB_LLM_URL`). T0 + Docker: `deploy/local/up.sh --pgvector`,
`pip install "psycopg[binary]"`, then set `MEMLAB_PG_DSN` as it prints.

## How it fits

The concepts are in [`../PRIMER.md`](../PRIMER.md); the minimal from-scratch version is next door in
[`../memory-core/`](../memory-core/) — this lab never imports it. It reuses, and reproduces in
`tests/test_repo_numbers.py`, the 07.1 agent loop and tool contract
([`agent-core`](../../agent-fundamentals/agent-core/)), 07.4's hashing embedder and RRF
([`rag-from-scratch`](../../retrieval-rag/rag-from-scratch/)), the serving lab's prefix-cache rules and
`minengine.perf.step_cost` ([`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/)),
`scalelab.capacity.cost_per_call`
([`06-gateway/scaling-admission-cost`](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/))
and 07.2's Wilson interval. Scope, delegation and audit follow the
[identity primer](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §2, §3.5, §8, §9;
the durable job follows the durable-execution primer §3.2–§3.4 and the lra-gcp primer §3.3 and §3.13
(`07-application-agent-framework/long-running-durable/`). Prices and where to get GPUs:
[`COMPUTE.md`](../../../COMPUTE.md).

## The library (`memlab/`, ~4,500 lines, numpy + aiohttp)

| Module | Lines | The idea |
|---|---:|---|
| `store/sqlite.py` | ~580 | one file: typed rows, float32 BLOB vectors scored in numpy, FTS5 (`porter unicode61`) ranked by `bm25()`, RRF k = 60, the `(tenant, user)` partition in every query, as-of reads, a forget that follows provenance and purges FTS5, the WAL and freed pages; the job's lease and checkpoint tables |
| `store/pgvector.py` | ~370 | the same store on Postgres + pgvector: DDL, hybrid search as RRF in SQL, a recursive forget, the job tables; psycopg lazy; every statement parsed offline by `pglast` in the tests |
| `records.py`, `deletion.py` | ~180 | the typed record (kind, scope, source, trust, provenance, validity, TTL, deletion key); the deletion report and the byte-level residue check |
| `memory.py` | ~215 | the write path: `WritePolicy` (kinds per source, confidence floor, screening, quarantine), resolution ADD / NOOP / UPDATE / ADD_HISTORY / FLAG, idempotency keys unique per partition and a refused different write under a known key, the pinned profile, audit |
| `service.py` | ~350 | the memory service (aiohttp): scope from a verified stand-in token, `Idempotency-Key`, forget by subject, review and promote, audit JSON lines; its client and `RemoteMemory` |
| `agent.py`, `llm.py`, `extract.py` | ~815 | the 07.1 loop with memory as tools, implicit or pinned, three prompt layouts, provenance by taint and a confirm-gated forget; an OpenAI-compatible chat client and the scripted model; the extraction and answering grammar |
| `fakeserver.py`, `cachebench.py` | ~710 | a fake OpenAI-compatible server (chat with tool calls, embeddings, vLLM-named metrics, block-hash prefix cache with `cache_salt`, vLLM's whole-cache reset); the layout bench with the roofline TTFT and the price table (with the provider's caching minimum), each column labelled measured or modelled |
| `consolidate.py` | ~300 | consolidation as a durable run: a weekly run id and window, lease, checkpoints, a crash hook, resolution, reflection, TTL marking; sharding and the printed GCP commands (image, identities, Cloud SQL, schedule, cleanup) |
| `harness.py`, `report.py` | ~420 | the planted-facts generator, grader, Wilson intervals, mode comparison, recall versus budget and the knee; JSON and Markdown reports |
| `audit.py`, `embedders.py`, `env.py`, `__main__.py` | ~540 | the identity lab's `AuditEvent` with `memory.*` types and GenAI memory names; the hashing and HTTP embedders; tier detection; the CLI |

## Deploy

[`deploy/`](deploy/) — [`local/`](deploy/local/) (compose: the service, the fake model server, optional
`pgvector/pgvector:0.8.6-pg17`; `up.sh` / `down.sh` with `DRY_RUN=1`), [`any-gpu/`](deploy/any-gpu/) (the 04
serving lab's `serve.sh` with tool calling, `--enable-prompt-tokens-details` and a pooling-runner embedder),
[`gcp/`](deploy/gcp/) (the consolidation job on Cloud Run jobs and Cloud Scheduler, the model on the serving
lab's Cloud Run GPU, a dated table of managed memory stores). Each has a README with cost and cleanup.

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Computed, simulated, measured, illustrative.** Harness numbers are *computed* by a scripted model and a
  hashing embedder: exact and reproducible, not a model's accuracy. The fake server's cached tokens follow vLLM's
  rules and its TTFTs a roofline model: *simulated*. Only a real server or embedder gives *measured* numbers. The
  pgvector sample in notebook 01 is *sample output in the documented format (illustrative)*.
- **This lab's policy and harness differ from memory-core's, on purpose.** `memory.WritePolicy` is stricter than
  PRIMER §2's (a 0.7 floor instead of 0.6; inferred facts may write semantic memory only; a fifth source,
  `consolidation`, for the job's own writes, carrying the trust of its strongest evidence). The harness has other
  users, questions and a 128-token budget, so its mode ranking is not PRIMER §6's; and its knee is relative (the
  smallest budget reaching 95% of the best recall, `harness.knee(frac=0.95)`) where memory-core's is absolute
  (within 2 points of the best). Neither harness is evidence about a real model: both script the answerer.
- **The token is a stand-in.** HMAC-signed, with a real verifier's claims and failure modes, not its
  cryptography; production verifies RS256/ES256 tokens against your IdP (identity primer §3.5).
- **A forget reaches what this process can reach.** The report counts the store, derived facts, checkpoints,
  idempotency rows, a prefix cache, an eval set and the audit log; backups, WAL archives and other services'
  copies expire on their own schedule, and the deletion policy must say how long. vLLM cannot evict one tenant's
  cached blocks: rotate the tenant's `cache_salt`, and keep the dev-mode full reset for the operator.
- **The memory service is SQLite-only.** The consolidation job runs on Postgres + pgvector; `memlab serve` does not
  (its idempotency table is SQLite SQL), so a Postgres deployment of the service is a port you would write.
- **SQLite features vary.** FTS5 `secure-delete` needs SQLite 3.42 or newer (verify; Colab's may be older) — the
  store feature-detects it and falls back to `optimize`. The deploy paths are checked with `bash -n`,
  `DRY_RUN=1`, YAML checks and Postgres's parser, not run against real infrastructure here.

## Verify list (facts dated 2026-09-26 that move)

- vLLM **v0.30.0**: `--enable-auto-tool-choice --tool-call-parser hermes` for Qwen2.5; `--enable-prompt-tokens-details`
  for `usage.prompt_tokens_details.cached_tokens`; `--runner pooling` for embeddings (no `--task`); `cache_salt` at
  most 128 characters, no `@ / \`, first block only; default block size 16; `/reset_prefix_cache` (dev mode,
  `VLLM_SERVER_DEV_MODE=1`: the whole cache, every tenant, answers `{"success": bool}`; no eviction by salt).
- pgvector **0.8.6** (`pgvector/pgvector:0.8.6-pg17`): `vector` indexes up to 2,000 dimensions; HNSW `m = 16`,
  `ef_construction = 64`, `hnsw.ef_search = 40`; `hnsw.iterative_scan` since 0.8.0; filters apply after the index scan.
- SQLite: FTS5 `secure-delete` since 3.42.0; `bm25()` k1 = 1.2, b = 0.75, IDF floored at 1e-6; `PRAGMA
  secure_delete` default off unless compiled in.
- Cloud Run jobs and Cloud Scheduler: the v2 `jobs/<job>:run` URI with OAuth; `--task-timeout`, `--set-secrets`,
  `--set-cloudsql-instances`, `--service-account` and `--command/--args` on `gcloud run jobs create`; the
  `host=/cloudsql/<connection name>` DSN; whether `roles/run.invoker` suffices to run a job;
  `CLOUD_RUN_TASK_INDEX` / `CLOUD_RUN_TASK_COUNT`.
- Model and embedder ids, sizes and licences (`Qwen/Qwen2.5-1.5B-Instruct`, `BAAI/bge-small-en-v1.5`); prices in
  `cachebench.PRICES` (scalelab's, 5 Sep 2026) and the 4,096-token caching minimum on Gemini 3.x (scaling primer
  §5.5; request or prefix?); the OpenTelemetry GenAI memory attributes (status *development*).

MIT licensed.
