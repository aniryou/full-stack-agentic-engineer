# memory-lab — put an agent's memory on real storage, measure what it costs, and prove it forgot

After this lab, you can run the long-term memory of an agent as a service. The service uses SQLite with FTS5 and
vectors (or Postgres + pgvector). It takes the scope from a verified token, its writes are idempotent, and it gives
a pinned profile or a retrieval for each turn. You can measure what each memory layout does to the prefix cache and
to the bill. You can consolidate episodes into facts as a scheduled job that survives a crash. You can grade the
whole system on a planted-facts benchmark, send a forget to every copy and examine the result on disk.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (30 min). They are about what an agent remembers and about the write path.
2. Run `python3 -m pip install -e ".[dev]" && python3 -m memlab demo`. It takes less than a minute. It writes two
   memories to a SQLite file and recalls one. Then it forgets the address and counts the bytes of the address on
   disk (0 after the purge).
3. Open [`notebooks/01_a_memory_store_on_sqlite.ipynb`](notebooks/01_a_memory_store_on_sqlite.ipynb) (T0). It
   shows the store, hybrid search and `bm25()` by hand. It also shows why `DELETE` is not a forget.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, and it is free. T0 + Docker is the local compose stack. T1 is one small GPU
(a Colab/Kaggle T4 or a rented card). T3 is the Google Cloud deployment. It is optional, and the lab prints its
commands.*

Each notebook starts with *the one-minute version*. Then it shows worked examples that use the library. Then it
gives 4–5 exercises (implement the key function, predict a number, select a setting), and a check that prints ✅
comes after each exercise. The notebook ends with *in a design review*. The answers are in
[`solutions/`](solutions/). The notebooks take about 6.5 hours in all.

| # | Notebook | Tier | You will be able to… | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`a_memory_store_on_sqlite`](notebooks/01_a_memory_store_on_sqlite.ipynb) | T0 (+Docker for pgvector) | Store typed records with float32 vectors and FTS5 in one file, with a partition by tenant and user. Fuse cosine and `bm25()` with RRF (k = 60). Reproduce `bm25()` exactly, with its sign and its IDF floor. Calculate the size of a memory. Count the copies that a `DELETE` leaves, and purge them. | §1, §3, §7 | ~1.5 h |
| 02 | [`a_memory_service_and_an_agent`](notebooks/02_a_memory_service_and_an_agent.ipynb) | T0 (T1: a real model with tool calls) | Serve memory over HTTP. The scope comes from a verified token, and the service returns 400 for a body that names a scope. Make a retried turn write only one time. Run the 07.1 loop with memory as tools, before every turn, or as a pinned profile. Put the "remember" of a poisoned page in quarantine. Refuse the "forget" of a page. | §2, §6, §8 | ~1.5 h |
| 03 | [`memory_layouts_and_the_prefix_cache`](notebooks/03_memory_layouts_and_the_prefix_cache.ipynb) | T0 simulated (T1: vLLM, cached tokens measured) | Predict and measure `cached_tokens` for each turn in three layouts: memory before the history, a pinned profile, and memory at the tail. Find where the prefix breaks. Calculate what the misses cost in TTFT (roofline) and in dollars on a hosted API with a caching minimum. Give each tenant its own `cache_salt`. | §5, §8 | ~1.5 h |
| 04 | [`consolidation_as_a_scheduled_job`](notebooks/04_consolidation_as_a_scheduled_job.ipynb) | T0 (T3 printed) | Consolidate episodes into facts with rules for supersede and precedence. Run the consolidation as a durable job, with a weekly run id, a weekly schedule to match it, a one-statement lease and checkpoints. Stop the job by force in the middle of a write, and resume it to the same state. Forget episodes by TTL. Print the commands for the Cloud Run job and Cloud Scheduler. | §7 | ~1 h |
| 05 | [`evaluate_forget_and_audit`](notebooks/05_evaluate_forget_and_audit.ipynb) | T0 (T1: a real embedder) | Grade five memory designs on 84 planted-fact questions in the shapes of LongMemEval and LoCoMo, with Wilson intervals. Select the budget at the recall knee. Forget an address everywhere: records, derived facts, idempotency rows, the prefix cache and the eval set. Show on disk that the address is gone. Read the audit trail. | §4, §5, §7, §8 | ~1 h |

Everything runs on a laptop first. The model is **scripted**: a template extractor and answerer. Thus every
accuracy number is reproducible, and you can attribute it to the memory system. The embedder is the **hashing
embedder** of 07.4's `ragkit`. It is lexical, not semantic. The paraphrase subset measures what this lexical match
costs.

The inference server in notebook 03 is a **fake**. Its prefix-cache hits obey the block rules of vLLM, its TTFTs come
from a roofline model, and its numbers have the label *simulated*.

You can put a real model, embedder or vLLM behind `MEMLAB_LLM_URL` / `MEMLAB_EMBED_URL`. Then the answers, the
embeddings and the cached-token counts become measurements. The prefill times stay a roofline estimate, and the
dollars stay a price table. Every table says which column is which. If Docker, Postgres or GCP is not available, the lab
prints the exact commands. For a step that a container answers when it is available, the lab also shows sample
output in the documented format (illustrative).

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the SQLite store, the service, the agent, the fake server, the job, the harness | every notebook, all tests |
| **T0 + Docker** | a laptop with Docker | the service, the fake model server and Postgres + pgvector in compose | [`deploy/local/`](deploy/local/), the last section of notebook 01 |
| **T1** | one GPU: Colab/Kaggle T4 (free) or a 24 GB card | Qwen2.5-1.5B-Instruct with tool calls and `cached_tokens`, and `bge-small-en-v1.5` on the pooling runner of vLLM | [`deploy/any-gpu/`](deploy/any-gpu/), notebooks 02, 03, 05 |
| **T3** | GCP | consolidation as a Cloud Run job on Cloud Scheduler, and the model on the Cloud Run GPU of the 04 serving lab | [`deploy/gcp/`](deploy/gcp/) (printed, no Terraform) |

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

For T1, start vLLM with the flags in [`deploy/any-gpu/`](deploy/any-gpu/). Then run
`export MEMLAB_LLM_URL=http://127.0.0.1:8000 MEMLAB_EMBED_URL=http://127.0.0.1:8001`. After that, run notebooks 02, 03
and 05 again (or `python3 -m memlab cachebench --url $MEMLAB_LLM_URL`).

For T0 + Docker, run `deploy/local/up.sh --pgvector`. Then run `pip install "psycopg[binary]"`. Then set
`MEMLAB_PG_DSN` as the script prints it.

## How it fits

The concepts are in [`../PRIMER.md`](../PRIMER.md). The minimal version from scratch is next to this lab, in
[`../memory-core/`](../memory-core/), and this lab never imports it. This lab reuses these parts, and
`tests/test_repo_numbers.py` reproduces them:

- the 07.1 agent loop and tool contract ([`agent-core`](../../agent-fundamentals/agent-core/)),
- 07.4's hashing embedder and RRF ([`rag-from-scratch`](../../retrieval-rag/rag-from-scratch/)),
- the prefix-cache rules of the serving lab and `minengine.perf.step_cost`
  ([`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/)),
- `scalelab.capacity.cost_per_call`
  ([`06-gateway/scaling-admission-cost`](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/)),
- the Wilson interval of 07.2.

Scope, delegation and audit are as the
[identity primer](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §2, §3.5, §8, §9
describe them. The durable job is as the
[durable-execution primer](../../long-running-durable/PRIMER.md#3-the-five-invariants) §3.2–§3.4 and the
[lra-gcp primer](../../long-running-durable/lra-gcp/docs/primer.md) §3.3 and §3.13 describe it (07.3,
[`long-running-durable`](../../long-running-durable/README.md)). For prices and for where to get GPUs, see
[`COMPUTE.md`](../../../COMPUTE.md).

## The library (`memlab/`, ~4,500 lines, numpy + aiohttp)

| Module | Lines | The idea |
|---|---:|---|
| `store/sqlite.py` | ~580 | The store is one file. It has typed rows, float32 BLOB vectors that numpy scores, FTS5 (`porter unicode61`) that `bm25()` ranks, and RRF k = 60. The `(tenant, user)` partition in every query, and as-of reads. A forget that goes along the provenance and purges FTS5, the WAL and the freed pages. The lease and checkpoint tables of the job. |
| `store/pgvector.py` | ~370 | The same store on Postgres + pgvector: DDL, hybrid search as RRF in SQL, a recursive forget and the job tables. The import of psycopg is lazy. In the tests, `pglast` parses every statement offline. |
| `records.py`, `deletion.py` | ~180 | The typed record (kind, scope, source, trust, provenance, validity, TTL, deletion key). The deletion report and the byte-level residue check. |
| `memory.py` | ~215 | The write path: `WritePolicy` (kinds per source, confidence floor, screening, quarantine). Resolution ADD / NOOP / UPDATE / ADD_HISTORY / FLAG. Idempotency keys that are unique in each partition, and the refusal of a different write under a known key. The pinned profile, and audit. |
| `service.py` | ~350 | The memory service (aiohttp): scope from a verified stand-in token, `Idempotency-Key`, forget by subject, review and promote, audit JSON lines. Its client and `RemoteMemory`. |
| `agent.py`, `llm.py`, `extract.py` | ~815 | The 07.1 loop with memory as tools, implicit or pinned. Three prompt layouts, provenance by taint and a confirm-gated forget. An OpenAI-compatible chat client and the scripted model. The grammar of extraction and of the answers. |
| `fakeserver.py`, `cachebench.py` | ~710 | A fake OpenAI-compatible server (chat with tool calls, embeddings, metrics with vLLM names, a block-hash prefix cache with `cache_salt`, the whole-cache reset of vLLM). The layout bench with the roofline TTFT and the price table (with the caching minimum of the provider). Each column has the label measured or modelled. |
| `consolidate.py` | ~300 | Consolidation as a durable run: a weekly run id and window, lease, checkpoints, a crash hook, resolution, reflection, the TTL mark. Sharding, and the printed GCP commands (image, identities, Cloud SQL, schedule, cleanup). |
| `harness.py`, `report.py` | ~420 | The planted-facts generator, the grader, Wilson intervals, the mode comparison, recall against budget and the knee. JSON and Markdown reports. |
| `audit.py`, `embedders.py`, `env.py`, `__main__.py` | ~540 | The `AuditEvent` of the identity lab, with `memory.*` types and GenAI memory names. The hashing and HTTP embedders. Tier detection. The CLI. |

## Deploy

[`deploy/`](deploy/) has three targets:

- [`local/`](deploy/local/): compose with the service, the fake model server and an optional
  `pgvector/pgvector:0.8.6-pg17`. `up.sh` / `down.sh` accept `DRY_RUN=1`.
- [`any-gpu/`](deploy/any-gpu/): the `serve.sh` of the 04 serving lab with tool calling,
  `--enable-prompt-tokens-details` and a pooling-runner embedder.
- [`gcp/`](deploy/gcp/): the consolidation job on Cloud Run jobs and Cloud Scheduler, and the model on the Cloud
  Run GPU of the serving lab. It also has a dated table of managed memory stores.

Each target has a README with the cost and the cleanup.

## Regenerating notebooks

The builder makes `notebooks/` (exercises) and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Computed, simulated, measured, illustrative.** A scripted model and a hashing embedder *compute* the harness
  numbers. These numbers are exact and reproducible, but they are not the accuracy of a model. The cached tokens
  of the fake server obey the rules of vLLM, and its TTFTs come from a roofline model. These numbers are
  *simulated*. Only a real server or embedder gives *measured* numbers. The pgvector sample in notebook 01 is
  *sample output in the documented format (illustrative)*.
- **This lab's policy and harness differ from memory-core's, on purpose.** `memory.WritePolicy` is stricter than
  the policy of PRIMER §2. It has a 0.7 floor instead of 0.6. Inferred facts can write semantic memory only. A
  fifth source, `consolidation`, is for the writes of the job itself, and it carries the trust of its strongest
  evidence.

  The harness has other users, other questions and a 128-token budget. Thus its mode ranking is not the ranking of
  PRIMER §6. Also, its knee is relative: the smallest budget that gets 95% of the best recall
  (`harness.knee(frac=0.95)`). But the knee of memory-core is absolute: within 2 points of the best. Neither harness is
  evidence about a real model, because both use a scripted answerer.
- **The token is a stand-in.** It is HMAC-signed. It has the claims and the failure modes of a real verifier, but
  not its cryptography. In production, the service verifies RS256/ES256 tokens against your IdP (identity primer
  §3.5).
- **A forget reaches what this process can reach.** The report counts the store, derived facts, checkpoints,
  idempotency rows, a prefix cache, an eval set and the audit log. Backups, WAL archives and the copies in other
  services expire on their own schedule. The deletion policy must say how long they stay. vLLM cannot evict the
  cached blocks of one tenant. Replace the `cache_salt` of the tenant with a new one, and keep the full reset of
  dev mode for the operator.
- **The memory service is SQLite-only.** The consolidation job runs on Postgres + pgvector. `memlab serve` does
  not, because its idempotency table is in SQLite SQL. Thus, if you want a Postgres deployment of the service, you
  must write that port yourself.
- **SQLite features vary.** FTS5 `secure-delete` needs SQLite 3.42 or newer (verify: the SQLite of Colab can be
  older). The store detects if the feature is available. If it is not available, the store uses `optimize`. The
  lab examines the deploy paths with `bash -n`, `DRY_RUN=1`, YAML checks and the parser of Postgres. It does not
  run them against real infrastructure here.

## Verify list (facts dated 2026-09-26 that move)

- vLLM **v0.30.0**: `--enable-auto-tool-choice --tool-call-parser hermes` for Qwen2.5.
  `--enable-prompt-tokens-details` for `usage.prompt_tokens_details.cached_tokens`. `--runner pooling` for
  embeddings (no `--task`). `cache_salt` has at most 128 characters and no `@ / \`, and it applies to the first
  block only. The default block size is 16. `/reset_prefix_cache` needs dev mode (`VLLM_SERVER_DEV_MODE=1`). It
  resets the whole cache, for every tenant, and answers `{"success": bool}`. There is no eviction by salt.
- pgvector **0.8.6** (`pgvector/pgvector:0.8.6-pg17`): `vector` indexes up to 2,000 dimensions. HNSW `m = 16`,
  `ef_construction = 64`, `hnsw.ef_search = 40`. `hnsw.iterative_scan` since 0.8.0. Filters apply after the index
  scan.
- SQLite: FTS5 `secure-delete` since 3.42.0. `bm25()` k1 = 1.2, b = 0.75, with the IDF floor at 1e-6. `PRAGMA
  secure_delete` is off by default, unless a compile-time option of the SQLite build sets it on.
- Cloud Run jobs and Cloud Scheduler: the v2 `jobs/<job>:run` URI with OAuth. `--task-timeout`, `--set-secrets`,
  `--set-cloudsql-instances`, `--service-account` and `--command/--args` on `gcloud run jobs create`. The
  `host=/cloudsql/<connection name>` DSN. Is `roles/run.invoker` sufficient to run a job? `CLOUD_RUN_TASK_INDEX` /
  `CLOUD_RUN_TASK_COUNT`.
- Model and embedder ids, sizes and licences (`Qwen/Qwen2.5-1.5B-Instruct`, `BAAI/bge-small-en-v1.5`). Prices in
  `cachebench.PRICES` (scalelab's, 5 Sep 2026), and the 4,096-token caching minimum on Gemini 3.x (scaling primer
  §5.5, request or prefix?). The OpenTelemetry GenAI memory attributes (status *development*).

MIT licensed.
