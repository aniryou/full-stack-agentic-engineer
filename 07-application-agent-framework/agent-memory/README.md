# agent-memory — remember what matters, for whom and for how long, and prove you forgot

After this topic, you can do these things:

- Decide what an agent writes to memory, from which source and for whom.
- Retrieve memory by similarity, recency and importance inside a token budget for each turn.
- Put memory in the prompt in a layout that keeps the prefix cache. Calculate the price of a turn with the prefix
  cache and without it.
- Consolidate episodes into facts on a schedule.
- Send a deletion to every copy.
- Keep the memory of another tenant, and the memory of an attacker, out of the prompt.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §1–§3 (40 min). These sections give the typed memory record, the write path and
   retrieval. They also show where the generative-agents paper and its code disagree.
2. Run `cd memory-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q`. The 85 tests take
   approximately 20 s. For the fastest result, in less than one second, run
   `python3 -c "from memcore import hits_per_turn, hit_rate, LAYOUTS; print({l: f'{hit_rate(hits_per_turn(l)):.0%}'
   for l in LAYOUTS})"`.
   It prints the prefix-cache hit rate of four memory layouts (88%, 58%, 88%, 72%). Then open
   [`memory-core/notebooks/01_records_and_the_write_path.ipynb`](memory-core/notebooks/01_records_and_the_write_path.ipynb).
3. For real storage and a real server, go to [`memory-lab/`](memory-lab/). The lab puts the same design on SQLite (FTS5 +
   vectors) and on Postgres + pgvector, behind a memory service over HTTP. It measures cache hits on vLLM. It also
   examines a deletion on the bytes of the database file.

## What you get

*Tiers: **T0** = laptop or Colab CPU, free. Every concept here runs at T0. **T0 + Docker** = Postgres + pgvector on
your machine, also free. **T1** = one small GPU (Colab/Kaggle T4 or a rented card) for a real embedder or for a small
chat model with tool calls. **T3** = the Google Cloud deployment. It is optional, and the lab prints it but does not
run it.*

Each notebook starts with "The one-minute version" and works examples against the code. It gives exercises
with a check that prints ✅, and it ends with "In a design review". The finished versions are in
[`memory-core/solutions/`](memory-core/solutions/). The topic is module 07.6 in the learning path.

| Path | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`memory-core`](memory-core/) `01_records_and_the_write_path` | Name the kinds of memory and their lifetimes. Write a typed record. Gate writes by source, confidence and screening. Merge (ADD / UPDATE / NOOP) and keep superseded facts. Make a retried turn write one time only. | §1, §2 | 1 h | T0 |
| `02_retrieval_and_the_planted_facts_harness` | Score by similarity, recency and importance in both generative-agents forms. Pack a token budget. Build a planted-facts benchmark in the task shapes of LongMemEval and LoCoMo. Read accuracy with a Wilson interval, recall, stale answers and abstention. Find the knee. | §3, §4 | 1.5 h | T0 |
| `03_the_context_budget_and_the_prefix_cache` | Predict the prefix-cache hits of three memory layouts from the block rules of vLLM. Change lost hits into TTFT (simulated) and dollars per turn. Calculate the price of extraction. Salt the cache for each tenant. | §5 | 1 h | T0 |
| `04_consolidation_forgetting_and_deletion` | Consolidate episodes into facts with precedence and supersession as a durable job (run id, lease, checkpoints, a crash and a resume). Forget by decay, TTL and caps. Follow a deletion to every copy. | §7 | 1.5 h | T0 |
| `05_memory_tools_and_memory_poisoning` | Compare three designs: `remember` / `recall` / `forget` tools, retrieval before every turn, and a pinned profile. Stop a poisoned page before it becomes an instruction that stays. Fence memory as data. Divide the controls between the gateway (06) and the agent (07). | §6, §8 | 1 h | T0 |
| [`memory-lab`](memory-lab/) | The same design on SQLite with FTS5 and vectors and on a pgvector twin. A memory service and an agent over HTTP. Prefix hits measured on vLLM. Consolidation as a scheduled job. A deletion examined on disk. | §1–§9 | ~6.5 h | T0, T0 + Docker, T1, T3 printed |

## Run it

```bash
cd memory-core
python3 -m pip install -e ".[dev]"     # the library needs numpy; dev adds pytest and Jupyter
python3 -m pytest -q                    # 85 tests, ~20 s (one skip: the Colab injector check, not used here)
python3 tools/build_notebooks.py        # (re)build notebooks/ and solutions/ from notebooks_src/
python3 -m jupyterlab notebooks         # do the exercises
```

The library needs only numpy. It needs no model:

```python
from memcore import MemoryStore, Scope, Writer, extract, retrieve, DAY
store, alice = MemoryStore(), Scope("acme", "alice")
w = Writer(store)
for day, text in [(0, "I live in Lisbon."), (3, "I moved to Porto.")]:
    for rec in extract(text, alice, at=day * DAY):
        print(w.write(rec).action, rec.text)      # ADD ..., ADD ..., ADD ..., UPDATE The user's home city is Porto.
print([r.value for r in retrieve(store, alice, "What is the user's home city?", now=5 * DAY, kinds=("semantic",)).records])
```

## How it fits

Read the [agent-core loop and tool contracts](../agent-fundamentals/agent-core/) (07.1) first. This topic replaces
its baseline, "memory is the transcript". This topic reuses these topics, and cites them section by section:

- The [agent platform lab](../agent-fundamentals/gcp-agent-platform-lab/) (07.2: sessions, context layout, evals,
  injection defences).
- [durable execution](../long-running-durable/README.md) (07.3: idempotency, leases, budgets, scheduled runs).
- [retrieval-rag](../retrieval-rag/) (07.4: the hashing embedder, hybrid search, the partitions and tombstones of the
  vector-database primer).
- The [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) §5 (04.3: the prefix cache whose
  block rules set the price of the context budget).
- The [identity
  primer](../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) (06: ASI06, delegated identity,
  tenancy, the audit event).

[sandboxed-execution](../sandboxed-execution/) (07.5) is its neighbour. Both topics treat tool output as untrusted.
For where each tier runs and what it costs, see [`COMPUTE.md`](../../COMPUTE.md). For the learning path, see
[`CURRICULUM.md`](../../CURRICULUM.md).

## Going further / caveats

- **What is real at T0, and what is not.** Records, the write policy, retrieval, the harness, consolidation, deletion
  and the agent all run offline, and tests examine them. A **scripted model** (rules, no weights) takes the place of
  the LLM that extracts and answers. A **lexical hashing embedder** takes the place of a real embedder. Thus every
  number is reproducible. The paraphrase subset shows what a real embedder must buy (the T1 step of the lab). A
  roofline model (the same one as `minengine.perf`) gives the prefill times, which are **SIMULATED**. The prices are
  dated list prices `(verify)`.
- **The harness has built-in advantages. Read them before you quote it.** Slot hints in the planted statements
  flatter raw episodes. The whole memory of one user is only slightly larger than a 60-token profile. The scripted
  model recalls only when a question names a slot. PRIMER §4 and §6 quantify each advantage. The comparison that
  decides a design uses a real model (T1).
- **The benchmarks are shapes, not data.** The harness copies the task types of LongMemEval and LoCoMo. It bundles
  nothing and downloads nothing (the LoCoMo data is CC BY-NC 4.0).
- **Deletion on disk is the job of the lab.** The core shows every surface that a deletion must reach. The lab proves
  the deletion on a real SQLite file. On that file, FTS5 and WAL keep deleted text until you make them release it.
- **Product facts change.** mem0 went ADD-only in 2.0.0. Letta archived its Python server. The memory
  conventions of OpenTelemetry are still in development. The date of the Verify list in the primer is 2026-09-26.
