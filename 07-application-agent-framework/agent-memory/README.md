# agent-memory — remember what matters, for whom and for how long, and prove you forgot

After this topic you can decide what an agent writes to memory, from which source and for whom; retrieve it by
similarity, recency and importance inside a per-turn token budget; lay it out so the prefix cache survives and price a
turn with and without it; consolidate episodes into facts on a schedule; carry a deletion to every copy; and keep
another tenant's memory, and an attacker's, out of the prompt.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §1–§3 (40 min): the typed memory record, the write path, and retrieval — including
   where the generative-agents paper and its code disagree.
2. `cd memory-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q` — 85 tests, about 20 s. The
   fastest win, under a second:
   `python3 -c "from memcore import hits_per_turn, hit_rate, LAYOUTS; print({l: f'{hit_rate(hits_per_turn(l)):.0%}'
   for l in LAYOUTS})"`
   prints the prefix-cache hit rate of four memory layouts (88%, 58%, 88%, 72%). Then open
   [`memory-core/notebooks/01_records_and_the_write_path.ipynb`](memory-core/notebooks/01_records_and_the_write_path.ipynb).
3. When you want real storage and a real server: [`memory-lab/`](memory-lab/) puts the same design on SQLite (FTS5 +
   vectors) and Postgres + pgvector, behind a memory service over HTTP, measures cache hits on vLLM, and checks a
   deletion on the bytes of the database file.

## What you get

*Tiers: **T0** = laptop or Colab CPU, free — every concept here runs at T0; **T0 + Docker** = Postgres + pgvector on
your machine, still free; **T1** = one small GPU (Colab/Kaggle T4 or a rented card) for a real embedder or a small
chat model with tool calls; **T3** = the Google Cloud deployment, optional and printed rather than run.* Each notebook
opens with "The one-minute version", works examples against the code, sets exercises with a check that prints ✅,
and closes with "In a design review". Finished versions are in [`memory-core/solutions/`](memory-core/solutions/).
Module 07.6 in the learning path.

| Path | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`memory-core`](memory-core/) `01_records_and_the_write_path` | name the kinds of memory and their lifetimes; write a typed record; gate writes by source, confidence and screening; merge (ADD / UPDATE / NOOP) and keep superseded facts; make a retried turn write once | §1, §2 | 1 h | T0 |
| `02_retrieval_and_the_planted_facts_harness` | score by similarity, recency and importance in both generative-agents forms; pack a token budget; build a planted-facts benchmark in LongMemEval's and LoCoMo's task shapes; read accuracy with a Wilson interval, recall, stale answers and abstention; find the knee | §3, §4 | 1.5 h | T0 |
| `03_the_context_budget_and_the_prefix_cache` | predict the prefix-cache hits of three memory layouts from vLLM's block rules; turn lost hits into TTFT (simulated) and dollars per turn; price extraction; salt the cache per tenant | §5 | 1 h | T0 |
| `04_consolidation_forgetting_and_deletion` | consolidate episodes into facts with precedence and supersession as a durable job (run id, lease, checkpoints, a crash and a resume); forget by decay, TTL and caps; follow a deletion to every copy | §7 | 1.5 h | T0 |
| `05_memory_tools_and_memory_poisoning` | compare `remember` / `recall` / `forget` tools, retrieval before every turn and a pinned profile; stop a poisoned page from becoming a standing instruction; fence memory as data; split the controls between the gateway (06) and the agent (07) | §6, §8 | 1 h | T0 |
| [`memory-lab`](memory-lab/) | the same design on SQLite with FTS5 and vectors and a pgvector twin, a memory service and an agent over HTTP, prefix hits measured on vLLM, consolidation as a scheduled job, a deletion checked on disk | §1–§9 | ~6.5 h | T0, T0 + Docker, T1, T3 printed |

## Run it

```bash
cd memory-core
python3 -m pip install -e ".[dev]"     # the library needs numpy; dev adds pytest and Jupyter
python3 -m pytest -q                    # 85 tests, ~20 s (one skip: the Colab injector check, not used here)
python3 tools/build_notebooks.py        # (re)build notebooks/ and solutions/ from notebooks_src/
python3 -m jupyterlab notebooks         # do the exercises
```

The library needs nothing but numpy, and no model:

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

Read the [agent-core loop and tool contracts](../agent-fundamentals/agent-core/) (07.1) first; this topic replaces its
"memory is the transcript" baseline. It reuses — and cites, section by section — the
[agent platform lab](../agent-fundamentals/gcp-agent-platform-lab/) (07.2: sessions, context layout, evals, injection
defences), durable execution (07.3: idempotency, leases, budgets, scheduled runs), [retrieval-rag](../retrieval-rag/)
(07.4: the hashing embedder, hybrid search, the vector-database primer's partitions and tombstones), the
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) §5 (04.3: the prefix cache whose block rules
price the context budget) and the [identity
primer](../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
(06: ASI06, delegated identity, tenancy, the audit event). [sandboxed-execution](../sandboxed-execution/) (07.5) is its
neighbour: both treat tool output as untrusted. Where each tier runs and what it costs: [`COMPUTE.md`](../../COMPUTE.md);
the learning path: [`CURRICULUM.md`](../../CURRICULUM.md).

## Going further / caveats

- **What is real at T0, and what is not.** Records, the write policy, retrieval, the harness, consolidation, deletion
  and the agent all run and are tested offline. A **scripted model** (rules, no weights) stands in for the extraction
  and answering LLM, and a **lexical hashing embedder** for a real one — so every number is reproducible, and the
  paraphrase subset shows what a real embedder must buy (the lab's T1 step). Prefill times are **SIMULATED** with a
  roofline model (the same one as `minengine.perf`); prices are dated list prices `(verify)`.
- **The harness has built-in advantages; read them before quoting it.** Slot hints in the planted statements flatter
  raw episodes, a user's whole memory barely exceeds a 60-token profile, and the scripted model recalls only when a
  question names a slot. PRIMER §4 and §6 quantify each; the comparison that decides a design is a real model (T1).
- **The benchmarks are shapes, not data.** The harness copies LongMemEval's and LoCoMo's task types; it bundles and
  downloads nothing (LoCoMo's data is CC BY-NC 4.0).
- **Deletion on disk is the lab's job.** The core shows every surface a deletion must reach; the lab proves it on a
  real SQLite file, where FTS5 and WAL keep deleted text until you make them let go.
- **Product facts move.** mem0 went ADD-only in 2.0.0, Letta's Python server was archived, OpenTelemetry's memory
  conventions are still in development; the primer's Verify list is dated 2026-09-26.
