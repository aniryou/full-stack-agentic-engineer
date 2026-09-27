# memory-core — build an agent's long-term memory from typed records to a deletion that reaches every copy

After this you can explain every decision an agent's memory makes — what it writes and from which source, what it
retrieves inside a token budget, where that memory sits in the prompt and what the prefix cache does with it, when
episodes become facts, and what "forget" has to touch — because you will have filled in the code that makes it, in
`memcore`: standard library + numpy, no model, no network (~1,700 lines).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (what an agent remembers; the write path).
2. `python3 -m pip install -e ".[dev]" && python3 -m pytest -q` — 85 tests in about 20 s (one skipped: the Colab
   injector check), including "every number the
   primer quotes is recomputed" and "memcore reproduces ragkit's vectors, minengine's step times and the scaling
   primer's $0.0070 call".
3. Open [`notebooks/01_records_and_the_write_path.ipynb`](notebooks/01_records_and_the_write_path.ipynb) and watch
   six writes come out as ADD, NOOP, UPDATE, QUARANTINE, REJECT, REJECT.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU, no model and no network.* Each notebook opens
with "The one-minute version", works examples against the code, sets exercises with a check cell that prints ✅, and
ends with "In a design review". Finished versions are in [`solutions/`](solutions/). About 6 hours for the notebooks,
7.5 with the primer
(the repo's curriculum, module 07.6).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_records_and_the_write_path`](notebooks/01_records_and_the_write_path.ipynb) | cost the transcript-as-memory baseline; write episodic, semantic and procedural records; extract after a turn; write a policy check and a merge rule; predict a write sequence; make an idempotency key | §1, §2 | 1 h | T0 |
| [`02_retrieval_and_the_planted_facts_harness`](notebooks/02_retrieval_and_the_planted_facts_harness.ipynb) | see the lexical embedder's paraphrase miss; implement both generative-agents scores and watch the code form serve stale facts; pack a budget; run the planted-facts harness with Wilson intervals; find the knee | §3, §4 | 1.5 h | T0 |
| [`03_the_context_budget_and_the_prefix_cache`](notebooks/03_the_context_budget_and_the_prefix_cache.ipynb) | implement vLLM's three hit rules; derive each layout's hits in closed form; turn them into TTFT (simulated) and dollars per session; price extraction; fix a layout; salt the cache per tenant | §5 | 1 h | T0 |
| [`04_consolidation_forgetting_and_deletion`](notebooks/04_consolidation_forgetting_and_deletion.ipynb) | implement the consolidation rules and the lease rule; run a job through a crash and a resume; compare facts with raw episodes; predict decay; follow a deletion through provenance, caches, logs and backups | §7 | 1.5 h | T0 |
| [`05_memory_tools_and_memory_poisoning`](notebooks/05_memory_tools_and_memory_poisoning.ipynb) | compare tools, implicit retrieval and a pinned profile on the harness; contain a poisoned page with taint; fence memory; choose a profile; write the injection golden case; split controls between 06 and 07 | §6, §8 | 1 h | T0 |

## Run it

```bash
cd memory-core
python3 -m pip install -e ".[dev]"            # or: python3 -m pip install -r requirements.txt
python3 -m pytest -q                           # 85 tests, ~20 s (one skipped by design)
python3 -m jupyterlab notebooks                # do the exercises
```

The library needs only numpy:

```python
from memcore import MemoryAgent, MemoryStore, Scope, UserTurn, DAY
agent = MemoryAgent(MemoryStore(), Scope("acme", "alice"), mode="pinned")
agent.start_session("s1", 0)
agent.run(UserTurn("I prefer window seats."), now=DAY)            # extracted and written after the turn
agent.start_session("s2", 2 * DAY)                                # the profile is pinned for the session
print(agent.run(UserTurn("Book me a flight to Rome.", needs=("seat_preference",)), now=2 * DAY).text)
# Done (window).
print([(e.event_type, e.decision) for e in agent.audit])          # every memory read and write is audited
```

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`memcore/records.py`](memcore/records.py) | ~100 | one typed record for episodic, semantic and procedural memory: scope, source, provenance, confidence, importance, validity, TTL, deletion key |
| [`memcore/store.py`](memcore/store.py) | ~135 | scope is a partition, not a filter; ragkit's crc32 hashing embedder, re-implemented; a flat index and a full-text index that both delete |
| [`memcore/write.py`](memcore/write.py) | ~220 | extraction after the turn; the write policy (kinds per source, a confidence floor, screening); ADD / UPDATE / ADD_HISTORY / NOOP with superseded facts kept and late, older values kept as history; idempotency keys that name the step, not the content |
| [`memcore/retrieve.py`](memcore/retrieve.py) | ~100 | the generative-agents score in the paper's form and the code's; greedy packing into a token budget; as-of filters; reads are writes |
| [`memcore/budget.py`](memcore/budget.py) | ~270 | vLLM's prefix-cache rules and a salted block-hash cache; secret per-tenant salts and their rotation; three memory layouts; prefill time on the roofline (SIMULATED); dollars per turn at dated prices with the provider's caching minimum; `Budget` |
| [`memcore/consolidate.py`](memcore/consolidate.py) | ~195 | episodes to facts with precedence, supersession in valid time and flags; a run id, a lease with a heartbeat and a fence, checkpoints and a crash hook; reflection with explicit exits |
| [`memcore/forget.py`](memcore/forget.py) | ~150 | decay, TTL and caps; `propagate()` to every surface on word boundaries, with a review list for what provenance finds and content cannot, and `residue()` to prove it |
| [`memcore/agent.py`](memcore/agent.py) | ~250 | agent-core's loop with `remember` / `recall` / `forget`; tools, implicit and pinned modes (re-pinned when a pinned slot changes); taint; fencing; audit events with the identity lab's field names |
| [`memcore/harness.py`](memcore/harness.py) | ~270 | the planted-facts generator in LongMemEval's and LoCoMo's task shapes (and its two artefacts, switchable); evaluation, Wilson and per-user intervals, recall against budget, the knee, the three modes compared |

## What the tests prove

`tests/` has one focused test per concept (77, plus 8 notebook-tooling checks; offline, ~20 s):

- **Formulas pinned to hand-computed values:** the generative-agents worked example in both forms (A 2.0000, B 2.1364,
  C 0.4286; A 3.0000, B 3.7487, C 1.3571) and the code form's oldest-first recency; the hashing embedder's buckets and
  cosines (0.2236); packing; every layout's cached tokens per turn in closed form; retention decay (0.225); the Wilson
  interval (45/50 → 0.7864–0.9565) (`test_write_retrieve.py`, `test_budget.py`, `test_consolidate_forget.py`,
  `test_agent_harness.py`); a hosted API's bill ignores the layout below its 4,096-token caching minimum.
- **Behaviour that must hold:** search never crosses a partition; a delete removes the record, its vector and its
  full-text postings with no tombstone; a retried write writes once even when its extraction is reworded (and a
  different write under a known key is refused); a weaker source cannot overwrite a stronger one; an older value
  never supersedes a newer one, in the writer or in a consolidation run over out-of-order windows; a consolidation job
  that crashes after a slot's writes and before its checkpoint, resumed by another worker after the lease expires,
  writes each fact once (random ids would duplicate); a slow worker that lost its lease stops before writing; a
  deletion matches values on word boundaries, lists paraphrases for review and rotates the tenant's cache salt; a
  poisoned page is quarantined and rejected, and a permissive policy lets it answer the adversarial question;
  `forget` is confirm-gated; the memory budget shapes packing and writes and model calls fail closed; a write to a
  pinned slot re-pins the profile.
- **The fixture's limits, pinned:** the slot hints flatter raw episodes; with thirty more facts per user the pinned
  profile's lead disappears; every user misses the same question, so a per-user interval collapses to a point.
- **Reproduced repo numbers** (`test_repo_numbers.py`), each pinned by hand and, where the other lab is in the
  checkout, cross-checked against the original by path import: `ragkit.embed.HashingEmbedder`'s vectors; the serving
  lab's `expected_cached_tokens` (48, 32, 16) against its notebook source; `minengine.kv` lookups; `minengine.perf.step_cost`
  (78.7 vs 15.1 ms on an L4); `capacity.ttft_s` (0.0970 s); `scalelab.capacity.cost_per_call` ($0.007005) and
  `agentlab`'s `token_cost` ($0.002335); `agentlab`'s `wilson_interval` and `count_tokens`; the identity lab's
  `AuditEvent` field names and `args_digest`.
- **The primer's numbers** (`test_primer_numbers.py`): every computed number in [`../PRIMER.md`](../PRIMER.md) is
  recomputed and must appear verbatim.

## Caveats: what is simplified

A **scripted model** stands in for the LLM: `write.extract` is a template extractor, `harness.read_answer` a template
reader, `agent.scripted_model` a rule set that calls tools — and obeys instructions in pages, on purpose; it calls
`recall` only when a turn names a slot, so "tools miss the task" is a rule, not an observation. The
**embedder is lexical** (crc32 feature hashing), so paraphrases miss by design. Tokens are `len(text) // 4` and token
ids are 4-character chunks, so text prefixes are token prefixes (a real tokenizer may shift one token at a boundary).
Prefill times are **SIMULATED** with the roofline model of `minengine.perf` (datasheet GPU numbers, 0.6/0.8
efficiencies, verify); prices are dated list prices `(verify)`, with the provider's caching minimum modelled as a
threshold on the request and its cache assumed to hit the prefix vLLM's would. The prefix cache has no LRU pressure;
like vLLM v0.30.0 it offers only a full reset, no eviction by salt. The store and the idempotency journal are
in-memory; SQLite, pgvector and on-disk deletion are the lab's.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs this
package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../memory-lab/`](../memory-lab/) for SQLite with FTS5 and vectors and a pgvector twin, a memory service over
HTTP that takes its scope from a verified token, prefix-cache hits measured on vLLM, consolidation as a scheduled job
with a crash and a resume, and a deletion checked on the bytes of the database file. Where each tier runs and what it
costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
