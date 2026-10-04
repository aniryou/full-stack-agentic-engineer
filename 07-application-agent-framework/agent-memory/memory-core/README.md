# memory-core — build an agent's long-term memory from typed records to a deletion that reaches every copy

After this core, you can explain every decision that the memory of an agent makes:

- What it writes, and from which source.
- What it retrieves inside a token budget.
- Where that memory is in the prompt, and what the prefix cache does with it.
- When episodes become facts.
- What "forget" must touch.

You can explain these decisions because you will fill in the code that makes them, in `memcore`. The package uses
the standard library + numpy, with no model and no network (~1,700 lines).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (what an agent remembers, and the write path).
2. Run `python3 -m pip install -e ".[dev]" && python3 -m pytest -q`. The 85 tests take approximately 20 s.
   The suite skips one test: the Colab injector check. The tests include these two:
   - "every number the primer quotes is recomputed"
   - "memcore reproduces ragkit's vectors, minengine's step times and the scaling primer's $0.0070 call"
3. Open [`notebooks/01_records_and_the_write_path.ipynb`](notebooks/01_records_and_the_write_path.ipynb). Look at the
   six writes. Their results are ADD, NOOP, UPDATE, QUARANTINE, REJECT, REJECT.

## What you get

*Tier T0 = laptop or Colab CPU, free. Everything here runs with no GPU, no model and no network.*

Each notebook starts with "The one-minute version" and works examples against the code. It gives exercises with a
check cell that prints ✅, and it ends with "In a design review". The finished versions are in
[`solutions/`](solutions/). The notebooks take approximately 6 hours, and 7.5 hours with the primer
(the curriculum of the repo, module 07.6).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_records_and_the_write_path`](notebooks/01_records_and_the_write_path.ipynb) | Calculate the cost of the transcript-as-memory baseline. Write episodic, semantic and procedural records. Extract after a turn. Write a policy check and a merge rule. Predict a sequence of writes. Make an idempotency key. | §1, §2 | 1 h | T0 |
| [`02_retrieval_and_the_planted_facts_harness`](notebooks/02_retrieval_and_the_planted_facts_harness.ipynb) | See the paraphrase miss of the lexical embedder. Implement both generative-agents scores, and see the code form serve stale facts. Pack a budget. Run the planted-facts harness with Wilson intervals. Find the knee. | §3, §4 | 1.5 h | T0 |
| [`03_the_context_budget_and_the_prefix_cache`](notebooks/03_the_context_budget_and_the_prefix_cache.ipynb) | Implement the three hit rules of vLLM. Derive the hits of each layout in closed form. Change them into TTFT (simulated) and dollars per session. Calculate the price of extraction. Repair a layout. Salt the cache for each tenant. | §5 | 1 h | T0 |
| [`04_consolidation_forgetting_and_deletion`](notebooks/04_consolidation_forgetting_and_deletion.ipynb) | Implement the consolidation rules and the lease rule. Run a job through a crash and a resume. Compare facts with raw episodes. Predict decay. Follow a deletion through provenance, caches, logs and backups. | §7 | 1.5 h | T0 |
| [`05_memory_tools_and_memory_poisoning`](notebooks/05_memory_tools_and_memory_poisoning.ipynb) | Compare tools, implicit retrieval and a pinned profile on the harness. Contain a poisoned page with taint. Fence memory. Select a profile. Write the injection golden case. Divide the controls between 06 and 07. | §6, §8 | 1 h | T0 |

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

Read the modules in this order. Each module starts with a docstring that gives the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`memcore/records.py`](memcore/records.py) | ~100 | One typed record for episodic, semantic and procedural memory: scope, source, provenance, confidence, importance, validity, TTL, deletion key. |
| [`memcore/store.py`](memcore/store.py) | ~135 | Scope is a partition, not a filter. The crc32 hashing embedder of ragkit, implemented again. A flat index and a full-text index that both delete. |
| [`memcore/write.py`](memcore/write.py) | ~220 | Extraction after the turn. The write policy (kinds per source, a confidence floor, screening). ADD / UPDATE / ADD_HISTORY / NOOP, which keeps superseded facts and keeps late, older values as history. Idempotency keys that name the step, not the content. |
| [`memcore/retrieve.py`](memcore/retrieve.py) | ~100 | The generative-agents score in the form of the paper and in the form of the code. Greedy packing into a token budget. As-of filters. Reads are writes. |
| [`memcore/budget.py`](memcore/budget.py) | ~270 | The prefix-cache rules of vLLM and a salted block-hash cache. Secret salts for each tenant, and their rotation. Three memory layouts. Prefill time on the roofline (SIMULATED). Dollars per turn at dated prices, with the caching minimum of the provider. `Budget`. |
| [`memcore/consolidate.py`](memcore/consolidate.py) | ~195 | Episodes to facts with precedence, supersession in valid time, and flags. A run id, a lease with a heartbeat and a fence, checkpoints and a crash hook. Reflection with explicit exits. |
| [`memcore/forget.py`](memcore/forget.py) | ~150 | Decay, TTL and caps. `propagate()` to every surface on word boundaries, with a review list for what provenance finds and content cannot find. `residue()` to prove it. |
| [`memcore/agent.py`](memcore/agent.py) | ~250 | The loop of agent-core with `remember` / `recall` / `forget`. Tools, implicit and pinned modes (the agent pins the profile again when a pinned slot changes). Taint. Fencing. Audit events with the field names of the identity lab. |
| [`memcore/harness.py`](memcore/harness.py) | ~270 | The planted-facts generator in the task shapes of LongMemEval and LoCoMo (and its two artefacts, which you can switch on and off). Evaluation, Wilson and per-user intervals, recall against budget, the knee, the three modes compared. |

## What the tests prove

`tests/` has one focused test for each concept (77, plus 8 checks of the notebook tooling). The tests run offline in
~20 s:

- **Formulas pinned to hand-computed values:**
  - The generative-agents worked example in both forms: (A 2.0000, B 2.1364, C 0.4286) and (A 3.0000, B 3.7487,
    C 1.3571).
  - The oldest-first recency of the code form.
  - The buckets and cosines (0.2236) of the hashing embedder.
  - Packing.
  - The cached tokens per turn of every layout, in closed form.
  - Retention decay (0.225).
  - The Wilson interval (45/50 gives 0.7864–0.9565).
  - These tests are in `test_write_retrieve.py`, `test_budget.py`, `test_consolidate_forget.py` and
    `test_agent_harness.py`.
  - The bill of a hosted API ignores the layout below its 4,096-token caching minimum.
- **Behaviour that must hold:**
  - A search never crosses a partition.
  - A delete removes the record, its vector and its full-text postings, with no tombstone.
  - A retried write writes one time only, also when its extraction uses different words. The writer refuses a
    different write under a known key.
  - A weaker source cannot overwrite a stronger source.
  - An older value never supersedes a newer value. This is true in the writer, and in a consolidation run over
    windows that are out of order.
  - A consolidation job can crash after the writes of a slot and before its checkpoint. When the lease expires,
    another worker resumes the job. The job writes each fact one time only (random ids cause duplicates).
  - A slow worker that lost its lease stops before it writes.
  - A deletion matches values on word boundaries, lists paraphrases for review and rotates the cache salt of the
    tenant.
  - The policy quarantines and rejects a poisoned page. A permissive policy lets the page answer the adversarial
    question.
  - `forget` is confirm-gated.
  - The memory budget shapes packing and writes. Model calls fail closed.
  - A write to a pinned slot pins the profile again.
- **The limits of the fixture, pinned:**
  - The slot hints flatter raw episodes.
  - With thirty more facts per user, the lead of the pinned profile disappears.
  - Every user misses the same question. Thus a per-user interval collapses to a point.
- **Reproduced repo numbers** (`test_repo_numbers.py`). The tests pin each number by hand. When the other lab is in
  the checkout, they also compare the number with the original by path import:
  - The vectors of `ragkit.embed.HashingEmbedder`.
  - The `expected_cached_tokens` (48, 32, 16) of the serving lab, against its notebook source.
  - `minengine.kv` lookups.
  - `minengine.perf.step_cost` (78.7 against 15.1 ms on an L4).
  - `capacity.ttft_s` (0.0970 s).
  - `scalelab.capacity.cost_per_call` ($0.007005) and the `token_cost` of `agentlab` ($0.002335).
  - The `wilson_interval` and `count_tokens` of `agentlab`.
  - The `AuditEvent` field names and `args_digest` of the identity lab.
- **The numbers of the primer** (`test_primer_numbers.py`). The tests recompute every computed number in
  [`../PRIMER.md`](../PRIMER.md). Each number must appear verbatim.

## Caveats: what is simplified

A **scripted model** takes the place of the LLM:

- `write.extract` is a template extractor.
- `harness.read_answer` is a template reader.
- `agent.scripted_model` is a rule set that calls tools. It obeys instructions in pages, on purpose. It calls
  `recall` only when a turn names a slot. Thus "tools miss the task" is a rule, not an observation.

The **embedder is lexical** (crc32 feature hashing). Thus paraphrases miss by design. Tokens are `len(text) // 4`
and token ids are 4-character chunks. Thus text prefixes are token prefixes (a real tokenizer can move one token at a
boundary).

The roofline model of `minengine.perf` (datasheet GPU numbers, 0.6/0.8 efficiencies, verify) gives the prefill
times, which are **SIMULATED**. The prices are dated list prices `(verify)`. The model treats the caching minimum of the
provider as a threshold on the request. It assumes that the cache of the provider hits the same prefix as the cache
of vLLM.

The prefix cache has no LRU pressure. Like vLLM v0.30.0, it gives only a full reset, and no eviction by salt. The
store and the idempotency journal are in memory. SQLite, pgvector and on-disk deletion are the job of the lab.

## Regenerating notebooks

The builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three commands and the tests. On Colab, the first cell of each notebook clones the repo and
installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../memory-lab/`](../memory-lab/). The lab has these parts:

- SQLite with FTS5 and vectors, and a pgvector twin.
- A memory service over HTTP that takes its scope from a verified token.
- Prefix-cache hits measured on vLLM.
- Consolidation as a scheduled job, with a crash and a resume.
- A deletion examined on the bytes of the database file.

For where each tier runs and what it costs, see [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
