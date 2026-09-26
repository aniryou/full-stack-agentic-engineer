# orchestrator-core — predict what a routing or scaling decision does to a fleet, on a laptop

A fleet of LLM engines you can run on a CPU: **`fleetsim`**, a pure-Python discrete-event simulator of replicas with
prefix caches, the routers in front of them, the Kubernetes HPA that sizes them, prefill/decode disaggregation and
KV-cache tiers — so you can say which router, which autoscaling signal and which split a workload needs before you
spend GPU hours.

## Start here

1. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 57 tests in about 4 s, offline.
2. Run the [snippet below](#run-it): power of two choices against the llm-d-style endpoint picker on agent sessions;
   it prints hit rate and TTFT p95 (simulated) in under a second.
3. Open [`notebooks/01_why_llm_load_balancing_is_different.ipynb`](notebooks/01_why_llm_load_balancing_is_different.ipynb)
   with [PRIMER §1](../PRIMER.md#1-why-a-layer-above-the-engine) beside it.

## What you get

*Tier: T0 = laptop or Colab CPU, free.* Everything here is T0: standard library only, deterministic under a seed,
every simulation a few seconds on a CPU.

| Notebook | You will be able to… | Time | Tier |
|---|---|---|---|
| [`01_why_llm_load_balancing_is_different`](notebooks/01_why_llm_load_balancing_is_different.ipynb) | say what a request costs in prefill, KV and residency; rank five load-only routers on a heterogeneous fleet; explain power of two choices and herding on stale metrics; hold a burst in a router queue with priorities and a per-endpoint cap sized by Little's law | ~1.5 h | T0 |
| [`02_cache_aware_routing_and_the_load_tradeoff`](notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) | compute chain hashes and the upstream scorer formulas; place prefix hashing, bounded loads, the weighted endpoint picker and the affinity filter's TTFT gate on the locality-vs-load spectrum; tune the prefix weight; survive a hot prefix; judge how stale an approximate index gets; route LoRA adapters | ~1.5 h | T0 |
| [`03_autoscaling_on_the_right_signal`](notebooks/03_autoscaling_on_the_right_signal.ipynb) | compare GPU util, running, KV and queue under load, and five signals on a traffic step; apply the HPA rule with stabilization, policies and Pending pods; pick targets from a load test; budget cold start and headroom; explain why a request-count target does not transfer to a chat + RAG mix and a prefill-backlog + KV signal does; decide on scale-to-zero once the node goes too | ~1.5 h | T0 |
| [`04_prefill_decode_disaggregation`](notebooks/04_prefill_decode_disaggregation.ipynb) | measure the prefill stall; price the KV transfer on L4 vs H100; search every xPyD split of eight GPUs against the analytic ratio; bound the decode batch by KV and ITL; try smaller chunks first; disaggregate conditionally | ~2 h | T0 |
| [`05_kv_cache_tiers_and_agent_sessions`](notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) | size an agent fleet's KV working set; show why routing cannot fix a capacity problem; decide fetch vs recompute; run a two-tier LRU; predict sticky vs random vs shared tiers; size DRAM from a replay | ~2 h | T0 |

Times are rough, include the matching [PRIMER](../PRIMER.md) section, and together come to roughly the 9 hours the
repo's curriculum (`CURRICULUM.md` at the repo root) budgets for the primer and this core. Each notebook has a **Tier**
line, "The one-minute version", worked examples, exercises with `# YOUR CODE HERE` followed by a check cell that
prints ✅, and "In a design review" drills; worked answers are in [`solutions/`](solutions/).

## Run it

```bash
cd orchestrator-core
python3 -m pip install -r requirements.txt   # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                          # 57 tests, ~4 s
python3 -m jupyterlab notebooks               # do the exercises
```

```python
from fleetsim import L4_8B, Fleet, PowerOfTwo, agentic, epp, table

for router in (PowerOfTwo(seed=1), epp((3, 2, 2))):              # load only vs prefix + queue + KV scoring
    reqs = agentic(0.6, 240, seed=3, groups=6, system=2000, tool=500, output=100, turns=(4, 10), think=(1, 4))
    s = Fleet(L4_8B, 4, router).run(reqs).summary(ttft_slo=1.0)
    print(router.name, round(s["hit_rate"], 2), round(s["ttft_p95"], 2))   # simulated
```

On Colab, the first cell of each notebook clones the repository and installs this package (see the repository's
[`COLAB.md`](../../../COLAB.md)).

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all of it plus the tests.

## What the model leaves out

Every number `fleetsim` prints is **simulated**: an engine model built from spec-sheet arithmetic (an 8B model on
an L4 by default), not a measurement. It is for building intuition and checking designs. Read every simulated
number with these in mind; each is also stated where it matters in the notebooks and the primer.

- **Prefill compute is linear in tokens.** Attention FLOPs (about 4 × layers × d_model × context per token) are
  omitted: +10 % for a 6,000-token prompt on the 8B model, +16 % at 10,000, +50 % at 30,000. Long-context prefill,
  P/D transfer budgets (notebook 04) and recompute costs (notebook 05) are optimistic by those amounts.
- **No per-chunk efficiency loss.** A 256-token prefill chunk costs the same per token as a 2,048-token one, so the
  chunk-budget sweep in notebook 04 is the optimistic end.
- **Metric staleness is a maximum age.** A scraped snapshot is re-read on the first look after it expires — no
  scrape jitter, no per-replica phase.
- **P/D fleets are fixed-size.** `Fleet` refuses an autoscaler or flow control together with a prefill pool; the
  KV transfer is priced for the blocks the decode replica lacks at hand-off.
- **KV tiers are exclusive** (demote on evict); real offload keeps an inclusive copy in the lower tier.
- **The HPA sees a starting replica as Pending for its whole cold start**; a real pod past its image pull is Running
  with no sample (the same on a scale-up, counted at the target on a scale-down). Flow control checks a request's
  TTL when it reaches the head of the router queue.
- **Adapter loads cost an illustrative 0.2 s** that stalls the step (`EngineProfile.lora_load_s`).

The engine profiles `L4_8B` and `H100_8B` are derived in `replica.engine_profile()` from spec-sheet numbers (verify
them against layer 01's device catalogue). The simulator stops where real systems start: no network, no Envoy, no
Kubernetes objects, no GPU. The next step is [`../inference-gateway-lab`](../inference-gateway-lab/): the same
decisions as a real async router in front of OpenAI-compatible backends, HPA recommendations from scraped metrics,
the llm-d Router on kind and GKE Inference Gateway on GCP.

## The library (read in this order)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `fleetsim/workload.py` | ~220 | a request is its lengths plus the chain hashes of its KV blocks; Poisson and bursty arrivals; chat, RAG and closed-loop agent sessions with growing histories, shared system prompts, LoRA ids and priorities |
| `fleetsim/replica.py` | ~315 | a replica is a stateful cache: token-budget continuous batching with chunked prefill, a paged block pool with a hash prefix cache (LRU, tail blocks first), preemption by recompute, adapter slots, a roofline step time |
| `fleetsim/routers.py` | ~340 | round-robin, least-outstanding, power-of-two, prefix hashing, consistent hashing with bounded loads, and the llm-d EPP shape (filters → weighted `prefix-cache` / `queue` / `kv-cache-utilization` / `token-load` scorers → max-score picker; the `prefix-cache-affinity-filter` with its TTFT gate and break counter), approximate vs precise prefix indexes |
| `fleetsim/sim.py` | ~245 | the event loop: arrivals, engine steps, stale metric scrapes, session follow-ups, flow control (a router queue with per-endpoint caps, priorities, TTL and queue-bound shedding), P/D hand-offs, autoscaler ticks, cold starts |
| `fleetsim/metrics.py` | ~75 | TTFT, ITL, TPOT percentiles, goodput, hit rate, imbalance, shed requests, adapter loads, GPU-hours — and text tables and sparklines |
| `fleetsim/autoscale.py` | ~205 | the HPA exactly as `kube-controller-manager` computes it: milli-unit ratio, tolerance, Pending and missing pods, several metrics (the largest wins), stabilization windows, Pods/Percent policies, External metrics from zero; request, KV and prefill-backlog signals; cold-start anatomy |
| `fleetsim/disagg.py` | ~75 | KV transfer time, decode batch at an ITL SLO, analytic P:D sizing, and a search over every xPyD split |
| `fleetsim/kvtier.py` | ~130 | fetch vs recompute break-even, agent-session working sets, exclusive LRU tiers with demotion, a session replay over local and shared tiers |

Each module opens with a docstring stating the one idea it teaches. **Size:** about 1,600 lines, of which about
1,100 are code (the rest docstrings and comments) — over the 500–1,000 lines the curriculum aims for in a core. The
overrun is deliberate: this layer's plan asks for routing, flow control, the HPA with its policies, P/D and KV tiers
in one simulator, and each piece is kept faithful enough to the upstream component that its numbers mean something.
Read `workload.py`, `replica.py`, `routers.py` and `sim.py` first; the other modules are independent of each other.

MIT licensed.
