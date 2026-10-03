# orchestrator-core — predict what a routing or scaling decision does to a fleet, on a laptop

This is a fleet of LLM engines that you can run on a CPU. **`fleetsim`** is a discrete-event simulator in Python
only. It simulates replicas with prefix caches, the routers in front of them and the Kubernetes HPA that sizes them.
It also simulates prefill/decode disaggregation and KV-cache tiers. With it, you can say which router, which
autoscaling signal and which split a workload needs, before you spend GPU hours.

## Start here

1. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. It runs 66 tests in about 30 s, offline.
2. Run the [snippet in "Run it"](#run-it). It compares power of two choices with the llm-d-style endpoint picker on
   agent sessions. In less than a second, it prints the hit rate and the TTFT p95 (simulated).
3. Open [`notebooks/01_why_llm_load_balancing_is_different.ipynb`](notebooks/01_why_llm_load_balancing_is_different.ipynb).
   Read [PRIMER §1](../PRIMER.md#1-why-a-layer-above-the-engine) with it.

## What you get

*Tier: T0 = laptop or Colab CPU, free.* All of this core is T0. It uses the standard library only, it is
deterministic under a seed, and each simulation takes a few seconds on a CPU.

| Notebook | You will be able to… | Time | Tier |
|---|---|---|---|
| [`01_why_llm_load_balancing_is_different`](notebooks/01_why_llm_load_balancing_is_different.ipynb) | Say what a request costs in prefill, KV and residency. Rank five load-only routers on a heterogeneous fleet. Explain power of two choices, and herding on stale metrics. Hold a burst in a router queue with priorities and a per-endpoint cap that Little's law sizes | ~1.5 h | T0 |
| [`02_cache_aware_routing_and_the_load_tradeoff`](notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) | Calculate chain hashes and the upstream scorer formulas. Put prefix hashing, bounded loads, the weighted endpoint picker and the TTFT gate of the affinity filter on the spectrum from locality to load. Adjust the prefix weight. Keep the fleet up under a hot prefix. Find how stale an approximate index becomes. Route LoRA adapters | ~1.5 h | T0 |
| [`03_autoscaling_on_the_right_signal`](notebooks/03_autoscaling_on_the_right_signal.ipynb) | Compare GPU util, running, KV and queue under load, and five signals on a traffic step. Apply the HPA rule with stabilization, policies and Pending pods. Select targets from a load test. Make a budget for cold start and headroom. Explain why a request-count target does not transfer to a chat + RAG mix, and why a prefill-backlog + KV signal does. Decide on scale-to-zero when the node also goes | ~1.5 h | T0 |
| [`04_prefill_decode_disaggregation`](notebooks/04_prefill_decode_disaggregation.ipynb) | Measure the prefill stall. Calculate the cost of the KV transfer on L4 against H100. Search every xPyD split of eight GPUs, and compare the best split with the analytic ratio. Find the limit of the decode batch from KV and ITL. Try smaller chunks first. Use conditional disaggregation | ~2 h | T0 |
| [`05_kv_cache_tiers_and_agent_sessions`](notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) | Find the size of the KV working set of an agent fleet. Show why routing cannot repair a capacity problem. Decide between fetch and recompute. Run a two-tier LRU. Predict the results of sticky, random and shared tiers. Find the DRAM size from a replay | ~2 h | T0 |

The times are approximate and include the related [PRIMER](../PRIMER.md) section. Their sum is approximately the 9
hours that the curriculum of the repository ([`CURRICULUM.md`](../../../CURRICULUM.md)) gives to the primer and
this core. Each notebook has these parts:

- A **Tier** line.
- "The one-minute version".
- Worked examples.
- Exercises with `# YOUR CODE HERE`, each with a check cell after it that prints ✅.
- "In a design review" drills.

The worked answers are in [`solutions/`](solutions/).

## Run it

```bash
cd orchestrator-core
python3 -m pip install -r requirements.txt   # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                          # 66 tests, ~30 s
python3 -m jupyterlab notebooks               # do the exercises
```

```python
from fleetsim import L4_8B, Fleet, PowerOfTwo, agentic, epp, table

for router in (PowerOfTwo(seed=1), epp((3, 2, 2))):              # load only vs prefix + queue + KV scoring
    reqs = agentic(0.6, 240, seed=3, groups=6, system=2000, tool=500, output=100, turns=(4, 10), think=(1, 4))
    s = Fleet(L4_8B, 4, router).run(reqs).summary(ttft_slo=1.0)
    print(router.name, round(s["hit_rate"], 2), round(s["ttft_p95"], 2))   # simulated
```

On Colab, the first cell of each notebook clones the repository and installs this package (see the
[`COLAB.md`](../../../COLAB.md) of the repository).

The builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources, then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all of it, and the tests too.

## What the model leaves out

Every number that `fleetsim` prints is **simulated**. It comes from an engine model that uses spec-sheet arithmetic
(an 8B model on an L4 by default), and it is not a measurement. Use the model to get intuition and to examine
designs. Read every simulated number with the items in this list in mind. The notebooks and the primer also state
each item where it is important.

- **Prefill compute is linear in tokens.** The model omits the attention FLOPs (about 4 × layers × d_model ×
  context per token). They add +10 % for a 6,000-token prompt on the 8B model, +16 % at 10,000 and +50 % at
  30,000. Thus long-context prefill, P/D transfer budgets (notebook 04) and recompute costs (notebook 05) are
  optimistic by those quantities.
- **No per-chunk efficiency loss.** A 256-token prefill chunk has the same cost per token as a 2,048-token chunk.
  Thus the chunk-budget sweep in notebook 04 is the optimistic end.
- **Metric staleness is a maximum age.** The simulator reads a scraped snapshot again on the first look after it
  expires. The simulator has no scrape jitter and no per-replica phase.
- **P/D fleets have a constant size.** `Fleet` refuses an autoscaler or flow control together with a prefill pool.
  The cost of the KV transfer is for the blocks that the decode replica does not have at hand-off.
- **KV tiers are exclusive** (the simulator demotes a block when it evicts it). Real offload keeps an inclusive copy in the lower tier.
- **The HPA sees a replica as Pending for all of its cold start.** A real pod after its image pull is
  Running with no sample. On a scale-up, the real HPA treats such a pod in the same way as a Pending pod. On a
  scale-down, the real HPA counts such a pod at the target. Flow
  control examines the TTL of a request when the request gets to the head of the router queue.
- **Adapter loads cost an illustrative 0.2 s** that stalls the step (`EngineProfile.lora_load_s`).

`replica.engine_profile()` derives the engine profiles `L4_8B` and `H100_8B` from spec-sheet numbers (verify them
against the device catalogue of layer 01). The simulator stops where real systems start. It has no network, no
Envoy, no Kubernetes objects and no GPU. The next step is [`../inference-gateway-lab`](../inference-gateway-lab/).
The lab makes the same decisions in a real async router in front of OpenAI-compatible backends. It also has HPA
recommendations from scraped metrics, the llm-d Router on kind and GKE Inference Gateway on GCP.

## The library (read in this order)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `fleetsim/workload.py` | ~220 | A request is its lengths and the chain hashes of its KV blocks. Poisson and bursty arrivals. Chat, RAG and closed-loop agent sessions with histories that grow, shared system prompts, LoRA ids and priorities |
| `fleetsim/replica.py` | ~315 | A replica is a stateful cache. It has token-budget continuous batching with chunked prefill. It has a paged block pool with a hash prefix cache (LRU, tail blocks first). It also has preemption by recompute, adapter slots and a roofline step time |
| `fleetsim/routers.py` | ~340 | Round-robin, least-outstanding, power-of-two, prefix hashing, consistent hashing with bounded loads. The shape of the llm-d EPP: filters, then weighted `prefix-cache` / `queue` / `kv-cache-utilization` / `token-load` scorers, then a max-score picker. The `prefix-cache-affinity-filter` with its TTFT gate and break counter. Approximate and precise prefix indexes |
| `fleetsim/sim.py` | ~245 | The event loop. It has arrivals, engine steps, stale metric scrapes and session follow-ups. It has flow control (a router queue with per-endpoint caps, priorities, TTL and queue-bound shedding). It also has P/D hand-offs, autoscaler ticks and cold starts |
| `fleetsim/metrics.py` | ~75 | TTFT, ITL and TPOT percentiles, goodput, hit rate, imbalance, shed requests, adapter loads, GPU-hours. Also text tables and sparklines |
| `fleetsim/autoscale.py` | ~205 | The HPA exactly as `kube-controller-manager` calculates it. This includes the milli-unit ratio, tolerance, Pending and missing pods, and more than one metric (the largest wins). It also includes stabilization windows, Pods/Percent policies and External metrics from zero. Request, KV and prefill-backlog signals. The anatomy of a cold start |
| `fleetsim/disagg.py` | ~75 | KV transfer time, the decode batch at an ITL SLO, an analytic calculation of the P:D size, and a search over every xPyD split |
| `fleetsim/kvtier.py` | ~130 | The break-even of fetch against recompute, agent-session working sets, exclusive LRU tiers with demotion, a session replay over local and shared tiers |

Each module starts with a docstring that states the one idea that it teaches. **Size:** about 1,600 lines. About
1,100 of them are code, and the rest are docstrings and comments. The size of 1,600 lines is more than the 500–1,000 lines that the
curriculum aims for in a core.

The overrun is intentional, because this layer's plan asks for routing, flow control,
the HPA with its policies, P/D and KV tiers in one simulator. The overrun has a second cause: each piece stays sufficiently faithful to the
upstream component for its numbers to mean something.

Read `workload.py`, `replica.py`, `routers.py` and `sim.py` first. The other modules are independent of each other.

MIT licensed.
