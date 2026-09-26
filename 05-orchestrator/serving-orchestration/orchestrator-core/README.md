# orchestrator-core

A fleet of LLM engines you can run on a laptop: **`fleetsim`**, a pure-Python discrete-event simulator of replicas
with prefix caches, the routers in front of them, the Kubernetes HPA that sizes them, prefill/decode disaggregation
and KV-cache tiers — plus five fill-in notebooks. Standard library only, deterministic under a seed, every run a
few seconds on a CPU.

Every number it prints is **simulated**: an engine model built from spec-sheet arithmetic (an 8B model on an L4 by
default), not a measurement. It is for building intuition and checking designs — which signal, which router, which
split — before you spend GPU hours. The concepts are in [`../PRIMER.md`](../PRIMER.md); the step up to real servers
is [`../inference-gateway-lab`](../inference-gateway-lab/).

## Quick start

```bash
cd orchestrator-core
python3 -m pip install -r requirements.txt   # only to run the notebooks/tests
python3 -m pytest -q                          # 43 tests, ~2 s
python3 -m jupyterlab notebooks               # do the exercises
```

```python
from fleetsim import L4_8B, Fleet, PowerOfTwo, agentic, epp, table

for router in (PowerOfTwo(seed=1), epp((3, 2, 2))):              # load only vs prefix + queue + KV scoring
    reqs = agentic(0.6, 240, seed=3, groups=6, system=2000, tool=500, output=100, turns=(4, 10), think=(1, 4))
    s = Fleet(L4_8B, 4, router).run(reqs).summary(ttft_slo=1.0)
    print(router.name, round(s["hit_rate"], 2), round(s["ttft_p95"], 2))   # simulated
```

## The library (read in this order)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `fleetsim/workload.py` | ~220 | a request is its lengths plus the chain hashes of its KV blocks; Poisson and bursty arrivals; chat, RAG and closed-loop agent sessions with growing histories, shared system prompts and LoRA ids |
| `fleetsim/replica.py` | ~300 | a replica is a stateful cache: token-budget continuous batching with chunked prefill, a paged block pool with a hash prefix cache (LRU, tail blocks first), preemption by recompute, a roofline step time |
| `fleetsim/routers.py` | ~330 | round-robin, least-outstanding, power-of-two, prefix hashing, consistent hashing with bounded loads, and the llm-d EPP shape (filters → weighted `prefix-cache` / `queue` / `kv-cache-utilization` / `token-load` scorers → max-score picker), approximate vs precise prefix indexes |
| `fleetsim/sim.py` | ~180 | the event loop: arrivals, engine steps, stale metric scrapes, session follow-ups, P/D hand-offs, autoscaler ticks, cold starts |
| `fleetsim/metrics.py` | ~70 | TTFT, ITL, TPOT percentiles, goodput, hit rate, imbalance, GPU-hours — and text tables and sparklines |
| `fleetsim/autoscale.py` | ~175 | the HPA exactly as `kube-controller-manager` computes it: milli-unit ratio, tolerance, not-ready pods, stabilization windows, Pods/Percent policies, External metrics from zero; cold-start anatomy |
| `fleetsim/disagg.py` | ~75 | KV transfer time, decode batch at an ITL SLO, analytic P:D sizing, and a search over every xPyD split |
| `fleetsim/kvtier.py` | ~125 | fetch vs recompute break-even, agent-session working sets, exclusive LRU tiers with demotion, a session replay over local and shared tiers |

Each module opens with a docstring stating the one idea it teaches. The engine profiles `L4_8B` and `H100_8B` are
derived in `replica.engine_profile()` from spec-sheet numbers (verify them against layer 01's device catalogue).

## The notebooks

Each has a **Tier** line, "The one-minute version", worked examples, exercises with `# YOUR CODE HERE` and a check
cell that prints ✅, and "In a design review" drills. Solutions are in `solutions/`.

1. **`01_why_llm_load_balancing_is_different`** — what a request costs in prefill, KV and residency; five load-only routers on a heterogeneous fleet; power of two choices; Little's law; herding on stale metrics.
2. **`02_cache_aware_routing_and_the_load_tradeoff`** — chain hashing; the upstream scorer formulas; the locality-vs-load spectrum from prefix hashing to the EPP; the prefix-weight knob; bounded loads; a hot prefix; how stale an approximate index gets.
3. **`03_autoscaling_on_the_right_signal`** — GPU util vs running vs KV vs queue under load; the HPA rule, stabilization and policies; five signals on a traffic step; cold start and headroom; picking a target; scale-to-zero economics.
4. **`04_prefill_decode_disaggregation`** — the prefill stall; KV transfer cost on L4 vs H100; every xPyD split of eight GPUs vs the analytic ratio; smaller chunks first; conditional disaggregation.
5. **`05_kv_cache_tiers_and_agent_sessions`** — agent working sets; why routing cannot fix a capacity problem; fetch vs recompute; a two-tier LRU; sticky vs random vs shared tiers; sizing DRAM from a replay.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all of it plus the tests. On Colab, the first cell of each notebook clones the repository and
installs this package (see the repository's `COLAB.md`).

## When you outgrow this

The simulator stops where real systems start: no network, no Envoy, no Kubernetes objects, no GPU. The
[`inference-gateway-lab`](../inference-gateway-lab/) runs the same decisions as a real async router in front of
OpenAI-compatible backends, turns scraped metrics into HPA recommendations and manifests, and deploys the llm-d
Router locally and GKE Inference Gateway on GCP. MIT licensed.
