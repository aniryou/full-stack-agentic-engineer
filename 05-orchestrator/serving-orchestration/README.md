# serving-orchestration — route, scale and split a fleet of LLM engines

After this topic you can say, with numbers, which replica an LLM request should go to, how many replicas a fleet
needs and which signal should decide it, and when to split prefill from decode or move KV cache out of HBM — and
then run those decisions as a real router in front of real or emulated engines.

## Start here

1. Read [PRIMER.md](PRIMER.md): "The one-minute version", then §1 — why a replica is a cache and round-robin fails.
2. `cd orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 66 tests in about
   30 s; then open [`01_why_llm_load_balancing_is_different`](orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb).
3. Put a real router in front of three emulated engines, still on a laptop CPU:
   [`inference-gateway-lab/notebooks/01_router_in_process.ipynb`](inference-gateway-lab/notebooks/01_router_in_process.ipynb).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* "T0 + Docker" is a laptop with Docker, still free.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the three decisions above the engine — which replica, how many, how to split the work — with worked numbers: §1 why a layer above the engine · §2 routing signals and algorithms · §3 flow control and priorities · §4 autoscaling · §5 prefill/decode disaggregation · §6 KV cache beyond HBM · §7 multi-model, multi-LoRA and model routing · §8 large MoE topologies · §9 the Kubernetes-native stack · §10 where to run it; then a design-review walkthrough and six drills | read alongside the core | — |
| [`orchestrator-core/`](orchestrator-core/) | predict what a routing, autoscaling or disaggregation choice does to TTFT, hit rate and GPU-hours, in `fleetsim` — a pure-Python discrete-event simulator of a fleet (workloads, engine replicas with prefix caches, routers, the Kubernetes HPA algorithm, P/D, KV tiers) — through five fill-in notebooks | ~9 h with the primer | T0 |
| [`inference-gateway-lab/`](inference-gateway-lab/) | run the llm-d endpoint picker's decision logic as a real async router (`igwlab`) in front of OpenAI-compatible backends (emulated or vLLM), turn scraped metrics into HPA decisions, and deploy the llm-d Router on kind and GKE Inference Gateway on GCP | ~6 h (01–04) + 2 h (05) | T0 → T3 |

Every number the core prints is **simulated** — a model of an engine built from spec-sheet arithmetic — and labelled
so. The lab measures real (or explicitly emulated) servers.

### Work it in this order

Read the primer section, then do the matching core notebook, then the lab notebook where there is one. Times are
rough and cover the whole row; they come from the repo's curriculum ([`CURRICULUM.md`](../../CURRICULUM.md), modules
05.1–05.6).

| Step | Primer | Core notebook (T0) | Lab notebook | Time |
|---|---|---|---|---|
| 1 | §1, §2.1–2.2, §3 | [`01_why_llm_load_balancing_is_different`](orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb) — request cost spread, round-robin vs least-outstanding vs power-of-two, herding on stale metrics, flow control: a router queue with priorities and a per-endpoint cap sized by Little's law | [`01_router_in_process`](inference-gateway-lab/notebooks/01_router_in_process.ipynb) (T0) | 3 h |
| 2 | §2.3–2.6, §7 | [`02_cache_aware_routing_and_the_load_tradeoff`](orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) — chain hashes, EPP scorers, prefix hashing vs bounded loads vs weighted scoring vs the affinity filter's TTFT gate, hot prefixes, approximate vs precise indexes, LoRA adapter affinity | [`02_scorer_weights_and_hot_prefixes`](inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) (T0) | 3 h |
| 3 | §4 | [`03_autoscaling_on_the_right_signal`](orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb) — the HPA rule, tolerance, stabilization, policies, Pending pods; GPU util vs queue vs KV vs in-flight requests vs prefill backlog, on chat and on a chat + RAG mix; cold start and scale-to-zero with the node | [`03_autoscaling_recommender`](inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) (T0; manifests for T3) | 3 h |
| 4 | §5 | [`04_prefill_decode_disaggregation`](orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb) — the prefill stall, KV transfer cost, P:D sizing and the split search, chunk budget first, conditional disaggregation | — | 2 h |
| 5 | §6 | [`05_kv_cache_tiers_and_agent_sessions`](orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) — agent working sets, fetch vs recompute, tiered LRU, sticky vs shared tiers | — | 2 h |
| 6 | §7–10 | — (§7's adapter routing is in `02`) | [`04_local_stack_with_llm_d`](inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb) (T0 + Docker or kind), [`05_gke_inference_gateway`](inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) (T3; offline it plans and inspects) | 2 h (+2 h on GCP) |

Finish with the primer's "In a design review": a two-minute walkthrough and six drills with answers.

## Run it

```bash
cd orchestrator-core
python3 -m pip install -r requirements.txt     # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                           # 66 tests, ~30 s, pinned to hand-computed numbers
python3 -m jupyterlab notebooks                # do the exercises; finished versions are in solutions/

cd ../inference-gateway-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                           # 84 tests, ~30 s, offline
python3 -m jupyterlab notebooks
```

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | What runs | Where | Cost |
|---|---|---|---|
| **T0** | all five core notebooks and their tests; lab notebooks 01–03 against fake backends; lab 04–05 in offline mode | laptop, Colab CPU, CI — no GPU, no network | $0 |
| **T0 + Docker** | the lab's local stack: compose (simulated backends, router, Prometheus) or kind with the llm-d Router in standalone mode and `llm-d-inference-sim` | a laptop with Docker | $0 |
| **T1/T2** | the lab's router and autoscaler in front of real vLLM on one or two GPUs | Colab/Kaggle T4, or any rented GPU box | free on Colab/Kaggle; a rented 24 GB GPU ~$0.3–0.7/h (verify) |
| **T3** | GKE Inference Gateway: InferencePool, the endpoint picker, InferenceObjective priorities, an HPA on Managed Prometheus metrics, an L4 Spot pool that scales from zero | GCP, via the lab's Terraform and manifests | ~$0.16/h with the GPU pool at 0, ~$0.44/h with one L4 Spot node (assumed prices, verify) |

Prices and where to get GPUs: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

This topic sits between one engine and the gateway in front of the fleet: it takes the engine as given and hands
admission and cost to the gateway.

| | Read | For |
|---|---|---|
| before | [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) | prefill vs decode, TTFT and TPOT |
| before | the [KV cache](../../04-inference-engine/kv-cache/kv-cache-primer.md) and [paged attention](../../04-inference-engine/paged-attention/paged-attention-primer.md) primers | the blocks a replica caches and reuses |
| before | [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md); the [GPU deployment primer](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §2 and §8 | fabrics and cold start; prefill vs decode and disaggregation in one page |
| beside | layer 04's `serving-engine` topic ([`04-inference-engine/serving-engine/`](../../04-inference-engine/serving-engine/)) and layer 03's `gpu-scheduling` topic ([`03-kubernetes-gpu/gpu-scheduling/`](../../03-kubernetes-gpu/gpu-scheduling/)) | batching, chunked prefill and prefix caching inside one engine; pods, GPUs and startup latency |
| after | [`06-gateway/scaling-admission-cost/agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) | admission control, rate limits and cost per conversation |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/README.md) | the agent sessions whose shared prompts and growing histories shape every routing and caching decision |

## Caveats

- **Simulated vs measured.** The core's numbers come from an engine model, not a GPU; the lab's T0 latencies are real
  HTTP against emulated engine timing. Use both to compare designs, never as GPU numbers — and expect router rankings
  to change with the engine and the workload (PRIMER §2.5).
- **Untested on real infrastructure here.** The lab's Docker, kind, GPU and GKE paths were validated by construction
  (`bash -n`, `DRY_RUN=1`, Terraform `validate`, Kubernetes schema checks), not executed.
- **Dated facts.** Product versions (llm-d Router v0.10.0, Gateway API Inference Extension v1.6.2, vLLM 0.30.0),
  GKE details and prices are as of September 2026 and marked `(verify)`; the primer's
  [Verify list](PRIMER.md#verify-list) collects them.
