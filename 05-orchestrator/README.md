# 05 · Orchestrator

Turn many inference-engine replicas into one serving system: after this layer you can decide which replica each
request goes to, how many replicas run and on which signal, and whether prefill and decode share a GPU — and defend
each choice with numbers in a design review.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the fleet: it turns many engine replicas (04) on the cluster (03) into one service, below the
gateway (06) that decides whether a request runs at all.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`serving-orchestration/`](serving-orchestration/README.md) | route on prefix affinity and load (the llm-d endpoint picker), hold bursts with flow control and priorities, autoscale on work in flight instead of GPU utilisation, size prefill/decode disaggregation, and keep agent sessions' KV cache in tiers beyond HBM — with a primer, a pure-Python fleet simulator and a real router you can deploy on kind and GKE | ~9 h primer + core; ~8 h lab | T0 → T3 |

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

## Start here

1. Read [`serving-orchestration/PRIMER.md`](serving-orchestration/PRIMER.md): "The one-minute version" and §1.
2. `cd serving-orchestration/orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`
   — 58 tests in about 5 s; then open
   [`01_why_llm_load_balancing_is_different`](serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb).
3. Follow the step table in the topic [`README.md`](serving-orchestration/README.md): each primer section pairs
   with a core notebook and, where there is one, a lab notebook.

## What is inside `serving-orchestration/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](serving-orchestration/PRIMER.md) | ten sections: why a layer above the engine, routing signals and algorithms (power of two choices, bounded-load hashing, the endpoint picker's filters → scorers → picker), flow control and priorities, autoscaling (the HPA algorithm exactly, which signal, cold start, scale to zero), prefill/decode disaggregation, KV cache beyond HBM, multi-model and LoRA routing, wide-EP, the Kubernetes-native stack (InferencePool, llm-d Router, GKE Inference Gateway, Dynamo), where to run it; a design-review walkthrough, drills, glossary, sources and a dated verify list | read |
| [`orchestrator-core/`](serving-orchestration/orchestrator-core/) | the minimal implementation, package `fleetsim`: a discrete-event simulator of workloads, replicas with prefix caches, routers, flow control, the HPA, P/D and KV tiers, standard library only; five fill-in notebooks that **predict** | T0 |
| [`inference-gateway-lab/`](serving-orchestration/inference-gateway-lab/) | the detailed lab, package `igwlab`: an async router that re-implements the endpoint picker's decision logic in front of emulated or real vLLM backends, the HPA recommender ported from kube-controller-manager, a shared-prefix agent benchmark; deploys for docker compose, kind with the llm-d Router, any GPU box and GKE Inference Gateway (Terraform); five notebooks that **run** it | T0 → T3 |

## Run it

```bash
cd serving-orchestration/orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../inference-gateway-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
```

Then `python3 -m jupyterlab notebooks` in either directory, or the Colab links below. Tiers and costs are in the
topic README's [Run it](serving-orchestration/README.md#run-it); the deploy paths (compose, kind, any GPU box, GKE)
are in the lab's [Deploy paths](serving-orchestration/inference-gateway-lab/README.md#deploy-paths).

## How it fits

**Needed first:** [`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (prefill vs
decode, TTFT and TPOT), layer 04's [KV cache](../04-inference-engine/kv-cache/kv-cache-primer.md) and
[paged attention](../04-inference-engine/paged-attention/paged-attention-primer.md) primers, and layer 01's
[`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (fabrics, cold start); layer
04's [prefix caching](../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching) makes §2's routing
signals concrete. In the [curriculum's spiral](../CURRICULUM.md#31-why-this-order) this layer follows 03, so the
HPA and LeaderWorkerSet already mean something. **Leads to** layer 06, where [admission control, rate limits and cost](../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
decide whether a request runs before the router decides where, and to the agent workloads in
[`07-application-agent-framework`](../07-application-agent-framework/README.md), whose multi-turn sessions shape
every routing and caching decision here.

## Caveats

- Every number the simulator prints is **simulated** (an engine model from spec-sheet arithmetic), and the lab's T0
  latencies are real HTTP against emulated engine timing: both compare designs, neither is a GPU measurement.
- Product versions, GKE details and prices are as of September 2026 and marked `(verify)`; the primer's
  [Verify list](serving-orchestration/PRIMER.md#verify-list) collects them.
- Not covered yet: SLO planners (thin), fleet observability, multi-cluster and multi-region serving, request
  cancellation and migration, cost-aware routing across GPU types — see
  [`CURRICULUM.md` §2](../CURRICULUM.md#2-what-each-layer-has-and-what-it-does-not-cover-yet).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`serving-orchestration/inference-gateway-lab/`** — [01_router_in_process](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/01_router_in_process.ipynb) · [02_scorer_weights_and_hot_prefixes](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) · [03_autoscaling_recommender](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) · [04_local_stack_with_llm_d](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb) · [05_gke_inference_gateway](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/01_router_in_process.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/02_scorer_weights_and_hot_prefixes.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/03_autoscaling_recommender.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/04_local_stack_with_llm_d.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/05_gke_inference_gateway.ipynb)
- **`serving-orchestration/orchestrator-core/`** — [01_why_llm_load_balancing_is_different](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb) · [02_cache_aware_routing_and_the_load_tradeoff](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) · [03_autoscaling_on_the_right_signal](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb) · [04_prefill_decode_disaggregation](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb) · [05_kv_cache_tiers_and_agent_sessions](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/01_why_llm_load_balancing_is_different.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/02_cache_aware_routing_and_the_load_tradeoff.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/03_autoscaling_on_the_right_signal.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/04_prefill_decode_disaggregation.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/05_kv_cache_tiers_and_agent_sessions.ipynb)
<!-- colab-links:end -->
