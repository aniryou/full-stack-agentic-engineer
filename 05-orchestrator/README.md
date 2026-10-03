# 05 · Orchestrator

Turn many inference-engine replicas into one serving system. After this layer, you can decide these things:

- Which replica each request goes to.
- How many replicas run, and on which signal.
- If prefill and decode share a GPU or not.

You can also defend each choice with numbers in a design review.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, memory, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, model routing, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the fleet. It changes many engine replicas (04) on the cluster (03) into one service. It is below the
gateway (06), which decides if a request runs at all.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`serving-orchestration/`](serving-orchestration/README.md) | route on prefix affinity and load (the llm-d endpoint picker). Hold bursts with flow control and priorities. Autoscale on work in flight, not on GPU utilisation. Calculate the size of a prefill/decode disaggregation. Keep the KV cache of agent sessions in tiers beyond HBM. The topic has a primer, a pure-Python fleet simulator and a real router that you can deploy on kind and GKE. | ~9 h primer + core, ~8 h lab | T0 to T3 |

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab or Kaggle T4, or a rented card). T2
is a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.*

## Start here

1. Read "The one-minute version" and §1 of [`serving-orchestration/PRIMER.md`](serving-orchestration/PRIMER.md).
2. Run `cd serving-orchestration/orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`.
   It runs 66 tests in about 30 s. Then open
   [`01_why_llm_load_balancing_is_different`](serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb).
3. Do the steps in the step table of the topic [`README.md`](serving-orchestration/README.md). Each primer section
   goes with a core notebook, and with a lab notebook if one exists for that section.

## What is inside `serving-orchestration/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](serving-orchestration/PRIMER.md) | The ten sections of the primer tell why a layer above the engine is necessary. They give routing signals and algorithms (power of two choices, bounded-load hashing, and the filters, then the scorers, then the picker of the endpoint picker). They explain flow control and priorities, and autoscaling (the exact HPA algorithm, which signal, cold start, scale to zero). They also explain prefill/decode disaggregation, KV cache beyond HBM, multi-model and LoRA routing, and wide-EP. Then they give the Kubernetes-native stack (InferencePool, llm-d Router, GKE Inference Gateway, Dynamo) and where to run it. The primer also has a design-review walkthrough, drills, a glossary, sources and a dated verify list. | read |
| [`orchestrator-core/`](serving-orchestration/orchestrator-core/) | This is the minimal implementation, package `fleetsim`. It is a discrete-event simulator of workloads, replicas with prefix caches, routers, flow control, the HPA, P/D and KV tiers. It uses the standard library only. It has five fill-in notebooks that **predict**. | T0 |
| [`inference-gateway-lab/`](serving-orchestration/inference-gateway-lab/) | The detailed lab, package `igwlab`, has an async router in front of emulated or real vLLM backends. The router re-implements the decision logic of the endpoint picker. The lab also has the HPA recommender, ported from kube-controller-manager, and a shared-prefix agent benchmark. It has deploy paths for docker compose, kind with the llm-d Router, any GPU box and GKE Inference Gateway (Terraform). It has five notebooks that **run** these parts. | T0 to T3 |

## Run it

```bash
cd serving-orchestration/orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../inference-gateway-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
```

Then run `python3 -m jupyterlab notebooks` in either directory, or use the Colab links in "Run in Colab". The
[Run it](serving-orchestration/README.md#run-it) section of the topic README gives the tiers and the costs. The
[Deploy paths](serving-orchestration/inference-gateway-lab/README.md#deploy-paths) section of the lab README gives
the deploy paths (compose, kind, any GPU box, GKE).

## How it fits

**Needed first:**

- [`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (prefill compared
  with decode, TTFT and TPOT).
- The [KV cache](../04-inference-engine/kv-cache/kv-cache-primer.md) and
  [paged attention](../04-inference-engine/paged-attention/paged-attention-primer.md) primers of layer 04.
- [`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) of layer 01 (fabrics, cold
  start).

The [prefix caching](../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching) section of layer 04 makes the
routing signals of §2 concrete. In the [curriculum's spiral](../CURRICULUM.md#31-why-this-order), this layer comes
after 03. Thus the HPA and LeaderWorkerSet already mean something to you.

**Leads to** layer 06. There, [admission control, rate limits and cost](../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
decide if a request runs, before the router decides where it runs. This layer also leads to the agent workloads in
[`07-application-agent-framework`](../07-application-agent-framework/README.md). Their multi-turn sessions shape
every routing and caching decision in this layer.

## Caveats

- Every number that the simulator prints is **simulated** (an engine model from spec-sheet arithmetic). The T0
  latencies of the lab come from real HTTP requests to engines with emulated timing. Both compare designs. Neither
  is a GPU measurement.
- Product versions, GKE details and prices are as of September 2026, and they have the tag `(verify)`. The
  [Verify list](serving-orchestration/PRIMER.md#verify-list) of the primer collects them.
- This layer does not cover these subjects yet: SLO planners (thin), fleet observability, and multi-cluster and
  multi-region serving. It also does not cover request cancellation and migration, or cost-aware routing across GPU
  types. See
  [`CURRICULUM.md` §2](../CURRICULUM.md#2-what-each-layer-has-and-what-it-does-not-cover-yet).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`serving-orchestration/inference-gateway-lab/`** — [01_router_in_process](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/01_router_in_process.ipynb) · [02_scorer_weights_and_hot_prefixes](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) · [03_autoscaling_recommender](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) · [04_local_stack_with_llm_d](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb) · [05_gke_inference_gateway](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/01_router_in_process.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/02_scorer_weights_and_hot_prefixes.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/03_autoscaling_recommender.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/04_local_stack_with_llm_d.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/inference-gateway-lab/solutions/05_gke_inference_gateway.ipynb)
- **`serving-orchestration/orchestrator-core/`** — [01_why_llm_load_balancing_is_different](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb) · [02_cache_aware_routing_and_the_load_tradeoff](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) · [03_autoscaling_on_the_right_signal](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb) · [04_prefill_decode_disaggregation](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb) · [05_kv_cache_tiers_and_agent_sessions](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/01_why_llm_load_balancing_is_different.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/02_cache_aware_routing_and_the_load_tradeoff.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/03_autoscaling_on_the_right_signal.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/04_prefill_decode_disaggregation.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/05-orchestrator/serving-orchestration/orchestrator-core/solutions/05_kv_cache_tiers_and_agent_sessions.ipynb)
<!-- colab-links:end -->
