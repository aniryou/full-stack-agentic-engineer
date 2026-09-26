# serving-orchestration — orchestrating a fleet of engines

Routing, autoscaling, prefill/decode disaggregation and KV-cache tiers: the decisions that sit between one inference
engine (layer 04) and the gateway in front of the fleet (layer 06). Three artifacts, used in this order:

| | What | Tier |
|---|---|---|
| [`PRIMER.md`](PRIMER.md) | the concepts, with worked numbers: §1 why a layer above the engine · §2 routing signals and algorithms · §3 flow control and priorities · §4 autoscaling · §5 prefill/decode disaggregation · §6 KV cache beyond HBM · §7 multi-model, multi-LoRA and model routing · §8 large MoE topologies · §9 the Kubernetes-native stack · §10 where to run it | read |
| [`orchestrator-core/`](orchestrator-core/) | `fleetsim`, a pure-Python discrete-event simulator of a fleet (workloads, engine replicas with prefix caches, routers, the Kubernetes HPA algorithm, P/D, KV tiers) and five fill-in notebooks | T0 |
| [`inference-gateway-lab/`](inference-gateway-lab/) | `igwlab`: a real async router in front of OpenAI-compatible backends (fake or vLLM), an HPA recommender, a local stack with the llm-d Router, and GKE Inference Gateway | T0 → T3 |

Every number the core prints is **simulated** — a model of an engine built from spec-sheet arithmetic — and labelled
so. The lab measures real (or explicitly fake) servers.

## How to work it

Read the primer section, then do the matching core notebook, then the lab notebook where there is one.

| Step | Primer | Core notebook (T0) | Lab notebook |
|---|---|---|---|
| 1 | §1, §2.1–2.2, §3 | `01_why_llm_load_balancing_is_different` — request cost spread, round-robin vs least-outstanding vs power-of-two, herding on stale metrics, flow control: a router queue with priorities and a per-endpoint cap sized by Little's law | `01_router_in_process` (T0) |
| 2 | §2.3–2.6, §7 | `02_cache_aware_routing_and_the_load_tradeoff` — chain hashes, EPP scorers, prefix hashing vs bounded loads vs weighted scoring vs the affinity filter's TTFT gate, hot prefixes, approximate vs precise indexes, LoRA adapter affinity | `02_scorer_weights_and_hot_prefixes` (T0) |
| 3 | §4 | `03_autoscaling_on_the_right_signal` — the HPA rule, tolerance, stabilization, policies, Pending pods; GPU util vs queue vs KV vs in-flight requests vs prefill backlog, on chat and on a chat + RAG mix; cold start and scale-to-zero with the node | `03_autoscaling_recommender` (T0; manifests for T3) |
| 4 | §5 | `04_prefill_decode_disaggregation` — the prefill stall, KV transfer cost, P:D sizing and the split search, chunk budget first, conditional disaggregation | — |
| 5 | §6 | `05_kv_cache_tiers_and_agent_sessions` — agent working sets, fetch vs recompute, tiered LRU, sticky vs shared tiers | — |
| 6 | §7–10 | — (§7's adapter routing is in `02`) | `04_local_stack_with_llm_d` (T0 + Docker or kind), `05_gke_inference_gateway` (T3; offline it plans and inspects) |

Budget about 9 hours for the primer and the core, and 8 more for the lab (see [`CURRICULUM.md`](../../CURRICULUM.md),
modules 05.1–05.6).

## Run tiers

| Tier | What runs | Where |
|---|---|---|
| **T0** | all five core notebooks and their tests; lab notebooks 01–03 against fake backends; lab 04–05 in offline mode | laptop, Colab CPU, CI — no GPU, no network |
| **T0 + Docker** | the lab's local stack: compose (simulated backends, router, Prometheus) or kind with the llm-d Router in standalone mode and `llm-d-inference-sim` | a laptop with Docker |
| **T1/T2** | the lab's router and autoscaler in front of real vLLM on one or two GPUs | any rented GPU box; see [`COMPUTE.md`](../../COMPUTE.md) |
| **T3** | GKE Inference Gateway: InferencePool, the endpoint picker, InferenceObjective priorities, an HPA on Managed Prometheus metrics, an L4 Spot pool that scales from zero | GCP, via the lab's Terraform and manifests |

## Quick start (T0)

```bash
cd orchestrator-core
python3 -m pip install -r requirements.txt     # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                           # the whole library's behaviour, pinned to hand-computed numbers
python3 -m jupyterlab notebooks                # do the exercises; finished versions are in solutions/
```

## Builds on, and leads to

- Below: the engine (`04-inference-engine/` — [KV cache](../../04-inference-engine/kv-cache/kv-cache-primer.md),
  [paged attention](../../04-inference-engine/paged-attention/paged-attention-primer.md), and the
  [`serving-engine`](../../04-inference-engine/serving-engine/README.md) topic for batching, chunked prefill and
  prefix caching), Kubernetes GPU scheduling and startup latency
  ([`03-kubernetes-gpu/gpu-scheduling`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md)), fabrics and cold start
  ([`01-hardware-gpu-fabric/roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)),
  prefill vs decode and disaggregation in one page
  ([GPU deployment primer](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §2 and §8).
- Above: admission control, rate limits and cost per conversation
  ([`06-gateway/scaling-admission-cost/agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)),
  and the agent workloads whose sessions shape every routing and caching decision
  ([`07-application-agent-framework`](../../07-application-agent-framework/README.md)).
