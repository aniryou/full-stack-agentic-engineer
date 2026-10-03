# serving-orchestration — route, scale and split a fleet of LLM engines

After this topic, you can give the numbers for these decisions:

- Which replica is the correct one for an LLM request.
- How many replicas a fleet needs, and which signal is the correct one to decide that number.
- When to divide prefill from decode, or move KV cache out of HBM.

Then you can run those decisions as a real router in front of real or emulated engines.

## Start here

1. Read [PRIMER.md](PRIMER.md): first "The one-minute version", then §1. Section 1 tells why a replica is a cache
   and why round-robin fails.
2. Run `cd orchestrator-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 66 tests
   take about 30 s. Then open [`01_why_llm_load_balancing_is_different`](orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb).
3. Put a real router in front of three emulated engines. This step also runs on a laptop CPU:
   [`inference-gateway-lab/notebooks/01_router_in_process.ipynb`](inference-gateway-lab/notebooks/01_router_in_process.ipynb).

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* "T0 + Docker" is
a laptop with Docker, also at no cost.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain the three decisions above the engine, with worked numbers: which replica, how many, and how to divide the work. The primer has ten sections. First, why a layer above the engine (§1), routing signals and algorithms (§2), and flow control and priorities (§3). Then autoscaling (§4), prefill/decode disaggregation (§5) and KV cache beyond HBM (§6). Then multi-model, multi-LoRA and model routing (§7), and large MoE topologies (§8). Last, the Kubernetes-native stack (§9) and where to run it (§10). Then a design-review walkthrough and six drills. | read alongside the core | — |
| [`orchestrator-core/`](orchestrator-core/) | Predict what a routing, autoscaling or disaggregation choice does to TTFT, hit rate and GPU-hours, through five fill-in notebooks. The tool is `fleetsim`, a pure-Python discrete-event simulator of a fleet (workloads, engine replicas with prefix caches, routers, the Kubernetes HPA algorithm, P/D, KV tiers). | ~9 h with the primer | T0 |
| [`inference-gateway-lab/`](inference-gateway-lab/) | Run the decision logic of the llm-d endpoint picker as a real async router (`igwlab`) in front of OpenAI-compatible backends (emulated or vLLM). Change scraped metrics into HPA decisions. Deploy the llm-d Router on kind, and GKE Inference Gateway on GCP. | ~6 h (01–04) + 2 h (05) | T0 to T3 |

Every number that the core prints is **simulated**, and the core labels it so. The simulation is a model of an
engine, built from spec-sheet arithmetic. The lab measures real (or explicitly emulated) servers.

### Work it in this order

Read the primer section, then do the core notebook for that section, then the lab notebook if there is one. The times are
approximate and cover the full row. They come from the curriculum of the repo ([`CURRICULUM.md`](../../CURRICULUM.md),
modules 05.1–05.6).

| Step | Primer | Core notebook (T0) | Lab notebook | Time |
|---|---|---|---|---|
| 1 | §1, §2.1–2.2, §3 | [`01_why_llm_load_balancing_is_different`](orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb): the spread of request cost, round-robin against least-outstanding against power-of-two, the herd that stale metrics cause. Also flow control: a router queue with priorities and a per-endpoint cap, with its size from Little's law. | [`01_router_in_process`](inference-gateway-lab/notebooks/01_router_in_process.ipynb) (T0) | 3 h |
| 2 | §2.3–2.6, §7 | [`02_cache_aware_routing_and_the_load_tradeoff`](orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb): chain hashes, EPP scorers, prefix hashing against bounded loads against weighted scoring against the TTFT gate of the affinity filter. Also hot prefixes, approximate against precise indexes, LoRA adapter affinity. | [`02_scorer_weights_and_hot_prefixes`](inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) (T0) | 3 h |
| 3 | §4 | [`03_autoscaling_on_the_right_signal`](orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb): the HPA rule, tolerance, stabilization, policies, Pending pods. Then GPU util against queue against KV against in-flight requests against prefill backlog, on chat and on a chat + RAG mix. Also cold start and scale-to-zero with the node. | [`03_autoscaling_recommender`](inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) (T0, manifests for T3) | 3 h |
| 4 | §5 | [`04_prefill_decode_disaggregation`](orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb): the prefill stall, KV transfer cost, the calculation of P:D and the split search, the chunk budget first, conditional disaggregation. | — | 2 h |
| 5 | §6 | [`05_kv_cache_tiers_and_agent_sessions`](orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb): agent working sets, fetch against recompute, tiered LRU, sticky against shared tiers. | — | 2 h |
| 6 | §7–10 | None. The adapter routing of §7 is in `02`. | [`04_local_stack_with_llm_d`](inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb) (T0 + Docker or kind), [`05_gke_inference_gateway`](inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) (T3. Offline, it plans and inspects.) | 2 h (+2 h on GCP) |

At the end, do the "In a design review" section of the primer: a two-minute walkthrough and six drills with answers.

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

On Colab, the first cell of each notebook clones the repo and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | What runs | Where | Cost |
|---|---|---|---|
| **T0** | All five core notebooks and their tests. Lab notebooks 01–03 against fake backends. Lab 04–05 in offline mode. | laptop, Colab CPU, CI, with no GPU and no network | $0 |
| **T0 + Docker** | The local stack of the lab: compose (simulated backends, router, Prometheus), or kind with the llm-d Router in standalone mode and `llm-d-inference-sim`. | a laptop with Docker | $0 |
| **T1/T2** | The router and the autoscaler of the lab in front of real vLLM on one or two GPUs. | Colab/Kaggle T4, or any rented GPU box | free on Colab/Kaggle, a rented 24 GB GPU ~$0.3–0.7/h (verify) |
| **T3** | GKE Inference Gateway: InferencePool, the endpoint picker, InferenceObjective priorities, an HPA on Managed Prometheus metrics, an L4 Spot pool that scales from zero. | GCP, through the Terraform and the manifests of the lab | ~$0.16/h with the GPU pool at 0, ~$0.44/h with one L4 Spot node (assumed prices, verify) |

Prices and where to get GPUs: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

This topic is between one engine and the gateway in front of the fleet. It takes the engine as given. It gives
admission and cost to the gateway.

| | Read | For |
|---|---|---|
| before | [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) | prefill against decode, TTFT and TPOT |
| before | the [KV cache](../../04-inference-engine/kv-cache/kv-cache-primer.md) and [paged attention](../../04-inference-engine/paged-attention/paged-attention-primer.md) primers | the blocks that a replica caches and uses again |
| before | [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), and §2 and §8 of the [GPU deployment primer](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) | fabrics and cold start. Prefill against decode, and disaggregation, on one page. |
| beside | the `serving-engine` topic of layer 04 ([`04-inference-engine/serving-engine/`](../../04-inference-engine/serving-engine/)) and the `gpu-scheduling` topic of layer 03 ([`03-kubernetes-gpu/gpu-scheduling/`](../../03-kubernetes-gpu/gpu-scheduling/)) | batching, chunked prefill and prefix caching in one engine. Pods, GPUs and startup latency. |
| after | [`06-gateway/scaling-admission-cost/agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) | admission control, rate limits and cost per conversation |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/README.md) | the agent sessions. Their shared prompts and their histories, which become longer, shape every routing and caching decision. |

## Caveats

- **Simulated vs measured.** The numbers of the core come from an engine model, not from a GPU. For the T0 latencies of
  the lab, the HTTP is real, but the engine times are emulated. Use both to compare designs, and never as GPU numbers. Expect
  that the ranking of routers changes with the engine and the workload (PRIMER §2.5).
- **Untested on real infrastructure here.** The Docker, kind, GPU and GKE paths of the lab have only checks by
  construction (`bash -n`, `DRY_RUN=1`, Terraform `validate`, Kubernetes schema checks). Nobody executed them.
- **Dated facts.** The product versions (llm-d Router v0.10.0, Gateway API Inference Extension v1.6.2, vLLM 0.30.0)
  are as of September 2026. The GKE details and the prices are also as of September 2026. All of them have the mark
  `(verify)`. The
  [Verify list](PRIMER.md#verify-list) of the primer collects them.
