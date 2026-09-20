# 05 · Orchestrator — Dynamo, llm-d, Ray Serve

Coordinates many inference-engine replicas into one serving system: where a request
goes, how many replicas exist, and how prefill/decode are split across hardware.

**Covers:** request routing (KV-cache-aware, prefix-aware, load-aware), autoscaling
replicas to load, prefill/decode disaggregation, multi-model & LoRA routing,
KV-cache offload/transfer between workers, SLA-driven scheduling, serving-layer
observability.

**Signal keywords:** Dynamo, llm-d, Ray Serve, disaggregation, prefill/decode split,
KV-aware routing, prefix cache routing, replica autoscaling, LoRA routing, SLO routing.

_Empty — awaiting content. Drop serving-orchestration material here._

> Note: `06-gateway/scaling-admission-cost/agentic-scaling-lab` touches this layer
> (capacity planning, autoscaling) but is filed under Gateway — see the root CLAUDE.md.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

_No notebooks yet._
<!-- colab-links:end -->
