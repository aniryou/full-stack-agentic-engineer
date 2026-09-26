# serving-engine — inside an inference engine

**The step loop, scheduling, batching, caching, speculation and quantization.** What happens on one engine
instance — vLLM, SGLang, TensorRT-LLM — between "a request arrived" and "tokens are streaming out", and which
knob trades what. The neighbouring topics in this layer cover the data structures and kernels the engine is built
on ([`kv-cache`](../kv-cache/), [`paged-attention`](../paged-attention/), [`flash-attention`](../flash-attention/));
this topic is the engine itself.

## What is here

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](PRIMER.md) | the concepts in twelve numbered sections — anatomy, continuous batching, chunked prefill, KV management, prefix caching, sampling and structured output, speculative decoding, quantization, parallelism, multi-LoRA, measurement, engines and where to run them — each formula with a worked number and the core function that computes it; then "In a design review", glossary, sources and a dated Verify list | reading |
| [`mini-engine-core/`](mini-engine-core/) | the minimal implementation: `minengine`, a numpy "nano-vLLM" (~960 lines) — a tiny model reading K/V through block tables, the KV cache manager with prefix caching, the scheduler, the sampler, speculative decoding, quantization and a roofline simulator — with six notebooks and 67 tests | T0 |
| [`vllm-serving-lab/`](vllm-serving-lab/) | the detailed implementation: `servelab` — size a model before serving it, an open/closed-loop load generator with streaming TTFT/ITL capture, a `/metrics` parser, knob sweeps against an SLO, an OpenAI-compatible fake server for T0, and deploy targets (any GPU box, Cloud Run GPU, GKE) | T0 → T1 → T3 |

## Before you start

- [`00-foundations/transformers`](../../00-foundations/transformers/) — attention, the KV cache, decoding
  (primer §7); [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) — weights, KV bytes,
  TTFT and TPOT on one page.
- [`01 gpu-primer`](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) — HBM, bandwidth and why a decode step
  is a memory read (§3).
- The three kernel topics next door: [`kv-cache`](../kv-cache/kv-cache-primer.md),
  [`paged-attention`](../paged-attention/paged-attention-primer.md) (block tables, copy-on-write — which block-hash
  prefix caching never needs, primer §5),
  [`flash-attention`](../flash-attention/flash-attention-primer.md) (tiling, online softmax).

## Order to work it

Read the primer sections, do the core notebook (T0), then run the lab notebook — at T0 against its fake server
first, then on a GPU if you have one. Module numbers match [`CURRICULUM.md`](../../CURRICULUM.md).

| Module | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| 04.1 The step loop | §1 Anatomy of an engine · §2 Continuous batching | [`01_the_step_loop_and_continuous_batching`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) | [`01_size_before_you_serve`](vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb), [`02_serve_and_measure`](vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) | T0 → T1 |
| 04.2 Chunked prefill and the KV budget | §3 Chunked prefill and prefill/decode interference · §4 KV cache management revisited | [`02_chunked_prefill_and_the_token_budget`](mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | [`03_knobs_and_tradeoffs`](vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) | T0 → T1 |
| 04.3 Prefix caching | §5 Prefix caching | [`03_prefix_caching`](mini-engine-core/notebooks/03_prefix_caching.ipynb) | [`04_prefix_caching_for_agents`](vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) | T0 → T1 |
| 04.4 Sampling and structured output | §6 Sampling and structured output | [`04_sampling_and_structured_output`](mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) | — | T0 |
| 04.5 Speculative decoding | §7 Speculative decoding | [`05_speculative_decoding`](mini-engine-core/notebooks/05_speculative_decoding.ipynb) | [`05_speculation_and_quantization_in_vllm`](vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 → T1 |
| 04.6 Quantization | §8 Quantization | [`06_quantization`](mini-engine-core/notebooks/06_quantization.ipynb) | [`05_speculation_and_quantization_in_vllm`](vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 → T1 (FP8 needs sm_89+) |
| 04.7 Parallelism, LoRA, measurement, deployment | §9 Parallelism inside the engine · §10 Multi-LoRA serving · §11 Measuring an engine · §12 Engines and where to run them | [`01`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) exercise 1.6 (tensor parallelism); [`02`](mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) worked example 3 (open loop vs saturation, goodput). §10 has no notebook exercise: its numbers come from `perf.lora_params()` and the adapter-salted block hashes, pinned in `tests/test_perf.py` and `tests/test_kv.py` | [`03_knobs_and_tradeoffs`](vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) exercises 3.1–3.5 (measurement) and 3.6 (tensor parallelism on two T4s, §9); [`06_deploy_on_cloud_run_gpu`](vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) (§12) | T0 → T1 (3.6 is T2; deployment T3) |

Each notebook ends with "In a design review" — the two-minute explanation and its drills. The primer's own
design-review section covers the whole topic.

## Tiers

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | every core notebook; the lab's sizing notebook; the lab's measurement notebooks against its bundled fake server (clearly labelled); every latency from `minengine.perf` is **simulated** | laptop, Colab CPU or CI — $0 |
| **T1** | the lab against a real vLLM server with a 0.5–2B model: TTFT/ITL, knob sweeps, prefix caching, n-gram speculation, quantized checkpoints | Colab/Kaggle T4 (free; fp16 only), any 24 GB GPU (~$0.3–0.7/hr), GCP L4 Spot |
| **T2** | optional: tensor parallelism across two GPUs (§9) | Kaggle 2×T4 (PCIe) or a rented NVLink pair |
| **T3** | the lab's Cloud Run GPU deployment (scale to zero) and GKE Deployment | GCP, pay per use; see the lab's `deploy/` READMEs for cleanup |

Prices, free tiers and how to obtain GPUs on GCP and elsewhere: [`COMPUTE.md`](../../COMPUTE.md).

## Where it connects

- **Down the stack.** Why decode is memory-bound and where the step-time knee comes from:
  [`01 roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §2–3. CUDA Graphs, which
  engines capture for decode, and the all-reduces tensor parallelism runs on:
  [`02 cuda-and-nccl`](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §4–5. How the engine's pod gets its GPUs:
  [`03 gpu-scheduling`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md).
- **Up the stack.** Many engine replicas, routed by prefix-cache affinity and load, autoscaled on queue depth and
  KV usage, split into prefill and decode pools: [`05-orchestrator`](../../05-orchestrator/). Admission, rate limits
  and cost in front of the fleet: [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/).
  The agent workloads that shape all of it — long stable prefixes, append-only histories, tool output quoted
  back: [`07-application-agent-framework`](../../07-application-agent-framework/).
