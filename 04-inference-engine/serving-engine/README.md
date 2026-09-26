# serving-engine — what one inference engine does, and which knob trades what

After this topic you can explain, with numbers, what an engine like vLLM, SGLang or TensorRT-LLM does between "a
request arrived" and "tokens are streaming out" — continuous batching, chunked prefill, KV management, prefix
caching, sampling, speculation, quantization — and then size, measure and tune a real vLLM server against an SLO.

## Start here

1. Read [PRIMER.md](PRIMER.md): "The one-minute version", then §1 Anatomy of an engine and §2 Continuous batching.
2. `cd mini-engine-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 67 tests in about
   15 s; then open [`01_the_step_loop_and_continuous_batching`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb).
3. Size a real model before serving it, still on a laptop:
   [`vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb`](vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the engine in twelve sections — §1 anatomy · §2 continuous batching · §3 chunked prefill · §4 KV cache management · §5 prefix caching · §6 sampling and structured output · §7 speculative decoding · §8 quantization · §9 parallelism · §10 multi-LoRA · §11 measuring an engine · §12 engines and where to run them — each formula with a worked number and the core function that computes it; then "In a design review", a glossary, sources and a dated Verify list | read alongside the core | — |
| [`mini-engine-core/`](mini-engine-core/) | build the engine yourself in `minengine`, a numpy "nano-vLLM" (~960 lines): a tiny model reading K/V through block tables, the KV cache manager with prefix caching, the scheduler, the sampler, speculative decoding, quantization and a roofline simulator; six fill-in notebooks | ~11 h with the primer | T0 |
| [`vllm-serving-lab/`](vllm-serving-lab/) | size a model from its `config.json`, drive vLLM with an open- or closed-loop load generator, read its `/metrics`, sweep its flags against an SLO, and deploy it on any GPU box, Cloud Run GPU or GKE (`servelab`; a fake vLLM makes every notebook run at T0); six notebooks | ~12 h | T0 → T1 (T2, T3 optional) |

### Work it in this order

Read the primer sections, do the core notebook (T0), then run the lab notebook — at T0 against its fake server
first, then on a GPU if you have one. Module numbers and times come from the repo's curriculum ([`CURRICULUM.md`](../../CURRICULUM.md), modules 04.1–04.7).

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

## Run it

```bash
cd mini-engine-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 67 tests, ~15 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../vllm-serving-lab
python3 -m pip install -e ".[dev]"
python3 -m pytest -q                           # 66 tests, a few seconds, offline
python3 -m jupyterlab notebooks
```

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | every core notebook; the lab's sizing notebook; the lab's measurement notebooks against its bundled fake server (clearly labelled); every latency from `minengine.perf` is **simulated** | laptop, Colab CPU or CI — $0 |
| **T1** | the lab against a real vLLM server with a 0.5–2B model: TTFT/ITL, knob sweeps, prefix caching, n-gram speculation, quantized checkpoints | Colab/Kaggle T4 (free; fp16 only), any 24 GB GPU (~$0.3–0.7/hr, verify), GCP L4 Spot |
| **T2** | optional: tensor parallelism across two GPUs (§9, the lab's exercise 3.6) | Kaggle 2×T4 (PCIe) or a rented NVLink pair |
| **T3** | the lab's Cloud Run GPU deployment (scale to zero) and GKE Deployment | GCP, pay per use; see the lab's `deploy/` READMEs for cleanup |

Prices, free tiers and how to obtain GPUs on GCP and elsewhere: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`00-foundations/transformers`](../../00-foundations/transformers/) (primer §7); [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) | attention, the KV cache, decoding; weights, KV bytes, TTFT and TPOT on one page |
| before | [`01 gpu-primer`](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) §3; [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §2–3 | HBM, bandwidth, why a decode step is a memory read and where the step-time knee comes from |
| before | the kernel topics next door: [`kv-cache`](../kv-cache/kv-cache-primer.md), [`paged-attention`](../paged-attention/paged-attention-primer.md), [`flash-attention`](../flash-attention/flash-attention-primer.md) | block tables and copy-on-write (which block-hash prefix caching never needs, primer §5); tiling and online softmax |
| beside | layer 02's [cuda-and-nccl primer](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §4–5 and layer 03's [gpu-scheduling](../../03-kubernetes-gpu/gpu-scheduling/README.md) topic | CUDA Graphs and the all-reduces tensor parallelism runs on; how the engine's pod gets its GPUs |
| after | [`vllm-internals`](../vllm-internals/README.md) | the same mechanisms read in vLLM's source, with line numbers |
| after | [`05-orchestrator`](../../05-orchestrator/README.md) | many replicas, routed by prefix-cache affinity and load, autoscaled on queue depth and KV usage, split into prefill and decode pools |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/); [`07-application-agent-framework`](../../07-application-agent-framework/) | admission, rate limits and cost in front of the fleet; the agent workloads — long stable prefixes, append-only histories — that shape all of it |

## Caveats

- **Simulated vs measured.** The core's latencies come from a roofline model driving its real scheduler, and the
  lab's fake server uses the same kind of model; both are labelled SIMULATED. The lab measures only against a real
  `vllm serve`.
- **Two sets of sizing inputs.** The core sizes KV with round inputs (0.9 of 24 GB minus a flat 1 GB); the lab
  models vLLM v0.30.0's defaults (0.92 of the driver-reported total minus profiled overheads). PRIMER §4 sets them
  side by side: 2,164 vs 2,363 blocks for Llama-3.1-8B on an L4. The startup log's `Available KV cache memory` is
  the measurement.
- **Checked by construction.** The GPU, Cloud Run and GKE paths are checked here with `bash -n`, `DRY_RUN=1`,
  Terraform `validate` and Kubernetes schema checks, not run on real infrastructure; expect to adjust quotas and
  regions on first use.
- **Dated facts.** vLLM v0.30.0 defaults, GPU prices and Cloud Run details are as of September 2026 and marked
  `(verify)`; the primer's [Verify list](PRIMER.md#verify-list) collects them.
