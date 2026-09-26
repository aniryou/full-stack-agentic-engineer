# 04 · Inference engine

Understand what one engine instance does between "a request arrived" and "tokens are streaming out": after this
layer you can size a model's KV cache before paying for a GPU, explain continuous batching, chunked prefill, prefix
caching, speculation and quantization with numbers, read vLLM's scheduler and block pool in its source, and measure
a real server against an SLO.

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

This layer is one replica: it runs the model (00) on its GPUs (01, 02), inside a pod (03), as one of the
orchestrator's replicas (05).

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* "T0 + torch" is T0 with CPU PyTorch installed
(Colab has it). Times are rough and include the exercises.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`kv-cache/`](kv-cache/kv-cache-primer.md) | compute KV bytes per token and per request, and say why decode rereads all of it every step — a primer, a worked notebook and a practice notebook | ~2 h | T0 + torch (Colab CPU) |
| [`paged-attention/`](paged-attention/paged-attention-primer.md) | explain fragmentation and how block tables, refcounts and copy-on-write fix it — a primer, `paged_attention_minimal.py` and a practice notebook | ~1.5 h | T0 |
| [`flash-attention/`](flash-attention/flash-attention-primer.md) | explain tiling and online softmax from zero ([primer](flash-attention/flash-attention-primer.md)), then defend a kernel or backend choice with exact byte counts, the FA2/FA3/FA4 changes, decode kernels, paged KV and a Triton forward pass ([deep dive](flash-attention/flash-attention-deep-dive.md)); `fa_calculators.py` computes every number (49 tests); two notebooks: [practice](flash-attention/flash_attention_practice.ipynb) (five exercises) and the [deep-dive companion](flash-attention/flash_attention_deep_dive.ipynb) | ~1.5 h primer + practice; ~4 h deep dive (rough) | T0 (kernel timing T1) |
| [`serving-engine/`](serving-engine/README.md) | explain and simulate the engine itself — step loop, continuous batching, chunked prefill, KV management, prefix caching, sampling and structured output, speculative decoding, quantization, parallelism, LoRA, measurement — then size, measure and tune a real vLLM: a [PRIMER](serving-engine/PRIMER.md), [`mini-engine-core`](serving-engine/mini-engine-core/) (a numpy "nano-vLLM", 6 notebooks) and [`vllm-serving-lab`](serving-engine/vllm-serving-lab/) (sizing, a load generator, `/metrics`, a fake vLLM for T0, Cloud Run and GKE deploys, 6 notebooks) | ~11 h primer + core; ~12 h lab | T0 → T1 (T2, T3 optional) |
| [`quantization/`](quantization/README.md) | say what INT4, FP8, NVFP4 or an FP8 KV cache buys for a given model on a given GPU — decode speed, prefill speed or concurrency — and what each runs as on that GPU generation; make 4 bits accurate with GPTQ, AWQ or SmoothQuant; then produce, serve and evaluate a real quantized checkpoint: a [PRIMER](quantization/PRIMER.md) (the deep dive behind serving-engine §8), [`quant-core`](quantization/quant-core/) (numpy formats, GPTQ, AWQ, SmoothQuant, KV quantization and a per-GPU cost model; 5 notebooks) and [`quant-lab`](quantization/quant-lab/) (llm-compressor checkpoints, FP16 vs INT4 vs FP8 in vLLM, lm-eval with error bars, FP8 KV, the NVFP4/MXFP4 layouts; a bundled tiny model and a fake server for T0; 5 notebooks) | ~10 h primer + core; ~9 h lab | T0 → T1 (T3 optional) |
| [`vllm-internals/`](vllm-internals/README.md) | follow a request through vLLM's source: the process split, the token-budget scheduler, block-hash prefix caching and its eviction order, how the KV pool is sized, the model runner, backends and flags — a [deep primer](vllm-internals/vllm-internals-primer.md), a [source map](vllm-internals/source-map.md) with a reading plan, and a [notebook](vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb) that re-implements the parts vLLM does differently | four ~2 h sittings (about 8.5 h) + the notebook | T0 (observing it T1) |

## Start here

1. Read the three kernel primers in order: [KV cache](kv-cache/kv-cache-primer.md) →
   [paged attention](paged-attention/paged-attention-primer.md) →
   [FlashAttention](flash-attention/flash-attention-primer.md), doing each topic's practice notebook.
2. `cd serving-engine/vllm-serving-lab && python3 -m pip install -e ".[dev]" && python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384`
   — under a second, and it prints the KV blocks and concurrency an 8B model gets on a 24 GB L4.
3. Work [`serving-engine/`](serving-engine/README.md) by its module table (primer section → core notebook → lab
   notebook), then go deeper where you need it: [`quantization/`](quantization/README.md) for number formats and
   calibration, [`vllm-internals/`](vllm-internals/README.md) to read the real engine.

Reading order across the layer: kv-cache → paged-attention → flash-attention → serving-engine → quantization and
vllm-internals (either order; quantization's §4 kernels and §6 FP8 KV cite vllm-internals §6.3 and §8, so read them
side by side). The FlashAttention deep dive can wait until after serving-engine; it pays off most next to vLLM's
attention backends (vllm-internals primer §6).

## Run it

```bash
cd flash-attention && python3 -m pytest -q                         # 49 tests, a few seconds (numpy)
cd ../serving-engine/mini-engine-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q  # 67 tests, ~5 s
cd ../vllm-serving-lab
python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 66 tests, a few seconds, offline
cd ../../quantization/quant-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q  # 78 tests, ~7 s
cd ../quant-lab
python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 86 tests, ~11 s, offline (one needs Terraform, else skipped)
```

Then `python3 -m jupyterlab notebooks` in any serving-engine or quantization directory, or the Colab links below. The
vllm-internals notebook needs only the standard library plus the serving lab installed (`pip install -e` above).

## How it fits

**Needed first:** [`00-foundations/transformers`](../00-foundations/transformers/) (attention and decoding) and
[`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (weights, KV bytes, TTFT
and TPOT) — enough for the kernel topics and serving-engine §1–8 at T0. The
[curriculum's spiral](../CURRICULUM.md#31-why-this-order) visits this layer twice on purpose: right after 00 for the
concepts, then again after layers 01 and 02 with a GPU, when layer 01's
[`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (why a decode step is a memory
read) is needed for serving-engine §9–12, the measurements, quantization and the two deep dives. Layer 02's [`cuda-and-nccl`](../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) (CUDA Graphs, the all-reduces tensor
parallelism runs on) and layer 03's [`gpu-scheduling`](../03-kubernetes-gpu/gpu-scheduling/README.md) (how the engine's pod
gets its GPUs) sit between them. Two layer-00 topics bring workloads that change how an engine is run:
[mixture-of-experts](../00-foundations/mixture-of-experts/PRIMER.md) (§6: fused MoE kernels and expert parallelism)
and [rl-and-thinking-models](../00-foundations/rl-and-thinking-models/PRIMER.md) (§7: long, heavy-tailed outputs
against the KV budget). Leads to [`05-orchestrator`](../05-orchestrator/README.md), which
routes across many engine replicas by prefix-cache affinity and load, autoscales them and splits prefill from decode.

## Caveats

- Every latency the mini engine, `quantcore.cost` and the labs' fake servers print is **simulated** from a roofline
  model; the labs' numbers become measurements only against a real `vllm serve` (T1 and up). FP8 math needs an Ada
  or newer GPU (a free T4 runs INT4 and INT8 only), and NVFP4 needs a rented Blackwell GPU.
- vLLM facts are pinned to v0.30.0 and `main` at `5840d95` (September 2026) and marked `(verify)` where they move;
  each primer ends with a dated verify list.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`flash-attention/`** — [flash_attention_deep_dive](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/flash-attention/flash_attention_deep_dive.ipynb) · [flash_attention_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/flash-attention/flash_attention_practice.ipynb)
- **`kv-cache/`** — [01_kv_cache_worked](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/01_kv_cache_worked.ipynb) · [02_kv_cache_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/02_kv_cache_practice.ipynb)
- **`paged-attention/`** — [paged_attention_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/paged-attention/paged_attention_practice.ipynb)
- **`quantization/quant-core/`** — [01_number_formats_and_error](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/01_number_formats_and_error.ipynb) · [02_granularity_and_outliers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/02_granularity_and_outliers.ipynb) · [03_gptq_awq_and_smoothquant_from_scratch](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) · [04_activation_and_kv_cache_quantization](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/04_activation_and_kv_cache_quantization.ipynb) · [05_choosing_a_scheme](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/05_choosing_a_scheme.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/01_number_formats_and_error.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/02_granularity_and_outliers.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/03_gptq_awq_and_smoothquant_from_scratch.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/04_activation_and_kv_cache_quantization.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/05_choosing_a_scheme.ipynb)
- **`quantization/quant-lab/`** — [01_quantize_a_checkpoint](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/01_quantize_a_checkpoint.ipynb) · [02_serve_and_compare_schemes](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/02_serve_and_compare_schemes.ipynb) · [03_measure_the_accuracy_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/03_measure_the_accuracy_cost.ipynb) · [04_kv_cache_quantization_in_vllm](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/04_kv_cache_quantization_in_vllm.ipynb) · [05_fp4_and_the_blackwell_path](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/05_fp4_and_the_blackwell_path.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/01_quantize_a_checkpoint.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/02_serve_and_compare_schemes.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/03_measure_the_accuracy_cost.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/04_kv_cache_quantization_in_vllm.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/05_fp4_and_the_blackwell_path.ipynb)
- **`serving-engine/mini-engine-core/`** — [01_the_step_loop_and_continuous_batching](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) · [02_chunked_prefill_and_the_token_budget](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) · [03_prefix_caching](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/03_prefix_caching.ipynb) · [04_sampling_and_structured_output](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) · [05_speculative_decoding](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/05_speculative_decoding.ipynb) · [06_quantization](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/01_the_step_loop_and_continuous_batching.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/02_chunked_prefill_and_the_token_budget.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/03_prefix_caching.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/04_sampling_and_structured_output.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/05_speculative_decoding.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/06_quantization.ipynb)
- **`serving-engine/vllm-serving-lab/`** — [01_size_before_you_serve](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb) · [02_serve_and_measure](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) · [03_knobs_and_tradeoffs](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) · [04_prefix_caching_for_agents](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) · [05_speculation_and_quantization_in_vllm](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) · [06_deploy_on_cloud_run_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/01_size_before_you_serve.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/02_serve_and_measure.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/03_knobs_and_tradeoffs.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/04_prefix_caching_for_agents.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/05_speculation_and_quantization_in_vllm.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/06_deploy_on_cloud_run_gpu.ipynb)
- **`vllm-internals/`** — [01_block_hashes_and_eviction](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb)
<!-- colab-links:end -->
