# 04 · Inference engine

Understand what one engine instance does between "a request arrived" and "tokens are streaming out". After this
layer, you can do these things:

- Calculate the size of a model's KV cache before you pay for a GPU.
- Explain continuous batching, chunked prefill, prefix caching, speculation and quantization with numbers.
- Read the scheduler and the block pool of vLLM in its source.
- Measure a real server against an SLO.

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

This layer is one replica. It runs the model (00) on its GPUs (01, 02), in a pod (03). It is one of the replicas of
the orchestrator (05).

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab or Kaggle T4, or a rented card). T2
is a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* The times are
approximate and include the exercises.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`kv-cache/`](kv-cache/kv-cache-primer.md) | calculate the KV bytes per token and per request, and explain why decode reads all of these bytes again at each step. The topic has a primer, a worked notebook and a practice notebook. The notebooks use numpy through kernel-core. Torch is optional in one comparison cell. | ~2 h | T0 |
| [`paged-attention/`](paged-attention/paged-attention-primer.md) | explain fragmentation and how block tables, refcounts and copy-on-write are the solution to it. The topic has a primer, `paged_attention_minimal.py` and a practice notebook. | ~1.5 h | T0 |
| [`flash-attention/`](flash-attention/flash-attention-primer.md) | explain tiling and online softmax from zero ([primer](flash-attention/flash-attention-primer.md)), and give the reasons for the selection of a kernel or a backend ([deep dive](flash-attention/flash-attention-deep-dive.md)). These reasons come from exact byte counts, the FA2/FA3/FA4 changes, decode kernels, paged KV and a Triton forward pass. `fa_calculators.py` calculates every number (49 tests). The topic has two notebooks: [practice](flash-attention/notebooks/flash_attention_practice.ipynb) (five exercises) and the [deep-dive companion](flash-attention/notebooks/flash_attention_deep_dive.ipynb). | ~1.5 h primer + practice, ~4 h deep dive (rough) | T0 (T1 to measure kernel times) |
| [`kernel-core/`](kernel-core/README.md) | show in code that a KV cache changes the cost but not the output. You can also show that block tables with refcounts and copy-on-write do not change attention, and that tiled online-softmax attention is exact. You can calculate every number of the three kernel primers again. The package is `kerncore`. It uses numpy, it has kv, paged and flash parts, and it imports fa_calculators.py. It has 51 tests. | ~2 h with the kv-cache notebooks | T0 |
| [`serving-engine/`](serving-engine/README.md) | explain and simulate the engine itself. The subjects are the step loop, continuous batching, chunked prefill, KV management, prefix caching, sampling and structured output, speculative decoding, quantization, parallelism, LoRA and measurement. Then you can calculate the size of a real vLLM deployment, measure it and adjust it. The topic has a [PRIMER](serving-engine/PRIMER.md), [`mini-engine-core`](serving-engine/mini-engine-core/) (a numpy "nano-vLLM", 6 notebooks) and [`vllm-serving-lab`](serving-engine/vllm-serving-lab/). The lab has size calculations, a load generator, `/metrics`, a fake vLLM for T0, Cloud Run and GKE deploys, and 6 notebooks. | ~11 h primer + core, ~12 h lab | T0 to T1 (T2, T3 optional) |
| [`quantization/`](quantization/README.md) | tell what INT4, FP8, NVFP4 or an FP8 KV cache gives a given model on a given GPU: decode speed, prefill speed or concurrency. You can also tell what each of them runs as on that GPU generation. You can make 4 bits accurate with GPTQ, AWQ or SmoothQuant. Then you can produce, serve and evaluate a real quantized checkpoint. The topic has a [PRIMER](quantization/PRIMER.md) (the deep dive behind serving-engine §8), [`quant-core`](quantization/quant-core/) and [`quant-lab`](quantization/quant-lab/). The core has numpy formats, GPTQ, AWQ, SmoothQuant, KV quantization and a per-GPU cost model. The core has 5 notebooks. The lab has llm-compressor checkpoints, FP16 against INT4 against FP8 in vLLM, lm-eval with error bars, FP8 KV and the NVFP4/MXFP4 layouts. It also has a bundled small model and a fake server for T0. The lab has 5 notebooks. | ~10 h primer + core, ~9 h lab | T0 to T1 (T3 optional) |
| [`vllm-internals/`](vllm-internals/README.md) | read the source of vLLM along the path of a request. The path goes through the process split, the token-budget scheduler, and block-hash prefix caching with its eviction order. It also goes through the calculation of the KV pool size, the model runner, the backends and the flags. The topic has a [deep primer](vllm-internals/vllm-internals-primer.md), a [source map](vllm-internals/source-map.md) with a plan to read the source, and a [notebook](vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb). The notebook makes a new implementation of the parts that vLLM does in a different way. | four sessions of ~2 h (about 8.5 h) + the notebook | T0 (T1 to observe it) |

## Start here

1. Read the three kernel primers in this order: [KV cache](kv-cache/kv-cache-primer.md),
   [paged attention](paged-attention/paged-attention-primer.md), then
   [FlashAttention](flash-attention/flash-attention-primer.md). With each primer, do the practice notebook of its topic (all numpy, T0).
2. Run `cd serving-engine/vllm-serving-lab && python3 -m pip install -e ".[dev]" && python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384`.
   It takes less than a second. It prints the KV blocks and the concurrency that an 8B model gets on a 24 GB L4.
3. Do [`serving-engine/`](serving-engine/README.md) in the order of its module table. For each module, read the
   primer section, then do the core notebook, then the lab notebook. Then get more detail where you need it. Use
   [`quantization/`](quantization/README.md) for number formats and calibration. Use
   [`vllm-internals/`](vllm-internals/README.md) to read the real engine.

The order to read the layer is: kv-cache, paged-attention, flash-attention, serving-engine, then quantization and
vllm-internals. Read the last two in either order. Quantization's §4 kernels and §6 FP8 KV cite vllm-internals §6.3
and §8. Thus, read these sections side by side. The FlashAttention deep dive can wait until you finish
serving-engine. It gives the most value next to vLLM's attention backends (vllm-internals primer §6).

## Run it

```bash
cd kernel-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q   # 51 tests, <1 s (numpy)
cd ../flash-attention
python3 -m pip install numpy pytest && python3 -m pytest -q         # 49 tests, a few seconds (numpy)
cd ../serving-engine/mini-engine-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q  # 75 tests, ~50 s
cd ../vllm-serving-lab
python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 74 tests, ~30 s, offline
cd ../../quantization/quant-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q  # 86 tests, ~30 s
cd ../quant-lab
python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 94 tests, ~35 s, offline (one needs Terraform, else skipped)
```

Then run `python3 -m jupyterlab notebooks` in any serving-engine or quantization directory. For the KV-cache
notebooks, run `python3 -m jupyterlab ../kv-cache` from `kernel-core`. You can also use the Colab links in "Run in
Colab". The vllm-internals notebook needs only the standard library and the installed serving lab (the
`pip install -e` command in the code block of this section).

## How it fits

**Needed first:** [`00-foundations/transformers`](../00-foundations/transformers/) (attention and decoding) and
[`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (weights, KV bytes, TTFT
and TPOT). These two are sufficient for the kernel topics and for serving-engine §1–8 at T0.

The [curriculum's spiral](../CURRICULUM.md#31-why-this-order) visits this layer two times on purpose. The first visit
is immediately after 00, for the concepts. The second visit is after layers 01 and 02, with a GPU. At that time, you
need layer 01's [`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (why a decode step
is a memory read). You need it for serving-engine §9–12, the measurements, quantization and the two deep dives.

Layer 02's [`cuda-and-nccl`](../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) is between layer 01 and this layer. It
covers CUDA Graphs and the all-reduces that tensor parallelism runs on. Layer 03's
[`gpu-scheduling`](../03-kubernetes-gpu/gpu-scheduling/README.md) is also between layer 01 and this layer. It covers
how the engine's pod gets its GPUs.

Two layer-00 topics bring workloads that change how you run an engine. The first is
[mixture-of-experts](../00-foundations/mixture-of-experts/PRIMER.md) (§6: fused MoE kernels and expert parallelism).
The second is [rl-and-thinking-models](../00-foundations/rl-and-thinking-models/PRIMER.md) (§7: long, heavy-tailed
outputs against the KV budget). Layer 00's
[distillation](../00-foundations/distillation/PRIMER.md#7-a-distilled-draft-for-speculative-decoding) (§7) trains the
draft model that speculative decoding (serving-engine §7) runs. The draft model is a student of its target.

This layer leads to [`05-orchestrator`](../05-orchestrator/README.md). The orchestrator routes across many engine
replicas by prefix-cache affinity and load. It autoscales them and separates prefill from decode.

## Caveats

- Every latency that the mini engine, `quantcore.cost` and the labs' fake servers print is a **simulated** value from
  a roofline model. The numbers of the labs become measurements only against a real `vllm serve` (T1 and up). FP8
  math needs an Ada or newer GPU (a free T4 runs INT4 and INT8 only). NVFP4 needs a rented Blackwell GPU.
- The vLLM facts are those of v0.30.0 and of `main` at `5840d95` (September 2026). A fact that changes over time has
  the `(verify)` tag. Each primer ends with a dated verify list.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`flash-attention/`** — [flash_attention_deep_dive](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/flash-attention/notebooks/flash_attention_deep_dive.ipynb) · [flash_attention_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/flash-attention/notebooks/flash_attention_practice.ipynb)
- **`kv-cache/`** — [01_kv_cache_worked](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/notebooks/01_kv_cache_worked.ipynb) · [02_kv_cache_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/notebooks/02_kv_cache_practice.ipynb)
- **`paged-attention/`** — [paged_attention_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/paged-attention/notebooks/paged_attention_practice.ipynb)
- **`quantization/quant-core/`** — [01_number_formats_and_error](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/01_number_formats_and_error.ipynb) · [02_granularity_and_outliers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/02_granularity_and_outliers.ipynb) · [03_gptq_awq_and_smoothquant_from_scratch](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) · [04_activation_and_kv_cache_quantization](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/04_activation_and_kv_cache_quantization.ipynb) · [05_choosing_a_scheme](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/notebooks/05_choosing_a_scheme.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/01_number_formats_and_error.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/02_granularity_and_outliers.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/03_gptq_awq_and_smoothquant_from_scratch.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/04_activation_and_kv_cache_quantization.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-core/solutions/05_choosing_a_scheme.ipynb)
- **`quantization/quant-lab/`** — [01_quantize_a_checkpoint](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/01_quantize_a_checkpoint.ipynb) · [02_serve_and_compare_schemes](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/02_serve_and_compare_schemes.ipynb) · [03_measure_the_accuracy_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/03_measure_the_accuracy_cost.ipynb) · [04_kv_cache_quantization_in_vllm](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/04_kv_cache_quantization_in_vllm.ipynb) · [05_fp4_and_the_blackwell_path](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/notebooks/05_fp4_and_the_blackwell_path.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/01_quantize_a_checkpoint.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/02_serve_and_compare_schemes.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/03_measure_the_accuracy_cost.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/04_kv_cache_quantization_in_vllm.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/quantization/quant-lab/solutions/05_fp4_and_the_blackwell_path.ipynb)
- **`serving-engine/mini-engine-core/`** — [01_the_step_loop_and_continuous_batching](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) · [02_chunked_prefill_and_the_token_budget](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) · [03_prefix_caching](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/03_prefix_caching.ipynb) · [04_sampling_and_structured_output](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) · [05_speculative_decoding](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/05_speculative_decoding.ipynb) · [06_quantization](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/01_the_step_loop_and_continuous_batching.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/02_chunked_prefill_and_the_token_budget.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/03_prefix_caching.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/04_sampling_and_structured_output.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/05_speculative_decoding.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/mini-engine-core/solutions/06_quantization.ipynb)
- **`serving-engine/vllm-serving-lab/`** — [01_size_before_you_serve](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb) · [02_serve_and_measure](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) · [03_knobs_and_tradeoffs](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) · [04_prefix_caching_for_agents](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) · [05_speculation_and_quantization_in_vllm](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) · [06_deploy_on_cloud_run_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/01_size_before_you_serve.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/02_serve_and_measure.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/03_knobs_and_tradeoffs.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/04_prefix_caching_for_agents.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/05_speculation_and_quantization_in_vllm.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/serving-engine/vllm-serving-lab/solutions/06_deploy_on_cloud_run_gpu.ipynb)
- **`vllm-internals/`** — [01_block_hashes_and_eviction](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb)
<!-- colab-links:end -->
