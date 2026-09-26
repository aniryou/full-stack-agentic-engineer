# 04 · Inference engine — vLLM / SGLang / TensorRT-LLM

Understand what one engine instance does between "a request arrived" and "tokens are streaming out": after this
layer you can size a model's KV cache before paying for a GPU, explain continuous batching, chunked prefill, prefix
caching, speculation and quantization with numbers, read vLLM's scheduler and block pool in its source, and measure
a real server against an SLO.

## Where this layer sits

```
   05 orchestrator       which replica · how many replicas · prefill/decode split
 ▶ 04 inference engine   one replica: the step loop, the KV cache, batching, attention kernels
   03 kubernetes-gpu     pods, nodes and GPUs the engine runs on
   01 hardware           HBM bandwidth and FLOPs that set every step time
```

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* Times are rough and include the exercises.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`kv-cache/`](kv-cache/kv-cache-primer.md) | compute KV bytes per token and per request, and say why decode rereads all of it every step — a primer, a worked notebook and a practice notebook | ~2 h | T0 |
| [`paged-attention/`](paged-attention/paged-attention-primer.md) | explain fragmentation and how block tables, refcounts and copy-on-write fix it — a primer, `paged_attention_minimal.py` and a practice notebook | ~1.5 h | T0 |
| [`flash-attention/`](flash-attention/flash-attention-primer.md) | explain tiling and online softmax from zero ([primer](flash-attention/flash-attention-primer.md)), then defend a kernel or backend choice with exact byte counts, the FA2/FA3/FA4 changes, decode kernels, paged KV and a Triton forward pass ([deep dive](flash-attention/flash-attention-deep-dive.md)); `fa_calculators.py` computes every number (49 tests); two notebooks: [practice](flash-attention/flash_attention_practice.ipynb) (five exercises) and the [deep-dive companion](flash-attention/flash_attention_deep_dive.ipynb) | ~1.5 h primer + practice; ~4 h deep dive (rough) | T0 (kernel timing T1) |
| [`serving-engine/`](serving-engine/README.md) | explain and simulate the engine itself — step loop, continuous batching, chunked prefill, KV management, prefix caching, sampling and structured output, speculative decoding, quantization, parallelism, LoRA, measurement — then size, measure and tune a real vLLM: a [PRIMER](serving-engine/PRIMER.md), [`mini-engine-core`](serving-engine/mini-engine-core/) (a numpy "nano-vLLM", 6 notebooks) and [`vllm-serving-lab`](serving-engine/vllm-serving-lab/) (sizing, a load generator, `/metrics`, a fake vLLM for T0, Cloud Run and GKE deploys, 6 notebooks) | ~11 h primer + core; ~12 h lab | T0 → T1 (T2, T3 optional) |
| [`vllm-internals/`](vllm-internals/README.md) | follow a request through vLLM's source: the process split, the token-budget scheduler, block-hash prefix caching and its eviction order, how the KV pool is sized, the model runner, backends and flags — a [deep primer](vllm-internals/vllm-internals-primer.md), a [source map](vllm-internals/source-map.md) with a reading plan, and a [notebook](vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb) that re-implements the parts vLLM does differently | three ~2 h sittings + the notebook | T0 (observing it T1) |

## Start here

1. Read the three kernel primers in order: [KV cache](kv-cache/kv-cache-primer.md) →
   [paged attention](paged-attention/paged-attention-primer.md) →
   [FlashAttention](flash-attention/flash-attention-primer.md), doing each topic's practice notebook.
2. `cd serving-engine/vllm-serving-lab && python3 -m pip install -e ".[dev]" && python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384`
   — under a second, and it prints the KV blocks and concurrency an 8B model gets on a 24 GB L4.
3. Work [`serving-engine/`](serving-engine/README.md) by its module table (primer section → core notebook → lab
   notebook), then read the real engine with [`vllm-internals/`](vllm-internals/README.md).

Reading order across the layer: kv-cache → paged-attention → flash-attention → serving-engine → vllm-internals.
The FlashAttention deep dive can wait until after serving-engine; it pays off most next to vLLM's attention backends
(vllm-internals primer §6).

## Run it

```bash
cd flash-attention && python3 -m pytest -q                         # 49 tests, ~1 s (numpy)
cd ../serving-engine/mini-engine-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q  # 67 tests, ~15 s
cd ../vllm-serving-lab
python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 66 tests, a few seconds, offline
```

Then `python3 -m jupyterlab notebooks` in either serving-engine directory, or the Colab badges below. The
vllm-internals notebook needs only the standard library plus the serving lab installed (`pip install -e` above).

## How it fits

Builds on [`00-foundations/transformers`](../00-foundations/transformers/) (attention and decoding),
[`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (weights, KV bytes, TTFT
and TPOT) and layer 01's [`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (why a
decode step is a memory read). Layer 02 (CUDA graphs, the all-reduces tensor parallelism runs on) and layer 03 (how
the engine's pod gets its GPUs) sit between them. Leads to [`05-orchestrator`](../05-orchestrator/README.md), which
routes across many engine replicas by prefix-cache affinity and load, autoscales them and splits prefill from decode.

## Caveats

- Every latency the mini engine and the lab's fake server print is **simulated** from a roofline model; the lab's
  numbers become measurements only against a real `vllm serve` (T1 and up).
- vLLM facts are pinned to v0.30.0 and `main` at `5840d95` (September 2026) and marked `(verify)` where they move;
  each primer ends with a dated verify list.

## Scope of this layer

**Covers:** paged attention & KV-cache management, continuous/in-flight batching, prefill vs decode, attention
kernels (flash attention), tensor/pipeline parallelism, speculative decoding, quantization,
throughput-vs-latency tuning.

**Signal keywords:** vLLM, SGLang, TensorRT-LLM, KV cache, paged attention, flash attention, continuous batching,
prefill/decode, speculative decoding, quantization, FP8, tensor parallel, tokens/sec, TTFT, ITL.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`flash-attention/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/flash-attention/flash_attention_practice.ipynb) `flash_attention_practice.ipynb`

**`kv-cache/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/01_kv_cache_worked.ipynb) `01_kv_cache_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/kv-cache/02_kv_cache_practice.ipynb) `02_kv_cache_practice.ipynb`

**`paged-attention/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/04-inference-engine/paged-attention/paged_attention_practice.ipynb) `paged_attention_practice.ipynb`
<!-- colab-links:end -->
