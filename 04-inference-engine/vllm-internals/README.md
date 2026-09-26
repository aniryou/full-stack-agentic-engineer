# vllm-internals — follow a request through vLLM's source and explain every number it prints

After this topic you can open vLLM's code and say why a request waited, where its time went, why the KV cache holds
as many tokens as it does and which flag moves which number — every mechanism cited to a file and function in vLLM
`main` at commit `5840d95` (2026-09-25, fetched 2026-09-26; latest PyPI release then: 0.30.0).

## Start here

1. Read [`vllm-internals-primer.md`](vllm-internals-primer.md): "The one-minute version", then §1–§4 (process map,
   a request's life, the scheduler, the KV cache).
2. Open [`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb) and
   re-implement the parts vLLM does differently from the mini engine.
3. Clone vLLM at `5840d95` and follow [`source-map.md`](source-map.md)'s reading plan, one sitting at a time.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour.* Times are rough.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`vllm-internals-primer.md`](vllm-internals-primer.md) | explain the API-server / EngineCore split, the token-budget scheduler, admission and preemption, block-hash prefix caching and its eviction order, how the pool is sized from `gpu_memory_utilization`, the model runner and CUDA graphs, attention-backend selection, sampling and speculation, quantization and loading, KV connectors, and the flags, metrics and log lines that expose all of it | §1–4 first, then the rest as needed | T0 |
| [`source-map.md`](source-map.md) | find any of those mechanisms in the code: a concept → file → symbol index with line numbers at `5840d95`, and a reading plan of three sittings of about two hours with named line ranges and the branches to step over | three ~2 h sittings | T0 |
| [`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb) | implement the capped longest hit, predict the free queue, write `free_blocks` (uncached blocks to the head, LRU refresh) and the full-sequence admission gate with evictable hits — each exercise followed by a check, the last replaying the primer's preemption case (§3.7) at small scale; a final section recomputes every worked number in the primer, using the serving lab's `servelab.sizing` for the KV budgets | ~1.5 h | T0 |

## Run it

```bash
python3 -m pip install -e ../serving-engine/vllm-serving-lab   # the notebook's §4.7/§8.2 checks import servelab.sizing
python3 -m jupyterlab notebooks
git clone https://github.com/vllm-project/vllm && git -C vllm checkout 5840d95   # the source the line numbers refer to
```

| Part | Tier | Needs |
|---|---|---|
| `vllm-internals-primer.md`, `source-map.md` | T0 | a browser or a clone of vLLM at `5840d95` |
| `notebooks/01_block_hashes_and_eviction.ipynb` | T0 | Python 3 standard library; runs on a laptop or Colab CPU |
| Primer §13.2 one-process debugging recipe | T1 | one GPU (a free Colab or Kaggle T4 is enough with a 0.6B model; primer §13.1 lists options) |
| Observing metrics, preemption, spec-decode acceptance | T1 | the serving lab on any single GPU |
| Tensor/pipeline parallelism, P/D disaggregation (§5.6, §10) | T2 | a multi-GPU box; RDMA for realistic KV transfer |

## How it fits

The concepts come first, in [`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md) (continuous batching,
chunked prefill, prefix caching, speculation, quantization) and the kernel primers
[`kv-cache`](../kv-cache/kv-cache-primer.md), [`paged-attention`](../paged-attention/paged-attention-primer.md) and
[`flash-attention`](../flash-attention/flash-attention-primer.md). The mini engine
[`../serving-engine/mini-engine-core/`](../serving-engine/mini-engine-core/) builds the same scheduler and block
cache from scratch (`minengine/kv.py`, `minengine/scheduler.py`, notebooks `01`–`03`); primer §4.1 maps its names
onto vLLM's. To watch it run, [`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/) serves a
real model, scrapes `/metrics` and sweeps the flags of primer §11. For kernels, the FlashAttention
[deep dive](../flash-attention/flash-attention-deep-dive.md) goes below primer §6; for many replicas, continue to
[`05-orchestrator`](../../05-orchestrator/README.md).

## Caveats

- Line numbers and defaults are those of `main` at `5840d95`; they move with every release. Anything not confirmed in
  source is marked `(verify)`.
- The notebook's KV-budget checks use the serving lab's estimate of vLLM's defaults, not a measurement; the startup
  log's `Available KV cache memory` is the measurement.

## What you should be able to explain afterwards

- Why the scheduler has no prefill or decode phase, and what one step of `Scheduler.schedule` does.
- How many KV blocks a given model gets on a given GPU, and why the default context length can stop a 24 GB card
  from starting.
- Why a fully cached prompt still computes one block, and in which order cached blocks are evicted.
- Why a preempted request usually cannot return until the request that displaced it finishes, and what that does to
  everything queued behind it.
- Which flag trades time-to-first-token against inter-token latency, and which one only moves memory.
- Where stop strings, EOS, grammar masks and speculative verification happen, and in which process.
