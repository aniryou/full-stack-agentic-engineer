# vLLM internals

How vLLM actually serves a request, read from its source: the process split between the API server and
EngineCore, the token-budget scheduler, the block-hash prefix cache and its eviction order, how the KV pool is
sized from `gpu_memory_utilization`, the model runner and CUDA graphs, attention-backend selection, sampling and
speculative decoding, quantization and loading, KV connectors for disaggregation, and the flags, metrics and log
lines that expose all of it. Every mechanism is cited to a file and function in vLLM `main` at commit `5840d95`
(2026-09-25, fetched 2026-09-26; latest PyPI release then: 0.30.0).

The concepts come first in the serving-engine topic; this topic is the "now read the real engine" layer on top.

## How to work through it

1. **Concepts**: [`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md) (continuous batching, chunked
   prefill, prefix caching, speculation, quantization) and the background primers
   [`../kv-cache/kv-cache-primer.md`](../kv-cache/kv-cache-primer.md),
   [`../paged-attention/paged-attention-primer.md`](../paged-attention/paged-attention-primer.md) and
   [`../flash-attention/`](../flash-attention/). The mini engine
   [`../serving-engine/mini-engine-core/`](../serving-engine/mini-engine-core/) builds the same scheduler and block
   cache from scratch (`minengine/kv.py`, `minengine/scheduler.py`, notebooks `01`–`03`); the primer's Section 4.1
   maps its names onto vLLM's.
2. **The engine**: [`vllm-internals-primer.md`](vllm-internals-primer.md), sections 1–4 first (process map, a
   request's life, scheduler, KV cache), then 5–10 as needed, then the flag table (11) and observability (12).
3. **The code**: [`source-map.md`](source-map.md), a concept → file → symbol index with line numbers at `5840d95`
   and a reading plan of three sittings of about two hours with named line ranges.
4. **Re-implement the parts vLLM does differently**:
   [`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb). Four exercises,
   each followed by a check that tests your code: the capped longest hit, a free-queue prediction, `free_blocks`
   with uncached blocks to the head and LRU refresh, and the full-sequence admission gate with evictable hits,
   which the check uses to replay the primer's preemption case (Section 3.7) at small scale. The last section
   recomputes every worked number in the primer, using the serving lab's `servelab.sizing` for the KV budgets.
5. **Watch it run**: [`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/) serves a real
   model, scrapes `/metrics`, and sweeps the flags from section 11.

## Tiers

| Part | Tier | Needs |
|---|---|---|
| `vllm-internals-primer.md`, `source-map.md` | T0 | a browser or a clone of vLLM at `5840d95` |
| `notebooks/01_block_hashes_and_eviction.ipynb` | T0 | Python 3 standard library; runs on a laptop or Colab CPU |
| Primer §13.2 one-process debugging recipe | T1 | one GPU (a free Colab or Kaggle T4 is enough with a 0.6B model; primer §13.1 lists options) |
| Observing metrics, preemption, spec-decode acceptance | T1 | the serving lab on any single GPU |
| Tensor/pipeline parallelism, P/D disaggregation (§5.6, §10) | T2 | a multi-GPU box; RDMA for realistic KV transfer |

## What you should be able to explain afterwards

- Why the scheduler has no prefill or decode phase, and what one step of `Scheduler.schedule` does.
- How many KV blocks a given model gets on a given GPU, and why the default context length can stop a 24 GB card
  from starting.
- Why a fully cached prompt still computes one block, and in which order cached blocks are evicted.
- Why a preempted request usually cannot return until the request that displaced it finishes, and what that does to
  everything queued behind it.
- Which flag trades time-to-first-token against inter-token latency, and which one only moves memory.
- Where stop strings, EOS, grammar masks and speculative verification happen, and in which process.
