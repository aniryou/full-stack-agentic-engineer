# vllm-internals — follow a request through vLLM's source and explain every number it prints

After this topic, you can open the code of vLLM and explain these things:

- why a request waited,
- where its time went,
- why the KV cache holds the number of tokens that it holds,
- which flag changes which number.

For each mechanism, the topic gives a citation to a file and a function in vLLM `main` at commit `5840d95`
(2026-09-25, fetched 2026-09-26). The latest PyPI release at that time was 0.30.0.

## Start here

1. Read [`vllm-internals-primer.md`](vllm-internals-primer.md): first "The one-minute version", then §1–§4.
   These sections are the process map, the life of a request, the scheduler and the KV cache.
2. Open [`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb).
   Write your own code for the parts that vLLM does differently from the mini engine.
3. Clone vLLM at `5840d95`. Then read the code in the order of the plan in [`source-map.md`](source-map.md).
   Read one sitting of the plan at a time.

## What you get

*Tiers: T0 = a laptop or a Colab CPU, free. T1 = one small GPU (Colab/Kaggle T4 or a rented card). T2 = a multi-GPU
box, rented for an hour.* The times are approximate.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`vllm-internals-primer.md`](vllm-internals-primer.md) | explain the API-server / EngineCore split, the token-budget scheduler, and admission and preemption. You can also explain block-hash prefix caching and its eviction order, and how vLLM calculates the size of the pool from `gpu_memory_utilization`. You can also explain the model runner and CUDA graphs, the selection of the attention backend, sampling and speculation, and quantization and weight loading. You can also explain KV connectors, and the flags, metrics and log lines that show all of it | §1–4 first, then the other sections when you need them | T0 |
| [`source-map.md`](source-map.md) | find each of those mechanisms in the code. The file has an index from concept to file to symbol, with line numbers at `5840d95`. It also has a plan to read the code in four sittings of about two hours. Each slot gives named line ranges, the branches to step over and a line count | four ~2 h sittings (about 8.5 h) | T0 |
| [`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb) | implement the capped longest hit and predict the free queue. You can also write `free_blocks` (uncached blocks to the head, LRU refresh) and the full-sequence admission gate with evictable hits. A check comes after each exercise. The last check replays the preemption case of the primer (§3.7) at small scale. A final section calculates each worked number in the primer again. For the KV budgets, it uses `servelab.sizing` from the serving lab | ~1.5 h | T0 |

## Run it

```bash
python3 -m pip install -e ../serving-engine/vllm-serving-lab   # the notebook's §4.7/§8.2 checks import servelab.sizing
python3 -m jupyterlab notebooks
git clone https://github.com/vllm-project/vllm && git -C vllm checkout 5840d95   # the source the line numbers refer to
```

| Part | Tier | Needs |
|---|---|---|
| `vllm-internals-primer.md`, `source-map.md` | T0 | a browser or a clone of vLLM at `5840d95` |
| `notebooks/01_block_hashes_and_eviction.ipynb` | T0 | Python 3 standard library. The notebook runs on a laptop or a Colab CPU |
| Primer §13.2, the one-process recipe to debug vLLM | T1 | one GPU. A free Colab or Kaggle T4 is sufficient with a 0.6B model. Primer §13.1 lists the options |
| To monitor metrics, preemption and spec-decode acceptance | T1 | the serving lab on any single GPU |
| Tensor/pipeline parallelism, P/D disaggregation (§5.6, §10) | T2 | a multi-GPU box, and RDMA for a realistic KV transfer |

## How it fits

Learn the concepts first. They are in [`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md) (continuous
batching, chunked prefill, prefix caching, speculation, quantization). They are also in the kernel primers
[`kv-cache`](../kv-cache/kv-cache-primer.md), [`paged-attention`](../paged-attention/paged-attention-primer.md) and
[`flash-attention`](../flash-attention/flash-attention-primer.md).

The mini engine [`../serving-engine/mini-engine-core/`](../serving-engine/mini-engine-core/) builds the same
scheduler and block cache from nothing (`minengine/kv.py`, `minengine/scheduler.py`, notebooks `01` to `03`). Primer
§4.1 gives the vLLM name for each of its names. To see vLLM run, use
[`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/). It serves a real model, scrapes
`/metrics` and sweeps the flags of primer §11.

For kernels, the FlashAttention [deep dive](../flash-attention/flash-attention-deep-dive.md) goes deeper than primer
§6. For many replicas, continue to [`05-orchestrator`](../../05-orchestrator/README.md).

## Caveats

- The line numbers and the defaults are those of `main` at `5840d95`. They change with each release. Each item that
  the source did not confirm has the tag `(verify)`.
- The KV-budget checks of the notebook use the estimate of the vLLM defaults from the serving lab. This estimate is
  not a measurement. The measurement is the `Available KV cache memory` line in the startup log.

## What you should be able to explain afterwards

- Why the scheduler has no prefill or decode phase, and what one step of `Scheduler.schedule` does.
- How many KV blocks a given model gets on a given GPU. Also, why the default context length can prevent the start of
  vLLM on a 24 GB card.
- Why a fully cached prompt still computes one block, and in which order the pool evicts cached blocks.
- Why a preempted request usually cannot return until the request that displaced it finishes. Also, what that does
  to all the requests in the queue behind it.
- Which flag trades time-to-first-token against inter-token latency, and which flag only changes memory.
- Where stop strings, EOS, grammar masks and speculative verification occur, and in which process.
