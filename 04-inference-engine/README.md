# 04 · Inference engine — vLLM / SGLang / TensorRT-LLM

The engine that runs a model on one or a few GPUs: turning weights + a batch of requests into
tokens, as fast and cheaply as possible.

**Covers:** paged attention & KV-cache management, continuous/in-flight batching, prefill vs
decode, attention kernels (flash attention), tensor/pipeline parallelism, speculative decoding,
quantization, throughput-vs-latency tuning.

**Signal keywords:** vLLM, SGLang, TensorRT-LLM, KV cache, paged attention, flash attention,
continuous batching, prefill/decode, speculative decoding, quantization, FP8, tensor parallel,
tokens/sec, TTFT, ITL.

## Current contents
- **`flash-attention/`** — flash attention from a minimal implementation (`flash_attention_minimal.py`)
  + a practice notebook.
- **`paged-attention/`** — paged attention (the KV-cache paging idea behind vLLM): minimal impl + practice.
- **`kv-cache/`** — KV cache mechanics: a worked notebook + a practice notebook.

_Drop more inference-engine material here (batching, quantization, speculative decoding, engine tuning)._

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
