# KV Cache on GPUs — A Primer

*Assumes no prior knowledge of attention internals. Builds up to why the KV cache is the single biggest constraint in LLM serving.*

**Tier and notebooks.** Reading this is T0. The two notebooks, [`01_kv_cache_worked.ipynb`](01_kv_cache_worked.ipynb) and [`02_kv_cache_practice.ipynb`](02_kv_cache_practice.ipynb), import **PyTorch** and matplotlib: they run on a CPU (T0 + torch), so use a Colab CPU runtime, where both are preinstalled, or `pip install torch matplotlib` locally. A GPU is optional; the timing cells use CUDA when one is present. Sizes below are in binary units (1 KiB = 1,024 bytes, 1 GiB = 2³⁰ bytes) with decimal GB in brackets; the notebooks print decimal GB (1 GB = 10⁹ bytes).

---

## 1. The setup: how an LLM actually writes text

A language model generates text one token at a time. To produce token 501, it runs a full forward pass over the sequence so far, gets a probability distribution over the vocabulary, samples one token, appends it, and repeats.

Inside each transformer layer sits an **attention** block. For every token, attention produces three vectors from that token's current representation:

- **Q (query)** — "what am I looking for?"
- **K (key)** — "what do I offer, as a thing to be looked at?"
- **V (value)** — "what information do I actually contribute?"

A token's output is computed by comparing its **Q** against the **K** of every token before it, turning those comparison scores into weights, and taking a weighted sum of the corresponding **V** vectors.

Two properties matter enormously:

1. **Causality.** Token *i* can only attend to tokens ≤ *i*. It never sees the future.
2. Because of causality, token *i*'s **K** and **V** depend only on tokens up to *i*. Once computed, **they never change** — not when token 502 arrives, not ever.

---

## 2. The problem, and the fix

**Naive approach:** to generate each new token, re-run the whole sequence through the model and recompute K and V for all previous tokens. Generating a 1,000-token response means recomputing token 1's K and V a thousand times, identical every time. Cost grows quadratically with output length.

**KV cache:** keep every token's K and V vectors in GPU memory after computing them once. Then each generation step only needs to:

1. Compute Q, K, V for the **one** new token
2. Append that new K and V to the cache
3. Attend the new Q against the **whole cached** K and V

Work per step drops from "reprocess everything" to "process one token, read the cache."

```
Step N:                    Step N+1:
┌──────────────────┐       ┌──────────────────┬───┐
│  K cache (N)     │  -->  │  K cache (N)     │ +1│
├──────────────────┤       ├──────────────────┼───┤
│  V cache (N)     │       │  V cache (N)     │ +1│
└──────────────────┘       └──────────────────┴───┘
   read all of it            append one column,
   to attend                 read all of it again
```

Note that **Q is not cached**. A past token's query was used once, to compute that token's output, and is never needed again. Only K and V are re-read.

This is not an optimization you can turn off in practice. Every production inference engine does it. The interesting question is not *whether* to cache but *what it costs*.

---

## 3. Two phases with completely different hardware behaviour

This distinction is the heart of GPU inference.

**Prefill** — processing the input prompt. All prompt tokens go through the model at once, in parallel, as large matrix multiplications. This fills the cache. It is **compute-bound**: the GPU's tensor cores are the bottleneck. Latency here is your *time to first token*.

**Decode** — generating output, one token per forward pass. There's only one token of new work, but you must read every model weight and the entire KV cache out of memory to do it. It is **memory-bandwidth-bound**: the tensor cores sit mostly idle waiting on data. Latency here is your *tokens per second*.

Why bandwidth-bound? An H100 does roughly 1,000 TFLOP/s in fp16 but moves only about 3.35 TB/s from HBM. To keep the arithmetic units busy you'd need ~300 floating-point operations per byte loaded. Attention over a cache does about **1 FLOP per byte** in fp16 with full multi-head attention: each cached K or V element (2 bytes) is loaded once and used in one multiply-add (2 FLOPs) for its one query head. In general it is `2g/b` FLOP per byte, with `g` query heads sharing each KV head and `b` bytes per cached value, so GQA with `g = 4` (Llama 3 8B) gets to 4 FLOP/B. Either way you are off by about two orders of magnitude (the [FlashAttention deep dive](../flash-attention/flash-attention-deep-dive.md) derives it). The GPU is a delivery truck stuck in traffic, not an engine short on horsepower.

Useful approximation for decode speed:

```
tokens/sec  ≈  HBM bandwidth ÷ (bytes of weights + bytes of KV cache read per token)
```

---

## 4. How big is the cache, concretely

```
KV bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
           ↑
        one for K, one for V
```

**Worked example — Llama 3 8B** (32 layers, 8 KV heads, head_dim 128, fp16 = 2 bytes):

```
per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB
```

| Scenario | KV cache |
|---|---|
| 1 user, 8K (8,192-token) context | 1 GiB (1.07 GB) |
| 1 user, 128K (131,072-token) context | 16 GiB (17.2 GB); the notebooks' 128,000 tokens give 15.6 GiB (16.8 GB) |
| 32 users, 8K context each | 32 GiB (34.4 GB) |

The model's *weights* are a fixed ~16 GB (15 GiB) in fp16. On an 80 GB H100, that batch of 32 users has already consumed more memory than the model itself. The cache is the part that scales with your traffic and your context lengths — the weights just sit there.

**Now remove the architectural trick.** Llama 2 13B uses full multi-head attention — 40 layers, 40 KV heads:

```
per token = 2 × 40 × 40 × 128 × 2 = 819,200 bytes = 800 KiB
```

Six times larger per token, for a model only 1.6× bigger. That gap is grouped-query attention, explained below.

**Second-order effect:** at 128K context, decode reads ~16 GB of weights *plus* ~17 GB of cache per token — twice the traffic, so roughly **half the tokens per second** compared to a short context. Long context costs you capacity *and* speed.

---

## 5. Fragmentation, and why vLLM exists

Knowing the size is one thing; allocating it is another. The naive implementation reserves `max_sequence_length` worth of contiguous memory per request up front, because the cache grows as generation proceeds and you don't want to reallocate mid-stream. But a request configured for 4K output that stops after 200 tokens wastes 95% of its reservation. Measured waste in early serving stacks ran 60–80%.

**PagedAttention** (the idea behind vLLM) borrows virtual memory from operating systems. The cache is split into fixed-size blocks — say 16 tokens each — allocated on demand and stored non-contiguously, with a per-request block table mapping logical positions to physical blocks. Waste drops to well under 4%, and blocks become *shareable*: two requests with the same system prompt can point at the same physical blocks instead of storing two copies.

That sharing generalises into **prefix caching** — reuse the cache for any common prefix across requests. For workloads with long fixed preambles (system prompts, tool schemas, retrieved documents, multi-turn history that only grows at the end), this is often the largest single win available, because it converts repeated prefill into a cache lookup.

---

## 6. The techniques for shrinking it

Roughly in order of how universally they're adopted:

**Grouped-query attention (GQA) / multi-query attention (MQA).** Keep all query heads, but let groups of them share one K/V head. Llama 3 8B has 32 query heads and 8 KV heads — a 4× cache reduction. MQA takes it to a single shared KV head. Quality loss is small; this is now standard.

**KV quantization.** Store the cache in fp8 or int8 instead of fp16 — 2× smaller and 2× less bandwidth per token. int4 is possible with more care. Keys are generally more sensitive to quantization than values.

**Multi-head latent attention (MLA).** DeepSeek's approach: compress K and V into a small shared low-rank latent vector and reconstruct on the fly. Far more aggressive than GQA, trading a little compute for a lot of memory.

**Sliding-window / local attention.** Each token attends only to the last *W* tokens, so the cache stops growing at *W*. Usually combined with a few "attention sink" tokens at the start of the sequence, which turn out to be necessary for stability.

**Eviction policies** (H2O, SnapKV and relatives). Observe that attention is concentrated on a minority of tokens; drop the rest from the cache. Lossy, workload-dependent, but effective.

**Cross-layer sharing.** Share cache entries between adjacent layers rather than storing one set per layer.

**Offloading.** Push cold cache blocks to host RAM or NVMe and pull them back on demand. Buys capacity, costs latency over PCIe.

---

## 7. The mental model to keep

> The KV cache is the model's short-term memory of the current conversation. It makes generation linear instead of quadratic in cost, but it lives in the same finite, slow-to-read HBM as the weights, and it grows with **every token, every user, and every layer**.

Three consequences worth internalising:

1. **Context length is a serving cost, not just a training capability.** A 1M-token context window is an architecture claim; whether you can afford to *serve* it is a KV cache question.
2. **Batch size is capped by cache memory, not compute.** Throughput per GPU is largely "how many concurrent sequences fit in leftover HBM."
3. **Decode speed is a bandwidth division problem.** If you want faster tokens, shrink the bytes read per token — or buy more bandwidth.

---

## Glossary

| Term | Meaning |
|---|---|
| **HBM** | High-bandwidth memory on the GPU package. Big and slow relative to on-chip SRAM. Where the cache lives. |
| **Prefill** | Processing the prompt. Parallel, compute-bound. |
| **Decode** | Generating tokens one at a time. Sequential, bandwidth-bound. |
| **TTFT** | Time to first token — dominated by prefill. |
| **GQA / MQA** | Sharing K/V heads across query heads to shrink the cache. |
| **PagedAttention** | Block-based, non-contiguous cache allocation; the core idea in vLLM. |
| **Prefix caching** | Reusing cached K/V for shared prompt prefixes across requests. |
| **Continuous batching** | Adding and retiring requests mid-batch as sequences finish, instead of waiting for the slowest. |
| **Arithmetic intensity** | FLOPs per byte loaded. Low intensity ⇒ bandwidth-bound. |

---

## Verify list (dated 2026-09-26)

Product and paper facts this primer states; the sizes are computed from them.

| Item | Value used | Why it needs checking |
|---|---|---|
| H100 peak and bandwidth (§3) | ~1,000 TFLOP/s dense fp16 (989 on the datasheet), 3.35 TB/s HBM (SXM), 80 GB | datasheet; PCIe and NVL parts differ |
| Llama 3 8B shape (§4, §6) | 32 layers, 32 query heads, 8 KV heads, head_dim 128; ~16 GB of fp16 weights | the model's `config.json` |
| Llama 2 13B shape (§4) | 40 layers, 40 KV heads (full MHA), head_dim 128 | the model's `config.json` |
| Fragmentation (§5) | 60–80% waste in early serving stacks; under 4% with paging; 16-token blocks | PagedAttention paper and vLLM's default block size |
| Decode intensity (§3) | `2g/b` FLOP/B: 1 for fp16 MHA, 4 for Llama 3 8B's GQA | derived in the FlashAttention deep dive; kernels that pack a GQA group reach it, others do not |
