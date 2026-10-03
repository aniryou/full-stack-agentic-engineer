# KV Cache on GPUs — A Primer

*This primer does not assume that you know how attention operates inside. It shows, step by step, why the KV cache is the largest single constraint in LLM serving.*

**Tier and notebooks.** All of this primer is T0. The two notebooks, [`notebooks/01_kv_cache_worked.ipynb`](notebooks/01_kv_cache_worked.ipynb) and [`notebooks/02_kv_cache_practice.ipynb`](notebooks/02_kv_cache_practice.ipynb), run on numpy and matplotlib through [`kernel-core`](../kernel-core/README.md) (`kerncore.kv`). They run on a laptop or on a Colab CPU runtime, with no GPU and no PyTorch. Torch is optional: only one comparison cell at the end of 01 uses it. That cell has a clear label, and it does not run when the runtime has no torch.

Here and in the notebooks, sizes are in binary units (1 KiB = 1,024 bytes, 1 GiB = 2³⁰ bytes). Decimal GB (1 GB = 10⁹ bytes) follow in brackets. `kernel-core/tests/test_primer_numbers.py` pins every size on this page.

---

## 1. The setup: how an LLM actually writes text

A language model generates text one token at a time. To make token 501, the model runs a full forward pass over the sequence up to that point. It gets a probability distribution over the vocabulary, samples one token and adds that token to the end of the sequence. Then it does these steps again.

Each transformer layer contains an **attention** block. For each token, attention makes three vectors from the current representation of that token:

- **Q (query)**: "what am I looking for?"
- **K (key)**: "what do I offer, as a thing to be looked at?"
- **V (value)**: "what information do I actually contribute?"

To calculate the output of a token, attention compares its **Q** with the **K** of each token before it. It changes those comparison scores into weights. Then it calculates a weighted sum of the **V** vectors of the same tokens.

Two properties are of primary importance:

1. **Causality.** Token $i$ can attend only to tokens $\le i$. It never sees the future.
2. Because of causality, the **K** and **V** of token $i$ depend only on the tokens up to $i$. After the model calculates them, **they never change**. They do not change when token 502 arrives, and they do not change at any later time.

---

## 2. The problem, and the fix

**Naive approach:** for each new token, this approach runs the whole sequence through the model again. It calculates K and V again for all the previous tokens. For a 1,000-token response, it calculates the K and V of token 1 a thousand times, and the result is the same each time. The cost increases quadratically with the output length.

**KV cache:** keep the K and V vectors of each token in GPU memory after the model calculates them one time. Then each step of the generation must do only three things:

1. Calculate Q, K, V for the **one** new token
2. Add that new K and V to the end of the cache
3. Run attention for the new Q over the **whole cached** K and V

The work per step decreases from "reprocess everything" to "process one token, read the cache."

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

Note that **the engine does not cache Q**. The query of a past token had one use: the calculation of the output of that token. The model never needs it again. The model reads only K and V again.

In practice, you cannot turn off this optimization. Every inference engine in production does it. The question of interest is not *if* you cache, but *what it costs*.

---

## 3. Two phases with completely different hardware behaviour

This distinction is the core of GPU inference.

**Prefill** is the phase that processes the input prompt. All the prompt tokens go through the model at the same time, in parallel, as large matrix multiplications. This phase fills the cache. It is **compute-bound**: the tensor cores of the GPU are the bottleneck. The latency of this phase is your *time to first token*.

**Decode** is the phase that generates the output, one token per forward pass. There is only one token of new work. But to do that work, you must read every model weight and the full KV cache from memory. Decode is **memory-bandwidth-bound**: the tensor cores are idle most of the time while they wait for data. The latency of this phase is your *tokens per second*.

Why is decode bandwidth-bound? An H100 does roughly 1,000 TFLOP/s in fp16, but it moves only about 3.35 TB/s from HBM. To keep the arithmetic units busy, you need ~300 floating-point operations per byte loaded.

Attention over a cache does about **1 FLOP per byte** in fp16 with full multi-head attention. The kernel loads each cached K or V element (2 bytes) one time. It uses that element in one multiply-add (2 FLOPs) for its one query head. In general, the value is ${2g/b}$ FLOP per byte, where $g$ query heads share each KV head and each cached value has $b$ bytes. Thus GQA with `g = 4` (Llama 3 8B) gets to 4 FLOP/B.

In both cases, the value is about two orders of magnitude too low (the [FlashAttention deep dive](../flash-attention/flash-attention-deep-dive.md) derives it). The GPU is a delivery truck in a traffic jam, not an engine without sufficient horsepower.

A useful approximation for decode speed:

$$
\text{tokens/sec} \approx \frac{\text{HBM bandwidth}}{\text{bytes of weights} + \text{bytes of KV cache read per token}}
$$

---

## 4. How big is the cache, concretely

$$
\begin{aligned}
\text{KV bytes} = \underbrace{2}_{\text{one for K, one for V}} &\times \text{layers} \times \text{kv_heads} \times \text{head_dim} \\
&\times \text{seq_len} \times \text{batch} \times \text{bytes_per_value}
\end{aligned}
$$

**Worked example — Llama 3 8B** (32 layers, 8 KV heads, head_dim 128, fp16 = 2 bytes):

```
per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB
```

| Scenario | KV cache |
|---|---|
| 1 user, 8K (8,192-token) context | 1 GiB (1.07 GB) |
| 1 user, 128K (131,072-token) context | 16 GiB (17.2 GB); the notebooks' 128,000 tokens give 15.6 GiB (16.8 GB) |
| 32 users, 8K context each | 32 GiB (34.4 GB) |

The *weights* of the model are a fixed ~16 GB (15 GiB) in fp16. On an 80 GB H100, that batch of 32 users already uses more memory than the model itself. The cache is the part that scales with your traffic and your context lengths. The weights do not change in size.

Look at it from the other direction. After the weights, an 80 GB H100 has ~64 GB left. That memory holds at most **59** sessions of 8K tokens with an fp16 cache, or **119** with an fp8 cache (`kerncore.kv.sessions_per_gpu`). That is an upper bound: an engine also reserves memory for activations and for its allocator.

**Now remove the architectural trick.** Llama 2 13B uses full multi-head attention, with 40 layers and 40 KV heads:

```
per token = 2 × 40 × 40 × 128 × 2 = 819,200 bytes = 800 KiB
```

Six times larger per token, for a model only 1.6× bigger. That gap is the effect of grouped-query attention, which §6 explains.

**Second-order effect:** at 128K context, decode reads ~16 GB of weights *plus* ~17 GB of cache per token. That is two times the traffic, so you get roughly **half the tokens per second** of a short context. Long context costs you capacity *and* speed.

---

## 5. Fragmentation, and why vLLM exists

To know the size is one task. To allocate the memory is a different task. The simple implementation reserves contiguous memory for `max_sequence_length` tokens per request, at the start. It does this because the cache grows during generation, and you do not want to reallocate the memory in the middle of the stream. But a request that has a 4K output setting and stops after 200 tokens wastes 95% of its reservation. In early serving stacks, the measured waste was 60–80%.

**PagedAttention** (the idea behind vLLM) takes the virtual-memory method of operating systems. The engine divides the cache into fixed-size blocks — say 16 tokens each. It allocates the blocks on demand and stores them non-contiguously. A block table for each request maps logical positions to physical blocks.

Waste drops to well under 4%. Also, the blocks become *shareable*: two requests with the same system prompt can point at the same physical blocks. The engine then does not store two copies.

The general form of these shared blocks is **prefix caching**: the engine reuses the cache for any common prefix across requests. Some workloads have long preambles that do not change (system prompts, tool schemas, retrieved documents, multi-turn history that grows only at the end). For these workloads, prefix caching often gives the largest single gain available, because it changes repeated prefill into a cache lookup.

---

## 6. The techniques for shrinking it

This list is in the approximate order of how universal their use is:

**Grouped-query attention (GQA) / multi-query attention (MQA).** The model keeps all the query heads, but each group of query heads shares one K/V head. Llama 3 8B has 32 query heads and 8 KV heads — a 4× cache reduction. MQA goes further, to a single shared KV head. The loss of quality is small. This technique is now standard.

**KV quantization.** Store the cache in fp8 or int8 instead of fp16 — 2× smaller and 2× less bandwidth per token. int4 is possible with more care. Keys are usually more sensitive to quantization than values.

**Multi-head latent attention (MLA).** This is the method of DeepSeek. The model compresses K and V into a small, shared, low-rank latent vector, and reconstructs them when it needs them. It compresses much more than GQA. It uses a small quantity of extra compute to save a large quantity of memory.

**Sliding-window / local attention.** Each token attends only to the last $W$ tokens, so the cache does not grow past $W$. Models usually combine it with a few "attention sink" tokens at the start of the sequence. These tokens prove to be necessary for stability.

**Eviction policies** (H2O, SnapKV and related methods). These policies use an observation: attention goes mostly to a minority of tokens. They evict the other tokens from the cache. They are lossy and their result depends on the workload, but they are effective.

**Cross-layer sharing.** Adjacent layers share cache entries. The model does not store one set for each layer.

**Offloading.** Move cold cache blocks to host RAM or NVMe, and move them back on demand. This gives more capacity, but it costs latency over PCIe.

---

## 7. The mental model to keep

> The KV cache is the short-term memory of the model for the current conversation. It makes the cost of generation linear instead of quadratic. But it lives in the same finite, slow-to-read HBM as the weights, and it grows with **every token, every user, and every layer**.

Three consequences to keep in mind:

1. **Context length is a serving cost, not just a training capability.** A 1M-token context window is an architecture claim. The KV cache tells you if you can afford to *serve* it.
2. **Batch size is capped by cache memory, not compute.** Throughput per GPU is mostly the answer to "how many concurrent sequences fit in leftover HBM."
3. **Decode speed is a bandwidth division problem.** If you want faster tokens, decrease the number of bytes that decode reads per token. Or buy more bandwidth.

---

## Glossary

| Term | Meaning |
|---|---|
| **HBM** | High-bandwidth memory on the GPU package. It is large and slow, compared with on-chip SRAM. The cache lives in it. |
| **Prefill** | The phase that processes the prompt. Parallel, compute-bound. |
| **Decode** | The phase that generates tokens one at a time. Sequential, bandwidth-bound. |
| **TTFT** | Time to first token. Most of it comes from prefill. |
| **GQA / MQA** | Query heads share K/V heads, so the cache is smaller. |
| **PagedAttention** | Block-based, non-contiguous cache allocation. The core idea in vLLM. |
| **Prefix caching** | The reuse of cached K/V for shared prompt prefixes across requests. |
| **Continuous batching** | The engine adds requests to the batch and removes them from it when sequences finish. It does not wait for the slowest sequence. |
| **Arithmetic intensity** | FLOPs per byte loaded. If the intensity is low, the work is bandwidth-bound. |

---

## Verify list (dated 2026-09-26)

This primer states these product facts and paper facts. The sizes on this page are calculations from them.

| Item | Value used | Why it needs checking |
|---|---|---|
| H100 peak and bandwidth (§3) | ~1,000 TFLOP/s dense fp16 (989 on the datasheet), 3.35 TB/s HBM (SXM), 80 GB | The datasheet. PCIe and NVL parts are different. |
| Llama 3 8B shape (§4, §6) | 32 layers, 32 query heads, 8 KV heads, head_dim 128, ~16 GB of fp16 weights | the model's `config.json` |
| Llama 2 13B shape (§4) | 40 layers, 40 KV heads (full MHA), head_dim 128 | the model's `config.json` |
| Fragmentation (§5) | 60–80% waste in early serving stacks, under 4% with paging, 16-token blocks | PagedAttention paper and vLLM's default block size |
| Decode intensity (§3) | ${2g/b}$ FLOP/B: 1 for fp16 MHA, 4 for Llama 3 8B's GQA | The FlashAttention deep dive derives it. Kernels that pack a GQA group reach it. Other kernels do not. |
