# PagedAttention: A Primer

## Why the KV cache is the problem

Serving throughput for autoregressive LLMs is, to first order, a batching problem: decode is memory-bandwidth-bound — each step reads all the weights and every cached K and V to produce one token per sequence — so the tensor cores sit mostly idle unless many sequences share each forward pass, and the binding constraint on batch size is memory. Weights are static and activations are small, so the dynamic consumer of memory is the KV cache. During decoding, each new token attends over the keys and values of every token before it; recomputing those projections each step would make generation quadratic in sequence length, so serving engines cache them. The cache grows by one token's worth of K and V per layer per step, lives for the entire request, and its final size is unknown in advance because you don't know how long the model will generate.

The footprint is large. Per token, the cache costs `2 × n_layers × n_kv_heads × d_head × bytes_per_element` (the 2 covers K and V). For LLaMA-13B in FP16 — 40 layers, 40 heads of dimension 128 — that works out to roughly 800 KB per token, so a single 2,048-token sequence occupies about 1.6 GB. On a 40 GB A100, the FP16 weights already take 26 GB, leaving room for only a handful of max-length sequences if each one gets a full reservation. Batch size, and therefore throughput, is bottlenecked by exactly this arithmetic.

Before vLLM, production systems (FasterTransformer, Orca) stored each request's KV cache as a single contiguous tensor, reserved up front at the maximum possible sequence length. Contiguity plus unknown lengths produces three distinct kinds of waste. Internal fragmentation: slots reserved for output that never materializes, because most requests stop well short of the maximum. Reservation waste: slots that will eventually be used but sit empty now, and cannot serve any other request in the meantime. External fragmentation: gaps between variable-sized contiguous allocations that are too small to fit a new request. The PagedAttention paper measured that only about 20–40% of KV cache memory in these systems held actual token state. Most of the scarcest resource on the GPU was reserved air.

## The core idea: virtual memory, applied to attention

PagedAttention (Kwon et al., SOSP 2023 — the paper that introduced vLLM) ports the operating-systems playbook: processes see contiguous virtual address spaces, physical memory is divided into fixed-size frames, and a page table maps one to the other. The observation is that nothing about attention actually requires the KV cache to be physically contiguous — that was an implementation convenience inherited from dense tensor frameworks.

Concretely, the KV cache is partitioned into fixed-size **KV blocks**, each holding keys and values for a small fixed number of tokens (16 by default in vLLM). A sequence sees an ordered list of logical blocks; a per-sequence **block table** maps each logical block to a physical block in a large pre-allocated pool on the GPU. Physical blocks can live anywhere in the pool, and they are allocated on demand — a new physical block is grabbed only when the current one fills.

```
Logical view (one sequence)       Block table         Physical block pool
[ blk0 ][ blk1 ][ blk2 ][ blk3 ]  blk0 -> phys 7      [0][1][2][3][4][5][6][7]...
  full    full    full   9/16     blk1 -> phys 1       scattered, non-contiguous,
                                  blk2 -> phys 4       shared across sequences
                                  blk3 -> phys 2
```

Two consequences follow. First, waste collapses to at most one partially filled block per sequence — under 4% in the paper's measurements, versus 60–80% before. Second, and more interestingly, the indirection decouples memory layout from sequence identity, which is what unlocks sharing and flexible scheduling.

## The kernel

The price of giving up contiguity is that the attention kernel can no longer read K and V as one strided tensor. The PagedAttention kernel takes the block table as an input, walks the logical blocks of each sequence, gathers the corresponding physical blocks, computes attention scores block by block, and combines the partial results using an online-softmax accumulation — the same numerically stable running-max-and-rescale trick that FlashAttention uses for tiling, applied here to a gather-from-blocks access pattern. In the paper's microbenchmarks this costs roughly 20–26% extra latency on the attention kernel itself relative to a contiguous layout, but attention is only a fraction of end-to-end step time, and the batching gains it enables dominate by a wide margin.

It is worth being precise about the relationship to FlashAttention, since the two are often confused: FlashAttention is about *computing* attention IO-efficiently — never materializing the full score matrix in HBM. PagedAttention is about *storing* the KV cache flexibly. They are orthogonal and compose; modern kernels (FlashAttention-2/3, FlashInfer) accept paged KV layouts natively, and "paged KV" has become the standard interface contract for attention kernels in serving stacks.

## Sharing and copy-on-write

Block-table indirection makes KV sharing almost free. Physical blocks carry reference counts, and multiple sequences' block tables may point to the same physical block. When a sequence needs to write into a block whose refcount exceeds one, the engine performs copy-on-write at block granularity: allocate a fresh block, copy the contents, decrement the old refcount, and redirect the block table.

Three workloads benefit. In parallel sampling (n completions from one prompt), all samples share the prompt's blocks and diverge only in their generated suffixes, so at most the last prompt block ever needs copying; savings are modest since only the prompt is shared, roughly 6–10% in the paper. In beam search, beams continually fork from shared ancestors, so large and dynamically changing fractions of the cache are shared — the paper measured up to ~55% memory savings at wide beam widths, a sharing pattern that is essentially unimplementable with contiguous allocation. Across requests, identical prefixes (system prompts, few-shot exemplars) can map to the same physical blocks: this is the mechanism underneath prefix caching, later generalized by SGLang's RadixAttention, which organizes cached prefixes in a radix tree over the paged pool. The prompt-caching products now offered by frontier-lab APIs are the commercial expression of the same block-reuse idea.

## Scheduling: continuous batching and preemption

Because memory is allocated block by block as sequences grow, the scheduler no longer needs to know output lengths to admit requests — it admits until the physical pool runs low. Combined with continuous (iteration-level) batching from Orca, where finished sequences exit the batch immediately and waiting ones join at any step, the effective batch size rises dramatically: vLLM reported 2–4× throughput over FasterTransformer and Orca at equivalent latency, with larger gains for long sequences, big models, and complex decoding.

The interesting design question is what happens when the pool is exhausted mid-flight, since blocks were promised optimistically. vLLM preempts victim sequences with one of two mechanisms. Swapping copies a sequence's blocks out to CPU RAM and back later, bounded by PCIe bandwidth. Recomputation simply frees the blocks and, when the sequence is rescheduled, replays its full prompt-plus-generated-so-far as a single prefill — which is often faster than swapping, because prefill is one large parallel pass while swap moves many small blocks over a slow bus. Eviction is all-or-nothing per sequence: every block of a sequence is touched on every step, so partial residency would be useless. Which mechanism wins depends on block size and hardware — small blocks favor recomputation, large ones favor swapping.

## Design trade-offs worth knowing

Block size is the central tuning knob. Smaller blocks mean finer-grained sharing and less internal waste per sequence, but larger block tables, more gather overhead, and worse memory coalescing in the kernel; larger blocks invert all of that. Sixteen tokens is the empirical middle ground and has proven remarkably sticky as a default.

Paging is also orthogonal to everything that shrinks the cache itself, and the effects multiply. GQA and MQA cut `n_kv_heads`; DeepSeek-style MLA compresses K and V into a low-rank latent; FP8/INT8 KV quantization halves or quarters bytes per element. All of these reduce the size of each block's contents while paging governs how blocks are placed, shared, and evicted.

The sharpest critique came from vAttention (2024), which argues that PagedAttention re-implements virtual memory in user space — software block tables, rewritten kernels — when CUDA's virtual memory management APIs can keep the cache virtually contiguous while physically paging underneath, letting unmodified kernels run. It is a fair architectural point, but in practice the paged model won: TensorRT-LLM, HF TGI, SGLang, and LMDeploy all adopted paged KV caches, and newer KV-centric architectures — disaggregated prefill/decode systems like Mooncake and DistServe, which ship KV blocks between prefill and decode workers — build directly on block-managed caches as the unit of transfer.

## Numbers to keep in your pocket

A 13B FP16 model costs ~800 KB of KV cache per token, so ~1.6 GB per 2K-token sequence; pre-paging systems achieved only 20–40% KV memory utilization versus >96% with paging; the default block holds 16 tokens; the kernel pays ~20–26% overhead on attention in isolation; and the headline result is 2–4× serving throughput over FasterTransformer and Orca at matched latency.

## Sources

Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 (arXiv:2309.06180) — the primary source, and unusually readable. Yu et al., "Orca," OSDI 2022, for continuous batching. Zheng et al., "SGLang" (arXiv:2312.07104) for RadixAttention. Prabhu et al., "vAttention" (arXiv:2405.04437) for the counter-argument. Dao et al., FlashAttention 1/2, for the compute-side complement.

## Verify list (dated 2026-09-26)

Paper and product facts this primer states; nothing here was re-measured.

| Item | Value used | Why it needs checking |
|---|---|---|
| LLaMA-13B shape and memory | 40 layers, 40 heads of dimension 128; ~800 KB of FP16 KV per token; 26 GB of FP16 weights on a 40 GB A100 | model config and the paper's setup |
| KV utilization before and after paging | 20–40% of KV memory held token state before; waste 60–80% before and under 4% with paging | PagedAttention paper (Kwon et al., SOSP 2023) |
| Kernel overhead | 20–26% extra attention-kernel latency versus a contiguous layout | paper microbenchmark, on the paper's GPUs and kernel |
| Sharing savings | ~6–10% for parallel sampling, up to ~55% for wide beam search | paper figures |
| Throughput | 2–4× over FasterTransformer and Orca at matched latency | paper figure; engines have changed a lot since |
| Default block size | 16 tokens in vLLM | engine default; attention backends may choose another size |
| Adoption | TensorRT-LLM, HF TGI, SGLang and LMDeploy use paged KV caches; FlashAttention-2/3 and FlashInfer accept paged layouts | project docs at the release in use |
