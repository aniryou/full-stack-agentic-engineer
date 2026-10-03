# Reading the machine: rooflines, memory hierarchy, fabrics, and the cost of a token

*A primer for layer 01. Snapshot: September 2026. Each product fact has a date and the mark (verify).
Each formula has a worked number. Each computed number comes from a function in
[`roofline-core/`](roofline-core/), and the tests of that core pin it.*

This primer turns a GPU datasheet and a network diagram into predictions that you can defend. The
predictions are:

- how long an LLM step takes, and which resource limits it,
- what a collective costs on each kind of link,
- how long a new replica takes to load,
- how often a large job fails,
- what a token costs.

The primer assumes that you know the qualitative picture from two other primers.
[`gpu-primer`](../gpu-primer/gpu-primer.md) explains why a GPU has the shape that it has.
[`gpu-deployment`](../gpu-deployment/gpu-deployment-primer.md) explains scale-up against scale-out, and
the parallelism menu. This primer leaves questions of size (does it fit, TTFT/TPOT budgets) to
[`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md).

You can learn everything here at tier T0 on a laptop. [`gpu-bench-lab/`](gpu-bench-lab/) measures the
same quantities on the hardware that you have (T0 CPU, T1 one GPU, T2 several, T3 on GCP).

---

## The one-minute version

A GPU is a memory system with arithmetic attached to it.

- Every kernel pays **max(FLOPs ÷ peak, bytes ÷ bandwidth)**. The ratio peak ÷ bandwidth is the
  **ridge point**, about 295 FLOP per byte for an H100 in bf16. It is the arithmetic intensity that a
  kernel needs before the math units are the limit.
- An LLM step streams every weight one time, whatever the number of tokens in the step. Thus its
  intensity is approximately **the number of tokens in the step**. A 2K-token prefill is compute-bound
  (TTFT ≈ FLOPs ÷ peak), and a batch-1 decode step is at ~1 FLOP/B (time per token ≈ bytes ÷ bandwidth).
  Batching moves the weight GEMMs of decode toward the ridge. But the **KV-cache reads** of each sequence
  grow as fast as its attention FLOPs. They keep attention at a few FLOP/B, and they become the largest
  cost in the step.
- Quantization and MoE change **bytes**.
- Between GPUs, every transfer costs **$\alpha + n/\beta$**. The tensor-parallel all-reduces of decode
  are latency-bound. The all-reduces of prefill are bandwidth-bound, and they do not decrease when the TP
  degree increases. Also, β per GPU falls ~9× from NVLink to a 400 Gb/s NIC. Thus tensor parallelism stays
  inside the NVLink domain.
- Three fleet numbers then come from the same arithmetic. Cold start is **bytes ÷ the slowest tier**.
  Failure rates **add** (a checkpoint every $\sqrt{2\,\delta M}$). The cost is
  **$/M tokens = $/GPU-hr ÷ (tokens/s × 3600 × utilisation) × 10⁶**.

---

## 1. Spec-sheet literacy

A datasheet is a small set of numbers. Five of them carry almost every argument in this layer:

| Line on the sheet | What to extract | Trap |
|---|---|---|
| Tensor TFLOPS by precision | the **dense** peak for the precision that you run | headline figures with the mark \* assume 2:4 sparsity: exactly 2× dense |
| FP32 TFLOPS | the non-tensor (CUDA-core) rate | an H100 has 67 FP32 TFLOP/s against 989 bf16 tensor. Code that does not use the tensor cores loses 14.8× of the chip's throughput. |
| Memory GB and TB/s | capacity (what fits) and bandwidth (how fast decode runs) | Capacity is binary. An "80 GB" H100 carries 80 GiB = 85.9 × 10⁹ bytes. The driver reports a small quantity less (with `nvidia-smi`, verify). The core uses 80 × 10⁹ for its plans, which is a safe value. |
| Interconnect GB/s | per-direction bandwidth | The datasheets give NVLink and PCIe as bidirectional totals. H100 "900 GB/s" is 450 each way. PCIe Gen5 x16 "128 GB/s" is 63 each way. |
| Network | bytes, not bits | a 400 Gb/s NIC moves 50 GB/s |

Two more lines are important for plans. **TDP** (or TBP) is the board power that the cooling and the
power delivery must support continuously. It is 72 W for an L4, 700 W for an H100 SXM, 1,400 W for a
GB300 (verify). Under a continuous tensor load, the clocks decrease below boost to stay inside that
limit. Thus a measured peak is lower than the datasheet value (measure it: `gpu-bench-lab` notebook
`01_measure_your_roofline`).

**Form factor** changes the part. An H100 PCIe card has fewer SMs, HBM2e at ~2 TB/s and a 350 W limit
(verify). Thus "H100" alone does not identify the part. Always name SXM, PCIe or NVL.

### Where a peak comes from

A peak is units × work per clock × clock (`roofline.specs.peak_from_clock()`):

$$
\text{peak} = \text{SMs} \times \text{dense tensor FLOP/clock/SM} \times \text{boost clock}
$$

| GPU | SMs | FLOP/clock/SM | Boost clock | Peak | Note |
|---|---|---|---|---|---|
| A100 | 108 | 2,048 | 1.41 GHz | 311.9 TFLOP/s | datasheet: 312 |
| H100 | 132 | 4,096 | 1.83 GHz | 989.4 TFLOP/s | datasheet: 989.4 dense = 1,979\* / 2 |
| T4 | 40 | 1,024 | 1.59 GHz | 65.1 TFLOP/s | fp16. Turing has no bf16. |
| L4 | 58 | 1,024 | 2.04 GHz | 121.2 TFLOP/s | 242\* / 2 |

The per-SM tensor rate doubled from Ampere to Hopper. Most of the jump from A100 to H100 comes from this
rate, not from the clock. Each time the precision becomes half as wide (bf16, then fp8, then fp4), the
rate doubles again on parts that support it.

### Caches are for reuse inside a kernel

The last-level cache is 50 MB of L2 on an H100, 48 MB on an L4, and 256 MB of Infinity Cache on an
MI300X. It serves re-reads of tiles *within* a kernel (§4). It cannot hold a model. A decode step streams
15 GB of Llama-3.1-8B weights once per step, three hundred times the H100's L2. Thus nothing stays in the
cache from one step to the next. Use HBM bandwidth for the decode plan, not the cache.

> **In a design review.** "I read the dense peak for our precision, the HBM bandwidth and capacity, the
> per-direction link bandwidth in bytes, and the power limit. The TFLOPS with an asterisk and the
> bidirectional link figures are both exactly 2× the value that a transfer or a dense GEMM gets."

---

## 2. The roofline model

The roofline of Williams, Waterman and Patterson (2009) puts two ceilings on a kernel. A kernel does $F$
FLOPs and moves $B$ bytes to and from memory. Its **arithmetic intensity** is $I = F/{B}$. In
`roofline.roofline`, the functions `attainable()`, `ridge_point()` and `time_kernel()` calculate these
three values, in this order:

$$
\begin{aligned}
\text{attainable FLOP/s} &= \min(\text{peak},\ I \times \text{bandwidth}) \\
\text{ridge point} &= \text{peak} \,/\, \text{bandwidth} \quad [\text{FLOP/byte}] \\
\text{kernel time} &\ge \max(F \,/\, \text{peak},\ B \,/\, \text{bandwidth})
\end{aligned}
$$

```
   FLOP/s
     │                     ridge (I = peak / BW)
peak ┤. . . . . . . . . . . ┌───────────────────────────  compute-bound: buy FLOPs,
     │                     ╱                              narrower precision
     │                   ╱
     │                 ╱   memory-bound: move fewer bytes
     │               ╱     (fuse, tile, batch, quantize)
     │             ╱  slope = bandwidth
     └───────────┴─────────┴──────────────────────────── I (FLOP/byte, log scale)
```

Worked ridges (dense, `roofline.specs.Device.ridge()`):

- H100 bf16 989.4 / 3.35 = **295 FLOP/B** (fp8: 591)
- L4 121 / 0.30 = 403 (fp8: 808)
- T4 fp16 65 / 0.32 = 203
- A100 80GB 153, H200 206, B200 281, MI300X 247

That list holds two lessons. First, compute-first generations increase the ridge (A100 153 → H100 295).
Memory refreshes decrease it again (H200 is an H100 with more bandwidth: 206). Second, a *narrower*
precision doubles the ridge. Thus fp8 gives a gain only at twice the intensity.

### Where kernels land

The byte counts in the table are **compulsory** traffic. In compulsory traffic, the kernel reads each
operand one time and writes each result one time. A perfectly fused and tiled kernel moves this traffic
(`roofline.roofline.elementwise()`, `reduction()`, `gemm()`):

| Kernel | FLOPs | Bytes | Intensity | On an H100 (bf16) |
|---|---|---|---|---|
| vector add `z = x + y`, bf16 | n | 3n × 2 | 1/6 = 0.17 | 0.56 TFLOP/s, 0.06% of peak |
| sum of n fp32 | n | 4n | 0.25 | 0.84 TFLOP/s |
| GEMV: 1 token × 4096×4096 weight | 2dk | ≈ dk × 2 | 1.0 | 3.35 TFLOP/s, 0.34% of peak |
| GEMM 64 × 4096 × 4096 | 2mnk | (mk + kn + mn) × 2 | 62 | 208 TFLOP/s, memory-bound |
| GEMM 4096 × 4096 × 4096 | 2n³ | 3n² × 2 | 1,365 | 989 TFLOP/s, compute-bound |

Remember the GEMM formula:

$$
\text{GEMM intensity} = \frac{2mnk}{(mk + kn + mn) \cdot b} \qquad \text{square } n\text{:} \quad \frac{2n}{3b}
$$

A square bf16 GEMM is compute-bound on an H100 when 2n/6 ≥ 295, that is, when n ≥ 887. A GEMM with m = 1
(one token through a weight matrix) has intensity ≈ 2/b = 1 FLOP/B at bf16, whatever its size. An 8192³
GEMM does 1.1 × 10¹² FLOPs and cannot finish in less than 1.11 ms on an H100.

### What the roofline leaves out

The roofline is a bound, not a prediction. Real kernels do not reach either ceiling, for these reasons:

- The sustained clock is below the boost clock.
- A kernel rarely overlaps loads and math perfectly.
- The last wave of thread blocks leaves SMs idle (wave quantization).
- In small kernels, launch latency is most of the time. For example, a 128³ GEMM is 4 MFLOP and 98 KB: 4 ns
  of math at peak, 29 ns on the roofline, microseconds in practice.

The model also assumes compulsory traffic, and §4 shows how far a real tiling is from that. Use the model
to decide *which* ceiling to attack. Then measure how near you get (`gpu-bench-lab` fits a measured
roofline to your device).

---

## 3. LLM inference on the roofline

### 3.1 The FLOPs and bytes of one step

One engine step runs every layer one time over the tokens that the engine schedules in it. For a
prefill, these tokens are the whole prompt. For decode, they are one token per sequence.
`roofline.llm.ModelConfig` counts these values from the `config.json` of a model (see the
[transformer primer §8](../../00-foundations/transformers/docs/transformer-primer.md#82-reading-a-model-config)):

$$
\begin{aligned}
P &= L \times (\text{attention} + \text{MLP}) \\
&\qquad \text{per-token matmul params (MoE: } \mathrm{top\_k} \text{ experts, not all)} \\
\text{FLOPs(step)} &= 2 \times \text{tokens} \times P \\
&\quad {}+ 2 \times \mathrm{rows\_of\_logits} \times \text{vocab} \times d \\
&\quad {}+ 4 \times L \times \text{heads} \times \mathrm{head\_dim} \times \text{positions attended} \\
\text{bytes(step)} &= \text{weights streamed once (MoE: only experts hit)} \\
&\quad {}+ \text{KV read} + \text{KV written} \\
\text{KV per token} &= 2\,(K,V) \times L \times \mathrm{kv\_heads} \times \mathrm{head\_dim} \times \text{bytes}
\end{aligned}
$$

For Llama-3.1-8B, the values are:

- 8.03 B parameters,
- 6.98 B per-token matmul parameters,
- a 525 M-parameter LM head,
- in bf16, 131,072 B (128 KiB) of KV per token.

A decode step streams **15.0 GB**: every layer plus the LM head. The input embedding is a gather of one
row per token, not a stream of the table. The model assumes that activations stay on chip (fused), and
efficiencies default to 1. Thus every time that follows is an ideal lower bound (`roofline.llm.prefill()`,
`decode()`).

### 3.2 Prefill: intensity ≈ tokens in the step

The step reads the weights one time for the whole prompt, but the FLOPs grow with the prompt. Thus, at
2-byte weights, the intensity is near the token count. For Llama-3.1-8B on an H100, the bound flips
between 300 and 320 tokens. This is exactly at the ridge, measured in tokens:

Prefill, 2,048 tokens, H100, bf16:

- FLOPs = 2.97 × 10¹³, bytes = 15.3 GB, intensity = 1,941 FLOP/B. Thus the step is compute-bound.
- compute time 30.0 ms, memory time 4.57 ms. Thus the ideal TTFT ≈ 30 ms (L4: 245 ms).

This is why engines schedule prefill in chunks of hundreds to thousands of tokens (chunked prefill, layer
04). A chunk below ~300 tokens leaves the tensor cores of an H100 idle. A chunk far above that size only
adds latency. It is also why TTFT grows with prompt length. TTFT grows linearly at first, and faster for
long prompts, where the quadratic attention term grows (8K tokens: 133 ms, 4.4× the 2K figure). The
version of this calculation for capacity plans (MFU, queues, TTFT budgets) is in
[capacity planning, formula 4](../../00-foundations/gpu-capacity-planning/PRIMER.md).

### 3.3 Decode: batch 1 is a weight stream

At batch 1, each token re-reads 15 GB for ~16 GFLOPs. The intensity is ≈ 1. The time per token is
bytes ÷ bandwidth, whatever the FLOPs on the datasheet. At 1K context, it is **4.52 ms on an H100
(221 tokens/s), 50.5 ms on an L4 (19.8 tokens/s)**. Batching shares the weight read across sequences. But
each sequence also reads its own KV cache:

| Batch (2K context, H100, bf16) | Intensity | Step time | Aggregate tok/s | Per user tok/s | KV share of bytes |
|---|---|---|---|---|---|
| 1 | 1.1 | 4.56 ms | 219 | 219 | 2% |
| 8 | 7.5 | 5.12 ms | 1,562 | 195 | 13% |
| 32 | 21.8 | 7.05 ms | 4,542 | 142 | 36% |
| 64 | 32.0 | 9.61 ms | 6,659 | 104 | 53% |
| 128 | 41.7 | 14.74 ms | 8,682 | 68 | 70% |
| 208 (HBM limit) | 47.2 | 21.16 ms | 9,832 | 47 | 79% |

Throughput increases with batch, and the speed per user decreases. This is the trade that you adjust
against an ITL SLO. The table stops where HBM does: 208 sequences of 2K tokens fit beside the weights with
10% headroom (`max_batch_by_memory()`). At 4K context and a 20 ms ITL, the largest batch is 96
(`best_batch_under_itl()`). HBM capacity allows 104.

### 3.4 KV reads cap decode intensity

When the batch grows, more sequences share the weight read, and its cost per sequence goes toward zero.
But the KV read is per sequence. It grows with context exactly as fast as the attention FLOPs of that
sequence. The limit (`decode_intensity_limit()`):

$$
\begin{aligned}
\text{decode intensity as batch} \to \infty &= \frac{\text{FLOPs per token}}{\text{KV bytes per token}} \\
&\approx \frac{2 \times (\text{heads} \,/\, \mathrm{kv\_heads})}{\mathrm{kv\_bytes}} \quad \text{at long context} \\
&\phantom{\approx}\ \text{(GQA group of 4: 4 FLOP/B)}
\end{aligned}
$$

```
Llama-3.1-8B, bf16 KV:   1K context → 115.7 FLOP/B    4K → 32.0    32K → 7.5
```

Against an H100 ridge of 295, no batch size makes a 4K-context decode *step* compute-bound as a whole.
Treat the step as one kernel (`decode()`). Then its crossover batch exists only at short contexts: 297
at c = 0, 440 at c = 128, 854 at c = 256. It never exists beyond ~392 tokens
(`max_context_for_compute_bound()`). Also, HBM becomes full first (104 sequences of 4K tokens fit beside
the weights with 10% headroom).

**Per kernel, not per step.** That average mixes two kernels with opposite shapes. The weight GEMMs
multiply a $[\text{batch} \times d]$ activation by every weight matrix. Their intensity is
≈ batch × 2 / weight bytes at any context. They are compute-bound from batch 296
(`gemm_crossover_batch()`; FP8 halves the bytes and doubles the peak, so also 296). For them, the rule
"decode turns compute-bound around batch ≈ ridge" in
[capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) is exactly correct.

Attention reads the KV cache of each sequence at 2 × (heads / kv_heads) / kv_bytes = 4 FLOP/B (bf16 KV),
whatever the batch. Thus attention is never compute-bound.

An engine runs the kernels one after the other. Thus the tighter bound is the **sum of per-kernel
roofline times** (`decode_split()`). It is not $\max(\Sigma F \div \text{peak},\ \Sigma B \div \text{BW})$,
which assumes that GEMM math overlaps KV streaming perfectly. The two bounds agree while both kernels are
memory-bound, as in every decode row in §3.3–3.6 and §8. They become different when the GEMMs cross the
ridge:

```
Llama-3.1-8B, H100, FP8 weights and KV, 2K context, batch 400 (476 fit):
  one kernel    18.27 ms, memory-bound
  per kernel    GEMMs 3.03 ms (compute-bound) + attention 16.03 ms (memory-bound) = 19.07 ms
```

After the GEMM crossover, throughput increases no more: 20,978 tokens/s here at any larger batch. Each
extra sequence only adds KV time. The KV-cache techniques in layer 04 work against this
([KV cache](../../04-inference-engine/kv-cache/kv-cache-primer.md),
[PagedAttention](../../04-inference-engine/paged-attention/paged-attention-primer.md)). GQA/MQA/MLA and FP8
KV decrease the bytes. Paging and prefix caching stop the waste of bytes. It is also the argument for
attention–FFN disaggregation: the two kernels on separate GPU pools, each at its own batch.

### 3.5 Quantization moves bytes

In the memory-bound regime, the speedup of a step is old bytes ÷ new bytes. The schemes differ in *which*
bytes they cut (Llama-3.1-8B, H100, 2K context, `roofline.llm.SCHEMES`):

| Scheme | Weight bytes | KV bytes | Math | Batch 1 | Speedup | Batch 64 | Speedup | Compute-bound from (c = 0) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 2 | 2 | bf16 | 4.56 ms | 1.00× | 9.61 ms | 1.00× | batch 297 |
| FP8 (W8A8 + KV8) | 1 | 1 | fp8 | 2.28 ms | 2.00× | 4.81 ms | 2.00× | 297 |
| W8A16 | 1 | 2 | bf16 | 2.32 ms | 1.97× | 7.37 ms | 1.30× | 149 |
| W4A16 (INT4, AWQ/GPTQ) | 0.5 | 2 | bf16 | 1.20 ms | 3.80× | 6.25 ms | 1.54× | 75 |
| W4A16 + FP8 KV | 0.5 | 1 | bf16 | 1.16 ms | 3.93× | 3.69 ms | 2.61× | 74 |

FP8 halves every byte and doubles the peak. Thus it is exactly 2× in either regime. Weight-only INT4 is
nearly 4× at batch 1 but only 1.5× at batch 64. The reason is that half the bytes at batch 64 are KV
cache, which INT4 did not touch. INT4 also decreases the compute-bound crossover to ~75, because the math
still runs at the bf16 peak.

Quantization also gives **capacity**: FP8 weights and KV let 476 sequences of 2K tokens fit on the H100
instead of 208. (INT4 group scales add ~2/`group_size` bytes per weight. The numbers here ignore them.)

### 3.6 MoE: which weights a step streams

The *total* parameters set the memory size, because every expert lives in HBM. But the FLOPs of a token
follow the *active* parameters. A decode step streams only the experts to which the router sends its
tokens. With uniform routing, one layer touches $E\,(1 - (1 - k/E)^{T})$ distinct experts for $T$ tokens
(`experts_touched()`):

| Batch (1K context) | Mixtral-8x7B (8 experts, top-2) | Step bytes on H200 | Step time | Qwen3-30B-A3B (128, top-8) |
|---|---|---|---|---|
| 1 | 2.00 experts/layer | 25.6 GB | 5.34 ms | 8.0 / 128 |
| 4 | 5.47 | 65.1 GB | 13.57 ms | 29.1 |
| 16 | 7.92 | 94.4 GB | 19.66 ms | 82.4 |
| 64 | 8.00 | 101.7 GB | 21.20 ms | 125.9 |

Mixtral (46.7 B total, 12.9 B active) decodes like a 13 B model only at batch 1. At batch 16, it streams
nearly all 93 GB of its weights every step. Fine-grained MoE keeps this gain up to larger batches. At high
batch, the cost per token still decreases, because the batch shares the full stream.

The consequence for the roofline: FLOPs follow the *active* parameters, but the bytes approach the
*total* parameters. Each expert sees only $B \cdot k/E$ of the batch. Thus the batch at which the weight
stream reaches the ridge scales with total ÷ active. At c = 0 on an H200 (ridge 206;
`decode_crossover_batch()`): 207 for Llama-3.1-8B, 754 for Mixtral-8x7B (total/active 3.6) and 2,055 for
Qwen3-30B-A3B (9.1, or 9.9 without the input embedding). The step gathers the input embedding. It does
not stream it or multiply it.

Thus serving large MoE models is about large batches. Expert parallelism is how systems get them.
Attention runs data-parallel on many GPUs, and all their tokens meet at each expert. For the calculation
of size, see [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md). For the all-to-all
that this causes, see §5.

[00-foundations/mixture-of-experts](../../00-foundations/mixture-of-experts/PRIMER.md) (§2–3, §5–6)
explains the router and the experts themselves. It tells how the router routes tokens, why a router must
balance the load, and how the dispatch and combine of expert parallelism work.

---

## 4. The memory hierarchy and why tiling/fusion win

Each level down the hierarchy is larger and slower (for approximate H100 bandwidths and latencies, see
[gpu-primer §3](../gpu-primer/gpu-primer.md#3-memory-is-the-whole-game)):

| Level | Size and speed | Holds |
|---|---|---|
| registers | 256 KB per SM (33 MB total) | accumulators of the tile that the kernel computes |
| SMEM / L1 | up to 228 KB per SM | operand tiles, staged by TMA / async copy |
| L2 | 50 MB, shared | re-reads of panels by nearby thread blocks |
| HBM | 80 GB at 3.35 TB/s | everything else: weights, KV cache, activations |
| host DRAM | over PCIe Gen5 (63 GB/s/dir) | offload, load, swap |

Every level has its own roofline. To keep the tensor cores busy, a kernel must reuse each byte that it
fetches from level X approximately peak ÷ bandwidth(X) times. For HBM, that is 295 on an H100. Tiling is
how a kernel makes that reuse.

### 4.1 Tiling

A GEMM computes each $T_m \times T_n$ output tile. To do this, it streams a $T_m \times k$ panel of $A$
and a $k \times T_n$ panel of $B$ through on-chip memory. Each $k$-step loads $(T_m + T_n)$ elements and
does $2\,T_m T_n$ FLOPs (`roofline.roofline.tile_intensity()`):

$$
\text{tile intensity} = \frac{2\,T_m T_n}{(T_m + T_n) \cdot b}
$$

In bf16: 16×16 → 8, 64×64 → 32, 128×128 → 64, 128×256 → 85, 256×256 → 128 FLOP/B.

Larger tiles give more reuse, but they must fit. Take a 128 × 256 tile with 64-wide $k$-slices in bf16 and
four pipeline stages. It needs 4 × (128 + 256) × 64 × 2 = 192 KiB of shared memory (`tile_smem_bytes()`).
Its fp32 accumulators need 128 × 256 × 4 = 128 KiB of registers. These are 84% of an H100 SM's shared
memory and half its register file. That is why production GEMM tiles stop at approximately this size.

Even then, 85 FLOP/B is below the HBM ridge of 295. Assume that every tile re-reads its panels from HBM.
Then a 4096³ bf16 GEMM with 128 × 128 tiles will move 2.18 GB — 21.7× the compulsory 101 MB — at 63 FLOP/B
(`tiled_gemm_bytes()`). With no tiling at all (two loads per multiply-add), it will sit at 0.5 FLOP/B.

The rest of the reuse comes from the **L2**. Thread blocks that run at the same time share panels. Kernels
also order (swizzle) their tiles, and on Hopper they multicast them to clusters. They do this so that HBM sees
almost compulsory traffic. Tiling occurs at every level of the hierarchy, not at only one.

### 4.2 Fusion

Elementwise operations are at ~0.2 FLOP/B. A chain of unfused elementwise operations sends every
intermediate result to HBM and back (`fusion_bytes()`). Take $k$ ops on $n$ elements, where $e$ of the ops
read a second full tensor (a residual):

$$
\text{unfused: } (2k + e) \cdot n \cdot b \ \text{bytes} \qquad \text{fused: } (2 + e) \cdot n \cdot b
$$

Take 4 ops (bias, GELU, dropout, residual add) on a 4096 × 8192 bf16 activation, with $e$ = 1. On an H100:

- unfused 604 MB → 180 µs,
- fused 201 MB → 60 µs.

Fusion changes no FLOPs and removes 67% of the bytes. The fused kernel still reads each distinct input one
time and writes the result one time.

FlashAttention is the standard example. It tiles Q, K and V into shared memory, computes softmax online, and
never writes the $N \times N$ score matrix. See the
[FlashAttention primer](../../04-inference-engine/flash-attention/flash-attention-primer.md). Its §3
calculates ~32 FLOP/B for the simple version.

Layer 02 explains how warps map onto sectors, banks and tiles: coalescing, shared memory bank conflicts,
occupancy ([`02-cuda-nccl-runtime`](../../02-cuda-nccl-runtime/README.md): `cuda-and-nccl/PRIMER.md`
§2–3).

---

## 5. Fabrics quantitatively

### 5.1 The link ladder

The table gives the theoretical bandwidth per direction (`roofline.fabric.LINKS`). The $\alpha$ values are
illustrative per-step latencies, with the software included, for use in the model. Measure your own
values with nccl-tests:

| Link | Scope | GB/s per direction | α used here |
|---|---|---|---|
| NVLink 5 (B200, GB200, GB300): 18 links × 50 GB/s | between GPUs in an 8-GPU node or a 72-GPU rack | 900 | 2 µs |
| NVLink 4 (H100, H200): 18 × 25 GB/s | between GPUs, 8 per node through NVSwitch | 450 | 2 µs |
| NVLink 3 (A100): 12 × 25 GB/s | between GPUs, 8 per node | 300 | 2 µs |
| PCIe Gen5 x16 (32 GT/s, 128b/130b) | between a GPU and a CPU / NIC / NVMe | 63 | 4 µs |
| PCIe Gen4 x16 | same, one generation older | 31.5 | 4 µs |
| InfiniBand XDR, 800 Gb/s NIC | between nodes, per NIC | 100 | 5 µs |
| InfiniBand NDR / RoCE, 400 Gb/s NIC | between nodes, per NIC | 50 | 5–6 µs |
| 100 GbE with TCP | commodity network | 12.5 | 25 µs |

NVSwitch makes the NVLink domain all-to-all: any GPU pair gets the full per-GPU bandwidth. The size of that
domain is the most important property (8 GPUs in an HGX node, 72 in an NVL72 rack). The
[deployment primer §5](../gpu-deployment/gpu-deployment-primer.md#5-the-interconnect-hierarchy) shows the
cliff between scale-up and scale-out.

### 5.2 The α-β model and the ring all-reduce

| Pattern | Time for $n$ bytes over $p$ GPUs | In `roofline.fabric` |
|---|---|---|
| point-to-point | $t = \alpha + n/\beta$ | `transfer_time()` |
| ring all-reduce | $t = 2(p-1)\,\alpha + 2\,(p-1)/p \cdot n/\beta$ | `ring_allreduce_time()` |
| all-gather / RS | $t = (p-1)\,\alpha + (p-1)/p \cdot n/\beta$ | `ring_allgather_time()` |
| recursive doubling | $t = \lceil \log_2 p \rceil \cdot (\alpha + n/\beta)$ | `recursive_doubling_allreduce_time()` |
| crossover | ring latency term = bandwidth term at $n = p\,\alpha\,\beta$ | `allreduce_crossover_bytes()` |

The bandwidth term of the ring, $\sim 2n/\beta$, is optimal and independent of $p$. Its latency term grows
with $p$. Recursive doubling has the opposite shape.

Take 8 H100s over NVLink 4 as a worked example. The two terms of the ring meet at **7.2 MB** (over a
400 Gb/s NIC: 2.0 MB). A 1 GiB all-reduce takes 4.20 ms — algbw 255 GB/s. For it, nccl-tests reports
**$\text{busbw} = \text{algbw} \times 2(p-1)/p$ = 447 GB/s**, the number to compare with the link's 450
(`busbw()`). Layer 02 covers busbw, NCCL algorithms and protocols in depth.

### 5.3 Tensor parallelism: the collective inside every layer

Megatron-style tensor parallelism divides the matrices of each layer across $p$ GPUs. It does an
all-reduce of a $[\text{tokens} \times d_{\text{model}}]$ activation two times per layer: after
attention's output projection and after the MLP's down projection (`tp_allreduces_per_step()`,
`tp_comm_time()`). For Llama-3.1-70B, that is **160 all-reduces per step**:

Decode, batch 1, TP=8 on H100 NVLink:

- message = 1 token × 8,192 × 2 B = 16 KiB (440× below the 7.2 MB crossover: latency-bound)
- weights = each GPU streams its shard in 5.20 ms
- comm/step = 160 × ring all-reduce = 4.49 ms (latency-optimal algorithm: 0.98 ms)

Prefill, 4,096 tokens, TP=8:

- message = 4,096 × 8,192 × 2 B = 67 MB (bandwidth-bound)
- compute = 73.6 ms per GPU
- comm/step = 46.2 ms over NVLink 4 (387.0 ms if every ring hop crossed a 400 Gb/s NIC: 8 GPUs in 8 nodes)

Prefill, 4,096 tokens, TP=16 across two 8-GPU nodes (`tp_comm_time_across_nodes()`):

- compute = 36.8 ms per GPU
- comm/step = 74.7 ms with rails (8 NICs per node, each GPU sends its n/8 share over its own NIC). It is
  262.6 ms with one NIC per node, and 426.7 ms as one flat ring through the NICs.

There are two lessons. First, at decode, the cost is the count of collectives times the latency of the
algorithm. This is why engines ship latency-optimized all-reduce kernels. It is also why TP's per-GPU
efficiency decreases when $p$ grows (batch-1 tokens per GPU-second: 23.3 at TP=2, 12.9 at TP=8 with a
ring).

Second, at prefill, the cost is bandwidth, and it does not decrease with $p$ (the bandwidth term of a ring
is $\sim 2n/\beta$, whatever the degree). But the compute of each GPU halves each time that $p$ doubles.
TP=8 already spends 46 ms communicating per 74 ms of compute.

Across nodes, NCCL spreads the traffic over every rail. Thus the NIC of each GPU carries approximately its
n/8 share. But TP=16 over two nodes still spends ~75 ms per 4K-token step in all-reduce against ~37 ms of
compute per GPU. The total is 111 ms per step against TP=8's 120 ms (no overlap), for 1.9× the
GPU-seconds.

Without a NIC per GPU, it is much worse (263 ms with one NIC per node, 427 ms as a flat ring). Beyond the
node, two things set the cost: its NIC aggregate (8 × 50 GB/s with rails) and a larger $\alpha$. These
links also carry data-, pipeline- and expert-parallel traffic.

Thus the rule from the
[deployment primer §4](../gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu):
tensor (and expert) parallelism inside the NVLink domain, pipeline and data parallelism across it. The
all-to-all of expert parallelism has busbw factor $(p-1)/{p}$ and the same α-β shape.

### 5.4 Topology: rails, fat trees, oversubscription, bisection

Scale-out networks for GPU clusters are **rail-optimized fat trees**. Each node has one NIC per GPU. NIC
$i$ of every node in a group connects to the rail-$i$ leaf switch, and the leaves connect to the spines:

```
spines         S0          S1          S2          S3        every leaf has uplinks to every spine
                │ ╲      ╱ │ ╲      ╱ │ ╲      ╱ │
leaves      rail-0      rail-1       ...       rail-7       one leaf per rail, per group of 32 nodes
              │           │                       │
node 0      NIC0        NIC1         ...        NIC7        GPU i drives NIC i (over a PCIe switch)
node 1      NIC0        NIC1         ...        NIC7
 ...
node 31     NIC0        NIC1         ...        NIC7
```

GPUs with the same index in the group are one switch hop apart. Cross-rail traffic goes up to a spine
(three hops). This does not occur if NCCL's PXN first moves it over NVLink to the local GPU on the
destination's rail (`switch_hops()`). Rails are the reason that every GPU can drive its own NIC at the
same time. A hierarchical 1 GiB all-reduce over 4 nodes × 8 GPUs takes **8.3 ms with 8 NICs per node and 36.4 ms
with one** (`hierarchical_allreduce_time()`).

A two-tier folded Clos of radix-$R$ switches is **non-blocking** (1:1) when each leaf gives half its ports
to hosts and half to spines. Its maximum is $R^2/2$ endpoints (`leaf_spine()`). With radix-64 switches,
2,048 endpoints need 64 leaves and 32 spines.

With 48 ports down and 16 up, the network serves 3,072 endpoints with 64 leaves and 16 spines — at **3:1
oversubscription**. This divides the **bisection bandwidth** (the capacity across the worst half/half cut,
`bisection_gbs()`) from 76,800 GB/s to 25,600 GB/s. Independent inference replicas (mostly north-south
traffic) can accept oversubscription. Training all-reduces, expert all-to-all and disaggregated KV transfer
(layer 05) need bisection.

### 5.5 GPUDirect RDMA and staged copies

Without GPUDirect RDMA, the system copies a GPU-to-remote-GPU message to host memory, sends it, and copies
it up again. A copy through a chain of hops costs the sum (store-and-forward), or the slowest hop plus one
chunk per other hop (pipelined, `staged_transfer_time()`):

1 GiB over [PCIe Gen5 63, NDR 50, PCIe Gen5 63] GB/s:

- host-staged, store-and-forward 55.6 ms
- GPUDirect RDMA, pipelined (1 MiB) 21.5 ms ≈ bytes / the NIC

A carefully pipelined staged path can get near the same bandwidth for large messages, but not the same
latency. It also uses host memory bandwidth and CPU. For full speed, GPUDirect RDMA needs the NIC and the
GPU in the same PCIe switch tree. The topology matrix shows this.

### 5.6 NUMA and `nvidia-smi topo -m`

`nvidia-smi topo -m` prints the path between every GPU and NIC pair (`fabric.TOPO_LEGEND`). From best to
worst, the paths are:

- **NV#** (a bonded set of # NVLinks)
- **PIX** (at most one PCIe bridge: the same PCIe switch)
- **PXB** (several PCIe bridges, no host bridge: several switches under one host bridge)
- **PHB** (through a PCIe host bridge, typically the CPU)
- **NODE** (between host bridges within a NUMA node)
- **SYS** (across the inter-socket link, QPI/UPI)

Its CPU and NUMA affinity columns tell which socket each GPU connects to. Three rules:

1. Put tensor-parallel groups on NV# pairs, never across SYS.
2. Give each GPU the NIC in its own PCIe switch tree: PIX or PXB, never through the host bridge. Then
   GPUDirect RDMA skips the CPU.
3. Pin each GPU's process, its data loaders and its pinned host buffers to that GPU's NUMA node. If you do
   not, host-to-device copies cross the socket link.

On HGX H100 servers, each GPU's NIC shows PIX or PXB. The code changes with how the tray lays out its PCIe
switches (verify). `gpu-bench-lab` parses the matrix on your machine (`topo.py`) and measures P2P bandwidth
per path.

---

## 6. Storage and cold start

A checkpoint is `params × bytes per param` (`roofline.storage.checkpoint_bytes()`): Llama-3.1-8B is
16.1 GB in bf16, Llama-3.1-70B 141.1 GB. A new replica must fetch those bytes and copy them into HBM. The
slowest tier on the path sets the time:

$$
t_{\text{weights}} \approx \frac{\text{bytes}}{\min(\text{fetch bandwidth},\ \text{host} \to \text{device bandwidth} \times \text{GPUs})}
$$

The table uses **assumed** single-reader bandwidths (`storage.ASSUMED_GBS`). Replace them with
measurements from `gpu-bench-lab` notebook `04_weights_loading_and_cold_start`. With these values, 141 GB
takes:

| Source (assumed GB/s) | Time for 70B bf16 |
|---|---|
| one object-store or internet stream (0.1) | 1,411 s = 23.5 min |
| network block device (1.2) | 118 s |
| local NVMe (7) | 20.2 s |
| 128 parallel range streams, capped by a 100 Gb/s NIC (12.5) | 11.3 s |
| host page cache (20) | 7.1 s |
| host to 8 GPUs, pinned, PCIe Gen5 (8 × 50) | 0.35 s |

### 6.1 Parallel and streamed loading

One HTTP stream is slow. Many range reads add up until the NIC, the disk or the service limits them
(`parallel_gbs()`). When you stream chunks straight into GPU memory, fetch and copy overlap. Thus the time
goes toward the time of the slower hop, not the sum (`load_time(..., streamed=True)`). This is the idea
behind safetensors loaders that read directly into device memory. It is also the idea behind model
streamers such as the Run:ai Model Streamer load format in vLLM (verify).

Host-to-device copies from pinned memory run near the PCIe rate. But for pageable memory, the driver must
stage the data through a pinned bounce buffer. The assumptions here are 25 against 12 GB/s on PCIe Gen4.
Measure your own values with `gpu-bench-lab` notebook `02_memory_bandwidth_and_transfers`.

### 6.2 The anatomy of a cold start

```
scale-up ─► provision node ─► pull image ─► fetch weights ─► copy to GPU ─► engine init + warm-up ─► ready
            (warm pool?)      (streaming?)   (parallel? cache?) (pinned? streamed?) (CUDA graphs, compile, KV profiling)
```

The table assumes these stage times: provision 120 s, image 60 s, engine init 90 s (`cold_start()`):

| Scenario | Total | Largest stage |
|---|---|---|
| new node, one stream | 1,681 s | fetch weights, 84% |
| new node, parallel + streamed weights | 281 s | provision node, 43% |
| warm node, checkpoint in page cache | 97 s | engine init + warm-up, 93% |

The lesson is the order of attack. First, work on the weights (parallelism, streaming, a local or
regional cache, smaller precision). Then work on the node (warm pools, pre-pulled or streamed images,
layer 03). Then work on the engine (layer 04).

A budget sets the bandwidth that you need. Take a 70B replica that must be ready in 120 s on a warm node pool.
Assume a 20 s image pull and 60 s of engine init. This leaves 40 s for streamed weights: 141.1 GB / 40 s =
**3.53 GB/s** of fetch, i.e. 36 parallel streams at 0.1 GB/s. Cold start is also why autoscaling on LLM
replicas needs headroom and scale-ahead signals (layer 05).

---

## 7. Reliability at scale

### 7.1 Failure rates add

Assume independent failures at a constant rate. Then $N$ components with MTBF $M$ fail as a group every
$M/{N}$ (`roofline.reliability.cluster_mtbf()`). The chance of a clean run of $t$ hours is
$\exp(-N t / M)$ (`p_survive()`). The best public data point is the Llama 3 report (Meta, 2024): **419
unexpected interruptions in 54 days of pre-training on 16,384 H100s**. About 78% of them have hardware as
their cause, and GPU faults are the largest share. Calculate a per-GPU figure from these values
(`component_mtbf_from_observation()`):

$M_{\text{gpu}}$ = 16,384 × 1,296 h / 419 = 50,677 GPU-hours ≈ 5.8 years

```
   8 GPUs → an interruption every 6,335 h       P(clean 24 h) = 0.996
1,024 GPUs → every 49.5 h                         P(clean 24 h) = 0.616
16,384 GPUs → every 3.09 h (7.76 per day)
```

Each GPU lasts years, but the job does not last a shift. The figure combines GPUs, hosts, network and
software. Real fleets add infant mortality and correlated failures (a switch, a power feed). Silent data
corruption shows as incorrect numbers, not as crashes. Thus use the figure as a first-order rate for plans,
and monitor the health signals (DCGM, XID errors, layer 02).

### 7.2 How often to checkpoint: Young/Daly

Take a synchronous job that writes a checkpoint every $\tau$ seconds, and each checkpoint takes $\delta$
seconds. The job loses $\delta/\tau$ to the checkpoints. Per failure, it also loses about $\tau/2$ of
recomputed work plus a restart $R$ (`wasted_fraction()`):

$$
\begin{aligned}
\text{waste} &\approx \frac{\delta}{\tau} + \frac{\tau/2 + R}{M} \\
&\qquad \text{minimised at } \tau^{*} = \sqrt{2\,\delta M} \quad \text{(Young, 1974)} \\
\text{at } \tau^{*}\text{:}\quad \text{waste} &= \sqrt{2\delta/M} + R/M
\end{aligned}
$$

Daly (2006) adds higher-order terms: `young_daly_interval(..., higher_order=True)`.

```
restart cost R = 0 (ignored) — 16,384 GPUs (M = 3.09 h = 11,135 s):
   δ = 60 s → τ* = 19.3 min (Daly 18.6), waste 10.4%    (checkpointing hourly: 17.8%)
   δ = 10 s → τ* =  7.9 min,             waste  4.2%    (hourly: 16.4%)
4,096 GPUs, δ = 30 s → τ* = 27.2 min, waste 3.7%

with an assumed restart R = 10 min (reschedule, reload, re-initialise): R/M = 5.4% more
   δ = 60 s → waste 15.8%        δ = 10 s → waste 9.6%
```

Waste scales as $\sqrt{\delta}$. Thus a 4× faster checkpoint halves that part. This is the case for
asynchronous checkpoints to host memory and local disk, then to storage in the background. Restart time
adds $R/{M}$ on top, and it does not move $\tau^{*}$. At this failure rate, a 10-minute restart (5.4%)
costs more than everything a 10 s checkpoint wastes (4.2%). Thus fast restart is a lever next to fast
checkpoints: hot spare nodes, checkpoints replicated in peer memory, pre-initialised jobs.

### 7.3 Inference: replicas are failure domains

Serving replicas fail independently. Thus the question becomes "how many spares?". A TP=8 replica is down
when any of its GPUs is down: replica MTBF = M/8 = 6,335 h. With a 48 h MTTR (assumed: detect, drain,
repair), the availability is $a = \text{MTBF}/(\text{MTBF} + \text{MTTR})$ = 0.99248
(`replica_availability()`). If you need 8 replicas up (`p_at_least()`, `replicas_for()`):

- deploy 8: P(≥ 8 up) = 0.941
- deploy 9 → 0.998
- deploy 10 → 0.99995. Thus 99.9% needs 10 replicas = 80 GPUs for 64 GPUs of capacity (16 spare).
- the same capacity as 16 × TP=4 (FP8, so the model fits 4 GPUs) → 18 replicas = 72 GPUs (8 spare)
- the same capacity as 64 × single-GPU replicas (a smaller model) → 66 replicas = 66 GPUs (2 spare)

Larger replicas are larger blast radii: the spare capacity that you carry grows with the replica size.
This is one more reason to use the smallest TP degree that meets the latency SLO (§5.3). It is also why
training and inference clusters have different shapes
([deployment primer §9](../gpu-deployment/gpu-deployment-primer.md#9-training-clusters-vs-inference-clusters)).

---

## 8. The cost of a token

### 8.1 From $/GPU-hour to $/M tokens

The conversion, as `cost_per_million_tokens()` computes it:

$$
\text{\$/M tokens} = \frac{\text{\$/GPU-hr} \times \text{GPUs}}{\text{tokens/s} \times 3600 \times \text{utilisation}} \times 10^{6}
$$

The table takes decode throughput from §3. This throughput is an upper bound, so these costs are lower
bounds. The table takes GCP list prices from the research snapshot: L4 ~$0.70/hr, H100 ~$11/GPU-hr on
demand, ~$3.7/GPU-hr Spot, us-central1, September 2026 (verify). The current prices are in
[`COMPUTE.md`](../../COMPUTE.md). The model is Llama-3.1-8B at 2K context. One rule sets every batch: the
largest batch that meets an **ITL of 10 ms** (100 tokens/s per user) and fits in HBM
(`best_batch_under_itl()`):

| Option | Batch | ITL | tok/s | Per user | $/M output tokens at 100% | at 60% |
|---|---|---|---|---|---|---|
| H100 on demand, bf16 | 68 (ITL-bound) | 9.93 ms | 6,847 | 101 | $0.446 | $0.744 |
| H100 Spot, bf16 | 68 (ITL-bound) | 9.93 ms | 6,847 | 101 | $0.150 | $0.250 |
| H100 on demand, FP8 | 193 (ITL-bound) | 9.98 ms | 19,345 | 100 | $0.158 | $0.263 |
| L4 on demand, bf16 — misses the SLO | 20 (HBM-bound) | 67.9 ms | 294 | 14.7 | $0.660 | $1.10 |

The L4 cannot meet the SLO at any batch, because at batch 1 it needs 50.9 ms to stream the weights. Thus
its row is at its HBM limit, which is a slower product. Even so, the H100 gives the lower-cost token. Its
bandwidth and capacity together deliver 23× the L4's tokens/s for 15.7× the price.

FP8 then decreases the H100's cost by another 2.8×: exactly 2.0× from halving every byte, and 1.4× more
from the bigger batch (193 vs 68). The same ITL permits this larger batch.

Prefill tokens have an even lower cost. The same H100 processes a 2K prompt at an ideal ~68,000 tokens/s,
an order of magnitude above decode. This is the root of the difference between the input and output
prices of hosted APIs. It is also why prefix caching (which skips prefill) is a cost lever for agents.

### 8.2 Utilisation

A fleet sized for its peak is idle off-peak: utilisation = mean load ÷ peak (`utilisation()`). A day at
20% / 100% / 60% of peak in thirds averages 60%. Thus every $/M in §8.1 is 1.67× higher. Utilisation is
the place where serving economics succeed or fail. These things control it:

- autoscaling and scale-to-zero (layers 03 and 05),
- batch traffic that fills the troughs,
- admission control (layer 06), which keeps the peak within its plan.

### 8.3 Rent or own

Ownership costs a constant sum per hour (depreciation plus fixed opex), plus energy while the hardware is
busy. Rental costs the hourly rate only for the hours that you use. Ownership costs less above this
utilisation (`owned_cost_per_hour()`, `breakeven_utilisation()`):

$$
u^{*} = \frac{\text{fixed per hour}}{\text{rent per hour} - \text{energy per hour}}
$$

An 8 × H100 server has these values:

- $300k over 4 years + $30k/yr opex (assumed),
- 10.2 kW at full load (verify),
- PUE 1.3,
- $0.10/kWh.

These give $1.50 per GPU-hour fixed + $0.166 per busy GPU-hour energy. The break-even utilisation is:

- against on demand at $11/GPU-hr: 13.8%,
- against Spot at $3.7: 42.4%,
- against $2.0 rental: 81.7%.

The general rule "own above ~60–70% utilisation" in the
[deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)
is the break-even against low-cost rental. Against on-demand list prices, ownership costs less from a much
lower utilisation, if you can get the hardware, the power and the people. The same equation gives the price
of managed capacity. The Provisioned-Throughput break-even in the agentic scaling lab
([`01-scaling-primer.md` §3.5](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md#35-provisioned-throughput))
is the utilisation of committed units against pay-as-you-go. Agent workloads
([layer 07](../../07-application-agent-framework/README.md)) set the tokens per task that multiply it.

---

## 9. The accelerator landscape (September 2026 snapshot)

The table gives these values:

- dense peaks, in TFLOP/s,
- memory as the vendor gives it,
- the ridge, computed at bf16 (fp16 for the T4),
- scale-up bandwidth as the vendor gives it, with the size of the domain in parentheses. NVLink and xGMI
  are bidirectional totals per GPU. TPU ICI is per chip, as published.

`roofline.specs.table()` generates the table. The tag applies to **every entry (verify)**.

| Device | Mem GB | TB/s | bf16/fp16 TF | fp8 TF | fp4 TF | Ridge FLOP/B | Scale-up GB/s (domain) | TDP W |
|---|---|---|---|---|---|---|---|---|
| NVIDIA T4 | 16 | 0.32 | 65 | – | – | 203 | PCIe only | 70 |
| NVIDIA L4 | 24 | 0.3 | 121 | 242 | – | 403 | PCIe only | 72 |
| NVIDIA A100 SXM 80GB | 80 | 2.039 | 312 | – | – | 153 | 600 (8) | 400 |
| NVIDIA H100 SXM | 80 | 3.35 | 989 | 1,979 | – | 295 | 900 (8) | 700 |
| NVIDIA H200 SXM | 141 | 4.8 | 989 | 1,979 | – | 206 | 900 (8) | 700 |
| NVIDIA B200 (HGX) | 180 | 8 | 2,250 | 4,500 | 9,000 | 281 | 1,800 (8) | 1,000 |
| NVIDIA GB200 (per GPU, NVL72) | 186 | 8 | 2,500 | 5,000 | 10,000 | 312 | 1,800 (72) | 1,200 |
| NVIDIA GB300 (per GPU, NVL72) | 288 | 8 | 2,500 | 5,000 | 15,000 | 312 | 1,800 (72) | 1,400 |
| NVIDIA RTX PRO 6000 Blackwell Server | 96 | 1.6 | 500 | 1,000 | 2,000 | 312 | PCIe only | 600 |
| AMD Instinct MI300X | 192 | 5.3 | 1,307 | 2,615 | – | 247 | 896 (8) | 750 |
| AMD Instinct MI325X | 256 | 6 | 1,307 | 2,615 | – | 218 | 896 (8) | 1,000 |
| AMD Instinct MI355X | 288 | 8 | 2,500 | 5,000 | 10,000 | 312 | 1,075 (8) | 1,400 |
| Google TPU v5e | 16 | 0.819 | 197 | – | – | 241 | 200 (256) | – |
| Google TPU v6e (Trillium) | 32 | 1.64 | 918 | – | – | 560 | 448 (256) | – |
| Google TPU7x (Ironwood) | 192 | 7.37 | 2,307 | 4,614 | – | 313 | 1,200 (9216) | – |

Derived entries:

- The table divides the 4 PFLOPS sparse-FP4 headline of the RTX PRO 6000 by two, step by step, to get its
  tensor rates.
- The table assumes that Ironwood's bf16 is half its 4,614 TFLOPS fp8.
- B200 is the HGX board (192 GB physical, 180 GB usable).
- FP4 on the MI355X is MXFP4.

The T4 has no bf16, tf32 or fp8 paths. NVIDIA Vera Rubin and AMD's MI400 series are not in the catalogue.
They are at the stage of announcement or ramp-up. See the
[deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)
(verify).

How to read the table for inference:

- **Bandwidth sets decode speed per replica** (§3.3): 0.3 TB/s (L4) to 8 TB/s, a 27× range. Compare parts
  on $/M tokens (§8), not on $/hour or TFLOPS.
- **Capacity sets what fits and how big the batch can be** (§3.4–3.5): 16 GB (T4, TPU v5e) to 288 GB
  (GB300, MI355X). An H200 is an H100 plus capacity and bandwidth. For decode, that is most of what is
  important.
- **The ridge sets how much batching the FLOPs need**: from 153 (A100 80GB) to 560 (TPU v6e). A high ridge
  is good for prefill and training, but it gives no benefit to small-batch decode.
- **The scale-up domain sets how far TP and EP can go without the NIC** (§5.3). The domain is 1 for PCIe
  cards, 8 for HGX and MI3xx platforms, 72 for NVL72 racks, and 256 to 9,216 chips over TPU ICI.
- **Precision support is a generation marker**: fp8 from Ada/Hopper/MI300, fp4 from Blackwell/MI355X.
- **Power has gone from 70 W to 1,400 W per accelerator**: facilities, not procurement, are the limit for
  owned deployments
  ([deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)).

---

## 10. Getting hardware

You can learn every concept in this topic at T0. You use hardware to measure the concepts. The table shows
what each tier gives you here:

| Tier | Where | What you can see | Cost (verify) (see [`COMPUTE.md`](../../COMPUTE.md)) |
|---|---|---|---|
| T0 | laptop, Colab CPU, CI | the core notebooks, and your CPU's own roofline and disk throughput with `gpu-bench-lab` (numpy backend) | $0 |
| T1 | Colab or Kaggle T4 (free, not guaranteed), a rented 24 GB GPU, GCP L4 Spot or Cloud Run L4 | a real GPU roofline (GEMM sweep by dtype), HBM bandwidth, pinned against pageable host copies, weight loads | free to ~$0.7/hr |
| T2 | Kaggle 2×T4 (free, PCIe only, no NVLink), RunPod / Vast / Lambda 2–8× A100/H100 SXM for an hour, GCP `a2-highgpu-2g` | P2P bandwidth over PCIe against NVLink, `nvidia-smi topo -m` on real topologies, all-reduce busbw | $0 to ~$25 per session |
| T3 | GCP via Terraform (`gpu-bench-lab/deploy/gcp/terraform`) | the same suite on a Spot L4 VM with results in a bucket. Change the machine type for NVLink. | pay per use, Spot, auto-stop |

### 10.1 On Google Cloud

GPU families (September 2026, verify):

- **G2** (L4, for example `g2-standard-4` = 1 L4, ~$0.70/hr on demand),
- **N1 + T4**,
- **A2** (A100 40 GB `a2-highgpu-*`, 80 GB `a2-ultragpu-*`),
- **A3** (H100). On demand, it is available only as `a3-highgpu-8g`, ~$88/hr. Smaller A3 shapes come
  through Spot or flex-start. **A3 Mega** adds GPUDirect-TCPXO to the network.
- **A3 Ultra** (H200) and **A4** (B200, NVLink 1.8 TB/s per GPU), **A4X** (GB200 NVL72) and **A4X Max**
  (GB300 NVL72). These use GPUDirect RDMA over ConnectX NICs on a rail-aligned network. The NICs are
  ConnectX-7 on A3 Ultra and A4, and probably ConnectX-8 on A4X Max (for each machine type, verify).
- **G4** (RTX PRO 6000 Blackwell, 96 GB).

**Cloud Run** has L4 and RTX PRO 6000 GPUs, with per-second billing and scale to zero. **TPUs**: v5e and
v6e are available as Cloud TPU VMs or through GKE. TPU7x "Ironwood" (GA April 2026) is available through
GKE (verify).

The ability to get the capacity is as important as the price. **On-demand** is the simplest, and it is the
most difficult to get for large parts. **Spot** is 60–91% lower in cost, and the provider can preempt it
at any time. That is acceptable for benchmarks and for stateless replicas with headroom. **Dynamic
Workload Scheduler flex-start** puts a request in a queue and provisions all of it at the same time (up to
7 days). You use it through Kueue ProvisioningRequest (layer 03).

In **calendar mode**, you reserve a future block. With **reservations**, you hold capacity that you pay
for, if you use it or not.

Two account facts control everything. First, you cannot use GPUs on a Free Trial billing account. Second,
GPU quota (`GPUS_ALL_REGIONS` plus per-type regional quota) often starts at 0. Request it early. Requests
for T4 and L4 usually get a fast approval. Requests for A100 and H100 rarely get approval for a new
individual account.

### 10.2 Elsewhere

**Colab** gives a free T4 (16 GB, approximately 15–30 GPU-hours a week, not guaranteed). **Kaggle** gives
2×T4 or a P100 (30 GPU-hours a week). Kaggle is the free two-GPU box, PCIe only. Together they cover T1
and most of T2.

**Vast.ai** (marketplace) and **RunPod** (per-second) rent single GPUs and multi-GPU NVLink boxes as
containers. They give no driver or kernel control, and no Kubernetes. **Lambda** rents full VMs.
**Modal** runs serverless Python on GPUs, with monthly free credits. A local kind cluster serves the
Kubernetes layers (03, 05) without GPUs.

Prices and quotas change monthly. The maintained list is [`COMPUTE.md`](../../COMPUTE.md). The learning
order is [`CURRICULUM.md`](../../CURRICULUM.md).

---

## In a design review

**The two-minute walkthrough.** "I start from four datasheet numbers: the dense peak at our precision, HBM
bandwidth and capacity, and the per-direction link bandwidth. I also use one ratio, the ridge: 295 FLOP per
byte on an H100. An LLM step streams the weights one time, so its intensity is approximately its token
count. Prefill of a 2K prompt is compute-bound: TTFT is FLOPs over peak, ~30 ms ideal for an 8B model.
Decode is memory-bound: time per token is bytes over bandwidth, 4.5 ms at batch 1.

"Batching shares the weight read across the batch, and the GEMMs reach the ridge near batch 300. But
attention reads the KV cache of each sequence at a few FLOP per byte. Thus at 4K context, the whole step
averages at most 32 FLOP/B, and HBM capacity limits the batch first. Because of this, my levers are
bytes: FP8 weights and KV, GQA, paging, prefix caching.

"Across GPUs, I use α-β. TP makes 160 all-reduces per step for a 70B model. They are latency-bound at
decode and bandwidth-bound at prefill. That cost does not decrease when TP grows, but the compute of each
GPU does. Beyond the node, the share of each GPU goes over a NIC 9× slower than NVLink. Thus TP stays in
the NVLink domain.

"Then I look at the fleet. Cold start is bytes over the slowest tier. Failure rates add, so large jobs
write a checkpoint every $\sqrt{2\delta M}$, and serving carries spare replicas. Last,
$/M tokens is $/GPU-hr over tokens/s × utilisation."

**Drill.**

1. *Our H100s' tensor cores are mostly idle during decode. Is it a good decision to buy B200s for their
   FLOPs?* Decode is memory-bound, so idle tensor cores are normal. The B200's gain for decode is its
   bandwidth (8 vs 3.35 TB/s, 2.4×) and capacity (larger batch). It is not its 2.3× bf16 peak. Measure
   the achieved HBM bandwidth (DCGM's DRAM-active, not the "GPU utilisation" counter). Then compare
   $/M tokens.
2. *Why not TP=16 across two 8-GPU nodes for a model that does not fit in one?* Then communication becomes
   larger than compute. Here is the cost for a 70B model, even on rails with the traffic spread over
   every NIC. A 4K-token prefill step spends ~75 ms in all-reduce against ~37 ms of compute per GPU. Thus
   twice TP=8's GPUs decrease the step time by only a small fraction. Also, with one NIC per node it
   is 263 ms. Fit the model in one NVLink domain (FP8, an H200/B200 node, an NVL72 rack). Or use
   PP=2 × TP=8. It keeps the all-reduces on NVLink and sends one activation across the node per step.
3. *Will FP8 double decode throughput?* It does so only if every byte halves. FP8 weights *and* FP8 KV
   give exactly 2× in the memory-bound regime. They also approximately double the batch that fits.
   Weight-only 8-bit gives ~2× at batch 1 but 1.3× at batch 64 × 2K context, where KV is half the bytes.
4. *A new 70B replica takes 25 minutes to become ready. What do you change first?* Divide the time into
   its parts. One object-store stream at 0.1 GB/s is 23.5 of those minutes. First, use parallel range
   reads (NIC-bound at ~11 s), streaming into GPU memory, and a regional or local cache. Then use warm
   pools and pre-pulled images. Then change the engine init.
5. *How often does a 16K-GPU job need a checkpoint, and what decreases that cost?* At the Llama 3 failure
   rate, the job is interrupted every ~3.1 h. With a 60 s checkpoint, write one every
   √(2 × 60 × 11,135) s ≈ 19 min. This means losing ~10% plus R/M for restarts (~16% with a 10-minute
   restart). Asynchronous checkpoints (10 s) decrease the interval to ~8 min and the checkpoint waste to
   ~4%. Then the restart is the largest cost, so make it fast too.
6. *We need 8 TP=8 replicas up at 99.9%. How many do we deploy, and can we use fewer GPUs?* Deploy ten
   (80 GPUs) at a 48 h MTTR. Smaller failure domains need fewer spares. If FP8 lets the model fit on TP=4,
   18 replicas (72 GPUs) deliver the same capacity at the same target.

---

## Glossary

| Term | Meaning |
|---|---|
| Arithmetic intensity | FLOPs done per byte moved to or from a memory level |
| Roofline | attainable FLOP/s = min(peak, intensity × bandwidth) |
| Ridge point | peak ÷ bandwidth: the intensity at which a kernel is no longer memory-bound |
| Dense and sparse peak | headline tensor rates assume 2:4 structured sparsity and are 2× the dense rate |
| Compulsory traffic | bytes that a kernel must move if it reads every operand one time and writes every result one time |
| Tile / tiling | to compute an output block from operand panels held on chip, so that the kernel reuses each fetched byte |
| Fusion | to run several operations in one kernel, so that intermediate results never touch HBM |
| TDP / TBP | the board power limit. Clocks throttle to stay inside it. |
| KV cache | cached attention keys and values: 2 × layers × kv_heads × head_dim × bytes per token |
| W8A8 / W4A16 | weight bits / activation bits. Weight-only schemes dequantize to bf16 for the math. |
| MoE, top-k | mixture of experts. The router sends each token to $k$ of $E$ expert MLPs. |
| α-β model | transfer time = latency $\alpha$ + bytes ÷ bandwidth $\beta$ |
| algbw / busbw | nccl-tests: size ÷ time, and that value scaled by the collective's factor (2(p−1)/p for all-reduce) |
| Tensor parallelism (TP) | the division of each layer's matrices across GPUs, with two all-reduces per layer |
| NVLink domain | GPUs joined by NVLink/NVSwitch at full bandwidth: 8 per HGX node, 72 per NVL72 rack |
| Rail-optimized | NIC $i$ of every node connects to leaf switch $i$, so same-index GPUs are one hop apart |
| PXN | NCCL moves data over NVLink to the GPU on the destination's rail before the network hop |
| Oversubscription | a leaf's downlink bandwidth ÷ uplink bandwidth (1:1 = non-blocking) |
| Bisection bandwidth | capacity across the worst cut that divides the endpoints in half |
| GPUDirect RDMA | a NIC reads and writes GPU memory directly, and does not go through host memory |
| NUMA affinity | the CPU socket (and memory) that a GPU or NIC connects to |
| MTBF / MTTR | mean time between failures / to repair |
| Young/Daly interval | the checkpoint interval $\sqrt{2\,\delta M}$ (with Daly's corrections) that gives the least lost time |
| Utilisation | mean load ÷ provisioned capacity |
| Break-even utilisation | the utilisation above which ownership costs less than the rental of the same capacity |

---

## Sources

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore
  Architectures", *CACM* 52(4), 2009.
- NVIDIA datasheets: T4, L4, A100, H100, H200, HGX B200 / DGX B200, GB200 NVL72, GB300 NVL72, RTX PRO 6000
  Blackwell Server Edition. The *NVIDIA A100 Tensor Core GPU Architecture* (2020) and *NVIDIA H100 Tensor
  Core GPU Architecture* (2022) whitepapers: SM counts, per-SM tensor rates, cache sizes.
- AMD Instinct MI300X, MI325X and MI355X product pages and datasheets.
- Google Cloud TPU documentation (v5e, v6e, TPU7x) and GPU machine-family documentation (G2, A2, A3, A4,
  A4X, G4), and Cloud Run GPU documentation.
- R. Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023. It analyses inference cost and
  model partitions in the same spirit.
- J. Kaplan et al., "Scaling Laws for Neural Language Models", 2020. It is the source of the count of 2N
  FLOPs per token.
- M. Shoeybi et al., "Megatron-LM", 2019. It describes tensor parallelism with two all-reduces per layer.
- P. Patarasuk, X. Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations", *JPDC*
  69(2), 2009. R. Thakur, R. Rabenseifner, W. Gropp, "Optimization of Collective Communication Operations in
  MPICH", *IJHPCA* 19(1), 2005.
- NVIDIA nccl-tests, `doc/PERFORMANCE.md`: the definitions of algbw and busbw.
- M. Al-Fares, A. Loukissas, A. Vahdat, "A Scalable, Commodity Data Center Network Architecture",
  SIGCOMM 2008: fat trees from commodity switches. NVIDIA DGX SuperPOD reference architecture:
  rail-optimized scale-out.
- NVIDIA `nvidia-smi` documentation (`topo -m` legend) and GPUDirect RDMA documentation.
- J. W. Young, "A first order approximation to the optimum checkpoint interval", *CACM* 17(9), 1974.
  J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps", *FGCS* 22(3),
  2006.
- Llama Team, Meta, "The Llama 3 Herd of Models", arXiv:2407.21783, 2024. Its §3.3.4 covers the
  reliability of a 16K-GPU training run.
- W. Kwon et al., PagedAttention (SOSP 2023). T. Dao et al., FlashAttention (2022–2024). A. Agrawal et al.,
  Sarathi-Serve (OSDI 2024). The arithmetic of this layer gives the reasons for these engine techniques
  (layer 04).
- Model configurations from each model's published `config.json`: Llama-3.1-8B/70B, Mixtral-8x7B,
  Qwen2.5-1.5B, Qwen3-30B-A3B.

---

## Verify list

Product facts in this primer and in `roofline-core/roofline/specs.py`, as of September 2026:

- Every entry of the §9 table, and the status of the parts outside it (Vera Rubin, MI400 series). The
  values are: dense peaks per precision, memory capacity and bandwidth, scale-up bandwidth and domain size,
  TDP. The derived entries are especially important. These are the RTX PRO 6000 tensor rates (from its
  sparse FP4 headline) and TPU7x bf16 (assumed half of fp8). They are also B200 usable memory (180 of
  192 GB), GB200 memory (186 GB), GB300 TDP and MI355X xGMI bandwidth. Also what the TPU ICI
  figures mean.
- SM counts and boost clocks behind `peak_from_clock()`: T4 40 / 1.59 GHz, L4 58 / 2.04 GHz, A100 108 /
  1.41 GHz. For the H100 SXM, 132 / 1.83 GHz, which 989.4 TFLOP/s implies.
- H100 PCIe differences (SMs, HBM2e ~2 TB/s, 350 W). The MiB figure that `nvidia-smi` reports for an
  "80 GB" H100.
- Link rates: NVLink 3/4/5 per-GPU link counts and speeds, PCIe Gen4/Gen5 x16, InfiniBand NDR/XDR. The
  GPU–NIC `topo -m` code on HGX H100 servers (PIX or PXB). The $\alpha$ values are illustrative, not
  product facts.
- DGX H100 maximum system power (10.2 kW), which §8.3 uses. The $300k server price and opex are
  assumptions.
- GCP prices (L4 ~$0.70/hr, H100 ~$11/GPU-hr on demand as `a3-highgpu-8g` ≈ $88/hr, Spot ~$3.7/GPU-hr).
  The Spot discount range (60–91%). Machine families with their GPUs and NICs, and the availability of small
  A3 shapes. DWS flex-start (up to 7 days) and calendar mode. Cloud Run GPU types, the TPU7x GA date
  (2026-04-22), and quota behaviour for new accounts.
- Free and low-cost tiers: Colab (T4, hours per week), Kaggle (2×T4 or P100, 30 GPU-hours per week), and
  the Vast.ai, RunPod, Lambda and Modal offers. The maintained list is in [`COMPUTE.md`](../../COMPUTE.md).
- The Run:ai Model Streamer as a vLLM load format. The loader behaviour of safetensors.
- The Llama 3 interruption figures (419 unexpected in 54 days on 16,384 GPUs, ~78% hardware).
- Assumptions, not product facts (replace them with your own): $\alpha$ values, storage tier bandwidths,
  cold-start stage times. Also the 10-minute restart, the 48 h MTTR, the server price and opex.
