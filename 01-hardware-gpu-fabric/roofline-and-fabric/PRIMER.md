# Reading the machine: rooflines, memory hierarchy, fabrics, and the cost of a token

*A primer for layer 01. Snapshot: September 2026. Product facts are dated and marked (verify); every
formula has a worked number, and every computed number comes from a function in
[`roofline-core/`](roofline-core/) and is pinned by its tests.*

This primer turns a GPU datasheet and a network diagram into predictions you can defend: how long an
LLM step takes and which resource bounds it, what a collective costs on each kind of link, how long a
new replica takes to load, how often a large job fails, and what a token costs. It assumes the
qualitative picture from [`gpu-primer`](../gpu-primer/gpu-primer.md) (why a GPU is shaped the way it is)
and [`gpu-deployment`](../gpu-deployment/gpu-deployment-primer.md) (scale-up vs scale-out, the parallelism
menu), and it hands sizing questions (does it fit, TTFT/TPOT budgets) to
[`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md). Everything here is
learnable at tier T0 on a laptop; [`gpu-bench-lab/`](gpu-bench-lab/) measures the same
quantities on the hardware you have (T0 CPU, T1 one GPU, T2 several, T3 on GCP).

---

## The one-minute version

A GPU is a memory system with arithmetic attached. Every kernel pays **max(FLOPs ÷ peak, bytes ÷
bandwidth)**, and the ratio peak ÷ bandwidth — the **ridge point**, about 295 FLOP per byte for an H100
in bf16 — is the arithmetic intensity a kernel needs before the math units are the limit. An LLM step
streams every weight once however many tokens it carries, so its intensity is roughly **the number of
tokens in the step**: a 2K-token prefill is compute-bound (TTFT ≈ FLOPs ÷ peak), a batch-1 decode step
sits at ~1 FLOP/B (time per token ≈ bytes ÷ bandwidth), and batching lifts decode's weight GEMMs toward
the ridge while each sequence's **KV-cache reads**, which grow as fast as its attention FLOPs, keep
attention at a few FLOP/B and come to dominate the step. Quantization and MoE change **bytes**. Between
GPUs every transfer costs **α + n/β**: decode's tensor-parallel all-reduces are latency-bound, prefill's
are bandwidth-bound and do not shrink as the TP degree grows, and β per GPU falls ~9× from NVLink to a
400 Gb/s NIC — so tensor parallelism stays inside the NVLink domain. Then three fleet numbers follow from
the same arithmetic: cold start is **bytes ÷ the slowest tier**, failure rates **add** (checkpoint every
√(2 δ M)), and **$/M tokens = $/GPU-hr ÷ (tokens/s × 3600 × utilisation) × 10⁶**.

---

## 1. Spec-sheet literacy

A datasheet is a handful of numbers. Five of them carry almost every argument in this layer:

| Line on the sheet | What to extract | Trap |
|---|---|---|
| Tensor TFLOPS by precision | the **dense** peak for the precision you run | headline figures marked \* assume 2:4 sparsity: exactly 2× dense |
| FP32 TFLOPS | the non-tensor (CUDA-core) rate | an H100's 67 FP32 TFLOP/s vs 989 bf16 tensor: code that misses the tensor cores forgoes 14.8× of the chip's throughput |
| Memory GB and TB/s | capacity (what fits) and bandwidth (how fast decode runs) | capacity is binary: an "80 GB" H100 carries 80 GiB = 85.9 × 10⁹ bytes, of which the driver reports a little less (verify with `nvidia-smi`); the core plans with 80 × 10⁹, which is conservative |
| Interconnect GB/s | per-direction bandwidth | NVLink and PCIe are marketed as bidirectional totals: H100 "900 GB/s" is 450 each way; PCIe Gen5 x16 "128 GB/s" is 63 each way |
| Network | bytes, not bits | a 400 Gb/s NIC moves 50 GB/s |

Two more lines matter for planning. **TDP** (or TBP) is the board power the cooling and power delivery
must sustain — 72 W for an L4, 700 W for an H100 SXM, 1,400 W for a GB300 (verify) — and under sustained
tensor load clocks drop below boost to stay inside it, so a measured peak lands below the datasheet
(measure it: `gpu-bench-lab` notebook `01_measure_your_roofline`). **Form factor** changes the part: an
H100 PCIe card has fewer SMs, HBM2e at ~2 TB/s and a 350 W limit (verify), so "H100" alone is ambiguous —
always name SXM, PCIe or NVL.

### Where a peak comes from

A peak is units × work per clock × clock (`roofline.specs.peak_from_clock()`):

```
peak = SMs × dense tensor FLOP/clock/SM × boost clock

A100:  108 × 2,048 × 1.41 GHz = 311.9 TFLOP/s   (datasheet: 312)
H100:  132 × 4,096 × 1.83 GHz = 989.4 TFLOP/s   (datasheet: 989.4 dense = 1,979* / 2)
T4:     40 × 1,024 × 1.59 GHz =  65.1 TFLOP/s   (fp16; Turing has no bf16)
L4:     58 × 1,024 × 2.04 GHz = 121.2 TFLOP/s   (242* / 2)
```

The per-SM tensor rate doubled from Ampere to Hopper; that, not the clock, is most of the A100 → H100
jump. Every halving of precision (bf16 → fp8 → fp4) doubles the rate again on parts that support it.

### Caches are for reuse inside a kernel

The last-level cache — 50 MB of L2 on an H100, 48 MB on an L4, 256 MB of Infinity Cache on an MI300X —
serves re-reads of tiles *within* a kernel (§4). It cannot hold a model: a decode step streams 15 GB of
Llama-3.1-8B weights once per step, three hundred times the H100's L2, so nothing survives from one step
to the next. Plan decode on HBM bandwidth, not on cache.

> **In a design review** — "I read the dense peak for our precision, the HBM bandwidth and capacity, the
> per-direction link bandwidth in bytes, and the power limit. The asterisked TFLOPS and the bidirectional
> link figures are both exactly 2× what a transfer or a dense GEMM will see."

---

## 2. The roofline model

Williams, Waterman and Patterson's roofline (2009) bounds a kernel by two ceilings. A kernel does F FLOPs
and moves B bytes to and from memory; its **arithmetic intensity** is I = F / B.

```
attainable FLOP/s = min( peak ,  I × bandwidth )          roofline.roofline.attainable()
ridge point       = peak / bandwidth   [FLOP/byte]        roofline.roofline.ridge_point()
kernel time      >= max( F / peak ,  B / bandwidth )      roofline.roofline.time_kernel()

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

Worked ridges (dense, `roofline.specs.Device.ridge()`): H100 bf16 989.4 / 3.35 = **295 FLOP/B** (fp8: 591);
L4 121 / 0.30 = 403 (fp8: 808); T4 fp16 65 / 0.32 = 203; A100 80GB 153; H200 206; B200 281; MI300X 247. Two lessons
hide in that list. Compute-first generations push the ridge up (A100 153 → H100 295), and memory refreshes
pull it back down (H200 is an H100 with more bandwidth: 206). And a *narrower* precision doubles the ridge:
fp8 only pays off at twice the intensity.

### Where kernels land

Byte counts below are **compulsory** traffic — every operand read once, every result written once — which
is what a perfectly fused and tiled kernel moves (`roofline.roofline.elementwise()`, `reduction()`, `gemm()`):

| Kernel | FLOPs | Bytes | Intensity | On an H100 (bf16) |
|---|---|---|---|---|
| vector add `z = x + y`, bf16 | n | 3n × 2 | 1/6 = 0.17 | 0.56 TFLOP/s, 0.06% of peak |
| sum of n fp32 | n | 4n | 0.25 | 0.84 TFLOP/s |
| GEMV: 1 token × 4096×4096 weight | 2dk | ≈ dk × 2 | 1.0 | 3.35 TFLOP/s, 0.34% of peak |
| GEMM 64 × 4096 × 4096 | 2mnk | (mk + kn + mn) × 2 | 62 | 208 TFLOP/s, memory-bound |
| GEMM 4096 × 4096 × 4096 | 2n³ | 3n² × 2 | 1,365 | 989 TFLOP/s, compute-bound |

The GEMM formula is the one to remember:

```
GEMM intensity = 2mnk / ((mk + kn + mn) · b)           square n:  2n / (3b)
```

A square bf16 GEMM is compute-bound on an H100 once 2n/6 ≥ 295, i.e. n ≥ 887; a GEMM with m = 1 (one
token through a weight matrix) has intensity ≈ 2/b = 1 FLOP/B at bf16, whatever its size. An 8192³ GEMM
does 1.1 × 10¹² FLOPs and cannot finish in less than 1.11 ms on an H100.

### What the roofline leaves out

It is a bound, not a prediction. Real kernels miss both ceilings: sustained clocks sit below boost, a
kernel rarely overlaps loads and math perfectly, the last wave of thread blocks leaves SMs idle (wave
quantization), and tiny kernels are dominated by launch latency (a 128³ GEMM is 4 MFLOP and 98 KB: 4 ns
of math at peak, 29 ns on the roofline, microseconds in practice). It also assumes compulsory traffic; §4
shows how far a real tiling is from that. Use the model to decide *which* ceiling to attack, then measure
how close you get (`gpu-bench-lab` fits a measured roofline to your device).

---

## 3. LLM inference on the roofline

### 3.1 The FLOPs and bytes of one step

One engine step runs every layer once over the tokens scheduled in it — the whole prompt for a prefill,
one token per sequence for decode. From a model's `config.json` (see the
[transformer primer §8](../../00-foundations/transformers/docs/transformer-primer.md#82-reading-a-model-config)),
`roofline.llm.ModelConfig` counts:

```
per-token matmul params  P = L × (attention + MLP)          (MoE: top_k experts, not all)
FLOPs(step)  = 2 × tokens × P  +  2 × rows_of_logits × vocab × d  +  4 × L × heads × head_dim × positions attended
bytes(step)  = weights streamed once (MoE: only experts hit)  +  KV read  +  KV written
KV per token = 2 (K,V) × L × kv_heads × head_dim × bytes    Llama-3.1-8B, bf16: 131,072 B (128 KiB)
```

For Llama-3.1-8B: 8.03 B parameters; 6.98 B per-token matmul parameters; a 525 M-parameter LM head. A
decode step streams **15.0 GB** — every layer plus the LM head; the input embedding is a gather of one row
per token, not a stream of the table. Activations are assumed to stay on chip (fused), and efficiencies
default to 1, so every time below is an ideal lower bound (`roofline.llm.prefill()`, `decode()`).

### 3.2 Prefill: intensity ≈ tokens in the step

The weights are read once for the whole prompt while FLOPs grow with it, so at 2-byte weights the
intensity is close to the token count. For Llama-3.1-8B on an H100 the bound flips between 300 and 320
tokens — right at the ridge, measured in tokens:

```
prefill, 2,048 tokens, H100, bf16:
  FLOPs = 2.97 × 10¹³    bytes = 15.3 GB    intensity = 1,941 FLOP/B   → compute-bound
  compute time 30.0 ms   memory time 4.57 ms  → ideal TTFT ≈ 30 ms (L4: 245 ms)
```

This is why engines schedule prefill in chunks of hundreds to thousands of tokens (chunked prefill, layer
04): a chunk below ~300 tokens leaves an H100's tensor cores idle, a chunk far above it only adds latency.
It is also why TTFT grows with prompt length — linearly at first, faster for long prompts, where the
quadratic attention term grows (8K tokens: 133 ms, 4.4× the 2K figure). The planning version of this
(MFU, queueing, TTFT budgets) lives in [capacity planning, formula 4](../../00-foundations/gpu-capacity-planning/PRIMER.md).

### 3.3 Decode: batch 1 is a weight stream

At batch 1 each token re-reads 15 GB for ~16 GFLOPs: intensity ≈ 1. Time per token is bytes ÷ bandwidth
regardless of the FLOPs on the datasheet — **4.52 ms on an H100 (221 tokens/s), 50.5 ms on an L4
(19.8 tokens/s)** at 1K context. Batching shares the weight read across sequences, but each sequence also
reads its own KV cache:

| Batch (2K context, H100, bf16) | Intensity | Step time | Aggregate tok/s | Per user tok/s | KV share of bytes |
|---|---|---|---|---|---|
| 1 | 1.1 | 4.56 ms | 219 | 219 | 2% |
| 8 | 7.5 | 5.12 ms | 1,562 | 195 | 13% |
| 32 | 21.8 | 7.05 ms | 4,542 | 142 | 36% |
| 64 | 32.0 | 9.61 ms | 6,659 | 104 | 53% |
| 128 | 41.7 | 14.74 ms | 8,682 | 68 | 70% |
| 208 (HBM limit) | 47.2 | 21.16 ms | 9,832 | 47 | 79% |

Throughput climbs with batch and per-user speed falls: the trade you tune against an ITL SLO. The table
stops where HBM does: 208 sequences of 2K tokens fit beside the weights with 10% headroom
(`max_batch_by_memory()`). At 4K context and a 20 ms ITL the largest batch is 96
(`best_batch_under_itl()`); HBM capacity allows 104.

### 3.4 KV reads cap decode intensity

As batch grows the weight read amortizes away, but the KV read is per sequence and grows with context
exactly as fast as that sequence's attention FLOPs. The limit (`decode_intensity_limit()`):

```
decode intensity as batch → ∞  =  FLOPs per token / KV bytes per token
                               ≈  2 × (heads / kv_heads) / kv_bytes   at long context (GQA group of 4: 4 FLOP/B)

Llama-3.1-8B, bf16 KV:   1K context → 115.7 FLOP/B    4K → 32.0    32K → 7.5
```

Against an H100 ridge of 295, no batch size makes a 4K-context decode *step* compute-bound as a whole.
Treated as one kernel (`decode()`), the step's crossover batch exists only at very short contexts — 297
at c = 0, 440 at c = 128, 854 at c = 256 — and never beyond ~392 tokens
(`max_context_for_compute_bound()`); and HBM runs out first (104 sequences of 4K tokens fit beside the
weights with 10% headroom).

**Per kernel, not per step.** That average blends two kernels with opposite shapes. The weight GEMMs
multiply a [batch × d] activation by every weight matrix: intensity ≈ batch × 2 / weight bytes at any
context, compute-bound from batch 296 (`gemm_crossover_batch()`; FP8 halves the bytes and doubles the
peak, so also 296) — the planning rule "decode turns compute-bound around batch ≈ ridge" in
[capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) is exactly right for them.
Attention reads each sequence's own KV cache at 2 × (heads / kv_heads) / kv_bytes = 4 FLOP/B (bf16 KV)
whatever the batch: never compute-bound. An engine runs the kernels one after another, so the tighter bound
is the **sum of per-kernel roofline times** (`decode_split()`), not max(ΣF ÷ peak, ΣB ÷ BW), which assumes
GEMM math overlaps KV streaming perfectly. They agree while both kernels are memory-bound — every decode
row in §3.3–3.6 and §8 — and part once the GEMMs cross the ridge:

```
Llama-3.1-8B, H100, FP8 weights and KV, 2K context, batch 400 (476 fit):
  one kernel    18.27 ms, memory-bound
  per kernel    GEMMs 3.03 ms (compute-bound) + attention 16.03 ms (memory-bound) = 19.07 ms
```

Past the GEMM crossover, throughput stops rising — 20,978 tokens/s here at any larger batch — and each
extra sequence only adds KV time. That is what the KV-cache techniques in layer 04 fight
([KV cache](../../04-inference-engine/kv-cache/kv-cache-primer.md),
[PagedAttention](../../04-inference-engine/paged-attention/paged-attention-primer.md)): GQA/MQA/MLA and
FP8 KV shrink the bytes; paging and prefix caching stop wasting them. It is also the argument for
attention–FFN disaggregation: the two kernels on separate GPU pools, each at its own batch.

### 3.5 Quantization moves bytes

In the memory-bound regime a step's speedup is old bytes ÷ new bytes. The schemes differ in *which*
bytes they cut (Llama-3.1-8B, H100, 2K context; `roofline.llm.SCHEMES`):

| Scheme | Weight bytes | KV bytes | Math | Batch 1 | Speedup | Batch 64 | Speedup | Compute-bound from (c = 0) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 2 | 2 | bf16 | 4.56 ms | 1.00× | 9.61 ms | 1.00× | batch 297 |
| FP8 (W8A8 + KV8) | 1 | 1 | fp8 | 2.28 ms | 2.00× | 4.81 ms | 2.00× | 297 |
| W8A16 | 1 | 2 | bf16 | 2.32 ms | 1.97× | 7.37 ms | 1.30× | 149 |
| W4A16 (INT4, AWQ/GPTQ) | 0.5 | 2 | bf16 | 1.20 ms | 3.80× | 6.25 ms | 1.54× | 75 |
| W4A16 + FP8 KV | 0.5 | 1 | bf16 | 1.16 ms | 3.93× | 3.69 ms | 2.61× | 74 |

FP8 halves every byte and doubles the peak, so it is exactly 2× in either regime. Weight-only INT4 is
nearly 4× at batch 1 but only 1.5× at batch 64, because half the bytes there are KV cache it did not
touch — and it lowers the compute-bound crossover to ~75 because the math still runs at the bf16 peak.
Quantization also buys **capacity**: FP8 weights and KV let 476 sequences of 2K tokens fit on the H100
instead of 208. (INT4 group scales add ~2/group_size bytes per weight; ignored here.)

### 3.6 MoE: which weights a step streams

Memory is sized by *total* parameters — every expert lives in HBM — while a token's FLOPs follow *active*
ones. A decode step streams only the experts its tokens are routed to; with uniform routing one layer
touches E(1 − (1 − k/E)^T) distinct experts for T tokens (`experts_touched()`):

| Batch (1K context) | Mixtral-8x7B (8 experts, top-2) | Step bytes on H200 | Step time | Qwen3-30B-A3B (128, top-8) |
|---|---|---|---|---|
| 1 | 2.00 experts/layer | 25.6 GB | 5.34 ms | 8.0 / 128 |
| 4 | 5.47 | 65.1 GB | 13.57 ms | 29.1 |
| 16 | 7.92 | 94.4 GB | 19.66 ms | 82.4 |
| 64 | 8.00 | 101.7 GB | 21.20 ms | 125.9 |

Mixtral (46.7 B total, 12.9 B active) decodes like a 13 B model only at batch 1; by batch 16 it streams
nearly all 93 GB of its weights every step. Fine-grained MoE keeps the saving to larger batches, and at
high batch the cost per token still falls because the full stream is shared. The roofline consequence:
FLOPs follow *active* parameters while the bytes approach *total* ones — each expert sees only B·k/E of
the batch — so the batch at which the weight stream reaches the ridge scales with total ÷ active. At
c = 0 on an H200 (ridge 206; `decode_crossover_batch()`): 207 for Llama-3.1-8B, 754 for Mixtral-8x7B
(total/active 3.6) and 2,055 for Qwen3-30B-A3B (9.1, or 9.9 without the input embedding, which is
gathered, not streamed or multiplied). Serving large MoE models is therefore about big
batches, and expert parallelism is how systems reach them: attention runs data-parallel on many GPUs and
all their tokens meet at each expert (see [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md)
for the sizing side, §5 for the all-to-all it implies).

---

## 4. The memory hierarchy and why tiling/fusion win

Each level down the hierarchy is larger and slower (rough H100 bandwidths and latencies:
[gpu-primer §3](../gpu-primer/gpu-primer.md#3-memory-is-the-whole-game)):

```
registers   256 KB per SM (33 MB total)    accumulators of the tile being computed
SMEM / L1   up to 228 KB per SM            operand tiles, staged by TMA / async copy
L2          50 MB, shared                  re-reads of panels by neighbouring thread blocks
HBM         80 GB at 3.35 TB/s             everything else: weights, KV cache, activations
host DRAM   over PCIe Gen5 (63 GB/s/dir)   offload, loading, swapping
```

Every level has its own roofline: to keep the tensor cores busy a kernel must reuse each byte fetched from
level X about peak ÷ bandwidth(X) times. For HBM that is 295 on an H100. Tiling is how a kernel manufactures
that reuse.

### 4.1 Tiling

A GEMM computes each Tm × Tn output tile by streaming a Tm × k panel of A and a k × Tn panel of B through
on-chip memory. Each k-step loads (Tm + Tn) elements and does 2·Tm·Tn FLOPs
(`roofline.roofline.tile_intensity()`):

```
tile intensity = 2 · Tm · Tn / ((Tm + Tn) · b)        bf16:  16×16 → 8   64×64 → 32   128×128 → 64
                                                             128×256 → 85   256×256 → 128 FLOP/B
```

Bigger tiles mean more reuse, but they must fit. A 128 × 256 tile with 64-wide k-slices in bf16, four
pipeline stages deep, needs 4 × (128 + 256) × 64 × 2 = 192 KiB of shared memory (`tile_smem_bytes()`), and
its fp32 accumulators need 128 × 256 × 4 = 128 KiB of registers — 84% of an H100 SM's shared memory and
half its register file. That is why production GEMM tiles top out around this size.

Even then 85 FLOP/B is below the HBM ridge of 295. If every tile re-read its panels from HBM, a 4096³
bf16 GEMM with 128 × 128 tiles would move 2.18 GB — 21.7× the compulsory 101 MB — at 63 FLOP/B
(`tiled_gemm_bytes()`); with no tiling at all (two loads per multiply-add) it would sit at 0.5 FLOP/B. The
remaining reuse comes from the **L2**: thread blocks that run at the same time share panels, and kernels
order (swizzle) their tiles and, on Hopper, multicast them to clusters so that HBM sees close to
compulsory traffic. Tiling happens at every level of the hierarchy, not just one.

### 4.2 Fusion

Elementwise operations sit at ~0.2 FLOP/B; chaining them unfused round-trips every intermediate through
HBM (`fusion_bytes()`):

```
k ops on n elements, e of which read a second full tensor (a residual):
   unfused (2k + e) · n · b bytes            fused (2 + e) · n · b
4 ops (bias, GELU, dropout, residual add) on a 4096 × 8192 bf16 activation, e = 1:
   unfused 604 MB → 180 µs on an H100       fused 201 MB → 60 µs
```

Fusion changes no FLOPs and removes 67% of the bytes: the fused kernel still reads each distinct input
once and writes the result once. FlashAttention is the canonical case — tile Q, K and
V into shared memory, compute softmax online, never write the N × N score matrix — see the
[FlashAttention primer](../../04-inference-engine/flash-attention/flash-attention-primer.md) (its §3
prices the naive version at ~32 FLOP/B). How warps map onto sectors, banks and tiles — coalescing, shared
memory bank conflicts, occupancy — is the subject of layer 02
([`02-cuda-nccl-runtime`](../../02-cuda-nccl-runtime/README.md): `cuda-and-nccl/PRIMER.md` §2–3).

---

## 5. Fabrics quantitatively

### 5.1 The link ladder

Bandwidth per direction, theoretical (`roofline.fabric.LINKS`); the α values are illustrative per-step
latencies including software, for use in the model — measure yours with nccl-tests:

| Link | Scope | GB/s per direction | α used here |
|---|---|---|---|
| NVLink 5 (B200, GB200, GB300): 18 links × 50 GB/s | GPU ↔ GPU in an 8-GPU node or a 72-GPU rack | 900 | 2 µs |
| NVLink 4 (H100, H200): 18 × 25 GB/s | GPU ↔ GPU, 8 per node via NVSwitch | 450 | 2 µs |
| NVLink 3 (A100): 12 × 25 GB/s | GPU ↔ GPU, 8 per node | 300 | 2 µs |
| PCIe Gen5 x16 (32 GT/s, 128b/130b) | GPU ↔ CPU / NIC / NVMe | 63 | 4 µs |
| PCIe Gen4 x16 | same, one generation older | 31.5 | 4 µs |
| InfiniBand XDR, 800 Gb/s NIC | node ↔ node, per NIC | 100 | 5 µs |
| InfiniBand NDR / RoCE, 400 Gb/s NIC | node ↔ node, per NIC | 50 | 5–6 µs |
| 100 GbE with TCP | commodity network | 12.5 | 25 µs |

NVSwitch makes the NVLink domain all-to-all: any GPU pair gets the full per-GPU bandwidth. The size of
that domain is the property that matters most (8 GPUs in an HGX node, 72 in an NVL72 rack); the
[deployment primer §5](../gpu-deployment/gpu-deployment-primer.md#5-the-interconnect-hierarchy) draws the
cliff between scale-up and scale-out.

### 5.2 The α-β model and the ring all-reduce

```
point-to-point:      t = α + n / β                                      transfer_time()
ring all-reduce:     t = 2(p−1)·α + 2·(p−1)/p · n/β                     ring_allreduce_time()
all-gather / RS:     t = (p−1)·α + (p−1)/p · n/β                        ring_allgather_time()
recursive doubling:  t = ⌈log₂ p⌉ · (α + n/β)                           recursive_doubling_allreduce_time()
crossover:           ring latency term = bandwidth term at n = p·α·β    allreduce_crossover_bytes()
```

The ring's bandwidth term, ~2n/β, is optimal and independent of p; its latency term grows with p.
Recursive doubling has the opposite shape. Worked on 8 H100s over NVLink 4: the ring's two terms meet
at **7.2 MB** (over a 400 Gb/s NIC: 2.0 MB). A 1 GiB all-reduce takes 4.20 ms — algbw 255 GB/s — and
nccl-tests would report **busbw = algbw × 2(p−1)/p = 447 GB/s**, the number to compare with the link's 450
(`busbw()`; layer 02 covers busbw, NCCL algorithms and protocols in depth).

### 5.3 Tensor parallelism: the collective inside every layer

Megatron-style tensor parallelism splits each layer's matrices across p GPUs and all-reduces a
[tokens × d_model] activation twice per layer — after attention's output projection and after the MLP's
down projection (`tp_allreduces_per_step()`, `tp_comm_time()`). For Llama-3.1-70B that is **160
all-reduces per step**:

```
decode, batch 1, TP=8 on H100 NVLink:
  message   = 1 token × 8,192 × 2 B = 16 KiB           (440× below the 7.2 MB crossover: latency-bound)
  weights   = each GPU streams its shard in 5.20 ms
  comm/step = 160 × ring all-reduce = 4.49 ms          (latency-optimal algorithm: 0.98 ms)

prefill, 4,096 tokens, TP=8:
  message   = 4,096 × 8,192 × 2 B = 67 MB               (bandwidth-bound)
  compute   = 73.6 ms per GPU
  comm/step = 46.2 ms over NVLink 4    (387.0 ms if every ring hop crossed a 400 Gb/s NIC: 8 GPUs in 8 nodes)

prefill, 4,096 tokens, TP=16 across two 8-GPU nodes      tp_comm_time_across_nodes()
  compute   = 36.8 ms per GPU
  comm/step = 74.7 ms with rails (8 NICs per node, each GPU sends its n/8 share over its own NIC)
              262.6 ms with one NIC per node        426.7 ms as one flat ring through the NICs
```

Two lessons. At decode, the count of collectives times the algorithm's latency is the cost — which is why
engines ship latency-optimized all-reduce kernels and why TP's per-GPU efficiency falls as p grows (batch-1
tokens per GPU-second: 23.3 at TP=2, 12.9 at TP=8 with a ring). At prefill, bandwidth is the cost, and it
does not shrink with p (a ring's bandwidth term is ~2n/β whatever the degree) while each GPU's compute
halves every time p doubles: TP=8 already spends 46 ms communicating per 74 ms of compute. Across nodes
NCCL spreads the traffic over every rail, so each GPU's NIC carries roughly its n/8 share — yet TP=16
over two nodes still spends ~75 ms per 4K-token step in all-reduce against ~37 ms of compute per GPU:
111 ms per step against TP=8's 120 ms (no overlap), for 1.9× the GPU-seconds. Without a NIC per GPU it
is far worse (263 ms with one NIC per node, 427 ms as a flat ring). Past the node the cost is set by its
NIC aggregate (8 × 50 GB/s with rails) and a larger α, on links that also carry data-, pipeline- and
expert-parallel traffic. Hence the rule from the [deployment primer §4](../gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu):
tensor (and expert) parallelism inside the NVLink domain, pipeline and data parallelism across it.
Expert parallelism's all-to-all has busbw factor (p−1)/p and the same α-β shape.

### 5.4 Topology: rails, fat trees, oversubscription, bisection

Scale-out networks for GPU clusters are **rail-optimized fat trees**. Each node has one NIC per GPU; NIC
*i* of every node in a group plugs into the rail-*i* leaf switch, and leaves connect to spines:

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

Same-index GPUs in the group are one switch hop apart; cross-rail traffic climbs to a spine (three hops)
unless NCCL's PXN first moves it over NVLink to the local GPU on the destination's rail
(`switch_hops()`). Rails are why every GPU can drive its own NIC at once: a hierarchical 1 GiB
all-reduce over 4 nodes × 8 GPUs takes **8.3 ms with 8 NICs per node and 36.4 ms with one**
(`hierarchical_allreduce_time()`).

A two-tier folded Clos of radix-R switches is **non-blocking** (1:1) when each leaf gives half its ports to
hosts and half to spines; it tops out at R²/2 endpoints (`leaf_spine()`). With radix-64 switches: 2,048
endpoints need 64 leaves and 32 spines. Giving 48 ports down and 16 up serves 3,072 endpoints with 64
leaves and 16 spines — at **3:1 oversubscription**, which divides the **bisection bandwidth** (the
capacity across the worst half/half cut, `bisection_gbs()`) from 76,800 GB/s to 25,600 GB/s. Independent
inference replicas (mostly north-south traffic) tolerate oversubscription; training all-reduces, expert
all-to-all and disaggregated KV transfer (layer 05) need bisection.

### 5.5 GPUDirect RDMA and staged copies

Without GPUDirect RDMA, a GPU-to-remote-GPU message is copied to host memory, sent, and copied up again.
Copies through a chain of hops cost either the sum (store-and-forward) or the slowest hop plus one chunk
per other hop (pipelined; `staged_transfer_time()`):

```
1 GiB over [PCIe Gen5 63, NDR 50, PCIe Gen5 63] GB/s:
   host-staged, store-and-forward    55.6 ms
   GPUDirect RDMA, pipelined (1 MiB)  21.5 ms   ≈ bytes / the NIC
```

A carefully pipelined staged path can approach the same bandwidth for large messages, but not the latency,
and it spends host memory bandwidth and CPU. GPUDirect RDMA needs the NIC and GPU to share a PCIe switch
for full speed — which is what the topology matrix shows.

### 5.6 NUMA and `nvidia-smi topo -m`

`nvidia-smi topo -m` prints, for every GPU and NIC pair, the path between them (`fabric.TOPO_LEGEND`),
best to worst: **NV#** (a bonded set of # NVLinks) → **PIX** (at most one PCIe bridge) → **PXB** (several
PCIe bridges, no host bridge) → **PHB** (through a PCIe host bridge, typically the CPU) → **NODE** (between
host bridges within a NUMA node) → **SYS** (across the inter-socket link, QPI/UPI). Its CPU and NUMA
affinity columns say which socket each GPU hangs off. Three rules: place tensor-parallel groups on NV#
pairs, never across SYS; give each GPU the NIC it reaches by PIX; pin each GPU's process, its data
loaders and its pinned host buffers to that GPU's NUMA node, or host-to-device copies cross the socket
link. `gpu-bench-lab` parses the matrix on your machine (`topo.py`) and measures P2P bandwidth per path.

---

## 6. Storage and cold start

A checkpoint is `params × bytes per param` (`roofline.storage.checkpoint_bytes()`): Llama-3.1-8B is
16.1 GB in bf16, Llama-3.1-70B 141.1 GB. A new replica must fetch those bytes and copy them into HBM, and
the time is set by the slowest tier on the path:

```
t_weights ≈ bytes / min(fetch bandwidth, host→device bandwidth × GPUs)
```

With **assumed** single-reader bandwidths (`storage.ASSUMED_GBS` — replace them with measurements from
`gpu-bench-lab` notebook `04_weights_loading_and_cold_start`), 141 GB takes:

| Source (assumed GB/s) | Time for 70B bf16 |
|---|---|
| one object-store or internet stream (0.1) | 1,411 s = 23.5 min |
| network block device (1.2) | 118 s |
| local NVMe (7) | 20.2 s |
| 128 parallel range streams, capped by a 100 Gb/s NIC (12.5) | 11.3 s |
| host page cache (20) | 7.1 s |
| host → 8 GPUs, pinned, PCIe Gen5 (8 × 50) | 0.35 s |

### 6.1 Parallel and streamed loading

One HTTP stream is slow; many range reads add up until the NIC, disk or service caps them
(`parallel_gbs()`). Streaming chunks straight into GPU memory overlaps fetch and copy, so the time tends to
the slower hop rather than the sum (`load_time(..., streamed=True)`) — the idea behind safetensors loaders
that read directly into device memory and model streamers such as the Run:ai Model Streamer load format in
vLLM (verify). Host-to-device copies from pinned memory run near the PCIe rate, while pageable memory must
be staged by the driver through a pinned bounce buffer — the assumptions here are 25 vs 12 GB/s on PCIe
Gen4; measure yours with `gpu-bench-lab` notebook `02_memory_bandwidth_and_transfers`.

### 6.2 The anatomy of a cold start

```
scale-up ─► provision node ─► pull image ─► fetch weights ─► copy to GPU ─► engine init + warm-up ─► ready
            (warm pool?)      (streaming?)   (parallel? cache?) (pinned? streamed?) (CUDA graphs, compile, KV profiling)
```

With stage times assumed as provision 120 s, image 60 s, engine init 90 s (`cold_start()`):

| Scenario | Total | Largest stage |
|---|---|---|
| new node, one stream | 1,681 s | fetch weights, 84% |
| new node, parallel + streamed weights | 281 s | provision node, 43% |
| warm node, checkpoint in page cache | 97 s | engine init + warm-up, 93% |

The lesson is the order of attack: first the weights (parallelism, streaming, a local or regional cache,
smaller precision), then the node (warm pools, pre-pulled or streamed images — layer 03), then the engine
(layer 04). A budget turns into a bandwidth requirement: a 70B replica that must be ready in 120 s on a
warm node pool, with a 20 s image pull and 60 s of engine init (assumed), leaves 40 s for streamed weights:
141.1 GB / 40 s = **3.53 GB/s** of fetch, i.e. 36 parallel streams at 0.1 GB/s. Cold start is
also why autoscaling on LLM replicas needs headroom and scale-ahead signals (layer 05).

---

## 7. Reliability at scale

### 7.1 Failure rates add

With independent failures at a constant rate, N components with MTBF M fail as a group every M / N
(`roofline.reliability.cluster_mtbf()`), and the chance of a clean run of t hours is exp(−N t / M)
(`p_survive()`). The best public data point is the Llama 3 report (Meta, 2024): **419 unexpected
interruptions in 54 days of pre-training on 16,384 H100s**, about 78% attributed to hardware (GPU faults
the largest share). Backing out a per-GPU figure (`component_mtbf_from_observation()`):

```
M_gpu = 16,384 × 1,296 h / 419 = 50,677 GPU-hours ≈ 5.8 years

   8 GPUs → an interruption every 6,335 h       P(clean 24 h) = 0.996
1,024 GPUs → every 49.5 h                         P(clean 24 h) = 0.616
16,384 GPUs → every 3.09 h (7.76 per day)
```

Each GPU lasts years; the job does not last a shift. The figure lumps GPUs, hosts, network and software
together, real fleets add infant mortality and correlated failures (a switch, a power feed), and silent data
corruption shows up as bad numbers rather than crashes — so treat it as a first-order planning rate and
watch the health signals (DCGM, XID errors — layer 02).

### 7.2 How often to checkpoint: Young/Daly

A synchronous job that checkpoints every τ seconds, taking δ seconds per checkpoint, loses δ/τ to
checkpointing and, per failure, about τ/2 of recomputed work plus a restart R (`wasted_fraction()`):

```
waste ≈ δ/τ + (τ/2 + R)/M        minimised at   τ* = √(2 δ M)   (Young, 1974)
at τ*: waste = √(2δ/M) + R/M       Daly (2006) adds higher-order terms: young_daly_interval(..., higher_order=True)

restart cost R = 0 (ignored) — 16,384 GPUs (M = 3.09 h = 11,135 s):
   δ = 60 s → τ* = 19.3 min (Daly 18.6), waste 10.4%    (checkpointing hourly: 17.8%)
   δ = 10 s → τ* =  7.9 min,             waste  4.2%    (hourly: 16.4%)
4,096 GPUs, δ = 30 s → τ* = 27.2 min, waste 3.7%

with an assumed restart R = 10 min (reschedule, reload, re-initialise): R/M = 5.4% more
   δ = 60 s → waste 15.8%        δ = 10 s → waste 9.6%
```

Waste scales as √δ, so a 4× faster checkpoint halves that part — the case for asynchronous checkpointing to
host memory and local disk, then to storage in the background. Restart time adds R/M on top and does not
move τ*: at this failure rate a 10-minute restart (5.4%) costs more than everything a 10 s checkpoint
wastes (4.2%), so fast restart — hot spare nodes, checkpoints replicated in peer memory, pre-initialised
jobs — is a lever alongside fast checkpoints.

### 7.3 Inference: replicas are failure domains

Serving replicas fail independently, so the question becomes "how many spares?". A TP=8 replica is down
when any of its GPUs is: replica MTBF = M/8 = 6,335 h, and with a 48 h MTTR (assumed: detect, drain,
repair) availability a = MTBF/(MTBF + MTTR) = 0.99248 (`replica_availability()`). Needing 8 replicas up
(`p_at_least()`, `replicas_for()`):

```
deploy  8 → P(≥ 8 up) = 0.941
deploy  9 → 0.998
deploy 10 → 0.99995      → 99.9% needs 10 replicas = 80 GPUs for 64 GPUs of capacity (16 spare)

the same capacity as 16 × TP=4 (FP8, so the model fits 4 GPUs) → 18 replicas = 72 GPUs (8 spare)
the same capacity as 64 × single-GPU replicas (a smaller model)  → 66 replicas = 66 GPUs (2 spare)
```

Bigger replicas are bigger blast radii: the spare capacity you carry grows with the replica size. This is
one more reason to use the smallest TP degree that meets the latency SLO (§5.3) — and why training and
inference clusters are shaped differently ([deployment primer §9](../gpu-deployment/gpu-deployment-primer.md#9-training-clusters-vs-inference-clusters)).

---

## 8. The cost of a token

### 8.1 From $/GPU-hour to $/M tokens

```
$/M tokens = ($/GPU-hr × GPUs) / (tokens/s × 3600 × utilisation) × 10⁶      cost_per_million_tokens()
```

With decode throughput from §3 (an upper bound, so these are lower bounds on cost) and GCP list prices
from the research snapshot — L4 ~$0.70/hr, H100 ~$11/GPU-hr on demand, ~$3.7/GPU-hr Spot, us-central1,
September 2026 (verify; current prices in [`COMPUTE.md`](../../COMPUTE.md)) — Llama-3.1-8B at 2K context.
One rule sets every batch: the largest that meets an **ITL of 10 ms** (100 tokens/s per user) and fits in
HBM (`best_batch_under_itl()`):

| Option | Batch | ITL | tok/s | Per user | $/M output tokens at 100% | at 60% |
|---|---|---|---|---|---|---|
| H100 on demand, bf16 | 68 (ITL-bound) | 9.93 ms | 6,847 | 101 | $0.446 | $0.744 |
| H100 Spot, bf16 | 68 (ITL-bound) | 9.93 ms | 6,847 | 101 | $0.150 | $0.250 |
| H100 on demand, FP8 | 193 (ITL-bound) | 9.98 ms | 19,345 | 100 | $0.158 | $0.263 |
| L4 on demand, bf16 — misses the SLO | 20 (HBM-bound) | 67.9 ms | 294 | 14.7 | $0.660 | $1.10 |

The L4 cannot meet the SLO at any batch — at batch 1 it needs 50.9 ms to stream the weights — so its row
sits at its HBM limit, a slower product. Even so the H100 gives the cheaper token: its bandwidth and
capacity together deliver 23× the L4's tokens/s for 15.7× the price. FP8 then cuts the H100's cost another
2.8×: exactly 2.0× from halving every byte, and 1.4× more from the bigger batch (193 vs 68) that the same
ITL allows. Prefill tokens are cheaper still: the same H100 ingests a 2K prompt at an ideal ~68,000
tokens/s, an order of magnitude above decode — the root of the input/output price asymmetry of hosted APIs,
and of why prefix caching (skipping prefill) is a cost lever for agents.

### 8.2 Utilisation

A fleet sized for its peak idles off-peak: utilisation = mean load ÷ peak (`utilisation()`). A day at
20% / 100% / 60% of peak in thirds averages 60%, so every $/M above is 1.67× higher. Utilisation is
where serving economics are won or lost: autoscaling and scale-to-zero (layers 03 and 05), batch traffic
filling the troughs, and admission control (layer 06) keeping the peak honest.

### 8.3 Rent or own

Owning costs a fixed amount per hour (depreciation plus fixed opex) plus energy while busy; renting costs
the hourly rate only for hours used. Owning wins above (`owned_cost_per_hour()`, `breakeven_utilisation()`):

```
u* = fixed per hour / (rent per hour − energy per hour)

8 × H100 server: $300k over 4 years + $30k/yr opex (assumed), 10.2 kW at full load (verify), PUE 1.3, $0.10/kWh
   → $1.50 per GPU-hour fixed + $0.166 per busy GPU-hour energy
   break-even vs on demand at $11/GPU-hr: 13.8%     vs Spot at $3.7: 42.4%     vs $2.0 rental: 81.7%
```

The "own above ~60–70% utilisation" rule of thumb in the
[deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)
is the break-even against cheap rental; against on-demand list prices ownership pays off far earlier — if
you can get the hardware, power and people. The same equation prices managed capacity: the
Provisioned-Throughput break-even in the agentic scaling lab
([`01-scaling-primer.md` §3.5](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md#35-provisioned-throughput))
is utilisation of committed units versus pay-as-you-go, and agent workloads
([layer 07](../../07-application-agent-framework/README.md)) set the tokens per task that multiply it.

---

## 9. The accelerator landscape (September 2026 snapshot)

Dense peaks, TFLOP/s; memory as marketed; ridge computed at bf16 (fp16 for the T4); scale-up bandwidth as
marketed — NVLink and xGMI are bidirectional totals per GPU, TPU ICI per chip as published — with the
size of the domain in parentheses. Generated by `roofline.specs.table()`; **every entry (verify)**.

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

Derived entries: the RTX PRO 6000 tensor rates are halved down from its 4 PFLOPS sparse-FP4 headline;
Ironwood's bf16 is assumed half its 4,614 TFLOPS fp8; B200 is the HGX board (192 GB physical, 180 GB
usable); FP4 on the MI355X is MXFP4. The T4 has no bf16, tf32 or fp8 paths. Announced or ramping but
not in the catalogue: NVIDIA Vera Rubin and AMD's MI400 series — see the
[deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)
(verify).

How to read it, for inference:

- **Bandwidth sets decode speed per replica** (§3.3): 0.3 TB/s (L4) to 8 TB/s — a 27× range. Compare
  parts on $/M tokens (§8), not on $/hour or TFLOPS.
- **Capacity sets what fits and how big the batch can be** (§3.4–3.5): 16 GB (T4, TPU v5e) to 288 GB
  (GB300, MI355X). An H200 is an H100 plus capacity and bandwidth; for decode that is most of what matters.
- **The ridge sets how much batching the FLOPs need**: from 153 (A100 80GB) to 560 (TPU v6e). A high ridge
  is fine for prefill and training, wasted on small-batch decode.
- **The scale-up domain sets how far TP and EP can go without the NIC** (§5.3): 1 for PCIe cards, 8 for HGX
  and MI3xx platforms, 72 for NVL72 racks, 256 to 9,216 chips over TPU ICI.
- **Precision support is a generation marker**: fp8 from Ada/Hopper/MI300, fp4 from Blackwell/MI355X.
- **Power has gone from 70 W to 1,400 W per accelerator**: facilities, not procurement, gate owned
  deployments ([deployment primer §6](../gpu-deployment/gpu-deployment-primer.md#6-the-rack-is-the-new-unit-of-deployment)).

---

## 10. Getting hardware

Every concept in this topic is learnable at T0; hardware is for measuring it. What each tier buys here:

| Tier | Where | What you can see | Cost (verify; see [`COMPUTE.md`](../../COMPUTE.md)) |
|---|---|---|---|
| T0 | laptop, Colab CPU, CI | the core notebooks; your CPU's own roofline and disk throughput with `gpu-bench-lab` (numpy backend) | $0 |
| T1 | Colab or Kaggle T4 (free, not guaranteed); a rented 24 GB GPU; GCP L4 Spot or Cloud Run L4 | a real GPU roofline (GEMM sweep by dtype), HBM bandwidth, pinned vs pageable host copies, weight loading | free – ~$0.7/hr |
| T2 | Kaggle 2×T4 (free, PCIe only, no NVLink); RunPod / Vast / Lambda 2–8× A100/H100 SXM for an hour; GCP `a2-highgpu-2g` | P2P bandwidth over PCIe vs NVLink, `nvidia-smi topo -m` on real topologies, all-reduce busbw | $0 – ~$25 per session |
| T3 | GCP via Terraform (`gpu-bench-lab/deploy/gcp/terraform`) | the same suite on a Spot L4 VM with results in a bucket; switch the machine type for NVLink | pay per use, Spot, auto-stop |

### 10.1 On Google Cloud

GPU families (September 2026, verify): **G2** (L4, e.g. `g2-standard-4` = 1 L4, ~$0.70/hr on demand); **N1 +
T4**; **A2** (A100 40 GB `a2-highgpu-*`, 80 GB `a2-ultragpu-*`); **A3** (H100: on demand only as
`a3-highgpu-8g`, ~$88/hr; smaller A3 shapes via Spot or flex-start; **A3 Mega** adds GPUDirect-TCPXO
networking); **A3 Ultra** (H200) and **A4** (B200, NVLink 1.8 TB/s per GPU), **A4X** (GB200 NVL72) and **A4X
Max** (GB300 NVL72), which use GPUDirect RDMA over ConnectX NICs on a rail-aligned network (ConnectX-7 on A3
Ultra and A4; likely ConnectX-8 on A4X Max — verify per machine type); **G4** (RTX PRO 6000 Blackwell, 96
GB). **Cloud Run** offers L4 and RTX PRO 6000 GPUs with per-second billing and scale to zero. **TPUs**: v5e
and v6e as Cloud TPU VMs or through GKE; TPU7x "Ironwood" (GA April 2026) through GKE (verify).

Obtainability matters as much as price. **On-demand** is simplest and scarcest for large parts. **Spot** is
60–91% cheaper and can be preempted at any time — fine for benchmarks and stateless replicas with headroom.
**Dynamic Workload Scheduler flex-start** queues a request and provisions all of it at once (up to 7 days;
used through Kueue ProvisioningRequest, layer 03); **calendar mode** reserves a future block;
**reservations** hold capacity you pay for whether used or not. Two account facts gate everything: GPUs are
not usable on a Free Trial billing account, and GPU quota (`GPUS_ALL_REGIONS` plus per-type regional quota)
often starts at 0 — request it early; T4 and L4 are usually approved quickly, A100 and H100 rarely for a new
individual account.

### 10.2 Elsewhere

**Colab** (free T4, 16 GB, roughly 15–30 GPU-hours a week, not guaranteed) and **Kaggle** (2×T4 or a P100,
30 GPU-hours a week — the free two-GPU box, PCIe only) cover T1 and most of T2. **Vast.ai** (marketplace)
and **RunPod** (per-second) rent single GPUs and multi-GPU NVLink boxes as containers — no driver or kernel
control, no Kubernetes; **Lambda** rents full VMs; **Modal** runs serverless Python on GPUs with monthly free
credits. A local kind cluster serves the Kubernetes layers (03, 05) without GPUs. Prices and quotas move
monthly: the maintained list is [`COMPUTE.md`](../../COMPUTE.md); the learning order is
[`CURRICULUM.md`](../../CURRICULUM.md).

---

## In a design review

**The two-minute walkthrough.** "I start from four datasheet numbers — dense peak at our precision, HBM
bandwidth and capacity, and the per-direction link bandwidth — and one ratio, the ridge: 295 FLOP per
byte on an H100. An LLM step streams the weights once, so its intensity is about its token count. Prefill
of a 2K prompt is compute-bound: TTFT is FLOPs over peak, ~30 ms ideal for an 8B model. Decode is
memory-bound: time per token is bytes over bandwidth, 4.5 ms at batch 1. Batching amortizes the weights —
the GEMMs reach the ridge near batch 300 — but attention reads each sequence's KV cache at a few FLOP per
byte, so at 4K context the whole step averages at most 32 FLOP/B and HBM capacity limits the batch first.
So my levers are bytes: FP8 weights and KV, GQA, paging, prefix caching. Across GPUs I use α-β: TP makes
160 all-reduces per step for a 70B model, latency-bound at decode and bandwidth-bound at prefill; that
cost does not shrink as TP grows while each GPU's compute does, and past the node each GPU's share rides
a NIC 9× slower than NVLink — so TP stays in the NVLink domain. Then the fleet: cold start is bytes over
the slowest tier, failure rates add so large jobs checkpoint every √(2δM) and serving carries spare
replicas, and $/M tokens is $/GPU-hr over tokens/s × utilisation."

**Drill.**

1. *Our H100s' tensor cores are mostly idle during decode. Should we buy B200s for their FLOPs?* — Decode
   is memory-bound; idle tensor cores are expected. The B200's gain for decode is its bandwidth (8 vs 3.35 TB/s,
   2.4×) and capacity (bigger batch), not its 2.3× bf16 peak. Measure achieved HBM bandwidth (DCGM's
   DRAM-active, not the "GPU utilisation" counter) and compare $/M tokens.
2. *Why not TP=16 across two 8-GPU nodes for a model that does not fit in one?* — Communication then
   outgrows compute. Priced for a 70B model: even on rails, with the traffic spread over every NIC, a
   4K-token prefill step spends ~75 ms in all-reduce against ~37 ms of compute per GPU, so twice TP=8's
   GPUs barely shorten the step; with one NIC per node it is 263 ms. Fit it in one NVLink domain (FP8, an
   H200/B200 node, an NVL72 rack) or use PP=2 × TP=8, which keeps the all-reduces on NVLink and sends one
   activation across the node per step.
3. *Will FP8 double decode throughput?* — Only if every byte halves: FP8 weights *and* FP8 KV give exactly 2×
   in the memory-bound regime and roughly double the batch that fits. Weight-only 8-bit gives ~2× at batch 1
   but 1.3× at batch 64 × 2K context, where KV is half the bytes.
4. *A new 70B replica takes 25 minutes to become ready. What do you change first?* — Decompose: one
   object-store stream at 0.1 GB/s is 23.5 of those minutes. Parallel range reads (NIC-bound at ~11 s),
   streaming into GPU memory, a regional or local cache; then warm pools and pre-pulled images; then
   engine init.
5. *How often should a 16K-GPU job checkpoint, and what would make it cheaper?* — At the Llama 3 failure
   rate the job is interrupted every ~3.1 h; with a 60 s checkpoint, every √(2 × 60 × 11,135) s ≈ 19 min,
   losing ~10% plus R/M for restarts (~16% with a 10-minute restart). Asynchronous checkpoints (10 s) cut
   the interval to ~8 min and the checkpoint waste to ~4%; then the restart dominates, so make it fast too.
6. *We need 8 TP=8 replicas up at 99.9%. How many do we deploy, and could we need fewer GPUs?* — Ten
   (80 GPUs) at a 48 h MTTR. Smaller failure domains need fewer spares: if FP8 lets the model fit on TP=4,
   18 replicas (72 GPUs) deliver the same capacity at the same target.

---

## Glossary

| Term | Meaning |
|---|---|
| Arithmetic intensity | FLOPs performed per byte moved to or from a memory level |
| Roofline | attainable FLOP/s = min(peak, intensity × bandwidth) |
| Ridge point | peak ÷ bandwidth: the intensity at which a kernel stops being memory-bound |
| Dense vs sparse peak | headline tensor rates assume 2:4 structured sparsity and are 2× the dense rate |
| Compulsory traffic | bytes a kernel must move if every operand is read once and every result written once |
| Tile / tiling | computing an output block from operand panels held on chip, to reuse each fetched byte |
| Fusion | running several operations in one kernel so intermediates never touch HBM |
| TDP / TBP | the board power limit; clocks throttle to stay inside it |
| KV cache | cached attention keys and values: 2 × layers × kv_heads × head_dim × bytes per token |
| W8A8 / W4A16 | weight bits / activation bits; weight-only schemes dequantize to bf16 for the math |
| MoE, top-k | mixture of experts; each token is routed to k of E expert MLPs |
| α-β model | transfer time = latency α + bytes ÷ bandwidth β |
| algbw / busbw | nccl-tests: size ÷ time, and that scaled by the collective's factor (2(p−1)/p for all-reduce) |
| Tensor parallelism (TP) | splitting each layer's matrices across GPUs; two all-reduces per layer |
| NVLink domain | GPUs joined by NVLink/NVSwitch at full bandwidth: 8 per HGX node, 72 per NVL72 rack |
| Rail-optimized | NIC i of every node wired to leaf switch i, so same-index GPUs are one hop apart |
| PXN | NCCL moving data over NVLink to the GPU on the destination's rail before the network hop |
| Oversubscription | a leaf's downlink bandwidth ÷ uplink bandwidth (1:1 = non-blocking) |
| Bisection bandwidth | capacity across the worst cut that splits the endpoints in half |
| GPUDirect RDMA | a NIC reading and writing GPU memory directly, bypassing host memory |
| NUMA affinity | which CPU socket (and memory) a GPU or NIC is attached to |
| MTBF / MTTR | mean time between failures / to repair |
| Young/Daly interval | the checkpoint spacing √(2 δ M) (with Daly's corrections) that minimises lost time |
| Utilisation | mean load ÷ provisioned capacity |
| Break-even utilisation | the utilisation above which owning beats renting the same capacity |

---

## Sources

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore
  Architectures", *CACM* 52(4), 2009.
- NVIDIA datasheets: T4, L4, A100, H100, H200, HGX B200 / DGX B200, GB200 NVL72, GB300 NVL72, RTX PRO 6000
  Blackwell Server Edition; *NVIDIA A100 Tensor Core GPU Architecture* (2020) and *NVIDIA H100 Tensor Core
  GPU Architecture* (2022) whitepapers — SM counts, per-SM tensor rates, cache sizes.
- AMD Instinct MI300X, MI325X and MI355X product pages and datasheets.
- Google Cloud TPU documentation (v5e, v6e, TPU7x) and GPU machine-family documentation (G2, A2, A3, A4,
  A4X, G4); Cloud Run GPU documentation.
- R. Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023 — inference cost and
  partitioning analysis in the same spirit.
- J. Kaplan et al., "Scaling Laws for Neural Language Models", 2020 — the 2N FLOPs-per-token accounting.
- M. Shoeybi et al., "Megatron-LM", 2019 — tensor parallelism with two all-reduces per layer.
- P. Patarasuk, X. Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations", *JPDC* 69(2),
  2009; R. Thakur, R. Rabenseifner, W. Gropp, "Optimization of Collective Communication Operations in MPICH",
  *IJHPCA* 19(1), 2005.
- NVIDIA nccl-tests, `doc/PERFORMANCE.md` — algbw and busbw definitions.
- M. Al-Fares, A. Loukissas, A. Vahdat, "A Scalable, Commodity Data Center Network Architecture",
  SIGCOMM 2008 — fat trees from commodity switches; NVIDIA DGX SuperPOD reference architecture — rail-optimized
  scale-out.
- NVIDIA `nvidia-smi` documentation (`topo -m` legend) and GPUDirect RDMA documentation.
- J. W. Young, "A first order approximation to the optimum checkpoint interval", *CACM* 17(9), 1974;
  J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps", *FGCS* 22(3), 2006.
- Llama Team, Meta, "The Llama 3 Herd of Models", arXiv:2407.21783, 2024 — §3.3.4, reliability of a
  16K-GPU training run.
- W. Kwon et al., PagedAttention (SOSP 2023); T. Dao et al., FlashAttention (2022–2024); A. Agrawal et al.,
  Sarathi-Serve (OSDI 2024) — the engine techniques this layer's arithmetic motivates (layer 04).
- Model configurations from each model's published `config.json`: Llama-3.1-8B/70B, Mixtral-8x7B,
  Qwen2.5-1.5B, Qwen3-30B-A3B.

---

## Verify list

Product facts in this primer and in `roofline-core/roofline/specs.py`, as of September 2026:

- Every entry of the §9 table, and the status of the parts outside it (Vera Rubin, MI400 series): dense
  peaks per precision, memory capacity and bandwidth, scale-up bandwidth and domain size, TDP. Especially the derived ones — RTX PRO 6000 tensor rates (from its sparse FP4
  headline), TPU7x bf16 (assumed half of fp8), B200 usable memory (180 of 192 GB), GB200 memory (186 GB),
  GB300 TDP, MI355X xGMI bandwidth — and the semantics of TPU ICI figures.
- SM counts and boost clocks behind `peak_from_clock()` (T4 40 / 1.59 GHz, L4 58 / 2.04 GHz, A100 108 /
  1.41 GHz, H100 SXM 132 / 1.83 GHz implied by 989.4 TFLOP/s).
- H100 PCIe differences (SMs, HBM2e ~2 TB/s, 350 W); the MiB figure `nvidia-smi` reports for an "80 GB" H100.
- Link rates: NVLink 3/4/5 per-GPU link counts and speeds; PCIe Gen4/Gen5 x16; InfiniBand NDR/XDR.
  The α values are illustrative, not product facts.
- DGX H100 maximum system power (10.2 kW) used in §8.3; the $300k server price and opex are assumptions.
- GCP prices (L4 ~$0.70/hr, H100 ~$11/GPU-hr on demand as `a3-highgpu-8g` ≈ $88/hr, Spot ~$3.7/GPU-hr),
  Spot discount range (60–91%), machine families and their GPUs and NICs, A3 small-shape availability,
  DWS flex-start (up to 7 days) and calendar mode, Cloud Run GPU types, TPU7x GA date (2026-04-22), quota
  behaviour for new accounts.
- Free and cheap tiers: Colab (T4, hours per week), Kaggle (2×T4 or P100, 30 GPU-hours per week), Vast.ai,
  RunPod, Lambda, Modal offerings — maintained in `COMPUTE.md`.
- The Run:ai Model Streamer as a vLLM load format; loader behaviour of safetensors.
- The Llama 3 interruption figures (419 unexpected in 54 days on 16,384 GPUs; ~78% hardware).
- Assumptions, not product facts (replace with your own): α values, storage tier bandwidths, cold-start
  stage times, the 10-minute restart, the 48 h MTTR, the server price and opex.
