# GPU Primer — Exercises

Companion to `gpu-primer.md`. Everything here is answerable from the primer plus arithmetic.

Work through Parts A–D first. Answers are all in Part E at the end, so don't scroll.

**Reference constants you'll need:**

| | |
|---|---|
| H100 HBM bandwidth | 3.35 TB/s |
| H100 HBM capacity | 80 GB |
| H100 BF16 Tensor Core peak | 990 TFLOP/s |
| H100 FP32 CUDA core peak | 67 TFLOP/s |
| Roofline crossover (H100, BF16) | ~295 FLOP/byte |
| NVLink 5 per-GPU bandwidth | 1.8 TB/s |
| PCIe Gen5 x16 | ~64 GB/s |
| Warp size | 32 |
| Bytes per parameter, BF16 | 2 |
| FLOPs per parameter per token, forward | 2 |
| FLOPs per parameter per token, training step | 6 |
| Adam training memory | ~16 bytes/param |

---

## Part A — Recall

Answer in one or two sentences each. No arithmetic needed.

**A1.** A CPU and a GPU both have to wait ~500 cycles for a DRAM access. Describe how each one spends that wait, and what hardware each spends its transistor budget on as a result.

**A2.** What is a warp, and why is it the unit that actually matters rather than the thread?

**A3.** Shared memory and L1 cache sit at the same level of the hierarchy and have similar latency. What is the fundamental difference between them from the programmer's point of view?

**A4.** Define arithmetic intensity. What are its units, and what question does it let you answer before you write any code?

**A5.** An H100 has ~67 TFLOP/s of FP32 and ~990 TFLOP/s of BF16. These are not the same units of hardware. Explain what changed between them.

**A6.** BF16 and FP16 are both 16 bits. Why did BF16 win for training?

**A7.** What does TMA do that a normal `memcpy_async` doesn't, and what problem was it introduced to solve?

**A8.** Distinguish *scale-up* from *scale-out*. Which one does tensor parallelism require, and why?

**A9.** Name the four layers of the NVIDIA software stack between "PyTorch eager" and "inline PTX", in order, and say in one clause what each buys you.

**A10.** What is the structural difference between the CUDA Tile model and the SIMT model? What is being described in each case?

---

## Part B — Numerical exercises

Show your working. Round freely; order of magnitude and the reasoning are what matter.

### B1 — Memory budget

A 70B-parameter model, BF16.

(a) How much memory do the weights need?
(b) Does it fit for inference on one H100 (80 GB)? One H200 (141 GB)? One B300 (288 GB)?
(c) You want to *train* it with Adam. How much memory do the model states alone need, before activations? How many H100s is that, at minimum?
(d) Explain in one sentence why (c) is so much larger than (a).

### B2 — The decode roofline

A 30B model, BF16, running single-stream inference (batch size 1) on one H100.

(a) How many bytes must be read from HBM to produce one token?
(b) How many FLOPs are performed?
(c) What is the arithmetic intensity? Is this kernel memory-bound or compute-bound, and by what factor is it away from the crossover?
(d) What is the maximum achievable token rate?
(e) What fraction of the H100's peak BF16 throughput are you actually using?

Part (e) is the number to remember.

### B3 — What batching buys you

Same model and hardware as B2, now with batch size *B*. Assume weights are read once per step and shared across the batch, and ignore KV cache traffic.

(a) Write arithmetic intensity as a function of *B*.
(b) At what batch size does the kernel become compute-bound?
(c) Compute token throughput and percentage of peak at B = 64.
(d) What physically stops you from just setting *B* to the answer from (b)?

### B4 — Fusion

A model applies four elementwise operations in sequence to a 1 GB activation tensor: GELU, a scalar multiply, a bias add, and a cast.

(a) How much HBM traffic do four separate kernels generate?
(b) How much does one fused kernel generate?
(c) Compute both times on an H100 and the speedup.
(d) Roughly what is the arithmetic intensity here, and what does that tell you about whether optimising the *math* in these kernels is worth any effort?

### B5 — KV cache

A model with 80 layers, 8 KV heads, head dimension 128, BF16.

(a) Compute the KV cache size per token, per sequence.
(b) At 128K context, how much is that for a single sequence?
(c) The model has 70B parameters. Compare the KV cache for four concurrent 128K sequences against the weights.
(d) Now suppose the model used plain multi-head attention with 64 KV heads instead of 8. Recompute (b). What does this tell you about why GQA exists?

### B6 — Coalescing

A warp reads 32 float32 values from a row-major 1024×1024 matrix. HBM transactions are 32 bytes.

(a) The warp reads 32 consecutive elements of one row. How many bytes are fetched, and how many are used?
(b) The warp reads 32 elements down one column. Same two numbers.
(c) What is the efficiency ratio between them?

### B7 — Divergence

Inside a warp, lane *i* executes a loop that runs `i + 1` times (lane 0 runs once, lane 31 runs 32 times).

(a) How many iterations of warp-time does this cost?
(b) How much useful lane-work is actually performed?
(c) What is the lane utilisation?

### B8 — Interconnect

An 8-way tensor-parallel 70B model does a prefill pass on a 4096-token prompt. Hidden size 8192, BF16, 80 layers, two all-reduces per layer. Approximate each all-reduce as moving one copy of the activation tensor per GPU.

(a) Size of one activation tensor.
(b) Total communication volume per forward pass.
(c) Time over NVLink 5, and over PCIe Gen5 x16.
(d) The compute itself takes roughly 180 ms across the 8 GPUs. Express each interconnect's cost as a percentage overhead, and state the conclusion.

### B9 — MFU

A training run on 8,192 H100s sustains 1.2M tokens/second on a 405B-parameter model.

(a) What is the aggregate achieved FLOP/s?
(b) What is the aggregate peak?
(c) What is the Model FLOPs Utilisation? Is this good?

### B10 — Quantisation

You quantise the 70B model from B1 to MXFP4 (4 bits per weight).

(a) New weight footprint. Does it now fit on one H100?
(b) Batch-1 decode token rate before and after, on an H100.
(c) Where did the speedup come from: more FLOPs, or fewer bytes? Justify from the roofline.

---

## Part C — Diagnostics

For each, say what's happening and what you'd do next.

**C1.** Nsight Compute reports your kernel at 87% of peak DRAM throughput and 3% of peak Tensor Core throughput. A colleague suggests replacing your inner loop with a lower-instruction-count formulation. Good idea?

**C2.** A different kernel reports 12% DRAM throughput and 78% Tensor Core throughput. What is the state of this kernel, and what is the next lever?

**C3.** `torch.compile` gave you a 2.5× speedup on training but essentially nothing on batch-1 decode. Why? What would actually help decode?

**C4.** A hand-written attention kernel hits 60% of peak on A100 but only 25% of peak on H100. It has not been modified. What is the likely explanation?

**C5.** Your GNN message-passing kernel achieves 4% of peak on an H100. Your matmul kernel achieves 70%. Is the GNN kernel broken?

**C6.** On your home cluster, a model that fits comfortably in one GPU's VRAM runs *slower* when you shard it 4-way with tensor parallelism across four consumer cards. Explain, and give two alternatives.

---

## Part D — Synthesis

Longer answers. These are the ones worth being able to give out loud.

**D1.** Explain why prefill and decode have opposite bottlenecks, and why that motivates disaggregated serving. What kind of hardware would you want for each half?

**D2.** FlashAttention performs *more* floating-point operations than a naive attention implementation, yet is several times faster. Explain, and state the general principle it illustrates.

**D3.** You are speccing hardware for a workload that is half physics-informed neural network training (FP64-sensitive PDE residuals) and half GNN message passing over irregular meshes. Which parts of the last three GPU generations' progress help you, which don't, and what does AMD's MI430X/MI450X split have to do with it?

**D4.** A vendor moves the scale-up domain from 8 GPUs to 72. Concretely, what does that change about how you would parallelise a large model, and why?

**D5.** State the argument for why "a GPU is a memory system with arithmetic attached" is a more useful mental model than "a GPU is a fast calculator." Use at least three specific results from Part B as evidence.

---
---

## Part E — Answers

### Part A

**A1.** The CPU tries to *avoid* the wait: it spends transistors on large multi-level caches, prefetchers, branch prediction, and out-of-order execution so a single instruction stream rarely stalls. The GPU *accepts* the wait and hides it by keeping thousands of other threads resident, switching to a ready warp on every stall. Freed of most control logic, it spends the area on ALUs and memory bandwidth instead. Latency-optimised versus throughput-optimised.

**A2.** A warp is 32 threads sharing a single instruction pointer, executing in lockstep. It matters because it is the unit the hardware schedules and the unit that memory coalescing operates over. Cost is charged per warp, not per thread, which is why 32 threads doing different things costs the sum of their paths rather than the max of their work.

**A3.** L1 is hardware-managed; you influence it only indirectly through access patterns. Shared memory is a software-managed scratchpad that you explicitly allocate, load, and synchronise on. CPUs have no equivalent, and deciding what to stage there is most of what kernel optimisation consists of.

**A4.** FLOPs performed per byte moved from HBM, in FLOP/byte. It tells you, before writing code, whether your ceiling is the ALUs or the memory bus, by comparing against the machine's crossover ratio (~295 FLOP/byte on an H100). If you're below it, arithmetic optimisations cannot help you.

**A5.** The FP32 number is scalar fused-multiply-add on the CUDA cores, one operation per lane. The BF16 number comes from Tensor Cores, dedicated units that consume small matrix tiles and issue a whole matrix multiply-accumulate per instruction, fed cooperatively by a warp or warpgroup. A kernel that doesn't reach the Tensor Cores leaves roughly 95% of the chip idle.

**A6.** BF16 keeps FP32's 8-bit exponent and gives up mantissa bits. FP16 does the opposite. Training is far more sensitive to dynamic range than to precision (gradients underflow), so BF16 trains stably without loss scaling.

**A7.** TMA is a dedicated DMA engine: one thread issues a descriptor and the hardware moves an entire multidimensional tile, including address generation and boundary handling. It removes the index arithmetic and register pressure that used to dominate the inner loop, and frees the warp to do nothing but issue Tensor Core work.

**A8.** Scale-up grows the NVLink domain, where GPUs can treat each other's memory as nearly local (1.8 TB/s). Scale-out adds nodes over InfiniBand or Ethernet, roughly an order of magnitude slower. Tensor parallelism needs an all-reduce *inside every layer*, so it requires scale-up; put it across a scale-out boundary and communication dominates.

**A9.** `torch.compile` (traces and fuses automatically, emitting Triton), Triton (write kernels at block-of-elements level, compiler handles threads and coalescing), CUTLASS/CuTe (C++ templates and a layout algebra for matmul-shaped problems), CUDA C++ (full control, warp-level intrinsics).

**A10.** SIMT describes what *one thread* does, and you reason about thread indices. CUDA Tile describes operations on *tiles and arrays*, NumPy-style, and the compiler derives the thread mapping, memory movement, asynchrony, and Tensor Core selection. It is a complementary model added alongside SIMT, not a replacement.

---

### Part B

**B1.**
(a) 70e9 × 2 bytes = **140 GB**.
(b) H100 (80 GB): no. H200 (141 GB): technically yes, with ~1 GB left, so no room for KV cache — not usable in practice. B300 (288 GB): yes, comfortably.
(c) 16 bytes/param × 70e9 = **1.12 TB** for weights, gradients, and the two Adam moments. At 80 GB per H100 that is **14 GPUs minimum** for states alone, and realistically more once activations and fragmentation are accounted for.
(d) Inference needs weights only; training additionally holds gradients and two FP32 optimiser moments per parameter, roughly 8× the footprint.

**B2.**
(a) All the weights: 30e9 × 2 = **60 GB**.
(b) 2 FLOPs/param/token × 30e9 = **60 GFLOP**.
(c) 60e9 FLOP / 60e9 bytes = **1 FLOP/byte**. Memory-bound, and about **295× below** the crossover.
(d) 3.35e12 / 60e9 = **~56 tokens/second**.
(e) Achieved throughput = 60 GFLOP × 56/s = 3.35 TFLOP/s. Against 990 TFLOP/s peak, that is **0.34%**.

You are using a third of one percent of the arithmetic hardware. Nothing you do to the math will change that.

**B3.**
(a) Bytes stay at 60 GB (weights read once); FLOPs become B × 60 GFLOP. Intensity = **B FLOP/byte**.
(b) When B ≥ **~295**.
(c) B = 64: throughput = 64 × 56 = **~3,600 tokens/s**; achieved = 214 TFLOP/s = **~22% of peak**. A 64× throughput gain at no arithmetic cost.
(d) KV cache memory. Each concurrent sequence needs its own cache, and at long context that grows faster than the weights (see B5). Memory capacity, not compute, sets the practical batch ceiling.

**B4.**
(a) Each kernel reads 1 GB and writes 1 GB. Four kernels = **8 GB**.
(b) Read once, write once = **2 GB**.
(c) 8e9/3.35e12 = **2.4 ms** unfused; 2e9/3.35e12 = **0.6 ms** fused. **4× speedup.**
(d) A handful of FLOPs per element against 8 bytes of traffic, so intensity is around 0.5–1 FLOP/byte. Three hundred times below the crossover. Optimising the arithmetic is worthless; the only lever is traffic, which is exactly what fusion removes.

**B5.**
(a) 2 (K and V) × 80 layers × 8 heads × 128 dim × 2 bytes = **327,680 bytes ≈ 320 KB per token**.
(b) 327,680 × 131,072 = **~43 GB for one sequence**.
(c) Four sequences = **~172 GB**, against 140 GB of weights. The cache is larger than the model.
(d) With 64 KV heads it is 8× larger: **~343 GB for a single sequence**. That is more than any single accelerator has. GQA exists because multi-head attention's KV cache makes long context physically impossible, not merely slow.

**B6.**
(a) 32 × 4 = 128 bytes needed; the hardware merges these into 4 × 32-byte transactions. **128 fetched, 128 used, 100%.**
(b) Stride is 4096 bytes, so every lane lands in a different transaction: **32 × 32 = 1024 bytes fetched, 128 used, 12.5%.**
(c) **8×** more traffic for identical useful work. Same instruction count, same FLOPs, eight times slower.

**B7.**
(a) The warp runs until its longest lane finishes: **32 iterations** of warp-time.
(b) Sum of 1 to 32 = **528 lane-iterations** of useful work.
(c) 528 / (32 × 32) = **51.6%**. Half the machine idle, from a loop bound that merely varies across lanes.

**B8.**
(a) 4096 × 8192 × 2 bytes = **67 MB**.
(b) 67 MB × 2 per layer × 80 layers = **~10.7 GB**.
(c) NVLink: 10.7e9/1.8e12 = **~6 ms**. PCIe: 10.7e9/64e9 = **~167 ms**.
(d) Against 180 ms of compute: NVLink adds **~3%**, PCIe adds **~93%** and nearly doubles the pass. Tensor parallelism is viable on NVLink and not viable on PCIe. This is the whole reason NVLink domains exist and the reason consumer multi-GPU rigs can't do TP.

**B9.**
(a) 6 × 405e9 × 1.2e6 = **2.92e18 FLOP/s**.
(b) 8192 × 990e12 = **8.11e18 FLOP/s**.
(c) **36% MFU.** Just under the 40–50% "good" band — respectable for a run at that scale, with room to look at communication overlap and pipeline bubbles.

**B10.**
(a) 70e9 × 0.5 bytes = **35 GB**. Yes, it now fits on one 80 GB H100 with ~45 GB left for KV cache.
(b) Before: 3.35e12/140e9 = 24 tok/s (if it fit). After: 3.35e12/35e9 = **~96 tok/s**. 4×.
(c) **Fewer bytes.** At batch 1 you sit at ~1 FLOP/byte, hundreds of times below the crossover, so the extra FP4 arithmetic throughput is irrelevant. The entire gain is that you moved a quarter as much data across the same bus. This is the general shape of quantisation wins in inference.

---

### Part C

**C1.** No. You are memory-bound at 87% of the bandwidth ceiling and using 3% of the arithmetic. Reducing instruction count changes nothing. The levers are all traffic: fuse with neighbouring kernels, fix the access pattern for coalescing, stage reuse in shared memory, or use a narrower dtype.

**C2.** Healthy and compute-bound. Bandwidth is not the limit, so fusion and access-pattern work will not help. The next lever is precision: move BF16 to FP8 or FP4 if accuracy permits, which roughly doubles throughput per step down the ladder. After that, check you're on the current Tensor Core path (warpgroup MMA, TMA-fed pipelines) rather than an older instruction.

**C3.** `torch.compile` mostly wins by fusing elementwise ops, which removes HBM round trips. Training has many such ops between large matmuls, so there is a lot to remove. Batch-1 decode is dominated by the unavoidable read of every weight, and no amount of fusion removes that. Real fixes: increase batch size (B3), quantise (B10), or reduce the bytes-per-token another way, e.g. speculative decoding or an MoE that activates a subset of weights.

**C4.** It was written against the Ampere execution model. H100's additional peak comes from features it isn't using: TMA for tile movement, warpgroup-level MMA, warp specialisation, thread block clusters. Note also that H100 gained roughly 3× the FLOPS but only ~1.7× the bandwidth over A100, so an unchanged kernel drifts toward being memory-bound as well. Same code, worse fit.

**C5.** Almost certainly not broken. Message passing is a gather/scatter over irregular neighbour lists: accesses are uncoalesced (B6 territory), the work is latency-bound rather than bandwidth-bound, and it does not decompose into dense tiles, so the Tensor Cores stay idle by construction. "Percent of peak" is close to meaningless here because peak is a dense-matmul number. Judge it against achieved bandwidth and against a CPU baseline instead.

**C6.** Consumer cards have had no NVLink since the 40-series, so your shards are communicating over PCIe. Tensor parallelism needs an all-reduce inside every layer, and B8 shows what that costs on PCIe. Alternatives: (i) don't shard at all, since it fits — run one GPU per model and use the other three for concurrent replicas; (ii) if you later need a model that doesn't fit, use pipeline parallelism, which is point-to-point and far less communication-intensive, or quantise it back down to fitting.

---

### Part D

**D1.** Prefill processes the whole prompt at once, so every weight read is amortised over thousands of tokens: intensity is high, it saturates the Tensor Cores, and it is compute-bound. Decode produces one token per step per sequence, re-reading every weight each time: intensity near 1, and it is bandwidth-bound (B2). Run them on the same hardware and you either starve the ALUs during decode or waste bandwidth during prefill, and long prefills block short decodes in the queue. Disaggregating lets you size each pool independently: prefill wants maximum FLOPS and does not need much memory bandwidth per token, decode wants maximum HBM bandwidth and capacity for KV cache. NVIDIA has taken this far enough to build a separate SKU for the prefill half, Rubin CPX. The KV cache produced by prefill then has to be shipped to the decode pool, which is why these designs live inside a fast scale-up domain.

**D2.** Naive attention materialises the full N×N score matrix in HBM, writes it, reads it back for softmax, writes again, reads again for the value multiply. FlashAttention tiles the computation so that block of scores is produced, consumed, and discarded entirely in shared memory, using an online-softmax reformulation so the full matrix is never needed. It recomputes some quantities, so it does more arithmetic, but it moves an order of magnitude fewer bytes. Since the kernel was memory-bound, trading FLOPs for bytes is trading an abundant resource for the scarce one. The general principle: **on a memory-bound kernel, redundant computation is free and data movement is the only real cost.** Recomputation in gradient checkpointing is the same trade.

**D3.** Most of the last three generations' progress is precision-ladder progress on dense low-precision matmul: FP8 on Hopper, FP4 and microscaling on Blackwell, and the Transformer Engine machinery around them. None of that helps FP64 PDE residuals, and FP64 rates have not grown remotely in line with the headline numbers. What *does* help you: HBM capacity and bandwidth growth, which benefits every workload; TMA and async pipelines, which help any tiled kernel; and the improved tooling, since Triton and cuTile lower the cost of writing the custom kernels irregular workloads need anyway. The GNN half is worse served still, being gather/scatter-bound and structurally unable to use Tensor Cores (C5). AMD's split is the direct consequence: from MI400 it ships MI450X for AI with FP32/FP64 logic stripped out and MI430X for HPC with FP4/FP8/BF16 stripped out, each reclaiming the die area. The practical implication is that "the best AI chip" and "the best chip for your workload" are actively diverging, and you should be benchmarking FP64 throughput and achieved bandwidth on irregular access rather than reading TFLOPS off a slide.

**D4.** With an 8-GPU domain, tensor parallelism can only go 8-wide before it crosses onto the slow network, so a large model must be split further by pipeline parallelism, which brings bubbles and complicated scheduling, or by sharding optimiser state. With 72 GPUs in one domain you can run much wider TP, and expert parallelism for MoE becomes practical because the all-to-all routing stays on NVLink. The knock-on effects are that pipeline depth can shrink or disappear, removing bubble overhead; larger models fit in a single NVLink domain's aggregate memory; and inference can hold much bigger KV caches across the domain, enabling longer context and higher batch. Broadly, the more of your communication that stays inside the scale-up domain, the more the whole rack behaves like one large GPU, which is precisely the design intent behind NVL72 and NVL144.

**D5.** Every quantitative result points the same way. Batch-1 decode uses 0.34% of the arithmetic hardware and is limited entirely by weight reads (B2). Four elementwise kernels take 4× longer than one fused kernel for identical arithmetic, because the difference is 8 GB of traffic versus 2 GB (B4). An uncoalesced access pattern is 8× slower with the same instruction count and the same FLOPs (B6). Quantisation to FP4 gives a 4× decode speedup that comes entirely from moving fewer bytes, with the added arithmetic throughput contributing nothing (B10). Tensor parallelism is viable or not viable depending purely on whether the interconnect is NVLink or PCIe (B8). In every case the arithmetic was never the constraint. The chip is best understood as a memory hierarchy with bandwidth at each level, and the design problem is keeping data as high in that hierarchy as possible for as long as possible; the ALUs are then, in effect, free. This is also why the headline specs on modern accelerators are HBM capacity and bandwidth, and why the roofline model predicts performance better than any FLOPS count.
