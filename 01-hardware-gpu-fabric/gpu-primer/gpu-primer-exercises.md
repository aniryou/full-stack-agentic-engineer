# GPU Primer — Exercises

This file is the companion to `gpu-primer.md`. You can answer all of the questions here with the primer and arithmetic.

Do Parts A–D first. All of the answers are in Part E at the end. Thus, do not scroll down to them.

**Reference constants that you will need:**

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

Give each answer in one or two sentences. You do not need arithmetic.

**A1.** A CPU and a GPU must both wait ~500 cycles for a DRAM access. Describe how each one uses that wait. Then say, as a result, on which hardware each one spends its transistor budget.

**A2.** What is a warp? Why is the warp the unit that is really important, and not the thread?

**A3.** Shared memory and the L1 cache are at the same level of the hierarchy. They have almost the same latency. For the programmer, what is the basic difference between them?

**A4.** Give the definition of arithmetic intensity. What are its units? Which question does it let you answer before you write any code?

**A5.** An H100 has ~67 TFLOP/s of FP32 and ~990 TFLOP/s of BF16. These two numbers do not come from the same units of hardware. Explain what changed between them.

**A6.** BF16 and FP16 both have 16 bits. Why did BF16, and not FP16, become the format for training?

**A7.** What does TMA do that a usual `memcpy_async` does not do? Which problem was the reason to add TMA?

**A8.** Tell the difference between *scale-up* and *scale-out*. Which one is necessary for tensor parallelism? Why?

**A9.** Name, in sequence, the four layers of the NVIDIA software stack between "PyTorch eager" and "inline PTX". For each layer, say in one clause what it gives you.

**A10.** What is the difference in structure between the CUDA Tile model and the SIMT model? What does each model describe?

---

## Part B — Numerical exercises

Show the steps of your calculation. You can round the numbers freely. The order of magnitude and your logic are the important things.

### B1 — Memory budget

Use a 70B-parameter model in BF16.

(a) How much memory do the weights need?

(b) For inference, does it fit on one H100 (80 GB)? Does it fit on one H200 (141 GB)? Does it fit on one B300 (288 GB)?

(c) You want to *train* it with Adam. How much memory do the model states alone need, before the activations? What is the minimum number of H100s for that memory?

(d) In one sentence, explain why (c) is so much larger than (a).

### B2 — The decode roofline

A 30B model in BF16 does single-stream inference (batch size 1) on one H100.

(a) How many bytes must the GPU read from HBM to make one token?

(b) How many FLOPs does the GPU do?

(c) What is the arithmetic intensity? Is this kernel memory-bound or compute-bound? By what factor is it away from the crossover?

(d) What is the maximum token rate that it can get?

(e) What fraction of the peak BF16 throughput of the H100 do you use in practice?

The number from part (e) is the one to remember.

### B3 — What batching buys you

Use the same model and hardware as B2, but now with batch size $B$. Assume that the GPU reads the weights one time per step, and that all of the batch shares them. Ignore the KV cache traffic.

(a) Write arithmetic intensity as a function of $B$.

(b) At what batch size does the kernel become compute-bound?

(c) Calculate the token throughput and the percentage of peak at $B$ = 64.

(d) Why can you not set $B$ to the answer from (b)? Which physical limit stops you?

### B4 — Fusion

A model applies four elementwise operations in sequence to a 1 GB activation tensor: GELU, a scalar multiply, a bias add and a cast.

(a) How much HBM traffic do four separate kernels cause?

(b) How much traffic does one fused kernel cause?

(c) Calculate the two times on an H100. Then calculate the speedup.

(d) Approximately, what is the arithmetic intensity here? What does that tell you about the value of an effort to optimise the *math* in these kernels?

### B5 — KV cache

The model has 80 layers, 8 KV heads and head dimension 128, in BF16.

(a) Calculate the KV cache size per token, per sequence.

(b) At 128K context, how much is that for a single sequence?

(c) The model has 70B parameters. Compare the KV cache for four concurrent 128K sequences with the weights.

(d) Now assume that the model uses plain multi-head attention with 64 KV heads, not 8. Calculate (b) again. What does this tell you about the reason for GQA?

### B6 — Coalescing

A warp reads 32 float32 values from a row-major 1024×1024 matrix. HBM transactions are 32 bytes.

(a) The warp reads 32 consecutive elements of one row. How many bytes does the hardware fetch? How many does it use?

(b) The warp reads 32 elements down one column. Give the same two numbers.

(c) What is the efficiency ratio between them?

### B7 — Divergence

In a warp, lane $i$ executes a loop that runs ${i+1}$ times (lane 0 runs it one time, lane 31 runs it 32 times).

(a) How many iterations of warp-time does this cost?

(b) How much useful lane-work does the warp do in fact?

(c) What is the lane utilisation?

### B8 — Interconnect

An 8-way tensor-parallel 70B model does a prefill pass on a 4096-token prompt. The hidden size is 8192, the format is BF16, the model has 80 layers, and each layer does two all-reduces. As an approximation, each all-reduce moves one copy of the activation tensor per GPU.

(a) Calculate the size of one activation tensor.

(b) Calculate the total communication volume per forward pass.

(c) Calculate the time over NVLink 5 and over PCIe Gen5 x16.

(d) The compute itself takes approximately 180 ms across the 8 GPUs. Give the cost of each interconnect as a percentage overhead. Then state the conclusion.

### B9 — MFU

A training run on 8,192 H100s keeps a steady rate of 1.2M tokens/second on a 405B-parameter model.

(a) What is the total achieved FLOP/s?

(b) What is the total peak?

(c) What is the Model FLOPs Utilisation? Is this good?

### B10 — Quantisation

You quantise the 70B model from B1 to MXFP4 (4 bits per weight).

(a) What is the new weight footprint? Does it fit on one H100 now?

(b) Calculate the batch-1 decode token rate on an H100, before and after.

(c) Where did the speedup come from: more FLOPs, or fewer bytes? Use the roofline to show why.

---

## Part C — Diagnostics

For each case, say what occurs. Then say what you will do next.

**C1.** Nsight Compute shows your kernel at 87% of peak DRAM throughput and 3% of peak Tensor Core throughput. A colleague recommends that you replace your inner loop with a formulation that has a lower instruction count. Is this a good idea?

**C2.** A different kernel shows 12% DRAM throughput and 78% Tensor Core throughput. What is the state of this kernel? What is the next lever?

**C3.** `torch.compile` gave you a 2.5× speedup on training, but almost nothing on batch-1 decode. Why? What will really help decode?

**C4.** A hand-written attention kernel gets 60% of peak on A100, but only 25% of peak on H100. The kernel is unchanged. What is the probable explanation?

**C5.** Your GNN message-passing kernel gets 4% of peak on an H100. Your matmul kernel gets 70%. Is the GNN kernel broken?

**C6.** On your home cluster, a model fits easily in the VRAM of one GPU. The model runs *slower* when you shard it 4-way with tensor parallelism across four consumer cards. Explain why, and give two alternatives.

---

## Part D — Synthesis

These questions need longer answers. It is worth the effort to be able to say these answers aloud.

**D1.** Explain why prefill and decode have opposite bottlenecks. Also explain why that is the reason for disaggregated serving. Which type of hardware do you want for each half?

**D2.** FlashAttention does *more* floating-point operations than a simple attention implementation, but it is several times faster. Explain why, and state the general principle that it shows.

**D3.** You write the hardware specification for a workload. Half of the workload is physics-informed neural network training (FP64-sensitive PDE residuals). The other half is GNN message passing over irregular meshes. Which parts of the progress of the last three GPU generations help you, and which parts do not? What is the relation of AMD's MI430X/MI450X split to this?

**D4.** A vendor changes the scale-up domain from 8 GPUs to 72. In concrete terms, how does that change the way that you parallelise a large model? Why?

**D5.** Here are two mental models. The first is "a GPU is a memory system with arithmetic attached". The second is "a GPU is a fast calculator." State the argument that the first is the more useful mental model. As evidence, use a minimum of three specific results from Part B.

---

---

## Part E — Answers

### Part A

**A1.** The CPU tries to *prevent* the wait. It spends transistors on large multi-level caches, prefetchers, branch prediction and out-of-order execution. Thus, a single instruction stream rarely stalls.

The GPU *accepts* the wait and hides it. It keeps thousands of other threads resident, and on every stall it changes to a warp that is ready. The GPU does not need most of the control logic. Thus, it spends the area on ALUs and memory bandwidth instead. The CPU is latency-optimised, and the GPU is throughput-optimised.

**A2.** A warp is 32 threads that share a single instruction pointer and execute in lockstep. The warp is important because the hardware schedules it as one unit. It is also the unit that memory coalescing operates on. The cost is per warp, not per thread. Thus, when 32 threads do different things, the cost is the sum of their paths, not the maximum of their work.

**A3.** The hardware manages L1. You have an effect on it only indirectly, through your access patterns. Shared memory is a software-managed scratchpad that you explicitly allocate, load and synchronise on. CPUs have no equivalent. Most of kernel optimisation is the decision about what data to stage there.

**A4.** Arithmetic intensity is the number of FLOPs done per byte moved from HBM, in FLOP/byte. It tells you if your ceiling is the ALUs or the memory bus. It tells you this before you write code. To find the answer, compare it with the crossover ratio of the machine (~295 FLOP/byte on an H100). If you are below the crossover, arithmetic optimisations cannot help you.

**A5.** The FP32 number is scalar fused-multiply-add on the CUDA cores, with one operation per lane. The BF16 number comes from Tensor Cores. These are dedicated units that consume small matrix tiles. They issue a whole matrix multiply-accumulate per instruction, and a warp or warpgroup feeds them cooperatively. A kernel that does not reach the Tensor Cores leaves approximately 95% of the chip idle.

**A6.** BF16 keeps the 8-bit exponent of FP32 and gives up mantissa bits. FP16 does the opposite. Training is much more sensitive to dynamic range than to precision (gradients underflow). Thus, training in BF16 is stable without loss scaling.

**A7.** TMA is a dedicated DMA engine. One thread issues a descriptor, and the hardware moves an entire multidimensional tile. The hardware also generates the addresses and handles the boundaries. TMA removes the index arithmetic and the register pressure that were before the largest part of the inner loop. The warp is then free to do only one thing: issue Tensor Core work.

**A8.** Scale-up makes the NVLink domain larger. In that domain, each GPU can use the memory of the other GPUs as almost local (1.8 TB/s). Scale-out adds nodes over InfiniBand or Ethernet, which is approximately an order of magnitude slower. Tensor parallelism needs an all-reduce *inside every layer*, so it needs scale-up. If you put it across a scale-out boundary, communication becomes the largest cost.

**A9.** The four layers are:

- `torch.compile`. It traces and fuses automatically, and its output is Triton.
- Triton. You write kernels at the level of a block of elements. The compiler handles threads and coalescing.
- CUTLASS/CuTe. It gives C++ templates and a layout algebra for matmul-shaped problems.
- CUDA C++. It gives full control and warp-level intrinsics.

**A10.** SIMT describes what *one thread* does, and you think in terms of thread indices. CUDA Tile describes operations on *tiles and arrays*, in the NumPy style. The compiler then derives the thread mapping, the memory movement, the asynchrony and the Tensor Core selection. CUDA Tile is a complementary model, added next to SIMT. It is not a replacement.

---

### Part B

**B1.**

(a) 70e9 × 2 bytes = **140 GB**.

(b) On an H100 (80 GB), it does not fit. On an H200 (141 GB), it fits technically, with ~1 GB left. Thus, there is no space for the KV cache, and you cannot use it in practice. On a B300 (288 GB), it fits easily.

(c) 16 bytes/param × 70e9 = **1.12 TB** for weights, gradients and the two Adam moments. At 80 GB per H100, that is **14 GPUs minimum** for the states alone. In practice, you need more when you add the activations and fragmentation.

(d) Inference needs only the weights. Training also holds gradients and two FP32 optimiser moments per parameter, for approximately 8× the footprint.

**B2.**

(a) The GPU reads all the weights: 30e9 × 2 = **60 GB**.

(b) 2 FLOPs/param/token × 30e9 = **60 GFLOP**.

(c) 60e9 FLOP / 60e9 bytes = **1 FLOP/byte**. The kernel is memory-bound, and it is about **295× below** the crossover.

(d) 3.35e12 / 60e9 = **~56 tokens/second**.

(e) The achieved throughput is 60 GFLOP × 56/s = 3.35 TFLOP/s. Against 990 TFLOP/s peak, that is **0.34%**.

You use a third of one percent of the arithmetic hardware. Nothing that you do to the math will change that.

**B3.**

(a) The bytes stay at 60 GB, because the GPU reads the weights one time. The FLOPs become $B$ × 60 GFLOP. Intensity = **$B$ FLOP/byte**.

(b) The kernel becomes compute-bound when $B$ ≥ **~295**.

(c) At $B$ = 64, throughput = 64 × 56 = **~3,600 tokens/s**. The achieved rate = 214 TFLOP/s = **~22% of peak**. This is a 64× throughput gain at no arithmetic cost.

(d) The KV cache memory stops you. Each concurrent sequence needs its own cache, and at long context that cache grows faster than the weights (see B5). Memory capacity, not compute, sets the practical batch ceiling.

**B4.**

(a) Each kernel reads 1 GB and writes 1 GB. Four kernels = **8 GB**.

(b) One read and one write = **2 GB**.

(c) 8e9/3.35e12 = **2.4 ms** unfused. 2e9/3.35e12 = **0.6 ms** fused. The result is a **4× speedup.**

(d) Each element needs a few FLOPs against 8 bytes of traffic. Thus, the intensity is approximately 0.5–1 FLOP/byte. That is three hundred times below the crossover. An effort to optimise the arithmetic has no value. The only lever is traffic, and fusion removes exactly that traffic.

**B5.**

(a) 2 (K and V) × 80 layers × 8 heads × 128 dim × 2 bytes = **327,680 bytes ≈ 320 KB per token**.

(b) 327,680 × 131,072 = **~43 GB for one sequence**.

(c) Four sequences = **~172 GB**, against 140 GB of weights. The cache is larger than the model.

(d) With 64 KV heads, it is 8× larger: **~343 GB for a single sequence**. That is more than any single accelerator has. GQA exists because the KV cache of multi-head attention makes long context physically impossible, not only slow.

**B6.**

(a) The warp needs 32 × 4 = 128 bytes. The hardware merges these into 4 × 32-byte transactions. **128 fetched, 128 used, 100%.**

(b) The stride is 4096 bytes. Thus, every lane goes into a different transaction: **32 × 32 = 1024 bytes fetched, 128 used, 12.5%.**

(c) There is **8×** more traffic for the same useful work. The instruction count is the same and the FLOPs are the same, but the access is eight times slower.

**B7.**

(a) The warp runs until its longest lane finishes: **32 iterations** of warp-time.

(b) The sum of 1 to 32 = **528 lane-iterations** of useful work.

(c) 528 / (32 × 32) = **51.6%**. Half of the machine is idle. The only cause is a loop bound that changes from lane to lane.

**B8.**

(a) 4096 × 8192 × 2 bytes = **67 MB**.

(b) 67 MB × 2 per layer × 80 layers = **~10.7 GB**.

(c) NVLink: 10.7e9/1.8e12 = **~6 ms**. PCIe: 10.7e9/64e9 = **~167 ms**.

(d) Against 180 ms of compute, NVLink adds **~3%**. PCIe adds **~93%** and almost doubles the time of the pass. Tensor parallelism is viable on NVLink and not viable on PCIe. This is the whole reason for NVLink domains. It is also the reason that consumer multi-GPU machines cannot do TP.

**B9.**

(a) 6 × 405e9 × 1.2e6 = **2.92e18 FLOP/s**.

(b) 8192 × 990e12 = **8.11e18 FLOP/s**.

(c) **36% MFU.** This is just below the 40–50% "good" band. For a run at that scale, it is a respectable result. There is still space to examine communication overlap and pipeline bubbles.

**B10.**

(a) 70e9 × 0.5 bytes = **35 GB**. Yes, it now fits on one 80 GB H100, with ~45 GB left for KV cache.

(b) Before: 3.35e12/140e9 = 24 tok/s (if it fit). After: 3.35e12/35e9 = **~96 tok/s**. The rate is 4× higher.

(c) **Fewer bytes.** At batch 1, you are at ~1 FLOP/byte, hundreds of times below the crossover. Thus, the added FP4 arithmetic throughput has no effect. All of the gain comes from one thing: you moved a quarter as much data across the same bus. This is the general shape of the gains from quantisation in inference.

---

### Part C

**C1.** No. You are memory-bound at 87% of the bandwidth ceiling, and you use 3% of the arithmetic. A lower instruction count changes nothing. All of the levers are about traffic. Fuse with adjacent kernels, correct the access pattern for coalescing, stage reuse in shared memory, or use a narrower dtype.

**C2.** The kernel is healthy and compute-bound. Bandwidth is not the limit. Thus, fusion and work on the access pattern will not help.

The next lever is precision. If accuracy permits, change from BF16 to FP8 or FP4. Each step down the ladder approximately doubles the throughput. After that, make sure that you use the current Tensor Core path (warpgroup MMA, TMA-fed pipelines), not an older instruction.

**C3.** Most of the gain from `torch.compile` comes from the fusion of elementwise ops, which removes HBM round trips. Training has many such ops between large matmuls, so there is much to remove. In batch-1 decode, the largest cost is the read of every weight, and you cannot prevent that read. No quantity of fusion removes it. The real solutions are:

- Increase the batch size (B3).
- Quantise (B10).
- Reduce the bytes-per-token in a different way, for example with speculative decoding or an MoE that activates a subset of weights.

**C4.** The kernel targets the Ampere execution model. The increase in peak on H100 comes from features that the kernel does not use. These features are TMA for tile movement, warpgroup-level MMA, warp specialisation and thread block clusters.

Also, over A100, H100 gained approximately 3× the FLOPS but only ~1.7× the bandwidth. Thus, an unchanged kernel also moves slowly toward the memory-bound condition. The code is the same, but the fit is worse.

**C5.** It is almost certainly not broken. Message passing is a gather/scatter over irregular neighbour lists. Its accesses are uncoalesced (B6 territory), and the work is latency-bound, not bandwidth-bound. The work does not divide into dense tiles, so by construction the Tensor Cores stay idle.

"Percent of peak" tells you almost nothing here, because peak is a dense-matmul number. Compare the kernel with achieved bandwidth and with a CPU baseline instead.

**C6.** Consumer cards have had no NVLink since the 40-series. Thus, your shards communicate over PCIe. Tensor parallelism needs an all-reduce inside every layer, and B8 shows the cost of that on PCIe. The alternatives are:

- (i) Do not shard the model at all, because it fits. Run one GPU per model. Use the other three for concurrent replicas.
- (ii) If you later need a model that does not fit, use pipeline parallelism. It is point-to-point and needs much less communication. Or quantise the model until it fits again.

---

### Part D

**D1.** Prefill processes the whole prompt at one time. Thus, thousands of tokens share the cost of each weight read. The intensity is high, prefill saturates the Tensor Cores, and it is compute-bound. Decode makes one token per step per sequence, and it reads every weight again each time. Its intensity is near 1, and it is bandwidth-bound (B2).

If you run them on the same hardware, either the ALUs do not get sufficient work during decode, or you waste bandwidth during prefill. Also, long prefills block short decodes in the queue.

Disaggregation lets you set the size of each pool independently. Prefill wants maximum FLOPS and does not need much memory bandwidth per token. Decode wants maximum HBM bandwidth and capacity for KV cache. NVIDIA went so far with this that it built a separate SKU for the prefill half, Rubin CPX. The KV cache that prefill makes must then go to the decode pool. That is why these designs are inside a fast scale-up domain.

**D2.** Simple attention materialises the full $N \times N$ score matrix in HBM. It writes the matrix, then reads it back for softmax. Then it writes again, and reads again for the value multiply. FlashAttention tiles the computation, and it makes, uses and discards each block of scores entirely in shared memory. It uses an online-softmax reformulation, so it never needs the full matrix.

It recomputes some quantities, so it does more arithmetic. But it moves an order of magnitude fewer bytes. The kernel was memory-bound. Thus, an exchange of FLOPs for bytes is an exchange of an abundant resource for the scarce one. The general principle is this: **on a memory-bound kernel, redundant computation is free and data movement is the only real cost.** Recomputation in gradient checkpointing is the same trade.

**D3.** Most of the progress of the last three generations is progress on the precision ladder for dense low-precision matmul. It includes FP8 on Hopper, FP4 and microscaling on Blackwell, and the Transformer Engine machinery around them. None of that helps FP64 PDE residuals. Also, FP64 rates have grown much less than the headline numbers.

These things *do* help you:

- The growth of HBM capacity and bandwidth, which helps every workload.
- TMA and async pipelines, which help any tiled kernel.
- The improved tools. Triton and cuTile decrease the cost to write the custom kernels that irregular workloads need anyway.

The progress helps the GNN half even less. It is gather/scatter-bound, and its structure makes it unable to use Tensor Cores (C5).

AMD's split is the direct result. From MI400, AMD ships MI450X for AI, with the FP32/FP64 logic removed. It also ships MI430X for HPC, with FP4/FP8/BF16 removed. Each chip gets back that die area. The practical result is that "the best AI chip" and "the best chip for your workload" now move apart. Thus, measure FP64 throughput and achieved bandwidth on irregular access with benchmarks, and do not read TFLOPS off a slide.

**D4.** With an 8-GPU domain, tensor parallelism can go only 8-wide before it crosses onto the slow network. Thus, you must divide a large model more. You can use pipeline parallelism, which brings bubbles and complex schedules, or you can shard the optimiser state. With 72 GPUs in one domain, you can run much wider TP. Also, expert parallelism for MoE becomes practical, because the all-to-all routing stays on NVLink.

The knock-on effects are these:

- Pipeline depth can decrease or go away, and this removes the bubble overhead.
- Larger models fit in the aggregate memory of a single NVLink domain.
- Inference can hold much larger KV caches across the domain. This permits longer context and a higher batch.

In general, when more of your communication stays inside the scale-up domain, the whole rack operates more like one large GPU. That is exactly the design intent of NVL72 and NVL144.

**D5.** Every quantitative result points in the same direction:

- Batch-1 decode uses 0.34% of the arithmetic hardware, and only the weight reads limit it (B2).
- Four elementwise kernels take 4× longer than one fused kernel for the same arithmetic. The cause is 8 GB of traffic against 2 GB (B4).
- An uncoalesced access pattern is 8× slower with the same instruction count and the same FLOPs (B6).
- Quantisation to FP4 gives a 4× decode speedup. All of it comes from fewer bytes moved, and the added arithmetic throughput gives nothing (B10).
- Only the interconnect, NVLink or PCIe, decides if tensor parallelism is viable or not viable (B8).

In every case, the arithmetic was never the constraint. The best way to understand the chip is as a memory hierarchy with bandwidth at each level. The design problem is to keep data as high in that hierarchy as possible, for as long as possible. The ALUs are then, in effect, free. This is also why the headline specs on modern accelerators are HBM capacity and bandwidth. It is also why the roofline model predicts performance better than any FLOPS count.
