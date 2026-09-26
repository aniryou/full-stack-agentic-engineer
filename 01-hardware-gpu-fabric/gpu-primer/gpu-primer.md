# A GPU Primer

*From first principles up to the August 2026 state of the art.*

---

## 1. Why a GPU is shaped the way it is

Start with the transistor budget. If you build a CPU core from gates, you find that the arithmetic units are a small fraction of the die. Most of the area goes to making a *single* instruction stream finish fast: branch predictors, out-of-order scheduling, register renaming, and three levels of cache. That is a latency-optimising design. It assumes you have one thread that matters and you want its answer now.

A GPU makes the opposite bet. It assumes you have tens of thousands of independent pieces of work and you only care about total work per second. So it deletes almost all the control logic, amortises what remains across many arithmetic lanes, and spends the reclaimed area on ALUs and memory bandwidth.

That single trade explains nearly everything else:

| | CPU | GPU |
|---|---|---|
| Optimises | latency of one stream | throughput of many |
| Cores | tens of complex cores | thousands of simple lanes |
| Hides memory latency by | large caches, prefetch, OoO | switching to another thread |
| Fails badly when | work is embarrassingly parallel | work is branchy and serial |

The key phrase is **latency hiding by oversubscription**. A CPU tries to avoid waiting for memory. A GPU accepts the wait and keeps thousands of other threads resident so it always has something else to issue. This is why "occupancy" is a GPU concept and not a CPU one.

---

## 2. The execution model

### The hierarchy

You write a **kernel**, a function that one thread runs. You launch it over a grid:

```
grid  →  thread blocks  →  warps (32 threads)  →  threads
```

- **Thread**: one lane of execution, with its own registers.
- **Warp**: 32 threads that execute *in lockstep*. This is the real unit of scheduling. AMD calls its equivalent a *wavefront* (64 lanes, or 32 on RDNA).
- **Thread block** (CTA): a group of warps guaranteed to run on the same physical core, able to share fast scratchpad memory and synchronise with a barrier.
- **Grid**: all the blocks in the launch.

Physically, the chip is a set of **SMs** (Streaming Multiprocessors). An H100 has 132. Each SM has its own register file, scratchpad, warp schedulers, and arithmetic units. A block is assigned to exactly one SM and lives there until it finishes.

### SIMT, and why divergence hurts

NVIDIA calls the model **SIMT**: Single Instruction, Multiple Threads. In practice it is SIMD hardware wearing a scalar programming model. All 32 lanes of a warp share one instruction pointer.

So this code:

```cuda
if (threadIdx.x % 2 == 0) { A(); } else { B(); }
```

does not run A and B in parallel. The warp executes A with the odd lanes masked off, then executes B with the even lanes masked off. You paid for both. This is **warp divergence**, and it is the first performance trap everyone hits.

Since Volta (2017), each thread has its own program counter, so divergent threads can make independent forward progress and re-converge. The cost model is unchanged, but a class of deadlocks disappeared.

### What changed recently

**Thread block clusters** (Hopper, 2022) added a level between block and grid. A cluster is a group of blocks guaranteed to be co-resident on the same GPC, which lets them read each other's scratchpad directly. This is called **distributed shared memory**, and it is a genuinely new tier that did not exist when CUDA was taught as grid/block/thread.

---

## 3. Memory is the whole game

If you internalise one section, make it this one.

### The hierarchy, with rough H100 numbers

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | 256 KB per SM | ~100 TB/s | ~1 cycle |
| Shared memory / L1 | 228 KB per SM | ~20 TB/s | ~30 cycles |
| L2 cache | 50 MB | ~5 TB/s | ~200 cycles |
| HBM (global) | 80 GB | 3.35 TB/s | ~500 cycles |
| Host RAM over PCIe | ~TB | ~64 GB/s | ~microseconds |

Two things stand out. First, each step down is roughly an order of magnitude worse. Second, **shared memory is a software-managed scratchpad**, not a cache. You explicitly stage data into it. CPUs have nothing equivalent, and learning to use it well is most of what "optimising a CUDA kernel" means.

### Coalescing

HBM is read in fixed transactions (typically 32 bytes). If the 32 lanes of a warp read 32 consecutive floats, the hardware merges them into a few wide transactions. If they read 32 scattered addresses, you issue 32 separate transactions and waste most of the bytes you paid for. Access pattern matters more than instruction count.

### The roofline model

This is the mental model that makes GPU performance predictable.

Define **arithmetic intensity** as FLOPs performed per byte moved from HBM. Every kernel sits somewhere on:

```
achievable FLOP/s = min( peak FLOP/s ,  arithmetic intensity × memory bandwidth )
```

For an H100, peak BF16 tensor throughput is about 990 TFLOP/s against 3.35 TB/s of bandwidth. The crossover is around **295 FLOPs per byte**. Below that intensity you are memory-bound and the ALUs idle no matter what you do.

Where common operations land:

- **Elementwise ops** (ReLU, add, layernorm): intensity near 1. Hopelessly memory-bound. This is why kernel *fusion* is the single highest-leverage optimisation in deep learning: it removes round trips to HBM.
- **Attention, naively written**: materialises an N×N score matrix in HBM. Memory-bound and quadratic in memory, which is exactly the problem FlashAttention solves by tiling the computation so the score matrix never leaves shared memory.
- **Large dense matmul**: intensity scales with tile size. This is the one thing that genuinely saturates the ALUs.
- **LLM decoding, batch size 1**: you read every weight in the model to produce one token. At 2 FLOPs per parameter and 2 bytes per parameter, intensity is **1 FLOP per byte** at BF16 (2 at FP8, 4 at FP4). Catastrophically memory-bound, and the reason a 70B model at batch 1 runs at a fraction of a percent of peak FLOPS.

That last point is worth sitting with. **Most inference is bandwidth-bound, not compute-bound.** It is why HBM capacity and bandwidth, not TFLOPS, are the headline numbers on modern accelerators, and why batching, quantisation, and KV-cache compression are the levers that actually move throughput.

---

## 4. Tensor Cores: the biggest change since you last looked

If you learned CUDA before roughly 2018, you learned a machine whose fundamental operation was the fused multiply-add on a scalar lane. That machine no longer describes where the performance is.

Starting with Volta, each SM contains **Tensor Cores**: dedicated units that consume small matrix tiles and produce a matrix multiply-accumulate in a few cycles, `D = A×B + C`. They are fed cooperatively by a whole warp (or, on Hopper and later, a warpgroup of four warps) rather than by individual threads.

The magnitude of the shift:

| Unit | H100 throughput |
|---|---|
| FP32 "CUDA core" | ~67 TFLOP/s |
| BF16 Tensor Core | ~990 TFLOP/s |
| FP8 Tensor Core | ~1,980 TFLOP/s |

A kernel that does not use Tensor Cores leaves roughly 95% of the chip unused. This is the central reason that hand-written CUDA from the pre-Volta era is no longer competitive with library code, and why almost nobody writes matmuls from scratch any more.

### The precision ladder

Each generation added a narrower number format, because neural networks tolerate low precision far better than scientific computing does:

- **FP32** → baseline.
- **TF32** (Ampere): 19-bit internal format, drop-in replacement for FP32 matmul, ~8× faster.
- **FP16 / BF16** (Volta / Ampere): BF16 keeps FP32's exponent range and sacrifices mantissa, which makes it far more robust in training. It won.
- **FP8** (Hopper): two variants, E4M3 for weights and activations, E5M2 for gradients. Needs per-tensor scaling to manage range, which is what the "Transformer Engine" automates.
- **FP4 / MXFP4** (Blackwell): 4-bit with a shared exponent per small block of values, called *microscaling*. Blackwell's headline inference numbers are FP4 numbers.

The pattern is consistent: as precision halves, throughput roughly doubles and memory traffic halves. Since you are usually memory-bound, the second effect often matters more than the first.

### A caveat relevant to scientific ML

This entire trend is aimed at low-precision dense linear algebra. Workloads that need FP64 (traditional HPC, stiff PDE solvers) or that are dominated by irregular gather/scatter (GNNs, sparse meshes, neighbour lists) benefit far less. GNN message passing in particular is memory-latency-bound and does not map cleanly to Tensor Cores. AMD has made this split explicit: from the MI400 generation it is bifurcating its line into an AI part (FP4/FP8/BF16) and an HPC part (FP32/FP64), stripping the other's logic from each die to reclaim area. If you work on physics-informed models, this divergence in the hardware roadmap is a thing to track, because the FP64 story is quietly getting worse on the AI parts.

---

## 5. The modern kernel: asynchrony and specialisation

The other large change since the classic CUDA era is that a fast kernel is now an explicitly *pipelined* program, not a loop that loads and computes.

- **Async copy** (Ampere): copy from HBM to shared memory without staging through registers, so compute on tile *n* overlaps with the load of tile *n+1*.
- **TMA, the Tensor Memory Accelerator** (Hopper): a dedicated DMA engine. A single thread issues a descriptor and the hardware moves an entire multidimensional tile, handling addressing and boundary conditions. This removes an enormous amount of index arithmetic from the inner loop.
- **Warp specialisation**: rather than every warp doing the same thing, some warps become *producers* that only issue TMA loads while others become *consumers* that only issue Tensor Core instructions, coordinated through asynchronous barriers. The SM starts to look like a small dataflow machine.

**FlashAttention** is the canonical worked example of all of this. It is not a better algorithm in the FLOPs sense; it does slightly more arithmetic. It is faster because it is *IO-aware*: it tiles the computation so intermediate results stay in shared memory, uses an online-softmax reformulation so it never needs the full score matrix, and on Hopper it pipelines TMA loads against warpgroup matmuls. It made long-context transformers practical, and it is the best single case study for how modern GPU thinking works.

---

## 6. When one GPU is not enough

Modern frontier work does not treat a GPU as the unit of compute. The unit is a rack.

### Interconnect tiers

- **Within a node**: **NVLink**. Direct GPU-to-GPU links at 1.8 TB/s per GPU on Blackwell, roughly 14× a PCIe Gen5 x16 slot. NVSwitch chips make it an all-to-all fabric rather than point-to-point.
- **Across nodes**: InfiniBand or high-end Ethernet, roughly 400–800 Gb/s per GPU, 50–100 GB/s each way. Per direction and within one generation that is about 9× below NVLink (whose 1.8 TB/s is both directions added); see the [roofline primer §5.1](../roofline-and-fabric/PRIMER.md#51-the-link-ladder).

That gap defines the standard vocabulary. **Scale-up** means growing the NVLink domain, the set of GPUs that can treat each other's memory as nearly local. **Scale-out** means adding nodes over the slower network. The dominant hardware trend of the last two years is scale-up domains getting dramatically larger: GB200 NVL72 puts 72 GPUs in one liquid-cooled NVLink domain, and Vera Rubin NVL144 extends that further. AMD is pursuing the same idea with its Helios rack and the open UALink standard as an NVLink alternative.

### Parallelism strategies

You split a model across GPUs along different axes, and each choice buys different communication:

| Strategy | What is split | Communication |
|---|---|---|
| **Data parallel** | the batch | all-reduce of gradients each step |
| **Tensor parallel** | individual matrices | all-reduce *inside* every layer, so it needs NVLink |
| **Pipeline parallel** | layers across devices | point-to-point, but introduces bubbles |
| **Expert parallel** | MoE experts | all-to-all routing of tokens |
| **Context parallel** | the sequence | for very long context attention |

Real training runs compose three or four of these at once. The collective operations themselves (all-reduce, all-gather, reduce-scatter, all-to-all) are implemented by **NCCL**, and tuning NCCL is a real part of large-scale training work.

---

## 7. The software stack in 2026

Layers, from highest to lowest:

1. **PyTorch eager.** Still where most work starts. Every operation dispatches to a pre-written kernel; the tax is a round trip to HBM between operations.
2. **`torch.compile`.** Traces your model and generates fused kernels via the Inductor backend, which emits Triton. Usually the first thing to try, and often gets a large fraction of the available win for no effort.
3. **Triton.** A Python DSL where you write kernels at the level of *blocks of elements* rather than individual threads. The compiler handles coalescing, shared memory allocation, and scheduling within a block. This is where most custom kernel work happens today, and it is a far gentler on-ramp than CUDA C++.
4. **CUTLASS / CuTe.** NVIDIA's C++ template library for matmul-shaped problems. CuTe's layout algebra is the serious tool for expressing tiling and data movement. Steep, but this is what production matmul and attention kernels are built from.
5. **CUDA C++ with inline PTX.** Still the floor when you need something the layers above will not express.

### CUDA Tile, the notable new arrival

Since CUDA 13.0 (August 2025), NVIDIA has been building a *second* programming model alongside SIMT, called **CUDA Tile**. Instead of describing what one thread does, you describe operations on tiles and arrays, NumPy-style, and the compiler handles thread mapping, memory movement, asynchrony, and Tensor Core selection. It ships as **cuTile** for Python and, since CUDA 13.3, for C++ as well, backed by a new Tile IR that other compilers can target. Tile support was extended back to Ampere and Ada in 13.2.

This matters for you specifically: it is the first structural change to the CUDA programming model in twenty years, and it targets exactly the gap between "PyTorch is too slow here" and "I do not want to hand-write warp-level PTX". It converges conceptually with Triton, which is a reasonable signal that block/tile-level abstraction is where kernel programming is settling.

### Serving and inference

- **vLLM** and **SGLang** are the standard serving engines. Their central innovation, PagedAttention, applies virtual-memory paging to the KV cache to eliminate fragmentation.
- **TensorRT-LLM** is NVIDIA's own, faster on NVIDIA hardware and less flexible.
- **Disaggregated serving** is the current architectural direction: run *prefill* (compute-bound, processes the whole prompt) and *decode* (memory-bound, one token at a time) on separate pools of hardware, because they have opposite bottlenecks. NVIDIA has gone as far as building a distinct SKU for this, Rubin CPX, aimed at massive-context prefill.

### Non-NVIDIA

**ROCm** is AMD's stack. It is materially better than its reputation suggests: PyTorch, JAX, Triton, vLLM, and FlashAttention all work, and HIP gives you a near-mechanical port path from CUDA. The gap is now less about basic support and more about the depth of tuned kernels and the long tail of libraries. Google's **TPUs** (compiled through XLA, programmed via JAX) and Amazon's **Trainium** are the other volume alternatives, both systolic-array designs with a compiler-first rather than kernel-first philosophy.

---

## 8. The hardware landscape, as of August 2026

**NVIDIA datacenter line:**

| Generation | Parts | Memory | Notes |
|---|---|---|---|
| Hopper (2022) | H100, H200 | 80 GB HBM3 / 141 GB HBM3e | FP8, TMA, thread block clusters |
| Blackwell (2024–25) | B200, GB200 | 192 GB HBM3e physical; 180 GB usable per B200 in HGX, 186 GB per GB200 GPU (verify, 2026-09) | FP4, dual-die, NVL72 racks |
| Blackwell Ultra (2025) | B300, GB300 | 288 GB HBM3e | current volume part |
| **Rubin** (2026) | Rubin + Vera CPU | 288 GB HBM4 | in production, volume H2 2026 |
| Rubin CPX | long-context prefill SKU | — | expected end of 2026 |
| Rubin Ultra → Feynman | 2027+ | — | announced roadmap only |

The usable figures are the ones the [roofline primer's catalogue](../roofline-and-fabric/PRIMER.md#9-the-accelerator-landscape-september-2026-snapshot) (`roofline.specs`) plans with; size memory from them, not from the physical stack count.

Rubin entered full production around CES 2026 with volume shipments targeting the second half of this year, so it is arriving right now but supply is constrained by HBM4 yields and TSMC N3 capacity, and hyperscalers absorb most early allocation. NVIDIA's claimed gains over Blackwell are roughly 3.5× training and 5× inference per GPU, with a 10× reduction in cost per token. Treat vendor multipliers as marketing until MLPerf lands, but the direction is real.

**AMD:** MI355X (CDNA 4) shipping now; MI400 series in 2026, splitting into MI450X for AI and MI430X for HPC, with rack-scale Helios systems and UALink. MI500 announced for 2027.

**Consumer and desk-side:** RTX 50-series remains the current GeForce generation, with no announced successor. The RTX PRO 6000 Blackwell (96 GB) is the top workstation card. DGX Spark is NVIDIA's small Arm-based desk-side box, and CUDA 13 unified the Arm toolkit partly to serve it.

**Practical note for a home cluster:** the constraint you will hit is not FLOPS, it is VRAM and interconnect. Consumer cards have no NVLink from the 40-series onward, so multi-GPU work falls back to PCIe, which makes tensor parallelism painful and pushes you toward pipeline parallelism, offloading, or simply smaller models. Quantisation buys capacity and bandwidth simultaneously, which is why it is disproportionately effective on hobbyist hardware.

---

## 9. Numbers worth memorising

**Model memory:**
- Weights: 2 bytes per parameter at BF16. A 70B model needs 140 GB, so it does not fit on one 80 GB H100.
- Training: roughly 16 bytes per parameter with Adam (weights, gradients, and two optimiser moments in FP32), before activations. This is why training needs an order of magnitude more memory than inference.
- KV cache: `2 × layers × kv_heads × head_dim × seq_len × batch × 2 bytes`. At long context and high batch, this exceeds the weights, which is what motivated MQA, GQA, and MLA.

**Rules of thumb:**
- Forward pass costs about 2 FLOPs per parameter per token; a training step costs about 6.
- If arithmetic intensity is below ~300 FLOP/byte on modern hardware, you are memory-bound and should stop optimising arithmetic.
- Achieving 40–50% of peak FLOPS (Model FLOPs Utilisation) on a large training run is good. Above 50% is excellent.

---

## 10. A path in, if you want to go deeper

Roughly in order of leverage:

1. Read the **FlashAttention** paper (v1 for the idea, v3 for the Hopper mechanics). It teaches the roofline model, tiling, and asynchrony in one artefact.
2. Do the **Triton tutorials**, then write a fused kernel for something in your own stack. Vector add, then softmax, then a fused layernorm, then attention.
3. Learn to read a **Nsight Compute** profile. Specifically: what fraction of peak memory bandwidth am I hitting, what fraction of peak Tensor Core throughput, and which of those is the ceiling. Most optimisation intuition comes from this loop rather than from reading.
4. Skim the **CUTLASS/CuTe** layout algebra even if you never write it, because it is the vocabulary production kernels are described in.
5. Try **cuTile**, since it is new enough that early familiarity is cheap and it is plausibly where the model is heading.
6. For distributed: read the **Megatron-LM** and **ZeRO** papers, then run a multi-GPU job and watch the NCCL traces.

The single reframe that carries the most weight: stop thinking of a GPU as a fast calculator and start thinking of it as a memory system with arithmetic attached.
