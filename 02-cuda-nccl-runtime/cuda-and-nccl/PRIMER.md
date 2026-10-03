# The GPU software substrate: CUDA's execution model, collectives, and how a container gets a GPU

*This is the primer for layer 02 of the stack, written in September 2026. Versions, profiles and product names
change. The rules under them do not change. The [Verify list](#verify-list) collects all the dated facts.*

This primer is about the software between the silicon ([layer 01](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md))
and the schedulers and engines above it. Those are layer 03, [`03-kubernetes-gpu/gpu-scheduling/`](../../03-kubernetes-gpu/gpu-scheduling/), and layer 04,
[`04-inference-engine/serving-engine/`](../../04-inference-engine/serving-engine/). The primer explains these topics:

- how a driver, a CUDA runtime and a compiled kernel agree to run,
- how a kernel executes (warps, occupancy, 32-byte memory transactions),
- why launches cost time,
- how GPUs exchange data (collectives and NCCL),
- how a container gets a GPU,
- how processes share one GPU,
- how to tell if a GPU is healthy.

The `gpusim` package in [`cuda-nccl-core/`](cuda-nccl-core/) calculates each formula in this primer. That core is T0
and runs on a laptop CPU. [`cuda-nccl-lab/`](cuda-nccl-lab/) measures the formulas for real (T1 to T3). This primer
assumes that you know the [GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md). It links to that
primer and does not repeat it.

---

## The one-minute version

- **Two gates decide if CUDA code runs.** First, the host *driver* must support the CUDA *runtime* that you
  used to build your app. A newer driver runs an older runtime. A newer runtime of the same major version runs by
  minor-version compatibility. A newer major version needs a newer driver, or the forward-compatibility package on
  a data-center GPU. Second, the binary must contain a *kernel image* for the GPU. That image is SASS for its
  compute capability (same major, equal or lower minor), or PTX that the driver can JIT-compile. If the binary has
  neither, you get *"no kernel image is available"*.
- **The warp is the unit.** 32 threads share one instruction stream. The memory system serves their requests in
  32-byte sectors. Consecutive 4-byte loads take 4 sectors per warp, and a column walk takes 32. Shared memory has
  32 banks of 4 bytes, and a word stride of $s$ costs a $\gcd(s, 32)$-way conflict. When a branch divides a warp,
  the warp runs both sides.
- **Memory-bound kernels cost bytes plus launches.** Tiling and fusion decrease the bytes. CUDA Graphs decrease
  the launches. This is why engines capture decode steps. Occupancy is a means to keep sufficient bytes in flight
  (Little's law). It is not a goal.
- **All-reduce = reduce-scatter + all-gather.** On a ring, that is ${2(p-1)}$ steps of ${S/p}$ bytes:
  $T = 2(p-1) \cdot \alpha + 2(p-1)/p \cdot S/B$. Below approximately $p \cdot \alpha \cdot B$ (7.2 MB for 8 H100s
  with illustrative numbers), the all-reduce is latency-bound. The all-reduces of tensor-parallel decode are well
  below that limit. Thus engines use algorithms with few steps. The **busbw** of nccl-tests normalises to the
  per-link bandwidth. Thus you can compare it with the spec.
- **A container brings CUDA. The host brings the driver.** The NVIDIA Container Toolkit injects `/dev/nvidia*`,
  `libcuda` and NVML when the container starts. It uses an OCI hook or a CDI spec. Do not build `libcuda` into an
  image.
- **Sharing:** MIG partitions the GPU (isolated, shapes that do not change). MPS overlaps processes (efficient, weakly
  isolated). Time-slicing takes turns (no isolation). For LLM serving, the batching in the engine is the best way
  to share a GPU.
- **Health:** "GPU util" is the share of *time* when a kernel ran. It does not show how much of the GPU the kernel
  used. Read SM active, tensor active and DRAM active. Sort XIDs by who must act.

---

## 1. The stack from driver to framework

### 1.1 The layers, and who ships each one

```
┌──────────────────────────────────────────────────────────────┐
│ framework / engine   PyTorch · vLLM · SGLang · TensorRT-LLM  │  pip wheel or image
│ kernel libraries     cuBLAS · cuDNN · NCCL · CUTLASS · FA    │  each .so is a *fatbin*:
│ JIT kernels          Triton: Python → PTX → SASS, first call │  SASS per sm + maybe PTX
│ CUDA runtime         libcudart.so.12 / .13                   │
├──────────────── container image ends here ───────────────────┤
│ user-mode driver     libcuda.so · libnvidia-ml.so (NVML)     │  host driver install,
│                      libnvidia-ptxjitcompiler.so (PTX JIT)   │  mounted into containers
├──────────────── user space / kernel space ───────────────────┤
│ kernel-mode driver   nvidia.ko, nvidia-uvm.ko, GSP firmware  │  host
│ device nodes         /dev/nvidia0.., /dev/nvidiactl, -uvm    │
├──────────────────────────────────────────────────────────────┤
│ GPU                  compute capability 9.0 (H100), 8.9 (L4) │  layer 01
└──────────────────────────────────────────────────────────────┘
```

Two version numbers are important. The **driver version** (for example `580.65.06`) belongs to the host. The
**CUDA version** that you used to build your app belongs to the runtime in your wheel or image
(`torch.version.cuda`, `nvcc --version`). The banner of `nvidia-smi` prints *"CUDA Version: 13.0"* next to the
driver. That value is the **newest runtime this driver supports**, not an installed toolkit
(`gpusim.compat.driver_cuda()`).

### 1.2 Gate 1: driver and runtime

| Case | Condition | Outcome |
|---|---|---|
| Backward compatibility | driver's CUDA >= app's CUDA | runs |
| Minor-version compatibility | same major, driver >= that major's floor (12.x: 525.60.13) | runs. But the driver cannot JIT-compile PTX from the newer toolkit (error 222), and APIs newer than the driver fail (error 36). |
| Newer major | for example, a CUDA 13 app on a 12.x driver | fails with 35, *"CUDA driver version is insufficient for CUDA runtime version"* |
| Forward compatibility | the `cuda-compat` package supplies a newer user-mode driver on an older kernel module | runs on data-center GPUs with a supported kernel-driver branch. If not, it fails with 804 (hardware) or 803 (branch). |

This table gives the minimum driver for each toolkit, on Linux x86-64, from the release notes. The full table, back
to CUDA 9.0, is `gpusim.compat.CUDA_MIN_DRIVER` (verify). For older drivers, `check()` answers *cannot judge*.

| CUDA | 11.0 | 12.2 | 12.4 | 12.8 | 12.9 | 13.0 | 13.1 | 13.2 | 13.3 | 13.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| driver >= | 450.51.05 | 535.54.03 | 550.54.14 | 570.26 | 575.51.03 | 580.65.06 | 590.44.01 | 595.45.04 | 610.43.02 | R615 (inferred) |

An example, calculated with `gpusim.compat.check()`: an app built with **CUDA 12.4** runs on an **H100** under
driver **535.183.01**. That driver supports CUDA 12.2.

- With SASS for sm_80 and sm_90, the app **runs** by minor-version compatibility.
- If the kernels contain only `compute_80` PTX, the app **fails with 222**. The 535 JIT cannot read PTX from 12.4.
- A **CUDA 13.0** build **fails with 35**. With `cuda-compat` 13.0, it runs on the H100, because R535 is a
  kernel-driver branch that this package supports (`compat.COMPAT_BRANCHES`, verify). Over R560, it fails with 803.
  On an RTX 4090, it fails with 804.

### 1.3 Gate 2: kernel image and GPU

Each NVIDIA GPU has a **compute capability** X.Y. `nvcc` compiles device code to **PTX** (a virtual ISA,
`compute_XY`). Then it compiles the PTX to **SASS** (machine code, `sm_XY`). A fatbin can hold more than one of
each (`-gencode
arch=compute_90,code=[sm_90,compute_90]`). PyTorch writes this as `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"`, and
`gpusim.compat.parse_targets()` parses it.

At load time, the driver selects the SASS that matches best. If no SASS matches, the driver JIT-compiles the PTX.
Then the first start is slow, and the driver caches the result in `~/.nv/ComputeCache`. If neither is possible,
you get error **209**.

```
SASS sm_XY  runs on CC X.Z with Z >= Y        sm_80 → 8.0, 8.6, 8.9       never 9.0, never 7.5
PTX  compute_XY  JITs to CC >= X.Y            compute_80 → 8.x, 9.0, 10.0, 12.0 ...
'a' targets (sm_90a, sm_100a)                 exactly one CC: arch-specific instructions (wgmma, tcgen05)
'f' targets (sm_100f, compute_100f; 12.9+)    the same family: same major, minor >= Y (verify)
```
(`gpusim.compat.sass_runs_on()`, `ptx_jits_to()`)

| GPU | Architecture | CC | First CUDA | Notes (verify) |
|---|---|---|---|---|
| V100 | Volta | 7.0 | 9.0 | CUDA 13 removed the Maxwell/Pascal/Volta targets |
| T4 | Turing | 7.5 | 10.0 | Colab and Kaggle free tier |
| A100, A30 | Ampere | 8.0 | 11.0 | MIG |
| A10G, A40, RTX 30 | Ampere | 8.6 | 11.1 | |
| L4, L40S, RTX 40 | Ada Lovelace | 8.9 | 11.8 | FP8 |
| H100, H200 | Hopper | 9.0 | 11.8 | sm_90a for wgmma/TMA kernels |
| B200, GB200 | Blackwell | 10.0 | 12.8 | FP4 |
| B300, GB300 | Blackwell Ultra | 10.3 | 12.9 | |
| RTX 5090, RTX PRO 6000 | Blackwell | 12.0 | 12.8 | consumer/workstation line. It is *not* binary-compatible with 10.0. |

The classic failure is this: you run a wheel built for `8.0 8.6 9.0` without `+PTX` on an RTX 5090 (12.0). It
fails with **209**. The wheel has no sm_12x SASS, and the driver cannot JIT-compile any PTX. If the wheel has
`9.0+PTX`, the driver JIT-compiles the PTX and the wheel runs (slowly at first). On a T4, the same wheel also
fails, because the driver JIT-compiles PTX only *upward*.

The same check predicts "no device" when the branch of the driver is older than the first toolkit that targets the
GPU. An example is a B200 on an R550 driver. The check uses an approximation of the minimum driver of each GPU
(verify).

### 1.4 Libraries, frameworks and the errors you will meet

The Linux wheels of PyTorch contain their own CUDA runtime, cuBLAS, cuDNN and NCCL as `nvidia-*` pip packages.
Thus a host needs **only a driver**. **Triton** compiles Python kernels at the first call, through PTX to a cubin
with its own bundled `ptxas`, and then caches them (`~/.triton/cache`). For each shape,
**cuBLAS/cuBLASLt** select GEMM kernels. **NCCL** is the topic of §5. `torch.cuda.get_arch_list()` lists the
SASS/PTX targets of your build.

| Error | Code | Meaning | Usual fix |
|---|---|---|---|
| CUDA driver version is insufficient for CUDA runtime version | 35 | The runtime is a newer major, or it is below the minor-compat floor. | Install a newer driver, use an older CUDA, or use `cuda-compat` (data center). |
| API call is not supported in the installed CUDA driver | 36 | A minor-compat app called a newer API. | Install a newer driver. |
| no CUDA-capable device is detected | 100 | The container has no injected GPU, or the driver does not know the GPU. | Use `--gpus`, a CDI device or an `nvidia.com/gpu` request, or install a newer driver. |
| no kernel image is available for execution on the device | 209 | The binary has no SASS for this CC and no usable PTX. | Build for this sm, or add `+PTX`. |
| the provided PTX was compiled with an unsupported toolchain | 222 | The PTX is newer than the JIT of the driver. | Include SASS in the binary, or install a newer driver. |
| system not yet initialized | 802 | The NVSwitch system has no fabric manager. | Start `nvidia-fabricmanager` (the same version as the driver). |
| system has unsupported display driver / cuda driver combination | 803 | The user-mode libcuda does not match the kernel module. | Remove libcuda from the image. Correct the compat branch. |
| forward compatibility was attempted on non supported HW | 804 | The app uses `cuda-compat` on GeForce. | Install a newer driver instead. |
| Failed to initialize NVML: Driver/library version mismatch | (NVML) | The user-space NVML is new, but the old kernel module stays loaded. | After a driver upgrade, reboot or load the module again. |

The table is `gpusim.compat.ERRORS`. Notebook 04 has drills for 35, 100, 209, 222, 803 and 804. `check()` does
not model 36, 802 or NVML.

---

## 2. The execution model

### 2.1 Grid, block, warp, SM

A kernel launch is a **grid** of **blocks** (CTAs) of up to 1,024 threads. The hardware schedules **warps** of 32
threads. Each block goes to one **SM** and stays there until it finishes. An SM has four sub-partitions, and each
one has its own warp scheduler, register-file slice and execution units. In each cycle, a scheduler issues one
instruction from a *ready* warp. The [GPU primer §2 and §5](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md#2-the-execution-model)
gives the rest of the story (tensor cores, TMA, clusters).

### 2.2 SIMT and divergence

A warp has one instruction stream. When the lanes do not agree on a branch, the warp runs **every path that any
lane takes**, and it masks off the other lanes. The cost per warp is the sum over the different paths that the
lanes take (`gpusim.simt.divergence()`):

```
if (tid % 2) A(10 instr) else B(10 instr)      every warp issues 20: SIMT efficiency 50%
if ((tid / 32) % 2) ...                         each warp takes one path: 100%
per-thread loop of L[t] iterations              each warp runs max(L) iterations  (loop_divergence())
```

The loop case is the case that causes problems for inference code. With one thread per sequence and long-tailed
lengths, SIMT efficiency was **24%** in notebook 01, and **96%** after a sort by length. This is why kernels over
ragged batches put the sequences into buckets by length. It is also why attention kernels divide the work into
*tiles of tokens*, not threads per sequence. Since Volta, each thread has its own program counter, and thus divergent
lanes can make independent progress. The cost model does not change.

### 2.3 Occupancy

**Occupancy** = resident warps / the SM's maximum. A block is resident only if all of its resources fit. Thus the
number of blocks per SM is the minimum over four limits (`gpusim.occupancy.occupancy()`, which uses the arithmetic of
NVIDIA's `cuda_occupancy.h`):

$$
\begin{aligned}
\text{by warps} \quad & \left\lfloor \frac{\texttt{max_warps}}{\lceil \texttt{threads}/32 \rceil} \right\rfloor \\
\text{by blocks} \quad & \texttt{max_blocks} \\
\text{by registers} \quad & \left\lfloor \frac{4 \times \lfloor 16384 / \operatorname{roundup}(\texttt{regs} \times 32, 256) \rfloor}{\texttt{warps_per_block}} \right\rfloor && \text{(4 sub-partitions)} \\
\text{by smem} \quad & \left\lfloor \frac{\texttt{smem_per_SM}}{\operatorname{roundup}(\texttt{smem_per_block} + 1\ \text{KB reserved}, 128)} \right\rfloor && (\text{CC} \ge 8.0)
\end{aligned}
$$

| CC (example) | threads/SM | warps/SM | blocks/SM | registers/SM | smem/SM | smem/block |
|---|---|---|---|---|---|---|
| 7.5 (T4) | 1,024 | 32 | 16 | 65,536 | 64 KB | 64 KB |
| 8.0 (A100) | 2,048 | 64 | 32 | 65,536 | 164 KB | 163 KB |
| 8.9 (L4) | 1,536 | 48 | 24 | 65,536 | 100 KB | 99 KB |
| 9.0 (H100) | 2,048 | 64 | 32 | 65,536 | 228 KB | 227 KB |
| 10.0 (B200) | 2,048 | 64 | 32 | 65,536 | 228 KB | 227 KB (verify) |
| 10.3 (B300) | 2,048 | 64 | 32 | 65,536 | 228 KB | 227 KB (verify) |
| 12.0 (RTX 5090, RTX PRO 6000) | 1,536 | 48 | 32 | 65,536 | 128 KB | 99 KB (verify) |

An example on an **L4**: a block of 256 threads that uses 64 registers per thread gives 2,048 registers per warp.
The 16,384 registers in each sub-partition hold 8 warps. Thus an SM holds 32 warps, which is **4 blocks and 67%
occupancy**, and the registers set the limit. If you add 48 KB of shared memory per block, each block needs 49 KB.
Then 2 blocks fit in 100 KB: **33%**.

The division into sub-partitions is important. At 80 registers, ⌊16,384 / 2,560⌋ = 6 warps per partition. Thus an
SM holds 24 warps, not ⌊65,536 / 2,560⌋ = 25.

### 2.4 Latency hiding, Little's law and waves

When a warp waits for memory, the scheduler replaces it with a ready warp. To keep a bandwidth $B$ with a latency
$L$, **$B \times L$ bytes must be in flight** (`gpusim.occupancy.bytes_in_flight()`). With an *assumed* loaded
latency of 600 ns, an H100 at 3.35 TB/s needs 2.0 MB in flight, which is **15 KB per SM**. Even at full occupancy
(2,048 threads), that is **1.86 loads of 4 bytes in flight per thread**, or 0.46 with 16-byte vector loads
(`loads_in_flight_per_thread()`). An L4 needs 3.1 KB per SM, and approximately half occupancy covers that.

Thus fast kernels on large GPUs hide latency with *vector loads, several independent loads per thread, and
asynchronous copies* (cp.async, TMA). GEMM and attention kernels run at 1 to 2 blocks per SM with large register
tiles. They still saturate the machine ("better performance at lower occupancy", Volkov 2010). Treat occupancy
as a means.

A grid runs in **waves** of `SMs × blocks_per_SM` blocks. 140 blocks, with one block per SM, on 132 SMs take two
waves. The second wave is 6% full. Thus the result is **53% efficiency** (`gpusim.occupancy.waves()`). Decode
kernels with few blocks have this problem. This is why split-K and split-KV exist.

---

## 3. Memory access patterns

### 3.1 Coalescing into 32-byte sectors

Global memory serves requests in **32-byte sectors**. An L1/L2 line is 128 bytes = 4 sectors. On compute
capability 6.0+, a warp request costs one transaction per **distinct sector** that its active lanes touch
(`gpusim.simt.coalescing()`):

| One warp reads (float32) | Sectors | Bytes moved | Efficiency |
|---|---|---|---|
| `a[lane]` (aligned) | 4 | 128 | 100% |
| `a[lane + 1]` (misaligned by 4 B) | 5 | 160 | 80% |
| `a[2 * lane]` | 8 | 256 | 50% |
| `a[32 * lane]` (a column of a row-major matrix) | 32 | 1,024 | 12.5% |
| `a[0]` in every lane | 1 | 32 | one transaction (a broadcast) |
| `double` / `float4` per lane, contiguous | 8 / 16 | 256 / 512 | 100% |

This has two consequences for inference. **Layout is performance**: struct-of-arrays is faster than
array-of-structs. Matrices have a layout in which the index that changes fastest is the index that consecutive
lanes walk. **Paged KV caches** stay coalesced, although their blocks are in different places in memory. Each block
stores the keys and values of its tokens contiguously (kilobytes per block), and thus a warp still reads whole sectors.
That is one reason why the blocks cannot be too small
([PagedAttention primer](../../04-inference-engine/paged-attention/paged-attention-primer.md)).

### 3.2 Shared memory and bank conflicts

Shared memory is a software-managed on-chip scratchpad (up to 228 KB per SM on an H100). It has **32 banks, each
4 bytes wide**. Word $w$ is in bank $w \bmod 32$. In one pass, each bank serves one word. If lanes want *different*
words in the same bank, the bank serves them one after the other. If lanes want the *same* word, they get a
broadcast (`gpusim.simt.bank_conflicts()`):

```
tile[lane]                   stride 1 word  → 32 banks, 1 pass
tile[2*lane]                 stride 2       → 2-way conflict
tile[s*lane]                 stride s       → gcd(s, 32)-way conflict
tile[lane][c] in float[32][32]              → 32-way conflict (every row starts in bank 0)
tile[lane][c] in float[32][33]              → conflict-free (row r starts in bank r)
8-byte / 16-byte accesses                   → served per half-warp / quarter-warp
```

### 3.3 Case study: the transpose

Take `out[j][i] = in[i][j]` with a warp across 32 consecutive `j`. The read takes 4 sectors per request, and the
write is a column walk at **32 sectors**. Thus the writes move 8× the bytes that they need.

If you put a 32×32 tile in shared memory first, both global phases become row-wise. The column walk then moves
into shared memory. There, a pitch of 32 floats causes a 32-way bank conflict, and a pitch of **33** makes it
conflict-free. Notebook 01 calculates all three numbers. The lab's `01_kernels_in_the_simulator` runs the kernels.

### 3.4 Tiled GEMM traffic

For $C[M,N] = A[M,K] \cdot B[K,N]$, a block that computes a $\mathit{BM} \times \mathit{BN}$ tile streams a
$\mathit{BM} \times K$ panel of $A$ and a $K \times \mathit{BN}$ panel of $B$ through shared memory. Thus the
kernel loads each $A$ element one time per *column* of tiles, and each $B$ element one time per *row*.
`gpusim.tiling.gemm_traffic()` calculates this, and a comparison with the simulated kernel `tiled_matmul()` confirms
it:

$$
\begin{aligned}
\text{global elements} &= M \cdot K \cdot \lceil N/\mathit{BN} \rceil + K \cdot N \cdot \lceil M/\mathit{BM} \rceil + M \cdot N \\
\text{naive: } & \mathit{BM} = \mathit{BN} = 1 \qquad \text{compulsory: each input once}
\end{aligned}
$$

| 4096³ GEMM, bf16 | Global traffic | FLOP/byte |
|---|---|---|
| naive (no reuse) | 274.9 GB | 0.5 |
| 32 × 32 tiles | 8.62 GB | 15.9 |
| 128 × 128 tiles | 2.18 GB | 63.0 |
| 128 × 256 tiles | 1.64 GB | 83.6 |
| compulsory | 0.10 GB | 1,365 |

Tiles cost shared memory: $\text{stages} \times (\mathit{BM} \cdot \mathit{BK} + \mathit{BK} \cdot \mathit{BN}) \times \text{bytes}$ (`tile_smem_bytes()`).
A 128×128×32 BF16 tile with 3 stages is 48 KB, which gives 2 blocks per SM on an L4 (§2.3). The last factor of
approximately 20× to compulsory traffic comes from **L2**, where concurrent blocks share panels. Layer 01 puts these
numbers on the roofline
([§4.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#4-the-memory-hierarchy-and-why-tilingfusion-win)).
Note the decode case: with $M$ = 8 tokens, the traffic is one read of the weight matrix, and no tile size helps.

### 3.5 Fusion

Each kernel boundary is a round trip through HBM. Take a row-wise softmax of a 4096 × 4096 BF16 score matrix
(`gpusim.tiling.softmax_traffic()`):

| Variant | Kernels | HBM traffic | At 3.35 TB/s |
|---|---|---|---|
| unfused: max, subtract, exp, sum, divide | 5 | 256 MiB (8RC + 4R elements) | 80 µs |
| fused, a row fits on chip | 1 | 64 MiB (2RC) | 20 µs |
| online (two passes, running max and sum) | 1 | 96 MiB (3RC) | 30 µs |

When the max increases, the online variant rescales its running sum by $\exp(m_{\text{old}} - m_{\text{new}})$.
This makes the result exact (`softmax_online()`). FlashAttention fuses the softmax *into* QKᵀ and PV. Thus the
score matrix never gets to HBM
([FlashAttention primer §4–5](../../04-inference-engine/flash-attention/flash-attention-primer.md)). Elementwise
chains fuse in the same way: $k$ unfused ops move $k$ times the bytes of one fused kernel (`elementwise_traffic()`).

### 3.6 L2: the shared last level

One L2 cache serves all SMs. It is approximately 4 MB on a T4, 40 MB on an A100, 48 MB on an L4, 50 MB on an
H100 (verify). In L2, the tiled GEMM of §3.4 gets back its last ~20×. Blocks that run at the same time on nearby
tiles read the same $A$ and $B$ panels. Thus the later readers hit in L2. This is why GEMM libraries launch tiles in
a *swizzled* order (nearby tiles together).

Decode shows the limit. A 17.5 GB weight shard is 350× an H100's L2. Thus each step streams its weights from HBM,
whatever the policy is. On CC 8.0+, you can pin small hot data with the access policy window of a stream
(`cudaAccessPolicyWindow`). `gpusim` counts only HBM bytes. Thus it gives a bound on what L2 can save, and it does
not simulate L2.

---

## 4. Streams, launch overhead and CUDA Graphs

### 4.1 Asynchronous launches and streams

A kernel launch **enqueues** work and returns. A **stream** is an in-order queue. Different streams can overlap:
copies with compute, NCCL communication with compute, prefetch with decode. The CPU runs ahead of the GPU until
something makes it wait. **Hidden synchronization points** are the usual cause: `tensor.item()`, `.cpu()`, a print
of a GPU tensor, `torch.cuda.synchronize()`, and some allocations.

### 4.2 When the CPU is the bottleneck

Each launch costs CPU time: microseconds in the driver, and more in the eager dispatch of a framework. If a kernel
is shorter than a launch, the GPU waits (`gpusim.tiling.step_time()`):

```
eager:  kernel i is enqueued at (i+1)·launch; it starts when enqueued AND the previous one is done
graph:  one launch for the whole captured sequence, then kernels back to back
assumed: an eager launch costs 5 µs, launching a whole graph 10 µs
384 kernels × 2 µs:                               eager 1,922 µs (GPU idle 60%)   graph 778 µs   → 2.5×
384 kernels × 20 µs:                              eager 7,685 µs   graph 7,690 µs  → no gain
```

A decode step at small batch is exactly the first case. It has hundreds of small GEMMs, norms, rotary embeddings,
attention and sampling kernels. Many of them are a few microseconds long.

### 4.3 CUDA Graphs, and why engines capture decode

A **CUDA Graph** records a sequence of kernels (and memcpys, and NCCL calls) one time. Then it replays the sequence
with one launch. The price is that the graph is rigid. The capture freezes the shapes and the memory addresses. Thus
you copy the inputs into static buffers. There must be no host synchronization inside the graph, and the control
flow does not change.

Thus engines **capture one graph per batch-size bucket** for decode. They pad each batch up to the nearest captured
size. Prefill has variable shapes. Thus it runs in eager mode or as *piecewise* graphs around attention. A capture
costs startup time and GPU memory. This is why vLLM's `--enforce-eager` exists: it saves both, but decode becomes
slower.

`torch.compile` works on the same overhead from the other side. Inductor fuses elementwise chains into Triton
kernels, and `mode="reduce-overhead"` puts the result in CUDA Graphs. You can also capture NCCL collectives. Custom
all-reduce kernels read the buffers of their peers directly over NVLink. These kernels register those buffer
addresses during the capture of the graph.

---

## 5. Collectives

### 5.1 Semantics

Each of $p$ ranks holds a buffer. What each rank holds after the operation defines the collective
(`gpusim.collectives.reference()`):

| Collective | After | Used in inference by |
|---|---|---|
| broadcast | every rank has the buffer of the root | weights or config from rank 0 |
| reduce | root has the elementwise sum | rarely |
| **all-reduce** | every rank has the sum | tensor parallelism, 2 per layer |
| **reduce-scatter** | rank $r$ has chunk $r$ of the sum | sequence parallelism and the first half of all-reduce |
| **all-gather** | every rank has the concatenation | sharded weights (FSDP), logits, sequence parallelism |
| **all-to-all** | rank $r$'s chunk $j$ goes to rank $j$ | MoE expert parallelism: dispatch and combine |
| send / recv | point-to-point | pipeline parallelism and KV transfer (layer 05) |

### 5.2 All-reduce = reduce-scatter + all-gather

Divide each buffer into $p$ chunks. In the **reduce-scatter**, at step $s = 1, \dots, p-1$, each rank $r$ adds its
partial of chunk $(r-s) \bmod p$ into its right neighbour. After ${p-1}$ steps, rank $r$ owns the finished chunk
$r$. In the **all-gather**, the finished chunks travel ${p-1}$ more hops. This trace is
`all_reduce(bufs, "ring").trace.table()` for 4 ranks. In the trace, `c2+` means "chunk 2, added in":

```
step  1 reduce-scatter   0->1:c3+  1->2:c0+  2->3:c1+  3->0:c2+
step  2 reduce-scatter   0->1:c2+  1->2:c3+  2->3:c0+  3->0:c1+
step  3 reduce-scatter   0->1:c1+  1->2:c2+  2->3:c3+  3->0:c0+     rank r now owns the sum of chunk r
step  4 all-gather       0->1:c0   1->2:c1   2->3:c2   3->0:c3
step  5 all-gather       0->1:c3   1->2:c0   2->3:c1   3->0:c2
step  6 all-gather       0->1:c2   1->2:c3   2->3:c0   3->0:c1      everyone has every sum
```

Each rank sends ${2(p-1)}$ messages of ${S/p}$, which is **$2(p-1)/p \cdot S$** bytes in total. Because no
point-to-point algorithm sends less, the ring is bandwidth-optimal. The ring reduces each chunk exactly one time and
then copies it. Thus all ranks end with bitwise-identical results. A different algorithm adds in a different order.
Thus the ring and tree results are different in the last bits (notebook 03 shows it).

### 5.3 The α-β cost of each algorithm

In the model, each step costs **$\alpha$** (launch, synchronization and hop latency), plus the bytes through the
busiest port divided by **$B$**. $B$ is the per-direction link bandwidth. Layer 01 introduces the model and writes
$B$ as $\beta$ ([§5.2](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#5-fabrics-quantitatively)). Each
algorithm in `gpusim.collectives` really moves numpy data. Its traced time is equal to the closed form
$T = a \cdot \alpha + c \cdot S/B$ (`cost_terms()`, `model_time()`):

| Algorithm | Steps $a$ | Bandwidth factor $c$ | Needs |
|---|---|---|---|
| ring all-reduce | ${2(p-1)}$ | ${2(p-1)/p}$ | a ring (any topology) |
| binomial tree all-reduce (reduce, then broadcast) | $2\lceil \log_2 p \rceil$ | $2\lceil \log_2 p \rceil$ | nothing, for small messages only |
| NCCL double binary tree (pipelined, a model and not a simulation) | $\approx 2\lceil \log_2 p \rceil$ + pipeline fill | $\approx 2$ | NCCL's tree algorithm |
| one-shot (every rank pulls all buffers) | 1 | ${p-1}$ | all-to-all links (NVSwitch) |
| two-shot (direct reduce-scatter + all-gather) | 2 | ${2(p-1)/p}$ | all-to-all links |
| in-switch reduction (NVLS, SHARP), $k$ chunks | ${k+1}$ | $(k+1)/k \to 1$ | a switch that reduces |
| ring reduce-scatter or all-gather | ${p-1}$ | ${(p-1)/p}$ | |
| pairwise all-to-all / direct all-to-all | ${p-1}$ / 1 | ${(p-1)/p}$ | |
| pipelined chain broadcast, $k$ chunks | ${p+k-2}$ | $(p+k-2)/k \to 1$ | |

A pipeline trades $\alpha$ for $\beta$. If you divide a message into $k$ pieces, you add $k$ steps, but each step
becomes smaller. The optimum is $k^{\ast} = \sqrt{S/(\alpha \cdot B)}$: 12 chunks for 128 MiB with the numbers in
§5.4 (`optimal_chunks()`). `best_algorithm()`, `sweep()` and `tp_comm()` pipeline at $k^{\ast}$, unless you give a
different value (`pipeline_chunks()`). In-switch reduction moves almost half the bytes of the ring ($S$ instead of
$2(p-1)/p \cdot S$ per GPU). The reason is that each GPU sends its data **once** and receives the result once.

### 5.4 algbw and busbw

nccl-tests reports two bandwidths ([PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)).
**$\text{algbw} = S/t$**, where $S$ is the full buffer (the gathered output for all-gather, the input for
reduce-scatter). **$\text{busbw} = \text{algbw} \times \text{factor}$**. The factor is the bandwidth term of the
optimal point-to-point algorithm. Thus busbw shows the **per-link** bandwidth for any value of $p$
(`gpusim.collectives.busbw()`):

| all-reduce | reduce-scatter, all-gather, all-to-all | broadcast, reduce, send/recv |
|---|---|---|
| ${2(p-1)/p}$ | ${(p-1)/p}$ | 1 |

This example uses **8 H100s** and the illustrative NVLink 4 numbers of layer 01: **$\alpha$ = 2 µs per step and
$B$ = 450 GB/s**. A 1 GiB ring all-reduce takes **4.20 ms**: algbw 255 GB/s, **busbw 447 GB/s**. The number to
compare with the 450 of the link is busbw.

There are two warnings. First, with **NVLS** (1 GiB in $k^{\ast}$ = 35 chunks), the modelled busbw gets to
744 GB/s (ceiling $2(p-1)/p \cdot B$ = 788). That is *above* the link rate, because the switch does the reduction
and the factor assumes that it did not. This is the expected result, not an error. What real NVLS reaches is a
question for nccl-tests.

Second, busbw well below the link rate inside one node usually means that NCCL does not use NVLink (§5.8).

### 5.5 Latency-bound or bandwidth-bound

The latency term and the bandwidth term are equal at **$S^{\ast} = a \cdot \alpha \cdot B / c$**
(`crossover_bytes()`). For the ring, **$S^{\ast} = p \cdot \alpha \cdot B$ = 7.2 MB** on the 8 H100s of §5.4. That
is approximately 440 tokens of an 8,192-wide BF16 activation. This is the ring sweep, simulated in the shape of
nccl-tests (`sweep()`, `format_sweep()`):

```
#  size (B)   time (us)   algbw (GB/s)   busbw (GB/s)   [simulated, alpha-beta model]
      65536        28.3           2.32           4.06
    1048576        32.1          32.69          57.20
   16777216        93.2         179.93         314.87
  134217728       550.0         244.05         427.09
 1073741824      4203.7         255.43         447.00
```

Below $S^{\ast}$, a ring all-reduce costs approximately $2(p-1) \cdot \alpha$ for any message size. Thus the
algorithm that wins is the one with the fewest steps. At 512 KiB on 8 GPUs, the model gives these times: two-shot
6.0 µs, in-switch 6.3 µs ($k^{\ast}$ = 1), one-shot 10.2 µs. The binomial tree takes 19.0 µs, and the ring takes
30.0 µs.

At 128 MiB, in-switch reduction wins (349 µs at $k^{\ast}$ = 12). Two-shot and ring come next (526 and 550 µs), and one-shot is
2.1 ms (`best_algorithm()`). This is, on a small scale, what the tuner of NCCL decides for each call.

### 5.6 How inference uses collectives

**Tensor parallelism** (Megatron-style) divides each attention and MLP layer column-wise, then row-wise. It does an
all-reduce of a `tokens × hidden × 2 B` activation **two times per layer**. For a 70B-class model (hidden 8,192, 80
layers) at TP=8, that is 160 all-reduces per step. Layer 01's
[§5.3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#5-fabrics-quantitatively) gives the cost at
batch 1. `gpusim.collectives.tp_comm()` gives this table:

| Step | Message | Ring | Two-shot | Regime |
|---|---|---|---|---|
| decode, batch 32 | 512 KiB | 4.81 ms/step (93% $\alpha$) | 0.97 ms/step | latency-bound |
| prefill, 8,192 tokens | 128 MiB | 88.0 ms/step (5% $\alpha$) | 84.2 ms/step | bandwidth-bound |

Each GPU streams its 17.5 GB weight shard in approximately 5.2 ms per decode step. A ring adds approximately 90% to
this time, and a two-shot adds approximately 19%. That is why engines have **custom all-reduce kernels**:

- the one-shot and two-shot kernels of vLLM and TensorRT-LLM over NVLink P2P buffers,
- NVLS where the switch supports it,
- all-reduce fused with the next RMSNorm.

The engines also capture all of it in the decode CUDA graph. *Sequence parallelism* replaces each all-reduce with a
reduce-scatter and an all-gather (the same bytes), so that the norms run on ${1/p}$ of the tokens.

**Expert parallelism** moves tokens, not partial sums. It does an all-to-all **dispatch** to the top-k experts of
each token, and an all-to-all **combine** back. Each one is `tokens × top_k × hidden × bytes` per GPU. For a
Mixtral-like layer (hidden 4,096, top-2) with 256 tokens per GPU, that is 4 MiB per direction. In the model, this
takes 22 µs pairwise or 10 µs direct (`model_time("all_to_all", ...)`).

Specialised kernels (DeepEP) have low-latency modes for decode and high-throughput modes for prefill.
[MoE primer §6](../../00-foundations/mixture-of-experts/PRIMER.md#6-running-moe-on-gpus) is about the MoE side of
the exchange: placement, the slowest rank, TP against EP, and wide-EP.

**Pipeline parallelism** sends one activation per stage boundary (send/recv). That is sufficiently small to cross
the scale-out network. **Data parallelism** (replicas) needs no collectives at inference. This gives the rule of the
[deployment primer §4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu):
TP and EP inside the NVLink domain, PP and DP across it.

### 5.7 NCCL in practice

- **Init.** One rank creates a unique id. All ranks call `ncclCommInitRank`. Then NCCL **detects the topology**
  (GPUs, NVLink, PCIe switches, NICs, NUMA) from NVML and sysfs. It searches for the best rings and trees, and builds
  **channels**. Each channel is one ring or tree instance, and one CTA runs it. More channels give more bandwidth,
  and they also take more SMs from compute.
- **Transports.** NCCL has these transports:
  - P2P over NVLink or PCIe (CUDA IPC),
  - SHM through host memory, when P2P is not possible,
  - NET over InfiniBand/RoCE verbs, sockets or a vendor plugin (for example GCP's gIB, AWS's aws-ofi-nccl),
  - CollNet (SHARP in the switch),
  - NVLS (NVLink SHARP on NVSwitch systems, NCCL 2.17+).
- **Protocols.** *LL* stores 8 bytes (4 data + 4 flag) for the lowest latency at 50% efficiency. *LL128* moves
  128-byte lines that carry 120 bytes of data (~94%), where the hardware keeps the order. *Simple* uses large chunks
  with fences: full bandwidth, higher latency. For each call, the tuner selects algorithm × protocol × channels
  from a latency-and-bandwidth model like the model in §5.3.
- **Algorithms** that you can see by name: Ring, Tree, CollnetDirect, CollnetChain, NVLS, NVLSTree, and PAT (NCCL
  2.23+, for all-gather and reduce-scatter at scale). The current release is 2.32 (September 2026).

| Variable | Use |
|---|---|
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET` | print the topology, the rings and the transports ("via P2P/IPC", "via NET/IB/0/GDRDMA") |
| `NCCL_TOPO_DUMP_FILE`, `NCCL_GRAPH_DUMP_FILE` | dump the detected topology and the selected graphs |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | pin the bootstrap interface and the NICs |
| `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_NVLS_ENABLE` | switch transports off or on, to bisect a problem |
| `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | force choices for experiments. Do not deploy them without a test. |
| `NCCL_NET_GDR_LEVEL`, `NCCL_NET_PLUGIN` | GPUDirect RDMA distance and the network plugin |

### 5.8 Debugging hangs and slowness

NCCL matches collectives **by issue order on a communicator**. It does not examine names or sizes. The $k$-th
call on each rank is the same collective. Thus, if one rank takes a data-dependent branch, crashes, goes OOM, or
issues calls in a different order, the other ranks stay blocked. A size mismatch can cause a hang, or it can corrupt
the data with no error.

`gpusim.collectives.first_mismatch()` finds the first call where the per-rank logs do not agree. You do the same
with the flight recorder dumps of PyTorch or with `NCCL_DEBUG=INFO` logs. The checklist:

1. **Make hangs into errors.** Set timeouts and async error handling (`TORCH_NCCL_ASYNC_ERROR_HANDLING`). Use the
   flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`, verify names). Use NCCL's RAS client (`ncclras`, NCCL 2.24+) to
   see the state of each rank. Use `py-spy dump` for Python stacks.
2. **Make sure that the control flow is rank-uniform.** Also make sure that each rank reached the same call count.
3. **Examine the path.** In a node, the `NCCL_DEBUG=INFO` log must show the P2P or NVLS transport. SHM inside one node means
   that P2P is off. Possible causes are ACS or IOMMU on PCIe, containers without a shared IPC namespace, or
   `NCCL_P2P_DISABLE`. Compare with `nvidia-smi topo -m`.
4. **Examine the environment.** Look for an incorrect `NCCL_SOCKET_IFNAME` and firewalls between nodes. Make sure
   that the network plugin is available. Look for a small `/dev/shm` in containers (`--ipc=host` or
   `--shm-size`). Look for NCCL versions that do not match. Make sure that NVSwitch systems have a fabric manager.
5. **Then measure.** Run nccl-tests on the same nodes. Compare busbw with the link rate (the lab's notebook 04).

---

## 6. How a container gets a GPU

### 6.1 What a CUDA process needs

Containers share the host kernel. Thus the **kernel-mode driver is always the host's**. A CUDA process in a
container needs four things:

- the **device nodes** (`/dev/nvidia0`, one per GPU, plus `/dev/nvidiactl`, `/dev/nvidia-uvm` and
  `/dev/nvidia-uvm-tools`, and `/dev/nvidia-caps/*` for MIG), with permission from its cgroup for each of them,
- the **user-mode driver libraries** (`libcuda.so`, `libnvidia-ml.so`, `libnvidia-ptxjitcompiler.so`, ...) at
  *exactly* the version of the host driver,
- the host utilities that it wants (`nvidia-smi`),
- the **CUDA userland** (runtime and libraries).

Only the last one belongs in the image.

### 6.2 The NVIDIA Container Toolkit

```
docker run --gpus all    pod asking for nvidia.com/gpu: 1        docker run --device nvidia.com/gpu=all
        │                (device plugin Allocate returns env vars               │
        │                 or CDI device names)                                  │
        ▼                          │                                            ▼
 legacy: OCI prestart hook ◄─ env ─┴─ CDI names ─►  CDI: a spec (e.g. /etc/cdi/nvidia.yaml, from
 nvidia-container-runtime-hook                      `nvidia-ctk cdi generate`) names device nodes,
 → nvidia-container-cli; added                      library mounts and hooks. The engine applies it
 by Docker's --gpus or by the                       natively (recent Docker, containerd, CRI-O; verify
 nvidia-container-runtime shim                      versions), or the nvidia-container-runtime shim
 (a runc wrapper)                                   does it ("cdi" mode)
        └───────────────────────────┬───────────────────────────────────────────────┘
                                    ▼
   inside: /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm        (created and allowed)
           libcuda.so.1 → host's libcuda.so.<driver version>    (bind-mounted from the host)
           libnvidia-ml.so.1, nvidia-smi                        (bind-mounted from the host)
           + the image: libcudart, cuBLAS, cuDNN, NCCL, torch
```

`NVIDIA_VISIBLE_DEVICES` (`all`, indices, UUIDs, `none`, `void`) and `NVIDIA_DRIVER_CAPABILITIES`
(`compute,utility` in CUDA base images) control which GPUs and which parts of the driver the toolkit injects. In
the container, `CUDA_VISIBLE_DEVICES` also masks and renumbers what the process sees. `gpusim.compat.origin()`
classifies any path as host-injected or image.

### 6.3 In Kubernetes

The device plugin advertises `nvidia.com/gpu` as an extended resource. At pod admission, its `Allocate` call
returns the device IDs, environment variables, mounts or CDI device names for the selected GPUs. The container
runtime then injects them as §6.2 shows (layer 03's primer,
[`03-kubernetes-gpu/gpu-scheduling/PRIMER.md`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md) §1 *What Kubernetes
sees*). On GKE, Google manages the device plugin and installs the driver on the node (§9). With the NVIDIA GPU
Operator, the operator installs the driver, toolkit, plugin and DCGM exporter as pods.

### 6.4 Failure modes (`gpusim.compat.explain_container()`)

| Story | Result |
|---|---|
| `docker run` without `--gpus` or a CDI device, `NVIDIA_VISIBLE_DEVICES` unset or `void`, or no GPU in the pod spec | error 100, no device |
| the Dockerfile copies `libcuda.so` from a build machine into the image | error 803: the user-mode and kernel-mode drivers do not agree |
| the image CUDA is a newer major than the host driver | error 35. `cuda-compat` in the image corrects it on data-center GPUs over a supported driver branch (804 on GeForce, 803 on other branches). |
| a wheel without SASS for the GPU and no usable PTX | error 209 |
| `nvidia-smi` works but `torch.cuda.is_available()` is False | a CPU-only wheel, or `CUDA_VISIBLE_DEVICES` masks every GPU |
| NCCL hangs or uses slow fallback paths in containers | `/dev/shm` is too small, or there is no shared IPC namespace for CUDA IPC |

---

## 7. Sharing a GPU

### 7.1 First, ask whether to share

A 70B model on 8 GPUs does not share them. Its engine batches hundreds of requests. **Continuous batching is the
most efficient way to share GPUs**, because it shares the weight reads. GPU-level sharing is for many *small*
things: small models, dev notebooks, low-QPS endpoints, CI.

### 7.2 MIG: partitions

Multi-Instance GPU divides one GPU into up to **seven instances**. It is available on the A100, A30, H100, H200
and B200 class, but not on L4, T4 or RTX 40. Each instance has its own SMs, L2 slice, memory controllers and
memory. This gives hardware isolation of performance, memory and faults.

The geometry has strict rules. There are 7 compute slices and 8 memory slices. A profile takes a constant number of
each, and it can *start* only at listed slice indices (`nvidia-smi mig -lgipp`, `gpusim.sharing.MIG`, verify):

| H100 80 GB profile | Compute | Memory slices | Allowed starts | Max |
|---|---|---|---|---|
| 1g.10gb | 1/7 | 1 | 0–6 | 7 |
| 1g.20gb | 1/7 | 2 | 0, 2, 4, 6 | 4 |
| 2g.20gb | 2/7 | 2 | 0, 2, 4 | 3 |
| 3g.40gb | 3/7 | 4 | 0, 4 | 2 |
| 4g.40gb | 4/7 | 4 | 0 | 1 |
| 7g.80gb | 7/7 | 8 | 0 | 1 |

A100 80 GB has the same shapes. A100 40 GB has half the memory (1g.5gb ... 7g.40gb), and H200 141 GB increases it
(1g.18gb ... 7g.141gb). `gpusim.sharing.pack()` searches for a layout:

```
4g.40gb + 2g.20gb + 1g.10gb           | 4g  |  -  |  -  |  -  | 2g  |  -  | 1g  |  .  |   fits
3g.40gb + 2g.20gb + 1g.10gb × 2       | 2g  |  -  | 1g  | 1g  | 3g  |  -  |  -  |  -  |   fits
3g.40gb + 3g.40gb + 1g.10gb           does not fit: 3 + 3 + 1 = 7 compute slices, but the two 3g take all 8 memory slices
1g × 3, then 4g, created in arrival order at the first free slice   the 4g no longer fits (first_fit())
```

Thus plan a layout for each node pool, and create the instances with explicit placements. Kubernetes exposes MIG
in one of two ways. The first is plain `nvidia.com/gpu` with one profile per node ("single" strategy). The second is
`nvidia.com/mig-1g.10gb`-style resources ("mixed").

GKE sets one partition size per node pool (§9). Layer 03 explains how to share GPUs at the cluster level
([`03-kubernetes-gpu/gpu-scheduling/PRIMER.md`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md) §9 *Sharing GPUs
at the cluster level*). A 7g instance is not all of the GPU. In MIG mode, each slice gets a constant number of SMs.
Thus the 7g of an A100 has 98 of its 108 SMs (verify).

### 7.3 MPS: overlap

The Multi-Process Service runs a server that merges the kernels of many client processes onto the GPU at the same
time. Thus small kernels from different processes fill the SMs together, and do not take turns. Since Volta, each
client has its own address space. You can set limits on a client with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and
`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`. Fault isolation is weaker: a fatal fault in one client can stop the other
clients on that GPU (verify for your driver). MPS is applicable to cooperative workloads of one team.

### 7.4 Time-slicing: turns

The device plugin (or GKE's `max_shared_clients_per_gpu`) advertises one GPU as $N$ schedulable replicas. The GPU
runs one context at a time, round-robin. There is **no memory isolation**. Thus one tenant can cause an OOM in the
other tenants. There is also no performance isolation.

Take $N$ busy tenants and a request that needs $W$ of GPU time, in quanta $q$, with a switch cost $s$. At best, the
request finishes after $W + (\lceil W/q \rceil - 1) \cdot ((N-1)(q+s) + s)$. That occurs when it arrives just as its
turn starts (`gpusim.sharing.timeslice_latency()`). For example, 10 ms of work with 4 busy tenants, 2 ms quanta and
50 µs switches takes **34.8 ms at best**. If the request arrives just after its turn, the late arrival adds one more
round of the turns of the other tenants, 6.2 ms. That gives **41.0 ms at worst** and **37.9 ms on average**.

Idle tenants cost nothing. This is why time-slicing is good for notebooks and dev work that comes in bursts.

### 7.5 Choosing

This is an idealised latency model that counts only SM capacity (`gpusim.sharing.shared_latency()`). One request
needs 10 ms alone, and 4 always-busy tenants share the GPU. `util` is the share of the GPU that the kernels of the
request fill. The time-slicing column is the mean over arrival times from §7.4 (34.8 ms at best, 41.0 ms at worst):

| Kernel size | Exclusive | Time-slicing | MPS | MIG (1g each) |
|---|---|---|---|---|
| small (util 0.2) | 10 ms | 37.9 ms | 10 ms | 14 ms |
| fills the GPU (util 1.0) | 10 ms | 37.9 ms | 40 ms | 70 ms |

| Need | Choose |
|---|---|
| tenants who must not affect each other, SLOs, untrusted code | MIG |
| many small cooperative processes, throughput first | MPS |
| idle-heavy notebooks, CI, demos | time-slicing |
| one large model | none: whole GPUs, batching in the engine |

---

## 8. Health and observability

### 8.1 nvidia-smi and NVML

`nvidia-smi` (on NVML) is the first tool to use. It gives:

- the banner,
- `-q` for everything,
- `--query-gpu=... --format=csv` for scripts,
- `topo -m` for the link matrix (layer 01 §5.6),
- `-q -d PERFORMANCE` for clock-event reasons,
- `-q -d ECC` and `-q -d ROW_REMAPPER` for memory health,
- `mig -lgip` / `-lgipp` for MIG.

### 8.2 DCGM, and why "GPU util" misleads

**DCGM** (the Data Center GPU Manager) samples the same counters across the fleet, and **dcgm-exporter** exposes
them to Prometheus. The field that everyone puts on a graph is `DCGM_FI_DEV_GPU_UTIL` (nvidia-smi's "GPU-Util"). It
is the **share of time when at least one kernel ran**. Take a kernel with 8 blocks on a 132-SM H100 that runs all
the time. It reads **100%**, but only **6%** of the SMs do anything (`gpusim.health.util_counters()`). Read these
fields instead:

| Field (dcgm-exporter) | Meaning | Reading it |
|---|---|---|
| `DCGM_FI_PROF_SM_ACTIVE` | share of SM-cycles with at least one warp resident | the real "how much of the GPU" |
| `DCGM_FI_PROF_SM_OCCUPANCY` | resident warps / maximum | headroom to hide latency (§2.3) |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | share of cycles when the tensor pipes are busy | high in prefill and large batches |
| `DCGM_FI_PROF_DRAM_ACTIVE` | share of cycles when the memory interface is busy | high in decode: memory-bound |
| `DCGM_FI_PROF_PCIE_*_BYTES`, `DCGM_FI_PROF_NVLINK_*_BYTES` | link traffic | model load, offload, collectives |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | framebuffer memory | Near-full is *normal* for engines that pre-allocate KV. Use the KV-usage metric of the engine. |
| `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_SM_CLOCK` | power, temperature, clocks | with the throttle reasons of §8.3 |
| `DCGM_FI_DEV_XID_ERRORS` | last XID error | triage in §8.4 |

**Find which fields your exporter exports.** The default counters file of stock dcgm-exporter
(`default-counters.csv`) has `DCGM_FI_PROF_SM_ACTIVE` and `DCGM_FI_PROF_SM_OCCUPANCY` only as comments. It does not
include `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (§8.3). Thus, by default, you get neither the first two rows of the table
nor the throttle bits.

The lab supplies a counters file that adds all three
([`deploy/any-gpu/dcgm-counters.csv`](cuda-nccl-lab/deploy/any-gpu/dcgm-counters.csv)). The field lists for each
exporter are `gpurt.dcgm.EXPORTED_BY`. [`deploy/gke/README.md`](cuda-nccl-lab/deploy/gke/README.md) shows which
alert rules can fire with each exporter.

The DCGM field list of Google's managed Prometheus (its `nvidia-dcgm` example, which GKE's managed package is similar to)
has SM active, occupancy and tensor active. But it has no XID or row-remap fields, and no clock-event or DRAM fields
(verify). Thus, on GKE, the §8.3–8.4 signals need a self-managed exporter with that counters file.

`gpusim.health.diagnose()` turns a sample into findings. Examples are *"busy but mostly empty"* (util high, SM
active low: small batch, launch gaps, small grids) and *"memory-bandwidth-bound"*. Notebook 05 shows both.

### 8.3 Throttling

`DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (the old name is `..._CLOCK_THROTTLE_REASONS`) is the bitmask of NVML
(`gpusim.health.decode_throttle()`):

- `0x1` idle,
- `0x2` application clocks setting,
- `0x4` software power cap,
- `0x8` hardware slowdown,
- `0x10` sync boost,
- `0x20` software thermal slowdown,
- `0x40` hardware thermal slowdown,
- `0x80` hardware power brake,
- `0x100` display clock setting.

A power cap under load is normal. Any `hw_*` reason or thermal slowdown is a ticket for the facilities team.

### 8.4 XIDs, ECC and row remapping

XIDs are the error reports of the driver (in `dmesg` and DCGM). What is important is **who acts**
(`gpusim.health.triage_xid()`). The catalogue is NVIDIA's XID documentation (verify):

| XID | Meaning | Owner | Action |
|---|---|---|---|
| 13, 31, 43 | graphics engine exception, GPU memory page fault, GPU stopped processing | app | Usually an out-of-bounds or illegal address in a kernel. Correct the kernel, then restart the app. |
| 45 | preemptive cleanup | app | Secondary. Read the XID that came before it. |
| 63 | row-remapping (or page-retirement) event | node | A remap is pending until the GPU resets. Drain the node, then reset the GPU. |
| 94 | contained ECC error | node | Only the affected app crashed. Restart it. Reset the GPU after you drain the node. |
| 48, 95 | double-bit ECC error, uncontained ECC error | hardware | Drain and reset now. If it occurs again, send the GPU for RMA. |
| 64 | row-remapper failure | hardware | Drain the node. Send the GPU for RMA. |
| 74, 79 | NVLink error, GPU fell off the bus | hardware | Drain, reboot, diagnose (PCIe, power, thermals). If it occurs again, send the GPU for RMA. |
| 92, 119 | high single-bit ECC rate, GSP RPC timeout | node | Schedule diagnostics. Reset the GPU. Update the driver. |

Since A100, the GPU **remaps rows** of HBM that show memory errors. It does not retire pages. The remap takes
effect at the next GPU reset. A remap *failure* means that you must replace the GPU. `gpusim.health.alerts()` maps
all of this to severities: page, ticket, notify.

### 8.5 Fleet practice

Run `dcgmi diag -r 1|2|3` (short to long) before a node joins a pool, and after hardware XIDs. Drain nodes
automatically on hardware XIDs and pending remaps. Alert on SM active and on the engine's own metrics, not on GPU
util. On NVSwitch systems (HGX H100/B200), keep `nvidia-fabricmanager` at the version of the driver. If not, CUDA
returns error 802.

---

## 9. On GCP and elsewhere

This section shows the same concepts on each platform. [`COMPUTE.md`](../../COMPUTE.md) gives the prices and the
availability.

| Concept | T0 (`gpusim`, any laptop) | T1/T2 (any GPU box) | T3 (GCP) |
|---|---|---|---|
| kernels, coalescing, occupancy | `simt`, `occupancy`, `tiling` | lab `01` and `02`: Numba kernels, simulator then GPU | same kernels on an L4 node |
| collectives, busbw | `collectives` | lab `03` and `04`: collectives over OS pipes (a real multi-process ring, no torch) or gloo at T0. NCCL and nccl-tests on 2+ GPUs. | 2-GPU nccl-tests Job on `g2-standard-24` (2 × L4) |
| compatibility, containers | `compat` | lab `05`: what the container sees | GKE-installed drivers, container images |
| sharing, health | `sharing`, `health` | MIG and DCGM on a rented A100/H100 VM | GKE MIG and time-sharing pools, DCGM metrics (lab `06`) |

**On GCP.** GKE installs the NVIDIA driver on GPU nodes for you. From control plane 1.32.2-gke.1297000, it
installs the default driver automatically. `gpu_driver_version` takes `DEFAULT`, `LATEST` (COS only) or
`INSTALLATION_DISABLED` (for the GPU Operator path). GKE puts the taint `nvidia.com/gpu=present:NoSchedule` on GPU
nodes (verify).

You configure GPU sharing per node pool. For MIG, use `guest_accelerator.gpu_partition_size` (for example
`"1g.10gb"`). For time-sharing or MPS, use `gpu_sharing_config` (`gpu_sharing_strategy = "TIME_SHARING" | "MPS"`,
`max_shared_clients_per_gpu`).

**Deep Learning VM** images have preinstalled drivers. For multi-node NCCL, A3 High (H100, `a3-highgpu-8g`) uses
GPUDirect-TCPX (verify). A3 Mega (H100) uses GPUDirect-TCPXO. A3 Ultra (H200), A4 (B200), A4X (GB200 NVL72) and
A4X Max (GB300 NVL72) use GPUDirect RDMA over ConnectX-7 NICs. They use a rail-aligned network with Google's NCCL
network plugin (gIB, verify names). DCGM metrics can go into Cloud Monitoring through GKE's managed collection
(verify).

The deploy assets in [`cuda-nccl-lab`](cuda-nccl-lab/) hold the Terraform and the Jobs. The Terraform is for a zonal
GKE Standard cluster with these node pools:

- an L4 Spot pool that scales from zero,
- an `l4x2` pool of `g2-standard-24` nodes (2 × L4, also Spot and from zero) that hosts the 2-GPU nccl-tests Job,
- optional time-sharing and MIG pools.

**Elsewhere.** **Colab** gives one T4 (CC 7.5, and thus no bf16 tensor cores). **Kaggle** gives **2 × T4 over PCIe**
for free. That is a real 2-GPU NCCL box without NVLink. There, `NCCL_DEBUG=INFO` shows P2P or SHM over PCIe, and
PCIe sets the limit for busbw.

**RunPod and Vast** give you a *container* on a host that another person owns. You cannot change the driver. Thus
select an image whose CUDA the host driver supports (§1.2). MIG and DCGM profiling are usually not available.
**Lambda and GCP** give you *VMs*: you own the driver, MIG, fabric manager and DCGM.

A laptop runs everything in `gpusim`. The lab's Numba kernels run in `NUMBA_ENABLE_CUDASIM=1`. A local kind cluster
with fake GPU capacity (layer 03) lets you try the scheduler, not CUDA. It has no device nodes to inject.

---

## In a design review

**The two-minute walkthrough.** "Each node pool pins one driver branch. Images contain only CUDA userland. We build
them at or below the CUDA version of that driver, or in its major. Each image passes a check that every kernel has
SASS for our GPUs (`torch.cuda.get_arch_list()`). The NVIDIA Container Toolkit injects the libcuda and the devices
of the host, and thus the user-mode and kernel-mode drivers always match.

"Our hot kernels are memory-bound. We count bytes and sectors, and we tile and fuse. At small batch, the launch path
is the largest cost. Thus we capture decode as CUDA Graphs per batch bucket.

"We run tensor parallelism inside the NVLink domain. Decode all-reduces are approximately 0.5 MB, far below the
~7 MB latency/bandwidth crossover. Thus we use a few-step all-reduce (two-shot or NVLS) inside the graph. Before any
model runs, we examine the fabric. We measure nccl-tests busbw and compare it with the NVLink rate.

"We give whole GPUs to large models. Small multi-tenant endpoints get MIG layouts that we plan. Dev notebooks get
time-slicing. We alert on SM active and engine metrics, not GPU util. Hardware XIDs drain nodes automatically."

**Drill questions**

1. *nvidia-smi on the nodes says CUDA 12.2, our image is CUDA 12.8. Do we ship?* Yes, if each kernel has SASS for
   our GPUs. Minor-version compatibility (driver >= 525.60.13) runs it. PTX-only kernels fail with 222. APIs newer
   than 12.2 fail with 36. A driver upgrade is the clean solution. A CUDA 13 image needs a driver upgrade, or
   `cuda-compat` on data-center GPUs.
2. *TP=8 decode spends 40% of the step in all-reduce, yet NVLink is nearly idle. Why?* The messages are hundreds of
   KB. Thus the 14 $\alpha$-steps of a ring are the largest cost. The all-reduce is latency-bound, not
   bandwidth-bound. Use fewer steps (two-shot or one-shot custom all-reduce, NVLS). Capture them in CUDA graphs.
   Fuse them with the norm. As an alternative, decrease the TP degree for this model.
3. *nccl-tests on one 8×H100 node shows all-reduce busbw of 100 GB/s. What do you check?* First, make sure that NCCL
   uses NVLink at all. The `NCCL_DEBUG=INFO` log must show P2P or NVLS, not SHM or NET. Then examine the
   fabric manager status and `nvidia-smi topo -m`. Look for `NCCL_P2P_DISABLE` left set, and for containers without
   shared IPC. On NVLink 4, the expected busbw is a few hundred GB/s, near the link rate, not 100.
4. *The dashboard shows 100% GPU utilization but low tokens/s. Is the GPU saturated?* Not necessarily. GPU util is
   time-based. Examine SM active and tensor active. Small-batch decode, launch gaps and small grids read 100% with a
   few percent of the SMs busy. Batch more requests. Capture CUDA graphs. Fuse the kernels.
5. *Seven small customer endpoints on one H100: MIG or time-slicing?* MIG with seven `1g.10gb` instances gives each
   endpoint an isolated, predictable slice. Time-slicing shares memory without limits. It also multiplies the
   latency by the number of busy tenants. If all seven are adapters of one base model, serve them from one engine
   with multi-LoRA instead.
6. *A node logged XID 79, another XID 13. Who acts?* XID 79 (the GPU fell off the bus) is hardware. Drain, reboot
   and diagnose. If it occurs again, send the GPU for RMA. XID 13 is usually a fault in the kernel of the
   application. Send it to the service owner. It is a problem of the node only if it occurs again across unrelated
   apps.

---

## Glossary

| Term | Meaning |
|---|---|
| SM | streaming multiprocessor: 4 sub-partitions, each with a warp scheduler and a register-file slice |
| warp | 32 threads issued together, the unit of scheduling |
| SIMT, divergence | one instruction stream per warp, and lanes on different branches serialize |
| occupancy | resident warps / maximum warps per SM |
| sector, line | the 32-byte unit of global-memory traffic (a 128-byte cache line = 4 sectors) |
| coalescing | the combination of the accesses of a warp into few sectors |
| bank conflict | lanes that hit different 4-byte words in the same one of 32 shared-memory banks |
| compute capability (CC) | the architecture version of a GPU, X.Y (SASS uses `sm_XY`) |
| SASS, PTX, fatbin | machine code, a virtual ISA that the driver JIT-compiles, a binary that holds more than one of each |
| minor-version compatibility | a newer runtime on an older driver of the same CUDA major, with limits |
| forward compatibility | `cuda-compat`: a newer user-mode driver on an older kernel module (data-center GPUs) |
| stream | an in-order queue of GPU work (different streams can overlap) |
| CUDA Graph | a captured sequence of GPU work replayed with one launch |
| collective | communication operation defined over all ranks of a communicator |
| $\alpha$, $B$ | per-step latency, per-direction link bandwidth (layer 01 writes $B$ as $\beta$) |
| algbw, busbw | nccl-tests' ${S/t}$, and ${S/t}$ × a per-collective factor comparable with link bandwidth |
| channel | one NCCL ring or tree instance, run by one CTA |
| LL, LL128, Simple | NCCL protocols: lowest latency, near-full bandwidth, full bandwidth |
| NVLS, SHARP | in-network reduction in NVSwitch (NVLink SHARP) or InfiniBand switches |
| CDI | Container Device Interface: a spec that names the devices, mounts and hooks to inject |
| MIG, MPS, time-slicing | partition, overlap, turns: the three ways to share a GPU |
| DCGM | Data Center GPU Manager, whose fields dcgm-exporter publishes to Prometheus |
| XID | a driver error code in the kernel log |
| row remapping | the replacement of faulty HBM rows by the GPU (A100+), with effect after a GPU reset |

---

## Sources

- NVIDIA, *CUDA C++ Programming Guide*: execution model, compute capabilities and their limits, binary and
  PTX compatibility. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*: coalescing, shared memory, occupancy. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA, *CUDA Compatibility* (minor-version and forward compatibility) and the *CUDA Toolkit Release Notes*
  (driver table). <https://docs.nvidia.com/deploy/cuda-compatibility/>. The check of `gpusim.compat` used these
  copies: the driver table in meson's `mesonbuild/modules/cuda.py`, and `NVIDIA_REQUIRE_CUDA` in NVIDIA's
  `nvidia/cuda` images.
- NVIDIA, `cuda_occupancy.h` (part of the CUDA Toolkit): the occupancy arithmetic that `gpusim` uses.
- NVIDIA, *Nsight Compute* documentation, Memory Workload Analysis: sectors per request, bank conflicts.
- V. Volkov, *Better Performance at Lower Occupancy*, GTC 2010.
- NVIDIA, *NCCL User Guide*, with the environment variables. <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>
- NVIDIA, nccl-tests `doc/PERFORMANCE.md`: algbw and busbw. <https://github.com/NVIDIA/nccl-tests>
- S. Jeaugey, *Massively Scale Your Deep Learning Training with NCCL 2.4*, NVIDIA Technical Blog, 2019 (double binary trees).
- R. Thakur, R. Rabenseifner, W. Gropp, *Optimization of Collective Communication Operations in MPICH*,
  IJHPCA 2005 (α-β costs of ring, tree, recursive halving and doubling).
- P. Patarasuk, X. Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations*, JPDC 2009.
- M. Shoeybi et al., *Megatron-LM*, 2019 (tensor parallelism, two all-reduces per layer). V. Korthikanti et al.,
  *Reducing Activation Recomputation in Large Transformer Models*, 2022 (sequence parallelism).
- vLLM source, `csrc/custom_all_reduce.cuh` (one- and two-stage all-reduce). DeepSeek-AI, DeepEP
  (expert-parallel all-to-all). <https://github.com/vllm-project/vllm>, <https://github.com/deepseek-ai/DeepEP>
- NVIDIA Container Toolkit documentation. Container Device Interface specification.
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/>, <https://github.com/cncf-tags/container-device-interface>
- NVIDIA, *Multi-Instance GPU User Guide*, *Multi-Process Service*, and GPU Operator, *Time-Slicing GPUs in Kubernetes*.
  <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/>, <https://docs.nvidia.com/deploy/mps/>
- NVIDIA, *DCGM* documentation (field identifiers) and dcgm-exporter, NVML API reference (clock event reasons),
  *XID Errors*. <https://github.com/NVIDIA/dcgm-exporter>, <https://docs.nvidia.com/deploy/xid-errors/>
- Google Cloud, GKE documentation: GPUs in Standard node pools, GPU sharing strategies (MIG, time-sharing, MPS),
  GPUDirect-TCPX/TCPXO and RDMA networks. <https://cloud.google.com/kubernetes-engine/docs>
- PyTorch documentation: CUDA semantics (streams, CUDA Graphs), `torch.compile`, distributed debugging.

---

## Verify list

The date of this list is 2026-09-26. Each fact in it can change. Do a check of each fact before you rely on it.

| Fact used here | Where in `gpusim` | Status |
|---|---|---|
| Minimum drivers for CUDA 9.0–13.3 (release notes). 13.2 = 595.45.04 and 13.3 = 610.43.02 come from meson's copy of the table. | `compat.CUDA_MIN_DRIVER` | checked against copies of the table, 2026-09-26, verify |
| CUDA 13.4 maps to driver branch R615 (exact minimum unknown) | `compat.INFERRED` | **inferred** from `nvidia-ml-py` 13.615.71 and CUDA runtime wheels 13.4 on PyPI, verify |
| Minor-compatibility floors 11.x 450.80.02, 12.x 525.60.13, 13.x 580.65.06 | `compat.MINOR_COMPAT_FLOOR` | release notes give 11.x >= 450.80.02, 12.x >= 525, 13.x >= 580, verify |
| Kernel-driver branches that each `cuda-compat` package runs over (13.0: R535, R550, R565, R570, R575) | `compat.COMPAT_BRANCHES` | read from `NVIDIA_REQUIRE_CUDA` in nvidia/cuda 12.4, 12.8, 13.0, 13.1 images, verify in the CUDA Compatibility guide |
| CUDA 13 removed Maxwell, Pascal and Volta targets (Turing kept). The first toolkit that targets the CC of a GPU gives an approximation of its minimum driver. | `compat.FIRST_CUDA` | verify, and also the approximation per GPU |
| CC of B300/GB300 (10.3), RTX 50 / RTX PRO 6000 (12.0), first toolkits 12.8 / 12.9, 'f' family targets | `compat.GPUS`, `compat.FIRST_CUDA` | verify |
| Forward compatibility on RTX PRO 6000 (server edition) | `compat.GPUS` (data-center flag) | verify |
| Per-SM limits for CC 10.0 | `occupancy.SMS` | verify |
| MIG profiles and placements for A100, H100, H200. A30 and B200 profiles are different. | `sharing.MIG` | verify with `nvidia-smi mig -lgipp` |
| SMs available to a 7g MIG instance (A100: 98 of 108) | primer §7.2 | verify |
| MPS fault-isolation behaviour on current drivers | `sharing.TRAITS` | verify |
| Latest NCCL 2.32.3 (PyPI `nvidia-nccl-cu12/cu13`, 2026-09-22). NVLS since 2.17, PAT since 2.23, RAS since 2.24. | primer §5.7 | version checked on PyPI, feature versions verify |
| NCCL and PyTorch environment-variable names (`TORCH_NCCL_*`) | primer §5.7–5.8 | verify for your versions |
| DCGM field names. `CLOCK_THROTTLE_REASONS` has the new name `CLOCKS_EVENT_REASONS`. | `health` | verify |
| XID meanings and recommended actions | `health.XIDS` | verify |
| The versions of Docker, containerd and CRI-O that support CDI natively. Docker's `--gpus` adds the prestart hook itself. | primer §6.2 | verify |
| L2 sizes: T4 4 MB, A100 40 MB, L4 48 MB, H100 50 MB | primer §3.6 | verify |
| vLLM flags (`--enforce-eager`, CUDA-graph capture sizes) | primer §4.3 | verify |
| GKE driver auto-install from 1.32.2-gke.1297000, `gpu_driver_version` values | primer §9 | from project FACTS (2026-09-26) |
| The GKE GPU taint `nvidia.com/gpu=present:NoSchedule`, the gIB plugin name, DCGM in Cloud Monitoring. A3 High uses GPUDirect-TCPX. | primer §9 | verify |
| A3 Mega uses GPUDirect-TCPXO. A3 Ultra, A4, A4X and A4X Max use GPUDirect RDMA on ConnectX-7. | primer §9 | from project FACTS |
| Kaggle 2 × T4 (PCIe), 30 GPU-hours per week | primer §9 | from project FACTS, verify quotas |
| Illustrative $\alpha$ = 2 µs and $B$ = 450 GB/s (NVLink 4), loaded DRAM latency 600 ns, eager launch 5 µs, graph launch 10 µs. NVLS busbw 744 GB/s (model). | primer §2.4, §4.2, §5 | assumptions and model output, not measurements. Measure your own values in the lab. |
