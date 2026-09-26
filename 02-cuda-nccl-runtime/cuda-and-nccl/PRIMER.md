# The GPU software substrate: CUDA's execution model, collectives, and how a container gets a GPU

*Layer 02 of the stack. Written September 2026. Versions, profiles and product names move; the rules
under them do not. Everything dated is collected in the [Verify list](#verify-list).*

This primer covers the software between the silicon ([layer 01](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md))
and the schedulers and engines above it (layer 03, [`03-kubernetes-gpu/gpu-scheduling/`](../../03-kubernetes-gpu/gpu-scheduling/), and layer 04,
[`04-inference-engine/serving-engine/`](../../04-inference-engine/serving-engine/)). It explains how a driver, a CUDA runtime and a compiled kernel agree to
run; how a kernel executes (warps, occupancy, 32-byte memory transactions); why launches cost time; how GPUs
exchange data (collectives and NCCL); how a container gets a GPU; how one GPU is shared; and how to tell whether
it is healthy. Every formula here is computed by the `gpusim` package in
[`cuda-nccl-core/`](cuda-nccl-core/) (T0, runs on a laptop CPU) and measured for real in
[`cuda-nccl-lab/`](cuda-nccl-lab/) (T1 to T3). It assumes the [GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md)
and links to it rather than repeating it.

---

## The one-minute version

- **Two gates decide whether CUDA code runs.** First, the host *driver* must support the CUDA *runtime* your app was
  built with. Newer drivers run older runtimes. A newer runtime of the same major runs by minor-version
  compatibility. A newer major needs a newer driver, or the forward-compatibility package on a data-center GPU.
  Second, the binary must carry a *kernel image* for the GPU: SASS for its compute capability (same major, equal or
  lower minor) or PTX the driver can JIT-compile. Otherwise you get *"no kernel image is available"*.
- **The warp is the unit.** 32 threads share one instruction stream. Their memory requests are served in 32-byte
  sectors: consecutive 4-byte loads take 4 sectors per warp, a column walk takes 32. Shared memory has 32 banks of 4
  bytes, and a word stride of s costs a gcd(s, 32)-way conflict. A split branch runs both sides.
- **Memory-bound kernels cost bytes plus launches.** Tiling and fusion cut bytes. CUDA Graphs cut launches, which is
  why engines capture decode steps. Occupancy is a means to keep enough bytes in flight (Little's law), not a goal.
- **All-reduce = reduce-scatter + all-gather.** On a ring that is 2(p−1) steps of S/p bytes:
  `T = 2(p−1)·α + 2(p−1)/p · S/B`. Below about `p·α·B` (7.2 MB for 8 H100s with illustrative numbers) it is
  latency-bound. Tensor-parallel decode all-reduces are well below that, so engines use few-step algorithms.
  nccl-tests' **busbw** normalises to the per-link bandwidth so it can be compared with the spec.
- **A container brings CUDA; the host brings the driver.** The NVIDIA Container Toolkit injects `/dev/nvidia*`,
  `libcuda` and NVML at container start, through an OCI hook or a CDI spec. Never bake `libcuda` into an image.
- **Sharing:** MIG partitions the GPU (isolated, fixed shapes), MPS overlaps processes (efficient, weakly isolated),
  time-slicing takes turns (no isolation). For LLM serving, the engine's batching is the best sharing.
- **Health:** "GPU util" is the share of *time* a kernel ran, not how much of the GPU it used. Read SM active,
  tensor active and DRAM active, and triage XIDs by who has to act.

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

Two version numbers matter. The **driver version** (for example `580.65.06`) belongs to the host. The
**CUDA version** your app was built with belongs to the runtime in your wheel or image (`torch.version.cuda`,
`nvcc --version`). The banner of `nvidia-smi` prints *"CUDA Version: 13.0"* next to the driver. That is the
**newest runtime this driver supports**, not an installed toolkit (`gpusim.compat.driver_cuda()`).

### 1.2 Gate 1: driver and runtime

| Case | Condition | Outcome |
|---|---|---|
| Backward compatibility | driver's CUDA >= app's CUDA | runs |
| Minor-version compatibility | same major, driver >= that major's floor (12.x: 525.60.13) | runs, but PTX from the newer toolkit cannot be JIT-compiled (error 222) and APIs newer than the driver fail (error 36) |
| Newer major | e.g. a CUDA 13 app on a 12.x driver | fails with 35, *"CUDA driver version is insufficient for CUDA runtime version"* |
| Forward compatibility | the `cuda-compat` package supplies a newer user-mode driver on an older kernel module | runs on data-center GPUs with a supported kernel-driver branch; otherwise 804 (hardware) or 803 (branch) |

The minimum driver per toolkit (Linux x86-64, from the release notes; the full table back to CUDA 9.0 is
`gpusim.compat.CUDA_MIN_DRIVER`, verify; for older drivers `check()` answers *cannot judge*):

| CUDA | 11.0 | 12.2 | 12.4 | 12.8 | 12.9 | 13.0 | 13.1 | 13.2 | 13.3 | 13.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| driver >= | 450.51.05 | 535.54.03 | 550.54.14 | 570.26 | 575.51.03 | 580.65.06 | 590.44.01 | 595.45.04 | 610.43.02 | R615 (inferred) |

Worked with `gpusim.compat.check()`. An app built with **CUDA 12.4** runs on an **H100** under driver
**535.183.01**, which supports CUDA 12.2:

- with SASS for sm_80 and sm_90, it **runs** by minor-version compatibility;
- if the kernels ship only `compute_80` PTX, it **fails with 222**: the 535 JIT cannot read PTX from 12.4;
- a **CUDA 13.0** build **fails with 35**. With `cuda-compat` 13.0 it runs on the H100, because R535 is a kernel-driver
  branch that package supports (`compat.COMPAT_BRANCHES`, verify); over R560 it fails with 803, on an RTX 4090 with 804.

### 1.3 Gate 2: kernel image and GPU

Every NVIDIA GPU has a **compute capability** X.Y. `nvcc` compiles device code to **PTX** (a virtual ISA,
`compute_XY`) and then to **SASS** (machine code, `sm_XY`). A fatbin can hold several of each (`-gencode
arch=compute_90,code=[sm_90,compute_90]`; PyTorch spells it `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"`, parsed by
`gpusim.compat.parse_targets()`). At load time the driver picks the best matching SASS. Failing that, it
JIT-compiles PTX (slow first start, cached in `~/.nv/ComputeCache`). Failing both, you get error **209**.

```
SASS sm_XY  runs on CC X.Z with Z >= Y        sm_80 → 8.0, 8.6, 8.9       never 9.0, never 7.5
PTX  compute_XY  JITs to CC >= X.Y            compute_80 → 8.x, 9.0, 10.0, 12.0 ...
'a' targets (sm_90a, sm_100a)                 exactly one CC: arch-specific instructions (wgmma, tcgen05)
'f' targets (sm_100f, compute_100f; 12.9+)    the same family: same major, minor >= Y (verify)
```
(`gpusim.compat.sass_runs_on()`, `ptx_jits_to()`)

| GPU | Architecture | CC | First CUDA | Notes (verify) |
|---|---|---|---|---|
| V100 | Volta | 7.0 | 9.0 | CUDA 13 dropped Maxwell/Pascal/Volta targets |
| T4 | Turing | 7.5 | 10.0 | Colab and Kaggle free tier |
| A100, A30 | Ampere | 8.0 | 11.0 | MIG |
| A10G, A40, RTX 30 | Ampere | 8.6 | 11.1 | |
| L4, L40S, RTX 40 | Ada Lovelace | 8.9 | 11.8 | FP8 |
| H100, H200 | Hopper | 9.0 | 11.8 | sm_90a for wgmma/TMA kernels |
| B200, GB200 | Blackwell | 10.0 | 12.8 | FP4 |
| B300, GB300 | Blackwell Ultra | 10.3 | 12.9 | |
| RTX 5090, RTX PRO 6000 | Blackwell | 12.0 | 12.8 | consumer/workstation line; *not* binary-compatible with 10.0 |

The classic failure: a wheel built for `8.0 8.6 9.0` without `+PTX`, run on an RTX 5090 (12.0), fails with
**209**. No sm_12x SASS is in it, and no PTX can be JIT-compiled. With `9.0+PTX` it would JIT and run
(slowly at first). On a T4 the same wheel fails too, because PTX only JITs *upward*. The same check predicts
"no device" when the driver's branch predates the first toolkit that targets the GPU, for example a B200 on an
R550 driver (an approximation of each GPU's minimum driver; verify).

### 1.4 Libraries, frameworks and the errors you will meet

PyTorch's Linux wheels carry their own CUDA runtime, cuBLAS, cuDNN and NCCL as `nvidia-*` pip packages, so
a host needs **only a driver**. **Triton** compiles Python kernels at first call through PTX to a cubin with its
own bundled `ptxas`, then caches them (`~/.triton/cache`). **cuBLAS/cuBLASLt** pick GEMM kernels per shape.
**NCCL** is §5. `torch.cuda.get_arch_list()` lists the SASS/PTX targets of your build.

| Error | Code | Meaning | Usual fix |
|---|---|---|---|
| CUDA driver version is insufficient for CUDA runtime version | 35 | runtime is a newer major, or below the minor-compat floor | upgrade the driver, or older CUDA, or `cuda-compat` (data center) |
| API call is not supported in the installed CUDA driver | 36 | minor-compat app called a newer API | upgrade the driver |
| no CUDA-capable device is detected | 100 | no GPU injected, or the driver does not know the GPU | `--gpus`, CDI device, `nvidia.com/gpu` request, newer driver |
| no kernel image is available for execution on the device | 209 | no SASS for this CC and no usable PTX | build for this sm, or add `+PTX` |
| the provided PTX was compiled with an unsupported toolchain | 222 | PTX newer than the driver's JIT | ship SASS, or upgrade the driver |
| system not yet initialized | 802 | NVSwitch system without fabric manager | start `nvidia-fabricmanager` (same version as the driver) |
| system has unsupported display driver / cuda driver combination | 803 | user-mode libcuda does not match the kernel module | remove libcuda from the image; fix the compat branch |
| forward compatibility was attempted on non supported HW | 804 | `cuda-compat` on GeForce | upgrade the driver instead |
| Failed to initialize NVML: Driver/library version mismatch | (NVML) | new user-space NVML, old kernel module still loaded | reboot or reload the module after a driver upgrade |

(`gpusim.compat.ERRORS`; notebook 04 drills 35, 100, 209, 222, 803 and 804. `check()` does not model 36, 802 or NVML.)

---

## 2. The execution model

### 2.1 Grid, block, warp, SM

A kernel launch is a **grid** of **blocks** (CTAs) of up to 1,024 threads. The hardware schedules **warps** of
32 threads. Each block is assigned to one **SM** and stays there until it finishes. An SM is four
sub-partitions, each with its own warp scheduler, register-file slice and execution units. Every cycle a
scheduler issues one instruction from a *ready* warp. The rest of the story (tensor cores, TMA, clusters) is in
the [GPU primer §2 and §5](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md#2-the-execution-model).

### 2.2 SIMT and divergence

A warp has one instruction stream. When lanes disagree on a branch, the warp runs **every path that any lane
takes**, masking the others off. Cost per warp = the sum over the distinct paths taken (`gpusim.simt.divergence()`):

```
if (tid % 2) A(10 instr) else B(10 instr)      every warp issues 20: SIMT efficiency 50%
if ((tid / 32) % 2) ...                         each warp takes one path: 100%
per-thread loop of L[t] iterations              each warp runs max(L) iterations  (loop_divergence())
```

The loop case is the one that bites inference code. With one thread per sequence and long-tailed lengths,
SIMT efficiency was **24%** in notebook 01 and **96%** after sorting by length. This is why kernels over
ragged batches bucket by length, and why attention kernels split work into *tiles of tokens*, not threads per
sequence. Since Volta, each thread has its own program counter, so divergent lanes can make independent
progress. The cost model is unchanged.

### 2.3 Occupancy

**Occupancy** = resident warps / the SM's maximum. A block is resident only if all of its resources fit, so
the blocks per SM are the minimum over four limits (`gpusim.occupancy.occupancy()`, following NVIDIA's
`cuda_occupancy.h`):

```
by warps      max_warps // ceil(threads/32)
by blocks     max_blocks
by registers  4 × (16384 // roundup(regs × 32, 256)) // warps_per_block      (4 sub-partitions)
by smem       smem_per_SM // roundup(smem_per_block + 1 KB reserved, 128)     (CC >= 8.0)
```

| CC (example) | threads/SM | warps/SM | blocks/SM | registers/SM | smem/SM | smem/block |
|---|---|---|---|---|---|---|
| 7.5 (T4) | 1,024 | 32 | 16 | 65,536 | 64 KB | 64 KB |
| 8.0 (A100) | 2,048 | 64 | 32 | 65,536 | 164 KB | 163 KB |
| 8.9 (L4) | 1,536 | 48 | 24 | 65,536 | 100 KB | 99 KB |
| 9.0 (H100) | 2,048 | 64 | 32 | 65,536 | 228 KB | 227 KB |
| 10.0 (B200) | 2,048 | 64 | 32 | 65,536 | 228 KB | 227 KB (verify) |

Worked on an **L4**. A block of 256 threads using 64 registers per thread gives 2,048 registers per warp. The
16,384 in each sub-partition hold 8 warps, so 32 warps per SM, which is **4 blocks and 67% occupancy**,
limited by registers. Add 48 KB of shared memory per block and each block needs 49 KB, so 2 blocks fit in
100 KB: **33%**. The sub-partition split matters. At 80 registers, 16,384 // 2,560 = 6 warps per partition,
so 24 per SM, not 65,536 // 2,560 = 25.

### 2.4 Latency hiding, Little's law and waves

A warp waiting on memory is swapped out for a ready one. To sustain bandwidth B with latency L, **B × L bytes
must be in flight** (`gpusim.occupancy.bytes_in_flight()`). With an *assumed* loaded latency of 600 ns, an
H100 at 3.35 TB/s needs 2.0 MB in flight, which is **15 KB per SM**. Even at full occupancy (2,048 threads)
that is **1.86 outstanding 4-byte loads per thread**, or 0.46 with 16-byte vector loads
(`loads_in_flight_per_thread()`). An L4 needs 3.1 KB per SM, which about half occupancy covers. So fast kernels on
big GPUs hide latency with *vector loads, several independent loads per thread, and asynchronous copies*
(cp.async, TMA). GEMM and attention kernels run at 1 to 2 blocks per SM with large register tiles and still
saturate the machine ("better performance at lower occupancy", Volkov 2010). Treat occupancy as a means.

A grid runs in **waves** of `SMs × blocks_per_SM` blocks. 140 single-block-per-SM blocks on 132 SMs take two
waves, and the second is 6% full, so **53% efficiency** (`gpusim.occupancy.waves()`). Decode kernels with few
blocks hit this, which is why split-K and split-KV exist.

---

## 3. Memory access patterns

### 3.1 Coalescing into 32-byte sectors

Global memory is served in **32-byte sectors**; an L1/L2 line is 128 bytes = 4 sectors. On compute capability 6.0+ a
warp request costs one transaction per **distinct sector** its active lanes touch (`gpusim.simt.coalescing()`):

| One warp reads (float32) | Sectors | Bytes moved | Efficiency |
|---|---|---|---|
| `a[lane]` (aligned) | 4 | 128 | 100% |
| `a[lane + 1]` (misaligned by 4 B) | 5 | 160 | 80% |
| `a[2 * lane]` | 8 | 256 | 50% |
| `a[32 * lane]` (a column of a row-major matrix) | 32 | 1,024 | 12.5% |
| `a[0]` in every lane | 1 | 32 | one transaction (a broadcast) |
| `double` / `float4` per lane, contiguous | 8 / 16 | 256 / 512 | 100% |

Two consequences for inference. **Layout is performance**: struct-of-arrays beats array-of-structs, and
matrices are stored so the fastest-moving index is the one consecutive lanes walk. **Paged KV caches** stay
coalesced even though blocks are scattered: each block stores its tokens' keys and values contiguously
(kilobytes per block), so a warp still reads whole sectors. That is one reason blocks cannot be tiny
([PagedAttention primer](../../04-inference-engine/paged-attention/paged-attention-primer.md)).

### 3.2 Shared memory and bank conflicts

Shared memory is a software-managed on-chip scratchpad (up to 228 KB per SM on an H100) with **32 banks, each
4 bytes wide**. Word w lives in bank `w mod 32`. In one pass each bank serves one word. Lanes that want
*different* words in the same bank are serialized, and lanes that want the *same* word get a broadcast
(`gpusim.simt.bank_conflicts()`):

```
tile[lane]                   stride 1 word  → 32 banks, 1 pass
tile[2*lane]                 stride 2       → 2-way conflict
tile[s*lane]                 stride s       → gcd(s, 32)-way conflict
tile[lane][c] in float[32][32]              → 32-way conflict (every row starts in bank 0)
tile[lane][c] in float[32][33]              → conflict-free (row r starts in bank r)
8-byte / 16-byte accesses                   → served per half-warp / quarter-warp
```

### 3.3 Case study: the transpose

`out[j][i] = in[i][j]` with a warp across 32 consecutive `j`: the read takes 4 sectors per request, and the
write is a column walk at **32 sectors**, so writes move 8× the bytes they need. Staging a 32×32 tile in
shared memory makes both global phases row-wise. The column walk moves into shared memory, where a pitch of
32 floats makes it a 32-way bank conflict and a pitch of **33** makes it conflict-free. Notebook 01 derives
all three numbers; the lab's `01_kernels_in_the_simulator` runs the kernels.

### 3.4 Tiled GEMM traffic

For C[M,N] = A[M,K] · B[K,N], a block computing a BM × BN tile streams a BM × K panel of A and a K × BN panel
of B through shared memory. So each A element is loaded once per *column* of tiles and each B element once
per *row* (`gpusim.tiling.gemm_traffic()`, checked against the simulated kernel `tiled_matmul()`):

```
global elements = M·K·⌈N/BN⌉ + K·N·⌈M/BM⌉ + M·N          naive: BM = BN = 1      compulsory: each input once
```

| 4096³ GEMM, bf16 | Global traffic | FLOP/byte |
|---|---|---|
| naive (no reuse) | 274.9 GB | 0.5 |
| 32 × 32 tiles | 8.62 GB | 15.9 |
| 128 × 128 tiles | 2.18 GB | 63.0 |
| 128 × 256 tiles | 1.64 GB | 83.6 |
| compulsory | 0.10 GB | 1,365 |

Tiles cost shared memory: `stages × (BM·BK + BK·BN) × bytes` (`tile_smem_bytes()`). A 128×128×32 BF16 tile,
3 stages deep, is 48 KB, which means 2 blocks per SM on an L4 (§2.3). The last factor of about 20× to
compulsory traffic comes from **L2**, where concurrent blocks share panels. Layer 01 puts these numbers on the
roofline ([§4.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#4-the-memory-hierarchy-and-why-tilingfusion-win)).
Note the decode case: with M = 8 tokens, the traffic is the weight matrix read once, and no tile size helps.

### 3.5 Fusion

Every kernel boundary is a round trip through HBM. A row-wise softmax of a 4096 × 4096 BF16 score matrix
(`gpusim.tiling.softmax_traffic()`):

| Variant | Kernels | HBM traffic | At 3.35 TB/s |
|---|---|---|---|
| unfused: max, subtract, exp, sum, divide | 5 | 256 MiB (8RC + 4R elements) | 80 µs |
| fused, a row fits on chip | 1 | 64 MiB (2RC) | 20 µs |
| online (two passes, running max and sum) | 1 | 96 MiB (3RC) | 30 µs |

The online variant rescales its running sum by `exp(m_old − m_new)` when the max grows, which makes it exact
(`softmax_online()`). FlashAttention fuses it *into* QKᵀ and PV so the score matrix never reaches HBM
([FlashAttention primer §4–5](../../04-inference-engine/flash-attention/flash-attention-primer.md)). Elementwise
chains fuse the same way: k ops unfused move k times the bytes of one fused kernel (`elementwise_traffic()`).

### 3.6 L2: the shared last level

One L2 cache serves all SMs: about 4 MB on a T4, 40 MB on an A100, 48 MB on an L4, 50 MB on an H100 (verify). It is
where §3.4's tiled GEMM recovers its last ~20×: blocks running at once on neighbouring tiles read the same A and B
panels, so later readers hit in L2, which is why GEMM libraries launch tiles in a *swizzled* order (nearby tiles
together). Decode shows the limit: a 17.5 GB weight shard is 350× an H100's L2, so each step streams its weights
from HBM whatever the policy. Small hot data can be pinned on CC 8.0+ with a stream's access policy window
(`cudaAccessPolicyWindow`). `gpusim` counts HBM bytes only, so it bounds what L2 can save rather than simulating it.

---

## 4. Streams, launch overhead and CUDA Graphs

### 4.1 Asynchronous launches and streams

A kernel launch **enqueues** work and returns. A **stream** is an in-order queue, and different streams may
overlap: copies with compute, NCCL communication with compute, prefetch with decode. The CPU runs ahead of
the GPU until something forces it to wait. **Hidden synchronization points** are the usual culprit:
`tensor.item()`, `.cpu()`, printing a GPU tensor, `torch.cuda.synchronize()`, and some allocations.

### 4.2 When the CPU is the bottleneck

Each launch costs CPU time: microseconds in the driver, more in a framework's eager dispatch. If a kernel is
shorter than a launch, the GPU waits (`gpusim.tiling.step_time()`):

```
eager:  kernel i is enqueued at (i+1)·launch; it starts when enqueued AND the previous one is done
graph:  one launch for the whole captured sequence, then kernels back to back
assumed: an eager launch costs 5 µs, launching a whole graph 10 µs
384 kernels × 2 µs:                               eager 1,922 µs (GPU idle 60%)   graph 778 µs   → 2.5×
384 kernels × 20 µs:                              eager 7,685 µs   graph 7,690 µs  → no gain
```

A decode step at small batch is exactly the first case: hundreds of small GEMMs, norms, rotary embeddings,
attention and sampling kernels, many of them a few microseconds long.

### 4.3 CUDA Graphs, and why engines capture decode

A **CUDA Graph** records a sequence of kernels (and memcpys, and NCCL calls) once and replays it with one
launch. The price is rigidity. Shapes and memory addresses are frozen at capture, so inputs are copied into
static buffers. There must be no host synchronization inside the graph, and control flow is fixed. Engines
therefore **capture one graph per batch-size bucket** for decode and pad each batch up to the nearest
captured size. Prefill, with its variable shapes, runs eagerly or as *piecewise* graphs around attention.
Capturing costs startup time and GPU memory, which is why vLLM's `--enforce-eager` exists: it saves both at
the cost of slower decode. `torch.compile` attacks the same overhead from the other side: Inductor fuses
elementwise chains into Triton kernels, and `mode="reduce-overhead"` wraps the result in CUDA Graphs.
NCCL collectives can be captured too. Custom all-reduce kernels, which read their peers' buffers directly over
NVLink, register those buffer addresses when the graph is captured.

---

## 5. Collectives

### 5.1 Semantics

p ranks each hold a buffer. A collective is defined by what each holds afterwards
(`gpusim.collectives.reference()`):

| Collective | After | Used in inference by |
|---|---|---|
| broadcast | every rank has root's buffer | weights or config from rank 0 |
| reduce | root has the elementwise sum | rarely |
| **all-reduce** | every rank has the sum | tensor parallelism, 2 per layer |
| **reduce-scatter** | rank r has chunk r of the sum | sequence parallelism; the first half of all-reduce |
| **all-gather** | every rank has the concatenation | sharded weights (FSDP), logits, sequence parallelism |
| **all-to-all** | rank r's chunk j goes to rank j | MoE expert parallelism: dispatch and combine |
| send / recv | point-to-point | pipeline parallelism; KV transfer (layer 05) |

### 5.2 All-reduce = reduce-scatter + all-gather

Split each buffer into p chunks. In the **reduce-scatter**, at step s = 1, …, p−1 every rank r adds its partial of
chunk `(r−s) mod p` into its right neighbour; after p−1 steps rank r owns the finished chunk r. In the
**all-gather**, finished chunks travel p−1 more hops. The trace below is `all_reduce(bufs, "ring").trace.table()`
for 4 ranks, where `c2+` means "chunk 2, added in":

```
step  1 reduce-scatter   0->1:c3+  1->2:c0+  2->3:c1+  3->0:c2+
step  2 reduce-scatter   0->1:c2+  1->2:c3+  2->3:c0+  3->0:c1+
step  3 reduce-scatter   0->1:c1+  1->2:c2+  2->3:c3+  3->0:c0+     rank r now owns the sum of chunk r
step  4 all-gather       0->1:c0   1->2:c1   2->3:c2   3->0:c3
step  5 all-gather       0->1:c3   1->2:c0   2->3:c1   3->0:c2
step  6 all-gather       0->1:c2   1->2:c3   2->3:c0   3->0:c1      everyone has every sum
```

Each rank sends 2(p−1) messages of S/p, so **2(p−1)/p · S** bytes in total. No point-to-point algorithm sends
less, which is why the ring is bandwidth-optimal. Every chunk is reduced exactly once and then copied, so
all ranks end with bitwise-identical results. A different algorithm adds in a different order, so ring and
tree results differ in the last bits (notebook 03 shows it).

### 5.3 The α-β cost of each algorithm

Model each step as costing **α** (launch, synchronization and hop latency) plus the bytes through the busiest
port divided by **B**, the per-direction link bandwidth. Layer 01 introduces the model and writes B as β
([§5.2](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#5-fabrics-quantitatively)). Every
algorithm in `gpusim.collectives` really moves numpy data, and its traced time equals the closed form
`T = a·α + c·S/B` (`cost_terms()`, `model_time()`):

| Algorithm | Steps a | Bandwidth factor c | Needs |
|---|---|---|---|
| ring all-reduce | 2(p−1) | 2(p−1)/p | a ring (any topology) |
| binomial tree all-reduce (reduce, then broadcast) | 2⌈log₂p⌉ | 2⌈log₂p⌉ | nothing; tiny messages only |
| NCCL double binary tree (pipelined; a model, not simulated) | ≈ 2⌈log₂p⌉ + pipeline fill | ≈ 2 | NCCL's tree algorithm |
| one-shot (every rank pulls all buffers) | 1 | p−1 | all-to-all links (NVSwitch) |
| two-shot (direct reduce-scatter + all-gather) | 2 | 2(p−1)/p | all-to-all links |
| in-switch reduction (NVLS, SHARP), k chunks | k+1 | (k+1)/k → 1 | a reducing switch |
| ring reduce-scatter or all-gather | p−1 | (p−1)/p | |
| pairwise all-to-all / direct all-to-all | p−1 / 1 | (p−1)/p | |
| pipelined chain broadcast, k chunks | p+k−2 | (p+k−2)/k → 1 | |

Pipelining trades α for β. Chunking a message into k pieces adds k steps but shrinks each one, and the optimum
is `k* = √(S/(α·B))`: 12 chunks for 128 MiB with the numbers below (`optimal_chunks()`). `best_algorithm()`, `sweep()`
and `tp_comm()` pipeline at k* unless told otherwise (`pipeline_chunks()`). In-switch reduction nearly halves the
ring's bytes (S instead of 2(p−1)/p · S per GPU) because each GPU sends its data **once** and receives the result
once.

### 5.4 algbw and busbw

nccl-tests reports two bandwidths ([PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)).
**algbw = S / t**, with S the full buffer (the gathered output for all-gather, the input for reduce-scatter).
**busbw = algbw × factor**, where the factor is the bandwidth term of the optimal point-to-point algorithm, so
that busbw reads the **per-link** bandwidth whatever p is (`gpusim.collectives.busbw()`):

| all-reduce | reduce-scatter, all-gather, all-to-all | broadcast, reduce, send/recv |
|---|---|---|
| 2(p−1)/p | (p−1)/p | 1 |

Worked with **8 H100s** and layer 01's illustrative NVLink 4 numbers, **α = 2 µs per step and B = 450 GB/s**.
A 1 GiB ring all-reduce takes **4.20 ms**: algbw 255 GB/s, **busbw 447 GB/s**. busbw is the number to hold
against the link's 450. Two warnings. With **NVLS** (1 GiB in k* = 35 chunks), modelled busbw reaches 744 GB/s
(ceiling 2(p−1)/p · B = 788), *above* the link rate, because the switch does the reduction and the factor assumes it
did not. That is expected, not an error; what real NVLS reaches is for nccl-tests to say.
And busbw well below the link rate inside one node usually means NCCL is not using NVLink (§5.8).

### 5.5 Latency-bound or bandwidth-bound

The latency and bandwidth terms are equal at **S* = a·α·B / c** (`crossover_bytes()`). For the ring,
**S* = p·α·B = 7.2 MB** on the 8 H100s above, which is about 440 tokens of an 8,192-wide BF16 activation.
The ring sweep, simulated in nccl-tests' shape (`sweep()`, `format_sweep()`):

```
#  size (B)   time (us)   algbw (GB/s)   busbw (GB/s)   [simulated, alpha-beta model]
      65536        28.3           2.32           4.06
    1048576        32.1          32.69          57.20
   16777216        93.2         179.93         314.87
  134217728       550.0         244.05         427.09
 1073741824      4203.7         255.43         447.00
```

Below S* a ring all-reduce costs about 2(p−1)·α whatever the message size, so the winning algorithm is the one
with the fewest steps. At 512 KiB on 8 GPUs the model gives two-shot 6.0 µs, in-switch 6.3 µs (k* = 1), one-shot
10.2 µs, binomial tree 19.0 µs and ring 30.0 µs. At 128 MiB, in-switch reduction wins (349 µs at k* = 12), then
two-shot and ring (526 and 550 µs), while one-shot is 2.1 ms (`best_algorithm()`). This is, in miniature, what
NCCL's tuner decides per call.

### 5.6 How inference uses collectives

**Tensor parallelism** (Megatron-style) splits each attention and MLP layer column-wise then row-wise, and
all-reduces a `tokens × hidden × 2 B` activation **twice per layer**. For a 70B-class model (hidden 8,192,
80 layers) at TP=8 that is 160 all-reduces per step (`gpusim.collectives.tp_comm()`; layer 01's
[§5.3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#5-fabrics-quantitatively) prices batch 1):

| Step | Message | Ring | Two-shot | Regime |
|---|---|---|---|---|
| decode, batch 32 | 512 KiB | 4.81 ms/step (93% α) | 0.97 ms/step | latency-bound |
| prefill, 8,192 tokens | 128 MiB | 88.0 ms/step (5% α) | 84.2 ms/step | bandwidth-bound |

Each GPU streams its 17.5 GB weight shard in about 5.2 ms per decode step, so a ring would add about 90% to it and a
two-shot about 19%. That is why engines ship **custom all-reduce kernels**: vLLM's and TensorRT-LLM's one-shot and
two-shot kernels over NVLink P2P buffers, NVLS where the switch supports it, and all-reduce fused with the following
RMSNorm. They also capture all of it in the decode CUDA graph. *Sequence parallelism* replaces each all-reduce with
a reduce-scatter and an all-gather (the same bytes) so that norms run on 1/p of the tokens.

**Expert parallelism** moves tokens, not partial sums: an all-to-all **dispatch** to each token's top-k
experts and an all-to-all **combine** back, each `tokens × top_k × hidden × bytes` per GPU. For a Mixtral-like
layer (hidden 4,096, top-2) with 256 tokens per GPU that is 4 MiB per direction: 22 µs pairwise or 10 µs
direct in the model (`model_time("all_to_all", ...)`). Specialised kernels (DeepEP) have low-latency modes for decode and high-throughput modes
for prefill; the MoE side of the exchange — placement, the slowest rank, TP vs EP and wide-EP — is
[MoE primer §6](../../00-foundations/mixture-of-experts/PRIMER.md#6-running-moe-on-gpus). **Pipeline parallelism** sends one activation per stage boundary (send/recv), which is small
enough to cross the scale-out network. **Data parallelism** (replicas) needs no collectives at inference. The
rule from the [deployment primer §4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu)
follows: TP and EP inside the NVLink domain, PP and DP across it.

### 5.7 NCCL in practice

- **Init.** One rank creates a unique id, all ranks call `ncclCommInitRank`, and NCCL **detects the
  topology** (GPUs, NVLink, PCIe switches, NICs, NUMA) from NVML and sysfs, searches for the best rings and
  trees, and builds **channels**. Each channel is one ring or tree instance run by one CTA. More channels
  mean more bandwidth and more SMs taken from compute.
- **Transports.** P2P over NVLink or PCIe (CUDA IPC), SHM through host memory when P2P is impossible, NET
  over InfiniBand/RoCE verbs, sockets or a vendor plugin (for example GCP's gIB, AWS's aws-ofi-nccl), CollNet
  (SHARP in the switch), NVLS (NVLink SHARP on NVSwitch systems, NCCL 2.17+).
- **Protocols.** *LL* stores 8 bytes (4 data + 4 flag) for the lowest latency at 50% efficiency. *LL128*
  moves 128-byte lines carrying 120 bytes of data (~94%) where the hardware guarantees ordering. *Simple*
  uses large chunks with fences: full bandwidth, higher latency. The tuner picks algorithm × protocol ×
  channels per call from a latency-and-bandwidth model like §5.3.
- **Algorithms** you may see named: Ring, Tree, CollnetDirect, CollnetChain, NVLS, NVLSTree, and PAT (NCCL
  2.23+, for all-gather and reduce-scatter at scale). The current release is 2.32 (September 2026).

| Variable | Use |
|---|---|
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET` | print topology, rings, transports ("via P2P/IPC", "via NET/IB/0/GDRDMA") |
| `NCCL_TOPO_DUMP_FILE`, `NCCL_GRAPH_DUMP_FILE` | dump the detected topology and the chosen graphs |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | pin the bootstrap interface and the NICs |
| `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_NVLS_ENABLE` | switch transports off or on, to bisect a problem |
| `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | force choices for experiments; do not ship them untested |
| `NCCL_NET_GDR_LEVEL`, `NCCL_NET_PLUGIN` | GPUDirect RDMA distance and the network plugin |

### 5.8 Debugging hangs and slowness

NCCL matches collectives **by issue order on a communicator** and checks neither names nor sizes. The k-th
call on every rank is the same collective. So one rank that takes a data-dependent branch, crashes, goes OOM,
or issues calls in a different order leaves the others blocked. A size mismatch can hang or silently corrupt.
`gpusim.collectives.first_mismatch()` finds the first call where per-rank logs disagree, which is what you do
with PyTorch's flight recorder dumps or `NCCL_DEBUG=INFO` logs. The checklist:

1. **Make hangs into errors.** Set timeouts and async error handling (`TORCH_NCCL_ASYNC_ERROR_HANDLING`), and
   use the flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`, verify names), NCCL's RAS client (`ncclras`,
   NCCL 2.24+) to see every rank's state, and `py-spy dump` for Python stacks.
2. **Check control flow is rank-uniform** and every rank reached the same call count.
3. **Check the path.** `NCCL_DEBUG=INFO` should say P2P or NVLS within a node. SHM inside one node means P2P
   is off: ACS or IOMMU on PCIe, containers without a shared IPC namespace, or `NCCL_P2P_DISABLE`. Compare
   `nvidia-smi topo -m`.
4. **Check the environment.** Wrong `NCCL_SOCKET_IFNAME`, firewalls between nodes, a missing network plugin,
   a small `/dev/shm` in containers (`--ipc=host` or `--shm-size`), mismatched NCCL versions, and no fabric
   manager on NVSwitch systems.
5. **Then measure.** Run nccl-tests on the same nodes and compare busbw with the link rate (the lab's notebook 04).

---

## 6. How a container gets a GPU

### 6.1 What a CUDA process needs

Containers share the host kernel, so the **kernel-mode driver is always the host's**. A CUDA process inside a
container needs four things: the **device nodes** (`/dev/nvidia0`, one per GPU, plus `/dev/nvidiactl`,
`/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools`, and `/dev/nvidia-caps/*` for MIG) allowed by its cgroup; the
**user-mode driver libraries** (`libcuda.so`, `libnvidia-ml.so`, `libnvidia-ptxjitcompiler.so`, ...) at *exactly*
the host driver's version; the host utilities it wants (`nvidia-smi`); and the **CUDA userland** (runtime and
libraries). Only the last belongs in the image.

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

Which GPUs and which parts of the driver get injected is controlled by `NVIDIA_VISIBLE_DEVICES` (`all`, indices,
UUIDs, `none`, `void`) and `NVIDIA_DRIVER_CAPABILITIES` (`compute,utility` in CUDA base images). Inside the
container, `CUDA_VISIBLE_DEVICES` further masks and renumbers what the process sees. `gpusim.compat.origin()`
classifies any path as host-injected or image.

### 6.3 In Kubernetes

The device plugin advertises `nvidia.com/gpu` as an extended resource. At pod admission its `Allocate` call
returns the device IDs, environment variables, mounts or CDI device names for the chosen GPUs, and the container
runtime injects them as above (layer 03's primer, [`03-kubernetes-gpu/gpu-scheduling/PRIMER.md`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md) §1 *What Kubernetes
sees*). On GKE, Google manages the device plugin and installs the driver on the node (§9). With the NVIDIA GPU
Operator, the operator installs the driver, toolkit, plugin and DCGM exporter as pods.

### 6.4 Failure modes (`gpusim.compat.explain_container()`)

| Story | Result |
|---|---|
| `docker run` without `--gpus` or a CDI device; `NVIDIA_VISIBLE_DEVICES` unset or `void`; no GPU in the pod spec | error 100, no device |
| the Dockerfile copies `libcuda.so` from a build machine into the image | error 803: user-mode and kernel-mode driver disagree |
| image CUDA newer major than the host driver | error 35; `cuda-compat` in the image fixes it on data-center GPUs over a supported driver branch (804 on GeForce, 803 on other branches) |
| a wheel without SASS for the GPU and no usable PTX | error 209 |
| `nvidia-smi` works but `torch.cuda.is_available()` is False | CPU-only wheel, or `CUDA_VISIBLE_DEVICES` masks every GPU |
| NCCL hangs or falls back to slow paths in containers | `/dev/shm` too small; no shared IPC namespace for CUDA IPC |

---

## 7. Sharing a GPU

### 7.1 First, ask whether to share

A 70B model on 8 GPUs does not share them: its engine batches hundreds of requests, and **continuous batching is
the most efficient sharing there is**, because it shares the weight reads. GPU-level sharing is for many *small*
things: small models, dev notebooks, low-QPS endpoints, CI.

### 7.2 MIG: partitions

Multi-Instance GPU (A100, A30, H100, H200, B200 class; not L4, T4 or RTX 40) splits one GPU into up to **seven
instances**, each with its own SMs, L2 slice, memory controllers and memory: hardware isolation of performance,
memory and faults. The geometry is rigid. There are 7 compute slices and 8 memory slices; a profile takes a fixed
number of each and may only *start* at listed slice indices (`nvidia-smi mig -lgipp`; `gpusim.sharing.MIG`, verify):

| H100 80 GB profile | Compute | Memory slices | Allowed starts | Max |
|---|---|---|---|---|
| 1g.10gb | 1/7 | 1 | 0–6 | 7 |
| 1g.20gb | 1/7 | 2 | 0, 2, 4, 6 | 4 |
| 2g.20gb | 2/7 | 2 | 0, 2, 4 | 3 |
| 3g.40gb | 3/7 | 4 | 0, 4 | 2 |
| 4g.40gb | 4/7 | 4 | 0 | 1 |
| 7g.80gb | 7/7 | 8 | 0 | 1 |

A100 80 GB has the same shapes, A100 40 GB halves the memory (1g.5gb ... 7g.40gb), and H200 141 GB scales it
(1g.18gb ... 7g.141gb). `gpusim.sharing.pack()` searches for a layout:

```
4g.40gb + 2g.20gb + 1g.10gb           | 4g  |  -  |  -  |  -  | 2g  |  -  | 1g  |  .  |   fits
3g.40gb + 2g.20gb + 1g.10gb × 2       | 2g  |  -  | 1g  | 1g  | 3g  |  -  |  -  |  -  |   fits
3g.40gb + 3g.40gb + 1g.10gb           does not fit: 3 + 3 + 1 = 7 compute slices, but the two 3g take all 8 memory slices
1g × 3, then 4g, created in arrival order at the first free slice   the 4g no longer fits (first_fit())
```

So plan a layout per node pool and create instances with explicit placements. Kubernetes exposes MIG either
as plain `nvidia.com/gpu` with one profile per node ("single" strategy) or as `nvidia.com/mig-1g.10gb`-style
resources ("mixed"). GKE sets one partition size per node pool (§9); layer 03 covers sharing at cluster level
([`03-kubernetes-gpu/gpu-scheduling/PRIMER.md`](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md) §9 *Sharing GPUs at the cluster level*). A 7g instance is not the
whole GPU: in MIG mode each slice gets a fixed number of SMs, so an A100's 7g has 98 of its 108 SMs (verify).

### 7.3 MPS: overlap

The Multi-Process Service runs a server that merges the kernels of many client processes onto the GPU at the same
time. Small kernels from different processes then fill the SMs together instead of taking turns. Since Volta, each
client has its own address space, and a client can be capped with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and
`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`. Fault isolation is weaker: a fatal fault in one client can take down the others
on that GPU (verify for your driver). MPS suits cooperative workloads of one team.

### 7.4 Time-slicing: turns

The device plugin (or GKE's `max_shared_clients_per_gpu`) advertises one GPU as N schedulable replicas. The GPU
runs one context at a time, round-robin. There is **no memory isolation**, so one tenant can OOM the others, and
no performance isolation. With N busy tenants, a request needing W of GPU time in quanta q with switch cost s
finishes after `W + (⌈W/q⌉ − 1)·((N−1)(q+s) + s)` at best, when it arrives just as its turn starts
(`gpusim.sharing.timeslice_latency()`): 10 ms of work with 4 busy tenants, 2 ms quanta and 50 µs switches takes
**34.8 ms at best**. Arriving just after its turn adds one more round of the others' turns, 6.2 ms, so **41.0 ms
at worst** and **37.9 ms on average**. Idle tenants cost nothing, which is why time-slicing suits notebooks and
bursty dev work.

### 7.5 Choosing

An idealised latency model, counting SM capacity only (`gpusim.sharing.shared_latency()`): one request needs
10 ms alone, 4 always-busy tenants share, and `util` is the share of the GPU its kernels fill. The time-slicing
column is the mean over arrival times from §7.4 (34.8 ms at best, 41.0 ms at worst):

| Kernel size | Exclusive | Time-slicing | MPS | MIG (1g each) |
|---|---|---|---|---|
| small (util 0.2) | 10 ms | 37.9 ms | 10 ms | 14 ms |
| saturating (util 1.0) | 10 ms | 37.9 ms | 40 ms | 70 ms |

| Need | Choose |
|---|---|
| tenants who must not affect each other, SLOs, untrusted code | MIG |
| many small cooperative processes, throughput first | MPS |
| idle-heavy notebooks, CI, demos | time-slicing |
| one big model | none: whole GPUs, batching in the engine |

---

## 8. Health and observability

### 8.1 nvidia-smi and NVML

`nvidia-smi` (on NVML) is the first look: the banner, `-q` for everything, `--query-gpu=... --format=csv` for
scripts, `topo -m` for the link matrix (layer 01 §5.6), `-q -d PERFORMANCE` for clock-event reasons,
`-q -d ECC` and `-q -d ROW_REMAPPER` for memory health, and `mig -lgip` / `-lgipp` for MIG.

### 8.2 DCGM, and why "GPU util" misleads

**DCGM** (the Data Center GPU Manager) samples the same counters fleet-wide, and **dcgm-exporter** exposes them to
Prometheus. The field everyone graphs, `DCGM_FI_DEV_GPU_UTIL` (nvidia-smi's "GPU-Util"), is the **share of
time at least one kernel was running**. A kernel with 8 blocks on a 132-SM H100 that runs all the time reads
**100%** while **6%** of the SMs do anything (`gpusim.health.util_counters()`). Read these instead:

| Field (dcgm-exporter) | Meaning | Reading it |
|---|---|---|
| `DCGM_FI_PROF_SM_ACTIVE` | share of SM-cycles with at least one warp resident | the real "how much of the GPU" |
| `DCGM_FI_PROF_SM_OCCUPANCY` | resident warps / maximum | latency-hiding headroom (§2.3) |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | share of cycles the tensor pipes are busy | high in prefill and big batches |
| `DCGM_FI_PROF_DRAM_ACTIVE` | share of cycles the memory interface is busy | high in decode: memory-bound |
| `DCGM_FI_PROF_PCIE_*_BYTES`, `DCGM_FI_PROF_NVLINK_*_BYTES` | link traffic | loading, offload, collectives |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | framebuffer memory | near-full is *normal* for engines that pre-allocate KV; use the engine's KV-usage metric |
| `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_SM_CLOCK` | power, temperature, clocks | with the throttle reasons below |
| `DCGM_FI_DEV_XID_ERRORS` | last XID error | triage below |

**Check which fields your exporter exports.** Stock dcgm-exporter's default counters file
(`default-counters.csv`) has `DCGM_FI_PROF_SM_ACTIVE` and `DCGM_FI_PROF_SM_OCCUPANCY` commented out and leaves out
`DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (§8.3), so by default you get neither the first two rows above nor the throttle
bits. The lab ships a counters file that adds all three
([`deploy/any-gpu/dcgm-counters.csv`](cuda-nccl-lab/deploy/any-gpu/dcgm-counters.csv)); the per-exporter field lists
are `gpurt.dcgm.EXPORTED_BY`, and [`deploy/gke/README.md`](cuda-nccl-lab/deploy/gke/README.md) shows which alert rules
each exporter lets fire. Google's managed-Prometheus DCGM field list (its `nvidia-dcgm` example, which GKE's managed
package resembles) has SM active, occupancy and tensor active but no XID or row-remap fields, nor clock-event or DRAM
ones (verify). On GKE, the §8.3–8.4 signals therefore need a self-managed exporter with that counters file.

`gpusim.health.diagnose()` turns a sample into findings such as *"busy but mostly empty"* (util high, SM active
low: small batch, launch gaps, tiny grids) and *"memory-bandwidth-bound"*. Notebook 05 works through both.

### 8.3 Throttling

`DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (formerly `..._CLOCK_THROTTLE_REASONS`) is NVML's bitmask
(`gpusim.health.decode_throttle()`): `0x1` idle, `0x2` application clocks setting, `0x4` software power cap,
`0x8` hardware slowdown, `0x10` sync boost, `0x20` software thermal slowdown, `0x40` hardware thermal slowdown,
`0x80` hardware power brake, `0x100` display clock setting. A power cap under load is normal. Any `hw_*` reason
or thermal slowdown is a facilities ticket.

### 8.4 XIDs, ECC and row remapping

XIDs are the driver's error reports (in `dmesg` and DCGM). What matters is **who acts**
(`gpusim.health.triage_xid()`; the catalogue is NVIDIA's XID documentation, verify):

| XID | Meaning | Owner | Action |
|---|---|---|---|
| 13, 31, 43 | graphics engine exception, GPU memory page fault, GPU stopped processing | app | usually an out-of-bounds or illegal address in a kernel: fix and restart the app |
| 45 | preemptive cleanup | app | secondary: read the XID before it |
| 63 | row-remapping (or page-retirement) event | node | a remap is pending until the GPU resets: drain and reset |
| 94 | contained ECC error | node | only the affected app died: restart it, reset when drained |
| 48, 95 | double-bit ECC error, uncontained ECC error | hardware | drain and reset now; recurring means RMA |
| 64 | row-remapper failure | hardware | drain, RMA |
| 74, 79 | NVLink error, GPU fell off the bus | hardware | drain, reboot, diagnose (PCIe, power, thermals); recurring means RMA |
| 92, 119 | high single-bit ECC rate, GSP RPC timeout | node | schedule diagnostics; reset; update the driver |

Since A100, the GPU **remaps rows** of HBM that show memory errors instead of retiring pages. The remap takes
effect at the next GPU reset, and a remap *failure* means replacement. `gpusim.health.alerts()` maps all of this
to severities: page, ticket, notify.

### 8.5 Fleet practice

Run `dcgmi diag -r 1|2|3` (quick to long) before a node joins a pool and after hardware XIDs. Drain nodes
automatically on hardware XIDs and pending remaps. Alert on SM active and on the engine's own metrics, not on GPU
util. On NVSwitch systems (HGX H100/B200) keep `nvidia-fabricmanager` at the driver's version, or CUDA returns
error 802.

---

## 9. On GCP and elsewhere

The same concepts on each platform; prices and obtainability are in [`COMPUTE.md`](../../COMPUTE.md).

| Concept | T0 (`gpusim`, any laptop) | T1/T2 (any GPU box) | T3 (GCP) |
|---|---|---|---|
| kernels, coalescing, occupancy | `simt`, `occupancy`, `tiling` | lab `01`–`02`: Numba kernels, simulator then GPU | same kernels on an L4 node |
| collectives, busbw | `collectives` | lab `03`–`04`: collectives over OS pipes (a real multi-process ring, no torch) or gloo at T0, NCCL and nccl-tests on 2+ GPUs | 2-GPU nccl-tests Job on `g2-standard-24` (2 × L4) |
| compatibility, containers | `compat` | lab `05`: what the container sees | GKE-installed drivers, container images |
| sharing, health | `sharing`, `health` | MIG and DCGM on a rented A100/H100 VM | GKE MIG and time-sharing pools, DCGM metrics (lab `06`) |

**On GCP.** GKE installs the NVIDIA driver on GPU nodes for you. From control plane 1.32.2-gke.1297000 it
installs the default driver automatically, and `gpu_driver_version` takes `DEFAULT`, `LATEST` (COS only) or
`INSTALLATION_DISABLED` (for the GPU Operator path). GPU nodes are tainted `nvidia.com/gpu=present:NoSchedule`
(verify). Sharing is configured per node pool: MIG with `guest_accelerator.gpu_partition_size` (for example
`"1g.10gb"`), time-sharing or MPS with `gpu_sharing_config` (`gpu_sharing_strategy = "TIME_SHARING" | "MPS"`,
`max_shared_clients_per_gpu`). **Deep Learning VM** images come with drivers preinstalled. For multi-node NCCL,
A3 High (H100, `a3-highgpu-8g`) uses GPUDirect-TCPX (verify) and A3 Mega (H100) GPUDirect-TCPXO; A3 Ultra (H200),
A4 (B200), A4X (GB200 NVL72) and A4X Max (GB300 NVL72) use GPUDirect RDMA over ConnectX-7 NICs on a rail-aligned
network with Google's NCCL network plugin (gIB, verify names). DCGM metrics can flow into Cloud Monitoring through
GKE's managed collection (verify). The deploy assets in [`cuda-nccl-lab`](cuda-nccl-lab/) hold the Terraform (a zonal
GKE Standard cluster with an L4 Spot pool that scales from zero, an `l4x2` pool of `g2-standard-24` nodes (2 × L4,
also Spot and from zero) that hosts the 2-GPU nccl-tests Job, and optional time-sharing and MIG pools) and the Jobs.

**Elsewhere.** **Colab** gives one T4 (CC 7.5, so no bf16 tensor cores). **Kaggle** gives **2 × T4 over PCIe**
for free: a real 2-GPU NCCL box without NVLink, where `NCCL_DEBUG=INFO` shows P2P or SHM over PCIe and busbw
is bounded by PCIe. **RunPod and Vast** hand you a *container* on someone's host: you cannot change the
driver, so choose an image whose CUDA the host driver supports (§1.2), and MIG and DCGM profiling are usually
unavailable. **Lambda and GCP** hand you *VMs*: you own the driver, MIG, fabric manager and DCGM. A laptop runs
everything in `gpusim`, and the lab's Numba kernels run in `NUMBA_ENABLE_CUDASIM=1`. A local kind cluster with
fake GPU capacity (layer 03) exercises the scheduler, not CUDA: it has no device nodes to inject.

---

## In a design review

**The two-minute walkthrough.** "Each node pool pins one driver branch. Images carry only CUDA userland,
built at or below that driver's CUDA version, or within its major, having checked that every kernel ships SASS
for our GPUs (`torch.cuda.get_arch_list()`). The NVIDIA Container Toolkit injects the host's libcuda and devices,
so user-mode and kernel-mode driver always match. Our hot kernels are memory-bound: we count bytes and sectors,
tile and fuse, and at small batch the launch path dominates, so we capture decode as CUDA Graphs per batch
bucket. We run tensor parallelism inside the NVLink domain. Decode all-reduces are about 0.5 MB, far below the
~7 MB latency/bandwidth crossover, so we use a few-step all-reduce (two-shot or NVLS) inside the graph. We
validate the fabric with nccl-tests busbw against the NVLink rate before any model runs. Whole GPUs go to big
models; small multi-tenant endpoints get planned MIG layouts; dev notebooks get time-slicing. We alert on SM
active and engine metrics, not GPU util, and hardware XIDs drain nodes automatically."

**Drill questions**

1. *nvidia-smi on the nodes says CUDA 12.2, our image is CUDA 12.8. Do we ship?* Yes, if every kernel has SASS
   for our GPUs: minor-version compatibility (driver >= 525.60.13) runs it. PTX-only kernels fail with 222, and
   APIs newer than 12.2 fail with 36. Upgrading the driver is the clean fix. A CUDA 13 image would need a
   driver upgrade, or `cuda-compat` on data-center GPUs.
2. *TP=8 decode spends 40% of the step in all-reduce, yet NVLink is nearly idle. Why?* The messages are
   hundreds of KB, so a ring's 14 α-steps dominate: it is latency-bound, not bandwidth-bound. Use fewer steps
   (two-shot or one-shot custom all-reduce, NVLS), capture them in CUDA graphs, fuse them with the norm, or
   lower the TP degree for this model.
3. *nccl-tests on one 8×H100 node shows all-reduce busbw of 100 GB/s. What do you check?* That NCCL uses
   NVLink at all: `NCCL_DEBUG=INFO` should report P2P or NVLS, not SHM or NET. Then fabric manager status,
   `nvidia-smi topo -m`, `NCCL_P2P_DISABLE` left set, and containers without shared IPC. On NVLink 4, busbw
   should be a few hundred GB/s, near the link rate, not 100.
4. *The dashboard shows 100% GPU utilization but low tokens/s. Is the GPU saturated?* Not necessarily. GPU util
   is time-based. Check SM active and tensor active. Small-batch decode, launch gaps and tiny grids read 100%
   with a few percent of SMs busy. Batch more, capture CUDA graphs, fuse.
5. *Seven small customer endpoints on one H100: MIG or time-slicing?* MIG with seven `1g.10gb` instances gives each
   an isolated, predictable slice. Time-slicing shares memory without limits and multiplies latency by the number of
   busy tenants. If all seven are adapters of one base model, serve them from one engine with multi-LoRA instead.
6. *A node logged XID 79, another XID 13. Who acts?* XID 79 (the GPU fell off the bus) is hardware: drain,
   reboot, diagnose, RMA if it repeats. XID 13 is usually the application's kernel faulting: send it to the
   service owner. It concerns the node only if it repeats across unrelated apps.

---

## Glossary

| Term | Meaning |
|---|---|
| SM | streaming multiprocessor: 4 sub-partitions, each with a warp scheduler and a register-file slice |
| warp | 32 threads issued together; the unit of scheduling |
| SIMT, divergence | one instruction stream per warp; lanes on different branches serialize |
| occupancy | resident warps / maximum warps per SM |
| sector, line | 32-byte unit of global-memory traffic; 128-byte cache line = 4 sectors |
| coalescing | a warp's accesses combining into few sectors |
| bank conflict | lanes hitting different 4-byte words in the same one of 32 shared-memory banks |
| compute capability (CC) | a GPU's architecture version, X.Y; `sm_XY` for SASS |
| SASS, PTX, fatbin | machine code; virtual ISA JIT-compiled by the driver; a binary holding several of each |
| minor-version compatibility | a newer runtime on an older driver of the same CUDA major, with limits |
| forward compatibility | `cuda-compat`: a newer user-mode driver on an older kernel module (data-center GPUs) |
| stream | in-order queue of GPU work; different streams may overlap |
| CUDA Graph | a captured sequence of GPU work replayed with one launch |
| collective | communication operation defined over all ranks of a communicator |
| α, B | per-step latency; per-direction link bandwidth (layer 01 writes B as β) |
| algbw, busbw | nccl-tests' S/t, and S/t × a per-collective factor comparable with link bandwidth |
| channel | one NCCL ring or tree instance, run by one CTA |
| LL, LL128, Simple | NCCL protocols: lowest latency, near-full bandwidth, full bandwidth |
| NVLS, SHARP | in-network reduction in NVSwitch (NVLink SHARP) or InfiniBand switches |
| CDI | Container Device Interface: a spec naming the devices, mounts and hooks to inject |
| MIG, MPS, time-slicing | partition, overlap, turns: the three ways to share a GPU |
| DCGM | Data Center GPU Manager; dcgm-exporter publishes its fields to Prometheus |
| XID | a driver error code in the kernel log |
| row remapping | replacing faulty HBM rows (A100+), effective after a GPU reset |

---

## Sources

- NVIDIA, *CUDA C++ Programming Guide*: execution model, compute capabilities and their limits, binary and
  PTX compatibility. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*: coalescing, shared memory, occupancy. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA, *CUDA Compatibility* (minor-version and forward compatibility) and the *CUDA Toolkit Release Notes*
  (driver table). <https://docs.nvidia.com/deploy/cuda-compatibility/>. Copies used to check `gpusim.compat`: the
  driver table in meson's `mesonbuild/modules/cuda.py`, and `NVIDIA_REQUIRE_CUDA` in NVIDIA's `nvidia/cuda` images.
- NVIDIA, `cuda_occupancy.h` (ships with the CUDA Toolkit): the occupancy arithmetic `gpusim` follows.
- NVIDIA, *Nsight Compute* documentation, Memory Workload Analysis: sectors per request, bank conflicts.
- V. Volkov, *Better Performance at Lower Occupancy*, GTC 2010.
- NVIDIA, *NCCL User Guide*, including environment variables. <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>
- NVIDIA, nccl-tests `doc/PERFORMANCE.md`: algbw and busbw. <https://github.com/NVIDIA/nccl-tests>
- S. Jeaugey, *Massively Scale Your Deep Learning Training with NCCL 2.4*, NVIDIA Technical Blog, 2019 (double binary trees).
- R. Thakur, R. Rabenseifner, W. Gropp, *Optimization of Collective Communication Operations in MPICH*,
  IJHPCA 2005 (α-β costs of ring, tree, recursive halving and doubling).
- P. Patarasuk, X. Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations*, JPDC 2009.
- M. Shoeybi et al., *Megatron-LM*, 2019 (tensor parallelism, two all-reduces per layer); V. Korthikanti et al.,
  *Reducing Activation Recomputation in Large Transformer Models*, 2022 (sequence parallelism).
- vLLM source, `csrc/custom_all_reduce.cuh` (one- and two-stage all-reduce); DeepSeek-AI, DeepEP
  (expert-parallel all-to-all). <https://github.com/vllm-project/vllm>, <https://github.com/deepseek-ai/DeepEP>
- NVIDIA Container Toolkit documentation; Container Device Interface specification.
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/>, <https://github.com/cncf-tags/container-device-interface>
- NVIDIA, *Multi-Instance GPU User Guide*; *Multi-Process Service*; GPU Operator, *Time-Slicing GPUs in Kubernetes*.
  <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/>, <https://docs.nvidia.com/deploy/mps/>
- NVIDIA, *DCGM* documentation (field identifiers) and dcgm-exporter; NVML API reference (clock event reasons);
  *XID Errors*. <https://github.com/NVIDIA/dcgm-exporter>, <https://docs.nvidia.com/deploy/xid-errors/>
- Google Cloud, GKE documentation: GPUs in Standard node pools, GPU sharing strategies (MIG, time-sharing, MPS),
  GPUDirect-TCPX/TCPXO and RDMA networking. <https://cloud.google.com/kubernetes-engine/docs>
- PyTorch documentation: CUDA semantics (streams, CUDA Graphs), `torch.compile`, distributed debugging.

---

## Verify list

Dated 2026-09-26. Everything below moves; check it before you rely on it.

| Fact used here | Where in `gpusim` | Status |
|---|---|---|
| Minimum drivers for CUDA 9.0–13.3 (release notes; 13.2 = 595.45.04 and 13.3 = 610.43.02 read from meson's copy of the table) | `compat.CUDA_MIN_DRIVER` | checked against copies of the table, 2026-09-26; verify |
| CUDA 13.4 maps to driver branch R615 (exact minimum unknown) | `compat.INFERRED` | **inferred** from `nvidia-ml-py` 13.615.71 and CUDA runtime wheels 13.4 on PyPI; verify |
| Minor-compatibility floors 11.x 450.80.02, 12.x 525.60.13, 13.x 580.65.06 | `compat.MINOR_COMPAT_FLOOR` | release notes give 11.x >= 450.80.02, 12.x >= 525, 13.x >= 580; verify |
| Kernel-driver branches each `cuda-compat` package runs over (13.0: R535, R550, R565, R570, R575) | `compat.COMPAT_BRANCHES` | read from `NVIDIA_REQUIRE_CUDA` in nvidia/cuda 12.4, 12.8, 13.0, 13.1 images; verify in the CUDA Compatibility guide |
| CUDA 13 removed Maxwell, Pascal and Volta targets (Turing kept); a GPU's minimum driver approximated by the first toolkit that targets its CC | `compat.FIRST_CUDA` | verify; the approximation per GPU too |
| CC of B300/GB300 (10.3), RTX 50 / RTX PRO 6000 (12.0); first toolkits 12.8 / 12.9; 'f' family targets | `compat.GPUS`, `compat.FIRST_CUDA` | verify |
| Forward compatibility on RTX PRO 6000 (server edition) | `compat.GPUS` (data-center flag) | verify |
| Per-SM limits for CC 10.0 | `occupancy.SMS` | verify |
| MIG profiles and placements for A100, H100, H200; A30 and B200 profiles differ | `sharing.MIG` | verify with `nvidia-smi mig -lgipp` |
| SMs available to a 7g MIG instance (A100: 98 of 108) | primer §7.2 | verify |
| MPS fault-isolation behaviour on current drivers | `sharing.TRAITS` | verify |
| Latest NCCL 2.32.3 (PyPI `nvidia-nccl-cu12/cu13`, 2026-09-22); NVLS since 2.17, PAT since 2.23, RAS since 2.24 | primer §5.7 | version checked on PyPI; feature versions verify |
| NCCL and PyTorch environment-variable names (`TORCH_NCCL_*`) | primer §5.7–5.8 | verify for your versions |
| DCGM field names; `CLOCK_THROTTLE_REASONS` renamed `CLOCKS_EVENT_REASONS` | `health` | verify |
| XID meanings and recommended actions | `health.XIDS` | verify |
| Docker/containerd/CRI-O native CDI support versions; Docker's `--gpus` adding the prestart hook itself | primer §6.2 | verify |
| L2 sizes: T4 4 MB, A100 40 MB, L4 48 MB, H100 50 MB | primer §3.6 | verify |
| vLLM flags (`--enforce-eager`, CUDA-graph capture sizes) | primer §4.3 | verify |
| GKE driver auto-install from 1.32.2-gke.1297000; `gpu_driver_version` values | primer §9 | from project FACTS (2026-09-26) |
| GKE GPU taint `nvidia.com/gpu=present:NoSchedule`; gIB plugin name; DCGM in Cloud Monitoring; A3 High uses GPUDirect-TCPX | primer §9 | verify |
| A3 Mega uses GPUDirect-TCPXO; A3 Ultra, A4, A4X and A4X Max use GPUDirect RDMA on ConnectX-7 | primer §9 | from project FACTS |
| Kaggle 2 × T4 (PCIe), 30 GPU-hours per week | primer §9 | from project FACTS; verify quotas |
| Illustrative α = 2 µs and B = 450 GB/s (NVLink 4), loaded DRAM latency 600 ns, eager launch 5 µs, graph launch 10 µs; NVLS busbw 744 GB/s (model) | primer §2.4, §4.2, §5 | assumptions and model output, not measurements: measure your own in the lab |
