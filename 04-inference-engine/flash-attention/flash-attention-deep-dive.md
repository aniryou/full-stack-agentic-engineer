# FlashAttention in depth: from the roofline to serving kernels

**Scope.** This is the second FlashAttention document in this folder. [The primer](flash-attention-primer.md) builds the idea from zero. It explains what attention computes, why the naive schedule is slow, online softmax and the lineage. This page assumes that you read the primer, or that you can explain those ideas yourself. It goes to the depth that is necessary to defend a kernel choice or a backend choice in a design review:

- exact byte counts
- the algebra with numerical safety
- what each FlashAttention generation changed, and why the hardware forced the change
- why decode needs a different kernel
- how paged KV gets to the kernel in vLLM and FlashInfer
- what each attention variant costs
- numerics
- how to make measurements that you can trust
- how to write the forward pass yourself in Triton.

**Tier.** Everything on this page is learnable on a CPU (T0). The functions in [`fa_calculators.py`](fa_calculators.py) calculate the derived numbers on this page, and [`test_fa_calculators.py`](test_fa_calculators.py) pins them. The companion notebook [`flash_attention_deep_dive.ipynb`](notebooks/flash_attention_deep_dive.ipynb) prints these numbers. It also runs every algorithm again against a naive reference in numpy, and it has exercises. The page calculates some one-line products inline: products of numbers that are already on the page, for example a time from bytes and a bandwidth.

Two checks on a CPU examine the Triton kernel of section 11. They examine its correctness, not its speed:

- One check compares its logic against a numpy substitute for Triton ([`test_triton_kernel_emulated.py`](test_triton_kernel_emulated.py)). This check runs here.
- The other check runs the real Triton front end through its interpreter (section 11.4). It needs `torch` and `triton`.

To measure the time of a real kernel (sections 10 and 11), you need T1. T1 is one small GPU, for example a free Colab or Kaggle T4 or any rented 24 GB card. [`COMPUTE.md`](../../COMPUTE.md) tells you where to get one. If you use a T4, know that FlashAttention-2 does not support Turing (section 7.4).

**Sources.** This page uses the upstream code as its source, downloaded on 2026-09-26 (the list is under Sources):

- the FlashAttention repo (FA2 CUDA kernels, FA3 Hopper kernels, FA4 CuTe-DSL kernels, the Triton version)
- vLLM's attention backends
- FlashInfer.

It was not possible to get the papers from this environment (this environment blocks arXiv). Thus, the figures that come from the papers have the tag **(verify)**. This page calculates every other performance number from stated assumptions.

**Notation.** Per head, $N$ is the sequence length ($N_q$ and $N_k$ when they are different) and $d$ is the head dimension. $b$ is the bytes per element (2 for bf16/fp16, 1 for FP8), and $\tau = 1/\sqrt{d}$ is the softmax scale.

For tile sizes, $B_r$ is the rows of Q per thread block (the kernels call it `kBlockM`). $B_c$ is the rows of K/V per tile (`kBlockN`). $H$ is the number of query heads, $H_{kv}$ the number of key/value heads, and $g = H / H_{kv}$ the group size. "CTA" is a thread block, SMEM is shared memory and HBM is device memory.

---

## The one-minute version

- Naive attention is memory-bound on every current GPU, and a larger problem cannot repair this. Its arithmetic intensity goes toward ${d/b}$ (64 FLOP/byte for $d$ = 128 in bf16). This is far below the ridge of an H100 (≈ 295) or an L4 (≈ 403). The solution is a schedule, not an approximation.
- The schedule depends on one algebraic fact. You can keep a partial softmax-weighted sum as a triple ${(m, l, o)}$, and any two triples merge exactly. That one operator gives you tiling (FA1/FA2), split-KV decode (Flash-Decoding), cascade attention over shared prefixes, and ring attention across GPUs.
- FA1 tiled the computation. FA2 changed the order of the loops, so each CTA owns a Q block. This keeps the output in registers, adds parallelism over the sequence, and removes inter-warp traffic. FA3 made the kernel asynchronous on Hopper (TMA, WGMMA, producer/consumer warpgroups, ping-pong), because the exponential unit was then a bottleneck. FA4 goes further on Blackwell, where the exponential unit now takes as long as the matrix multiplies.
- Decode is a different problem. One query row against a long KV cache has no reuse. Thus it is fully bandwidth-bound. The kernel must divide the KV sequence across CTAs. It must also put the query heads of a GQA group together in one tile. Decode attention time increases with batch × context × KV bytes per token.
- A server adds paging (block tables into the kernel), ragged batches, and a scheduler that selects a backend for each GPU generation. In vLLM on CUDA, that backend is FlashAttention (FA2 on Ampere, Ada and SM 12.x Blackwell, FA3 on Hopper), with two exceptions. If the GPU is SM 10.x datacenter Blackwell (B200, GB200), FlashInfer goes first for causal attention. If the KV cache is FP8 on a GPU without FA3 or FA4, vLLM uses FlashInfer instead.

---

## 1. Standard attention on the roofline

[The roofline primer (01)](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) explains the roofline itself (arithmetic intensity, ridge point, attainable FLOP/s). [The GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) explains the memory hierarchy that gives tiling its benefit. This section applies them.

### 1.1 FLOPs

Per head, $S = \tau \cdot QK^{\top}$ is an $N \times d$ by $d \times N$ product. That is $N^2 \cdot d$ multiply-adds, or $2N^2 d$ FLOPs. ${O = PV}$ is another $2N^2 d$. Softmax adds approximately 5 operations per score (max, subtract, exponentiate, sum, divide). FlashAttention's own benchmarks, the Triton tutorial and this page use one convention, which counts only the two matrix products:

$$
\mathrm{FLOPs}(\text{forward, one head}) = 4 \cdot N_q \cdot N_k \cdot d \qquad \text{causal: } \times 0.5 \qquad \text{backward: } \times 2.5
$$

That count does not show the softmax work, but the runtime shows it. Sections 4.3 and 5 are about exactly that difference.

### 1.2 Exact HBM traffic of the naive schedule

If you execute it literally, attention is three kernels. Count every byte that crosses HBM for one head, with ${Q, K, V, O, S, P}$ all in bf16:

| Kernel | Reads | Writes | Bytes | N = 4,096, d = 128 |
|---|---|---|---|---|
| 1. $S = \tau \cdot QK^{\top}$ | Q, K: $2Nd \cdot b$ | S: $N^2 \cdot b$ | $2Ndb + N^2 b$ | 35.65 MB |
| 2. $P = \operatorname{softmax}(S)$ | S: $N^2 \cdot b$ | P: $N^2 \cdot b$ | $2N^2 b$ | 67.11 MB |
| 3. ${O = PV}$ | P: $N^2 \cdot b$, V: $Nd \cdot b$ | O: $Nd \cdot b$ | $N^2 b + 2Ndb$ | 35.65 MB |
| **Total** | | | **$4N^2 b + 4Ndb$** | **138.41 MB** |

This is the lower limit for an unfused implementation. Real eager code is worse:

- Each separate elementwise pass over the scores (scale, mask fill, dropout) reads and writes $N^2$ again. The cost is $+2N^2 b$ = +67.1 MB per pass at $N$ = 4,096. A PyTorch sequence `matmul → div → masked_fill → softmax → dropout → matmul` has three of these extra passes.
- If the code keeps the scores in fp32 for the softmax, every $N^2$ term becomes two times as large (272.6 MB at $N$ = 4,096).

(`fa_calculators.naive_traffic()`.)

### 1.3 Arithmetic intensity: why it is memory-bound

$$
I = \frac{4N^2 d}{4N^2 b + 4Ndb} = \frac{N \cdot d}{b \cdot (N + d)} \quad\longrightarrow\quad \frac{d}{b} \quad \text{as } N \to \infty
$$

For $d$ = 128 in bf16, $I$ goes toward 64 FLOP/byte (62.1 at $N$ = 4,096, 63.8 at $N$ = 32,768). The length $N$ cancels. A longer sequence, a larger batch or more heads increase FLOPs and bytes by the same factor. Only the head dimension and the element size set the intensity of the naive schedule.

Each kernel in the schedule is below the ridge, not only the softmax:

| Kernel | Intensity (bf16, large N) | Why |
|---|---|---|
| $QK^{\top}$ | $2N^2 d / \big((2Nd + N^2) \cdot b\big)$ → ${2d/b}$ = 128 | an outer-product-shaped GEMM. The reduction dimension is only $d$, and the write of the $N \times N$ output is most of the traffic |
| softmax | ≈ 5 FLOPs / 4 bytes ≈ 1.25 | streaming only |
| ${PV}$ | $2N^2 d / \big((N^2 + 2Nd) \cdot b\big)$ → ${2d/b}$ = 128 | the read of the $N \times N$ input is most of the traffic |

Ridge points are the dense bf16 peak ÷ HBM bandwidth (datasheet values, **(verify)**):

- H100 SXM: 989.4 TFLOP/s ÷ 3.35 TB/s = **295 FLOP/B**
- L4: 121 ÷ 0.30 = **403**
- A100 80GB: 312 ÷ 2.039 = **153**. The 40 GB A100, at 1.555 TB/s, gives ≈ 200. The primer quotes this figure.
- B200: 2,250 ÷ 8.0 = **281**.

Naive attention at 64 FLOP/B is 2.4× (A100 80GB) to 6.3× (L4) below them.

### 1.4 Worked numbers: N = 4k and 32k, d = 128, bf16, one head

| | N = 4,096 | N = 32,768 |
|---|---|---|
| FLOPs ($4N^2 d$) | 8.59 GFLOP | 549.8 GFLOP |
| Naive HBM bytes | 138.4 MB | 8.62 GB |
| $S$ alone ($N^2 b$) | 33.6 MB | 2.15 GB |
| Intensity | 62.1 FLOP/B | 63.8 FLOP/B |
| **H100**: memory time / compute time | 41.3 µs / 8.7 µs | 2.57 ms / 0.56 ms |
| H100 attainable | 208 TFLOP/s (21% of peak) | 214 TFLOP/s (22%) |
| **L4**: memory time / compute time | 461 µs / 71 µs | 28.7 ms / 4.54 ms |
| L4 attainable | 18.6 TFLOP/s (15% of peak) | 19.1 TFLOP/s (16%) |

These values are lower limits. They assume that every kernel moves data at 100% of HBM bandwidth. Measured naive kernels are slower. (`fa_calculators.roofline()`.)

The second failure is capacity. At $N$ = 32,768, $S$ is 2.15 GB **per head, per sequence**. A 32-head layer needs 68.7 GB for $S$ alone, and the same quantity again for $P$. That is more than an L4's 24 GB, and most of an H100's 80 GB before you load one weight. At long context, the naive schedule is not only slow. It also does not fit.

### 1.5 What a fused schedule can reach

An exact schedule has at least this traffic: it reads ${Q, K, V}$ once and writes $O$ once. That is $4Nd \cdot b$ bytes, plus 4 bytes of fp32 log-sum-exp per row (section 3.5). The result is 4.2 MB at $N$ = 4,096 and 33.7 MB at $N$ = 32,768. These are intensities of 2,040 and 16,320 FLOP/B, far to the right of every ridge. A kernel that gets near this compulsory traffic is compute-bound at 8.7 µs (H100) and 71 µs (L4) for the 4k case.

Two things decide if real kernels get near this compulsory traffic: the loop schedule and the L2 cache. Section 3.4 continues from this point.

---

## 2. Online softmax: the algebra in full

The primer's section 5 shows the update rule and a worked example with symbolic values. This section does these things:

- It derives the rule with the scale.
- It proves the invariant.
- It gives the merge operator, on which every later section depends.
- It shows the base-2 form that the kernels actually execute.
- It lists the numerical guards.

### 2.1 Safe softmax needs two reductions

For one query row with raw scores $s_j = q \cdot k_j$:

$$
\operatorname{softmax}(\tau s)_j = \frac{\exp\big(\tau(s_j - m)\big)}{\sum_k \exp\big(\tau(s_k - m)\big)}, \qquad m = \max_k s_k
$$

The subtraction of $m$ changes nothing mathematically. It makes every exponent $\le 0$, so nothing overflows. But now a row needs two global reductions (the max, then the sum) before the kernel can normalize the first output value. If you do this naively, that is two extra passes over $S$.

### 2.2 The recurrence

Process the keys in blocks $B_1, B_2, \ldots$. Keep three quantities per row: a reference max $m$, a sum $l$, and an unnormalized output $o$ (a $d_v$-vector). Start from $(m, l, o) = (-\infty, 0, 0)$. For block $B$:

$$
\begin{aligned}
m' &= \max\big(m,\ \max_{j \in B} s_j\big) \\
\alpha &= \exp\big(\tau(m - m')\big) && \text{rescale factor for everything seen so far} \\
l' &= \alpha \cdot l + \sum_{j \in B} \exp\big(\tau(s_j - m')\big) \\
o' &= \alpha \cdot o + \sum_{j \in B} \exp\big(\tau(s_j - m')\big) \cdot v_j
\end{aligned}
$$

After the last block, ${O = o / l}$ and $\mathrm{LSE} = \tau \cdot m + \ln l$ (the log-sum-exp of the scaled scores).

**Invariant.** After the recurrence processes blocks $B_1 \ldots B_t$,

$$
\begin{aligned}
l_t &= \sum_{j\ \text{seen}} \exp\big(\tau(s_j - m_t)\big) \quad \text{and} \\
o_t &= \sum_{j\ \text{seen}} \exp\big(\tau(s_j - m_t)\big) \cdot v_j .
\end{aligned}
$$

It holds trivially for the empty prefix. If it holds at $t$, then

$$
\alpha \cdot l_t = \sum_{j\ \text{seen}} \exp\big(\tau(s_j - m_t)\big) \cdot \exp\big(\tau(m_t - m_{t+1})\big) = \sum_{j\ \text{seen}} \exp\big(\tau(s_j - m_{t+1})\big),
$$

and the addition of the new block's terms (already relative to $m_{t+1}$) gives the invariant at ${t+1}$. The same is true for $o$. At the end,

$$
\frac{o}{l} = \frac{\sum_j \exp\big(\tau(s_j - m)\big) \cdot v_j}{\sum_j \exp\big(\tau(s_j - m)\big)} = \sum_j \operatorname{softmax}(\tau s)_j \cdot v_j .
$$

The result is exact, for any block partition.

### 2.3 The state is a monoid: merge any two partial results

Two states from disjoint sets of keys $A$ and $B$ combine in the same way:

$$
\begin{aligned}
&\operatorname{merge}\big((m_a, l_a, o_a),\ (m_b, l_b, o_b)\big): \\
&\qquad m = \max(m_a, m_b) \\
&\qquad l = \exp\big(\tau(m_a - m)\big) \cdot l_a + \exp\big(\tau(m_b - m)\big) \cdot l_b \\
&\qquad o = \exp\big(\tau(m_a - m)\big) \cdot o_a + \exp\big(\tau(m_b - m)\big) \cdot o_b
\end{aligned}
$$

$\operatorname{merge}$ is associative and commutative, and $(-\infty, 0, 0)$ is its identity. Thus the states form a commutative monoid. Everything that follows depends on this fact. You can divide the keys in any way, and calculate the pieces anywhere. Then you can combine them in any order and any tree shape:

| Where the split occurs | Its name | Section |
|---|---|---|
| K/V tiles inside one CTA | FlashAttention's inner loop | 3, 4 |
| K/V ranges across CTAs, which a second kernel combines | split-KV, Flash-Decoding | 6.3 |
| shared prefix against per-request suffix | cascade attention (vLLM, FlashInfer) | 7.3, 7.4 |
| K/V shards across GPUs | ring attention, context parallelism | 8.4 |

**The LSE form.** Kernels usually store a finished partial result as a normalized output $\hat{O} = o/l$ and $L = \tau \cdot m + \ln l$. The same merge in that representation is:

$$
\begin{aligned}
L &= \max(L_a, L_b) + \ln\big(\exp(L_a - \max) + \exp(L_b - \max)\big) \\
\hat{O} &= \exp(L_a - L) \cdot \hat{O}_a + \exp(L_b - L) \cdot \hat{O}_b && \text{(the two weights sum to 1)}
\end{aligned}
$$

FA2's split-KV combine kernel (`combine_attn_seqk_parallel` in `flash_fwd_kernel.h`), vLLM's `merge_attn_states` and FlashInfer's `merge_state` implement this merge. This page uses the code of all three as a source.

### 2.4 The exp2 trick

GPUs have no natural-exponential instruction. The special-function unit (MUFU) calculates $2^x$ (`ex2.approx`). The fast intrinsic `__expf(x)` is a multiply by $\log_2 e$ and then an `ex2`. The accurate `expf` also adds range reduction. Thus kernels combine the softmax scale and $\log_2 e$ into one constant and work in base 2:

$$
\begin{aligned}
c &= \tau \cdot \log_2(e) && \text{(FA2: }\texttt{params.scale_softmax_log2}\text{;} \\
&&& \phantom{(}\text{Triton tutorial: }\texttt{qk_scale *= 1.44269504}\text{)} \\
p_j &= \mathrm{exp2}(s_j \cdot c - m \cdot c) && \text{one FFMA (fused multiply-add) + one EX2 per score} \\
\alpha &= \mathrm{exp2}\big((m - m') \cdot c\big)
\end{aligned}
$$

FA2's `softmax.h` gives the reason: "Instead of computing exp(x - max), we compute exp2(x * log_2(e) - max * log_2(e)). This allows the compiler to use the ffma instruction instead of fadd and fmul separately." The kernel takes the max on the raw scores (this is valid because $\tau > 0$).

A kernel can keep the log-sum-exp in either base. A mix of conventions is a real source of bugs (a factor of $\ln 2$). FA2 stores it in natural-log units (`row_max * softmax_scale + __logf(sum)`). The Triton tutorial stores it in base 2 (`m_i += tl.math.log2(l_i)` with `m_i` already in log2 units).

### 2.5 Numerical safety

- **No overflow, by construction.** Every exponent is $\le 0$. Thus $p \in (0, 1]$ and $\alpha \in (0, 1]$, for all magnitudes of the scores.
- **The final division is safe.** Any row with at least one unmasked key has $l$ ≥ 1, because the maximum adds $\exp(0) = 1$.
- **Underflow is harmless.** Terms below fp32's range flush to zero. Relative to the `1` from the max, they were too small to change the result.
- **Fully masked tiles and rows need a guard.** If every score in a row of a tile is $-\infty$ (a causal or sliding-window mask), then $m' = -\infty$, and $(-\infty) - (-\infty)$ is NaN. In that case, FA2 uses 0 as the max ("If max is -inf, then all elements must have been -inf (possibly due to masking). We don't want (-inf - (-inf)) since that would give NaN."). A row with no permitted key at all ends with $l$ = 0. FA2 then writes $O$ = 0 and $\mathrm{LSE} = +\infty$ ($-\infty$ in its split-KV path). FA3 writes $-\infty$, and vLLM's merge code maps both to $-\infty$ before the merge.
- **Accumulate in fp32.** $m$, $l$ and $o$ are in fp32 registers. $l$ can reach $N$ (when all scores are equal). bf16 cannot even represent every integer above 256.
- **Defer what you can defer.** FA2 divides by $l$ one time at the end, not at every block. It also moves the cross-thread reduction of $l$ to the end. The comment in the source is: "We don't do the reduce across threads here since we don't need to use the row_sum".
- **The reference need not be the true max.** The invariant needs only one thing: all terms of $l$ and $o$ must be relative to the same reference. FA4 uses this on Blackwell with *conditional rescaling*. It keeps the old max, unless the new max is larger by more than a threshold (`rescale_threshold = 8.0` log2 units for 16-bit inputs in `flash_attn/cute/flash_fwd_sm100.py`). Thus most of the time, it does not do the rescale multiply. The cost is that $p$ can now reach $2^8$, so the kernel asserts that `max_offset + rescale_threshold` stays below $\log_2$ of the input dtype's largest value. This limit is necessary because the $P$ that goes into the second matrix multiply is in that dtype.

### 2.6 A hand-worked two-block example

Use one query row, with $d$ = 4, thus $\tau$ = 0.5. There are four keys in two blocks of two. $V$ has two columns, so that the printed output stays small.

```
raw scores q·k:   block 1 = [4, 8]          block 2 = [6, 12]
values:           v1 = [1, 0]   v2 = [0, 1]   v3 = [1, 1]   v4 = [2, −1]
```

**Reference.** The scaled scores are `[2, 4, 3, 6]`, and the softmax is `[0.015219, 0.112457, 0.041371, 0.830953]`. The result is `O = [1.718495, −0.677125]`; `LSE = 6.185182`.

**Block 1.** `m = 8`. `p = exp(0.5·([4, 8] − 8)) = [e^−2, e^0] = [0.135335, 1]`. `l = 1.135335`. `o = 0.135335·v1 + 1·v2 = [0.135335, 1]`.

**Block 2.** `m' = 12`. `α = exp(0.5·(8 − 12)) = e^−2 = 0.135335`. `p = exp(0.5·([6, 12] − 12)) = [e^−3, 1] = [0.049787, 1]`.

```
l' = 0.135335 × 1.135335 + (0.049787 + 1)          = 0.153651 + 1.049787 = 1.203438
o' = 0.135335 × [0.135335, 1] + 0.049787·[1, 1] + 1·[2, −1]
   = [0.018316, 0.135335] + [0.049787, 0.049787] + [2, −1]          = [2.068103, −0.814878]
```

**Finalize.** `O = o'/l' = [1.718495, −0.677125]` and `LSE = 0.5·12 + ln 1.203438 = 6 + 0.185182 = 6.185182`. Both match the reference.

**The same, as the kernel executes it (base 2).** `c = 0.5·log2 e = 0.721348`. In block 1, `s·c = [2.885390, 5.770780]` and `m·c = 5.770780`, thus the exponents are `[−2.885390, 0]`. `exp2` gives `[0.135335, 1]`. These are the same numbers, with one FFMA and one EX2 each.

**The same, as a split-KV merge in LSE form.** Treat the two blocks as two independent splits. Split 1 alone gives `Ô₁ = [0.119203, 0.880797]` and `L₁ = 0.5·8 + ln 1.135335 = 4.126928`. Split 2 alone gives `Ô₂ = [1.952574, −0.905148]` and `L₂ = 0.5·12 + ln 1.049787 = 6.048587`. Then `L = ln(e^4.126928 + e^6.048587) = 6.185182`. The weights are `e^(L₁−L) = 0.127677` and `e^(L₂−L) = 0.872323`, and `O = 0.127677·Ô₁ + 0.872323·Ô₂ = [1.718495, −0.677125]`.

**The same, with a max that lags (FA4-style).** Use a threshold of 3 log2 units as an example (FA4 uses 8). The max increased by `(12 − 8)·c = 2.885 < 3`, so keep `m = 8`. Then block 2 gives `p = exp2([6, 12]·c − 8c) = [e^−1, e^2] = [0.367879, 7.389056]` and `l = 1.135335 + 7.756935 = 8.892271`. It also gives `o = [15.281327, −6.021177]` and `o/l = [1.718495, −0.677125]`. The kernel did no rescale, but the cost is a `p` of 7.39, which is above 1.

(`fa_calculators.online_trace()` and `merge_lse()`. The notebook runs all four variants again.)

---

## 3. FlashAttention-1: tiling and recomputation

### 3.1 The algorithm and its loop order

FA1 (Dao et al., 2022) divides $Q$ into $T_r = N/B_r$ row blocks and ${K, V}$ into $T_c = N/B_c$ blocks. Then it runs the recurrence of section 2.2 tile by tile. Its loop order is **outer over K/V, inner over Q**:

```
FA1 (paper, Algorithm 1, simplified; O, l, m live in HBM)
for j in 1..T_c:                         load K_j, V_j into SRAM            (each read once)
    for i in 1..T_r:                     load Q_i, O_i, l_i, m_i from HBM
        S_ij = τ Q_i K_jᵀ                (B_r × B_c, on chip)
        m~ = rowmax(S_ij);  P~ = exp(S_ij − m~);  l~ = rowsum(P~)
        m_new = max(m_i, m~);  l_new = e^(m_i − m_new) l_i + e^(m~ − m_new) l~
        O_i  <- diag(l_new)^-1 ( diag(l_i) e^(m_i − m_new) O_i + e^(m~ − m_new) P~ V_j )
        write O_i, l_i <- l_new, m_i <- m_new back to HBM

                 K/V block j (outer) ->
               ┌─────┬─────┬─────┬─────┐
   Q block i   │  1  │  5  │  9  │ 13  │      numbers = visit order
   (inner)     ├─────┼─────┼─────┼─────┤      every visit reads and writes O_i, l_i, m_i
       |       │  2  │  6  │ 10  │ 14  │
       v       ├─────┼─────┼─────┼─────┤
               │  3  │  7  │ 11  │ 15  │
               ├─────┼─────┼─────┼─────┤
               │  4  │  8  │ 12  │ 16  │
               └─────┴─────┴─────┴─────┘
```

Note two things. FA2 changes both of them. First, the output accumulator makes a round trip through HBM on every visit. Second, the kernel keeps $O_i$ normalized (divided by $l$) at every step.

### 3.2 Block sizes from SRAM

With $M$ elements of on-chip SRAM, the paper sets $B_c = \lceil M/(4d) \rceil$ and $B_r = \min(\lceil M/(4d) \rceil, d)$. Then $K_j$ and $V_j$ take $2 \cdot B_c \cdot d \approx M/2$, and $Q_i$ and $O_i$ take the same quantity at most. The score tile takes $B_r \cdot B_c \le d \cdot M/(4d) = M/4$. Everything is ${O(M)}$, and the analysis needs nothing more. Because of the constant factors, real kernels adjust tile sizes by hand. On an H100 (228 KB of SMEM per SM, 116,736 bf16 elements) with $d$ = 128, that gives $B_c$ = 228 and $B_r$ = 128 (`fa_calculators.fa1_block_sizes()`).

Production kernels use the same reasoning, with two improvements. First, they round tile sizes to shapes that suit the MMA instructions. Second, the score tile and the output accumulator are in *registers*, so SMEM holds only the Q, K and V tiles. FA2's H100/A100 configuration for $d$ = 128 is $B_r \times B_c$ = 128 × 64. This needs `(128 + 2·64)·128·2 B = 64 KB` of SMEM (section 4.5 has the full table from the source).

### 3.3 The IO-complexity result and a proof sketch

The paper's Theorem 2 (**(verify)** statement) has two parts. Standard attention needs $\Theta(Nd + N^2)$ HBM accesses. FlashAttention needs $\Theta(N^2 d^2/M)$, for $d \le M \le Nd$.

Proof sketch for the FlashAttention side:

1. The kernel reads $K$ and $V$ exactly once: $\Theta(Nd)$.
2. The outer loop runs $T_c = N/B_c = \Theta(Nd/M)$ times, because $B_c = \Theta(M/d)$.
3. Each outer iteration moves all of $Q$ and $O$ through SRAM: $\Theta(Nd)$. It reads $Q$, and it reads and writes $O$.
4. Total: $\Theta(Nd) + \Theta(Nd/M) \cdot \Theta(Nd) = \Theta(N^2 d^2/M)$.

Against the naive $\Theta(N^2)$ (for $N \gg d$), the ratio is of order $M/d^2$. That is an order-of-magnitude statement. The $\Theta$ hides a factor of about 3, and if you put numbers into $M/d^2$, the reduction is three times too large. Keep the constants instead. The naive schedule moves about $4N^2$ elements (section 1.2).

In the FA1 order, each of the $T_c = N/B_c$ outer iterations reads $Q$, reads $O$ and writes $O$ back (${3Nd}$). With the paper's $B_c = M/(4d)$, this gives:

$$
\begin{aligned}
\text{FA1 traffic} &\approx 2Nd + (N/B_c) \cdot 3Nd = 2Nd + 12\, N^2 d^2/M \\
\text{saving} &\approx 4N^2 / (12\, N^2 d^2/M) = M / (3d^2)
\end{aligned}
$$

$$
\begin{aligned}
&\text{FA2 order (no L2 reuse): K and V re-read once per Q block:}\quad 2N^2 d/B_r \\
&\text{saving} \approx 2B_r / d
\end{aligned}
$$

With the H100's $M$ = 116,736 elements, the FA1 reduction is $M/(3d^2)$ ≈ **2.4× at $d$ = 128** and **9.5× at $d$ = 64**. At $N$ = 4,096, a count of bytes with the paper's block sizes gives 59.9 MB against 138.4 MB (2.3×) for $d$ = 128. For $d$ = 64, it gives 15.8 MB against 136.3 MB (8.6×). The difference comes from the fp32 $m$ and $l$ round trips and from the rounded value of $N/B_c$. The FA2 order with $B_r$ = 128 saves $2B_r/d$ = 2× at $d$ = 128 (the table in section 3.4). (`fa_calculators.io_saving()`, `flash_traffic()`.)

Thus FlashAttention does not always move less data. The FA1 order wins only when $M > 3d^2$. At $d$ = 128, that is 49,152 bf16 elements, or 96 KB of SRAM. A100, H100 and L4 have that (164, 228 and 100 KB per SM). A T4, with 64 KB, does not, and on it the FA1-order traffic model at $d$ = 128 is 1.5× *worse* than naive. Because of the L2, real kernels actually move less than any of these values (section 3.4).

The paper measured the HBM reads and writes for GPT-2-sized attention ($N$ = 1,024, $d$ = 64, forward plus backward on an A100). They were 40.3 GB against 4.4 GB **(verify)**. That is a 9.2× reduction. It is of the same order as the constant-aware estimate for $d$ = 64, not the 28× that $M/d^2$ suggests. The naive baseline in that measurement also pays for the extra elementwise passes of section 1.2. Thus the two agree only in order of magnitude.

The lower bound is Proposition 3 (**(verify)**): no exact attention algorithm can use $o(N^2 d^2/M)$ HBM accesses for all $M$ in ${[d, Nd]}$. This is the argument. At $M = \Theta(Nd)$, such an algorithm makes ${o(Nd)}$ accesses. But ${Q, K, V}$ and $O$ have $\Theta(Nd)$ elements that start in HBM (or must end in HBM). This is a contradiction.

### 3.4 The model against a real GPU: count it

The IO-complexity model assumes that no data stays in a cache between SRAM and HBM. This table counts bytes under that model, with the actual block sizes (`fa_calculators.flash_traffic()`, one head, bf16, $d$ = 128):

| Schedule | N = 4,096: bytes / intensity | N = 32,768: bytes / intensity |
|---|---|---|
| Naive, three kernels | 138.4 MB / 62 | 8,623 MB / 64 |
| FA1 order (outer K/V, $B_c$ = 128) | 104.9 MB / 82 | 6,593 MB / 83 |
| FA1 order, the paper's $B_c$ = 228 (section 3.3) | 59.9 MB / 143 | 3,716 MB / 148 |
| FA2 order (outer Q, $B_r$ = 128) | 69.2 MB / 124 | 4,312 MB / 127 |
| FA2 order, causal | 36.7 MB / 117 | 2,173 MB / 127 |
| Compulsory ($4Nd \cdot b$ + LSE) | 4.2 MB / 2,040 | 33.7 MB / 16,320 |

There are two lessons.

**Only $B_r$ sets the per-tile intensity.** In the FA2 order, the kernel loads each K/V tile once per Q block, and the $B_r$ query rows of that CTA use it. That is $4 \cdot B_r \cdot B_c \cdot d$ FLOPs for $2 \cdot B_c \cdot d \cdot b$ bytes, or $2 \cdot B_r/b$ FLOP/B. In bf16, this is $B_r$. With $B_r$ = 128, that is 128 FLOP/B, which is still below the H100 ridge (295). $B_r$ cannot increase much more, because the output accumulator for $B_r \times d$ fp32 values must fit in registers.

**L2 closes the gap.** CTAs that work on the same (batch, head) run at the same time. They read the same $K$ and $V$ tiles again, and these tiles then come from L2, not from DRAM. On an H100, the $K$ and $V$ of one head are 2 MB at $N$ = 4,096 and 16.8 MB at $N$ = 32,768. The L2 is 50 MB.

FA3 makes this explicit. Its persistent scheduler goes through the heads in "sections". It sets the size of a section so that $K$ and $V$ fit in a 32 MB L2 budget. The source comment is: "we have to make sure K & V still fit into L2 cache, so we perform scheduling on 'sections' of the head & batch dimension" (`hopper/tile_scheduler.hpp`).

A profiler shows this L2 reuse as a high L2 hit rate and DRAM traffic near the compulsory row. The IO-complexity result explains why tiling wins. But the number that you measure depends on L2 residency and on how many SMs are busy.

### 3.5 The backward pass: recompute from the log-sum-exp

The forward saves $O$ ($N \times d$) and one fp32 number per row, $L_i = \tau \cdot m_i + \ln l_i$. When there is dropout, it also saves the RNG seed and offset. It does not save $P$. Here, $S$ includes the scale, and $\mathit{dO}$ is the gradient of the output:

$$
\begin{aligned}
P_{ij} &= \exp(S_{ij} - L_i) && \text{recomputed per tile: no max, no sum, } L \text{ already normalizes} \\
\mathit{dV} &= P^{\top} \; \mathit{dO} \\
\mathit{dP} &= \mathit{dO} \; V^{\top} \\
D_i &= \operatorname{rowsum}(\mathit{dO}_i \circ O_i) && \text{an } O(Nd) \text{ pre-pass} \\
\mathit{dS} &= P \circ (\mathit{dP} - D) && \text{the softmax Jacobian, row by row} \\
\mathit{dQ} &= \tau \cdot \mathit{dS} \; K \\
\mathit{dK} &= \tau \cdot \mathit{dS}^{\top} Q
\end{aligned}
$$

This is why $D$ can come from $O$:

$$
D_i = \sum_j P_{ij} \cdot \mathit{dP}_{ij} = \sum_j P_{ij} \, (\mathit{dO}_i \cdot v_j) = \mathit{dO}_i \cdot \sum_j P_{ij} v_j = \mathit{dO}_i \cdot O_i .
$$

FA2 calculates it in `dot_do_o`. The Triton tutorial calculates it in `_attn_bwd_preprocess` (`delta = tl.sum(o * do, axis=1)`).

For memory, at $N$ = 32,768 with 32 heads, the saved statistics are `32 × 32,768 × 4 B = 4.2 MB`, against 68.7 GB for $P$ in bf16.

This is how FA2's backward partitions the work (`flash_bwd_kernel.h`, `compute_dq_dk_dv_1colblock`). Each CTA owns one K/V column block and goes through all Q blocks in a loop. It keeps $K_j, V_j, \mathit{dK}_j, \mathit{dV}_j$ on chip, and writes $\mathit{dK}_j, \mathit{dV}_j$ one time. Every CTA adds to every $\mathit{dQ}_i$, so the kernel sums these contributions with fp32 `atomicAdd` into a `dq_accum` buffer. Atomics make the summation order non-deterministic, and thus also the last bits of $\mathit{dQ}$.

`deterministic=True` gives each CTA its own accumulator ("If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer"). The README says that this option is slightly slower and uses more memory.

### 3.6 Why the backward costs about 2.5× the forward

Count the matrix products. The forward has two ($QK^{\top}$, ${PV}$), that is $4N^2 d$ FLOPs. The backward has five, each $2N^2 d$. It recomputes $S = QK^{\top}$, then calculates $\mathit{dV} = P^{\top} \mathit{dO}$, $\mathit{dP} = \mathit{dO} \cdot V^{\top}$, $\mathit{dQ} = \mathit{dS} \cdot K$ and $\mathit{dK} = \mathit{dS}^{\top} Q$. That is $10N^2 d = 2.5 \times 4N^2 d$. The Triton tutorial's benchmark uses exactly this: `total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)`.

The wall-clock time is usually worse than 2.5×, for three reasons:

- The backward has more elementwise work per tile ($\mathit{dS}$).
- It holds more live tiles per CTA (fewer CTAs per SM).
- It pays for atomics or a second pass for $\mathit{dQ}$.

The FA2 paper reports that the backward reaches a lower fraction of peak than the forward **(verify)**.

Because of section 1, recomputation is a good exchange. To derive $P$ again costs one extra GEMM ($2N^2 d$ = 4.3 GFLOP per head at $N$ = 4,096, 4.3 µs on an H100 at peak). In return, the forward does not write $P$, and the backward does not read it back. That traffic is 2 × 33.6 MB = 67 MB, 20 µs at 3.35 TB/s. Also, recomputation needs no memory to hold $P$ between the two passes.

---

## 4. FlashAttention-2: work partitioning

FA2 (Dao, 2023) keeps the algorithm and changes the division of the work. [The CUDA primer (02)](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) explains the execution model that FA2 depends on (grids, CTAs, warps, occupancy, shared memory).

### 4.1 Swap the loops: one CTA per Q block

```
FA1                                        FA2
for j (K/V block):                         grid over (Q block i, batch, head)    <- one CTA each
    for i (Q block):                           load Q_i once
        load Q_i, O_i, l_i, m_i                for j (K/V block):
        update                                     load K_j, V_j
        store O_i, l_i, m_i                        update O_i, l_i, m_i in registers
                                               store O_i / l_i and LSE_i once
```

The launch in `flash_fwd_launch_template.h` is `dim3 grid(num_m_block, params.b, params.h)`. Thus the grid covers the Q blocks, and also batch and heads. This has three results.

1. **The accumulator never leaves the chip.** $O_i$, $m_i$ and $l_i$ stay in registers for the whole inner loop, and the kernel writes them one time. The largest term of the FA1 row in section 3.4, the round trip of $O$ through HBM, goes away. Only the repeated reads of $K$ and $V$ stay. $K$ and $V$ are read-only, and the L2 caches them well.
2. **Parallelism over the sequence.** FA1's kernel had parallelism over batch and heads only **(verify)**. With batch 1 and 16 heads (for example, a tensor-parallel shard), that is 16 CTAs for 108 or 132 SMs. FA2 at $N$ = 16,384, $B_r$ = 128 launches `16 × 128 = 2,048` CTAs.
3. **Causal work is easy to skip.** Each CTA knows its row range, so it stops at the diagonal (section 4.4).

### 4.2 Inside the CTA: split-Q instead of split-K

A CTA has 4 (sometimes 8) warps. The way that they divide a $B_r \times B_c$ tile sets how much data they must exchange with each other:

```
FA1, "split-K": warps share Q_i, split K_j/V_j       FA2, "split-Q": warps split Q_i, share K_j/V_j

            K_jᵀ columns ->                                   K_jᵀ columns ->
          ┌──────┬──────┬──────┬──────┐                     ┌───────────────────────────┐
  all     │  w0  │  w1  │  w2  │  w3  │             rows    │ w0: complete rows of S, P │
  rows    │      │      │      │      │             of Q_i  │ w1: complete rows of S, P │
  of Q_i  └──────┴──────┴──────┴──────┘                     │ w2: ...                   │
  every warp holds a partial rowmax, rowsum and             │ w3: ...                   │
  P_w V_w for ALL rows: write to SMEM, barrier,             └───────────────────────────┘
  read back, add                                            no inter-warp exchange; P stays in
                                                            registers as the A operand of P·V
```

In FA2's kernel, each warp owns interleaved 16-row slices of the Q tile. The kernel applies the mask at row `m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4` with a stride of `kNWarps * 16`. In a warp, 4 threads share one row of the MMA accumulator. Thus the row max and row sum are 4-thread shuffles (`Allreduce<4>` in `softmax.h`), not SMEM traffic. The kernel converts the probabilities to fp16/bf16 in registers (`convert_type<Element>(acc_s)`) and sends them directly into the second MMA as a register operand (`gemm_rs`). No part of $P$ or of the partial $O$ goes into shared memory.

### 4.3 Fewer non-matmul FLOPs, and why they cost so much

FA2 changes the arithmetic itself in these ways:

- It keeps $O$ unnormalized and divides by $l$ one time at the end. FA1 rescaled by $\operatorname{diag}(l)^{-1}$ at every block.
- It saves only the LSE for the backward, not $m$ and $l$ separately.
- It reduces $l$ across threads one time at the end, not at every block.
- It moves the scale into `exp2`, so each score costs one FFMA and one EX2 (section 2.4).

Each score has 512 matmul FLOPs and only a few elementwise operations. But these operations are important, because the units that execute them are much slower. The table gives the values per SM per clock (the throughput tables of the CUDA programming guide, and datasheet peaks, **(verify)** for B200):

| Per SM per clock | A100 | H100 SXM | B200 |
|---|---|---|---|
| dense bf16 tensor-core FLOPs | 2,048 | 4,096 | ≈ 8,192 |
| FP32 FMA instructions | 64 | 128 | 128 |
| MUFU instructions (EX2) | 16 | 16 | 16 |

Per score, with $d$ = 128, there are ${4d}$ = 512 MMA FLOPs, one EX2, and about 5 FP32 instructions. The FP32 instructions are:

- an FFMA for scale-and-subtract
- a max
- an add into $l$
- the $O$ rescale, amortized as $d/B_c$ = 2 FMULs at $B_c$ = 64
- a conversion that handles two values per instruction.

This table gives the clocks per score if each unit runs alone (`fa_calculators.clocks_per_score()`):

| | A100 | H100 | B200 |
|---|---|---|---|
| tensor cores | 0.25 | 0.125 | 0.0625 |
| EX2 on MUFU | 0.0625 (25% of MMA time) | 0.0625 (**50%**) | 0.0625 (**100%**) |
| ≈ 5 FP32 instructions | 0.078 (31%) | 0.039 (31%) | 0.039 (62.5%) |

Different units run at the same time when different warps supply them with work. Thus the total is not the sum of these values. But some kernels have warps that do MMA, then softmax, then MMA, all in lockstep. Such a kernel pays these fractions as idle tensor-core time. This table is the reason for FA3's overlap (the H100 column) and FA4's exponential emulation (the B200 column). The H100 row agrees with how the FA3 paper itself states the problem: 989 TFLOP/s of matmul against about 3.9 TFLOP/s of special functions **(verify)**.

### 4.4 Causal block skipping

This is from `compute_attn_1rowblock` in `flash_fwd_kernel.h`:

$$
\begin{aligned}
\texttt{n_block_max} &= \min\Big(\Big\lceil \frac{N_k}{B_c} \Big\rceil,\ \Big\lceil \frac{(\texttt{m_block} + 1) \cdot B_r + N_k - N_q}{B_c} \Big\rceil\Big) && \text{causal upper bound} \\
\texttt{n_block_min} &= \max\Big(0,\ \frac{\texttt{m_block} \cdot B_r + N_k - N_q - \texttt{window_left}}{B_c}\Big) && \text{sliding window only}
\end{aligned}
$$

The loop runs from `n_block_max − 1` *down* to `n_block_min`. Thus the $\lceil B_r/B_c \rceil$ diagonal tiles that need a mask (`n_masking_steps`) come first. The rest of the loop body contains no mask code. For $N$ = 512, $B_r$ = 128, $B_c$ = 64:

```
                   K/V blocks (B_c = 64) ->
                   0   1   2   3   4   5   6   7
Q block 0         ░░  ░░  ··  ··  ··  ··  ··  ··          ██  full tile: no mask code
Q block 1         ██  ██  ░░  ░░  ··  ··  ··  ··          ░░  diagonal tile: masked
Q block 2         ██  ██  ██  ██  ░░  ░░  ··  ··          ··  never loaded, never computed
Q block 3         ██  ██  ██  ██  ██  ██  ░░  ░░
                                                          20 of 32 tiles visited, 8 masked
```

At $N$ = 4,096, the same tiling visits 1,056 of 2,048 tiles (51.6%, 64 masked). At $N$ = 32,768, it visits 65,792 of 131,072 (50.2%). The "causal = 0.5 × FLOPs" convention is the large-$N$ limit (`fa_calculators.causal_tiles()`).

The causal mask also causes load imbalance. Here, Q block $m$ does ${2(m + 1)}$ tiles, so the last blocks are the longest. FA3's persistent scheduler gives out the longest tiles that are left first (`hopper/tile_scheduler.hpp`). The source comment is: "We use longest-processing-time-first scheduling: the longest remaining tile is assigned to the first SM that's free".

### 4.5 Tile sizes and the SMEM budget

These values come from `run_mha_fwd_hdim*` in `flash_fwd_launch_template.h`. SMEM is the Q tile plus one K tile and one V tile, $(B_r + 2B_c) \cdot d \cdot b$ (`fa_calculators.tile_smem_bytes()`):

| GPU, head dim | $B_r \times B_c$ | Warps | SMEM | Source comment |
|---|---|---|---|---|
| A100, H100, $d$ = 128 | 128 × 64 | 4 | 64 KB | "1st ones are good for H100, A100" |
| sm86/sm89 (A10, RTX 30/40, **L4**), $d$ = 128, non-causal | 128 × 32 | 4 | 48 KB | "128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM" |
| sm86/sm89, $d$ = 128, causal | 64 × 64 | 4 | 48 KB | "64 x 64 is the fastest for causal (because it's square)" |
| A100, $d$ = 256 | 128 × 64 | 8 | 128 KB | "For A100, we want to run with 128 x 64 (128KB smem)" |
| H100, $d$ = 256 | 64 × 64 | 4 | 96 KB | "For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM" |

The main loop overlaps loads with computation through Ampere's asynchronous copies (`cp.async`). While $S_j = Q K_j^{\top}$ runs, $V_j$ is in flight. Immediately after that GEMM, the kernel issues the load of $K_{j-1}$, so that the load overlaps the softmax and $P_j V_j$.

### 4.6 What FA2 achieved, and where it stopped

FA2 is about 2× faster than FA1 (FA2 paper **(verify)**). In the forward pass, it reaches 50–73% of the A100's peak, against 25–40% for FA1 (FA2 paper **(verify)**). The primer's "~70% of A100 peak" is the top of this range. In end-to-end training of GPT-style models, FA2 gives 225 TFLOP/s per A100, at 72% model FLOPs utilization (the repository README). On an H100, the same kernel reaches only about 35% of peak (FA3 paper **(verify)**). The reason is that it uses Ampere's synchronous warp-level `mma.sync` and `cp.async`, but Hopper's full throughput needs asynchronous warpgroup MMAs and the TMA.

---

## 5. FlashAttention-3: asynchrony on Hopper (and FA4 on Blackwell)

FA3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024) is a Hopper-specific rewrite on CUTLASS/CuTe. This section uses this source: `hopper/flash_fwd_kernel_sm90.h`, `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `hopper/softmax.h`, `hopper/tile_size.h`, `hopper/heuristics.h` and `hopper/tile_scheduler.hpp`.

### 5.1 What Hopper changed

- **WGMMA.** A *warpgroup* (4 warps, 128 threads) issues matrix multiplies, and they run asynchronously. The operands come from SMEM (the A operand can come from registers). The results go into registers. The warps continue to execute until they explicitly wait (`warpgroup_wait<N>()`).
- **TMA.** One thread issues a bulk copy of a whole tile from HBM to SMEM, and a tensor descriptor describes the tile. A transaction-counting barrier in SMEM (an mbarrier) monitors the completion. The main loop has no address arithmetic.
- **Register reallocation.** Warpgroups can give registers back and take more registers at run time (`warpgroup_reg_dealloc` / `warpgroup_reg_alloc`).
- **228 KB of SMEM per SM**. Hopper also has thread-block clusters and TMA multicast.
- **The exponential got relatively slower** (section 4.3). At $d$ = 128, EX2 alone takes half as long as the MMAs.

### 5.2 Warp specialization: one producer, two consumers

```
one CTA, d = 128, B_r × B_c = 128 × 176

WG0  producer    warpgroup_reg_dealloc<24>     one warp issues TMA: Q once, then K_j and V_j
                                               into a ring of kStages SMEM buffers
WG1  consumer    warpgroup_reg_alloc<240>      rows 0..63:   S = Q Kᵀ (WGMMA), softmax, O += P V
WG2  consumer    warpgroup_reg_alloc<240>      rows 64..127: the same
         pipeline_k, pipeline_v: barrier-guarded ring buffers
         consumer_wait(...) before using a stage, consumer_release(...) after
```

The register numbers come from the source (`LoadRegisterRequirement` 24 and `MmaRegisterRequirement` 240 for two consumer warpgroups with TMA loads). Everything in the design depends on them. A consumer thread holds its share of three tiles:

- the fp32 score tile (`64 × 176 / 128` = 88 registers)
- the fp32 output accumulator (`64 × 128 / 128` = 64)
- the bf16 probabilities (44).

That is about 196 registers before addresses and statistics. The SM has 65,536 registers, and `128 × 24 + 256 × 240 = 64,512`. Thus the registers that the producer gives back are what let one CTA with two consumer warpgroups fit (`fa_calculators.fa3_registers()`).

### 5.3 Ping-pong between warpgroups

The two consumer warpgroups use the tensor cores in turns, through named barriers (`warp_scheduler_barrier_sync` / `warp_scheduler_barrier_arrive`). A warpgroup waits for its turn and issues its GEMMs. Then it sends a signal to the other warpgroup, and does its softmax while the GEMMs of the other warpgroup run.

```
tensor cores   │ WG1: S, PV │ WG2: S, PV │ WG1: S, PV │ WG2: S, PV │ ...
MUFU / FP32    │ WG2 softmax │ WG1 softmax │ WG2 softmax │ WG1 softmax │ ...
               └──────────── time ───────────────────────────────────────>
```

The FA3 paper reports that this schedule increases the FP16 forward at $d$ = 128 from about 570 to 620–640 TFLOP/s **(verify)**.

### 5.4 Pipelining inside one warpgroup

In a warpgroup, the loop has a skew of one iteration. The source comment says: "Each step does gemm0 for iter n_block, gemm1 for iter n_block + 1, and softmax for iter n_block". The loop counts down, so `n_block + 1` is the previous block. In forward order:

```
step j:   issue  S_j = Q K_jᵀ               (WGMMA, async)
          issue  O  += P_(j−1) V_(j−1)      (WGMMA, async)
          wait for S_j only                 (warpgroup_wait<1>)
          softmax(S_j) -> P_j, α_j          <- MUFU/FP32 work overlaps the P·V GEMM still running
          wait for the P·V GEMM             (warpgroup_wait<0>)
          O *= α_j                          (rescale; O is now relative to m_j)
```

The cost is register pressure, because a score tile and a probability tile are live at the same time. SMEM also limits the tile size. The source notes that `128 × 192` hits the SMEM limit when $P$ stays in registers (`MmaPV_is_RS`). It notes that `128 × 144` hits the limit when $P$ goes through SMEM. This is why the $d$ = 128 tile in the table of section 5.5 is `128 × 176`.

### 5.5 Tile sizes (from `tile_size_fwd_sm90`)

| Head dim | bf16/fp16 $B_r \times B_c$ | FP8 $B_r \times B_c$ |
|---|---|---|
| 64 | 192 × 192 (192 × 128 causal, local, or paged without TMA) | 192 × 160 |
| 96 | 192 × 144 (192 × 128 local or paged without TMA) | 192 × 128 |
| 128 | 128 × 176 (128 × 128 causal, local, or paged without TMA) | 128 × 224 |
| 192 | 128 × 128 (128 × 112 if $d_v$ > 128) | 128 × 160 |
| 256 | 128 × 80 ("128 x 80 hits the limit of smem") | 128 × 128 |

$B_c$ = 176 is not a power of two, because `tile_size_fwd_sm90` sets the tile size to fill SMEM exactly. FP8 tiles are wider because each element has half the bytes.

### 5.6 FP8

- **Layout constraints.** Hopper's FP8 WGMMA needs both operands K-major in SMEM. For $P \cdot V$, $V$ is not K-major. FA3 transposes $V$ tiles in SMEM in the producer warpgroup (`Transpose_V = Is_FP8 && !V_colmajor`, a separate `pipeline_vt`). It also permutes the score registers, so that the fp32 accumulator layout matches the FP8 operand layout (`permute_Cregs_fp8`).
- **Scales.** The interface takes `q_descale`, `k_descale`, `v_descale`. The main loop indexes them per (batch, KV head). It combines `q_descale · k_descale` with the base-2 softmax scale, and applies `v_descale` to the output. vLLM gives them with shape `(num_sequences, num_kv_heads)`.
- **Use of the FP8 range for P.** Probabilities are at most 1. The e4m3 format has only 3 mantissa bits, and its smallest subnormal is $2^{-9}$. FA3's FP8 softmax multiplies the probabilities by $2^8$ (`Max_offset = 8`). The source comment is: "For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount". The multiplication by $2^8$ keeps small probabilities out of the flush-to-zero range. Section 9 gives the numbers for this effect.
- **Block quantization and incoherent processing (paper, (verify)).** Block quantization uses one scale per tile, not per tensor. Incoherent processing applies a random orthogonal rotation $M$ (a Hadamard transform with random signs) to $Q$ and $K$. Then $(QM)(KM)^{\top} = QMM^{\top}K^{\top} = QK^{\top}$, but the rotation spreads an outlier channel across all $d$ channels. With the fast Walsh-Hadamard transform, the rotation costs $O(d \log d)$ per row, and you can fuse it with the rotary embedding. The paper reports 2.6× lower numerical error than a baseline FP8 attention on inputs with outliers **(verify)**.

  Section 9.4 shows when each half helps. For e4m3, the rotation is worth its cost when an outlier channel is large in both $Q$ and $K$. Block scales are worth their cost when a few tokens are much larger than the rest. Block scales are much more important for integer formats than for e4m3.
- **Throughput (paper, (verify)).** The bf16/fp16 forward reaches up to 740 TFLOP/s (75% of 989). FP8 reaches near 1.2 PFLOP/s. FA3 is 1.5–2.0× faster than FA2.

### 5.7 Blackwell and FlashAttention-4 **(verify)**

The hardware changed in these ways:

- A single thread issues MMAs (`tcgen05`). They accumulate into a new per-SM *Tensor Memory* (TMEM), not into registers.
- Two CTAs can cooperate on one MMA tile.
- Dense bf16 throughput is about 2.25 PFLOP/s, 2.3× an H100. But the exponential unit and SMEM bandwidth did not increase by the same factor.

The table in section 4.3 shows the result. At $d$ = 128, EX2 now takes as long as the matrix multiplies.

This is what FA4 does about it, as `flash_attn/cute/flash_fwd_sm100.py` and `flash_attn/cute/softmax.py` show:

- **More specialized warps.** Softmax warps 0–3 and 4–7 work on two Q tiles in a ping-pong (`q_stage = 2`). A separate *correction* warpgroup (warps 8–11) rescales the output in TMEM, outside the critical path of the softmax. One warp (12) issues all MMAs. Warp 13 runs the epilogue, and warp 14 runs the loads.
- **Exponentials partly on the FMA pipe.** For an adjusted fraction of the `exp2` calls, a polynomial (`ex2_emulation_2`) does the calculation instead of the MUFU. The setting `ex2_emu_freq` is 8 to 32, dependent on dtype and configuration. The value 8 is the FP8 causal $d$ = 128 entry. On SM103, which "has fast native exp2", the setting is 0.
- **Conditional rescaling** with `rescale_threshold = 8.0` (section 2.5).
- The second MMA gets $P$ from TMEM (`OperandSource.TMEM`). FA4 also uses 2-CTA MMA instructions (`use_2cta_instrs`).
- The authors wrote FA4 in CuTe-DSL (Python). The package is `flash-attn-4` (`from flash_attn.cute import flash_attn_func`, as the README shows).

The reported FA4 numbers are up to 1,605 TFLOP/s bf16 on B200 (71% utilization), 1.3× cuDNN 9.13 and 2.7× Triton ([the primer's section 8](flash-attention-primer.md#8-the-lineage), **(verify)**). The primer also says that FA4 was initially slower than FA2 for decode, until the developers ported split-KV to it. On SM100, vLLM selects FA4 when FA4 is installed (`get_flash_attn_version` in `vllm/v1/attention/backends/fa_utils.py`). But for causal decoder attention on SM100, its backend priority list puts FlashInfer first (section 7.4).

---

## 6. Decode is a different kernel

### 6.1 One query row, a long KV cache: no reuse

In a decode step, each sequence adds one new query token per head. For one sequence, one layer and one KV head with $L$ cached tokens:

$$
\begin{aligned}
\text{bytes} &= 2 \cdot L \cdot d \cdot b && \text{read K and V once} \\
\text{FLOPs} &= 4 \cdot L \cdot d \cdot g && g \text{ query heads share this KV head (GQA)} \\
I &= 2g / b && \text{FLOP/byte: MHA bf16} = 1,\ g = 4 \to 4,\ g = 8 \to 8; \\
&&& \text{FP8 KV doubles it}
\end{aligned}
$$

Each of these values is one to two orders of magnitude below the ridge (295 on an H100, 403 on an L4). Decode attention is fully bandwidth-bound. Its time is the bytes divided by the achieved bandwidth, and to first order nothing else is important. Take a Llama-3-8B-shaped layer (32 query heads, 8 KV heads, $d$ = 128) at $L$ = 32,768. Per layer, it has 134 MB of K/V and 537 MFLOP, which take 40 µs at 3.35 TB/s. For a single sequence, that is 1.28 ms per token across 32 layers (`fa_calculators.decode_intensity()`).

### 6.2 Why the prefill kernel is the wrong shape

Take the prefill kernel (FA2's `flash_fwd_kernel`, one CTA per Q block, batch entry and head) and give it $N_q$ = 1:

- **Too few CTAs.** The grid is $\lceil N_q/B_r \rceil \times \text{batch} \times \text{heads}$. With batch 1 and a Llama-3-8B-shaped layer, that is 32 CTAs, one per query head, on a 132-SM H100 (76% of the SMs are idle). If the kernel puts the 4 query heads of each GQA group into one tile (section 6.4), the grid has 8 CTAs. Then 94% of the SMs are idle. In both cases, so few SMs cannot get near the full HBM bandwidth.
- **Each CTA walks all of $L$ serially**: $L/B_c$ tiles, one after the other.
- **A 128-row tile with 1 useful row** (4 with GQA packing) wastes almost all of each MMA. When the kernel is bandwidth-bound, this is not the bottleneck. But it shows that the kernel has the incorrect shape for this problem.

For decode, two things are important: the number of SMs that issue loads, and the bytes per useful FLOP. The libraries know this, and they dispatch the correct kernel for you. FA2's `flash_attn_func` (`mha_fwd`) and `flash_attn_with_kvcache` both find the case `seqlen_q = 1` and fold the GQA group into the sequence axis. Then they run the split-KV kernel of section 6.3 with a heuristic split count (`set_params_splitkv(..., num_splits = 0, ...)` in `flash_api.cpp`).

Only two callers get the shape that the list in this section describes. One is a caller that passes `num_splits = 1` to `flash_attn_with_kvcache`. The other is a kernel written by hand, for example the one in section 11.

### 6.3 Flash-Decoding: split the KV sequence, then reduce

```
q (g rows: the query heads of one KV head)
K/V cache (L tokens):   ├── split 0 ──┼── split 1 ──┼── ... ──┼── split S−1 ──┤
one CTA per (batch, KV head, split):  (Ô_0, L_0)    (Ô_1, L_1)           (Ô_S−1, L_S−1)     fp32 partials in HBM
combine kernel:         L = logsumexp_s(L_s)        O = Σ_s exp(L_s − L) · Ô_s                 (section 2.3)
```

In FA2, this is `flash_fwd_splitkv_kernel`, launched on `dim3 grid(num_m_block, num_splits, b * h)`. After it, `flash_fwd_splitkv_combine_kernel` runs. The combine kernel calculates three things, exactly as in section 2.3:

- the max of the split LSEs
- the log of the sum of exponentials
- the per-split weights `exp(lse − lse_logsum)`.

The FA2 changelog 2.2 says: "we split the loading across different thread blocks, with a separate kernel to combine results". Dao, Haziza, Massa and Sizov published the idea as Flash-Decoding in 2023. They reported decoding up to 8× faster at very long sequences **(verify)**.

**How many splits.** The function is `num_splits_heuristic` in `csrc/flash_attn/flash_api.cpp`, and `fa_calculators.num_splits_heuristic()` is a port of it:

1. If `batch × heads × query_blocks ≥ 0.8 × SMs`, do not divide the KV sequence (the grid already fills the GPU). FA2 passes `2 × SMs`, because two 128-thread CTAs fit on each SM.
2. If not, calculate the wave efficiency of each split count, $\text{waves} / \lceil \text{waves} \rceil$ with $\text{waves} = \text{CTAs} \times \text{splits} / \text{SMs}$. Do not examine a split count that gives the same partition as a smaller count. Return the **smallest** count within 85% of the best. More splits than necessary only add partial results to write and to combine.

FA3's version (`hopper/heuristics.h`, `get_num_splits` in `hopper/flash_api.cpp`) is different in four ways:

- It does not double the SM count.
- It has no rule that ignores split counts that give the same partition.
- It never divides the KV sequence when there are 4 or fewer KV blocks.
- It divides the KV sequence even when the GPU is full, if three conditions are true. One KV head is larger than a 50 MB L2 estimate, there are sufficient query blocks, and the mask is neither causal nor local.

For variable-length batches, which is how vLLM calls it, that static count is only an upper bound. FA3 calculates the bound as if the batch had one sequence. Then a small prepare kernel (`prepare_varlen_num_blocks` in `hopper/flash_prepare_scheduler.cu`) selects the split count of each sequence from the total work of the batch:

$$
\begin{aligned}
\texttt{blocks_per_sm} &= \Big\lceil \sum_{\text{sequences}} \text{KV blocks} \times 1.1 \times H_{kv} / \text{SMs} \Big\rceil \qquad \text{(10% margin)} \\
\text{splits} &= \operatorname{clamp}\Big(\Big\lceil \text{KV blocks of this sequence} / \texttt{blocks_per_sm} \Big\rceil,\ 1,\ \text{static bound}\Big)
\end{aligned}
$$

Outside CUDA graphs, vLLM passes `num_splits = 0` ("0 means use FA3's heuristics"). Inside them, it passes its CUDA-graph cap (`flash_attn_max_num_splits_for_cuda_graph`, 32 by default). In both cases, the dynamic count decides in the worked examples that follow.

The worked examples use Llama-3-8B shapes (32 query heads, 8 KV heads packed as in section 6.4, $d$ = 128). FA2 is what `flash_attn_with_kvcache` (or `flash_attn_func` with `seqlen_q = 1`) launches, and its split kernel uses $B_c$ = 128 (`fa_calculators.fa2_decode_splits()`). FA3 is what vLLM runs on an H100. With 16-token pages, the paged path cannot use the TMA, thus the key tile is 128 wide (`fa_calculators.fa3_decode_splits()`). vLLM builds FA3 from its fork `vllm-project/flash-attention`. For this case, the split logic of the fork is the same as upstream, **(verify)** on upgrade. The A100 and the L4 run FA2 in vLLM (section 7.4).

| Batch × context | GPU | CTAs without split | FA2: splits, CTAs (KV blocks per split) | FA3 in vLLM: splits, CTAs |
|---|---|---|---|---|
| 1 × 32k | H100 (132 SMs) | 8 | 29, 232 (9) | 15, 120 |
| 1 × 32k | A100 (108) | 8 | 24, 192 (11) | n/a |
| 1 × 32k | L4 (58) | 8 | 13, 104 (20) | n/a |
| 8 × 32k | H100 | 64 | 4, 256 (64) | 2, 128 |
| 64 × 4k | H100 | 512 | 1, 512 (32) | 1, 512 |

FA2's target is two CTAs per SM (it passes `2 × SMs`). The target of FA3's dynamic rule is approximately one wave of 132 CTAs. Both show the same thing. With a small batch, the number of CTAs that fill the machine sets the split count. When the batch fills the machine by itself, the split count decreases to 1.

The cost is small. At 1 × 32k on an H100 with FA2's 29 splits, the fp32 partial outputs are `29 × 8 × 4 × 128 × 4 B` = 475 KB (plus 3.7 KB of LSEs). The kernel writes them once and reads them back once. That is 0.96 MB against 134 MB of K/V per layer (0.7%). FA3's 15 splits make it half as large (`fa_calculators.split_partials_bytes()`).

The cost that is important is a different one. The split count depends on the batch size and the SM count, thus the order of the final sum also depends on them. Section 9.3 comes back to this.

### 6.4 GQA packing

The $g$ query heads that share a KV head need the same K and V. If the kernel treats them as $g$ rows of one Q tile, it loads K and V once per KV head. It does not load them once per query head. Also, the MMA gets ${M = g}$ useful rows instead of 1. All three libraries do this:

- FA2 moves the group into the sequence dimension when `seqlen_q == 1 and num_heads > num_heads_k`, and there is no sliding window, ALiBi or dropout. The code says: "Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case" (`seqlenq_ngroups_swapped` in `flash_api.cpp`).
- FA3 decides with `should_pack_gqa`. It packs when the unpacked tile efficiency `seqlen_q / round_up(seqlen_q, B_r)` is less than 0.9 × the packed one, `(seqlen_q·g) / round_up(seqlen_q·g, B_r)`. It always packs for variable-length batches.
- FlashInfer's decode wrapper has `use_tensor_cores`. Its description says: "Will be faster for large group size in grouped query attention."

MLA (section 8.1) is the extreme case. In MLA, 128 query heads share one latent vector per token. Packing increases decode attention from the range of 1 to 8 FLOP/B to within 20% of the H100's bf16 ridge. Thus kernel authors adjust it like a GEMM, not like a copy.

### 6.5 Decode attention time is proportional to batch × context × KV bytes

In each decode step, attention must read each cached token of each sequence in each layer:

$$
t_{\text{attention}} \approx \frac{\big(\sum_{\text{sequences}} L_s\big) \times 2 \cdot n_{\text{layers}} \cdot H_{kv} \cdot d \cdot b}{\text{achieved bandwidth}}
$$

The step reads the weights once, for any batch size. Take a Llama-3-8B-shaped model in bf16 on an H100 at 3.35 TB/s. It has 8.03 B parameters and 16.06 GB of weights. It has 128 KB of KV per token, from the formula of the [KV-cache primer](../kv-cache/kv-cache-primer.md). The weights alone take 4.79 ms per step:

| Batch × context | Tokens in batch | KV bytes | KV read time | Attention share of the step's bytes |
|---|---|---|---|---|
| 1 × 2k | 2,048 | 0.27 GB | 0.08 ms | 1.6% |
| 1 × 32k | 32,768 | 4.29 GB | 1.28 ms | 21% |
| 1 × 128k | 131,072 | 17.2 GB | 5.13 ms | 52% |
| 32 × 2k | 65,536 | 8.59 GB | 2.56 ms | 35% |
| 32 × 8k | 262,144 | 34.4 GB | 10.26 ms | 68% |
| 8 × 32k | 262,144 | 34.4 GB | 10.26 ms | 68% |

The share column is the same on all GPUs, because both terms divide by the same bandwidth. On an L4 at 0.30 TB/s, each time is 11.2× longer. Also, only the first two rows fit next to 16 GB of bf16 weights in 24 GB. KV bytes equal weight bytes at approximately 122,500 cached tokens per batch, that is, approximately 3,800 per sequence at batch 32. After that point, KV compression (GQA, MLA, FP8 KV) and paging efficiency change decode throughput more than any change to the matrix multiplies. [The capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md) shows how to calculate TPOT from this (`fa_calculators.decode_attention_bytes()`).

---

## 7. Paged KV and serving kernels

[The paged-attention primer](../paged-attention/paged-attention-primer.md) explains why serving engines keep the KV cache in fixed-size blocks. This section is about the effect of these blocks on the attention kernel. It also shows how two libraries and one engine connect them to the kernel. [The serving-engine topic (04)](../serving-engine/) is about the engine around the kernel (scheduler, KV manager, continuous batching).

### 7.1 What the kernel sees

A paged cache gives the kernel a pool and an indirection table, not a tensor for each sequence:

```
k_cache, v_cache : (num_blocks, block_size, H_kv, d)        one pool per layer, shared by all sequences
block_table      : (batch, max_blocks_per_seq) int32        logical block -> physical block, per sequence
seqused_k        : (batch,) int32                           valid tokens per sequence

logical token t of sequence b  ->  physical block  block_table[b][t // block_size],  row  t % block_size
```

The translation occurs in the main loop, for each K/V tile. In FA2's split-KV kernel (`compute_attn_1rowblock_splitkv`), it is:

```
block_table_idx    = n_block * kBlockN / page_block_size
block_table_offset = n_block * kBlockN - block_table_idx * page_block_size
K tile pointer     = k_ptr + block_table[block_table_idx] * k_batch_stride + block_table_offset * k_row_stride
```

The page size puts limits on the load path. In FA2's public `flash_attn_with_kvcache`, `page_block_size` must be a multiple of 256, thus a K/V tile never goes across two pages. FA3 accepts any page size ("page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.)"). But with small pages, the TMA's rectangular tile loads do not work. Thus FA3 changes to per-row asynchronous copies (`paged_kv_non_TMA`), which also make the tile smaller (section 5.5). The FlashAttention backend of vLLM accepts block sizes that are multiples of 16, through its own build of the kernels (`vllm.vllm_flash_attn`).

The paged-attention primer gives 20–26% kernel overhead for the original vLLM paged kernel. The cost for a given kernel depends on how contiguous the K/V rows of each page are. This is a layout decision (vLLM's layout is at the end of section 7.4).

### 7.2 FlashAttention's serving APIs

- **`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ..., block_table=None)`**. This function is for a ragged batch, packed along the token axis. `q` is `(total_q, H, d)`, and `cu_seqlens_q` holds the prefix sums of the per-sequence lengths (section 8.2). With `block_table`, K and V come from pages.
- **`flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None, ..., num_splits=0, return_softmax_lse=False)`**. This is the decode entry point. In one kernel, it appends the new `k, v` of the step into the cache in place (an option), applies the rotary embedding, and attends. `num_splits=0` means "use the heuristic" (section 6.3). It has no backward.
- **FA3 additions** (`hopper/flash_attn_interface.py`): `page_table`, and `cu_seqlens_q` and `max_seqlen_q` for multi-token queries against a cache (chunked prefill, speculative verification). There are also `qv` for MLA (section 8.1) and the FP8 `q_descale`, `k_descale`, `v_descale`. `scheduler_metadata` comes from `get_scheduler_metadata(...)`. This function calculates the split and tile schedule in advance, once per batch, so that all layers can use it. Then there are `pack_gqa` and `sm_margin` ("Can be tuned if some SMs are used for communication"). Last, `flash_attn_combine(out_partial, lse_partial)` is the split-KV reduction as a separate call.

For serving, know one semantic rule. Since v2.1, when $N_q \ne N_k$, the kernel aligns the causal mask to the **bottom-right** corner. For $N_q$ = 2, $N_k$ = 5, the permitted pattern is `1 1 1 1 0 / 1 1 1 1 1`. A chunk of new tokens that you append to a cache needs exactly this. Each new token sees the full cache and the new tokens before it.

### 7.3 FlashInfer: plan once per batch, run once per layer

FlashInfer (Ye et al., 2025) is a kernel library. SGLang, vLLM, TensorRT-LLM and others use it (README). Its decode and prefill wrappers divide the work into two parts:

```
wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")     # 128 MB workspace recommended
wrapper.plan(kv_page_indptr, kv_page_indices, kv_last_page_len,          # once per engine step (host)
             num_qo_heads, num_kv_heads, head_dim, page_size, ...)
for layer in layers:
    o = wrapper.run(q[layer], kv_cache[layer])                           # once per layer (device)
```

- **The page table is in CSR form**: `indptr` (`batch + 1` offsets), `indices` (all page ids, one after the other) and `last_page_len`. The last array holds the fill level of the last page of each sequence. vLLM and FlashAttention use a padded 2D block table instead.
- **`plan()` does the scheduling on the host**: from the lengths, it decides the split-KV partition and how to balance the load. It writes auxiliary arrays into the workspace. The workspace also holds "intermediate attention results in the split-k algorithm". The lengths are the same for all layers, thus one plan is sufficient for all of them. The example in the docstring makes one plan and runs 32 layers.
- **CUDA graphs**: `plan()` cannot run inside a CUDA graph or `torch.compile`. With `use_cuda_graph=True`, the wrapper uses preallocated index buffers, and the batch size cannot change.
- **What is behind `run()`**: decode kernels (with `use_tensor_cores=True` for large GQA groups), prefill/append kernels and MLA. There are also cascade attention for shared prefixes, POD-attention (prefill and decode fused in one launch) and block-sparse attention. The selectable backends are `auto`, `fa2`, `fa3`, `trtllm-gen`, `cute-dsl`, `cudnn` (decode wrapper docstring). The README lists SM 7.5 (T4) through Blackwell. But it also says that not every feature exists on every architecture.
- **Determinism knob**: `fixed_split_size` sets the split partition in pages to a constant value. The docstring says that this "will lead to deterministic softmax score reduction in the merge_states kernel, and therefore batch-size invariant outputs" (section 9.3).

The design point is this. The split decision depends only on the lengths in the batch. Thus the host calculates it once per step, which divides its cost across all layers. The per-layer launches also stay low-cost, and a CUDA graph can capture them.

### 7.4 How vLLM picks a backend and builds attention metadata

The sources of this section are `vllm/platforms/cuda.py`, `vllm/v1/attention/selector.py`, `vllm/v1/attention/backends/flash_attn.py` and `fa_utils.py` on `main`, fetched 2026-09-26. This code changes frequently. [The vLLM internals primer, section 6](../vllm-internals/vllm-internals-primer.md#6-attention-backends) has the engine-side wiring: the three backend classes, the per-GPU selection table, and when cascade is on. This section keeps the view from the kernel: which kernel a layer gets, and what arrives at its arguments.

**Selection.** For each attention layer, vLLM goes through a priority list. It takes the first backend whose `validate_configuration()` accepts the layer (head size, dtype, KV-cache dtype, block size, compute capability, sinks, sliding window, MLA, ...). An explicit `attention_config.backend` has priority over the list.

```
non-MLA attention on CUDA
  SM 10.x (B200, GB200), causal:   FLASHINFER > FLASH_ATTN > TRITON_ATTN > FLEX_ATTENTION > TURBOQUANT
  everything else (incl. SM 12.x): FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX_ATTENTION > TURBOQUANT
MLA on SM 9.0:                     FLASH_ATTN_MLA > FLASHMLA > FLASHINFER_MLA > TRITON_MLA > sparse variants

FlashAttention version:  SM 9.0 -> FA3;  SM 10.x -> FA4 if installed;  otherwise FA2 (SM 8.x, SM 12.x);
                         ALiBi forces FA2
FLASH_ATTN accepts:      fp16/bf16; KV cache auto/fp16/bf16, and fp8/fp8_e4m3 only with FA3 on SM 9.0
                         or FA4 (SM 10.x, or SM 9.0 at head size 512); block size multiple of 16;
                         head size multiple of 8 and <= 256 (<= 512 with FA4); compute capability >= 8.0
FLASHINFER accepts:      compute capability 8.0 to 12.1; KV cache incl. fp8/fp8_e4m3/fp8_e5m2
```

On the hardware tiers of this repository, the result is:

- A T4 (SM 7.5) gets `TRITON_ATTN`, for two reasons. FlashAttention needs SM 8.0. Also, vLLM currently sets a floor of SM 8.0 for FlashInfer ("FlashInfer supports SM75+, but is currently broken on SM75 (Turing) ... Temporarily raise the floor to SM80").
- An L4 (SM 8.9) gets `FLASH_ATTN`, which runs FA2 kernels.
- An H100 gets `FLASH_ATTN`, which runs FA3.
- A B200 gets `FLASHINFER` for causal decoder layers.
- An SM 12.x Blackwell card (RTX PRO 6000 Blackwell, RTX 50-series) gets `FLASH_ATTN`, which runs FA2.

The KV-cache dtype changes this. vLLM's FA2 path has no FP8 KV cache. Thus, with `kv_cache_dtype=fp8`, an L4 or an A100 fails `FLASH_ATTN`'s validation. The validation says: "FP8 KV cache requires FA3 on SM90, FA4 with head_size=512 on SM90, or FA4 on SM100". The GPU then gets `FLASHINFER`, which vLLM's CUDA build includes as a dependency. An H100 keeps `FLASH_ATTN` with FA3.

**Metadata.** Once per engine step, `FlashAttentionMetadataBuilder.build()` changes the scheduler's batch into the data that the kernel needs:

| Field | Meaning | Passed to the kernel as |
|---|---|---|
| `query_start_loc` | prefix sums of this step's query lengths (prefill chunks > 1, decodes = 1) | `cu_seqlens_q` |
| `seq_lens` | total context per request (cached + new) | `seqused_k` |
| `max_query_len`, `max_seq_len` | bounds for the grid | `max_seqlen_q`, `max_seqlen_k` |
| `block_table` | per-request page ids | `block_table` |
| `slot_mapping` | physical slot for each new token's K/V | used by the cache write |
| `scheduler_metadata` | FA3 only: schedule precomputed ahead of time (`aot_schedule`) with `get_scheduler_metadata` | `scheduler_metadata` |
| `max_num_splits` | set only when the step runs under a full CUDA graph | `num_splits` |
| `use_cascade`, `common_prefix_len`, ... | a prefix shared by every request in the batch | two kernel calls + merge |

Two details from the source are useful to quote in a review. On splits, the source says: "Setting num_splits > 1 may increase the memory usage, because the intermediate buffers of size [num_splits, num_heads, num_tokens, head_size] are allocated. Therefore, we only set num_splits when using cuda graphs." Also, if you set `VLLM_BATCH_INVARIANT`, then `max_num_splits = 1` (section 9.3).

For cascade attention, the kernel attends the shared prefix once for all query tokens of the batch (batch 1, non-causal). It attends the per-request suffixes separately (causal). Then `merge_attn_states` combines the two with their LSEs. This is section 2.3's merge. vLLM uses it so that it does not read the KV of a shared system prompt once per request.

**Forward**, per layer:

```
reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
                                            # scatter this step's new K/V rows into their pages
flash_attn_varlen_func(q, key_cache, value_cache, out,
                       cu_seqlens_q=query_start_loc, max_seqlen_q, seqused_k=seq_lens, max_seqlen_k,
                       softmax_scale, causal, window_size, block_table, softcap,
                       scheduler_metadata, q_descale, k_descale, v_descale, num_splits, ...)
                                            # one launch for the whole mixed batch: prefill chunks and decodes
```

The per-layer cache is one tensor of shape `[num_blocks, num_kv_heads, block_size, 2 × head_size]`. vLLM divides it into strided K and V views (`kv_cache.transpose(1, 2).split(head_size, dim=-1)`). This is why the kernels accept any strides, if the last dimension is contiguous.

---

## 8. Variants the kernel must support, and what they cost

An attention kernel in production is a family of template instantiations. Each variant in the table is a branch inside the tile loop or a change to the loop bounds. The cost column tells which resource it uses.

| Variant | What changes in the kernel | Cost | Notes from the sources |
|---|---|---|---|
| Causal | The loop stops at the diagonal. The kernel masks the diagonal tiles (section 4.4). | It saves ≈ 50% of FLOPs at large $N$. It causes load imbalance. | bottom-right aligned when $N_q \ne N_k$ (v2.1) |
| Sliding window `(left, right)` | `n_block_min` and `n_block_max` bound the loop | work ∝ $N \cdot W$ instead of $N^2/2$: at $N$ = 32k, $W$ = 4,096, 128 × 64 tiles, 15,840 tiles instead of 65,792 (24%) | FA2 v2.3 (Mistral 7B), FA3, FlashInfer `window_left` |
| ALiBi | adds $-\text{slope} \cdot \lvert i + N_k - N_q - j \rvert$ to each score | one FMA per score, nothing extra to load | vLLM changes to FA2 when ALiBi is on ("Cannot use FA version 3 with ALiBi") |
| Softcapping | $s \leftarrow c \cdot \tanh(s/c)$ before the softmax (Gemma-2, Grok) | one `tanh` per score on the same MUFU as `exp2`, the unit that section 4.3 shows is already the bottleneck | FA2 v2.6, FA3 (smaller FP8 tiles with softcap + local), FlashInfer `logits_soft_cap` |
| Dropout (training) | The kernel makes Philox random numbers for each score. The backward makes them again from the saved seed and offset (`rng_state`). | RNG instructions for each score. The kernel stores no mask. | There is no split-KV with dropout ("SplitKV is not implemented for dropout"). |
| MQA / GQA | K/V head index = `bidh / h_h_k_ratio` | Free in prefill. Decode gains come from packing (section 6.4). | query heads must be a multiple of KV heads |
| Head dim 64 / 128 / 256 | SMEM per tile ∝ $d$, accumulator registers ∝ $B_r \cdot d$ | A larger $d$ gives smaller tiles and fewer CTAs per SM. FA2 on H100: 128 × 64 at $d$ = 128, 64 × 64 at $d$ = 256. FA3: 128 × 176 against 128 × 80. The per-tile intensity $2B_r/b$ does not depend on $d$. | FA2 supports $d$ ≤ 256. vLLM's FA backend needs `d % 8 == 0`. |
| Attention sinks | an extra learnable per-head logit in the denominator (gpt-oss) | one extra term in $l$ per row | vLLM's FA call passes it as `s_aux`. vLLM needs SM 9.0+ for sinks with FA. |
| MLA | section 8.1 | brings decode to the ridge (242 FLOP/B in bf16 against 295 on an H100) | FA3 `qv`, FlashMLA, FlashInfer MLA |
| Ragged batches | section 8.2 | No padding FLOPs. Tile quantization stays. | the varlen APIs |
| Across GPUs | section 8.4 | communication to hide | ring attention, context parallelism |
| Sparse masks | section 8.5 | tiles skipped by metadata | block-sparse kernels, FlexAttention |

### 8.1 MLA and the absorbed-weight decode trick

Multi-head latent attention (DeepSeek-V2 and V3) caches two things per token and layer. (The dimensions in this section are DeepSeek-V3's, **(verify)**.) The first is a latent $c \in \mathbb{R}^{512}$. The second is one rotary key $k_{\text{rope}} \in \mathbb{R}^{64}$ that all 128 heads share. That is 576 values, or 1,152 bytes in bf16. The equivalent multi-head cache (128 heads × (192 key + 128 value dims)) is 80 KB per token per layer, 71× more (`fa_calculators.mla_cache_ratio()`).

The ratio is different if you do not count the 64 rotary dims in the key of each head. Without them (128-dim keys, as wide as the values), it is 57×. [The vLLM internals primer](../vllm-internals/vllm-internals-primer.md#64-gqa-and-mla-change-the-bytes-not-the-paging) uses this figure. But DeepSeek-V3's per-head key is 192 dims, thus 71× is the like-for-like number.

Each head calculates its key and value from the latent: $k_h = [W_{UK}^h c \,;\, k_{\text{rope}}]$ and $v_h = W_{UV}^h c$, with $W_{UK}^h, W_{UV}^h \in \mathbb{R}^{128 \times 512}$.

If decode uses these formulas directly, it must calculate the K and V of every cached token for every head, at every step. The trick is to move the up-projections to the query and output side. On that side, there is one token instead of $L$:

$$
\begin{aligned}
\text{score}_{h,j} &= q_{\text{nope}}^h \cdot (W_{UK}^h c_j) + q_{\text{rope}}^h \cdot k_{\text{rope},j} \\
&= \big((W_{UK}^h)^{\top} q_{\text{nope}}^h\big) \cdot c_j + q_{\text{rope}}^h \cdot k_{\text{rope},j}
&& \text{define } \tilde{q}_h = (W_{UK}^h)^{\top} q_{\text{nope}}^h \text{ (512-dim)} \\
\text{out}_h &= \sum_j p_{h,j} (W_{UV}^h c_j) = W_{UV}^h \Big(\sum_j p_{h,j} c_j\Big)
&& \text{attend over } c_j \text{ directly, then project} \\
&&& (W_{UV}^h \text{ can be folded into } W_O)
\end{aligned}
$$

Decode becomes multi-query attention with 128 query heads and a single shared "KV head". The key of this head is $[c \,;\, k_{\text{rope}}]$ (576-dim). Its value is $c$ itself (512-dim, the same memory as the first 512 key dimensions).

FA3 uses exactly this form. `q` carries the 64 rotary dims, `qv` the 512 absorbed dims, `k` is $k_{\text{rope}}$ and `v` is $c$. The kernel adds a second GEMM into the score accumulator (`HasQv`: $S = q \cdot k^{\top} + \mathit{qv} \cdot v^{\top}$). `tile_size_fwd_sm90` has a separate entry for `headdim = 64, headdim_v = 512`. When you pass `qv`, the default softmax scale is $(64 + 512)^{-1/2}$. Model code passes its own scale.

The arithmetic intensity is what makes MLA decode different. Per cached token, there are `2·576·128` FLOPs for the scores and `2·512·128` for the output. That is 278,528 FLOPs over 1,152 bytes: **242 FLOP/B in bf16**, and 484 with an FP8 latent and bf16 compute (`fa_calculators.mla_decode_intensity()`). Against the H100's bf16 ridge of 295, the bf16 case is at 82% of the ridge, just on the memory side of it. The FP8-latent case is above it (with FP8 matrix multiplies, the ridge also doubles, to approximately 590 at the dense FP8 peak, **(verify)**).

In both cases, absorbed MLA decode is at the ridge. It is not far below the ridge, as MHA or GQA decode is (1 to 8 FLOP/B). Thus kernel authors adjust FlashMLA and FA3's MLA path like GEMMs, and give their throughput in TFLOP/s and also in GB/s (section 10.1).

That intensity assumes that all 128 heads run on one GPU. Tensor parallelism over heads divides it by the TP degree (TP = 8 leaves 16 heads per GPU: 30 FLOP/B). Also, because all heads share the latent, TP cannot divide the latent by head. Thus each TP rank also holds the full latent cache. Both are reasons why engines often serve MLA models with data-parallel attention **(verify for a given engine)**. In data-parallel attention, each GPU holds all heads, and different GPUs get different requests.

Prefill usually does not absorb the weights. With $N$ query tokens, attention in the latent space costs `576 + 512` dims per head for scores and outputs, instead of `192 + 128`. That is 3.4× the quadratic FLOPs (`fa_calculators.mla_prefill_absorbed_ratio()`). But the calculation of K and V per head from the latent is linear in $N$, and all queries share its cost.

### 8.2 Ragged batches (varlen)

Serving batches have unequal lengths. If you pad them to the longest, the waste of FLOPs is quadratic. Lengths `[100, 3,000, 500]` padded to 3,000 cost `3 × 3,000² = 27.0 M` score pairs, against `100² + 3,000² + 500² = 9.26 M` actual, 2.9×. The varlen APIs pack the sequences one after the other and pass `cu_seqlens`.

The longest sequence sets the size of the grid. Each CTA calculates the real bounds of its sequence. If its Q block is past the end, the CTA exits immediately (`if (m_block * kBlockM >= binfo.actual_seqlen_q) return;` in FA2). The waste that stays is tile quantization: a 100-token sequence still uses a 128-row tile, which is 78% useful (`fa_calculators.padding_waste()`). FA3 packs GQA groups for all varlen batches to improve exactly this (section 6.4).

### 8.3 Head dimension, briefly

There are two reasons why $d$ = 256 runs slower per FLOP than $d$ = 128. First, the Q, K and V tiles increase with $d$. Thus the same SMEM holds fewer rows and fewer CTAs per SM (FA2 on H100 decreases to 64 × 64 to keep two CTAs per SM). Second, the fp32 output accumulator $B_r \times d$ and the score tile use the same limited registers. The per-tile intensity ($B_r$ FLOP/B in bf16) does not improve with $d$.

FA2's Python wrapper pads head dimensions that are not multiples of 8 (`torch.nn.functional.pad` to the next multiple of 8), and its C++ API rejects them. In vLLM, `d % 8 == 0` is necessary.

### 8.4 Across GPUs: ring attention and context parallelism

When one sequence does not fit on one GPU, shard it. **Ring attention** (Liu, Zaharia, Abbeel, 2023) gives each of $P$ GPUs ${N/P}$ tokens of $Q$, $K$ and $V$. In each of $P$ steps, a GPU attends its $Q$ shard to its current K/V shard. It merges the result into its ${(m, l, o)}$ state (section 2.3). While it calculates, it sends the K/V shard to its neighbour. The computation hides the communication when the compute time per step is more than the transfer time per step, per head:

$$
\frac{4 \cdot (N/P)^2 \cdot d}{F} \;\ge\; \frac{2 \cdot (N/P) \cdot d \cdot b}{W} \qquad\Longrightarrow\qquad N/P \;\ge\; \frac{b \cdot F}{2 \cdot W}
$$

For an H100 over NVLink (≈ 450 GB/s per direction, **(verify)**), ${N/P}$ ≥ 2,199 tokens at the 989 TFLOP/s peak. It is 1,333 at a more realistic 600 TFLOP/s. Over a 400 Gb/s (50 GB/s) network link, it is 12,000 to 19,800 tokens per GPU (`fa_calculators.ring_min_tokens_per_gpu()`). With a causal mask, the work of the steps is not equal (the queries of the first shard see almost nothing). Striped and zigzag partitions correct this, because they interleave the tokens across GPUs.

The alternative is all-to-all over heads (DeepSpeed-Ulysses). It gives each GPU all tokens for ${H/P}$ heads, and needs $P \le H$. At serving time, vLLM has decode context parallelism. It shards the KV cache across ranks and merges the partial outputs with their LSEs (`cp_lse_ag_out_rs`, `dcp_a2a_lse_reduce` in its FlashAttention backend).

### 8.5 Sparse and block-sparse attention, and FlexAttention

**Block-sparse attention** skips tiles by a block mask. The FA1 paper's block-sparse variant has IO complexity $\Theta(Nd + N^2 d^2 M^{-1} s)$, with $s$ the fraction of non-zero blocks **(verify)**. The cost model is simple: tiles visited × cost per tile, plus the metadata to find them. Sparsity finer than a tile saves nothing, because the kernel loads and multiplies the tile in all cases. FA4 has block-sparse paths in its CuTe kernels, and FlashInfer has block-sparse and variable block-sparse attention. Also, vLLM has separate backends for DeepSeek-style sparse MLA (`FLASHMLA_SPARSE`, `FLASHINFER_MLA_SPARSE`).

**FlexAttention** is from PyTorch (Dong et al., 2024). It removes the need for a new CUDA kernel for each variant. In Python, you write a `score_mod(score, b, h, q_idx, kv_idx)`, a `mask_mod(b, h, q_idx, kv_idx)`, or both.

`create_block_mask` calculates the mask once per block. For each Q block, it records which K/V blocks are empty (skipped), partial (mask calculated per element) and full (no mask code). This is the same three-way split as FA2's masked and unmasked loop phases, but the data controls it. `torch.compile` inlines the functions into a Triton flash kernel.

PyTorch reported approximately 90% of FlashAttention-2's performance in the forward pass and 85% in the backward pass **(verify)**. vLLM has a `FLEX_ATTENTION` backend. It also passes `mask_mod` functions to FA4 for masks that the built-in flags cannot describe (multimodal bidirectional prefixes, for example).

---

## 9. Numerics

### 9.1 bf16 in, fp32 accumulate

The tensor cores take bf16 or fp16 operands and accumulate in fp32. Thus $S$ and the output accumulator are fp32 in registers (or TMEM). Two roundings to the input dtype occur. The kernel converts $P$ before each $P \cdot V$ multiply (`convert_type<Element>(acc_s)` in FA2), and $O$ once at the end. The first rounding is the main source of error.

bf16 keeps 8 significant bits, a unit roundoff of $2^{-8}$ ≈ 0.39% per probability. fp16 keeps 11 bits, $2^{-11}$ ≈ 0.05%. For attention, fp16 is the more accurate input format when its range is sufficient.

Thus a kernel cannot agree with an fp32 reference bit for bit. Do not expect it to agree with another kernel either, because different tile sizes add in different orders. Use FlashAttention's own test criterion. Compare the kernel and a plain PyTorch implementation *in the same dtype* against an fp32 reference. The maximum error of the kernel must be at most two times the error of the baseline. The README says: "the maximum numerical error of FlashAttention is at most twice the numerical error of a baseline implementation in Pytorch".

### 9.2 Folding the softmax scale

The kernel never applies $\tau$ to the scores as a separate multiply. It takes the max on the raw fp32 scores. Then it folds $\tau \cdot \log_2 e$ (and, for FP8, `q_descale · k_descale`) into the single FFMA that makes the exponent (section 2.4). This has two consequences for reviewers:

- The scale must be positive, or the max-before-scale order is not valid.
- A model that needs an unusual scale (MLA with absorbed dimensions, section 8.1) must pass it explicitly. The reason is that the default is $1/\sqrt{\texttt{head_dim}}$ of the tensor that the kernel sees, whatever that tensor is.

### 9.3 The log-sum-exp, empty rows, and determinism

The LSE is the interface between kernels. The backward needs it (section 3.5). Split-KV, cascade attention, ring attention and decode context parallelism all merge partial results through it (section 2.3). In practice, three details cause problems:

- **Units.** FA2 and FA3 use the natural log. The Triton tutorial uses base 2. If you mix them, the result is incorrect by a factor of $\ln 2$.
- **Empty rows.** A row with no permitted key (a fully masked row, an empty split) has $l$ = 0. FA2 reports $\mathrm{LSE} = +\infty$ ($-\infty$ in split mode), and FA3 reports $-\infty$. vLLM's `merge_attn_states` maps $+\infty$ to $-\infty$. When both sides are empty, it returns 0 ("FA2 and FA3 have different behavior for when the sum-exp is 0").
- **Batch invariance.** The forward pass is deterministic for one configuration, but the KV split count depends on the batch size and the SM count (section 6.3). A different split count adds the partial results in a different order. Thus the same request can give slightly different logits, as a function of the other requests in its batch. The solutions keep the reduction order constant: vLLM sets `max_num_splits = 1` under `VLLM_BATCH_INVARIANT`, and FlashInfer's `fixed_split_size` sets the partition in pages to a constant value. (Its docstring cites the "defeating nondeterminism in LLM inference" analysis.) The backward has a second source: `atomicAdd` into $\mathit{dQ}$ (section 3.5), which `deterministic=True` removes.

### 9.4 FP8: error sources and mitigations

Section 6 of the notebook measures each source on synthetic data (seeded, numpy, with `fa_calculators.round_to_e4m3()`, which emulates e4m3fn):

| Error source | Mechanism | Measured in the notebook | Mitigation |
|---|---|---|---|
| Mantissa rounding of Q, K, V | e4m3 keeps 3 mantissa bits: up to 6.25% relative error per element | $QK^{\top}$ relative error 3.6% with per-tensor scales (N = 256, d = 128) | None within FP8. Keep the softmax, $l$ and $O$ in fp32. Measure end-to-end quality. |
| An outlier channel | One large channel sets a per-tensor scale. Thus, with integer formats, every other value loses precision. Also, in any format, a dot product in which one product is most of the sum gets the full relative error of that product. | Channel 7 × 20 in K only, with a random Hadamard rotation: INT8 6.1% to 1.3%, INT4 52% to 24%, e4m3 3.6% to 3.7% (unchanged). The same channel × 20 in both Q and K: e4m3 3.6% to 0.46%, INT8 1.1% to 0.14%, INT4 17% to 2.4%. | Incoherent processing (rotate Q and K). For e4m3, it helps when the outlier channels of Q and K align. For integer formats, it helps in all cases. |
| A few outlier tokens | a few large rows set a per-tensor scale | 4 of 256 K rows × 50. INT8 error on the scores of the other rows: 33% with one scale per tensor, 1.3% with one scale per 64 rows. For e4m3: 3.7% to 3.8% (unchanged). | Block quantization (one scale per tile, as in FA3) or per-head scales. This is decisive for integer formats. For e4m3, it is marginal, unless the range is more than its ~2^15 span of normal values. |
| Small probabilities in $P$ | $P$ ≤ 1. e4m3's smallest subnormal is $2^{-9}$. | A 4,096-key row with N(0, 2²) scores: 61.9% of probabilities flush to zero, and the row loses 2.7% of the probability mass. With FA3's $2^8$ offset, 0.7% flush, and the output error decreases from 2.2% to 1.6%. | Scale $P$ by $2^8$ before conversion (`Max_offset`). Accumulate $l$ in fp32 from unrounded values. |
| Scale choice | stale or per-tensor scales clip or waste range | not simulated | calibrated KV scales (vLLM `k_scale`, `v_scale`), per-(batch, head) descales as in FA3 |

Why does the rotation help a float format only when the outliers align? The rounding error of each element is relative. Thus the error of $q \cdot k = \sum q_i k_i$ has a variance proportional to $\sum q_i^2 k_i^2$. If one channel is large in both $q$ and $k$, that sum is almost only one term. Then $q \cdot k$ carries the full relative error of one rounded product.

The rotation does not change $q \cdot k$, but it spreads the product over all $d$ channels. The independent rounding errors of these channels partly cancel (approximately 8× less error here, at $d$ = 128). With the outlier in $K$ alone, the products were already spread, and the rotation changes nothing.

The alignment of real Q and K outliers is a property of the model. For rotary-embedding models, reports say that large values collect in the same dimensions of Q and K **(verify)**. Thus, measure it on the model's own activations before you decide.

Two review points follow. First, "FP8 attention" can mean an FP8 KV cache with bf16 compute (a bandwidth gain for decode, section 6.1). It can also mean FP8 matrix multiplies (a compute gain for prefill, section 5.6). The two have different error budgets. Second, a kernel-level error metric is not a quality metric. Accept FP8 attention on the basis of an end-to-end evaluation on the workload's own data.

---

## 10. Measuring attention kernels

### 10.1 Count by a stated convention

- **Prefill / training:** the count is $4 \cdot N_q \cdot N_k \cdot d$ FLOPs per head for the forward, and half of that for causal. The backward is ×2.5, and forward plus backward is ×3.5 (section 1.1). The Triton tutorial's `bench_flash_attention` uses exactly this convention. The convention does not count softmax FLOPs. Some tools count causal attention at full cost, or include softmax work. You cannot compare a number that does not give its convention.
- **Decode (MHA and GQA, group size up to 8):** report bandwidth. The bytes are the K/V that the kernel actually read ($2 \cdot L \cdot d \cdot b$ per KV head per sequence), plus Q and O. Divide the bytes by the time. Compare the result with the HBM peak. For such a kernel, TFLOP/s does not tell much (the intensity is 1 to 8 FLOP/B, section 6.1). Absorbed-MLA decode is the exception: at 242 to 484 FLOP/B, it is at the ridge (section 8.1). Thus report it on both axes: TFLOP/s against the dense peak, and GB/s against HBM bandwidth.

### 10.2 Which side of the roofline to report

A prefill kernel is on the compute side. Thus the honest figure is TFLOP/s as a fraction of the dtype's dense peak (989 bf16 on an H100, not the 1,979 sparse headline). An MHA or GQA decode kernel is on the bandwidth side. Thus the figure is GB/s as a fraction of HBM bandwidth (absorbed MLA decode, at the ridge, gets both).

For split-KV and paged kernels, report two byte counts. One is the bytes that the kernel had to read. The other is the bytes that it did read (partials, block tables, padding).

### 10.3 Procedure

1. **Correctness first.** Compare against an fp32 reference with the criterion of section 9.1. Do this for causal and non-causal, odd lengths ($N$ not a multiple of the tile), several head dims and GQA. Also compare the LSE, if the kernel returns it.
2. **Confirm which kernel ran.** `torch.nn.functional.scaled_dot_product_attention` has a silent fallback to another backend. It uses the fallback when the inputs do not agree with a constraint (dtype, head dim, mask type, GPU generation). Force the backend (`torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`). Then the call raises an error, and does not use the fallback. Also examine the kernel names in a profiler trace. In vLLM, the startup log names the selected attention backend.
3. **Warm up.** The first calls pay the cost of lazy initialization, JIT compilation, autotuning and the caching allocator.
4. **Time on the device.** Use CUDA events or `torch.utils.benchmark.Timer` (it synchronizes and repeats). Report the median and the spread, not the minimum.
5. **Mind the L2.** A decode-sized problem can stay in L2 from the previous repetition. One layer of one 4k sequence of a Llama-3-8B-shaped model is 16.8 MB. An H100 has 50 MB of L2. In that case, the measurement shows more than the HBM bandwidth. Flush the L2 between repetitions (`triton.testing.do_bench` does this by default **(verify)**). Or use a working set larger than L2.
6. **Mind launch overhead.** A decode attention kernel can run in tens of microseconds. That is the same order as the launch and Python overhead. For this reason, serving engines capture decode steps in CUDA graphs. Measure the kernel in the same way.
7. **Clocks and power.** Long runs throttle. Lock the clocks for A/B comparisons, or report the clock that you observed.
8. **Sweep the shapes that matter**: $N$, $d$, batch × heads (does the grid fill the GPU?), causal, GQA group, page size.

### 10.4 A minimal honest benchmark (T1)

This code did not run in this environment, because it has no GPU. Run it on any CUDA GPU with PyTorch 2.3 or later.

```python
import math, torch, torch.nn.functional as F
from torch.utils import benchmark
from torch.nn.attention import sdpa_kernel, SDPBackend

def naive(q, k, v):                                    # plain PyTorch in the input dtype: the error baseline
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    mask = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1)
    return torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1) @ v

def flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):     # raise rather than silently fall back
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 1. correctness on a small, odd shape: at most 2x the same-dtype baseline's error against fp32
q, k, v = (torch.randn(1, 4, 1000, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
ref = naive(q.float(), k.float(), v.float())
assert (flash(q, k, v).float() - ref).abs().max() <= 2 * (naive(q, k, v).float() - ref).abs().max()

# 2. timing on the shape that matters
B, H, N, D = 4, 32, 4096, 128
q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
flash(q, k, v)                                         # warm-up
m = benchmark.Timer(stmt="flash(q, k, v)", globals=dict(flash=flash, q=q, k=k, v=v)).blocked_autorange(min_run_time=2.0)
flops = 4 * B * H * N * N * D * 0.5                    # causal convention
print(f"median {m.median * 1e3:.3f} ms, IQR {m.iqr * 1e3:.3f} ms, {flops / m.median / 1e12:.0f} TFLOP/s")
```

On a T4, the `sdpa_kernel` context raises an error, because PyTorch's flash backend needs SM 8.0 or newer **(verify)**. That is the reason to force the backend. On a T4, use `SDPBackend.EFFICIENT_ATTENTION`, in fp16.

### 10.5 What to expect, by GPU class

The table is for the forward pass, $d$ = 128 and long sequences, with the dense peak in parentheses. Each value is a citation or an explicit assumption. Measure before you use any of them.

| GPU | Kernel | Expected | Basis |
|---|---|---|---|
| A100 80GB (312) | FA2 | 50–73% of peak, ≈ 156–228 TFLOP/s | FA2 paper **(verify)** |
| H100 SXM (989) | FA2 | ≈ 35%, ≈ 350 TFLOP/s | FA3 paper **(verify)** |
| H100 SXM | FA3 bf16 / FP8 | up to 740 TFLOP/s (75%) / near 1,200 | FA3 paper **(verify)** |
| B200 (2,250) | FA4 bf16 | up to 1,605 TFLOP/s (71%) | FA4 paper via the primer **(verify)** |
| L4 (121) | FA2 (sm89 tiles, section 4.5) | This page read no source. If it gets to the A100's 50–70% fraction, it gives 60–85 TFLOP/s. | assumption **(verify by measuring)** |
| T4 (65, fp16) | FA2 unsupported (README: Turing needs a separate project) | SDPA changes to its memory-efficient kernel | README |
| any | split-KV decode | bandwidth-bound: compare with a plain device-to-device copy on the same GPU as the practical ceiling | section 6 |

### 10.6 How attention's share of a step changes with context

**Prefill.** Per token and layer, the linear layers cost $2 \times \text{parameters}$ FLOPs. Causal attention costs $4 \times \text{context} \times H \times d \times 0.5$, on average over the prompt. The table gives the values for a Llama-3-8B-shaped layer (218.1 M parameters, 32 heads of 128) (`fa_calculators.prefill_attention_share()`):

| Prompt length | 2k | 8k | 32k | 128k |
|---|---|---|---|---|
| attention FLOPs / linear FLOPs | 0.04 | 0.15 | 0.62 | 2.46 |

The two are equal at 53,248 tokens. Attention usually runs at a lower fraction of peak than the large GEMMs. Thus its share of *time* gets to the crossover at a shorter prompt than its share of FLOPs.

**Decode.** As section 6.5 shows, the step reads the weights once and the KV cache once per cached token. Thus attention's share increases with batch × context. For the same model, KV bytes become larger than weight bytes at approximately 122,500 cached tokens per batch.

Both effects move long-context serving to the same point. At 32k tokens and more, attention, not the MLP, decides throughput, and KV size decides capacity ([the KV-cache primer](../kv-cache/kv-cache-primer.md), [the capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md)).

---

## 11. Implementing it yourself: an FA2 forward in Triton

[`flash_attention_minimal.py`](flash_attention_minimal.py) (part 5) is the complete algorithm in numpy. In [the practice notebook](notebooks/flash_attention_practice.ipynb), you write it yourself. This section changes it into a GPU kernel.

With Triton, you write the kernel at the level of tiles. A program instance works on a `BLOCK_M × BLOCK_N` tile, and the compiler controls the warps, the shared memory and the pipelining of loads. Thus the mapping from the algebra is direct. The upstream reference is the Triton tutorial `python/tutorials/06-fused-attention.py` (read for this page). On newer GPUs, the tutorial adds warp specialization and tensor descriptors (TMA).

### 11.1 The kernel

**Tier T1 for timing; T0 for correctness through Triton's interpreter (section 11.4).** The kernel did not run on a GPU in this repository.

[`test_triton_kernel_emulated.py`](test_triton_kernel_emulated.py) takes the code block of this section from this page. It runs the body of the code on a CPU, one program instance at a time. It runs the code against a numpy substitute for the `tl` operations that the code uses. It runs these cases:

- $N$ of 1, 64, 65, 200 and 300
- ragged tails
- `BLOCK_M` smaller and larger than `BLOCK_N`
- causal and not causal
- batch and head strides
- fp16 inputs.

The output must be within 5e-3 of an fp64 reference, and the LSE within 1e-3. The test also breaks three variants on purpose. In them, a mask hides the diagonal, a loop stops one tile early, or there is no rescale. Each of them must fail the same check.

This test examines the indexing, masking and algebra of the kernel, not Triton's compiler. Section 11.4 runs the real front end. The kernel takes inputs `(B, H, N, D)`, contiguous, in fp16 or bf16, with `D` a power of two.

The loads and stores use plain pointer arithmetic with masks. All Triton 2.x and 3.x releases and the interpreter accept this. Older versions of the tutorial used the block-pointer API (`tl.make_block_ptr`, `tl.advance`). Triton 3.8 deprecates this API, and Triton's `main` branch (fetched 2026-09-26) removes it. Tensor descriptors (`tl.make_tensor_descriptor`) replace it. The upstream tutorial now uses them to get access to the TMA.

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def attn_fwd(Q, K, V, O, LSE, sm_scale, N_CTX,
             stride_b, stride_h, stride_n,                       # shared by Q, K, V, O; last dim contiguous
             H: tl.constexpr, D: tl.constexpr,                   # D a power of two (tl.arange)
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)                                     # (1) which Q block
    pid_bh = tl.program_id(1)                                    # (2) which (batch, head)
    base = (pid_bh // H) * stride_b + (pid_bh % H) * stride_h
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)         # (3) Q block: loaded once, stays on chip

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)          # (4) identity state
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634                     # (5) fold log2(e)

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)            # (6) stop at the diagonal
    for start_n in range(0, hi, BLOCK_N):                        # (7) walk the K/V tiles
        offs_k = start_n + offs_n
        kt = tl.load(K + base + offs_k[None, :] * stride_n + offs_d[:, None],    # K tile as (D, BLOCK_N)
                     mask=offs_k[None, :] < N_CTX, other=0.0)
        s = tl.dot(q, kt) * qk_scale                              # (8) scores, fp32, log2 units
        valid = offs_k[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_k[None, :])
        s = tl.where(valid, s, float("-inf"))                    # (9) ragged tail and diagonal
        m_new = tl.maximum(m_i, tl.max(s, 1))                    # (10)
        alpha = tl.math.exp2(m_i - m_new)                        # (11)
        p = tl.math.exp2(s - m_new[:, None])                     # (12)
        l_i = l_i * alpha + tl.sum(p, 1)                         # (13)
        v = tl.load(V + base + offs_k[:, None] * stride_n + offs_d[None, :],
                    mask=offs_k[:, None] < N_CTX, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)    # (14) P rounded to the input dtype
        m_i = m_new

    acc = acc / l_i[:, None]                                     # (15) normalize once
    tl.store(LSE + pid_bh * N_CTX + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX)   # (16) base 2
    tl.store(O + base + offs_m[:, None] * stride_n + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v, causal=True, block_m=128, block_n=64, num_stages=2):
    B, H, N, D = q.shape
    assert q.stride() == k.stride() == v.stride() and q.stride(-1) == 1
    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, block_m), B * H)                      # the FA2 grid: Q blocks x (batch, head)
    attn_fwd[grid](q, k, v, o, lse, 1.0 / math.sqrt(D), N, q.stride(0), q.stride(1), q.stride(2),
                   H=H, D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
                   num_warps=4, num_stages=num_stages)
    return o, lse
```

### 11.2 Line by line

| Kernel line | What it is | Section | In `flash_attention_minimal.py` |
|---|---|---|---|
| (1) `pid_m = tl.program_id(0)` | the outer loop over Q blocks becomes the grid | 4.1 | `for i in range(0, N, block_q)` |
| (2) `pid_bh = tl.program_id(1)` | parallelism over batch and heads | 4.1 | one head only |
| (3) `q = tl.load(...)` once | Q block loaded once and kept on chip, rows past $N$ masked to zero | 4.1 | `Qi = Q[i:i + block_q]` |
| (4) `m_i, l_i, acc = −∞, 0, 0` | the merge identity | 2.3 | `mi`, `li`, `Oi` initialized |
| (5) `qk_scale = sm_scale · log2 e` | scale folded into the exponent | 2.4, 9.2 | `scale` and `np.exp` |
| (6) `hi` for `CAUSAL` | causal block skipping | 4.4 | `if causal and j > i + bq - 1: continue` |
| (7) `for start_n in range(0, hi, BLOCK_N)` | walk the K/V tiles. `offs_k` moves the K and V pointers. | 3.1 | `for j in range(0, N, block_kv)` |
| (8) `s = tl.dot(q, kt) * qk_scale` | first GEMM, fp32 accumulate | 1.1, 9.1 | `Sij = (Qi @ Kj.T) * scale` |
| (9) `tl.where(valid, s, −∞)` | diagonal tile mask and ragged tail | 4.4, 8.2 | `np.where(cols <= rows, Sij, -np.inf)` |
| (10) `m_new = max(m_i, rowmax(s))` | new reference max | 2.2 | `m_new = np.maximum(mi, Sij.max(axis=1))` |
| (11) `alpha = exp2(m_i − m_new)` | rescale factor for the old state | 2.2 | `alpha = np.exp(mi - m_new)` |
| (12) `p = exp2(s − m_new)` | probabilities relative to the new max | 2.2, 2.4 | `Pij = np.exp(Sij - m_new[:, None])` |
| (13) `l_i = l_i·alpha + rowsum(p)` | the denominator, updated for each tile | 2.2 | `li = alpha * li + Pij.sum(axis=1)` |
| (14) `acc = acc·alpha + dot(p.to(fp16/bf16), v)` | second GEMM. $P$ stays in registers, rounded to the input dtype. | 2.2, 4.2, 9.1 | `Oi = alpha[:, None] * Oi + Pij @ Vj` |
| (15) `acc / l_i` | normalize once, at the end | 4.3 | `O[i:i + bq] = Oi / li[:, None]` |
| (16) store `m_i + log2(l_i)` | LSE for the backward, base 2 | 2.4, 3.5 | `L[i:i + bq] = mi + np.log(li)` (natural log) |

This list shows what the Triton version adds to the numpy one, and what it still does not have:

- **Masking of ragged tails** (the last Q and K/V blocks when $N$ is not a multiple of the tile). The `mask=`/`other=` arguments of the loads and the `valid` mask on the scores do this.
- **No $-\infty$ guard is needed here**: the loop runs forward from key 0, and each row can see key 0. This includes the padded rows past $N$, which the kernel masks on the way out. Thus no row has a fully masked first tile. A kernel whose loop goes backwards from the diagonal (FA2, section 4.4) needs the guard of section 2.5. The same is true for a kernel that uses a sliding window, or that aligns the causal mask bottom-right with $N_q > N_k$.
- **Pipelining** comes from `num_stages`. On SM 8.0 and newer, the compiler multi-buffers the K and V loads. This is the `cp.async` scheme of section 4.5.
- **Left out**: the backward pass (section 4 of the notebook has the algorithm), variable-length batches, paged K/V, GQA packing, split-KV, FP8 and warp specialization. Each of them is a section of this page.

### 11.3 Testing it on a GPU (T1)

Do a test against an fp32 reference, with the criterion of section 9.1:

```python
q, k, v = (torch.randn(2, 8, 1000, 128, device="cuda", dtype=torch.float16) for _ in range(3))   # N not a tile multiple
o, lse = flash_attention(q, k, v, causal=True)
ref = naive(q.float(), k.float(), v.float())                     # naive() from section 10.4
assert (o.float() - ref).abs().max() <= 2 * (naive(q, k, v).float() - ref).abs().max()
s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(128)
s = s.masked_fill(torch.ones(1000, 1000, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
assert torch.allclose(lse.view(2, 8, 1000) * math.log(2), torch.logsumexp(s, dim=-1), atol=1e-3)   # base 2 -> natural
```

Then change `block_m`/`block_n`. The result must not depend on them, except for rounding. Also change `causal` and $N$ (1, a tile multiple, a tile multiple plus one). Then measure the time of the kernel with the procedure of section 10.3.

**Shared memory decides the tile on small GPUs.** The defaults (`BLOCK_M = 128`, `BLOCK_N = 64`, `D = 128`, two stages) put approximately 96 KB of Q, K and V tiles in shared memory (`fa_calculators.tile_smem_bytes(128, 64, 128, kv_stages=2)` = 98,304 B).

A T4 permits 64 KB per block. Thus on a T4, start at `block_m=64, block_n=32, num_stages=1` (32 KB of tiles), in fp16 (it has no bf16 tensor cores). Make sure that `tl.dot` uses its tensor cores there **(verify)**. An L4 permits approximately 99 KB per block, which is a tight fit for the defaults. If Triton raises `OutOfResources` for shared memory, divide the tile sizes by two.

### 11.4 Checking it on a CPU (T0) with Triton's interpreter

With `TRITON_INTERPRET=1`, `@triton.jit` kernels run on CPU tensors through numpy. This is slow and gives no performance information. But it is the real Triton front end, masks included. It needs `torch` and `triton` to be importable, which is true on a Colab CPU runtime **(verify)**. If not, run `pip install torch triton`. This code did not run in this repository, because it is not possible to install either package here.

```python
import os
os.environ["TRITON_INTERPRET"] = "1"          # must be set before triton is imported
import math, torch, triton, triton.language as tl
# ... paste attn_fwd and flash_attention from section 11.1 here ...

torch.manual_seed(0)
for N in (1, 64, 65, 200):                    # 1, a tile multiple, a tile multiple plus one, ragged
    for causal in (False, True):
        q, k, v = (torch.randn(1, 2, N, 64, dtype=torch.float16) for _ in range(3))
        o, lse = flash_attention(q, k, v, causal=causal, block_m=32, block_n=16)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(64)
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
        ref = torch.softmax(s, dim=-1) @ v.float()
        assert (o.float() - ref).abs().max() < 5e-3                     # fp16 P and O rounding
        assert torch.allclose(lse.view(1, 2, N) * math.log(2), torch.logsumexp(s, dim=-1), atol=1e-3)
print("forward pass and LSE match the reference on CPU")
```

This test finds indexing, masking and algebra errors. It tells nothing about speed, shared-memory limits or register pressure. Only a GPU run shows these (section 11.3).

---

## In a design review

### The two-minute walkthrough

"If you write attention the obvious way, it is memory-bound, and the softmax is not the only cause. Each kernel in the naive schedule is below the ridge. This includes the two GEMMs (128 FLOP per byte in bf16, because the reduction dimension is only the head dimension). The schedule as a whole goes toward ${d/b}$, 64 FLOP per byte. But an H100 needs 295 and an L4 needs 403 to be compute-bound. Longer sequences do not help: they only make the $N \times N$ intermediate too large to fit.

"FlashAttention keeps the math and changes the schedule. The key fact is about a partial softmax result, kept as a reference max, a sum and an unnormalized output. Such a result merges exactly with any other partial result. Thus we tile: each thread block owns a block of queries, streams K and V tiles through on-chip memory, and writes the output once. For the backward, the kernel keeps only one log-sum-exp per row. The backward calculates the probabilities again, at 2.5× the forward FLOPs, which costs less than a transfer of $N^2$ values.

"Then each generation went after the next bottleneck. FA2 repaired the work partition. It put the queries in the outer loop. It divided the warps by rows, so that nothing goes through shared memory. It also added parallelism over the sequence length.

"FA3 exists because on Hopper, the exponential unit takes half as long as the matrix multiplies. FA3 overlaps them with asynchronous tensor-core instructions, a producer warpgroup on the TMA and two consumer warpgroups in ping-pong. On Blackwell, the exponential takes as long as the multiplies. Thus FA4 emulates some exponentials on the FMA units and skips most rescales.

"Serving is a different regime. Decode has one query row per head against a long cache: no reuse, and 1 to 8 FLOP per byte. Thus the kernel divides the cache across thread blocks, and packs the query heads of a GQA group into one tile. It merges the partial results with the same operator. Its time is batch × context × KV bytes over bandwidth.

"vLLM passes a block table and per-sequence lengths into one variable-length call per layer. On Hopper, that call runs FA3. On SM 10.x datacenter Blackwell (B200, GB200), FlashInfer goes first. On a T4, vLLM uses Triton as the fallback. An FP8 KV cache on an L4 or A100 moves the call to FlashInfer."

### Drill questions

**1. Why can a larger batch or a longer sequence not make naive attention compute-bound?**
The reason is that FLOPs and bytes both increase with $B \cdot H \cdot N^2$. The ratio is $N \cdot d / \big(b(N + d)\big)$, which goes toward ${d/b}$. The intensity changes only with $d$, the dtype and the schedule.

**2. FlashAttention's backward does an extra matrix multiply. Why is it still faster?**
It calculates $P$ again from ${S - L}$ ($2N^2 d$ FLOPs, 4.3 µs per head at $N$ = 4k on an H100 at peak). The alternative is to write $P$ in the forward and read it in the backward (67 MB per head, 20 µs of HBM time).

Also, it does not hold $N^2$ per head in memory between the passes. Thus the kernel uses surplus FLOPs to save scarce bandwidth.

**3. What exactly does the kernel save from the forward, and why is that sufficient?**
The kernel saves $O$ and $L = \tau m + \ln l$ per row (and the RNG state if dropout is on). With $L$, $P_{ij} = \exp(S_{ij} - L_i)$ needs no max and no sum. Also, $D_i = \operatorname{rowsum}(\mathit{dO}_i \circ O_i)$ replaces the row reduction of the softmax Jacobian.

**4. What did FA2 change, and which change matters most on a small batch?**
FA2 changed the loop order (one CTA per Q block, output in registers, written once). It also added the split-Q warp partition (no shared-memory exchange) and deferred normalization. On a small batch, the decisive change is the grid over sequence blocks. At 16k tokens, batch 1 with 16 heads is 16 CTAs without it and 2,048 with it.

**5. Our H100 runs FA2 at approximately 35% of peak. What changes with FA3, and why is it Hopper-specific?**
FA3 uses asynchronous warpgroup MMAs and TMA loads, and a producer warpgroup that gives registers to two consumers. With ping-pong, the softmax of one warpgroup runs under the GEMMs of the other. Each warpgroup also has a skewed loop inside it.

It is Hopper-specific because those instructions are Hopper-specific. FA3 is necessary because at $d$ = 128 on an H100, the exponentials take 50% as long as the MMAs. The H100 does 16 EX2 per SM per clock, against 4,096 tensor FLOPs.

**6. A prefill-shaped attention kernel is fast in prefill and slow in decode. Why, and what kernel do you run instead?**
With one query row, the grid of the prefill kernel is batch × heads CTAs. That is 32 for batch 1 on a Llama-3-8B-shaped layer, and 8 when the kernel packs the GQA group into one tile. Each CTA reads the full cache alone, thus most SMs are idle and the kernel never uses all of the HBM bandwidth.

Divide the KV sequence across CTAs. Pack the GQA group. Then combine the $(\hat{O}, \mathrm{LSE})$ partials in a second step, which costs less than 1% more traffic. For batch 1 × 32k on an H100, FA3 as vLLM runs it selects 15 splits (120 CTAs, approximately one wave). If you use FA2, its heuristic selects 29 (232 CTAs, two per SM).

You rarely call the prefill kernel by accident. FA2's `flash_attn_func` and `flash_attn_with_kvcache` find the case `seqlen_q = 1` and change to packing and split-KV themselves. A kernel written by hand does not do this.

**7. How does vLLM select the attention kernel on an L4, an H100, a B200 and a T4?**
vLLM goes through a per-platform priority list. It takes the first backend whose validation accepts the layer. On Ampere, Ada and Hopper, FlashAttention is first (FA2 kernels on the L4, FA3 on the H100). On SM 10.x with causal attention, FlashInfer is first. A T4 gets Triton attention, because FlashAttention needs SM 8.0 and vLLM currently sets a floor of SM 8.0 for FlashInfer.

ALiBi forces FA2. The KV-cache dtype can have priority over the GPU. vLLM's FlashAttention path accepts an FP8 KV cache only with FA3 (SM 9.0) or FA4. Thus an L4 with `kv_cache_dtype=fp8` gets FlashInfer, not FA2.

**8. We want FP8 attention for a 128k-context deployment. What problems can occur, and how do we find them?**
First, decide if it is an FP8 KV cache (decode bandwidth) or FP8 matrix multiplies (prefill compute). The error sources and their mitigations are:

- 3-bit mantissa rounding: no solution within FP8.
- Outlier channels: a Hadamard rotation of Q and K. For e4m3, it helps when the outlier channels of Q and K align (approximately 8× less $QK^{\top}$ error in section 9.4's simulation). For integer formats, it helps in all cases.
- Outlier tokens: per-block or per-head scales, decisive for integer formats.
- Small probabilities that flush to zero: the $2^8$ offset.
- Stale scales: calibration.

On an L4 or A100, an FP8 KV cache also changes the backend that vLLM selects (drill 7). Examine the result with the end-to-end evaluation of the workload, not with a kernel error metric. Include long-context cases.

---

## Glossary

| Term | Meaning |
|---|---|
| Arithmetic intensity | FLOPs per byte moved from HBM. Below the ridge point, a kernel is memory-bound. |
| Ridge point | Peak FLOP/s ÷ HBM bandwidth: 295 FLOP/B for an H100 in bf16, 403 for an L4. |
| HBM / SMEM / L2 | Device memory, per-SM software-managed shared memory, and the GPU-wide cache between them. |
| CTA | Cooperative thread array: a thread block, scheduled on one SM. |
| Warp / warpgroup | 32 threads that execute together / 4 warps (128 threads) that issue one Hopper WGMMA. |
| MMA, WGMMA, tcgen05 | Tensor-core matrix multiply-accumulate: warp-level (Ampere), warpgroup-level and asynchronous (Hopper), single-thread-issued into Tensor Memory (Blackwell). |
| TMA | Tensor Memory Accelerator: Hopper's bulk asynchronous copy of a tile described by a descriptor. |
| `cp.async` | Ampere's asynchronous global-to-shared copy. FA2 uses it for a double buffer. |
| mbarrier | A shared-memory barrier that also counts bytes. It signals TMA completion and pipeline stages. |
| TMEM | Blackwell's per-SM Tensor Memory that holds MMA accumulators. |
| MUFU / EX2 | The special-function unit and its base-2 exponential instruction, 16 per SM per clock on A100 and H100. |
| FFMA | Fused floating-point multiply-add. The exp2 trick makes the exponent argument one FFMA. |
| Online softmax | Softmax in one pass. The kernel updates a reference max, and rescales the sum when the max changes. |
| ${(m, l, o)}$ state | Reference max, sum of exponentials, unnormalized output: the mergeable partial result. |
| LSE | Log-sum-exp of a row's scaled scores, $\tau m + \ln l$. The kernel saves it for the backward, and uses it to merge partials. |
| Split-Q / split-K | The division of a tile among warps by query rows (FA2, no exchange) or by key columns (FA1, exchange through SMEM). |
| Split-KV, Flash-Decoding | The division of the K/V sequence across CTAs, and the merge of the $(\hat{O}, \mathrm{LSE})$ partials in a second kernel. |
| GQA packing | The query heads that share a KV head go into one tile. Thus the kernel reads K/V once per KV head. |
| Wave efficiency | $\text{waves} / \lceil \text{waves} \rceil$ for a grid of CTAs over the SMs: how full the last wave is. |
| Tile quantization | Work wasted when a length is not a multiple of the tile size. |
| Block table / page table | Per-sequence map from logical KV blocks to physical blocks in a paged cache. |
| `cu_seqlens` | Prefix sums of sequence lengths for a packed, variable-length batch. |
| Warp specialization | Different warps (or warpgroups) do different jobs. For example, some load data and some calculate. |
| Ping-pong | Two consumer warpgroups that alternate on the tensor cores, so that the softmax of each one overlaps the GEMMs of the other. |
| Conditional rescaling | The kernel keeps a stale reference max until the new max is larger by more than a threshold (FA4). |
| Incoherent processing | A rotation of Q and K by the same orthogonal matrix before quantization, to spread the outliers. |
| `Max_offset` | FA3's $2^8$ scale factor on FP8 probabilities, to use e4m3's range. |
| Cascade attention | Attention over a shared prefix once for all requests, merged with per-request suffix attention. |
| Persistent kernel, LPT | A kernel whose CTAs loop over work items. LPT is longest-processing-time-first ordering for uneven (causal) tiles. |
| MLA, absorbed decode | Multi-head latent attention. Absorbed decode moves the key/value up-projections to the query and output side, so that decode attends over the latent cache directly. |

---

## Sources

**Papers and posts.** This page did not fetch them. On 2026-09-26, it was not possible to get to arXiv and the blogs from this environment. The page cites them from the literature, and the figures from them have the tag **(verify)** in the text.

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. arXiv:2307.08691.
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. arXiv:2407.08608.
- The FlashAttention-4 paper (2026), as summarized in [the primer](flash-attention-primer.md) **(verify title and figures)**.
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018. arXiv:1805.02867.
- Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021. arXiv:2112.05682.
- Dao, Haziza, Massa, Sizov. *Flash-Decoding for long-context inference.* Blog post, October 2023.
- Ye et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* 2025. arXiv:2501.01005 (cited by the FlashInfer README and vLLM's `merge_attn_states`).
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
- Liu, Zaharia, Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* 2023. arXiv:2310.01889. Brandon et al. *Striped Attention.* 2023. arXiv:2311.09431. Jacobs et al. *DeepSpeed Ulysses.* 2023. arXiv:2309.14509.
- DeepSeek-AI. *DeepSeek-V2.* 2024. arXiv:2405.04434 (MLA). *DeepSeek-V3 Technical Report.* 2024. arXiv:2412.19437.
- Dong, Feng, Guessous, Liang, He. *Flex Attention: A Programming Model for Generating Optimized Attention Kernels.* 2024. arXiv:2412.05496.
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. arXiv:1911.02150 (MQA). Ainslie et al. *GQA.* 2023. arXiv:2305.13245.
- Hong, Kung. *I/O complexity: The red-blue pebble game.* STOC 1981. Williams, Waterman, Patterson. *Roofline.* CACM 2009.

**Upstream code read for this page** (fetched 2026-09-26 from `raw.githubusercontent.com`, branch `main`). The quotes in the text are from these files:

- `Dao-AILab/flash-attention`: `README.md`; `flash_attn/flash_attn_interface.py`; `flash_attn/flash_attn_triton.py`; `csrc/flash_attn/flash_api.cpp`; `csrc/flash_attn/src/flash_fwd_kernel.h`, `softmax.h`, `flash_fwd_launch_template.h`, `flash_bwd_kernel.h`; `hopper/flash_attn_interface.py`, `flash_api.cpp`, `flash_fwd_kernel_sm90.h`, `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `softmax.h`, `tile_size.h`, `heuristics.h`, `tile_scheduler.hpp`, `flash_prepare_scheduler.cu`; `flash_attn/cute/README.md`, `flash_fwd_sm100.py`, `softmax.py`. (`hopper/README.md` returned 404. The FA3 installation notes are in the top-level README.)
- `vllm-project/flash-attention` (vLLM's fork, built as `vllm.vllm_flash_attn`): `hopper/flash_api.cpp`, `heuristics.h`, `tile_size.h`, `flash_prepare_scheduler.cu`.
- `triton-lang/triton`: `python/tutorials/06-fused-attention.py`. Also `python/triton/language/core.py` and `python/triton/runtime/interpreter.py`, on `main` and at `v3.8.0` (block pointers deprecated in 3.8, removed on `main`).
- `vllm-project/vllm`: `vllm/v1/attention/backends/flash_attn.py`, `fa_utils.py`, `flashinfer.py`, `triton_attn.py`; `vllm/v1/attention/selector.py`; `vllm/platforms/cuda.py`; `vllm/config/attention.py`; `requirements/cuda.txt`; `vllm/v1/attention/ops/merge_attn_states.py`, `triton_merge_attn_states.py`.
- `flashinfer-ai/flashinfer`: `README.md`, `flashinfer/decode.py`, `flashinfer/cascade.py`.

**In this repository:**

- [the FlashAttention primer](flash-attention-primer.md)
- [`flash_attention_minimal.py`](flash_attention_minimal.py)
- [`fa_calculators.py`](fa_calculators.py) and its tests
- [`test_triton_kernel_emulated.py`](test_triton_kernel_emulated.py)
- [the companion notebook](notebooks/flash_attention_deep_dive.ipynb)
- [the practice notebook](notebooks/flash_attention_practice.ipynb)
- [the paged-attention primer](../paged-attention/paged-attention-primer.md)
- [the KV-cache primer](../kv-cache/kv-cache-primer.md)
- [the vLLM internals primer](../vllm-internals/vllm-internals-primer.md) (section 6, attention backends from the engine's side)
- [the transformer primer](../../00-foundations/transformers/docs/transformer-primer.md)
- [the GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md)
- [the roofline primer (01)](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)
- [the CUDA primer (02)](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md)
- [the capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md).

---

## Verify list (2026-09-26)

Examine these again before you quote them. The page calculates all other values from stated inputs, or quotes them from the files in the Sources list.

1. **Paper figures.** FA1: Theorem 2 and Proposition 3 as stated, 40.3 GB against 4.4 GB HBM traffic (GPT-2-sized, A100), block-sparse IO bound. FA2: ≈ 2× over FA1, 50–73% of A100 peak forward against 25–40% for FA1. Also for FA2: FA1 parallelized over batch and heads only, the backward at a lower fraction of peak. FA3: FA2 at ≈ 35% on H100, 740 TFLOP/s (75%) bf16, ≈ 1.2 PFLOP/s FP8, 1.5–2.0× over FA2. Also for FA3: ping-pong from 570 to 620–640 TFLOP/s, 3.9 TFLOP/s of special functions. And: 2.6× lower FP8 error, Hadamard fused with RoPE. FA4: 1,605 TFLOP/s (71%) on B200, 1.3× cuDNN 9.13, 2.7× Triton, and the paper's title. Flash-Decoding: up to 8×. FlexAttention: approximately 90% of FA2 forward, 85% backward.
2. **Hardware.** Dense bf16 peaks and HBM bandwidth. H100 SXM: 989.4 TFLOP/s, 3.35 TB/s, 132 SMs, 228 KB SMEM/SM, 50 MB L2. A100 80GB: 312, 2.039 TB/s, 108 SMs, 164 KB SMEM/SM (A100 40GB: 1.555 TB/s). L4: 121, 0.30 TB/s, 58 SMs, 48 MB L2, 100 KB SMEM/SM (99 KB per block). B200: 2,250, 8.0 TB/s. T4: 65 (fp16), 0.32 TB/s, 64 KB SMEM per SM and per block. The per-SM per-clock throughputs in section 4.3, especially the B200 column (≈ 8,192 tensor FLOPs, 128 FP32 FMA, 16 MUFU). The Blackwell hardware description (tcgen05, TMEM, 2-CTA MMA). H100 NVLink ≈ 450 GB/s per direction. H100 dense FP8 peak ≈ 1,979 TFLOP/s (ridge ≈ 590).
3. **Model shapes.** Llama-3-8B: 8.03 B parameters, 32 layers, 32 query and 8 KV heads of 128, MLP width 14,336. DeepSeek-V3 MLA: 128 heads, latent 512, rotary 64, no-PE key 128, value 128. The claim that engines often serve MLA models with data-parallel attention. The claim that the Q and K outliers of rotary models share dimensions.
4. **Moving code facts** (read on `main`, 2026-09-26). vLLM backend priority lists and FA version selection. That vLLM's FlashAttention backend accepts an FP8 KV cache only with FA3 on SM 9.0 or FA4. The fall-through to FlashInfer on SM 8.x that this causes. vLLM's SM 8.0 floor for FlashInfer. FA3's split heuristic, dynamic per-sequence split and vLLM's CUDA-graph split cap of 32. That vLLM's fork agrees with upstream on them. FA2's `page_block_size` multiple of 256. FA3's arbitrary page size. FA3 tile sizes and register counts. FA4's `rescale_threshold = 8.0`, `ex2_emu_freq` values and warp roles. FlashInfer backends and `fixed_split_size`.
5. **Tooling.** That `triton.testing.do_bench` flushes L2 by default. That PyTorch's flash SDPA backend needs SM 8.0+. That Triton `tl.dot` uses tensor cores on a T4 in fp16. That the section 11 defaults fit an L4's shared memory, and the smaller tiles fit a T4's. That `torch` and `triton` are importable on a Colab CPU runtime. That the interpreter accepts the section 11.1 kernel (with launch options, for example `num_warps`).
6. **L4 expectation** in section 10.5 is an assumption, not a measurement. Measure it.
