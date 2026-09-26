# FlashAttention in depth: from the roofline to serving kernels

**Scope.** This is the second FlashAttention document in this folder. [The primer](flash-attention-primer.md) builds the idea from zero (what attention computes, why the naive schedule is slow, online softmax, the lineage). This page assumes you have read it, or can explain those ideas yourself, and goes to the depth needed to defend a kernel or backend choice in a design review: exact byte counts, the algebra with numerical safety, what each FlashAttention generation changed and why the hardware forced it, why decode needs a different kernel, how paged KV reaches the kernel in vLLM and FlashInfer, what each attention variant costs, numerics, how to measure honestly, and how to write the forward pass yourself in Triton.

**Tier.** Everything here is learnable on a CPU (T0). The companion notebook [`flash_attention_deep_dive.ipynb`](flash_attention_deep_dive.ipynb) and the calculators in [`fa_calculators.py`](fa_calculators.py) (pinned by [`test_fa_calculators.py`](test_fa_calculators.py)) reproduce every derived number on this page with numpy. Running a real kernel (sections 10 and 11) is T1: one small GPU, such as a free Colab or Kaggle T4 or any rented 24 GB card; see [COMPUTE.md](../../COMPUTE.md) for where to get one. On a T4, note that FlashAttention-2 does not support Turing (section 7.4).

**Sources.** Grounded in the upstream code, fetched on 2026-09-26 (listed under Sources): the FlashAttention repo (FA2 CUDA kernels, FA3 Hopper kernels, FA4 CuTe-DSL kernels, the Triton version), vLLM's attention backends, and FlashInfer. The papers could not be fetched from this environment (arXiv is blocked), so figures quoted from them carry **(verify)**. Every other performance number is derived on this page from stated assumptions.

**Notation.** Per head: sequence length `N` (or `N_q`, `N_k` when they differ), head dimension `d`, bytes per element `b` (2 for bf16/fp16, 1 for FP8), softmax scale `τ = 1/√d`. Tile sizes: `B_r` rows of Q per thread block (the kernels call it `kBlockM`), `B_c` rows of K/V per tile (`kBlockN`). `H` query heads, `H_kv` key/value heads, group size `g = H / H_kv`. "CTA" = thread block. SMEM = shared memory. HBM = device memory.

---

## The one-minute version

- Naive attention is memory-bound on every current GPU, and it cannot be fixed by making the problem bigger: its arithmetic intensity tends to `d/b` (64 FLOP/byte for `d = 128` in bf16), far below the ridge of an H100 (≈ 295) or an L4 (≈ 403). The fix is a schedule, not an approximation.
- The schedule rests on one algebraic fact: a partial softmax-weighted sum can be stored as a triple `(m, l, o)` and any two triples merge exactly. That single operator gives you tiling (FA1/FA2), split-KV decode (Flash-Decoding), cascade attention over shared prefixes, and ring attention across GPUs.
- FA1 tiled the computation; FA2 swapped the loops so each CTA owns a Q block, which keeps the output in registers, adds parallelism over the sequence, and removes inter-warp traffic; FA3 made the kernel asynchronous on Hopper (TMA, WGMMA, producer/consumer warpgroups, ping-pong) because the exponential unit had become a bottleneck; FA4 goes further on Blackwell, where the exponential unit now takes as long as the matrix multiplies.
- Decode is a different problem: one query row against a long KV cache has no reuse, so it is purely bandwidth-bound; the kernel must split the KV sequence across CTAs and pack the query heads of a GQA group together. Decode attention time grows with batch × context × KV bytes per token.
- Serving adds paging (block tables into the kernel), ragged batches, and a scheduler that picks a backend per GPU generation. In vLLM on CUDA that is FlashAttention (FA2 on Ampere/Ada, FA3 on Hopper) unless the GPU is a Blackwell part, where FlashInfer goes first for causal attention.

---

## 1. Standard attention on the roofline

The roofline itself (arithmetic intensity, ridge point, attainable FLOP/s) is covered in [the roofline primer (01)](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md); the memory hierarchy that makes tiling pay is in [the GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md). Here we apply it.

### 1.1 FLOPs

Per head, `S = τ·QKᵀ` is an `N×d` by `d×N` product: `N²·d` multiply-adds, `2N²d` FLOPs. `O = PV` is another `2N²d`. Softmax adds roughly 5 operations per score (max, subtract, exponentiate, sum, divide). The convention FlashAttention's own benchmarks and the Triton tutorial use, and this page uses, counts the two matrix products only:

```
FLOPs(forward, one head) = 4 · N_q · N_k · d          causal: × 0.5        backward: × 2.5
```

Softmax work is invisible in that count but not in the runtime; sections 4.3 and 5 are about exactly that gap.

### 1.2 Exact HBM traffic of the naive schedule

Executed literally, attention is three kernels. Count every byte that crosses HBM for one head, with `Q, K, V, O, S, P` all in bf16:

| Kernel | Reads | Writes | Bytes | N = 4,096, d = 128 |
|---|---|---|---|---|
| 1. `S = τ·QKᵀ` | Q, K: `2Nd·b` | S: `N²·b` | `2Ndb + N²b` | 35.65 MB |
| 2. `P = softmax(S)` | S: `N²·b` | P: `N²·b` | `2N²b` | 67.11 MB |
| 3. `O = PV` | P: `N²·b`, V: `Nd·b` | O: `Nd·b` | `N²b + 2Ndb` | 35.65 MB |
| **Total** | | | **`4N²b + 4Ndb`** | **138.41 MB** |

This is the floor for an unfused implementation. Real eager code is worse:

- Each separate elementwise pass over the scores (scale, mask fill, dropout) reads and writes `N²` again: +`2N²b` = +67.1 MB per pass at `N = 4,096`. A PyTorch sequence `matmul → div → masked_fill → softmax → dropout → matmul` has three such extra passes.
- If the scores are kept in fp32 for the softmax, every `N²` term doubles (272.6 MB at `N = 4,096`).

(`fa_calculators.naive_traffic()`.)

### 1.3 Arithmetic intensity: why it is memory-bound

```
I = 4N²d / (4N²b + 4Ndb) = N·d / (b·(N + d))   →   d / b   as N → ∞
```

For `d = 128` in bf16, `I → 64 FLOP/byte` (62.1 at `N = 4,096`, 63.8 at `N = 32,768`). The length `N` cancels: a longer sequence, a bigger batch or more heads all scale FLOPs and bytes together. The intensity of the naive schedule is fixed by the head dimension and the element size.

Every kernel in the schedule is individually below the ridge, not just the softmax:

| Kernel | Intensity (bf16, large N) | Why |
|---|---|---|
| `QKᵀ` | `2N²d / ((2Nd + N²)·b)` → `2d/b` = 128 | an outer-product-shaped GEMM: the reduction dimension is only `d`, and the `N×N` output write dominates |
| softmax | ≈ 5 FLOPs / 4 bytes ≈ 1.25 | pure streaming |
| `PV` | `2N²d / ((N² + 2Nd)·b)` → `2d/b` = 128 | the `N×N` input read dominates |

Ridge points (dense bf16 peak ÷ HBM bandwidth; datasheet values, **(verify)**): H100 SXM 989.4 TFLOP/s ÷ 3.35 TB/s = **295 FLOP/B**; L4 121 ÷ 0.30 = **403**; A100 80GB 312 ÷ 2.039 = **153**; B200 2,250 ÷ 8.0 = **281**. Naive attention at 64 FLOP/B sits 2.4× (A100) to 6.3× (L4) below them.

### 1.4 Worked numbers: N = 4k and 32k, d = 128, bf16, one head

| | N = 4,096 | N = 32,768 |
|---|---|---|
| FLOPs (`4N²d`) | 8.59 GFLOP | 549.8 GFLOP |
| Naive HBM bytes | 138.4 MB | 8.62 GB |
| `S` alone (`N²b`) | 33.6 MB | 2.15 GB |
| Intensity | 62.1 FLOP/B | 63.8 FLOP/B |
| **H100**: memory time / compute time | 41.3 µs / 8.7 µs | 2.57 ms / 0.56 ms |
| H100 attainable | 208 TFLOP/s (21% of peak) | 214 TFLOP/s (22%) |
| **L4**: memory time / compute time | 461 µs / 71 µs | 28.7 ms / 4.54 ms |
| L4 attainable | 18.6 TFLOP/s (15% of peak) | 19.1 TFLOP/s (16%) |

These are floors that assume every kernel streams at 100% of HBM bandwidth; measured naive kernels are slower. (`fa_calculators.roofline()`.)

The second failure is capacity. At `N = 32,768`, `S` is 2.15 GB **per head, per sequence**. A 32-head layer would need 68.7 GB for `S` alone, and the same again for `P`: more than an L4's 24 GB, and most of an H100's 80 GB before a single weight is loaded. The naive schedule is not just slow at long context; it does not fit.

### 1.5 What a fused schedule can reach

The least traffic any exact schedule can have is to read `Q, K, V` once and write `O` once: `4Nd·b` bytes (plus 4 bytes of fp32 log-sum-exp per row, section 3.5). That is 4.2 MB at `N = 4,096` and 33.7 MB at `N = 32,768`: intensities of 2,040 and 16,320 FLOP/B, far right of every ridge. A kernel that got close to this compulsory traffic would be compute-bound at 8.7 µs (H100) and 71 µs (L4) for the 4k case. Whether real kernels get close depends on the loop schedule and on the L2 cache, which is where section 3.4 picks up.

---

## 2. Online softmax: the algebra in full

The primer's section 5 shows the update rule and a worked example with symbolic values. This section derives it with the scale, proves the invariant, states the merge operator that every later section depends on, shows the base-2 form the kernels actually execute, and lists the numerical guards.

### 2.1 Safe softmax needs two reductions

For one query row with raw scores `s_j = q·k_j`:

```
softmax(τ s)_j = exp(τ(s_j − m)) / Σ_k exp(τ(s_k − m)),      m = max_k s_k
```

Subtracting `m` changes nothing mathematically and makes every exponent `≤ 0`, so nothing overflows. But now a row needs two global reductions (the max, then the sum) before the first output value can be normalized. Done naively, that is two extra passes over `S`.

### 2.2 The recurrence

Process the keys in blocks `B_1, B_2, …`. Carry three quantities per row: a reference max `m`, a sum `l`, and an unnormalized output `o` (a `d_v`-vector). Start from `(m, l, o) = (−∞, 0, 0)`. For block `B`:

```
m'  = max(m, max_{j∈B} s_j)
α   = exp(τ(m − m'))                                   rescale factor for everything seen so far
l'  = α·l + Σ_{j∈B} exp(τ(s_j − m'))
o'  = α·o + Σ_{j∈B} exp(τ(s_j − m')) · v_j
```

After the last block: `O = o / l` and `LSE = τ·m + ln l` (the log-sum-exp of the scaled scores).

**Invariant.** After processing blocks `B_1..B_t`, `l_t = Σ_{j seen} exp(τ(s_j − m_t))` and `o_t = Σ_{j seen} exp(τ(s_j − m_t))·v_j`. It holds trivially for the empty prefix. If it holds at `t`, then `α·l_t = Σ_{j seen} exp(τ(s_j − m_t))·exp(τ(m_t − m_{t+1})) = Σ_{j seen} exp(τ(s_j − m_{t+1}))`, and adding the new block's terms (already relative to `m_{t+1}`) gives the invariant at `t+1`; the same for `o`. At the end, `o/l = Σ_j exp(τ(s_j − m))·v_j / Σ_j exp(τ(s_j − m)) = Σ_j softmax(τs)_j·v_j`. Exact, for any block partition.

### 2.3 The state is a monoid: merge any two partial results

Two states built from disjoint sets of keys `A` and `B` combine the same way:

```
merge((m_a, l_a, o_a), (m_b, l_b, o_b)):
    m = max(m_a, m_b)
    l = exp(τ(m_a − m))·l_a + exp(τ(m_b − m))·l_b
    o = exp(τ(m_a − m))·o_a + exp(τ(m_b − m))·o_b
```

`merge` is associative and commutative, and `(−∞, 0, 0)` is its identity: the states form a commutative monoid. This is the load-bearing fact of everything that follows. You can split the keys any way you like, compute the pieces anywhere, and combine them in any order and any tree shape:

| Where the split happens | What it is called | Section |
|---|---|---|
| K/V tiles inside one CTA | FlashAttention's inner loop | 3, 4 |
| K/V ranges across CTAs, combined by a second kernel | split-KV, Flash-Decoding | 6.3 |
| shared prefix vs per-request suffix | cascade attention (vLLM, FlashInfer) | 7.3, 7.4 |
| K/V shards across GPUs | ring attention, context parallelism | 8.4 |

**The LSE form.** Kernels usually store a finished partial result as a normalized output `Ô = o/l` plus `L = τ·m + ln l`. The same merge in that representation:

```
L  = max(L_a, L_b) + ln( exp(L_a − max) + exp(L_b − max) )
Ô  = exp(L_a − L)·Ô_a + exp(L_b − L)·Ô_b                       (the two weights sum to 1)
```

This is what FA2's split-KV combine kernel (`combine_attn_seqk_parallel` in `flash_fwd_kernel.h`), vLLM's `merge_attn_states` and FlashInfer's `merge_state` implement (all read for this page).

### 2.4 The exp2 trick

GPUs have no natural-exponential instruction. The special-function unit (MUFU) computes `2^x` (`ex2.approx`); the fast intrinsic `__expf(x)` is a multiply by `log2 e` followed by `ex2`, and the accurate `expf` adds range reduction on top. Kernels therefore fold the softmax scale and `log2 e` into one constant and work in base 2:

```
c = τ · log2(e)                         (FA2: params.scale_softmax_log2; Triton tutorial: qk_scale *= 1.44269504)
p_j = exp2(s_j·c − m·c)                 one FFMA (fused multiply-add) + one EX2 per score
α   = exp2((m − m')·c)
```

FA2's `softmax.h` says why: "Instead of computing exp(x - max), we compute exp2(x * log_2(e) - max * log_2(e)). This allows the compiler to use the ffma instruction instead of fadd and fmul separately." The max itself is taken on the raw scores (valid because `τ > 0`).

The log-sum-exp can be kept in either base, and mixing conventions is a real bug source (a factor of `ln 2`): FA2 stores it in natural-log units (`row_max * softmax_scale + __logf(sum)`), the Triton tutorial stores it in base 2 (`m_i += tl.math.log2(l_i)` with `m_i` already in log2 units).

### 2.5 Numerical safety

- **No overflow, by construction.** Every exponent is `≤ 0`, so `p ∈ (0, 1]` and `α ∈ (0, 1]` whatever the magnitude of the scores.
- **The final division is safe.** Any row with at least one unmasked key has `l ≥ 1`, because the maximum contributes `exp(0) = 1`.
- **Underflow is harmless.** Terms below fp32's range flush to zero; they were negligible relative to the `1` from the max.
- **Fully masked tiles and rows need a guard.** If every score in a row of a tile is `−∞` (causal or sliding-window masking), then `m' = −∞` and `(−∞) − (−∞)` is NaN. FA2 substitutes 0 for the max in that case ("If max is -inf, then all elements must have been -inf (possibly due to masking). We don't want (-inf - (-inf)) since that would give NaN."). A row with no allowed key at all ends with `l = 0`; FA2 then writes `O = 0` and `LSE = +∞` (`−∞` in its split-KV path), FA3 writes `−∞`, and vLLM's merge code maps both to `−∞` before merging.
- **Accumulate in fp32.** `m`, `l` and `o` live in fp32 registers. `l` can reach `N` (all scores equal); bf16 cannot even represent every integer above 256.
- **Defer what can be deferred.** FA2 divides by `l` once at the end rather than every block, and delays the cross-thread reduction of `l` to the end too ("We don't do the reduce across threads here since we don't need to use the row_sum").
- **The reference need not be the true max.** The invariant only needs all terms of `l` and `o` to be relative to the same reference. FA4 exploits this on Blackwell with *conditional rescaling*: it keeps the old max unless the new one exceeds it by more than a threshold (`rescale_threshold = 8.0` log2 units for 16-bit inputs in `flash_attn/cute/flash_fwd_sm100.py`), skipping the rescale multiply most of the time. The price is that `p` can now reach `2^8`, so the kernel asserts that `max_offset + rescale_threshold` stays below `log2` of the input dtype's largest value (the `P` fed to the second matrix multiply is in that dtype).

### 2.6 A hand-worked two-block example

One query row, `d = 4` so `τ = 0.5`, four keys in two blocks of two. To keep the output printable, `V` has two columns.

```
raw scores q·k:   block 1 = [4, 8]          block 2 = [6, 12]
values:           v1 = [1, 0]   v2 = [0, 1]   v3 = [1, 1]   v4 = [2, −1]
```

**Reference.** Scaled scores `[2, 4, 3, 6]`; softmax `[0.015219, 0.112457, 0.041371, 0.830953]`; `O = [1.718495, −0.677125]`; `LSE = 6.185182`.

**Block 1.** `m = 8`. `p = exp(0.5·([4, 8] − 8)) = [e^−2, e^0] = [0.135335, 1]`. `l = 1.135335`. `o = 0.135335·v1 + 1·v2 = [0.135335, 1]`.

**Block 2.** `m' = 12`. `α = exp(0.5·(8 − 12)) = e^−2 = 0.135335`. `p = exp(0.5·([6, 12] − 12)) = [e^−3, 1] = [0.049787, 1]`.

```
l' = 0.135335 × 1.135335 + (0.049787 + 1)          = 0.153651 + 1.049787 = 1.203438
o' = 0.135335 × [0.135335, 1] + 0.049787·[1, 1] + 1·[2, −1]
   = [0.018316, 0.135335] + [0.049787, 0.049787] + [2, −1]          = [2.068103, −0.814878]
```

**Finalize.** `O = o'/l' = [1.718495, −0.677125]` and `LSE = 0.5·12 + ln 1.203438 = 6 + 0.185182 = 6.185182`. Both match the reference.

**The same, as the kernel executes it (base 2).** `c = 0.5·log2 e = 0.721348`. Block 1: `s·c = [2.885390, 5.770780]`, `m·c = 5.770780`, exponents `[−2.885390, 0]`, `exp2` gives `[0.135335, 1]`: the same numbers, one FFMA and one EX2 each.

**The same, as a split-KV merge in LSE form.** Treat the two blocks as two independent splits. Split 1 alone: `Ô₁ = [0.119203, 0.880797]`, `L₁ = 0.5·8 + ln 1.135335 = 4.126928`. Split 2 alone: `Ô₂ = [1.952574, −0.905148]`, `L₂ = 0.5·12 + ln 1.049787 = 6.048587`. Then `L = ln(e^4.126928 + e^6.048587) = 6.185182`, weights `e^(L₁−L) = 0.127677` and `e^(L₂−L) = 0.872323`, and `O = 0.127677·Ô₁ + 0.872323·Ô₂ = [1.718495, −0.677125]`.

**The same, with a lagging max (FA4-style).** Use a threshold of 3 log2 units for illustration (FA4 uses 8). The max grew by `(12 − 8)·c = 2.885 < 3`, so keep `m = 8`: block 2 gives `p = exp2([6, 12]·c − 8c) = [e^−1, e^2] = [0.367879, 7.389056]`, `l = 1.135335 + 7.756935 = 8.892271`, `o = [15.281327, −6.021177]`, and `o/l = [1.718495, −0.677125]`. No rescale was executed; the cost is a `p` of 7.39, above 1.

(`fa_calculators.online_trace()`, `merge_lse()`; the notebook reruns all four variants.)

---

## 3. FlashAttention-1: tiling and recomputation

### 3.1 The algorithm and its loop order

FA1 (Dao et al., 2022) splits `Q` into `T_r = N/B_r` row blocks and `K, V` into `T_c = N/B_c` blocks and runs the recurrence of section 2.2 tile by tile. Its loop order is **outer over K/V, inner over Q**:

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

Two things to notice, both of which FA2 changes: the output accumulator makes a round trip through HBM on every visit, and `O_i` is kept normalized (divided by `l`) at every step.

### 3.2 Block sizes from SRAM

With `M` elements of on-chip SRAM, the paper sets `B_c = ⌈M/(4d)⌉` and `B_r = min(⌈M/(4d)⌉, d)`: `K_j` and `V_j` take `2·B_c·d ≈ M/2`, `Q_i` and `O_i` at most as much, and the score tile `B_r·B_c ≤ d·M/(4d) = M/4`. Everything is `O(M)`, which is all the analysis needs; the constant factors are why real kernels tune tile sizes by hand. On an H100 (228 KB of SMEM per SM, 116,736 bf16 elements) with `d = 128`, that gives `B_c = 228` and `B_r = 128`.

Production kernels use the same reasoning with two refinements: tile sizes are rounded to shapes the MMA instructions like, and the score tile and the output accumulator live in *registers*, so SMEM holds only the Q, K and V tiles. FA2's H100/A100 configuration for `d = 128` is `B_r × B_c = 128 × 64`, which needs `(128 + 2·64)·128·2 B = 64 KB` of SMEM (section 4.5 has the full table from the source).

### 3.3 The IO-complexity result and a proof sketch

The paper's Theorem 2 (**(verify)** statement): standard attention needs `Θ(Nd + N²)` HBM accesses; FlashAttention needs `Θ(N²d²/M)`, for `d ≤ M ≤ Nd`.

Proof sketch for the FlashAttention side:

1. `K` and `V` are read exactly once: `Θ(Nd)`.
2. The outer loop runs `T_c = N/B_c = Θ(Nd/M)` times, because `B_c = Θ(M/d)`.
3. Each outer iteration streams all of `Q` and `O` through SRAM (read `Q`, read and write `O`): `Θ(Nd)`.
4. Total: `Θ(Nd) + Θ(Nd/M)·Θ(Nd) = Θ(N²d²/M)`.

Compared with the naive `Θ(N²)` (for `N ≫ d`), the ratio is about `M/d²`. For `d = 128`, `d² = 16,384` elements, which fits in every current SM, so FlashAttention always moves less; with `M = 116,736` the asymptotic saving is about 7× for `d = 128` and 28× for `d = 64`. The paper measured 40.3 GB versus 4.4 GB of HBM reads and writes for GPT-2-sized attention (`N = 1,024`, `d = 64`, forward plus backward on an A100) **(verify)**.

The lower bound (Proposition 3, **(verify)**): no exact attention algorithm can use `o(N²d²/M)` HBM accesses for all `M` in `[d, Nd]`. Argument: at `M = Θ(Nd)` such an algorithm would make `o(Nd)` accesses, but `Q, K, V` and `O` have `Θ(Nd)` elements that start in (or must end in) HBM. Contradiction.

### 3.4 The model against a real GPU: count it

The IO-complexity model assumes nothing is cached between SRAM and HBM. Counting bytes under that model with the actual block sizes (`fa_calculators.flash_traffic()`, one head, bf16, `d = 128`):

| Schedule | N = 4,096: bytes / intensity | N = 32,768: bytes / intensity |
|---|---|---|
| Naive, three kernels | 138.4 MB / 62 | 8,623 MB / 64 |
| FA1 order (outer K/V, `B_c = 128`) | 104.9 MB / 82 | 6,593 MB / 83 |
| FA2 order (outer Q, `B_r = 128`) | 69.2 MB / 124 | 4,312 MB / 127 |
| FA2 order, causal | 36.7 MB / 117 | 2,173 MB / 127 |
| Compulsory (`4Nd·b` + LSE) | 4.2 MB / 2,040 | 33.7 MB / 16,320 |

Two lessons.

**The per-tile intensity is set by `B_r` alone.** In the FA2 order each K/V tile is loaded once per Q block and used by the `B_r` query rows of that CTA: `4·B_r·B_c·d` FLOPs for `2·B_c·d·b` bytes, that is `2·B_r/b` FLOP/B, or simply `B_r` in bf16. With `B_r = 128` that is 128 FLOP/B, which is still below the H100 ridge (295). `B_r` cannot grow much further: the output accumulator for `B_r × d` fp32 values has to fit in registers.

**L2 closes the gap.** CTAs that work on the same (batch, head) run at the same time and re-read the same `K` and `V` tiles, which then come from L2 rather than DRAM. On an H100, `K` and `V` of one head are 2 MB at `N = 4,096` and 16.8 MB at `N = 32,768`, against 50 MB of L2. FA3 makes this explicit: its persistent scheduler walks heads in "sections" sized so that `K` and `V` fit in a 32 MB L2 budget ("we have to make sure K & V still fit into L2 cache, so we perform scheduling on 'sections' of the head & batch dimension", `hopper/tile_scheduler.hpp`). On a profiler this shows as a high L2 hit rate and DRAM traffic close to the compulsory row. The IO-complexity result explains why tiling wins; the number you measure depends on L2 residency and on how many SMs are busy.

### 3.5 The backward pass: recompute from the log-sum-exp

The forward saves `O` (`N×d`) and one fp32 number per row, `L_i = τ·m_i + ln l_i`, plus the RNG seed and offset when dropout is used. It does not save `P`. With `S` including the scale and `dO` the incoming gradient:

```
P_ij = exp(S_ij − L_i)                   recomputed per tile: no max, no sum, L already normalizes
dV   = Pᵀ dO
dP   = dO Vᵀ
D_i  = rowsum(dO_i ∘ O_i)                an O(Nd) pre-pass
dS   = P ∘ (dP − D)                      the softmax Jacobian, row by row
dQ   = τ · dS K          dK = τ · dSᵀ Q
```

Why `D` can be computed from `O`: `D_i = Σ_j P_ij·dP_ij = Σ_j P_ij (dO_i·v_j) = dO_i·Σ_j P_ij v_j = dO_i·O_i`. FA2 does it in `dot_do_o`; the Triton tutorial in `_attn_bwd_preprocess` (`delta = tl.sum(o * do, axis=1)`).

Memory: at `N = 32,768` with 32 heads, the saved statistics are `32 × 32,768 × 4 B = 4.2 MB`, against 68.7 GB for `P` in bf16.

Work partitioning in FA2's backward (`flash_bwd_kernel.h`, `compute_dq_dk_dv_1colblock`): each CTA owns one K/V column block, loops over all Q blocks, keeps `K_j, V_j, dK_j, dV_j` on chip, and writes `dK_j, dV_j` once. Every CTA contributes to every `dQ_i`, so those contributions are summed with fp32 `atomicAdd` into a `dq_accum` buffer. Atomics make the summation order, and therefore the last bits of `dQ`, non-deterministic; `deterministic=True` gives each CTA its own accumulator ("If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer"), which the README describes as slightly slower and using more memory.

### 3.6 Why the backward costs about 2.5× the forward

Count the matrix products. The forward has two (`QKᵀ`, `PV`), `4N²d` FLOPs. The backward has five, each `2N²d`: recompute `S = QKᵀ`, then `dV = PᵀdO`, `dP = dO·Vᵀ`, `dQ = dS·K`, `dK = dSᵀQ`. That is `10N²d = 2.5 × 4N²d`. The Triton tutorial's benchmark encodes exactly this: `total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)`.

Wall-clock is usually worse than 2.5×, because the backward has more elementwise work per tile (`dS`), holds more live tiles per CTA (fewer CTAs per SM), and pays atomics or a second pass for `dQ`; the FA2 paper reports the backward reaching a lower fraction of peak than the forward **(verify)**.

Recomputation is a good trade because of section 1: re-deriving `P` costs one extra GEMM (`2N²d` = 4.3 GFLOP per head at `N = 4,096`, 4.3 µs on an H100 at peak) and avoids writing `P` in the forward and reading it back in the backward (2 × 33.6 MB = 67 MB, 20 µs at 3.35 TB/s), plus the memory to hold it in between.

---

## 4. FlashAttention-2: work partitioning

FA2 (Dao, 2023) keeps the algorithm and changes who does what. The execution model it relies on (grids, CTAs, warps, occupancy, shared memory) is covered in [the CUDA primer (02)](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md).

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

The launch in `flash_fwd_launch_template.h` is `dim3 grid(num_m_block, params.b, params.h)`: the grid covers the Q blocks as well as batch and heads. Three things follow.

1. **The accumulator never leaves the chip.** `O_i`, `m_i` and `l_i` stay in registers for the whole inner loop and are written once. The FA1 round trip of `O` through HBM (the dominant term of the FA1 row in section 3.4) disappears; what remains is re-reading `K` and `V`, which are read-only and cache well in L2.
2. **Parallelism over the sequence.** FA1's kernel parallelized over batch and heads only **(verify)**. With batch 1 and 16 heads (a tensor-parallel shard, say) that is 16 CTAs for 108 or 132 SMs. FA2 at `N = 16,384`, `B_r = 128` launches `16 × 128 = 2,048` CTAs.
3. **Causal work is easy to skip.** Each CTA knows its row range, so it simply stops at the diagonal (section 4.4).

### 4.2 Inside the CTA: split-Q instead of split-K

A CTA has 4 (sometimes 8) warps. How they divide a `B_r × B_c` tile decides how much they must talk to each other:

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

In FA2's kernel each warp owns interleaved 16-row slices of the Q tile (the mask is applied at row `m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4` with a stride of `kNWarps * 16`). Within a warp, one row of the MMA accumulator is spread over 4 threads, so the row max and row sum are 4-thread shuffles (`Allreduce<4>` in `softmax.h`), not SMEM traffic. The probabilities are converted to fp16/bf16 in registers (`convert_type<Element>(acc_s)`) and fed straight into the second MMA as a register operand (`gemm_rs`). Nothing about `P` or the partial `O` touches shared memory.

### 4.3 Fewer non-matmul FLOPs, and why they cost so much

FA2's changes to the arithmetic itself:

- keep `O` unnormalized and divide by `l` once at the end (FA1 rescaled by `diag(l)⁻¹` every block);
- save only the LSE for the backward, not `m` and `l` separately;
- reduce `l` across threads once at the end, not every block;
- fold the scale into `exp2` so each score costs one FFMA and one EX2 (section 2.4).

Why a few elementwise operations matter against 512 matmul FLOPs per score: the units that execute them are much slower. Per SM per clock (CUDA programming guide throughput tables and datasheet peaks; **(verify)** for B200):

| Per SM per clock | A100 | H100 SXM | B200 |
|---|---|---|---|
| dense bf16 tensor-core FLOPs | 2,048 | 4,096 | ≈ 8,192 |
| FP32 FMA instructions | 64 | 128 | 128 |
| MUFU instructions (EX2) | 16 | 16 | 16 |

Per score, with `d = 128`: `4d = 512` MMA FLOPs, one EX2, and about 5 FP32 instructions (FFMA for scale-and-subtract, max, add into `l`, the `O` rescale amortized as `d/B_c` = 2 FMULs at `B_c = 64`, and a conversion that handles two values per instruction). Clocks per score if each unit ran alone:

| | A100 | H100 | B200 |
|---|---|---|---|
| tensor cores | 0.25 | 0.125 | 0.0625 |
| EX2 on MUFU | 0.0625 (25% of MMA time) | 0.0625 (**50%**) | 0.0625 (**100%**) |
| ≈ 5 FP32 instructions | 0.078 (31%) | 0.039 (31%) | 0.039 (63%) |

Different units run concurrently when different warps feed them, so these do not simply add. But a kernel whose warps do MMA, then softmax, then MMA in lockstep pays these fractions as idle tensor-core time. This table is the motivation for FA3's overlap (the H100 column) and FA4's exponential emulation (the B200 column). The H100 row agrees with the FA3 paper's own framing: 989 TFLOP/s of matmul against about 3.9 TFLOP/s of special functions **(verify)**.

### 4.4 Causal block skipping

From `compute_attn_1rowblock` in `flash_fwd_kernel.h`:

```
n_block_max = min( ceil(N_k / B_c),  ceil(((m_block + 1)·B_r + N_k − N_q) / B_c) )      causal upper bound
n_block_min = max( 0, (m_block·B_r + N_k − N_q − window_left) / B_c )                     sliding window only
```

The loop runs from `n_block_max − 1` *down* to `n_block_min`, so the `ceil(B_r/B_c)` diagonal tiles that need masking (`n_masking_steps`) come first and the rest of the loop body contains no mask code. For `N = 512`, `B_r = 128`, `B_c = 64`:

```
                   K/V blocks (B_c = 64) ->
                   0   1   2   3   4   5   6   7
Q block 0         ░░  ░░  ··  ··  ··  ··  ··  ··          ██  full tile: no mask code
Q block 1         ██  ██  ░░  ░░  ··  ··  ··  ··          ░░  diagonal tile: masked
Q block 2         ██  ██  ██  ██  ░░  ░░  ··  ··          ··  never loaded, never computed
Q block 3         ██  ██  ██  ██  ██  ██  ░░  ░░
                                                          20 of 32 tiles visited, 8 masked
```

At `N = 4,096` the same tiling visits 1,056 of 2,048 tiles (51.6%, 64 masked); at `N = 32,768`, 65,792 of 131,072 (50.2%). The "causal = 0.5 × FLOPs" convention is the large-`N` limit (`fa_calculators.causal_tiles()`).

Causal also creates load imbalance: Q block `m` does `2(m + 1)` tiles here, so the last blocks are the longest. FA3's persistent scheduler hands out the longest remaining tiles first ("We use longest-processing-time-first scheduling: the longest remaining tile is assigned to the first SM that's free", `hopper/tile_scheduler.hpp`).

### 4.5 Tile sizes and the SMEM budget

From `run_mha_fwd_hdim*` in `flash_fwd_launch_template.h` (SMEM = Q tile + one K and one V tile, `(B_r + 2B_c)·d·b`, `fa_calculators.tile_smem_bytes()`):

| GPU, head dim | `B_r × B_c` | Warps | SMEM | Source comment |
|---|---|---|---|---|
| A100, H100, `d = 128` | 128 × 64 | 4 | 64 KB | "1st ones are good for H100, A100" |
| sm86/sm89 (A10, RTX 30/40, **L4**), `d = 128`, non-causal | 128 × 32 | 4 | 48 KB | "128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM" |
| sm86/sm89, `d = 128`, causal | 64 × 64 | 4 | 48 KB | "64 x 64 is the fastest for causal (because it's square)" |
| A100, `d = 256` | 128 × 64 | 8 | 128 KB | "For A100, we want to run with 128 x 64 (128KB smem)" |
| H100, `d = 256` | 64 × 64 | 4 | 96 KB | "For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM" |

The main loop overlaps loads with compute using Ampere's asynchronous copies (`cp.async`): while `S_j = Q K_jᵀ` runs, `V_j` is in flight; right after that GEMM, the load of `K_(j−1)` is issued so it overlaps the softmax and `P_j V_j`.

### 4.6 What FA2 achieved, and where it stopped

About 2× over FA1 and 50–73% of the A100's peak in the forward pass, against 25–40% for FA1 (FA2 paper **(verify)**); 225 TFLOP/s per A100 (72% model FLOPs utilization) training GPT-style models end to end (the repository README). On an H100 the same kernel reaches only about 35% of peak (FA3 paper **(verify)**): it is written with Ampere's synchronous warp-level `mma.sync` and `cp.async`, while Hopper's full throughput needs asynchronous warpgroup MMAs and the TMA.

---

## 5. FlashAttention-3: asynchrony on Hopper (and FA4 on Blackwell)

FA3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024) is a Hopper-specific rewrite on CUTLASS/CuTe. The source read for this section is `hopper/flash_fwd_kernel_sm90.h`, `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `hopper/softmax.h`, `hopper/tile_size.h`, `hopper/heuristics.h` and `hopper/tile_scheduler.hpp`.

### 5.1 What Hopper changed

- **WGMMA.** Matrix multiplies are issued by a *warpgroup* (4 warps, 128 threads) and run asynchronously: operands come from SMEM (the A operand may come from registers), results land in registers, and the warps keep executing until they explicitly wait (`warpgroup_wait<N>()`).
- **TMA.** One thread issues a bulk copy of a whole tile, described by a tensor descriptor, from HBM to SMEM. Completion is tracked by a transaction-counting barrier in SMEM (an mbarrier). No address arithmetic in the main loop.
- **Register reallocation.** Warpgroups can give registers back and take more at run time (`warpgroup_reg_dealloc` / `warpgroup_reg_alloc`).
- **228 KB of SMEM per SM**, thread-block clusters, TMA multicast.
- **The exponential got relatively slower** (section 4.3): at `d = 128`, EX2 alone takes half as long as the MMAs.

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

The register numbers are the source's (`LoadRegisterRequirement` 24 and `MmaRegisterRequirement` 240 for two consumer warpgroups with TMA loads) and they are the whole point. A consumer thread holds its share of the fp32 score tile (`64 × 176 / 128` = 88 registers), the fp32 output accumulator (`64 × 128 / 128` = 64) and the bf16 probabilities (44), about 196 registers before addresses and statistics. The SM has 65,536 registers; `128 × 24 + 256 × 240 = 64,512`, so the producer's donation is what lets one CTA with two consumer warpgroups fit.

### 5.3 Ping-pong between warpgroups

The two consumer warpgroups take turns on the tensor cores through named barriers (`warp_scheduler_barrier_sync` / `warp_scheduler_barrier_arrive`): a warpgroup waits for its turn, issues its GEMMs, then signals the other one and does its softmax while the other's GEMMs run.

```
tensor cores   │ WG1: S, PV │ WG2: S, PV │ WG1: S, PV │ WG2: S, PV │ ...
MUFU / FP32    │ WG2 softmax │ WG1 softmax │ WG2 softmax │ WG1 softmax │ ...
               └──────────── time ───────────────────────────────────────>
```

The FA3 paper reports this schedule taking the FP16 forward at `d = 128` from about 570 to 620–640 TFLOP/s **(verify)**.

### 5.4 Pipelining inside one warpgroup

Within a warpgroup, the loop is skewed by one iteration. The source comment: "Each step does gemm0 for iter n_block, gemm1 for iter n_block + 1, and softmax for iter n_block" (the loop counts down, so `n_block + 1` is the previous block). In forward order:

```
step j:   issue  S_j = Q K_jᵀ               (WGMMA, async)
          issue  O  += P_(j−1) V_(j−1)      (WGMMA, async)
          wait for S_j only                 (warpgroup_wait<1>)
          softmax(S_j) -> P_j, α_j          <- MUFU/FP32 work overlaps the P·V GEMM still running
          wait for the P·V GEMM             (warpgroup_wait<0>)
          O *= α_j                          (rescale; O is now relative to m_j)
```

The price is register pressure (a score tile and a probability tile are live at the same time), and the tile size is squeezed by SMEM as well: the source notes that `128 × 192` hits the SMEM limit when `P` is kept in registers (`MmaPV_is_RS`) and `128 × 144` when it goes through SMEM, which is why the `d = 128` tile below is `128 × 176`.

### 5.5 Tile sizes (from `tile_size_fwd_sm90`)

| Head dim | bf16/fp16 `B_r × B_c` | FP8 `B_r × B_c` |
|---|---|---|
| 64 | 192 × 192 (192 × 128 causal, local, or paged without TMA) | 192 × 160 |
| 96 | 192 × 144 (192 × 128 local or paged without TMA) | 192 × 128 |
| 128 | 128 × 176 (128 × 128 causal, local, or paged without TMA) | 128 × 224 |
| 192 | 128 × 128 (128 × 112 if `d_v > 128`) | 128 × 160 |
| 256 | 128 × 80 ("128 x 80 hits the limit of smem") | 128 × 128 |

`B_c = 176` is not a power of two: the tile is sized to fill SMEM exactly. FP8 tiles are wider because each element is half the bytes.

### 5.6 FP8

- **Layout constraints.** Hopper's FP8 WGMMA wants both operands K-major in SMEM, which `V` is not for `P·V`. FA3 transposes `V` tiles in SMEM in the producer warpgroup (`Transpose_V = Is_FP8 && !V_colmajor`, a separate `pipeline_vt`), and permutes the score registers so the fp32 accumulator layout matches the FP8 operand layout (`permute_Cregs_fp8`).
- **Scales.** The interface takes `q_descale`, `k_descale`, `v_descale`; the main loop indexes them per (batch, KV head), folds `q_descale · k_descale` into the base-2 softmax scale, and applies `v_descale` to the output. vLLM passes them with shape `(num_sequences, num_kv_heads)`.
- **Using the FP8 range for P.** Probabilities are at most 1, and e4m3 has only 3 mantissa bits with a smallest subnormal of `2^−9`. FA3's FP8 softmax multiplies them by `2^8` (`Max_offset = 8`: "For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount"), which keeps small probabilities out of the flush-to-zero range; section 9 quantifies it.
- **Block quantization and incoherent processing (paper, (verify)).** One scale per tile rather than per tensor, and a random orthogonal rotation `M` (a Hadamard transform with random signs) applied to `Q` and `K`: `(QM)(KM)ᵀ = QMMᵀKᵀ = QKᵀ`, while an outlier channel is spread across all `d` channels. Via the fast Walsh-Hadamard transform the rotation costs `O(d log d)` per row and can be fused with the rotary embedding. The paper reports 2.6× lower numerical error than a baseline FP8 attention on inputs with outliers **(verify)**; section 9.4 shows when the rotation helps and when it does not.
- **Throughput (paper, (verify)).** bf16/fp16 forward up to 740 TFLOP/s (75% of 989), FP8 close to 1.2 PFLOP/s, 1.5–2.0× over FA2.

### 5.7 Blackwell and FlashAttention-4 **(verify)**

What changed in the hardware: MMAs (`tcgen05`) are issued by a single thread and accumulate into a new per-SM *Tensor Memory* (TMEM) instead of registers; two CTAs can cooperate on one MMA tile; dense bf16 throughput is about 2.25 PFLOP/s, 2.3× an H100, while the exponential unit and SMEM bandwidth did not scale with it. The table in section 4.3 shows the consequence: at `d = 128`, EX2 now takes as long as the matrix multiplies.

What FA4 does about it, as visible in `flash_attn/cute/flash_fwd_sm100.py` and `flash_attn/cute/softmax.py`:

- **More specialized warps.** Softmax warps 0–3 and 4–7 work on two Q tiles in a ping-pong (`q_stage = 2`); a separate *correction* warpgroup (warps 8–11) rescales the output in TMEM off the softmax's critical path; one warp (12) issues all MMAs; warp 13 runs the epilogue and warp 14 the loads.
- **Exponentials partly on the FMA pipe.** A tuned fraction of `exp2` calls is computed by a polynomial (`ex2_emulation_2`) instead of the MUFU (`ex2_emu_freq` 10 to 32 depending on the configuration; 0 on SM103, which "has fast native exp2").
- **Conditional rescaling** with `rescale_threshold = 8.0` (section 2.5).
- `P` is fed to the second MMA from TMEM (`OperandSource.TMEM`); 2-CTA MMA instructions (`use_2cta_instrs`).
- Written in CuTe-DSL (Python) and shipped as `flash-attn-4` (`from flash_attn.cute import flash_attn_func`, per the README).

Reported performance (from [the primer's section 8](flash-attention-primer.md#8-the-lineage), **(verify)**): up to 1,605 TFLOP/s bf16 on B200 (71% utilization), 1.3× cuDNN 9.13, 2.7× Triton; and a reminder that for decode FA4 was initially slower than FA2 until split-KV was ported. vLLM selects FA4 on SM100 when it is installed (`get_flash_attn_version` in `vllm/v1/attention/backends/fa_utils.py`), but for causal decoder attention on SM100 its backend priority list puts FlashInfer first (section 7.4).

---

## 6. Decode is a different kernel

### 6.1 One query row, a long KV cache: no reuse

In a decode step each sequence contributes one new query token per head. For one sequence, one layer and one KV head with `L` cached tokens:

```
bytes = 2·L·d·b                     read K and V once
FLOPs = 4·L·d·g                     g query heads share this KV head (GQA)
I     = 2g / b                      FLOP/byte:  MHA bf16 = 1,  g = 4 -> 4,  g = 8 -> 8;  FP8 KV doubles it
```

Every one of those is one to two orders of magnitude below the ridge (295 on an H100, 403 on an L4). Decode attention is pure bandwidth: its time is bytes divided by achieved bandwidth, and nothing else matters to first order. For a Llama-3-8B-shaped layer (32 query heads, 8 KV heads, `d = 128`) at `L = 32,768`: 134 MB of K/V and 537 MFLOP per layer, 40 µs at 3.35 TB/s, 1.28 ms per token across 32 layers for a single sequence (`fa_calculators.decode_intensity()`).

### 6.2 Why the prefill kernel is the wrong shape

Run the FA2 prefill kernel with `N_q = 1`:

- **Too few CTAs.** The grid is `ceil(1/B_r) × batch × heads`. With batch 1 and the 8 KV heads of a GQA model, that is 8 CTAs on a 132-SM H100: 94% of the SMs idle, and a handful of SMs cannot pull anywhere near full HBM bandwidth.
- **Each CTA walks all of `L` serially**, `L/B_c` tiles one after another.
- **A 128-row tile with 1 useful row** wastes 127/128 of every MMA. That is not the bottleneck when bandwidth-bound, but it shows the kernel is shaped for the wrong problem.

What matters for decode is the number of SMs issuing loads and the bytes per useful FLOP.

### 6.3 Flash-Decoding: split the KV sequence, then reduce

```
q (g rows: the query heads of one KV head)
K/V cache (L tokens):   ├── split 0 ──┼── split 1 ──┼── ... ──┼── split S−1 ──┤
one CTA per (batch, KV head, split):  (Ô_0, L_0)    (Ô_1, L_1)           (Ô_S−1, L_S−1)     fp32 partials in HBM
combine kernel:         L = logsumexp_s(L_s)        O = Σ_s exp(L_s − L) · Ô_s                 (section 2.3)
```

In FA2 this is `flash_fwd_splitkv_kernel`, launched on `dim3 grid(num_m_block, num_splits, b * h)`, followed by `flash_fwd_splitkv_combine_kernel`, which computes the max of the split LSEs, the log of the sum of exponentials, and the per-split weights `exp(lse − lse_logsum)` exactly as in section 2.3 (FA2 changelog 2.2: "we split the loading across different thread blocks, with a separate kernel to combine results"). The idea was published as Flash-Decoding by Dao, Haziza, Massa and Sizov in 2023, reporting up to 8× faster decoding at very long sequences **(verify)**.

**How many splits.** `num_splits_heuristic` in `csrc/flash_attn/flash_api.cpp`, ported in `fa_calculators.num_splits_heuristic()`:

1. If `batch × heads × query_blocks ≥ 0.8 × SMs`, do not split (the grid already fills the GPU). FA2 passes `2 × SMs` because two 128-thread CTAs fit per SM.
2. Otherwise compute the wave efficiency of each split count, `waves / ceil(waves)` with `waves = CTAs × splits / SMs`, skipping split counts that produce the same partition as a smaller one, and return the **smallest** count within 85% of the best. More splits than necessary only add partial results to write and combine.

FA3's version (`hopper/heuristics.h`) adds two rules: never split when there are 4 or fewer KV blocks, and split even when the GPU is full if one KV head is larger than a 50 MB L2 estimate, there are enough query blocks, and the mask is neither causal nor local.

Worked examples, Llama-3-8B shapes (32 query heads, 8 KV heads packed as in section 6.4, `d = 128`, `B_c = 128` in the split kernel), from `fa_calculators.fa2_decode_splits()`:

| Batch × context | GPU | CTAs without split | Splits chosen | CTAs launched | KV blocks per split |
|---|---|---|---|---|---|
| 1 × 32k | H100 (132 SMs) | 8 | 29 | 232 | 9 |
| 1 × 32k | A100 (108) | 8 | 24 | 192 | 11 |
| 1 × 32k | L4 (58) | 8 | 13 | 104 | 20 |
| 8 × 32k | H100 | 64 | 4 | 256 | 64 |
| 64 × 4k | H100 | 512 | 1 | 512 | 32 |

The cost is small: at 1 × 32k on an H100 the fp32 partials are `29 × 8 × 4 × 128 × 4 B` = 475 KB written and read back, against 134 MB of K/V per layer (0.7%). The cost that matters is elsewhere: the split count depends on batch size and SM count, so the order of the final sum does too. Section 9.3 comes back to this.

### 6.4 GQA packing

The `g` query heads that share a KV head need the same K and V. Treating them as `g` rows of one Q tile means K and V are loaded once per KV head rather than once per query head, and the MMA gets `M = g` useful rows instead of 1. All three libraries do it:

- FA2 swaps the group into the sequence dimension when `seqlen_q == 1 and num_heads > num_heads_k` (and there is no sliding window, ALiBi or dropout): "Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case" (`seqlenq_ngroups_swapped` in `flash_api.cpp`).
- FA3 decides with `should_pack_gqa`: pack when the unpacked tile efficiency `seqlen_q / round_up(seqlen_q, B_r)` is below 0.9 × the packed one `(seqlen_q·g) / round_up(seqlen_q·g, B_r)`; always for variable-length batches.
- FlashInfer's decode wrapper has `use_tensor_cores`: "Will be faster for large group size in grouped query attention."

The extreme case is MLA (section 8.1), where 128 query heads share one latent vector per token: packing turns decode attention from bandwidth-bound into nearly compute-bound.

### 6.5 Decode attention time is proportional to batch × context × KV bytes

Per decode step, attention must read every cached token of every sequence in every layer:

```
t_attention ≈ (Σ_sequences L_s) × 2·n_layers·H_kv·d·b / (achieved bandwidth)
```

The weights are read once per step whatever the batch. For a Llama-3-8B-shaped model in bf16 (8.03 B parameters, 16.06 GB of weights, 128 KB of KV per token per the [KV-cache primer](../kv-cache/kv-cache-primer.md)'s formula), on an H100 at 3.35 TB/s (weights alone: 4.79 ms per step):

| Batch × context | Tokens in batch | KV bytes | KV read time | Attention share of the step's bytes |
|---|---|---|---|---|
| 1 × 2k | 2,048 | 0.27 GB | 0.08 ms | 1.6% |
| 1 × 32k | 32,768 | 4.29 GB | 1.28 ms | 21% |
| 1 × 128k | 131,072 | 17.2 GB | 5.13 ms | 52% |
| 32 × 2k | 65,536 | 8.59 GB | 2.56 ms | 35% |
| 32 × 8k | 262,144 | 34.4 GB | 10.26 ms | 68% |
| 8 × 32k | 262,144 | 34.4 GB | 10.26 ms | 68% |

The share column is the same on any GPU (both terms divide by the same bandwidth); on an L4 at 0.30 TB/s every time is 11.2× longer, and only the first two rows fit next to 16 GB of bf16 weights in 24 GB. KV bytes equal weight bytes at about 122,500 cached tokens per batch: about 3,800 per sequence at batch 32. Past that point, KV compression (GQA, MLA, FP8 KV) and paging efficiency move decode throughput more than anything done to the matrix multiplies. Sizing TPOT from this is covered in [the capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md) (`fa_calculators.decode_attention_bytes()`).

---

## 7. Paged KV and serving kernels

[The paged-attention primer](../paged-attention/paged-attention-primer.md) explains why serving engines store the KV cache in fixed-size blocks. This section is about what that does to the attention kernel, and how two libraries and one engine wire it up. The engine around it (scheduler, KV manager, continuous batching) is the subject of [the serving-engine topic (04)](../serving-engine/).

### 7.1 What the kernel sees

A paged cache hands the kernel a pool and an indirection table instead of a tensor per sequence:

```
k_cache, v_cache : (num_blocks, block_size, H_kv, d)        one pool per layer, shared by all sequences
block_table      : (batch, max_blocks_per_seq) int32        logical block -> physical block, per sequence
seqused_k        : (batch,) int32                           valid tokens per sequence

logical token t of sequence b  ->  physical block  block_table[b][t // block_size],  row  t % block_size
```

The translation happens inside the main loop, per K/V tile. In FA2's split-KV kernel (`compute_attn_1rowblock_splitkv`):

```
block_table_idx    = n_block * kBlockN / page_block_size
block_table_offset = n_block * kBlockN - block_table_idx * page_block_size
K tile pointer     = k_ptr + block_table[block_table_idx] * k_batch_stride + block_table_offset * k_row_stride
```

The page size constrains the load path. FA2's public `flash_attn_with_kvcache` requires `page_block_size` to be a multiple of 256, so a K/V tile never straddles two pages. FA3 accepts any page size ("page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.)"), but small pages break the TMA's rectangular tile loads, so it falls back to per-row asynchronous copies (`paged_kv_non_TMA`), which also shrinks the tile (section 5.5). vLLM's FlashAttention backend accepts block sizes that are multiples of 16 through its own build of the kernels (`vllm.vllm_flash_attn`). The paged-attention primer quotes 20–26% kernel overhead for the original vLLM paged kernel; how much a given kernel pays depends on how contiguous each page's K/V rows are, which is a layout decision (vLLM's layout is at the end of section 7.4).

### 7.2 FlashAttention's serving APIs

- **`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ..., block_table=None)`**. A ragged batch packed along the token axis: `q` is `(total_q, H, d)` and `cu_seqlens_q` holds the prefix sums of the per-sequence lengths (section 8.2). With `block_table`, K and V come from pages.
- **`flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None, ..., num_splits=0, return_softmax_lse=False)`**. The decode entry point: optionally appends the step's new `k, v` into the cache in place, applies the rotary embedding, and attends, in one kernel. `num_splits=0` means "use the heuristic" (section 6.3). No backward.
- **FA3 additions** (`hopper/flash_attn_interface.py`): `page_table`; `cu_seqlens_q` and `max_seqlen_q` for multi-token queries against a cache (chunked prefill, speculative verification); `qv` for MLA (section 8.1); FP8 `q_descale`, `k_descale`, `v_descale`; `scheduler_metadata` from `get_scheduler_metadata(...)`, which precomputes the split and tile schedule once per batch so it can be reused by every layer; `pack_gqa`; `sm_margin` ("Can be tuned if some SMs are used for communication"); and `flash_attn_combine(out_partial, lse_partial)`, the split-KV reduction as a separate call.

One semantic to know for serving: since v2.1, when `N_q ≠ N_k` the causal mask is aligned to the **bottom-right** corner. For `N_q = 2`, `N_k = 5` the allowed pattern is `1 1 1 1 0 / 1 1 1 1 1`. That is exactly what a chunk of new tokens appended to a cache needs: each new token sees the whole cache plus the new tokens before it.

### 7.3 FlashInfer: plan once per batch, run once per layer

FlashInfer (Ye et al., 2025) is a kernel library used by SGLang, vLLM, TensorRT-LLM and others (README). Its decode and prefill wrappers split the work in two:

```
wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")     # 128 MB workspace recommended
wrapper.plan(kv_page_indptr, kv_page_indices, kv_last_page_len,          # once per engine step (host)
             num_qo_heads, num_kv_heads, head_dim, page_size, ...)
for layer in layers:
    o = wrapper.run(q[layer], kv_cache[layer])                           # once per layer (device)
```

- **The page table is in CSR form**: `indptr` (`batch + 1` offsets), `indices` (all page ids back to back), `last_page_len` (fill level of each sequence's last page). vLLM and FlashAttention use a padded 2D block table instead.
- **`plan()` does the scheduling on the host**: from the lengths it decides the split-KV partition and load balancing and writes auxiliary arrays into the workspace (the workspace also holds "intermediate attention results in the split-k algorithm"). The lengths are the same for every layer, so one plan serves all of them; the docstring's example plans once and runs 32 layers.
- **CUDA graphs**: `plan()` cannot run inside a CUDA graph or `torch.compile`; with `use_cuda_graph=True` the wrapper uses preallocated index buffers and the batch size is frozen.
- **What is behind `run()`**: decode kernels (with `use_tensor_cores=True` for large GQA groups), prefill/append kernels, MLA, cascade attention for shared prefixes, POD-attention (prefill and decode fused in one launch), block-sparse attention; selectable backends `auto`, `fa2`, `fa3`, `trtllm-gen`, `cute-dsl`, `cudnn` (decode wrapper docstring). The README lists SM 7.5 (T4) through Blackwell, with the caveat that not every feature exists on every architecture.
- **Determinism knob**: `fixed_split_size` fixes the split partition in pages, which "will lead to deterministic softmax score reduction in the merge_states kernel, and therefore batch-size invariant outputs" (section 9.3).

The design point: the split decision depends only on the batch's lengths, so computing it once per step on the host amortizes it over every layer and keeps the per-layer launches cheap and capturable.

### 7.4 How vLLM picks a backend and builds attention metadata

Read from `vllm/platforms/cuda.py`, `vllm/v1/attention/selector.py`, `vllm/v1/attention/backends/flash_attn.py` and `fa_utils.py` on `main`, fetched 2026-09-26; this code changes often.

**Selection.** For each attention layer vLLM walks a priority list and takes the first backend whose `validate_configuration()` accepts the layer (head size, dtype, KV-cache dtype, block size, compute capability, sinks, sliding window, MLA, ...). An explicit `attention_config.backend` overrides the list.

```
non-MLA attention on CUDA
  SM 10.x (B200, GB200), causal:   FLASHINFER > FLASH_ATTN > TRITON_ATTN > FLEX_ATTENTION > TURBOQUANT
  everything else:                 FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX_ATTENTION > TURBOQUANT
MLA on SM 9.0:                     FLASH_ATTN_MLA > FLASHMLA > FLASHINFER_MLA > TRITON_MLA > sparse variants

FlashAttention version:  SM 9.0 -> FA3;  SM 10.x -> FA4 if installed;  otherwise FA2;  ALiBi forces FA2
FLASH_ATTN accepts:      fp16/bf16; KV cache auto/fp16/bf16/fp8/fp8_e4m3; block size multiple of 16;
                         head size multiple of 8 and <= 256 (<= 512 with FA4); compute capability >= 8.0
```

On this repo's hardware tiers that means: a T4 (SM 7.5) gets `TRITON_ATTN`, because FlashAttention needs SM 8.0 and FlashInfer is currently floored at SM 8.0 in vLLM ("FlashInfer supports SM75+, but is currently broken on SM75 (Turing) ... Temporarily raise the floor to SM80"); an L4 (SM 8.9) gets `FLASH_ATTN` running FA2 kernels; an H100 gets `FLASH_ATTN` running FA3; a B200 gets `FLASHINFER` for causal decoder layers.

**Metadata.** Once per engine step, `FlashAttentionMetadataBuilder.build()` turns the scheduler's batch into what the kernel needs:

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

Two details from the source are worth quoting in a review. On splits: "Setting num_splits > 1 may increase the memory usage, because the intermediate buffers of size [num_splits, num_heads, num_tokens, head_size] are allocated. Therefore, we only set num_splits when using cuda graphs." And with `VLLM_BATCH_INVARIANT` set, `max_num_splits = 1` (section 9.3). For cascade attention, the shared prefix is attended once for all query tokens of the batch (batch 1, non-causal), the per-request suffixes separately (causal), and `merge_attn_states` combines the two with their LSEs: section 2.3's merge, used to avoid reading a shared system prompt's KV once per request.

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

The per-layer cache is one tensor of shape `[num_blocks, num_kv_heads, block_size, 2 × head_size]`, split into strided K and V views (`kv_cache.transpose(1, 2).split(head_size, dim=-1)`), which is why the kernels take arbitrary strides as long as the last dimension is contiguous.

---

## 8. Variants the kernel must support, and what they cost

A production attention kernel is a family of template instantiations. Each variant below is a branch inside the tile loop or a change to the loop bounds; the cost column says which resource it spends.

| Variant | What changes in the kernel | Cost | Notes from the sources |
|---|---|---|---|
| Causal | loop stops at the diagonal; diagonal tiles masked (section 4.4) | saves ≈ 50% of FLOPs at large `N`; creates load imbalance | bottom-right aligned when `N_q ≠ N_k` (v2.1) |
| Sliding window `(left, right)` | `n_block_min` and `n_block_max` bound the loop | work ∝ `N·W` instead of `N²/2`: at `N = 32k`, `W = 4,096`, 128 × 64 tiles, 15,840 tiles instead of 65,792 (24%) | FA2 v2.3 (Mistral 7B); FA3; FlashInfer `window_left` |
| ALiBi | adds `−slope·|i + N_k − N_q − j|` to each score | one FMA per score, nothing extra to load | vLLM falls back to FA2 when ALiBi is on ("Cannot use FA version 3 with ALiBi") |
| Softcapping | `s ← c·tanh(s/c)` before the softmax (Gemma-2, Grok) | one `tanh` per score on the same MUFU as `exp2`, the unit that section 4.3 shows is already the bottleneck | FA2 v2.6, FA3 (smaller FP8 tiles with softcap + local), FlashInfer `logits_soft_cap` |
| Dropout (training) | Philox random numbers generated per score in the kernel; the backward regenerates them from the saved seed and offset (`rng_state`) | RNG instructions per score; no mask stored | split-KV is not implemented with dropout ("SplitKV is not implemented for dropout") |
| MQA / GQA | K/V head index = `bidh / h_h_k_ratio` | free in prefill; decode gains come from packing (section 6.4) | query heads must be a multiple of KV heads |
| Head dim 64 / 128 / 256 | SMEM per tile ∝ `d`, accumulator registers ∝ `B_r·d` | larger `d` means smaller tiles and fewer CTAs per SM (FA2 on H100: 128 × 64 at `d = 128`, 64 × 64 at `d = 256`; FA3: 128 × 176 vs 128 × 80); the per-tile intensity `2B_r/b` does not depend on `d` | FA2 supports `d ≤ 256`; vLLM's FA backend `d % 8 == 0` |
| Attention sinks | an extra learnable per-head logit in the denominator (gpt-oss) | one extra term in `l` per row | passed as `s_aux` in vLLM's FA call; vLLM requires SM 9.0+ for sinks with FA |
| MLA | section 8.1 | turns decode compute-bound | FA3 `qv`, FlashMLA, FlashInfer MLA |
| Ragged batches | section 8.2 | no padding FLOPs; tile quantization remains | the varlen APIs |
| Across GPUs | section 8.4 | communication to hide | ring attention, context parallelism |
| Sparse masks | section 8.5 | tiles skipped by metadata | block-sparse kernels, FlexAttention |

### 8.1 MLA and the absorbed-weight decode trick

Multi-head latent attention (DeepSeek-V2 and V3; dimensions below are DeepSeek-V3's, **(verify)**) caches, per token and layer, a latent `c ∈ R^512` and one rotary key `k_rope ∈ R^64` shared by all 128 heads: 576 values, 1,152 bytes in bf16. The equivalent multi-head cache (128 heads × (192 key + 128 value dims)) would be 80 KB per token per layer, 71× more. Each head reconstructs its key and value from the latent: `k_h = [W_UK^h c ; k_rope]` and `v_h = W_UV^h c`, with `W_UK^h, W_UV^h ∈ R^(128×512)`.

Decode would reconstruct every cached token's K and V for every head, every step. The trick is to move the up-projections to the query and output side, where there is one token instead of `L`:

```
score_h,j = q_nope^h · (W_UK^h c_j) + q_rope^h · k_rope,j
          = ((W_UK^h)ᵀ q_nope^h) · c_j + q_rope^h · k_rope,j          define q~_h = (W_UK^h)ᵀ q_nope^h  (512-dim)

out_h     = Σ_j p_h,j (W_UV^h c_j) = W_UV^h ( Σ_j p_h,j c_j )          attend over c_j directly, then project
                                                                       (W_UV^h can be folded into W_O)
```

Decode becomes multi-query attention with 128 query heads and a single shared "KV head" whose key is `[c ; k_rope]` (576-dim) and whose value is `c` itself (512-dim, the same memory as the first 512 key dimensions). FA3 expresses exactly this: `q` carries the 64 rotary dims, `qv` the 512 absorbed dims, `k` is `k_rope` and `v` is `c`, and the kernel adds a second GEMM into the score accumulator (`HasQv`: `S = q·kᵀ + qv·vᵀ`); `tile_size_fwd_sm90` has a dedicated entry for `headdim = 64, headdim_v = 512`. The default softmax scale when `qv` is passed is `(64 + 512)^(−1/2)`; model code passes its own scale.

The arithmetic intensity is what makes MLA decode different: per cached token, `2·576·128` FLOPs for the scores plus `2·512·128` for the output, 278,528 FLOPs over 1,152 bytes, **242 FLOP/B in bf16** and 484 with an FP8 latent (`fa_calculators.mla_decode_intensity()`). That is at or above the H100's ridge: absorbed MLA decode is a compute-bound kernel, and FlashMLA and FA3's MLA path are tuned like GEMMs.

Prefill usually does not absorb. With `N` query tokens, attention in the latent space costs `576 + 512` dims per head for scores and outputs instead of `192 + 128`, 3.4× the quadratic FLOPs, while reconstructing K and V per head is linear in `N` and amortized over all queries.

### 8.2 Ragged batches (varlen)

Serving batches have unequal lengths. Padding them to the longest wastes FLOPs quadratically: lengths `[100, 3,000, 500]` padded to 3,000 cost `3 × 3,000² = 27.0 M` score pairs against `100² + 3,000² + 500² = 9.26 M` actual, 2.9×. The varlen APIs pack sequences back to back and pass `cu_seqlens`. The grid is sized by the longest sequence; each CTA computes its sequence's real bounds and exits at once if its Q block is past the end (`if (m_block * kBlockM >= binfo.actual_seqlen_q) return;` in FA2). What remains is tile quantization: a 100-token sequence still occupies a 128-row tile, 78% useful. FA3 packs GQA groups for all varlen batches to improve exactly this (section 6.4).

### 8.3 Head dimension, briefly

Why `d = 256` runs slower per FLOP than `d = 128`: the Q, K and V tiles grow with `d`, so the same SMEM holds fewer rows and fewer CTAs per SM (FA2 on H100 drops to 64 × 64 to keep two CTAs per SM), and the fp32 output accumulator `B_r × d` competes with the score tile for registers. The per-tile intensity (`B_r` FLOP/B in bf16) does not improve with `d`. Head dimensions that are not multiples of 8 are padded by FA2's Python wrapper (`torch.nn.functional.pad` to the next multiple of 8) and rejected by its C++ API; vLLM requires `d % 8 == 0`.

### 8.4 Across GPUs: ring attention and context parallelism

When one sequence does not fit on one GPU, shard it. **Ring attention** (Liu, Zaharia, Abbeel, 2023) gives each of `P` GPUs `N/P` tokens of `Q`, `K` and `V`. In each of `P` steps a GPU attends its `Q` shard to the K/V shard it currently holds, merges the result into its running `(m, l, o)` (section 2.3), and passes the K/V shard to its neighbour while it computes. Communication is hidden when per-step compute exceeds per-step transfer, per head:

```
4·(N/P)²·d / F  ≥  2·(N/P)·d·b / W        =>        N/P  ≥  b·F / (2·W)
```

For an H100 over NVLink (≈ 450 GB/s per direction, **(verify)**): `N/P ≥ 2,199` tokens at the 989 TFLOP/s peak, 1,333 at a more realistic 600 TFLOP/s; over a 400 Gb/s (50 GB/s) network link, 12,000 to 19,800 tokens per GPU (`fa_calculators.ring_min_tokens_per_gpu()`). With a causal mask the steps are uneven (the first shard's queries see almost nothing), which striped and zigzag partitions fix by interleaving tokens across GPUs. The alternative, all-to-all over heads (DeepSpeed-Ulysses), gives each GPU all tokens for `H/P` heads and needs `P ≤ H`. At serving time vLLM has decode context parallelism, which shards the KV cache across ranks and merges the partial outputs with their LSEs (`cp_lse_ag_out_rs`, `dcp_a2a_lse_reduce` in its FlashAttention backend).

### 8.5 Sparse and block-sparse attention, and FlexAttention

**Block-sparse attention** skips tiles by a block mask. The FA1 paper's block-sparse variant has IO complexity `Θ(Nd + N²d²M⁻¹s)`, with `s` the fraction of non-zero blocks **(verify)**. The cost model is simple: tiles visited × cost per tile, plus the metadata to find them. Sparsity finer than a tile saves nothing, because the tile is loaded and multiplied anyway. FA4 has block-sparse paths in its CuTe kernels, FlashInfer has block-sparse and variable block-sparse attention, and vLLM has dedicated backends for DeepSeek-style sparse MLA (`FLASHMLA_SPARSE`, `FLASHINFER_MLA_SPARSE`).

**FlexAttention** (PyTorch; Dong et al., 2024) removes the need for a new CUDA kernel per variant. You write a `score_mod(score, b, h, q_idx, kv_idx)` and/or a `mask_mod(b, h, q_idx, kv_idx)` in Python; `create_block_mask` evaluates the mask once per block and records, for each Q block, which K/V blocks are empty (skipped), partial (mask evaluated per element) and full (no mask code), the same three-way split as FA2's masked and unmasked loop phases, but driven by data; `torch.compile` inlines the functions into a Triton flash kernel. PyTorch reported forward performance around 85–90% of FlashAttention-2 **(verify)**. vLLM ships a `FLEX_ATTENTION` backend, and passes `mask_mod` functions to FA4 for masks the built-in flags cannot express (multimodal bidirectional prefixes, for example).

---

## 9. Numerics

### 9.1 bf16 in, fp32 accumulate

The tensor cores take bf16 or fp16 operands and accumulate in fp32, so `S` and the output accumulator are fp32 in registers (or TMEM). Two roundings to the input dtype happen: `P` is converted before every `P·V` multiply (`convert_type<Element>(acc_s)` in FA2), and `O` once at the end. The first dominates. bf16 keeps 8 significant bits, a unit roundoff of `2^−8 ≈ 0.39%` per probability; fp16 keeps 11 bits, `2^−11 ≈ 0.05%`. For attention, fp16 is the more accurate input format whenever its range suffices.

A kernel therefore cannot match an fp32 reference bit for bit, and it should not be expected to match another kernel either (different tile sizes sum in different orders). FlashAttention's own test criterion is the one to adopt: compare the kernel and a plain PyTorch implementation *in the same dtype* against an fp32 reference, and require the kernel's maximum error to be at most twice the baseline's (README: "the maximum numerical error of FlashAttention is at most twice the numerical error of a baseline implementation in Pytorch").

### 9.2 Folding the softmax scale

`τ` never touches the scores as a separate multiply. The max is taken on the raw fp32 scores, and `τ·log2 e` (plus, for FP8, `q_descale · k_descale`) is folded into the single FFMA that produces the exponent (section 2.4). Two consequences for reviewers: the scale must be positive for the max-before-scale order to be valid, and a model that needs an unusual scale (MLA with absorbed dimensions, section 8.1) must pass it explicitly, because the default is `1/√(head_dim)` of whatever tensor the kernel sees.

### 9.3 The log-sum-exp, empty rows, and determinism

The LSE is the interface between kernels. The backward needs it (section 3.5); split-KV, cascade attention, ring attention and decode context parallelism all merge partial results through it (section 2.3). Three details bite in practice:

- **Units.** Natural log in FA2 and FA3, base 2 in the Triton tutorial. Mixing them is off by a factor of `ln 2`.
- **Empty rows.** A row with no allowed key (a fully masked row, an empty split) has `l = 0`. FA2 reports `LSE = +∞` (`−∞` in split mode), FA3 `−∞`; vLLM's `merge_attn_states` maps `+∞` to `−∞` and returns 0 when both sides are empty ("FA2 and FA3 have different behavior for when the sum-exp is 0").
- **Batch invariance.** For a fixed configuration the forward pass is deterministic. But the number of KV splits depends on the batch size and the SM count (section 6.3), and a different number of splits sums the partial results in a different order, so the same request can produce slightly different logits depending on what else is in its batch. The fixes pin the reduction order: vLLM sets `max_num_splits = 1` under `VLLM_BATCH_INVARIANT`, and FlashInfer's `fixed_split_size` fixes the partition in pages (its docstring cites the "defeating nondeterminism in LLM inference" analysis). The backward has a second source: `atomicAdd` into `dQ` (section 3.5), removed by `deterministic=True`.

### 9.4 FP8: error sources and mitigations

The notebook's section 6 measures each source on synthetic data (seeded, numpy, `fa_calculators.round_to_e4m3()` emulating e4m3fn):

| Error source | Mechanism | Measured in the notebook | Mitigation |
|---|---|---|---|
| Mantissa rounding of Q, K, V | e4m3 keeps 3 mantissa bits: up to 6.25% relative error per element | `QKᵀ` relative error 3.6% with per-tensor scales (N = 256, d = 128) | none within FP8; keep the softmax, `l` and `O` in fp32; measure end-to-end quality |
| Outliers under amax scaling | one large channel sets the scale; with integer formats every other value loses precision | one channel × 20 in K: INT8 6.0% → 1.3% and INT4 52% → 24% with a random Hadamard rotation; e4m3 3.6% → 3.7% (unchanged: a float format's relative precision does not depend on magnitude) | per-block or per-head scales; incoherent processing (rotation), which matters most for integer KV caches and coarse scales |
| Small probabilities in `P` | `P ≤ 1`; e4m3's smallest subnormal is `2^−9` | a 4,096-key row with N(0, 2²) scores: 61.9% of probabilities flush to zero and 2.7% of the probability mass is lost; with FA3's `2^8` offset, 0.7% flush and the output error drops from 2.2% to 1.6% | scale `P` by `2^8` before conversion (`Max_offset`), accumulate `l` in fp32 from unrounded values |
| Scale choice | stale or per-tensor scales clip or waste range | not simulated | calibrated KV scales (vLLM `k_scale`, `v_scale`), per-(batch, head) descales as in FA3 |

Two review points follow. First, "FP8 attention" can mean FP8 KV cache with bf16 compute (bandwidth win for decode, section 6.1) or FP8 matrix multiplies (compute win for prefill, section 5.6); they have different error budgets. Second, a kernel-level error metric is not a quality metric: accept FP8 attention on the basis of an end-to-end evaluation on the workload's own data.

---

## 10. Measuring attention kernels

### 10.1 Count by a stated convention

- **Prefill / training:** `4·N_q·N_k·d` FLOPs per head for the forward, halved for causal, ×2.5 for the backward, ×3.5 for forward plus backward (section 1.1; the Triton tutorial's `bench_flash_attention` uses exactly this). Softmax FLOPs are not counted. Some tools count causal attention at full cost or include softmax work; a number without its convention is not comparable.
- **Decode:** report bandwidth. Bytes = K/V actually read (`2·L·d·b` per KV head per sequence) plus Q and O; divide by time; compare with the HBM peak. TFLOP/s for a decode kernel is a meaningless number (the intensity is 1 to 8 FLOP/B, section 6.1).

### 10.2 Which side of the roofline to report

A prefill kernel lives on the compute side, so the honest figure is TFLOP/s as a fraction of the dense peak for that dtype (989 bf16 on an H100, not the 1,979 sparse headline). A decode kernel lives on the bandwidth side, so the figure is GB/s as a fraction of HBM bandwidth. Split-KV and paged kernels should report both the bytes they had to read and the bytes they did read (partials, block tables, padding).

### 10.3 Procedure

1. **Correctness first.** Compare against an fp32 reference with the criterion of section 9.1, for causal and non-causal, odd lengths (`N` not a multiple of the tile), several head dims, GQA, and the LSE if it is returned.
2. **Confirm which kernel ran.** `torch.nn.functional.scaled_dot_product_attention` silently falls back to another backend when a constraint is not met (dtype, head dim, mask type, GPU generation). Force the backend (`torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`), which raises instead of falling back, and check kernel names in a profiler trace. In vLLM, the startup log names the selected attention backend.
3. **Warm up.** The first calls pay for lazy initialization, JIT compilation, autotuning and the caching allocator.
4. **Time on the device.** Use CUDA events or `torch.utils.benchmark.Timer` (which synchronizes and repeats); report the median and the spread, not the minimum.
5. **Mind the L2.** A decode-sized problem can sit in L2 from the previous repetition (one layer of one 4k sequence of a Llama-3-8B-shaped model is 16.8 MB, against 50 MB of L2 on an H100) and report more than the HBM bandwidth. Flush between repetitions (`triton.testing.do_bench` does this by default **(verify)**) or use a working set larger than L2.
6. **Mind launch overhead.** A decode attention kernel can run in tens of microseconds, the same order as launch and Python overhead. Serving engines capture decode steps in CUDA graphs for this reason; benchmark the same way.
7. **Clocks and power.** Long runs throttle. Either lock clocks for A/B comparisons or report the clock you observed.
8. **Sweep the shapes that matter**: `N`, `d`, batch × heads (does the grid fill the GPU?), causal, GQA group, page size.

### 10.4 A minimal honest benchmark (T1)

Not executed here (no GPU in this environment); run it on any CUDA GPU with PyTorch 2.3 or later.

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

On a T4 the `sdpa_kernel` context raises, because PyTorch's flash backend needs SM 8.0 or newer **(verify)**; that is the point of forcing it. Use `SDPBackend.EFFICIENT_ATTENTION` there, in fp16.

### 10.5 What to expect, by GPU class

Forward pass, `d = 128`, long sequences, dense peak in parentheses. All of these are either cited or explicitly assumptions; measure before relying on any of them.

| GPU | Kernel | Expected | Basis |
|---|---|---|---|
| A100 80GB (312) | FA2 | 50–73% of peak, ≈ 156–228 TFLOP/s | FA2 paper **(verify)** |
| H100 SXM (989) | FA2 | ≈ 35%, ≈ 350 TFLOP/s | FA3 paper **(verify)** |
| H100 SXM | FA3 bf16 / FP8 | up to 740 TFLOP/s (75%) / close to 1,200 | FA3 paper **(verify)** |
| B200 (2,250) | FA4 bf16 | up to 1,605 TFLOP/s (71%) | FA4 paper via the primer **(verify)** |
| L4 (121) | FA2 (sm89 tiles, section 4.5) | no source read; if it reaches the A100's 50–70% fraction, 60–85 TFLOP/s | assumption **(verify by measuring)** |
| T4 (65, fp16) | FA2 unsupported (README: Turing needs a separate project) | SDPA falls back to its memory-efficient kernel | README |
| any | split-KV decode | bandwidth-bound: compare with a plain device-to-device copy on the same GPU as the practical ceiling | section 6 |

### 10.6 How attention's share of a step changes with context

**Prefill.** Per token and layer, the linear layers cost `2 × parameters` FLOPs; causal attention costs `4 × context × H × d × 0.5` on average over the prompt. For a Llama-3-8B-shaped layer (218.1 M parameters, 32 heads of 128) (`fa_calculators.prefill_attention_share()`):

| Prompt length | 2k | 8k | 32k | 128k |
|---|---|---|---|---|
| attention FLOPs / linear FLOPs | 0.04 | 0.15 | 0.62 | 2.46 |

The two are equal at 53,248 tokens. Because attention usually runs at a lower fraction of peak than the large GEMMs, its share of *time* crosses over earlier than its share of FLOPs.

**Decode.** Section 6.5: the weights are read once per step, the KV cache once per cached token, so attention's share grows with batch × context. For the same model, KV bytes pass weight bytes at about 122,500 cached tokens per batch.

Both effects push long-context serving towards the same place: at 32k tokens and beyond, attention, not the MLP, decides throughput, and KV size decides capacity ([the KV-cache primer](../kv-cache/kv-cache-primer.md), [the capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md)).

---

## 11. Implementing it yourself: an FA2 forward in Triton

[`flash_attention_minimal.py`](flash_attention_minimal.py) (part 5) is the whole algorithm in numpy, and [the practice notebook](flash_attention_practice.ipynb) has you write it. This section turns it into a GPU kernel. Triton lets you write it at the level of tiles (a program instance works on a `BLOCK_M × BLOCK_N` tile; the compiler handles warps, shared memory, and pipelining of loads), which makes the mapping from the algebra direct. The upstream reference is the Triton tutorial `python/tutorials/06-fused-attention.py` (read for this page), which adds warp specialization and tensor descriptors (TMA) on newer GPUs.

### 11.1 The kernel

**Tier T1; not executed in this repository** (no GPU here). Written against Triton 3.x block pointers (`tl.make_block_ptr`); inputs `(B, H, N, D)`, contiguous, fp16 or bf16.

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def attn_fwd(Q, K, V, O, LSE, sm_scale, N_CTX,
             stride_b, stride_h, stride_n,                       # shared by Q, K, V, O; last dim contiguous
             H: tl.constexpr, D: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)                                     # (1) which Q block
    pid_bh = tl.program_id(1)                                    # (2) which (batch, head)
    base = (pid_bh // H) * stride_b + (pid_bh % H) * stride_h
    q_ptr = tl.make_block_ptr(Q + base, shape=(N_CTX, D), strides=(stride_n, 1),
                              offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    kt_ptr = tl.make_block_ptr(K + base, shape=(D, N_CTX), strides=(1, stride_n),   # K transposed view
                               offsets=(0, 0), block_shape=(D, BLOCK_N), order=(0, 1))
    v_ptr = tl.make_block_ptr(V + base, shape=(N_CTX, D), strides=(stride_n, 1),
                              offsets=(0, 0), block_shape=(BLOCK_N, D), order=(1, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)          # (4) identity state
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634                     # (5) fold log2(e)
    q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")          # (3) Q stays on chip

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)            # (6) stop at the diagonal
    for start_n in range(0, hi, BLOCK_N):
        kt = tl.load(kt_ptr, boundary_check=(1,), padding_option="zero")
        s = tl.dot(q, kt) * qk_scale                              # (7) scores, fp32, log2 units
        valid = (start_n + offs_n)[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= (start_n + offs_n)[None, :])
        s = tl.where(valid, s, float("-inf"))                    # (8) ragged tail and diagonal
        m_new = tl.maximum(m_i, tl.max(s, 1))                    # (9)
        alpha = tl.math.exp2(m_i - m_new)                        # (10)
        p = tl.math.exp2(s - m_new[:, None])                     # (11)
        l_i = l_i * alpha + tl.sum(p, 1)                         # (12)
        v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)    # (13) P rounded to the input dtype
        m_i = m_new
        kt_ptr = tl.advance(kt_ptr, (0, BLOCK_N))                # (14)
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]                                     # (15) normalize once
    tl.store(LSE + pid_bh * N_CTX + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX)   # (16) base 2
    o_ptr = tl.make_block_ptr(O + base, shape=(N_CTX, D), strides=(stride_n, 1),
                              offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    tl.store(o_ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))


def flash_attention(q, k, v, causal=True, block_m=128, block_n=64):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, block_m), B * H)                      # the FA2 grid: Q blocks x (batch, head)
    attn_fwd[grid](q, k, v, o, lse, 1.0 / math.sqrt(D), N, q.stride(0), q.stride(1), q.stride(2),
                   H=H, D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal, num_warps=4, num_stages=2)
    return o, lse
```

### 11.2 Line by line

| Kernel line | What it is | Section | In `flash_attention_minimal.py` |
|---|---|---|---|
| (1) `pid_m = tl.program_id(0)` | the outer loop over Q blocks becomes the grid | 4.1 | `for i in range(0, N, block_q)` |
| (2) `pid_bh = tl.program_id(1)` | parallelism over batch and heads | 4.1 | one head only |
| (3) `q = tl.load(q_ptr)` once | Q block loaded once, kept on chip | 4.1 | `Qi = Q[i:i + block_q]` |
| (4) `m_i, l_i, acc = −∞, 0, 0` | the merge identity | 2.3 | `mi`, `li`, `Oi` initialized |
| (5) `qk_scale = sm_scale · log2 e` | scale folded into the exponent | 2.4, 9.2 | `scale` and `np.exp` |
| (6) `hi` for `CAUSAL` | causal block skipping | 4.4 | `if causal and j > i + bq - 1: continue` |
| (7) `s = tl.dot(q, kt) * qk_scale` | first GEMM, fp32 accumulate | 1.1, 9.1 | `Sij = (Qi @ Kj.T) * scale` |
| (8) `tl.where(valid, s, −∞)` | diagonal tile mask and ragged tail | 4.4, 8.2 | `np.where(cols <= rows, Sij, -np.inf)` |
| (9) `m_new = max(m_i, rowmax(s))` | new reference max | 2.2 | `m_new = np.maximum(mi, Sij.max(axis=1))` |
| (10) `alpha = exp2(m_i − m_new)` | rescale factor for the old state | 2.2 | `alpha = np.exp(mi - m_new)` |
| (11) `p = exp2(s − m_new)` | probabilities relative to the new max | 2.2, 2.4 | `Pij = np.exp(Sij - m_new[:, None])` |
| (12) `l_i = l_i·alpha + rowsum(p)` | running denominator | 2.2 | `li = alpha * li + Pij.sum(axis=1)` |
| (13) `acc = acc·alpha + dot(p.to(bf16), v)` | second GEMM; `P` stays in registers, rounded to the input dtype | 2.2, 4.2, 9.1 | `Oi = alpha[:, None] * Oi + Pij @ Vj` |
| (14) `tl.advance(...)` | next K/V tile | 3.1 | `Kj, Vj = K[j:j + block_kv], V[j:j + block_kv]` |
| (15) `acc / l_i` | normalize once, at the end | 4.3 | `O[i:i + bq] = Oi / li[:, None]` |
| (16) store `m_i + log2(l_i)` | LSE for the backward, base 2 | 2.4, 3.5 | `L[i:i + bq] = mi + np.log(li)` (natural log) |

What the Triton version adds to the numpy one, and what it still leaves out:

- **Masking of ragged tails** (the last Q and K/V blocks when `N` is not a multiple of the tile) through `boundary_check`, `padding_option` and the `valid` mask.
- **No `−∞` guard is needed here**: the loop runs forward from key 0, which every row may see, so no row's first tile is fully masked. A kernel that walks backwards from the diagonal (FA2, section 4.4), uses a sliding window, or aligns the causal mask bottom-right with `N_q > N_k` needs the guard of section 2.5.
- **Pipelining** comes from `num_stages`: the compiler double-buffers the K and V loads, the `cp.async` scheme of section 4.5.
- **Left out**: the backward pass (the notebook's section 4 is the algorithm), variable-length batches, paged K/V, GQA packing, split-KV, FP8, warp specialization. Each is a section of this page.

### 11.3 Testing it

Against an fp32 reference, with the criterion of section 9.1:

```python
q, k, v = (torch.randn(2, 8, 1000, 128, device="cuda", dtype=torch.float16) for _ in range(3))   # N not a tile multiple
o, lse = flash_attention(q, k, v, causal=True)
ref = naive(q.float(), k.float(), v.float())                     # naive() from section 10.4
assert (o.float() - ref).abs().max() <= 2 * (naive(q, k, v).float() - ref).abs().max()
s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(128)
s = s.masked_fill(torch.ones(1000, 1000, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
assert torch.allclose(lse.view(2, 8, 1000) * math.log(2), torch.logsumexp(s, dim=-1), atol=1e-3)   # base 2 -> natural
```

Then vary `block_m`/`block_n` (the result must not depend on them beyond rounding), `causal`, `N` (1, a tile multiple, a tile multiple plus one), and time it with the procedure of section 10.3. On a T4, use fp16 (it has no bf16 tensor cores) and check that Triton's `tl.dot` uses its tensor cores there **(verify)**.

---

## In a design review

### The two-minute walkthrough

"Attention is memory-bound when written the obvious way, and not because of the softmax alone: every kernel in the naive schedule sits below the ridge, including the two GEMMs (128 FLOP per byte in bf16, because the reduction dimension is only the head dimension), and the schedule as a whole tends to `d/b`, 64 FLOP per byte, while an H100 needs 295 and an L4 403 to be compute-bound. Longer sequences don't help; they only make the `N×N` intermediate stop fitting.

"FlashAttention keeps the math and changes the schedule. The key fact is that a partial softmax result, stored as a running max, a running sum and an unnormalized output, merges exactly with any other partial result. So we tile: each thread block owns a block of queries, streams K and V tiles through on-chip memory, and writes the output once. The only thing saved for the backward is one log-sum-exp per row, and the backward recomputes the probabilities, 2.5× the forward FLOPs, which is cheaper than moving `N²` values.

"Each generation then chased the next bottleneck. FA2 fixed work partitioning: queries in the outer loop, warps split by rows so nothing goes through shared memory, parallelism over sequence length. FA3 exists because on Hopper the exponential unit takes half as long as the matrix multiplies: it overlaps them with asynchronous tensor-core instructions, a producer warpgroup on the TMA and two consumer warpgroups in ping-pong. On Blackwell the exponential takes as long as the multiplies, so FA4 emulates some exponentials on the FMA units and skips most rescales.

"Serving is a different regime. Decode has one query row per head against a long cache: no reuse, 1 to 8 FLOP per byte, so the kernel splits the cache across thread blocks, packs the query heads of a GQA group into one tile, and merges partial results with the same operator. Its time is batch × context × KV bytes over bandwidth. vLLM passes a block table and per-sequence lengths into one variable-length call per layer; on Hopper that runs FA3, on Blackwell FlashInfer goes first, on a T4 it falls back to Triton."

### Drill questions

**1. Why can't a bigger batch or a longer sequence make naive attention compute-bound?**
Because FLOPs and bytes both scale with `B·H·N²`; the ratio is `N·d / (b(N + d))`, which tends to `d/b`. The only levers are `d`, the dtype, and the schedule.

**2. FlashAttention's backward does an extra matrix multiply. Why is it still faster?**
It recomputes `P` from `S − L` (`2N²d` FLOPs, 4.3 µs per head at `N = 4k` on an H100 at peak) instead of writing `P` in the forward and reading it in the backward (67 MB per head, 20 µs of HBM time), and it avoids holding `N²` per head in memory between the passes. Surplus FLOPs buy scarce bandwidth.

**3. What exactly is saved from the forward, and why is it enough?**
`O` and `L = τm + ln l` per row (plus the RNG state if dropout is on). With `L`, `P_ij = exp(S_ij − L_i)` needs no max and no sum, and `D_i = rowsum(dO_i ∘ O_i)` replaces the softmax Jacobian's row reduction.

**4. What did FA2 change, and which change matters most on a small batch?**
Loop order (one CTA per Q block, output in registers, written once), split-Q warp partitioning (no shared-memory exchange), deferred normalization. On a small batch the decisive one is the grid over sequence blocks: batch 1 with 16 heads is 16 CTAs without it and 2,048 with it at 16k tokens.

**5. Our H100 runs FA2 at about 35% of peak. What changes with FA3, and why is it Hopper-specific?**
Asynchronous warpgroup MMAs and TMA loads, a producer warpgroup that donates registers to two consumers, ping-pong so one warpgroup's softmax runs under the other's GEMMs, and a skewed loop inside each warpgroup. It is Hopper-specific because those instructions are; the reason it is needed is that at `d = 128` the exponentials take 50% as long as the MMAs on an H100 (16 EX2 per SM per clock against 4,096 tensor FLOPs).

**6. The same attention call is fast in prefill and slow in decode. Why, and what should run instead?**
With one query row, the prefill grid is batch × heads CTAs (8 for batch 1 on a GQA model) and each walks the whole cache alone, so most SMs idle and HBM is never saturated. Use split-KV (29 splits, 232 CTAs for batch 1 × 32k on an H100) with GQA packing, and a combine step over `(Ô, LSE)` partials that costs under 1% extra traffic.

**7. How does vLLM pick the attention kernel on an L4, an H100, a B200 and a T4?**
It walks a per-platform priority list and takes the first backend that validates the layer. Ampere, Ada and Hopper: FlashAttention first (FA2 kernels on the L4, FA3 on the H100). SM 10.x with causal attention: FlashInfer first. T4: FlashAttention needs SM 8.0 and FlashInfer is currently floored at SM 8.0 in vLLM, so Triton attention. ALiBi forces FA2.

**8. We want FP8 attention for a 128k-context deployment. What can go wrong and how do we check?**
Decide first whether it is an FP8 KV cache (decode bandwidth) or FP8 matrix multiplies (prefill compute). Error sources: 3-bit mantissa rounding (no fix within FP8), outliers under amax scaling (per-block scales, rotations; rotations matter most for integer formats), small probabilities flushing to zero (the `2^8` offset), stale scales (calibration). Check with the workload's end-to-end evaluation, not a kernel error metric, and include long-context cases.

---

## Glossary

| Term | Meaning |
|---|---|
| Arithmetic intensity | FLOPs per byte moved from HBM. Below the ridge point a kernel is memory-bound. |
| Ridge point | Peak FLOP/s ÷ HBM bandwidth: 295 FLOP/B for an H100 in bf16, 403 for an L4. |
| HBM / SMEM / L2 | Device memory; per-SM software-managed shared memory; the GPU-wide cache between them. |
| CTA | Cooperative thread array: a thread block, scheduled on one SM. |
| Warp / warpgroup | 32 threads executing together / 4 warps (128 threads) that issue one Hopper WGMMA. |
| MMA, WGMMA, tcgen05 | Tensor-core matrix multiply-accumulate: warp-level (Ampere), warpgroup-level and asynchronous (Hopper), single-thread-issued into Tensor Memory (Blackwell). |
| TMA | Tensor Memory Accelerator: Hopper's bulk asynchronous copy of a tile described by a descriptor. |
| `cp.async` | Ampere's asynchronous global-to-shared copy, used by FA2 for double buffering. |
| mbarrier | A shared-memory barrier that also counts bytes, used to signal TMA completion and pipeline stages. |
| TMEM | Blackwell's per-SM Tensor Memory holding MMA accumulators. |
| MUFU / EX2 | The special-function unit and its base-2 exponential instruction; 16 per SM per clock on A100 and H100. |
| FFMA | Fused floating-point multiply-add; the exp2 trick makes the exponent argument one FFMA. |
| Online softmax | Computing softmax in one pass with a running max and a rescaled running sum. |
| `(m, l, o)` state | Reference max, sum of exponentials, unnormalized output: the mergeable partial result. |
| LSE | Log-sum-exp of a row's scaled scores, `τm + ln l`: saved for the backward, used to merge partials. |
| Split-Q / split-K | Dividing a tile among warps by query rows (FA2, no exchange) or by key columns (FA1, exchange through SMEM). |
| Split-KV, Flash-Decoding | Splitting the K/V sequence across CTAs and merging `(Ô, LSE)` partials in a second kernel. |
| GQA packing | Putting the query heads that share a KV head into one tile so K/V are read once per KV head. |
| Wave efficiency | `waves / ceil(waves)` for a grid of CTAs over the SMs: how full the last wave is. |
| Tile quantization | Work wasted when a length is not a multiple of the tile size. |
| Block table / page table | Per-sequence map from logical KV blocks to physical blocks in a paged cache. |
| `cu_seqlens` | Prefix sums of sequence lengths for a packed, variable-length batch. |
| Warp specialization | Different warps (or warpgroups) doing different jobs, such as loading versus computing. |
| Ping-pong | Two consumer warpgroups alternating on the tensor cores so each one's softmax overlaps the other's GEMMs. |
| Conditional rescaling | Keeping a stale reference max until it is exceeded by a threshold (FA4). |
| Incoherent processing | Rotating Q and K by the same orthogonal matrix before quantizing, to spread outliers. |
| `Max_offset` | FA3's `2^8` scaling of FP8 probabilities to use e4m3's range. |
| Cascade attention | Attending a shared prefix once for all requests and merging with per-request suffix attention. |
| Persistent kernel, LPT | A kernel whose CTAs loop over work items; longest-processing-time-first ordering for uneven (causal) tiles. |
| MLA, absorbed decode | Multi-head latent attention; moving the key/value up-projections to the query and output side so decode attends over the latent cache directly. |

---

## Sources

**Papers and posts.** Not fetched: arXiv and the blogs were unreachable from this environment on 2026-09-26. Cited from the literature; figures from them are marked **(verify)** in the text.

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

**Upstream code read for this page** (fetched 2026-09-26 from `raw.githubusercontent.com`, branch `main`; quotes in the text are from these files):

- `Dao-AILab/flash-attention`: `README.md`; `flash_attn/flash_attn_interface.py`; `flash_attn/flash_attn_triton.py`; `csrc/flash_attn/flash_api.cpp`; `csrc/flash_attn/src/flash_fwd_kernel.h`, `softmax.h`, `flash_fwd_launch_template.h`, `flash_bwd_kernel.h`; `hopper/flash_attn_interface.py`, `flash_fwd_kernel_sm90.h`, `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `softmax.h`, `tile_size.h`, `heuristics.h`, `tile_scheduler.hpp`; `flash_attn/cute/README.md`, `flash_fwd_sm100.py`, `softmax.py`. (`hopper/README.md` returned 404; FA3 installation notes are in the top-level README.)
- `triton-lang/triton`: `python/tutorials/06-fused-attention.py`.
- `vllm-project/vllm`: `vllm/v1/attention/backends/flash_attn.py`, `fa_utils.py`, `flashinfer.py`, `triton_attn.py`; `vllm/v1/attention/selector.py`; `vllm/platforms/cuda.py`; `vllm/v1/attention/ops/merge_attn_states.py`, `triton_merge_attn_states.py`.
- `flashinfer-ai/flashinfer`: `README.md`, `flashinfer/decode.py`, `flashinfer/cascade.py`.

**In this repository:** [the FlashAttention primer](flash-attention-primer.md), [`flash_attention_minimal.py`](flash_attention_minimal.py), [the practice notebook](flash_attention_practice.ipynb), [the paged-attention primer](../paged-attention/paged-attention-primer.md), [the KV-cache primer](../kv-cache/kv-cache-primer.md), [the transformer primer](../../00-foundations/transformers/docs/transformer-primer.md), [the GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md), [the roofline primer (01)](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), [the CUDA primer (02)](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md), [the capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md).

---

## Verify list (2026-09-26)

Re-check these before quoting them; everything else on the page is derived from stated inputs or quoted from the files listed above.

1. **Paper figures.** FA1: Theorem 2 and Proposition 3 as stated; 40.3 GB vs 4.4 GB HBM traffic (GPT-2-sized, A100); block-sparse IO bound. FA2: ≈ 2× over FA1; 50–73% of A100 peak forward vs 25–40% for FA1; FA1 parallelized over batch and heads only; backward at a lower fraction of peak. FA3: FA2 at ≈ 35% on H100; 740 TFLOP/s (75%) bf16; ≈ 1.2 PFLOP/s FP8; 1.5–2.0× over FA2; ping-pong 570 → 620–640 TFLOP/s; 3.9 TFLOP/s of special functions; 2.6× lower FP8 error; Hadamard fused with RoPE. FA4: 1,605 TFLOP/s (71%) on B200, 1.3× cuDNN 9.13, 2.7× Triton, and the paper's title. Flash-Decoding: up to 8×. FlexAttention: 85–90% of FA2.
2. **Hardware.** Dense bf16 peaks and HBM bandwidth: H100 SXM 989.4 TFLOP/s, 3.35 TB/s, 132 SMs, 228 KB SMEM/SM, 50 MB L2; A100 80GB 312, 2.039 TB/s, 108 SMs; L4 121, 0.30 TB/s, 58 SMs, 48 MB L2; B200 2,250, 8.0 TB/s; T4 65 (fp16), 0.32 TB/s. Per-SM per-clock throughputs in section 4.3, especially the B200 column (≈ 8,192 tensor FLOPs, 128 FP32 FMA, 16 MUFU) and the Blackwell hardware description (tcgen05, TMEM, 2-CTA MMA). H100 NVLink ≈ 450 GB/s per direction.
3. **Model shapes.** Llama-3-8B: 8.03 B parameters, 32 layers, 32 query and 8 KV heads of 128, MLP width 14,336. DeepSeek-V3 MLA: 128 heads, latent 512, rotary 64, no-PE key 128, value 128.
4. **Moving code facts** (read on `main`, 2026-09-26): vLLM backend priority lists and FA version selection; vLLM's SM 8.0 floor for FlashInfer; FA2's `page_block_size` multiple of 256; FA3's arbitrary page size; FA3 tile sizes and register counts; FA4's `rescale_threshold = 8.0`, `ex2_emu_freq` values and warp roles; FlashInfer backends and `fixed_split_size`.
5. **Tooling.** `triton.testing.do_bench` flushing L2 by default; PyTorch's flash SDPA backend requiring SM 8.0+; Triton `tl.dot` using tensor cores on a T4 in fp16.
6. **L4 expectation** in section 10.5 is an assumption, not a measurement: measure it.
