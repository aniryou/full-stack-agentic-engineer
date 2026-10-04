# %% [markdown]
# # 02 · LLM inference on the roofline
#
# **Tier:** T0. It is pure arithmetic on a model config and a device, with no GPU. To see real numbers,
# run a small model on a T4/L4 with `04-inference-engine/serving-engine/vllm-serving-lab` (T1).
# Then compare the result with the ceilings that this notebook calculates.
#
# ## The one-minute version
# One engine step streams every weight from HBM one time, whatever the number of tokens in the step. But
# the FLOPs grow with the tokens. Thus the arithmetic intensity of a step is approximately **its token count**
# (at 2-byte weights). A 2K-token prefill is far to the right of the ridge: it is compute-bound, and TTFT
# is FLOPs ÷ peak. A batch-1 decode step is at ~1 FLOP/B: it is memory-bound, and the time per token is
# bytes ÷ bandwidth.
#
# Batching moves the weight GEMMs of decode to the right, toward the ridge. But each sequence also has
# **KV-cache reads**, and these grow with context exactly as fast as the attention FLOPs of the sequence.
# The KV-cache reads keep attention at a few FLOP/B, and they come to dominate the step.
#
# After this notebook, you can do these things:
#
# - Say which bound applies (per step and per kernel).
# - Predict the step time within the roofline.
# - Explain what quantization and MoE really change: the **bytes**.
# - Explain why tiling and fusion give a gain on the memory hierarchy.
#
# Primer: `../PRIMER.md` §3–4.

# %%
from roofline import llm, specs

h100, l4, h200 = specs.get("h100-sxm"), specs.get("l4"), specs.get("h200")
m8 = llm.PRESETS["llama-3.1-8b"]
print(m8)
print(f"params {m8.params() / 1e9:.2f}B | per-token matmul params {m8.matmul_params_per_token() / 1e9:.2f}B "
      f"| LM head {m8.lm_head_params() / 1e6:.0f}M | KV per token {m8.kv_bytes_per_token():,.0f} B (bf16)")
print("weights streamed by one decode step:", f"{llm.streamed_weight_bytes(m8, 1) / 1e9:.2f} GB",
      "(the input embedding is a gather of one row per token, not a stream of the table)")

# %% [markdown]
# ## Prefill: intensity ≈ tokens in the step
# The engine reads the weights one time for the whole prompt. The FLOPs are $2 \times \text{tokens} \times \text{params}$ plus causal
# attention. Watch the bound change near the ridge (295 FLOP/B on an H100 in bf16). Here, the unit of that point is
# *tokens*. That is the reason that engines give prefill chunks of a few hundred to a few thousand tokens.

# %%
print(f"{'prompt':>7s} {'FLOP/B':>7s} {'bound':>8s} {'time ms':>8s}")
for s in (64, 128, 256, 300, 320, 512, 2048, 8192):
    st = llm.prefill(m8, h100, s)
    print(f"{s:7d} {st.intensity:7.0f} {st.bound:>8s} {st.time * 1e3:8.2f}")
st = llm.prefill(m8, h100, 2048)
print(f"\n2K prompt: {st.flops:.3g} FLOPs -> ideal TTFT {st.t_compute * 1e3:.1f} ms on H100; "
      f"{llm.prefill(m8, l4, 2048).t_compute * 1e3:.0f} ms on L4 (compute ceiling, before queueing)")

# %% [markdown]
# ## Decode: batch 1 is a weight stream; batching amortizes it
# At batch 1, each token reads ~15 GB again for ~16 GFLOPs: intensity ≈ 1. The time per token is
# bytes ÷ bandwidth, whatever the peak FLOP/s. Batching shares the weight read. But each
# sequence also reads its own KV cache (128 KiB per cached token here), and that read does not amortize.

# %%
cap = llm.max_batch_by_memory(m8, h100, 2048)          # the sweep stops where HBM does
print(f"{'batch':>5s} {'FLOP/B':>7s} {'step ms':>8s} {'agg tok/s':>9s} {'per user':>8s} {'KV share':>9s}")
for b in (1, 8, 32, 64, 128, cap):
    st = llm.decode(m8, h100, b, 2048)
    kv = b * 2049 * m8.kv_bytes_per_token() / st.bytes
    print(f"{b:5d} {st.intensity:7.1f} {st.time * 1e3:8.2f} {st.tokens_per_s:9.0f} {1 / st.time:8.1f} {kv:9.0%}"
          + ("   <- HBM limit at 2K context (10% headroom)" if b == cap else ""))

# %% [markdown]
# ## KV reads cap decode intensity
# As $\text{batch} \to \infty$, the weight read amortizes away. The average intensity of the step then tends to
# $\text{FLOPs per token} \div \text{KV bytes per token}$. For this GQA model, that is about 32 FLOP/B at 4K context.
# If you treat the step as one kernel, it can never reach the 295 of an H100 at 4K. The crossover of the whole step
# exists only when the context is very short, and the HBM capacity for KV runs out first anyway. (Per kernel, the
# picture is sharper: see "Per kernel, not per step" after Exercise 2.2.)

# %%
for c in (0, 128, 256, 1024, 4096, 32768):
    print(f"context {c:6d}: intensity limit {llm.decode_intensity_limit(m8, c):9.1f} FLOP/B | "
          f"whole step compute-bound from batch {llm.decode_crossover_batch(m8, h100, c)}")
print(f"the whole step, as one kernel, can reach the ridge only below ~{llm.max_context_for_compute_bound(m8, h100):.0f} "
      "tokens of context")
print("memory cap at 4K context:", llm.max_batch_by_memory(m8, h100, 4096), "sequences (10% headroom)")

# %% [markdown]
# ## Quantization moves bytes
# FP8 (W8A8 + FP8 KV) halves every byte and doubles the peak. In the memory-bound regime, the gain is
# exactly 2×. Weight-only INT4 (W4A16) decreases the weight bytes by 4×, but it does not change the KV. Thus its
# speedup decreases as batch × context grows. It also lowers the batch at which decode becomes compute-bound
# (the math still runs at the bf16 peak).

# %%
b1, b64 = llm.decode(m8, h100, 1, 2048), llm.decode(m8, h100, 64, 2048)
print(f"{'scheme':10s} {'b=1 ms':>7s} {'x':>5s} {'b=64 ms':>8s} {'x':>5s} {'compute-bound from':>19s}")
for name, kw in llm.SCHEMES.items():
    s1, s64 = llm.decode(m8, h100, 1, 2048, **kw), llm.decode(m8, h100, 64, 2048, **kw)
    print(f"{name:10s} {s1.time * 1e3:7.2f} {b1.time / s1.time:5.2f} {s64.time * 1e3:8.2f} "
          f"{b64.time / s64.time:5.2f} {llm.decode_crossover_batch(m8, h100, 0, **kw):>12} (c=0)")

# %% [markdown]
# ## MoE: which weights a step streams
# The *total* parameters set the memory size, because every expert lives in HBM. The *active*
# parameters set the FLOPs of a token. A decode step streams only the experts that its tokens hit. For Mixtral
# (8 experts, top-2), that is few experts at batch 1, and almost all of them by batch 16. Fine-grained MoE
# (128 experts, top-8) keeps the benefit to larger batches.

# %%
mix = llm.PRESETS["mixtral-8x7b"]
print(f"Mixtral: {mix.params() / 1e9:.1f}B total, {mix.active_params() / 1e9:.1f}B active")
for b in (1, 2, 4, 8, 16, 64):
    st = llm.decode(mix, h200, b, 1024)
    print(f"batch {b:3d}: experts/layer {llm.experts_touched(8, 2, b):4.2f} | step {st.bytes / 1e9:5.1f} GB "
          f"{st.time * 1e3:6.2f} ms on H200 | Qwen3-30B-A3B experts/layer {llm.experts_touched(128, 8, b):5.1f}/128")

# %% [markdown]
# ## Predict, then measure (for the T1 lab)
# Before you run a model on real hardware, write down its ceiling. The example is Qwen2.5-1.5B on a free Colab
# T4, in fp16 (a T4 has no bf16). A measured batch-1 decode must come in *below* this number.

# %%
q = llm.PRESETS["qwen2.5-1.5b"]
t4 = specs.get("t4")
st = llm.decode(q, t4, 1, 512, precision="fp16")
print(f"Qwen2.5-1.5B on T4: {st.bytes / 1e9:.2f} GB per token -> ceiling {st.tokens_per_s:.0f} tok/s at batch 1")

# %% [markdown]
# ## Exercise 2.1 — the decode step time
# Write `decode_step_time(weight_bytes, kv_bytes_per_seq, flops_per_token, batch, device, precision)`.
# It returns the roofline time (seconds) of a step with these properties:
#
# - It streams the weights one time.
# - It reads `kv_bytes_per_seq` for each sequence.
# - It does `flops_per_token` for each sequence.
#
# Treat the whole step as one kernel, as `llm.decode()` does. The next cell calculates the inputs for
# Llama-3.1-8B for you.

# %%
W = llm.streamed_weight_bytes(m8, 1)
def kv_seq(context): return (context + 1) * m8.kv_bytes_per_token()
def f_tok(context): return llm.decode(m8, h100, 1, context).flops

# %% exercise
def decode_step_time(weight_bytes, kv_bytes_per_seq, flops_per_token, batch, device, precision="bf16"):
    ### BEGIN SOLUTION
    t_compute = batch * flops_per_token / device.peak(precision)
    t_memory = (weight_bytes + batch * kv_bytes_per_seq) / device.bandwidth()
    return max(t_compute, t_memory)
    ### END SOLUTION

# %% check
for b, c in [(1, 1024), (16, 2048), (64, 2048), (256, 512)]:
    mine = decode_step_time(W, kv_seq(c), f_tok(c), b, h100)
    ref = llm.decode(m8, h100, b, c).time
    assert abs(mine - ref) / ref < 1e-3, (b, c, mine, ref)
print(f"✅ decode step = max(B·F/peak, (W + B·KV)/BW) — batch 64 at 2K context: "
      f"{decode_step_time(W, kv_seq(2048), f_tok(2048), 64, h100) * 1e3:.2f} ms")

# %% [markdown]
# ## Exercise 2.2 — the whole-step crossover batch, in closed form
# Set $B \cdot F/\text{peak} = (W + B \cdot \mathrm{KV})/\mathrm{BW}$. Then solve for $B$. Write
# `crossover(W, kv_bytes_per_seq, flops_per_token, ridge)`. It returns the smallest integer batch that is
# compute-bound, or `None` when no batch is ever compute-bound. Hint: divide through by $\mathrm{BW}$, and the
# ridge appears. The denominator can go negative.

# %% exercise
import math

def crossover(weight_bytes, kv_bytes_per_seq, flops_per_token, ridge):
    ### BEGIN SOLUTION
    denom = flops_per_token - ridge * kv_bytes_per_seq
    if denom <= 0:
        return None
    return math.ceil(ridge * weight_bytes / denom)
    ### END SOLUTION

# %% check
for c in (0, 128, 256, 1024, 4096):
    assert crossover(W, kv_seq(c), f_tok(c), h100.ridge()) == llm.decode_crossover_batch(m8, h100, c), c
print("✅ B* = ridge·W / (F − ridge·KV): 297 at c=0, 440 at c=128, None from c=1024 — KV reads never amortize")

# %% [markdown]
# ## Per kernel, not per step
# `decode()` and your `decode_step_time` calculate the step as one kernel: $\max(\sum F/\text{peak}, \sum B/\mathrm{BW})$.
# That calculation mixes two kernels with opposite shapes. The **weight GEMMs** multiply a $[\text{batch} \times d]$
# activation by every weight matrix. For them, $\text{intensity} \approx \text{batch} \times 2 / \text{weight bytes}$ at
# any context. They are compute-bound from $\text{batch} \approx \text{ridge}$, thus the capacity-planning rule is correct for them.
#
# **Attention** reads, for each sequence, the KV cache of that sequence: about $2 \times (\text{heads}/\mathrm{kv\_heads})/\mathrm{kv\_bytes} = 4$ FLOP/B
# here, whatever the batch. Thus attention is never compute-bound.
#
# An engine runs the kernels one after another. Thus the tighter bound is the *sum* of the per-kernel times
# (`llm.decode_split`). The two bounds agree while both kernels are memory-bound. They become different when the
# GEMMs cross the ridge. After that point, the throughput does not increase more, and each extra sequence only adds KV time.

# %%
fp8 = llm.SCHEMES["fp8"]
print(f"GEMM crossover (any context): batch {llm.gemm_crossover_batch(m8, h100)} in bf16, "
      f"{llm.gemm_crossover_batch(m8, h100, **fp8)} in FP8 (half the bytes, twice the peak)")
print(f"attention at 2K context: {llm.decode_split(m8, h100, 64, 2048).attention.intensity:.1f} FLOP/B (bf16 KV), "
      f"{llm.decode_split(m8, h100, 64, 2048, **fp8).attention.intensity:.1f} (FP8 KV)\n")
print(f"{'scheme':6s} {'batch':>5s} {'one kernel':>11s} {'GEMMs':>13s} {'attention':>14s} {'sum':>9s} {'tok/s':>7s}")
for name, kw, b in [("bf16", {}, 64), ("bf16", {}, 208), ("fp8", fp8, 193), ("fp8", fp8, 400), ("fp8", fp8, 476)]:
    one, sp = llm.decode(m8, h100, b, 2048, **kw), llm.decode_split(m8, h100, b, 2048, **kw)
    print(f"{name:6s} {b:5d} {one.time * 1e3:8.2f} ms {sp.gemms.time * 1e3:6.2f} ms ({sp.gemms.bound[:3]}) "
          f"{sp.attention.time * 1e3:6.2f} ms ({sp.attention.bound[:3]}) {sp.time * 1e3:6.2f} ms {b / sp.time:7.0f}")

# %% [markdown]
# ## Exercise 2.3 — predict a quantization speedup from bytes alone
# In the memory-bound regime, $\text{speedup} = \text{old bytes} \div \text{new bytes}$. Write
# `w4a16_speedup(batch, context)` for Llama-3.1-8B on an H100. The weights go from 2 bytes to 0.5, and the KV cache stays
# bf16. Use `W` and `kv_seq()` from the earlier cells ($W$ scales with bytes per weight). Then explain the gap between batch 1 and 64.

# %% exercise
def w4a16_speedup(batch, context):
    ### BEGIN SOLUTION
    old = W + batch * kv_seq(context)
    new = W / 4 + batch * kv_seq(context)
    return old / new
    ### END SOLUTION

# %% check
for b in (1, 64):
    ref = llm.decode(m8, h100, b, 2048).time / llm.decode(m8, h100, b, 2048, **llm.SCHEMES["w4a16"]).time
    assert abs(w4a16_speedup(b, 2048) - ref) < 0.01, (b, w4a16_speedup(b, 2048), ref)
print(f"✅ W4A16: {w4a16_speedup(1, 2048):.2f}x at batch 1 but {w4a16_speedup(64, 2048):.2f}x at batch 64 — "
      "the KV half of the bytes did not shrink")

# %% [markdown]
# ## Exercise 2.4 — pick a batch size and say what binds it
# An interactive product needs ITL ≤ `itl_s` at 8K context. Write `pick_batch(model, device, context, itl_s, **scheme)`.
# It returns `(batch, binding)`:
#
# - `batch` is the largest batch that meets the ITL **and** fits in memory
#   (use `llm.decode` and `llm.max_batch_by_memory`).
# - `binding` is `"latency"` or `"memory"`, the limit that stopped you.

# %% exercise
def pick_batch(model, device, context, itl_s, **scheme):
    ### BEGIN SOLUTION
    mem_kw = {k: v for k, v in scheme.items() if k in ("weight_bytes", "kv_bytes")}
    cap = llm.max_batch_by_memory(model, device, context, **mem_kw)
    b = 0
    while b < cap and llm.decode(model, device, b + 1, context, **scheme).time <= itl_s:
        b += 1
    return b, ("memory" if b == cap else "latency")
    ### END SOLUTION

# %% check
assert pick_batch(m8, h100, 8192, 0.015) == (32, "latency")
assert pick_batch(m8, h100, 8192, 0.025) == (52, "memory")
assert pick_batch(m8, h100, 8192, 0.015, **llm.SCHEMES["fp8"]) == (79, "latency")
assert pick_batch(m8, h100, 8192, 0.025, **llm.SCHEMES["fp8"]) == (119, "memory")
print("✅ at 8K context: bf16 → 32 (ITL-bound at 15 ms) or 52 (HBM-bound); FP8 more than doubles both")

# %% [markdown]
# ## Exercise 2.5 — when does an MoE stream all its experts?
# Write `experts_touched(E, k, T)` and `batch_for_fraction(E, k, frac)`. The first function returns the expected
# number of distinct experts that one layer reads for $T$ tokens, with uniform routing. The second function returns
# the smallest batch at which a decode step touches at least `frac` of the experts.

# %% exercise
def experts_touched(E, k, T):
    ### BEGIN SOLUTION
    return E * (1 - (1 - k / E) ** T)
    ### END SOLUTION


def batch_for_fraction(E, k, frac):
    ### BEGIN SOLUTION
    t = 1
    while experts_touched(E, k, t) < frac * E:
        t += 1
    return t
    ### END SOLUTION

# %% check
assert abs(experts_touched(8, 2, 1) - 2) < 1e-12 and abs(experts_touched(8, 2, 16) - llm.experts_touched(8, 2, 16)) < 1e-12
assert batch_for_fraction(8, 2, 0.9) == 9          # Mixtral
assert batch_for_fraction(128, 8, 0.9) == 36       # Qwen3-30B-A3B
print("✅ 90% of experts are streamed by batch 9 (8 experts, top-2) or 36 (128, top-8): "
      "MoE decode is 'active-params cheap' only at small batch")

# %% [markdown]
# ## The memory hierarchy: tiles and fusion (primer §4)
# Every level of the hierarchy has its own roofline. A GEMM calculates each $T_m \times T_n$ output tile from two
# panels, a $T_m \times k$ panel of $A$ and a $k \times T_n$ panel of $B$. It streams the panels through shared memory.
# Each $k$-step loads $(T_m + T_n)$ elements for $2 \cdot T_m \cdot T_n$ FLOPs, thus larger tiles reuse more. But
# the pipelined operand buffers must fit in the shared memory of an SM (228 KiB on an H100).
#
# Elementwise chains are the other extreme. They have nothing to reuse. Thus the only lever is to stop the
# round trips of intermediates through HBM.

# %%
from roofline import roofline as rl
for tm, tn in ((64, 64), (128, 128), (128, 256), (256, 256)):
    print(f"tile {tm:3d}x{tn:<3d}: {rl.tile_intensity(tm, tn):5.1f} FLOP/B per byte staged; "
          f"{rl.tile_smem_bytes(tm, tn, 64, 2, stages=4) / 1024:4.0f} KiB of SMEM at 4 stages of k = 64")
n = 4096
print(f"\n4096³ bf16 GEMM with 128x128 tiles and no reuse between tiles: {rl.tiled_gemm_bytes(n, n, n, 128, 128) / 1e9:.2f} GB "
      f"from HBM vs {rl.gemm(n, n, n).bytes / 1e6:.0f} MB compulsory — the L2 must supply the rest")

# %% [markdown]
# ## Exercise 2.6 — pick a tile, then fuse a chain
# **(a)** Write `pick_tile(candidates, smem_bytes, tile_k=64, stages=4, b=2)`. Look at the `(Tm, Tn)` candidates
# whose pipelined $A$ and $B$ buffers fit in `smem_bytes`. Of these, return the one with the highest tile intensity.
# Make a prediction first: does the winner reach the HBM ridge of the H100 (295) by itself?
#
# **(b)** Write `chain_bytes(n, inputs_per_op, b, fused)`. It returns the HBM bytes for a chain of elementwise ops on `n`
# elements. `inputs_per_op[i]` is the number of full-size tensors that op $i$ reads. They are the tensor that moves through the
# chain and, for a residual add, a second tensor. Without fusion, every op reads its inputs from HBM and writes its output
# back. With fusion, one kernel reads every distinct input one time and writes the result one time.

# %% exercise
def pick_tile(candidates, smem_bytes, tile_k=64, stages=4, b=2):
    ### BEGIN SOLUTION
    fits = [(tm, tn) for tm, tn in candidates if stages * (tm + tn) * tile_k * b <= smem_bytes]
    return max(fits, key=lambda t: 2 * t[0] * t[1] / ((t[0] + t[1]) * b))
    ### END SOLUTION


def chain_bytes(n, inputs_per_op, b=2, fused=True):
    ### BEGIN SOLUTION
    if fused:
        return (1 + sum(k - 1 for k in inputs_per_op) + 1) * n * b
    return sum(k + 1 for k in inputs_per_op) * n * b
    ### END SOLUTION

# %% check
cands = [(64, 64), (128, 128), (128, 256), (256, 128), (256, 256)]
best = pick_tile(cands, 228 * 1024)
assert best in ((128, 256), (256, 128)), best
assert rl.tile_intensity(*best) < h100.ridge()               # 85 < 295: reuse between tiles (L2) must do the rest
assert pick_tile(cands, 228 * 1024, stages=6) == (128, 128)   # a deeper pipeline forces a smaller tile
assert pick_tile(cands, 256 * 1024) == (256, 256)
ne = 4096 * 8192
chain = [1, 1, 1, 2]                                         # bias, GELU, dropout, residual add
for fused in (False, True):
    assert chain_bytes(ne, chain, fused=fused) == rl.fusion_bytes(ne, 4, 2, fused=fused, extra_inputs=1)
    assert chain_bytes(ne, [2, 2], fused=fused) == rl.fusion_bytes(ne, 2, 2, fused=fused, extra_inputs=2)
uf, f = chain_bytes(ne, chain, fused=False), chain_bytes(ne, chain, fused=True)
print(f"✅ best tile {best}: {rl.tile_intensity(*best):.0f} FLOP/B, below the 295 ridge — tiling in SMEM is not enough "
      f"on its own; fusing the chain: {uf / 1e6:.0f} → {f / 1e6:.0f} MB, "
      f"{uf / h100.bandwidth() * 1e6:.0f} → {f / h100.bandwidth() * 1e6:.0f} µs on an H100 ({1 - f / uf:.0%} of the bytes gone)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A decode step streams all the weights one time and the KV cache of each sequence.
# A prefill step does the same reads, but for thousands of tokens. Thus prefill is compute-bound, and TTFT is
# FLOPs over peak. Decode is memory-bound, and the time per token is bytes over bandwidth."
#
# "Batching amortizes the weight read. This is why throughput increases with batch. The weight GEMMs reach the
# ridge near $\text{batch} \approx \text{ridge}$. But attention reads the KV cache of each sequence at ~4 FLOP/B, whatever
# the batch. Thus at realistic contexts, KV bytes dominate the step, and the HBM capacity for KV caps the batch first."
#
# "Quantization is a bytes lever. In this memory-bound regime, FP8 weights and KV halve the step time. Weight-only INT4 gives
# a small gain when KV dominates."
#
# **Drill.**
# 1. *Do the 2.3× bf16 FLOPs of a B200 speed up our batch-32, 4K-context decode by 2.3×?* No. The decode
#    is memory-bound. The gain is the bandwidth ratio (8 against 3.35 TB/s ≈ 2.4×), plus the capacity for a larger batch.
# 2. *How do TTFT and ITL grow: TTFT with prompt length, ITL with batch?* TTFT is compute-bound: it is
#    linear in tokens while the matmuls dominate. Then it is superlinear as the quadratic attention term grows
#    (an 8K prompt takes 4.4× the 2K time, not 4×). ITL is nearly flat while the shared weight stream
#    dominates (small batch × short context). Then it grows approximately linearly with batch × context as the KV
#    reads take over. At 2K context, it is 4.56 ms at batch 1, 9.61 ms at 64 and 21.16 ms at 208.
# 3. *We quantized to W4A16 and saw 1.5×, not 4×. Bug?* No. At batch 64 × 2K context, KV is half
#    the bytes, and the KV stayed bf16. Quantize the KV cache (FP8) to move the rest.
