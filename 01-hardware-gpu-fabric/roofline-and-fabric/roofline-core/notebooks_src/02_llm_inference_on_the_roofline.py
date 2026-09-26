# %% [markdown]
# # 02 · LLM inference on the roofline
#
# **Tier:** T0 — pure arithmetic on a model config and a device; no GPU. To see real numbers,
# run a small model on a T4/L4 with `04-inference-engine/serving-engine/vllm-serving-lab` (T1)
# and compare with the ceilings computed here.
#
# ## The one-minute version
# One engine step streams every weight from HBM once, however many tokens it carries, while
# FLOPs grow with the tokens. So a step's arithmetic intensity is roughly **its token count**
# (at 2-byte weights). A 2K-token prefill sits far right of the ridge: compute-bound, and TTFT
# is FLOPs ÷ peak. A batch-1 decode step sits at ~1 FLOP/B: memory-bound, and time per token is
# bytes ÷ bandwidth. Batching moves decode right — until each sequence's **KV-cache reads**,
# which grow with context exactly as fast as its FLOPs, cap the intensity far below the ridge.
# After this notebook you can say which bound applies, predict step time within the roofline,
# and explain what quantization and MoE really change: **bytes**. Primer: `../PRIMER.md` §3.

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
# Weights are read once for the whole prompt; FLOPs are `2 × tokens × params` plus causal
# attention. Watch the bound flip near the ridge (295 FLOP/B on an H100 in bf16) — measured in
# *tokens*. That is why engines give prefill chunks of a few hundred to a few thousand tokens.

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
# At batch 1 every token re-reads ~15 GB for ~16 GFLOPs: intensity ≈ 1. The time per token is
# bytes ÷ bandwidth, whatever the peak FLOP/s. Batching shares the weight read — but each
# sequence also reads its own KV cache (128 KiB per cached token here), which does not amortize.

# %%
print(f"{'batch':>5s} {'FLOP/B':>7s} {'step ms':>8s} {'agg tok/s':>9s} {'per user':>8s} {'KV share':>9s}")
for b in (1, 8, 32, 64, 128, 256):
    st = llm.decode(m8, h100, b, 2048)
    kv = b * 2049 * m8.kv_bytes_per_token() / st.bytes
    print(f"{b:5d} {st.intensity:7.1f} {st.time * 1e3:8.2f} {st.tokens_per_s:9.0f} {1 / st.time:8.1f} {kv:9.0%}")

# %% [markdown]
# ## KV reads cap decode intensity
# As batch → ∞ the weight read amortizes away and intensity tends to
# `FLOPs per token ÷ KV bytes per token` — for this GQA model about 32 FLOP/B at 4K context.
# Batching can never reach an H100's 295. The crossover batch exists only at very short context,
# and the HBM capacity for KV runs out first anyway.

# %%
for c in (0, 128, 256, 1024, 4096, 32768):
    print(f"context {c:6d}: intensity limit {llm.decode_intensity_limit(m8, c):9.1f} FLOP/B | "
          f"compute-bound from batch {llm.decode_crossover_batch(m8, h100, c)}")
print(f"decode can reach the ridge only below ~{llm.max_context_for_compute_bound(m8, h100):.0f} tokens of context")
print("memory cap at 4K context:", llm.max_batch_by_memory(m8, h100, 4096), "sequences (10% headroom)")

# %% [markdown]
# ## Quantization moves bytes
# FP8 (W8A8 + FP8 KV) halves every byte and doubles the peak: exactly 2× in the memory-bound
# regime. Weight-only INT4 (W4A16) cuts weight bytes 4× but leaves KV alone, so its speedup
# shrinks as batch × context grows — and it lowers the batch at which decode turns compute-bound
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
# Memory is sized by *total* parameters (every expert lives in HBM); a token's FLOPs by *active*
# ones. A decode step streams only the experts its tokens hit — few at batch 1, almost all by
# batch 16 for Mixtral (8 experts, top-2). Fine-grained MoE (128 experts, top-8) keeps the
# benefit to larger batches.

# %%
mix = llm.PRESETS["mixtral-8x7b"]
print(f"Mixtral: {mix.params() / 1e9:.1f}B total, {mix.active_params() / 1e9:.1f}B active")
for b in (1, 2, 4, 8, 16, 64):
    st = llm.decode(mix, h200, b, 1024)
    print(f"batch {b:3d}: experts/layer {llm.experts_touched(8, 2, b):4.2f} | step {st.bytes / 1e9:5.1f} GB "
          f"{st.time * 1e3:6.2f} ms on H200 | Qwen3-30B-A3B experts/layer {llm.experts_touched(128, 8, b):5.1f}/128")

# %% [markdown]
# ## Predict, then measure (for the T1 lab)
# Before you run a model on real hardware, write down its ceiling. Qwen2.5-1.5B on a free Colab
# T4 (fp16 — a T4 has no bf16): a measured batch-1 decode must come in *below* this number.

# %%
q = llm.PRESETS["qwen2.5-1.5b"]
t4 = specs.get("t4")
st = llm.decode(q, t4, 1, 512, precision="fp16")
print(f"Qwen2.5-1.5B on T4: {st.bytes / 1e9:.2f} GB per token -> ceiling {st.tokens_per_s:.0f} tok/s at batch 1")

# %% [markdown]
# ## Exercise 2.1 — the decode step time
# Write `decode_step_time(weight_bytes, kv_bytes_per_seq, flops_per_token, batch, device, precision)`:
# the roofline time (seconds) of a step that streams the weights once, reads `kv_bytes_per_seq` for
# each sequence, and does `flops_per_token` for each. The inputs for Llama-3.1-8B are computed for you.

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
# ## Exercise 2.2 — the crossover batch, in closed form
# Set `B·F/peak = (W + B·KV)/BW` and solve for B. Write `crossover(W, kv_bytes_per_seq, flops_per_token, ridge)`
# returning the smallest integer batch that is compute-bound, or `None` when no batch ever is.
# (Hint: divide through by BW; the ridge appears; the denominator can go negative.)

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
# ## Exercise 2.3 — predict a quantization speedup from bytes alone
# In the memory-bound regime, speedup = old bytes ÷ new bytes. Write `w4a16_speedup(batch, context)`
# for Llama-3.1-8B on an H100: weights go from 2 bytes to 0.5, the KV cache stays bf16. Use `W`
# and `kv_seq()` above (W scales with bytes per weight). Then explain the gap between batch 1 and 64.

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
# An interactive product needs ITL ≤ `itl_s` at 8K context. Write `pick_batch(model, device, context, itl_s, **scheme)`
# returning `(batch, binding)` where `batch` is the largest batch that meets the ITL **and** fits in memory
# (use `llm.decode` and `llm.max_batch_by_memory`), and `binding` is `"latency"` or `"memory"` —
# whichever stopped you.

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
# Write `experts_touched(E, k, T)` (expected distinct experts one layer reads for T tokens,
# uniform routing) and `batch_for_fraction(E, k, frac)`: the smallest batch at which a decode
# step touches at least `frac` of the experts.

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
# ## In a design review
# **The two-minute version.** "A decode step streams all the weights once and each sequence's
# KV cache; a prefill step does the same reads but for thousands of tokens. So prefill is
# compute-bound — TTFT is FLOPs over peak — and decode is memory-bound — time per token is bytes
# over bandwidth. Batching amortizes the weight read, which is why throughput climbs with batch,
# but KV reads grow with context as fast as FLOPs do, so at realistic contexts decode never
# reaches the ridge; HBM capacity for KV caps the batch first. Quantization is a bytes lever:
# FP8 weights and KV halve step time in that regime; weight-only INT4 helps little once KV dominates."
#
# **Drill.**
# 1. *Would a B200's 2.3× bf16 FLOPs speed up our batch-32, 4K-context decode by 2.3×?* — No: it
#    is memory-bound; the gain is the bandwidth ratio (8 vs 3.35 TB/s ≈ 2.4×), plus capacity for a bigger batch.
# 2. *Why does TTFT grow linearly with prompt length but decode ITL barely moves with batch?* —
#    prefill is compute-bound (FLOPs ∝ tokens); decode is dominated by the weight stream, shared across the batch.
# 3. *We quantized to W4A16 and saw 1.5×, not 4×. Bug?* — no: at batch 64 × 2K context, KV is half
#    the bytes and stayed bf16; quantize the KV cache (FP8) to move the rest.
