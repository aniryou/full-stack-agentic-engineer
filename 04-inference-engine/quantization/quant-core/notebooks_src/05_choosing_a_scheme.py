# %% [markdown]
# # 05 · Choosing a scheme
#
# **Tier:** T0. It does arithmetic on a model config and a GPU spec, and runs in a few seconds. Every latency that
# this notebook prints is **SIMULATED**. Each latency comes from a roofline model,
# $\max(\text{bytes}/\text{bandwidth}, \text{FLOPs}/\text{peak})$, with assumed efficiencies. The GPU figures are
# dense datasheet values as of September 2026 `(verify)`. The lab's notebook 02 (T1) measures the same schemes on a
# real GPU.
#
# ## The one-minute version
# Select a scheme in three steps.
#
# - **What bounds the workload?** Decode reads every weight from memory at each step. Thus weight bytes bound decode
#   (and, at long context and high batch, KV bytes). FLOPs bound prefill. KV memory bounds concurrency.
# - **What can the GPU execute natively?** Weight-only formats (W4A16, W8A16) run on all GPUs from Turing up. They
#   cut bytes, not FLOPs. FP8 W8A8 needs Ada or newer, and FP4 W4A4 needs Blackwell. Blackwell has no INT8 W8A8.
#   An FP8 checkpoint on an A100 runs as weight-only FP8. `cost.supported` encodes the rules of vLLM.
# - **What accuracy can you afford?** Put the schemes in order, from least to most aggressive. Then take the first
#   scheme that meets the latency and concurrency targets *and* passes the eval (notebook 03, primer §8).
#
# After this notebook, you can make that table for any GPU and model in the repo, and defend it. You can also put a
# cost per million tokens on it.
#
# Primer: `../PRIMER.md` §1 *Why quantize* and §10 *Choosing a scheme*. The roofline comes from layer 01
# (`01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md` §2–3). The cost formula comes from its §8.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

from quantcore import cost as C, eval as E, quantize_model, TinyModel

L4, H100, T4 = C.GPUS["L4"], C.GPUS["H100-SXM"], C.GPUS["T4"]
m8 = C.MODELS["llama-3.1-8b"]


def show(rows, session=2000):
    print(f"{'scheme':11} {'KV':>3} {'runs as':11} {'weights':>8} {'dec b=1':>8} {'dec b=32':>9} {'prefill':>8} {'sess':>5}  note")
    for r in rows:
        if "prefill_ms" in r:
            print(f"{r['scheme']:11} {r['kv_bits']:3d} {r['runs_as']:11} {r['weights_gb']:6.2f}GB {r['decode_1_ms']:6.1f}ms "
                  f"{r['decode_b_ms']:7.1f}ms {r['prefill_ms']:6.0f}ms {r['sessions']:5d}  {r['note']}")
        else:
            print(f"{r['scheme']:11} {r['kv_bits']:3d} {'-':11} {r['weights_gb']:6.2f}GB {'':38}  {r['note']}")
    print("(SIMULATED: 80% of bandwidth, 60% of peak FLOP/s, 2 ms per step; decode at 1,000 tokens of context; "
          f"prefill of 1,800 tokens; sessions of {session:,} tokens)")

# %% [markdown]
# ## Worked example 1 — the serving-engine §8 table, reproduced
# serving-engine PRIMER §8 uses `minengine.perf` to print what each scheme gives for Llama-3.1-8B on a 24 GB L4.
# `quantcore.cost.step_cost` is the same model. If you define the L4 as minengine defines it (FP8 peak exactly 2×
# bf16), the numbers agree to the digit (`tests/test_repo_numbers.py`). The table in the next cell uses the
# 242.5 TFLOP/s FP8 peak of `roofline.specs`, not 242. At this precision, it prints the same numbers.

# %%
show(C.table(L4, m8, schemes=["bf16", "w8a16-fp8", "w4a16", "w8a8-fp8"]))

# %% [markdown]
# Read each row of the table by its bound. Decode at batch 1 is the weight read. Weights of 16.1, 9.1 and 5.7 GB give 65, 36 and
# 22 ms. The decrease is not 3.9×, because the 16-bit LM head, the KV read and the step overhead do not decrease.
# Prefill is compute: only FP8 W8A8 halves it. The number of sessions depends on the bytes that stay for KV: FP8 KV
# doubles it at any weight format.
#
# ## Worked example 2 — one GEMM on the roofline: where W4A16 stops paying
# vllm-internals §8.1 calculates the time of Llama-3.1-8B's `down_proj` ($K = 14{,}336$, $N = 4{,}096$) on an L4 for
# $M$ tokens in the step. `cost.gemm_time` gives the same result (W4A16 at 4.16 bits per weight, as that table
# counts).

# %%
for M in (1, 16, 64, 128, 256, 400, 462, 2048):
    row = [C.gemm_time(M, 14336, 4096, L4, s, w_bits=4.16 if s == "w4a16" else None) for s in ("bf16", "w4a16", "w8a8-fp8")]
    print(f"M = {M:5d}: " + "  ".join(f"{s} {r['t'] * 1e6:7.0f} us ({r['bound'][:3]})" for s, r in zip(("BF16", "W4A16", "FP8"), row))
          + f"   W4A16 speedup {row[0]['t'] / row[1]['t']:.2f}x")
for g in (L4, H100):
    print(f"{g.name}: W4A16 turns compute-bound above {C.crossover_tokens(14336, 4096, g, w_bits=4.16):.0f} tokens per step, "
          f"BF16 above {C.crossover_tokens(14336, 4096, g, 'bf16'):.0f}")

# %% [markdown]
# A weight-only kernel does BF16 math on dequantized weights. Up to ~120 tokens per step on an L4 (~85 on an H100),
# it is byte-bound and keeps the full byte ratio, ~3.8×. At that point, it hits the BF16 compute ceiling, but BF16
# itself is still byte-bound. Thus, from there, its speedup decreases (1.7× at 256 tokens). The speedup is gone where BF16
# also becomes compute-bound: ~460 tokens on an L4 (~330 on an H100).
#
# Thus decode batches of a few hundred still gain from INT4. Long prefill chunks (512 and up) do not gain. Real
# kernels lose the gain sooner, because dequantization has a cost. NVIDIA's ModelOpt measured weight-only NVFP4
# slower than BF16 in 10 of 12 shapes on Blackwell (its QAD note of 2026-09-16, verify). FP8 W8A8 halves both
# ceilings and helps at every M.
#
# ## Worked example 3 — what a checkpoint runs as, per GPU generation

# %%
print(f"{'':13}" + "".join(f"{s:>13}" for s in C.SCHEMES) + f"{'FP8 KV':>9}")
for g in C.GPUS.values():
    cells = [C.supported(g, s)["runs_as"] or "no" for s in C.SCHEMES]
    print(f"{g.name:13}" + "".join(f"{c:>13}" for c in cells) + f"{'yes' if C.supported(g, 'bf16', 8)['kv'] else 'no':>9}")

# %% [markdown]
# Read down the FP8 W8A8 column. An FP8 checkpoint loads on all GPUs from Turing up. But below Ada, it is a
# weight-only model (Marlin FP8) with BF16 math. This gives a gain in memory only. NVFP4 is W4A4 only on Blackwell
# (SM100/SM120, CUDA ≥ 12.8). On other GPUs, it runs as weight-only 4-bit.
#
# INT8 W8A8 is the Turing/Ampere way to make prefill faster. From compute capability 10.0, vLLM does not support INT8
# W8A8. A T4 has no FP8 KV cache in any vLLM backend. All the rules in worked example 3 are for vLLM 0.30.0 and main,
# from the capability checks of the kernels. Make sure that they apply to your version: the log line
# `Selected <kernel> for <module>` is the truth.
#
# ## Worked example 4 — three deployments

# %%
print("== a free Colab/Kaggle T4, Qwen2.5-1.5B (16 GB, fp16 only, no FP8 KV)")
show(C.table(T4, C.MODELS["qwen2.5-1.5b"], kv=(16,), schemes=["bf16", "w8a8-int8", "w4a16"]))
print("\n== an H100 80GB, Llama-3.1-70B")
show(C.table(H100, C.MODELS["llama-3.1-70b"], schemes=["bf16", "w8a8-fp8", "w4a16"], session=4000), session=4000)
print("\n== a B200, Llama-3.1-8B (verify: Blackwell FP4 figures)")
show(C.table(C.GPUS["B200"], m8, kv=(8,), schemes=["bf16", "w8a8-fp8", "w8a8-int8", "w4a4-nvfp4"]))

# %% [markdown]
# On a T4, a 1.5B model is so small that quantization is about speed: INT4 for decode, INT8 W8A8 for prefill.
#
# On an H100, a 70B model in 16-bit does not fit at all. In FP8 it fits. But the cost model of this notebook uses
# round memory inputs (0.9 × 80 GB − 1 GB). With these inputs, there is no room left for a 4K session.
#
# Take the defaults of vLLM on the 79.65 GiB that an H100 reports (the lab's `quantlab.kv.size`). With them, there is
# room for 2 such sessions, or 4 with FP8 KV. In both cases, there is no useful concurrency.
#
# On one GPU, only a 4-bit format serves the 70B model. With FP8, you need two GPUs with tensor parallelism, or the
# 141 GB of an H200. On a B200, NVFP4 W4A4 is the 4-bit format that also cuts prefill FLOPs. This is true if the model survives
# 4-bit activations (notebook 04, worked example 4).
#
# ## Worked example 5 — what it costs per million tokens
# The formula is `$/M tokens = $/GPU-hour ÷ (tokens/s × 3600 × utilisation) × 10⁶` (layer 01 §8.1). The prices are
# the GCP list prices of the research snapshot, us-central1, September 2026: L4 ~\$0.70/hr, H100 ~\$11/GPU-hr on
# demand `(verify)`. `COMPUTE.md` keeps them current.

# %%
for g, price in ((L4, 0.70), (H100, 11.0)):
    for s, kv in (("bf16", 16), ("w8a8-fp8", 8), ("w4a16", 8)):
        batch = min(256, C.sessions(g, m8, s, 2000, kv))
        tps = batch / C.step_cost(g, m8, s, [(1000, 1)] * batch, kv)["t"]
        print(f"{g.name:9} {s:9} KV{kv:<2} batch {batch:3d}: {tps:7,.0f} tok/s  ${C.cost_per_million(price, tps):.3f}/M at 100% "
              f"(${C.cost_per_million(price, tps, 0.6):.3f} at 60%)  SIMULATED")

# %% [markdown]
# Quantization decreases cost in two ways. It gives fewer bytes per step. It also gives more sessions, thus a larger
# batch shares each step. On the L4, KV memory sets the maximum batch (17 sessions in bf16). That is why FP8 weights
# and FP8 KV together cut the cost per token ~6× there, not 2×. The reason is that the batch is 5× larger, in a step that is not longer.
#
# On the H100, BF16 fits 209 sessions, and FP8 gets to the 256 cap that this notebook uses. There, FP8 cuts the cost
# to approximately half.
#
# ## Worked example 6 — the accuracy gate
# The eval decides which rows it permits at all. Here, the model is the tiny model of notebook 03. The budget has two
# limits: $\mathrm{KL} \le 0.05$ nats, and an accuracy drop within two standard errors of the difference
# (`E.diff_stderr`, unpaired). The paired McNemar z on the flips, `E.paired_z`, is the sharper test (primer §8). The
# next cell applies the budget:

# %%
tm = TinyModel()
Xt, yt = tm.sample(4000, "test")
Xc, _ = tm.sample(256, "calib")
ref = tm.forward(Xt)
for label, q in (("INT8 RTN", quantize_model(tm, "rtn", 8, None)), ("INT4 RTN g32", quantize_model(tm, "rtn", 4, 32)),
                 ("INT4 GPTQ g32", quantize_model(tm, "gptq", 4, 32, calib=Xc))):
    r = E.compare(ref, q.forward(Xt), yt)
    print(f"{label:14} KL {r['kl']:.4f}  acc {r['acc']:.1%} (fp {r['acc_ref']:.1%}; drop {r['acc_ref'] - r['acc']:+.1%} vs "
          f"2 x {r['diff_stderr']:.1%})  paired z {r['paired_z']:+.1f}  within budget: {E.within_budget(r, max_kl=0.05)}")

# %% [markdown]
# Here, the budget permits INT4 only with GPTQ: the recipe is part of the scheme. The 0.6-point drop of GPTQ is
# inside the unpaired noise bar. But the paired flips (85 lost, 61 gained, ${z = -2.0}$) show a small but probably
# real loss. Thus make sure that a budget says which test it means. `choose(rows, allowed=...)` takes the set of
# schemes that the budget permits.
#
# ## Exercise 5.1 — the W4A16 crossover by hand
# Take a linear with $K$ inputs, $N$ outputs and $M$ tokens. The FLOPs are ${2MKN}$. The bytes are
# $K \cdot N \cdot w + M \cdot (K \cdot a + 2N)$, with `w` bytes per weight (4.16 bits, which is 0.52 B) and `a = 2`
# bytes per activation. The time is the larger of $\text{FLOPs}/\text{peak}$ and $\text{bytes}/\text{bandwidth}$.
#
# Find the $M$ where the two are equal. Do this for the L4 (121 TFLOP/s bf16, 0.30 TB/s) and the H100 (989.4,
# 3.35). Set `m_l4` and `m_h100`.

# %% exercise
K_, N_ = 14336, 4096
### BEGIN SOLUTION
def crossover(peak_tflops, bw_tbs, w=4.16 / 8, a=2):
    P, B = peak_tflops * 1e12, bw_tbs * 1e12
    return K_ * N_ * w / B / (2 * K_ * N_ / P - (K_ * a + 2 * N_) / B)
m_l4, m_h100 = crossover(121, 0.30), crossover(989.4, 3.35)
### END SOLUTION

# %% check
assert round(m_l4) == 120 and round(m_h100) == 85
print(f"✅ {m_l4:.0f} tokens on an L4, {m_h100:.0f} on an H100: past that W4A16's edge shrinks, and where BF16 turns "
      "compute-bound too (~460 and ~330) it is a memory format, not a speed format")

# %% [markdown]
# ## Exercise 5.2 — what does an FP8 W8A8 checkpoint run as?
# Do not call `cost.supported`. Fill `runs_as` for an FP8 W8A8 checkpoint (`"w8a8-fp8"`) on each GPU. Give the
# scheme that the checkpoint actually executes on that GPU: `"w8a8-fp8"` or `"w8a16-fp8"` (weight-only). Use the
# compute capabilities (T4 7.5, A100 8.0, L4 8.9, H100 9.0, B200 10.0).

# %% exercise
runs_as = {"T4": None, "A100-80GB": None, "L4": None, "H100-SXM": None, "B200": None}
### BEGIN SOLUTION
runs_as = {g: ("w8a8-fp8" if C.GPUS[g].cc >= 8.9 else "w8a16-fp8") for g in runs_as}
### END SOLUTION

# %% check
assert runs_as == {g: C.supported(C.GPUS[g], "w8a8-fp8")["runs_as"] for g in runs_as}
print("✅ FP8 tensor cores start at Ada (sm_89): on a T4 or A100 the same checkpoint saves memory but not FLOPs")

# %% [markdown]
# ## Exercise 5.3 — pick a scheme for a free T4
# Serve Llama-3.1-8B on a T4 (16-bit KV only). You need at least **20** concurrent 2,000-token sessions. You also
# need a 1,800-token prefill in at most **700 ms**. Use `C.table(T4, m8, kv=(16,))`. Select the least aggressive
# scheme (order `C.ACCURACY_ORDER`) that meets both targets. Set `choice` to its name.

# %% exercise
### BEGIN SOLUTION
rows = C.table(T4, m8, kv=(16,))
choice = C.choose(rows, min_sessions=20, max_prefill_ms=700)["scheme"]
### END SOLUTION

# %% check
assert choice == "w4a16"
print("✅ w4a16: 16-bit weights do not fit a T4, 8-bit weights leave room for only 16 sessions; INT4 leaves 29. "
      "On a T4 the INT4 checkpoint is the only way to serve an 8B model at all, and GPTQ/AWQ is how it stays accurate")

# %% [markdown]
# ## Exercise 5.4 — cost per million tokens
# The L4 serves Llama-3.1-8B with FP8 weights and FP8 KV. It runs batch 32 in 44.2 ms per decode step (SIMULATED).
# At $0.70 per hour and 60% utilisation, what does a million output tokens cost? Set `usd_per_m`.

# %% exercise
### BEGIN SOLUTION
tps = 32 / 0.0442
usd_per_m = 0.70 / (tps * 3600 * 0.6) * 1e6
### END SOLUTION

# %% check
assert abs(usd_per_m - C.cost_per_million(0.70, 32 / 0.0442, 0.6)) < 1e-9 and 0.44 < usd_per_m < 0.46
print(f"✅ ${usd_per_m:.3f} per million output tokens (SIMULATED throughput, list price (verify))")

# %% [markdown]
# ## Exercise 5.5 — a 70B model on one 80 GB GPU
# The model is Llama-3.1-70B on one H100. Fill `fits` for `"bf16"`, `"w8a8-fp8"`, `"w4a16"` and `"w4a4-nvfp4"`.
# `fits` maps each scheme to the number of 4,000-token sessions with FP8 KV (0 means that the weights leave no
# room). Then set `smallest_ok` to the least aggressive scheme that serves at least 32 such sessions.

# %% exercise
fits, smallest_ok = {}, None
### BEGIN SOLUTION
fits = {s: C.sessions(H100, C.MODELS["llama-3.1-70b"], s, 4000, 8) for s in ("bf16", "w8a8-fp8", "w4a16", "w4a4-nvfp4")}
smallest_ok = next(s for s in C.ACCURACY_ORDER if s in fits and fits[s] >= 32)
### END SOLUTION

# %% check
assert fits == {"bf16": 0, "w8a8-fp8": 0, "w4a16": 48, "w4a4-nvfp4": 43} and smallest_ok == "w4a16"
print("✅ 141 GB in bf16, 72.7 GB in FP8 (no room left at these round inputs; 2-4 sessions at vLLM's real budget), "
      "39.5 GB in INT4: on one H100 a 70B model is a 4-bit model; FP8 means two GPUs with tensor parallelism, or an H200")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We selected the scheme from the bottleneck, the hardware and the eval, in that order.
# Our traffic is decode-heavy chat on L4s, thus weight bytes and KV memory decide. FP8 W8A8 halves the weights, and
# it halves prefill on the FP8 tensor cores of Ada. An FP8 KV cache doubles the sessions: 87 instead of 17 for an 8B
# model on a 24 GB L4, simulated. This is also what cuts cost per token ~6×, because a larger batch shares every
# weight read.
#
# "INT4 weight-only decodes even faster, but it does not make prefill faster. Above ~120 tokens per step on an L4,
# its GEMM is BF16-math-bound, and by ~460, plain BF16 is as fast. Thus we keep INT4 for memory-bound cases: a T4,
# or a 70B on one 80 GB GPU. We examined what each checkpoint runs as on each generation. FP8 on an A100 is
# weight-only, NVFP4 is W4A4 only on Blackwell, and INT8 W8A8 disappears on Blackwell. Every scheme passed the eval
# budget before it got to this table."
#
# **Drills**
# 1. *Why does INT4 give 3× faster decode but no faster prefill on an L4?* Decode is a weight read (bytes ÷ 4).
#    Prefill is BF16 math on dequantized weights. It is compute-bound above ~120 tokens per step. Past ~460, it is
#    not faster than BF16, because BF16 is also compute-bound there.
# 2. *We have A100s. Is FP8 worth it?* For memory and decode bytes, yes (weight-only FP8 through Marlin). For
#    prefill, no, because A100s have no FP8 tensor cores. On an A100, INT8 W8A8 (with SmoothQuant) is the way to
#    make prefill faster.
# 3. *Why did FP8 cut the L4's cost per token 6×, not 2×?* There are fewer bytes per step, and five times the
#    sessions. FP8 weights free memory, and FP8 KV halves each session. Thus each step serves a much larger batch.
