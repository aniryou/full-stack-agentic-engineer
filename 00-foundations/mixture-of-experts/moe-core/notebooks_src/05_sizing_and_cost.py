# %% [markdown]
# # 05 · Sizing and cost
#
# **Tier:** T0 — arithmetic, a second; every time and cost here is a roofline bound (simulated) with illustrative
# prices. What actually fits a 16 or 24 GB GPU, measured, is `../moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb`
# (T1, with CPU offload and INT4 experts; it prints this notebook's sizing without a GPU).
#
# ## The one-minute version
# Three parameter counts size three different things. **HBM holds the total** — every expert, hot or cold — plus
# the KV cache, which follows attention and does not care about experts. **Prefill FLOPs follow the active count**:
# 2 × active × tokens, as `gpu-capacity-planning/capacity.py` computes them. **A decode step's time follows the bytes
# it streams at your batch** — near active at batch 1, near total at serving batches (notebook 03). So: pick the
# GPU count (the EP degree, with data-parallel attention) that holds the weights plus your batch's KV, check the step
# against the inter-token latency target, then divide GPU-hours by tokens. The cheap-looking MoE is cheap only at
# large batch on enough GPUs; on one small GPU it is a big model with a small model's FLOPs, and offloading experts
# to the CPU turns every step into a PCIe transfer.
#
# Primer: `../PRIMER.md` §7 *Sizing and cost* and §8 *In a design review: failure modes*; prices and where to get
# GPUs: the repo's `COMPUTE.md`.

# %%
from moecore import ep as E
from moecore import sizing as S
from moecore import touched as T

M, L, D = S.MODELS, E.LINKS, T.DEVICES
print(f"catalogue as of {S.AS_OF}; entries marked (verify) reproduce published totals from partly derived configs")

# %% [markdown]
# ## Worked example 1 — memory by total
# Weights at bf16, FP8, and with the routed experts at 4.25 bits (MXFP4: four bits plus an 8-bit scale per 32
# values — how gpt-oss ships) and the rest at bf16. KV per token is set by attention alone.

# %%
print(f"{'model':24s} {'total':>8s} {'active':>7s} {'bf16 GB':>8s} {'FP8 GB':>7s} {'4.25b experts':>13s} {'KV KiB/token':>12s}")
for key in ("olmoe-1b-7b", "qwen1.5-moe-a2.7b", "gpt-oss-20b", "qwen3-30b-a3b", "mixtral-8x7b", "gpt-oss-120b",
            "llama4-scout", "qwen3-235b-a22b", "llama4-maverick", "deepseek-v3", "llama-3.1-70b"):
    c = M[key]
    print(f"{c.name:24s} {c.total() / 1e9:7.1f}B {c.active() / 1e9:6.1f}B {S.weight_bytes(c) / 1e9:8.1f} "
          f"{S.weight_bytes(c, 8) / 1e9:7.1f} {S.weight_bytes(c, 16, 4.25) / 1e9:13.1f} {c.kv_bytes_per_token() / 1024:12.0f}")

# %% [markdown]
# gpt-oss-120b's 65.2 GB is why it fits one 80 GB GPU, and gpt-oss-20b's 13.8 GB why it fits 16 GB — on paper: vLLM's
# MXFP4 path needs compute capability 8.0 and bf16, so not a T4 (the fact sheet has the details).
#
# ## Worked example 2 — the capacity primer's Mistral Large 3, reproduced
# 675B total, 41B active (numbers from `00-foundations/gpu-capacity-planning`, not a config). In FP8 that is
# ~675 GB of weights — more than 8×H100's usable HBM once KV is added — and at large batch every expert is streamed
# every step: 675 GB ÷ (8 × 4.8 TB/s) is the floor across 8 H200s. Prefill, meanwhile, costs like a 41B dense model.

# %%
h200, h100 = D["h200"], D["h100-sxm"]
print(f"decode floor, all experts streamed, 8 x H200: {S.decode_floor(675e9, 8, h200) * 1e3:.1f} ms per step")
print(f"prefill FLOPs for a 2,048-token prompt: {S.prefill_flops(41e9, 2048):.3g} (like a 41B dense model; "
      f"a 675B dense model would need {S.prefill_flops(675e9, 2048) / S.prefill_flops(41e9, 2048):.1f}x)")
print(f"DeepSeek-V3 FP8 on 8 x H200, same floor: {S.decode_floor(S.weight_bytes(M['deepseek-v3'], 8), 8, h200) * 1e3:.1f} ms")

# %% [markdown]
# ## Worked example 3 — plans: how many GPUs, how fast, what per million tokens
# `S.plan` walks GPU counts 1, 2, 4, … and returns the first that holds the weights plus `batch` sequences of KV and
# meets the ITL target (DP attention + EP for MoE; tensor parallel for the dense model). The H100/H200 EP cases
# assume all-to-all kernels (DeepEP-class; DeepSeek-V3 dispatches in FP8); DeepEP does not run on an L4, so the L4
# case uses vLLM's default `allgather_reducescatter` over an assumed PCIe link (`L["pcie-l4"]`, the lab's value).
# Prices are illustrative placeholders per GPU-hour — look up real ones in `COMPUTE.md`.

# %%
FP8 = dict(dispatch_elem=1, scale_block=128)
cases = [("Mixtral-8x7B, bf16, H100", M["mixtral-8x7b"], h100, 64, 4096, 50, "nvlink4", 16, 3.0, "ep", {}),
         ("Llama-3.1-70B, bf16, H100", M["llama-3.1-70b"], h100, 64, 4096, 50, "nvlink4", 16, 3.0, "tp", {}),
         ("DeepSeek-V3, FP8, H200", M["deepseek-v3"], h200, 256, 4096, 50, "nvlink4", 8, 3.0, "ep", FP8),
         ("Qwen3-30B-A3B, FP8, L4", M["qwen3-30b-a3b"], D["l4"], 32, 4096, 60, "pcie-l4", 8, 0.7, "ep",
          dict(exchange="agrs"))]
print(f"{'case':28s} {'batch':>5s} {'GPUs':>4s} {'fits':>6s} {'step ms':>8s} {'tok/s':>7s} {'$/M tok':>8s}")
for name, cfg, dev, b, c, itl, lk, bits, usd, lay, kw in cases:
    p = S.plan(cfg, dev, b, c, itl, L[lk], bits, usd_per_gpu_hr=usd, layout=lay, **kw)
    print(f"{name:28s} {b:5d} {p.gpus:4d} {p.sessions:6d} {p.step_ms:8.1f} {p.tok_s:7.0f} {p.usd_per_mtok:8.2f}")

# %% [markdown]
# Mixtral (12.9B active) needs two H100s and costs about half as much per token as the dense 70B on four — at batch
# 64. Whether a dense 70B and Mixtral are "similar quality" depends on the model versions and your evals (verify);
# the method, not the verdict, is the point.
#
# ## Worked example 4 — long context: KV takes over again
# Mixtral on 2 H100s: as context grows the KV cache, not the experts, sets both the memory (how many sequences fit)
# and the step time — and the batch that fits shrinks, so the expert stream is shared by fewer tokens.

# %%
mix = M["mixtral-8x7b"]
for c in (4096, 32_768, 131_072):
    fit = S.sessions(mix, h100, 2, c, layout="ep")
    b = min(16, fit)
    st = E.decode_on(mix, h100, b, c, 2, "ep", L["nvlink4"])
    print(f"context {c:>7,}: {fit:3d} sequences fit; at batch {b:2d} the step is {st['time'] * 1e3:5.1f} ms "
          f"({b / st['time']:5.0f} tok/s), KV {T.kv_share(mix, b, c):4.0%} of the bytes")

# %% [markdown]
# ## Exercise 5.1 — does Qwen3-30B-A3B fit one 24 GB L4?
# On one small GPU the margin is the question, so budget it the way vLLM does (`S.kv_tokens`: 0.92 of the 22.49 GiB
# an L4 reports, minus ~1.5 GiB of activations and CUDA graphs, minus the weights). Set `gb` to a dict of weight GB
# for `"bf16"`, `"fp8"` and `"int4-experts"` (4.25-bit experts, the rest bf16), and `fits` to the list of those
# that leave room for at least 4 sequences of 4K tokens on one L4.

# %% exercise
### BEGIN SOLUTION
q = M["qwen3-30b-a3b"]
opts = {"bf16": (16, None), "fp8": (8, None), "int4-experts": (16, 4.25)}
gb = {k: S.weight_bytes(q, b, e) / 1e9 for k, (b, e) in opts.items()}
fits = [k for k, (b, e) in opts.items() if S.kv_tokens(q, D["l4"], b, e) >= 4 * 4096]
### END SOLUTION

# %% check
assert round(gb["bf16"], 1) == 61.1 and round(gb["fp8"], 1) == 30.5 and round(gb["int4-experts"], 1) == 18.5
assert fits == ["int4-experts"]
tok = S.kv_tokens(M["qwen3-30b-a3b"], D["l4"], 16, 4.25)
print(f"✅ only 4-bit experts fit: 18.5 GB of weights leaves {tok:,} tokens of KV = {tok // 4096} sequences of 4K at "
      f"vLLM's defaults (the round 10%-headroom budget would say {S.sessions(M['qwen3-30b-a3b'], D['l4'], 1, 4096, 16, 4.25)}). "
      "A 30B MoE is a 30B model for memory - its 3B active count only helps the FLOPs")

# %% [markdown]
# ## Exercise 5.2 — prefill is priced by active parameters
# For an 8,192-token prompt, set `ratio` = prefill FLOPs of Llama-3.1-70B ÷ those of Mixtral-8x7B
# (`S.prefill_flops(cfg.active(), tokens)`), and `ttft_mixtral_ms` = Mixtral's prefill time on one H100 at 50% of
# the bf16 peak (the capacity primer's MFU assumption).

# %% exercise
### BEGIN SOLUTION
ratio = S.prefill_flops(M["llama-3.1-70b"].active(), 8192) / S.prefill_flops(M["mixtral-8x7b"].active(), 8192)
ttft_mixtral_ms = S.prefill_flops(M["mixtral-8x7b"].active(), 8192) / (h100.peak() * 0.5) * 1e3
### END SOLUTION

# %% check
assert round(ratio, 2) == 5.48 and round(ttft_mixtral_ms) == 427
print(f"✅ Mixtral prefills {ratio:.1f}x cheaper than a dense 70B ({ttft_mixtral_ms:.0f} ms for 8K tokens at 50% MFU, "
      "simulated): prefill is where MoE's FLOP saving shows in full")

# %% [markdown]
# ## Exercise 5.3 — the floor and the plan
# For DeepSeek-V3 in FP8 on 8 H200s, set `floor_ms` (all weights streamed once across 8 GPUs) and `plan_ms` (the
# `S.plan` step at batch 256, 4K context, 50 ms ITL, NVLink), then `overhead` = plan_ms − floor_ms. Explain where the
# overhead comes from. (Use FP8 dispatch, `**FP8`, as in worked example 3.)

# %% exercise
### BEGIN SOLUTION
ds = M["deepseek-v3"]
floor_ms = S.decode_floor(S.weight_bytes(ds, 8), 8, h200) * 1e3
plan_ms = S.plan(ds, h200, 256, 4096, 50, L["nvlink4"], 8, **FP8).step_ms
overhead = plan_ms - floor_ms
### END SOLUTION

# %% check
assert round(floor_ms, 1) == 17.5 and round(plan_ms, 1) == 23.2 and 5 < overhead < 7
st = E.decode_on(ds, h200, 256, 4096, 8, "ep", L["nvlink4"], 1, **FP8)
print(f"✅ floor {floor_ms:.1f} ms, plan {plan_ms:.1f} ms: the difference is each rank's KV reads, its replicated "
      f"non-expert weights, and {st['comm'] * 1e3:.1f} ms of all-to-alls (simulated)")

# %% [markdown]
# ## Exercise 5.4 — MoE on a free T4, with CPU offload
# OLMoE-1B-7B in fp16 (13.8 GB) on a 16 GB T4, budgeted the way vLLM does: `S.kv_room_gib(o, D["t4"])` is what is
# left for KV after 0.92 × the 15.0 GiB the T4 reports, ~1.5 GiB of overhead and the weights. Set `room_gib` to it,
# `offload_gib` to the smallest `--cpu-offload-gb` (rounded up to 0.5 GiB) that holds 4 sequences of 4K tokens
# (`S.min_offload_gib`), `penalty_ms` to what streaming it over the T4's PCIe Gen3 costs per step
# (`S.offload_step_s(offload_gib, pcie_gbs=12)`, ~12 GB/s effective, verify), and `slowdown` to (step + penalty) ÷
# step, with the step at the batch the offload was sized for: `T.decode_step(o, D["t4"], 4, 4096, precision="fp16")`.

# %% exercise
### BEGIN SOLUTION
o = M["olmoe-1b-7b"]
room_gib = S.kv_room_gib(o, D["t4"])
offload_gib = S.min_offload_gib(o, D["t4"], 4 * 4096)
penalty_ms = S.offload_step_s(offload_gib, pcie_gbs=12) * 1e3
step_ms = T.decode_step(o, D["t4"], 4, 4096, precision="fp16").time * 1e3
slowdown = (step_ms + penalty_ms) / step_ms
### END SOLUTION

# %% check
assert room_gib < 0 and offload_gib == 3.0 and round(penalty_ms) == 268 and 10 < slowdown < 13
assert S.kv_tokens(o, D["t4"], offload_gib=offload_gib) >= 4 * 4096
print(f"✅ no KV room at all ({room_gib:.2f} GiB); --cpu-offload-gb {offload_gib:g} -> +{penalty_ms:.0f} ms per step on a "
      f"{step_ms:.0f} ms batch-4 step: {slowdown:.0f}x slower (simulated; the lab's `python -m moelab fit` prints the same "
      "offload; vLLM streams offloaded weights every forward pass - verify for your kernel). A 4-bit checkpoint "
      f"({S.kv_tokens(o, D['t4'], 16, 4.25):,} tokens of KV with 4-bit experts), or llama.cpp computing the offloaded "
      "experts on the CPU, is the better trade on a T4")

# %% [markdown]
# ## Exercise 5.5 — dense or MoE for this workload?
# Using the plans in worked example 3, set `cost_ratio` = the dense 70B's $/M tokens ÷ Mixtral's. Then compare
# Mixtral with a dense model of its active size (`S.dense_equivalent(mix)`: same attention and embeddings, one MLP
# two experts wide) on one H200 at 1K context: `ratio[b]` = Mixtral's decode step ÷ the dense model's for batches
# 1, 4 and 16 (`T.decode_step`), and `memory_ratio` = Mixtral's weight bytes ÷ the dense model's. For a low-traffic
# internal tool that runs at batch 1 on one GPU, is Mixtral the cheap choice?

# %% exercise
### BEGIN SOLUTION
pm = S.plan(M["mixtral-8x7b"], h100, 64, 4096, 50, L["nvlink4"], usd_per_gpu_hr=3.0)
pd = S.plan(M["llama-3.1-70b"], h100, 64, 4096, 50, L["nvlink4"], usd_per_gpu_hr=3.0, layout="tp")
cost_ratio = pd.usd_per_mtok / pm.usd_per_mtok
twin = S.dense_equivalent(mix)
ratio = {b: T.decode_step(mix, h200, b, 1024).time / T.decode_step(twin, h200, b, 1024).time for b in (1, 4, 16)}
memory_ratio = S.weight_bytes(mix) / S.weight_bytes(twin)
### END SOLUTION

# %% check
assert 1.9 < cost_ratio < 2.1 and abs(ratio[1] - 1) < 1e-3 and 2.4 < ratio[4] < 2.6 and 3.3 < ratio[16] < 3.5
assert 3.5 < memory_ratio < 3.7
print(f"✅ at batch 64 the MoE is {cost_ratio:.1f}x cheaper per token than the dense 70B. Against a dense model of its "
      f"active size it decodes exactly as fast at batch 1 ({ratio[1]:.2f}x) on {memory_ratio:.1f}x the memory, and is "
      f"{ratio[4]:.1f}x and {ratio[16]:.1f}x slower at batch 4 and 16 - MoE pays off with traffic, not without")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We size an MoE with three numbers. Memory is the total parameter count plus KV: a
# 30B-A3B model needs 61 GB in bf16, so a 24 GB card only works with 4-bit experts. Prefill is the active count: 2 ×
# active × tokens, so Mixtral prefills about 5× cheaper than a dense 70B. Decode is the bytes streamed at our batch,
# which approach the total by modest batches, so we run MoE at large batch on enough GPUs — the EP degree — to hold
# weights and the batch's KV and meet the ITL; the exchanges come on top. At batch 64 that makes Mixtral about half
# the cost per token of a dense 70B on H100s in this model; at batch 1 on one GPU it decodes like a dense 13B
# while holding 47B, and offloading experts over PCIe makes it an order of magnitude slower. At long context the KV cache
# dominates again, exactly as for a dense model."
#
# **Drill questions**
# 1. *'It's only 3B active, it'll run on my 24 GB card.' Answer in numbers.* — 30.5B total = 61 GB bf16, 30.5 GB FP8;
#    only ~18.5 GB with 4-bit experts, leaving ~2 GiB of KV at vLLM's defaults: five 4K sequences.
# 2. *Why does the plan's step exceed the all-experts floor?* — KV reads, replicated attention weights per rank
#    (DP attention), and two all-to-alls per MoE layer.
# 3. *When does a dense model win?* — Low, bursty traffic (small batches), one small GPU, memory-bound long-context
#    workloads, or a slow interconnect between the GPUs that would hold the experts.
