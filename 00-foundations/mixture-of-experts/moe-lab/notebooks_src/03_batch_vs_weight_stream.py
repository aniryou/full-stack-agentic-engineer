# %% [markdown]
# # 03 · Batch versus weight stream: why an MoE decodes like a small model at batch 1 and a big one at 64
#
# **Tier:** T1: one GPU that runs `vllm serve` for an MoE and for a dense model. A closed-loop client measures
# the decode step time at each batch. T0: the same curves from the roofline with stated efficiencies
# (**simulated**). Every formula has a check against the numbers of layer 01.
#
# ## The one-minute version
#
# * A decode step streams the weights one time for the whole batch. A dense model streams all of them at any
#   batch. Thus its step time is flat until the batch makes it compute-bound.
# * An MoE streams only the experts that its tokens touch: $E(1 - (1 - k/E)^T)$ per layer for $T$ tokens with
#   uniform routing. At batch 1, that is $k$ experts, and the MoE decodes like its *active* size. By
#   $T \approx 3E/k$, almost every expert streams, and the MoE decodes like its *total* size. Skewed routing
#   touches fewer experts.
# * The step becomes compute-bound only when each streamed expert sees sufficient tokens
#   ($\text{batch} \cdot k/E$ of them). Thus the crossover batch scales with total/active: 207 for
#   Llama-3.1-8B, 754 for Mixtral-8x7B and 2,055 for Qwen3-30B-A3B on an H200 (layer 01 PRIMER §3.6). MoE
#   wants large batches.
# * The fused MoE kernel sorts the token–expert pairs by expert. It pads the rows of each expert to a tile
#   (`BLOCK_SIZE_M`). At batch 1, almost every row that it multiplies is padding. This causes no problem,
#   because the step is memory-bound. The padding is also the reason that the tuned kernel configs use the
#   batch size as their key.
#
# Concepts: PRIMER §5 "MoE at inference: which experts a step touches" and §6 "Running MoE on GPUs"
# ([`PRIMER.md`](../../PRIMER.md)). The formulas are those of layer 01 ([roofline PRIMER
# §3.6](../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), `roofline.llm.experts_touched`,
# `streamed_weight_bytes`, `decode`, `decode_crossover_batch`). `moelab.stream` implements them again, and the
# cells of this notebook reproduce them digit for digit. Notebook 03 of the `moe-core` of this topic predicts
# the same curves. This notebook adds the measurement.

# %%
import os
import numpy as np
from moelab import configs, env, stream

print(env.describe())
H200 = configs.gpu("H200")
MIX, Q3, L8 = configs.get("mixtral-8x7b"), configs.get("qwen3-30b-a3b"), configs.get("llama-3.1-8b")


def detected_gpu():
    names = {"T4": "T4", "L4": "L4", "4090": "RTX4090", "A100": "A100-80GB", "H100": "H100-80GB", "H200": "H200"}
    for g in env.gpus():
        for key, name in names.items():
            if key in g:
                return configs.gpu(name)
    return configs.gpu("L4")                        # T0: simulate an L4 (24 GB, the GCP G2 / Cloud Run GPU)


GPU = detected_gpu()
MOE, DENSE = ((configs.get("olmoe-1b-7b"), configs.get("qwen2.5-1.5b")) if GPU.memory_gib >= 20
              else (configs.get("granite-3b-a800m"), configs.get("qwen2.5-1.5b")))
print(f"GPU for predictions: {GPU.name} | MoE {MOE.name} (active {MOE.active_params() / 1e9:.2f} B of "
      f"{MOE.total_params() / 1e9:.2f} B) vs dense {DENSE.name} ({DENSE.total_params() / 1e9:.2f} B)")

# %% [markdown]
# ## Worked example: layer 01's MoE table, reproduced
#
# This is Mixtral-8x7B on an H200 at 1K context (layer 01 PRIMER §3.6, where
# `roofline-core/tests/test_primer_numbers.py` pins the numbers). At batch 1, a decode step reads 2.00 experts
# per layer and 25.6 GB (5.34 ms). At batch 16, a decode step reads 7.92 experts and 94.4 GB (19.66 ms).

# %%
print("batch  Mixtral experts/layer  step bytes  step time   Qwen3-30B-A3B experts/layer")
for b in (1, 4, 16, 64):
    s = stream.decode_step(MIX, H200, b, 1024)
    print(f"{b:5d}  {stream.experts_touched(8, 2, b):21.2f}  {s.bytes / 1e9:7.1f} GB  {s.time * 1e3:6.2f} ms  "
          f"{stream.experts_touched(128, 8, b):14.1f} / 128")

# %% [markdown]
# ## Exercise 3.1 — experts touched, closed form
#
# Each token selects $k$ *distinct* experts of $E$ uniformly. Thus the probability that one token does not
# select a given expert is exactly $1 - k/E$. For $T$ independent tokens, this probability is $(1 - k/E)^T$.
# Write `touched(E, k, T)`: the expected number of distinct experts that a layer reads for $T$ tokens (1 for a
# dense model, $E = 0$).

# %% exercise
def touched(E, k, T):
    ### BEGIN SOLUTION
    if E == 0:
        return 1.0
    return E * (1 - (1 - k / E) ** T)
    ### END SOLUTION

# %% check
assert [round(touched(8, 2, b), 2) for b in (1, 4, 16, 64)] == [2.0, 5.47, 7.92, 8.0]            # layer 01 §3.6
assert [round(touched(128, 8, b), 1) for b in (1, 4, 16, 64)] == [8.0, 29.1, 82.4, 125.9]
assert touched(0, 0, 32) == 1.0
mc, _ = stream.touched_mc(64, 8, 16, s=0.0, trials=300)
assert abs(mc / touched(64, 8, 16) - 1) < 0.02
print(f"✅ Mixtral 2.00 -> 7.92 experts by batch 16; OLMoE (64, top-8) at batch 16: {touched(64, 8, 16):.1f} "
      f"(Monte Carlo {mc:.1f})")

# %% [markdown]
# ## Exercise 3.2 — skewed routing, by Monte Carlo
#
# Real routers are not uniform (notebook 02). Sample the routing. For popularity `p` over $E$ experts, sample
# the $k$ distinct experts of each token in proportion to `p`. Use the **Gumbel top-k** trick: the top-k of
# `log p + Gumbel noise` is a sample without replacement. Write `touched_skewed(p, k, T, trials, seed)`. It
# returns the mean number of distinct experts per layer.

# %% exercise
def touched_skewed(p, k, T, trials=200, seed=0):
    ### BEGIN SOLUTION
    rng = np.random.default_rng(seed)
    logp = np.log(np.asarray(p, float))
    out = []
    for _ in range(trials):
        picks = np.argsort(-(logp + rng.gumbel(size=(T, len(logp)))), axis=1)[:, :k]
        out.append(len(np.unique(picks)))
    return float(np.mean(out))
    ### END SOLUTION

# %% check
uniform = np.full(64, 1 / 64)
assert abs(touched_skewed(uniform, 8, 16) / touched(64, 8, 16) - 1) < 0.03
zipf = stream.zipf_popularity(64, 1.0)
t_u, t_z = touched_skewed(uniform, 8, 32), touched_skewed(zipf, 8, 32)
assert t_z < t_u
print("   batch   uniform   Zipf(1.0)   [simulated routing, OLMoE shape]")
for b in (1, 4, 16, 64, 256):
    print(f"   {b:5d}   {touched_skewed(uniform, 8, b, 100):7.1f}   {touched_skewed(zipf, 8, b, 100):9.1f}")
print("✅ skew touches fewer experts per step (good for one GPU's weight stream) — and piles load on the hot "
      "ones (bad for EP ranks, notebook 04)")

# %% [markdown]
# ## Exercise 3.3 — bytes one decode step reads
#
# Write `step_bytes(model, batch, context, wb=2, kvb=2)`. Add these parameters:
#
# 1. For each layer: attention, router, shared expert and `touched(E, k, batch)` routed experts. Multiply this
#    sum by the number of layers.
# 2. The LM head.
# 3. The `batch` rows that the step gathers from the input embedding.
#
# Multiply the total by `wb` bytes. Then add the KV cache that each sequence reads,
# `batch · (context + 1) · kv_bytes_per_token`. `configs.Model` has `attn_params()`, `expert_params()`,
# `shared_params()`, `router_params()`, `kv_bytes_per_token(kvb)`, `vocab` and `d_model`.

# %% exercise
def step_bytes(model, batch, context, wb=2, kvb=2):
    ### BEGIN SOLUTION
    e = touched(model.n_experts, model.top_k, batch) if model.is_moe else 1
    per_layer = model.attn_params() + e * model.expert_params() + model.shared_params() + model.router_params()
    weights = (model.layers * per_layer + model.vocab * model.d_model + batch * model.d_model) * wb
    return weights + batch * (context + 1) * model.kv_bytes_per_token(kvb)
    ### END SOLUTION

# %% check
assert round(step_bytes(MIX, 1, 1024) / 1e9, 1) == 25.6 and round(step_bytes(MIX, 16, 1024) / 1e9, 1) == 94.4
for m in (MIX, Q3, L8, MOE, DENSE):
    for b in (1, 8, 64):
        assert np.isclose(step_bytes(m, b, 512), stream.decode_step(m, GPU, b, 512).bytes)
kv64 = 64 * 1025 * MIX.kv_bytes_per_token()
print(f"✅ Mixtral on an H200: 25.6 GB at batch 1, 94.4 GB at 16; at batch 64 its KV is only "
      f"{kv64 / step_bytes(MIX, 64, 1024):.0%} of the step's bytes — MoE shifts the KV/weights ratio toward weights")

# %% [markdown]
# ## Exercise 3.4 — the batch where decode turns compute-bound
#
# Use `stream.decode_step(model, gpu, batch, context)`. Its `.bound` is "compute" or "memory". Write
# `crossover(model, gpu, context=0)`: the smallest batch whose step is compute-bound. The step changes only
# one time. Thus double the batch until the step changes. Then bisect.

# %% exercise
def crossover(model, gpu, context=0, max_batch=1 << 20):
    ### BEGIN SOLUTION
    def cb(b):
        return stream.decode_step(model, gpu, b, context).bound == "compute"
    hi = 1
    while not cb(hi):
        hi *= 2
        if hi > max_batch:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if cb(mid) else (mid, hi)
    return hi
    ### END SOLUTION

# %% check
x = {m.name: crossover(m, H200) for m in (L8, MIX, Q3)}
assert list(x.values()) == [207, 754, 2055], x                               # layer 01 PRIMER §3.6
ratio = lambda m: (m.total_params() - m.vocab * m.d_model) / (m.active_params() - m.vocab * m.d_model)  # noqa: E731
for m in (MIX, Q3):
    assert abs(x[m.name] / x[L8.name] / ratio(m) - 1) < 0.01
print(f"✅ H200 crossovers {x}; each MoE's is Llama-3.1-8B's x (total / active without the gathered embedding): "
      f"{ratio(MIX):.1f} and {ratio(Q3):.1f}. On this GPU: {MOE.name} {crossover(MOE, GPU)}, {DENSE.name} {crossover(DENSE, GPU)}")

# %% [markdown]
# ## Worked example: the simulated ITL curves on this GPU
#
# `stream.simulate_itl` = the roofline step divided by stated efficiencies, plus a per-step overhead
# (`SimParams`: 70% of bandwidth, 50% of FLOP/s, 2.5 ms). These values are assumptions until Exercise 3.6
# calibrates them on your GPU. The context is 512 tokens.

# %%
P = stream.SimParams()
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
sim = {m.name: [stream.simulate_itl(m, GPU, b, 512, P) * 1e3 for b in BATCHES] for m in (MOE, DENSE)}
print(f"[SIMULATED on {GPU.name}]  batch  {MOE.name:>18s}  {DENSE.name:>12s}  MoE/dense  experts touched  KV share")
for i, b in enumerate(BATCHES):
    s = stream.decode_step(MOE, GPU, b, 512)
    print(f"{'':25s}{b:5d}  {sim[MOE.name][i]:15.1f} ms  {sim[DENSE.name][i]:9.1f} ms  {sim[MOE.name][i] / sim[DENSE.name][i]:8.2f}"
          f"  {stream.experts_touched(MOE.n_experts, MOE.top_k, b):8.1f} / {MOE.n_experts}  {s.kv_bytes / s.bytes:7.0%}")

# %% [markdown]
# The MoE starts near the dense model (a similar active size). It increases until its full expert set streams.
# Then both curves become flat until the KV reads and the compute become the most important costs.
#
# For both models, the throughput still increases with batch. For the MoE, this is because the cost per token
# decreases when many tokens share the stream of every expert. That is why MoE serving is about large batches. It is also why
# expert parallelism (notebook 04) exists: to put more tokens in front of each expert per step.
#
# ## Exercise 3.5 — how the fused MoE kernel lays tokens out
#
# The vLLM `moe_align_block_size(topk_ids, block_size, num_experts)` does these steps:
#
# 1. It flattens the $T \times k$ assignments.
# 2. It groups them by expert (from the lowest id to the highest).
# 3. It pads the group of each expert to a multiple of `block_size`, with a pad id equal to $T \cdot k$.
# 4. It returns `(sorted_token_ids, expert_ids per block, num_tokens_post_padded)`.
#
# Then the Triton kernel runs one `BLOCK_SIZE_M`-row tile per block against the expert of that block. Write
# `align(topk_ids, block_size, num_experts)` with the same semantics (experts with no tokens get no blocks).
# The check replays the example in the vLLM docstring.

# %% exercise
def align(topk_ids, block_size, num_experts):
    ### BEGIN SOLUTION
    flat = np.asarray(topk_ids).ravel()
    pad = flat.size
    sorted_ids, expert_ids = [], []
    for e in range(max(num_experts, int(flat.max()) + 1)):
        rows = np.flatnonzero(flat == e).tolist()
        if rows:
            n = -(-len(rows) // block_size) * block_size
            sorted_ids += rows + [pad] * (n - len(rows))
            expert_ids += [e] * (n // block_size)
    return np.array(sorted_ids), np.array(expert_ids), len(sorted_ids)
    ### END SOLUTION

# %% check
s_ids, e_ids, n = align([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], 4, 4)
assert s_ids.tolist() == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12] and n == 16   # vLLM's docstring
assert e_ids.tolist() == [1, 2, 3, 4]
rng = np.random.default_rng(0)
ids = stream.sample_topk(np.full(64, 1 / 64), 8, 24, rng)
for got, want in zip(align(ids, 16, 64), stream.align_block_size(ids, 16, 64)):
    assert np.array_equal(got, want)
print("   OLMoE shape (64 experts, top-8), BLOCK_SIZE_M = 16: share of kernel rows that are padding")
for b in (1, 8, 32, 128, 512):
    print(f"   batch {b:4d}: {stream.padding_waste(b, 8, 64, 16, trials=20):5.1%}")
print("✅ at batch 1 each of the 8 touched experts gets a 16-row tile for 1 real row; the step is memory-bound, "
      "so it costs almost nothing — by batch 512 padding is about a tenth of the rows")

# %% [markdown]
# vLLM uses the batch size $M$ as the key of its tuned kernel configs. `E=64,N=1024,device_name=...json` maps
# $M$ to `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, warps and stages. vLLM uses the nearest $M$. The tree has no
# tuned files for T4, L4 or A10 (main, Sep 2026). Thus on those GPUs, vLLM logs *"Using default MoE config.
# Performance might be sub-optimal!"* (notebook 05 finds that line in a start-up log).
# `benchmarks/kernels/benchmark_moe.py` generates a tuned file for your GPU (T1).
#
# ## Worked example: the grouped GEMM (torch, optional)
#
# When the rows are in order of expert, the matmuls of the experts are one *grouped* GEMM: a list of
# `[rows_e, d] @ [d, n]` problems in one launch. Torch 2.14 supplies it as
# `torch.nn.functional.grouped_mm(a, b, offs=...)` (offsets = cumulative rows per expert). In the next cell,
# it runs in bf16 on the CPU, and the cell compares it with a per-expert loop.

# %%
if env.has_torch():
    import torch
    torch.manual_seed(0)
    d, n, E_ = 32, 16, 4
    ids = torch.tensor([[2], [0], [3], [0], [1], [3], [3], [0]])
    order = ids.ravel().argsort(stable=True)
    counts = torch.bincount(ids.ravel(), minlength=E_)
    xs = torch.randn(8, d, dtype=torch.bfloat16)[order]
    w = torch.randn(E_, d, n, dtype=torch.bfloat16)
    loop = torch.cat([xs[int(a):int(b)] @ w[e] for e, (a, b) in
                      enumerate(zip([0] + counts.cumsum(0)[:-1].tolist(), counts.cumsum(0).tolist()))])
    try:
        grouped = torch.nn.functional.grouped_mm(xs, w, offs=counts.cumsum(0).to(torch.int32))
        print("grouped_mm == per-expert loop:", torch.allclose(grouped.float(), loop.float(), atol=1e-1, rtol=1e-2),
              "| rows per expert", counts.tolist())
    except (AttributeError, RuntimeError, TypeError) as e:          # older torch or unsupported dtype on CPU
        print("grouped_mm not available here:", type(e).__name__, "- the per-expert loop is the reference")
else:
    print("T0 without torch: skipped (the numpy align above is the layout; the GEMMs are per expert)")

# %% [markdown]
# ## Exercise 3.6 — calibrate the simulation from two measurements
#
# The efficiencies of the simulation are guesses. On a real GPU, measure the ITL at two or more memory-bound
# batches. Then fit `itl = overhead + t_mem / eff`. Here `t_mem` is the memory time of the step at 100%
# bandwidth (`stream.decode_step(...).t_memory`).
#
# Write `calibrate(points)`. For a list of `(t_mem, itl)` pairs, it returns `(eff, overhead)` by least squares
# (a straight line: slope 1/eff, intercept overhead). The check recovers known parameters from simulated
# points.

# %% exercise
def calibrate(points):
    ### BEGIN SOLUTION
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(1 / slope), float(intercept)
    ### END SOLUTION

# %% check
truth = stream.SimParams(mem_eff=0.62, compute_eff=0.5, overhead_s=4e-3)
pts = [(stream.decode_step(DENSE, GPU, b, 512).t_memory, stream.simulate_itl(DENSE, GPU, b, 512, truth)) for b in (1, 4, 16)]
eff, ovh = calibrate(pts)
assert abs(eff - 0.62) < 1e-6 and abs(ovh - 4e-3) < 1e-9
assert np.allclose(calibrate(pts), stream.fit_efficiency(pts))
print(f"✅ recovered eff {eff:.2f} and overhead {ovh * 1e3:.1f} ms; with measured points this replaces SimParams' guesses")

# %% [markdown]
# ## On a real GPU (T1): measure both curves
#
# Start the two servers. On a small GPU, start one at a time. If both fit, use ports 8000 and 8001. Then point
# the notebook at them.
#
# `stream.measure_itl` runs a closed loop at each concurrency with constant output lengths (`ignore_eos`).
# Thus the engine decodes a batch of about that size, and the median ITL is the step time. With
# `MOELAB_START_VLLM=1`, the cell starts and stops the servers itself.
#
# ```bash
# vllm serve allenai/OLMoE-1B-7B-0924-Instruct --max-model-len 4096 --port 8000       # L4 / 24 GB (bf16)
# vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 --port 8001
# MOELAB_URL=http://127.0.0.1:8000 MOELAB_DENSE_URL=http://127.0.0.1:8001 jupyter lab notebooks/
# ```
#
# On a T4, add `--dtype half`. Use `ibm-granite/granite-3.0-3b-a800m-instruct` as the MoE. Without offload,
# OLMoE in fp16 leaves no space for KV on 16 GB (notebook 05).

# %%
MEASURE = [1, 2, 4, 8, 16, 32]
measured = {}
targets = [(MOE, env.server_url()), (DENSE, env.server_url(dense=True))]
if any(u for _, u in targets):
    for m, url in targets:
        if url:
            name = env.served_model(url, env.auth_headers())
            pts = stream.sweep_itl(url, name, MEASURE, output_tokens=64, headers=env.auth_headers())
            measured[m.name] = {b: p.median_itl_ms for b, p in pts.items()}
elif env.may_start_vllm():
    flags = ["--max-model-len", "4096"] + (["--dtype", "half"] if not GPU.bf16 else [])
    for m in (MOE, DENSE):
        with env.VLLMServer(m.hf_id, flags) as url:
            pts = stream.sweep_itl(url, m.hf_id, MEASURE, output_tokens=64)
            measured[m.name] = {b: p.median_itl_ms for b, p in pts.items()}
if measured:
    for name, pts in measured.items():
        model = MOE if name == MOE.name else DENSE
        mem = [(stream.decode_step(model, GPU, b, 64 + 32).t_memory, v / 1e3) for b, v in pts.items() if b <= 8]
        eff, ovh = calibrate(mem)
        print(f"MEASURED {name}: " + "  ".join(f"b{b}={v:.1f}ms" for b, v in pts.items())
              + f"  -> bandwidth efficiency {eff:.2f}, overhead {ovh * 1e3:.1f} ms")
else:
    print("T0: no server (MOELAB_URL / MOELAB_DENSE_URL) and no local GPU+vLLM; the curves above are simulated.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "The cost of a decode step is the bytes that it streams. A dense model streams all its
# weights at any batch. An MoE streams only the experts that its tokens touch. That is $k$ per layer at batch
# 1, and almost all $E$ when the batch is more than a few times $E/k$. Thus OLMoE decodes approximately like a
# 1.3B model for one user and like a 7B model for thirty-two. Mixtral on an H200 goes from 25.6 GB to 94.4 GB
# per step between batch 1 and 16.
#
# "The step stays memory-bound until each expert sees sufficient tokens. Thus the compute-bound crossover
# moves out by total/active: 754 for Mixtral against 207 for Llama-3.1-8B.
#
# "For these reasons, we plan MoE capacity at high batch, where the cost per token is lowest. We treat batch-1 latency as
# the easy case. The simulation gives the shape, and two measured points calibrate it."
#
# **Drill 1.** *Our MoE has 1/5 the active parameters of the dense model, but only 1.3x its throughput at
# batch 64. Why?* At batch 64, it streams (almost) all experts in every step. The bytes depend on the total
# count, not on the active count. The active count sets the FLOPs, which are important only when the step is
# compute-bound.
#
# **Drill 2.** *Does skewed routing help or hurt?* On one GPU, it helps slightly (it streams fewer distinct
# experts). Under EP, it hurts (the GPU that holds the hot experts is the slowest).
#
# **Drill 3.** *Is the padding in the fused MoE kernel wasted money at batch 1?* It wastes FLOPs, not time.
# The step is memory-bound. Thus the idle tensor cores cost nothing. It is important near the compute-bound
# regime, where a tuned `BLOCK_SIZE_M` for each batch size recovers it.
