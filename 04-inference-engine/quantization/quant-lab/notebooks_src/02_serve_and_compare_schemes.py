# %% [markdown]
# # 02 · Serve and compare schemes: what INT4, FP8 and FP4 buy on the GPU you have
#
# **Tier:** T0 — the scheme rules come from vLLM's source, the speeds from a roofline engine emulator
# and a fake OpenAI-compatible server built on it: every speed here is **simulated**. T1 — set
# `QUANTLAB_URL` to a real `vllm serve` (started with `deploy/any-gpu/serve.sh`: FP16 vs INT4 vs FP8
# on a 24 GB card; on a T4 the INT4 path with `--dtype half`) and the same client measures it.
#
# ## The one-minute version
#
# A scheme changes two numbers: the **bytes** each step reads and the **FLOP/s** its GEMMs run at.
# Decode reads every weight once per step, so it follows bytes; prefill multiplies thousands of
# tokens against the same weights, so it follows FLOP/s.
#
# * **Weight-only INT4 (W4A16)** reads a quarter of the weight bytes and still multiplies in 16-bit:
#   decode ~3x faster, prefill no faster (the dequantization is extra work). It runs on everything
#   from a T4 (Marlin; Machete on Hopper).
# * **FP8 W8A8** halves bytes *and* doubles tensor-core FLOP/s — but only where FP8 tensor cores exist
#   (sm_89+: L4, RTX 4090, H100, B200). On a T4 or A100 vLLM runs the same checkpoint as weight-only
#   FP8 through Marlin: a memory win, no FLOP win.
# * **NVFP4 W4A4** quarters bytes and quadruples FLOP/s on Blackwell (sm_100+); elsewhere it is weight-only.
# * **INT8 W8A8** runs on INT8 tensor cores from Turing to Hopper and is refused on Blackwell.
#
# Concepts: PRIMER §1 "Why quantize, and what it can and cannot speed up", §4 (kernels) and §10
# "Choosing a scheme" ([`PRIMER.md`](../../PRIMER.md)); the roofline itself is layer 01's
# ([`../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md`](../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §1-3).

# %%
from quantlab import bench as B, env, report, serve
from quantlab.fakeserver import FakeServer

print(env.describe())
print(serve.matrix(gpus=("T4", "L4", "H100-80GB", "B200"), schemes=("bf16", "fp8", "w4a16", "w8a8-int8", "nvfp4")))

# %% [markdown]
# Read one row per GPU generation. The **kernel** column is what vLLM should log at startup
# (`Using MarlinLinearKernel for ...`, `Selected CutlassFP8ScaledMMLinearKernel for ...`): when your
# log says otherwise, the plan is wrong for your version — check it (verify) before trusting a
# benchmark.
#
# ## Worked example: one GEMM on the roofline
#
# Llama-3.1-8B's `down_proj` is `[M, 14,336] x [14,336, 4,096]` for `M` tokens in the step. Its time
# is `max(2MKN / peak, (KN w + MK a + MN 2) / bandwidth)`, with `w`, `a` the weight and activation
# bytes (W4A16 counts its group scales and zero points: 4.16 bits).

# %%
for M in (1, 16, 64, 256, 2048):
    t = {s: B.gemm_time(M, 14336, 4096, "L4", s) * 1e6 for s in ("bf16", "w4a16", "fp8")}
    print(f"   L4, M={M:5d}: " + "  ".join(f"{s} {v:7.0f} us" for s, v in t.items()))

# %% [markdown]
# ## Exercise 2.1 — the GEMM roofline
#
# Write `gemm_us(M, K, N, w_bytes, a_bytes, peak_tflops, bw_gbs)`: microseconds for the GEMM at 100%
# of peak, output written in 16-bit. The check reproduces vllm-internals §8.1's table (L4: 121 BF16
# and 242.5 FP8 dense TFLOP/s, 300 GB/s — datasheet numbers, verify).

# %% exercise
def gemm_us(M, K, N, w_bytes, a_bytes, peak_tflops, bw_gbs):
    ### BEGIN SOLUTION
    flops = 2.0 * M * K * N
    byts = K * N * w_bytes + M * K * a_bytes + M * N * 2
    return max(flops / (peak_tflops * 1e12), byts / (bw_gbs * 1e9)) * 1e6
    ### END SOLUTION

# %% check
w4 = (4 + 16 / 128 + 4 / 128) / 8
table = {M: (round(gemm_us(M, 14336, 4096, 2, 2, 121, 300)), round(gemm_us(M, 14336, 4096, w4, 2, 121, 300)),
             round(gemm_us(M, 14336, 4096, 1, 1, 242.5, 300))) for M in (1, 256, 2048)}
assert table == {1: (392, 102, 196), 256: (423, 248, 215), 2048: (1988, 1988, 992)}, table
print("✅ BF16 / W4A16 / FP8, microseconds:", table, "— vllm-internals §8.1, recomputed")

# %% [markdown]
# ## Exercise 2.2 — where the INT4 advantage starts to fade
#
# W4A16 is memory-bound (fast) while its FLOP time is below its byte time. Write
# `compute_bound_m(K, N, w_bytes, a_bytes, peak_tflops, bw_gbs)`: the smallest `M` at which the FLOP
# time reaches the byte time. Predict first: an L4's BF16 ridge is 121e12 / 300e9 = 403 FLOP/byte,
# and a W4A16 GEMM does about `2M / 0.52` FLOP per weight byte.

# %% exercise
def compute_bound_m(K, N, w_bytes, a_bytes, peak_tflops, bw_gbs):
    ### BEGIN SOLUTION
    M = 1
    while 2.0 * M * K * N / (peak_tflops * 1e12) < (K * N * w_bytes + M * K * a_bytes + M * N * 2) / (bw_gbs * 1e9):
        M += 1
    return M
    ### END SOLUTION

# %% check
l4 = compute_bound_m(14336, 4096, w4, 2, 121, 300)
h100 = compute_bound_m(14336, 4096, w4, 2, 989.4, 3350)
assert l4 == B.compute_bound_tokens(14336, 4096, "L4") and h100 == B.compute_bound_tokens(14336, 4096, "H100-80GB")
assert 110 <= l4 <= 130 and 75 <= h100 <= 95
print(f"✅ W4A16 turns compute-bound at M = {l4} on an L4 and {h100} on an H100 (vllm-internals: ~120 / ~85); "
      f"its edge over BF16 is gone by M = {B.crossover_tokens(14336, 4096, 'L4')} on the L4, and any dequantization "
      "cost makes it slower than BF16 above that")

# %% [markdown]
# ## Worked example: the whole engine, simulated — one user versus sixteen
#
# `bench.profile` turns model + GPU + scheme into a step-time model (weights streamed per decode
# step, KV bytes per token, the FLOP/s of the path the scheme takes on that GPU, 60% / 80% compute /
# memory efficiency — assumptions). `closed_loop` runs a continuous-batching emulator on it: each
# user sends a 1,024-token prompt and reads 128 tokens, then sends the next.

# %%
rows = []
for users in (1, 16):
    for r in B.compare(["bf16", "fp8", "w4a16"], "llama-3.1-8b-instruct", "L4", users=users, n_requests=2 * users,
                       prompt_len=1024, output_len=128):
        rows.append({"users": users, **{k: r[k] for k in ("scheme", "ttft_ms_mean", "tpot_ms_mean", "output_tok_s")}})
print("Llama-3.1-8B on an L4 [SIMULATED]")
print(report.markdown(rows))
t4 = B.compare(["bf16", "fp8", "w4a16"], "qwen2.5-1.5b-instruct", "T4", users=8, n_requests=16, prompt_len=1024, output_len=128)
print("\nQwen2.5-1.5B on a T4, 8 users [SIMULATED]")
print(report.markdown([{k: r[k] for k in ("scheme", "ttft_ms_mean", "tpot_ms_mean", "output_tok_s")} for r in t4]))

# %% [markdown]
# On the L4, FP8 improves both TTFT and TPOT; INT4 improves only TPOT. On the T4 the FP8 checkpoint
# is weight-only (no FP8 tensor cores): it reads half the bytes, so decode improves, but its TTFT
# is the BF16 one.
#
# ## Worked example: measuring over HTTP, the way you would measure a real server
#
# The fake server runs the same emulator in real time behind `/v1/completions` (SSE streaming). The
# client times each chunk — TTFT is the first token-carrying chunk, ITL the gaps — exactly as it would
# against `vllm serve`. Its `/version` says `"simulated": true`, so the label follows the server, not
# the code path. Set `QUANTLAB_URL` and this cell measures your server instead.

# %%
if env.server_url():
    url = env.server_url()
    live = B.run_http(url, env.served_model(url), users=4, n_requests=16, prompt_len=512, output_len=64,
                      headers=env.auth_headers())
    print(live)
else:
    for scheme in ("bf16", "w4a16"):
        with FakeServer(B.profile("qwen2.5-1.5b-instruct", "L4", scheme)) as url:
            r = B.run_http(url, "quantlab-fake", users=4, n_requests=8, prompt_len=512, output_len=32)
        print(f"{scheme:6s} TTFT {r['ttft_ms_mean']:6.1f} ms  TPOT {r['tpot_ms_mean']:5.1f} ms  "
              f"{r['output_tok_s']:6.0f} tok/s   {r['source']}")

# %% [markdown]
# ## Exercise 2.3 — predict a decode step before simulating it
#
# One decode step for `b` sequences at context `c` costs
# `overhead + max((2 x params_per_token x b + attn_flops_per_pair x b x c) / (peak x compute_eff), (streamed_bytes + b x c x kv_bytes) / (bw x memory_eff))`
# — every decoded token goes through every linear layer and the LM head, and attends to `c` positions.
# Write `step_ms(p, b, c)` from a `bench.Profile`'s fields (`params_per_token`, `attn_flops_per_pair`,
# `streamed_bytes`, `kv_bytes_per_token`, `peak_flops`, `mem_bw`, `compute_eff`, `memory_eff`, `overhead_s`).

# %% exercise
def step_ms(p, b, c):
    ### BEGIN SOLUTION
    compute = (2 * p.params_per_token * b + p.attn_flops_per_pair * b * c) / (p.peak_flops * p.compute_eff)
    memory = (p.streamed_bytes + b * c * p.kv_bytes_per_token) / (p.mem_bw * p.memory_eff)
    return (p.overhead_s + max(compute, memory)) * 1e3
    ### END SOLUTION

# %% check
for scheme in ("bf16", "fp8", "w4a16"):
    p = B.profile("llama-3.1-8b-instruct", "L4", scheme)
    assert abs(step_ms(p, 16, 1100) - p.step_time([1100] * 16) * 1e3) < 1e-9
    assert abs(step_ms(p, 256, 64) - p.step_time([64] * 256) * 1e3) < 1e-9          # a compute-bound decode too
bf, i4 = (B.profile("llama-3.1-8b-instruct", "L4", s) for s in ("bf16", "w4a16"))
print(f"✅ batch 16 at 1,100 tokens: bf16 {step_ms(bf, 16, 1100):.1f} ms, w4a16 {step_ms(i4, 16, 1100):.1f} ms per step "
      f"({step_ms(bf, 16, 1100) / step_ms(i4, 16, 1100):.2f}x) — less than the {bf.streamed_bytes / i4.streamed_bytes:.2f}x "
      "the weights alone give: the KV read and the per-step overhead do not shrink")

# %% [markdown]
# ## Exercise 2.4 — pick a scheme for three deployments, and say why
#
# Write `pick(gpu, prompt_len, output_len)`: among `bf16`, `fp8`, `w4a16`, `w8a8-int8` and `nvfp4`,
# keep the schemes `serve.plan` says the GPU runs, simulate 8 users (`B.compare`), and return the one
# with the lowest mean end-to-end latency `ttft + (output_len - 1) x tpot`. Speed only — accuracy is
# notebook 03's job.

# %% exercise
def pick(gpu, prompt_len, output_len):
    ### BEGIN SOLUTION
    ok = [s for s in ("bf16", "fp8", "w4a16", "w8a8-int8", "nvfp4") if serve.plan(s, gpu).supported]
    rows = B.compare(ok, "qwen2.5-1.5b-instruct", gpu, users=8, n_requests=16, prompt_len=prompt_len, output_len=output_len)
    return min(rows, key=lambda r: r["ttft_ms_mean"] + (output_len - 1) * r["tpot_ms_mean"])["scheme"]
    ### END SOLUTION

# %% check
chat_t4 = pick("T4", 256, 512)            # short prompts, long answers: decode-bound
rag_l4 = pick("L4", 4096, 32)              # long prompts, short answers: prefill-bound
b200 = pick("B200", 2048, 256)
assert chat_t4 == "w4a16", chat_t4                      # 4-bit weights win decode on a T4; NVFP4 is 4.5 bits there
assert rag_l4 in ("fp8", "w8a8-int8"), rag_l4           # only W8A8 halves prefill FLOP time
assert b200 == "nvfp4", b200                            # W4A4 on FP4 tensor cores
print(f"✅ T4 chat: {chat_t4}; L4 long-prompt RAG: {rag_l4}; B200: {b200} [SIMULATED speeds — gate each on an eval]")

# %% [markdown]
# On a T4, `nvfp4` is in the running only as a weight-only format: 4.5 bits read through Marlin,
# slightly more bytes than INT4 g128's 4.16, so INT4 wins — pick between INT4 checkpoints by their
# evals. On a B200 the FP4 result holds only if the W4A4 accuracy does (notebook 05 shows how badly
# FP4 *activations* can go without smoothing).
#
# ## On a real GPU (T1)
#
# `deploy/any-gpu/serve.sh` checks the scheme against the GPU and starts vLLM; then point this
# notebook at it. The commands per scheme on a 24 GB L4, and on a free T4:

# %%
for s in ("bf16", "fp8-online", "w4a16"):
    print(serve.plan(s, "L4").docker())
print(serve.plan("w4a16", "T4").command(), "   # T4: INT4 with fp16 activations")
print("then: QUANTLAB_URL=http://127.0.0.1:8000 jupyter lab notebooks/   (or python -m quantlab bench --url ...)")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Quantization buys bytes or FLOPs, and which one decides where it helps. On our
# L4s, INT4 weight-only makes decode 2-3x faster per step for an 8B model but does nothing for
# a long prompt's TTFT — its GEMMs still run in 16-bit: past ~120 tokens per step the dequantizing
# kernel is compute-bound, and by ~460 it is no faster than BF16. FP8 W8A8 halves both on the L4's FP8 tensor cores. On T4s the same FP8
# checkpoint would run weight-only through Marlin, so there INT4 is the speed lever. We confirm the
# kernel in vLLM's startup log, benchmark with the same client against the real server, and only then
# look at cost."
#
# **Drill 1.** *We moved an FP8 checkpoint from an L4 fleet to A100s and prefill got slower relative to
# BF16 expectations. Why?* — A100 has no FP8 tensor cores; vLLM falls back to W8A16 via Marlin (weight
# memory halves, math stays 16-bit, plus dequantization).
#
# **Drill 2.** *INT4 decode is 3x faster at batch 1 but only 1.5x at batch 64 — measurement error?* — No:
# the step still reads the weights once, but it also reads 64 sequences' KV cache, which INT4 weights do
# not shrink — at typical contexts about half the bytes (layer 01 PRIMER §3.5: 3.80x at batch 1, 1.54x at
# batch 64 for Llama-3.1-8B on an H100). The GEMMs are not compute-bound yet (that starts at ~120 tokens
# per step on an L4); past it W4A16's edge shrinks, and by ~460 it is gone.
#
# **Drill 3.** *Can we run INT8 W8A8 on B200s?* — vLLM refuses it on compute capability >= 10.0; use FP8
# or NVFP4 there.
