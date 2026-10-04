# %% [markdown]
# # 05 · Speculative decoding and quantization in vLLM: what each buys, read from the engine's counters
#
# **Tier:** T0: the arithmetic, plus the fake vLLM, which emulates speculation. In the fake vLLM, the acceptance
# is an explicit *assumption*, and the results are **simulated**. T1: the notebook prints the vLLM flags for
# each experiment. Against a real server (`SERVELAB_URL`), the acceptance cells read the spec-decode counters
# of vLLM itself.
#
# ## The one-minute version
#
# The two techniques attack the same fact: **decode is bound by memory bandwidth**. Each step reads all the
# weights to make one token per sequence.
#
# * **Speculative decoding** proposes $k$ tokens at a low cost: an n-gram lookup in the prompt, a small draft
#   model, or EAGLE/MTP heads. Then it verifies all of them in *one* forward pass of the target. With a
#   per-token acceptance $\alpha$, a step gives $(1 - \alpha^{k+1})/(1 - \alpha)$ tokens on average. The
#   rule of rejection sampling keeps the output distribution *exactly* the distribution of the target model.
#   Speculation pays while the step is memory-bound (small batches). The gain decreases when larger batches
#   make the step compute-bound.
# * **Quantization** decreases the bytes. Weight-only INT4 (AWQ/GPTQ) or FP8 weights decrease the bytes that
#   each decode step reads. Thus decode is faster, and there is more space for KV. FP8 *W8A8* also doubles
#   the tensor-core FLOP/s on GPUs that have FP8 units, and thus prefill is faster. FP8 KV decreases the
#   KV cache to half its size. You must measure the accuracy on your own evals.
#
# Concepts: PRIMER §7 "Speculative decoding" and §8 "Quantization" ([`PRIMER.md`](../../PRIMER.md)). The
# `mini-engine-core` of this topic implements the exact rejection sampler by hand.

# %%
import math
from servelab import env, metrics as M, sizing
from servelab.bench import Lengths, random_requests, run_closed_loop
from servelab.fake_engine import EngineConfig, FakeEngine, build_profile, profile, simulate, tiny_profile
from servelab.tune import FakeBackend, sweep, to_cli_flags, trials_table

print(env.describe())
print("expected tokens per verify step, (1 - a^(k+1)) / (1 - a):")
print("alpha " + "".join(f"  k={k:<4}" for k in (1, 2, 4, 8)))
for a in (0.5, 0.7, 0.9):
    print(f"{a:5.1f} " + "".join(f"{(1 - a ** (k + 1)) / (1 - a):8.2f}" for k in (1, 2, 4, 8)))

# %% [markdown]
# Each increase in $k$ gives less than the increase before it. At $\alpha = 0.7$, the 5th to 8th draft
# tokens add only 0.4 tokens per step, but the verifier must process all of them. The method names are as
# in the `SpeculativeConfig` of vLLM v0.30.0, and they change between releases. These are the vLLM flags
# for the three families:

# %%
for spec in ({"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
             {"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4},
             {"method": "eagle3", "model": "<an EAGLE-3 head trained for the target>", "num_speculative_tokens": 3}):
    print("vllm serve <target>", " ".join(to_cli_flags({"speculative_config": spec})))

# %% [markdown]
# ## Exercise 5.1 — expected tokens per verification step
#
# Write `expected_tokens(alpha, k)`. It returns the mean number of tokens that one verify step emits. The
# assumption: the verify step accepts each draft token independently with the probability `alpha`, and the first
# rejection stops the run. Add the one token that the target always gives. Make sure that the function
# also handles `alpha = 1`.

# %% exercise
def expected_tokens(alpha: float, k: int) -> float:
    ### BEGIN SOLUTION
    if alpha >= 1.0:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)
    ### END SOLUTION

# %% check
assert expected_tokens(0.0, 4) == 1.0 and expected_tokens(1.0, 4) == 5.0
assert math.isclose(expected_tokens(0.7, 4), 1 + 0.7 + 0.49 + 0.343 + 0.2401)
eng = FakeEngine(tiny_profile(num_blocks=2048, max_model_len=8192),
                 EngineConfig(num_speculative_tokens=4, spec_acceptance=0.7, seed=3))
(s,) = simulate(eng, [(0.0, [1, 2, 3], 4000)])
observed = (len(s.output) - 1) / (len(s.emit_times) - 1)
assert abs(observed / expected_tokens(0.7, 4) - 1) < 0.04
print(f"✅ formula {expected_tokens(0.7, 4):.3f} vs simulated {observed:.3f} tokens per step")

# %% [markdown]
# ## Worked example: speculation on the fake engine, one user versus eight
#
# `FakeBackend` changes the `speculative_config` of vLLM into the settings of the emulator. The acceptance
# rate is an assumption that you give to the backend, `FakeBackend(spec_acceptance=0.7)`. It never goes
# into the config. Thus the config stays one that `vllm serve --speculative-config` accepts. On a real
# engine, the acceptance is a property of your traffic and of the draft method. This is why you read it
# from `/metrics` and do not assume it.
#
# To do a load test of a real engine at a *selected* acceptance, vLLM v0.30.0 has a synthetic mode:
# `"rejection_sample_method": "synthetic"` with `"synthetic_acceptance_length"`. In this mode, the outputs
# are not the outputs of the target model. Use the mode for benchmarks only.
#
# The fake engine has no one-time start-up costs. Thus these sweeps do not do a warm-up.

# %%
work = lambda: random_requests(12, Lengths.fixed(256), Lengths.fixed(96), seed=9)  # noqa: E731
SPEC = {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}}
results = {}
for users in (1, 8):
    results[users] = sweep(FakeBackend("t4-qwen2.5-0.5b", spec_acceptance=0.7), [{}, SPEC], work,
                           concurrency=users, warmup=0)
    print(f"[SIMULATED] {users} concurrent user(s)")
    print(trials_table(results[users]))
spec_trial = results[1][1]
print(spec_trial.snapshot.table())

# %% [markdown]
# ## Exercise 5.2 — acceptance, the way vLLM's dashboards compute it
#
# vLLM exports three counters: `vllm:spec_decode_num_drafts` (the verify steps with drafts),
# `..._num_draft_tokens` and `..._num_accepted_tokens`. The PromQL in the vLLM documentation is
#
# $$
# \text{acceptance rate} = \frac{\operatorname{rate}(\text{accepted})}{\operatorname{rate}(\text{draft tokens})}
# $$
#
# and
#
# $$
# \text{mean acceptance length} = 1 + \frac{\operatorname{rate}(\text{accepted})}{\operatorname{rate}(\text{drafts})}
# $$
#
# The 1 counts the token that the target itself gives. Write `acceptance(before, after)`. It returns
# `(rate, mean_length)` from two scrapes.

# %%
from servelab.fakeserver import FakeServer
srv = FakeServer("t4-qwen2.5-0.5b", EngineConfig(num_speculative_tokens=4, spec_acceptance=0.7))
URL = srv.start()
before = M.scrape(URL)
run_closed_loop(URL, work(), concurrency=2)
after = M.scrape(URL)
srv.stop()

# %% exercise
def acceptance(before, after) -> tuple:
    ### BEGIN SOLUTION
    d = lambda name: after.value(name, 0.0) - before.value(name, 0.0)  # noqa: E731
    drafts, draft_tokens, accepted = d(M.SPEC_DRAFTS), d(M.SPEC_DRAFT_TOKENS), d(M.SPEC_ACCEPTED)
    return accepted / draft_tokens, 1 + accepted / drafts
    ### END SOLUTION

# %% check
rate, length = acceptance(before, after)
snap = M.snapshot(after, before)
assert math.isclose(rate, snap.spec_acceptance_rate) and math.isclose(length, snap.spec_mean_acceptance_length)
assert abs(length / expected_tokens(0.7, 4) - 1) < 0.1          # the counters and the formula agree
print(f"✅ [SIMULATED] acceptance rate {rate:.1%}, mean acceptance length {length:.2f} "
      f"(formula at a=0.7, k=4: {expected_tokens(0.7, 4):.2f})")

# %% [markdown]
# Note the difference between the two numbers. The **acceptance rate** is per draft token. Here it is 0.44,
# but $\alpha = 0.7$. The rate is less than $\alpha$ because the verify step gets to the later positions
# less frequently. The **mean acceptance length** is the number that sets the speedup.
#
# ## Exercise 5.3 — where speculation pays: batch size
#
# A verify step processes $\mathtt{batch} \times (k + 1)$ tokens and reads the weights one time. The draft
# work adds $k \times \mathtt{draft\_cost} \times (\text{weight read time})$. Write
# `spec_speedup(p, batch, context, alpha, k, draft_cost)`. It returns the baseline time per token divided
# by the speculative time per token:
#
# - The baseline time per token is one decode step per token.
# - The speculative time per token is one verify step plus the draft work, divided by the expected tokens
#   per step.
#
# Use the roofline as in notebook 03:
#
# $$
# \text{overhead} + \max\left(\frac{2 \cdot P \cdot \text{tokens}}{\text{FLOP/s}},
#   \frac{\text{weights} + \text{batch} \cdot \text{context} \cdot \text{kv}}{\text{bandwidth}}\right)
# $$
#
# Predict first. At which batch sizes does speculation no longer pay? Why is the context length important?
# Each operating point in the check must fit in the KV cache of the profile. A batch that the engine cannot
# hold tells you nothing about a real engine.

# %% exercise
def spec_speedup(p, batch: int, context: int, alpha: float, k: int, draft_cost: float = 0.0) -> float:
    ### BEGIN SOLUTION
    def step(tokens_per_seq):
        compute = 2 * p.active_params * batch * tokens_per_seq / p.flops
        memory = (p.weight_bytes + batch * context * p.kv_bytes_per_token) / p.mem_bw
        return p.overhead_s + max(compute, memory)
    base = step(1)
    spec = step(k + 1) + k * draft_cost * p.weight_bytes / p.mem_bw
    return base / (spec / expected_tokens(alpha, k))
    ### END SOLUTION

# %% check
L4_8B = build_profile("llama-3.1-8b-instruct", "L4", quantization="fp8", kv_cache_dtype="fp8", max_model_len=8192)
slots = lambda p: p.num_blocks * p.block_size  # noqa: E731 — KV token slots of the profile
sp = {b: spec_speedup(L4_8B, b, 256, 0.7, 4, draft_cost=0.05) for b in (1, 8, 32, 128, 256, 512)}
assert all(b * 256 <= slots(L4_8B) for b in sp), "every batch x context must fit the L4's KV cache"
for b, v in sp.items():
    print(f"   L4 FP8, batch {b:>3}, 256-token context: speedup {v:4.2f}x")
assert sp[1] > 2.0 and sp[128] < sp[1] and sp[256] < 1.0 and sp[512] < sp[256]
H100_8B = profile("h100-llama3.1-8b")     # bf16 on an H100: enough KV for 256 sequences of 1,024 tokens
assert 256 * 1024 <= slots(H100_8B) and 256 * 2048 > slots(H100_8B)
short_ctx, long_ctx = (spec_speedup(H100_8B, 256, c, 0.7, 4, draft_cost=0.05) for c in (256, 1024))
print(f"   H100 bf16, batch 256: 256-token context {short_ctx:4.2f}x, 1,024-token context {long_ctx:4.2f}x "
      f"(KV reads keep the step memory-bound); 2,048 would not fit ({slots(H100_8B):,} slots)")
assert long_ctx > short_ctx
print("✅ speculation is a latency tool while decode is memory-bound; once the verify step is compute-bound it costs throughput")

# %% [markdown]
# ## Exercise 5.4 — what each quantization format buys, prefill versus decode
#
# For Llama-3.1-8B on one L4, compare bf16, FP8 (W8A8: FP8 weights *and* FP8 tensor cores) and AWQ
# (4-bit weights, 16-bit compute). Write two functions:
#
# - `decode_ms(weight_bytes, bw)`. One batch-1 decode step is a weight read. Ignore the overhead and the KV.
# - `prefill_ms(active_params, tokens, tflops)`. Prefill is $2 \times \text{params} \times \text{tokens}$
#   FLOPs at `tflops` effective TFLOP/s.
#
# The check builds the three cases from `sizing.weight_bytes` and the L4 datasheet (50% of peak FLOP/s,
# 80% of bandwidth). AWQ computes in 16-bit, thus it gets the bf16 FLOP/s. The dequantization of the
# weights adds a cost on top, and this cost depends on the kernel. Marlin-style kernels hide most of it at a
# large batch. The check leaves this cost out, as an assumption to examine on your GPU (verify), not as a
# result.

# %% exercise
def decode_ms(weight_bytes: float, bw_bytes_per_s: float) -> float:
    ### BEGIN SOLUTION
    return weight_bytes / bw_bytes_per_s * 1e3
    ### END SOLUTION

def prefill_ms(active_params: float, tokens: int, tflops: float) -> float:
    ### BEGIN SOLUTION
    return 2 * active_params * tokens / (tflops * 1e12) * 1e3
    ### END SOLUTION

# %% check
m8, L4 = sizing.load_config("llama-3.1-8b-instruct"), sizing.GPUS["L4"]
P8 = sizing.param_count(m8).active
bw = L4.mem_bw_gbs * 1e9 * 0.8
cases = {"bf16": (sizing.weight_bytes(m8), L4.bf16_tflops * 0.5),
         "fp8 (W8A8)": (sizing.weight_bytes(m8, quantization="fp8"), L4.fp8_tflops * 0.5),
         "awq (W4A16)": (sizing.weight_bytes(m8, quantization="awq"), L4.bf16_tflops * 0.5)}
res = {k: (decode_ms(w, bw), prefill_ms(P8, 2048, tf)) for k, (w, tf) in cases.items()}
for k, (d, pf) in res.items():
    print(f"   {k:12s} weights {cases[k][0] / 1e9:5.2f} GB  decode step {d:5.1f} ms  prefill(2,048) {pf:6.0f} ms")
assert math.isclose(res["bf16"][0], sizing.weight_bytes(m8) / bw * 1e3)
assert 0.5 < res["fp8 (W8A8)"][0] / res["bf16"][0] < 0.6 and math.isclose(res["fp8 (W8A8)"][1] / res["bf16"][1], 0.5, rel_tol=0.01)
assert res["awq (W4A16)"][0] / res["bf16"][0] < 0.4 and res["awq (W4A16)"][1] >= res["bf16"][1]
print("✅ INT4 weight-only wins decode (bytes) but not prefill (still 16-bit math); FP8 W8A8 halves both on an L4")
print(f"   AWQ decode is {res['bf16'][0] / res['awq (W4A16)'][0]:.1f}x faster, not 16/4 = 4x: the embeddings and lm_head "
      f"({sizing.param_count(m8).embedding / 1e9:.2f} B params) stay 16-bit in real AWQ/GPTQ/FP8 checkpoints")

# %% [markdown]
# ## On a real GPU (T1)
#
# ```bash
# vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384   # L4/H100: FP8 units
# vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ --max-model-len 8192                                               # a pre-quantized AWQ checkpoint
# vllm serve Qwen/Qwen2.5-1.5B-Instruct --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'
# ```
#
# A T4 (compute capability 7.5) has no bfloat16 units and no FP8 units. On a T4:
#
# - Serve 16-bit models with `--dtype half`.
# - For memory, select AWQ/GPTQ checkpoints first.
# - Expect that FP8 options are not available there, or that they are weight-only (verify for your vLLM
#   version).
#
# Attention on a T4 runs on the Triton backend of vLLM. v0.30.0 selects this backend automatically (its
# FlashAttention backend must have sm_80+). No flag is necessary.
#
# The quantized sizes in this notebook keep the embeddings and `lm_head` in 16-bit. Real checkpoints do
# the same (5.73 GB for AWQ Llama-3.1-8B, 9.08 GB for FP8). Thus AWQ decode is ~2.8x faster than bf16. It is not
# the ~3.9x that the bytes of the linear layers alone suggest. The table in PRIMER §8 uses the same
# convention (5.7 GB, ~3x with the overhead and the KV read of its step model). Use the real size of the
# checkpoint, not 4 bits × params.
#
# N-gram speculation must have text that repeats itself (code edits, extraction, RAG answers that quote the
# context) to get to a useful acceptance rate. Measure the acceptance rate with exercise 5.2 against your
# own traffic before you turn speculation on.

# %%
if env.server_url():
    live = M.snapshot(M.scrape(env.server_url(), headers=env.auth_headers()))
    print("your server:", "spec decoding counters present" if live.spec_acceptance_rate is not None
          else "no spec-decode counters (speculation off)")
else:
    print("T0: no SERVELAB_URL set; the numbers above are simulated or computed from datasheets.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Decode is a problem of weight reads, thus we have two levers. Speculative decoding makes
# several tokens per weight read. With a draft acceptance of 0.7 and four drafts, a step gives 2.8 tokens.
# The rejection sampler keeps the output distribution identical to the distribution of the target model.
#
# "Speculation is a latency tool for small batches. At batch 256 with short contexts, the verify step is
# compute-bound, and speculation makes us slower. Long contexts keep the step memory-bound for a longer
# time. Because speculation helps at small batches, we turn speculation on for interactive traffic, and we
# monitor the mean acceptance length on the vLLM counters.
#
# "Quantization decreases the bytes. For an 8B model on an L4, AWQ INT4 makes batch-1 decode ~2.8x faster,
# but it does not make prefill faster. FP8 W8A8 decreases the decode time to almost half, decreases the
# prefill time to half, and releases memory for KV. FP8 KV doubles the tokens that we can hold. Each of
# these goes through an accuracy gate on our evals before we release it."
#
# **Drill 1.** *Acceptance rate is 40%. Does speculation work?* Look at the mean acceptance length instead
# ($1 + \text{accepted}/\text{drafts}$). 40% per draft token with $k = 4$ can still mean ~2.6 tokens per
# step. Judge by the measured TPOT at your batch size, not by the rate.
#
# **Drill 2.** *Does speculative decoding change outputs?* Not with exact rejection sampling. Accept with
# the probability $\min(1, p/q)$. On a rejection, sample again from the normalized residual
# $\max(0, p - q)$. Then the emitted tokens have the target distribution.
#
# **Drill 3.** *AWQ did nothing for our long-prompt TTFT (on one GPU it even got worse). Why?* Prefill is
# compute-bound, and W4A16 still multiplies in 16-bit. Thus there is no decrease in FLOPs.
#
# The dequantization on top costs what the kernel makes it cost. The cost is small with Marlin-style
# kernels at a large batch, and it is visible with other kernels. The gain is in decode and memory. Measure
# the TTFT for each kernel.
