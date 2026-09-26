# %% [markdown]
# # 05 · Speculative decoding and quantization in vLLM: what each buys, read from the engine's counters
#
# **Tier:** T0 — the arithmetic, plus the fake vLLM emulating speculation (acceptance is an explicit
# *assumption* there; results **simulated**). T1: the vLLM flags are printed for each experiment;
# against a real server (`SERVELAB_URL`) the acceptance cells read vLLM's own spec-decode counters.
#
# ## The one-minute version
#
# Both techniques attack the same fact: **decode is bound by memory bandwidth** — every step
# streams all the weights to produce one token per sequence.
#
# * **Speculative decoding** proposes `k` tokens cheaply (n-gram lookup in the prompt, a small
#   draft model, EAGLE/MTP heads) and verifies all of them in *one* target forward pass. With
#   per-token acceptance α, a step yields `(1 − α^(k+1)) / (1 − α)` tokens on average, and the
#   rejection-sampling rule keeps the output distribution *exactly* the target model's. It pays
#   while the step is memory-bound (small batches) and fades as batches make it compute-bound.
# * **Quantization** shrinks bytes: weight-only INT4 (AWQ/GPTQ) or FP8 weights cut the bytes each
#   decode step reads (faster decode, more room for KV); FP8 *W8A8* also doubles tensor-core
#   FLOP/s on GPUs that have FP8 units (faster prefill); FP8 KV halves the KV cache. Accuracy must
#   be measured on your own evals.
#
# Concepts: PRIMER §7 "Speculative decoding" and §8 "Quantization" ([`PRIMER.md`](../../PRIMER.md));
# the exact rejection sampler is implemented from scratch in this topic's `mini-engine-core`.

# %%
import math
from servelab import env, metrics as M, sizing
from servelab.bench import Lengths, random_requests, run_closed_loop
from servelab.fake_engine import EngineConfig, FakeEngine, build_profile, simulate, tiny_profile
from servelab.tune import FakeBackend, sweep, to_cli_flags, trials_table

print(env.describe())
print("expected tokens per verify step, (1 - a^(k+1)) / (1 - a):")
print("alpha " + "".join(f"  k={k:<4}" for k in (1, 2, 4, 8)))
for a in (0.5, 0.7, 0.9):
    print(f"{a:5.1f} " + "".join(f"{(1 - a ** (k + 1)) / (1 - a):8.2f}" for k in (1, 2, 4, 8)))

# %% [markdown]
# Diminishing returns in `k`: at α = 0.7 the 5th to 8th draft tokens add only 0.4 tokens per step,
# while the verifier must process all of them. vLLM flags for the three families (verify names
# against your vLLM version):

# %%
for spec in ({"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
             {"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4},
             {"method": "eagle3", "model": "<an EAGLE-3 head trained for the target>", "num_speculative_tokens": 3}):
    print("vllm serve <target>", " ".join(to_cli_flags({"speculative_config": spec})))

# %% [markdown]
# ## Exercise 5.1 — expected tokens per verification step
#
# Write `expected_tokens(alpha, k)`: the mean number of tokens one verify step emits when each draft
# token is accepted independently with probability `alpha` and the first rejection stops the run
# (plus the one token the target always contributes). Handle `alpha = 1`.

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
# `FakeBackend` turns vLLM's `speculative_config` into the emulator's knobs; the acceptance rate is
# an assumption you pass (0.7 here) — on a real engine it is a property of your traffic and the
# draft method, which is why you read it from `/metrics` rather than assume it.

# %%
work = lambda: random_requests(12, Lengths.fixed(256), Lengths.fixed(96), seed=9)  # noqa: E731
SPEC = {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "acceptance": 0.7}}
results = {}
for users in (1, 8):
    results[users] = sweep(FakeBackend("t4-qwen2.5-0.5b"), [{}, SPEC], work, concurrency=users, warmup=0)
    print(f"[SIMULATED] {users} concurrent user(s)")
    print(trials_table(results[users]))
spec_trial = results[1][1]
print(spec_trial.snapshot.table())

# %% [markdown]
# ## Exercise 5.2 — acceptance, the way vLLM's dashboards compute it
#
# vLLM exports three counters: `vllm:spec_decode_num_drafts` (verify steps with drafts),
# `..._num_draft_tokens` and `..._num_accepted_tokens`. Its documented PromQL is
# `acceptance rate = rate(accepted) / rate(draft tokens)` and
# `mean acceptance length = 1 + rate(accepted) / rate(drafts)` (the 1 counts the target's own
# token). Write `acceptance(before, after)` returning `(rate, mean_length)` from two scrapes.

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
# Note the difference between the two numbers: the **acceptance rate** is per draft token (0.44
# here even though α = 0.7, because later positions are reached less often), the **mean acceptance
# length** is what sets the speedup.
#
# ## Exercise 5.3 — where speculation pays: batch size
#
# A verify step processes `batch × (k + 1)` tokens and reads the weights once; drafting adds
# `k × draft_cost × (weight read time)`. Write `spec_speedup(p, batch, context, alpha, k,
# draft_cost)`: baseline time per token (one decode step per token) divided by speculative time
# per token (one verify step + drafting, divided by the expected tokens per step). Use the roofline
# as in notebook 03: `overhead + max(2·P·tokens / FLOP/s, (weights + batch·context·kv) / bandwidth)`.
# Predict first: at which batch sizes does speculation stop paying, and why does context length matter?

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
sp = {b: spec_speedup(L4_8B, b, 256, 0.7, 4, draft_cost=0.05) for b in (1, 8, 32, 128, 256, 512)}
for b, v in sp.items():
    print(f"   batch {b:>3}, 256-token context: speedup {v:4.2f}x")
assert sp[1] > 2.0 and sp[128] < sp[1] and sp[256] < 1.0 and sp[512] < sp[256]
long_ctx = spec_speedup(L4_8B, 256, 2048, 0.7, 4, draft_cost=0.05)
print(f"   batch 256, 2,048-token context: speedup {long_ctx:4.2f}x (KV reads keep the step memory-bound)")
assert long_ctx > sp[256]
print("✅ speculation is a latency tool while decode is memory-bound; once the verify step is compute-bound it costs throughput")

# %% [markdown]
# ## Exercise 5.4 — what each quantization format buys, prefill versus decode
#
# For Llama-3.1-8B on one L4 compare bf16, FP8 (W8A8: FP8 weights *and* FP8 tensor cores) and AWQ
# (4-bit weights, 16-bit compute). Write `decode_ms(weight_bytes, bw)` — one batch-1 decode step
# is a weight read (ignore overhead and KV) — and `prefill_ms(active_params, tokens, tflops)` —
# `2 × params × tokens` FLOPs at `tflops` effective TFLOP/s. The check builds the three cases from
# `sizing.weight_bytes` and the L4 datasheet (50% of peak FLOP/s, 80% of bandwidth).

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
         "awq (W4A16)": (sizing.weight_bytes(m8, quantization="awq"), L4.bf16_tflops * 0.5 * 0.9)}
res = {k: (decode_ms(w, bw), prefill_ms(P8, 2048, tf)) for k, (w, tf) in cases.items()}
for k, (d, pf) in res.items():
    print(f"   {k:12s} weights {cases[k][0] / 1e9:5.2f} GB  decode step {d:5.1f} ms  prefill(2,048) {pf:6.0f} ms")
assert math.isclose(res["bf16"][0], sizing.weight_bytes(m8) / bw * 1e3)
assert 0.5 < res["fp8 (W8A8)"][0] / res["bf16"][0] < 0.6 and math.isclose(res["fp8 (W8A8)"][1] / res["bf16"][1], 0.5, rel_tol=0.01)
assert res["awq (W4A16)"][0] / res["bf16"][0] < 0.4 and res["awq (W4A16)"][1] > res["bf16"][1]
print("✅ INT4 weight-only wins decode (bytes) but not prefill (still 16-bit math); FP8 W8A8 halves both on an L4")

# %% [markdown]
# ## On a real GPU (T1)
#
# ```bash
# vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384   # L4/H100: FP8 units
# vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ --max-model-len 8192                                               # a pre-quantized AWQ checkpoint
# vllm serve Qwen/Qwen2.5-1.5B-Instruct --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'
# ```
#
# A T4 (compute capability 7.5) has no bfloat16 and no FP8 units: serve 16-bit models with
# `--dtype half`, prefer AWQ/GPTQ checkpoints for memory, and expect FP8 options to be unavailable
# or weight-only there (verify for your vLLM version). N-gram speculation needs repetitive text
# (code edits, extraction, RAG answers that quote the context) to reach a useful acceptance rate;
# measure it with exercise 5.2 against your own traffic before turning it on.

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
# **Two minutes:** "Decode is a weight-streaming problem, so we have two levers. Speculative
# decoding produces several tokens per weight read: with a draft acceptance of 0.7 and four drafts
# a step yields 2.8 tokens, and the rejection sampler keeps the output distribution identical to
# the target model's. It is a latency tool for small batches — at batch 256 the verify step is
# compute-bound and speculation slows us down — so we enable it for interactive traffic and watch
# the mean acceptance length on vLLM's counters. Quantization cuts the bytes: for an 8B model on an
# L4, AWQ INT4 makes batch-1 decode ~2.8x faster but prefill slightly slower; FP8 W8A8 almost halves
# decode, halves prefill and frees room for KV, and FP8 KV doubles the tokens we can hold. Each gets
# an accuracy gate on our evals before it ships."
#
# **Drill 1.** *Acceptance rate is 40%. Is speculation working?* — Look at the mean acceptance
# length instead (1 + accepted/drafts): 40% per draft token with k = 4 can still mean ~2.6 tokens
# per step; judge by measured TPOT at your batch size, not by the rate.
#
# **Drill 2.** *Does speculative decoding change outputs?* — Not with exact rejection sampling:
# accept with probability min(1, p/q), resample from the normalized residual max(0, p − q) on
# rejection, and the emitted tokens follow the target distribution.
#
# **Drill 3.** *Why did AWQ make our long-prompt TTFT worse?* — Prefill is compute-bound and AWQ
# still computes in 16-bit, with dequantization on top; the win is in decode and memory.
