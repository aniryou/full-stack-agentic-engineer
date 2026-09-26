# %% [markdown]
# # 03 · Knobs and trade-offs: batch size, token budget, KV blocks and the latency-throughput curve
#
# **Tier:** T0 — every experiment restarts a fake vLLM with different flags (seconds each; all
# results **simulated**). T1: with a GPU, vLLM installed and `SERVELAB_START_VLLM=1`, the same
# sweeps restart a real `vllm serve` per config (a minute or two each) — same code, measured numbers.
#
# ## The one-minute version
#
# A decode step reads all the weights once for the whole batch, so batching is nearly free until
# the step becomes compute-bound or the KV reads catch up: **throughput rises with batch, per-token
# latency rises slowly**. Three flags shape the curve:
#
# | Flag | Raises | Costs |
# |---|---|---|
# | `--max-num-seqs` (batch cap) | throughput, fewer queued requests | a slower step for everyone (TPOT) |
# | `--max-num-batched-tokens` (step token budget) | prefill speed of long prompts (TTFT) | decode stalls when a big prefill chunk shares the step (ITL tail) |
# | KV blocks (`--gpu-memory-utilization`, `--max-model-len`, `--num-gpu-blocks-override`) | concurrent requests held | when too few: preemption and recompute |
#
# The way to choose is to measure **goodput at your SLO** for each setting on your workload, and to
# find the highest request rate that still meets the SLO. Concepts: PRIMER §2 "Continuous batching",
# §3 "Chunked prefill and prefill/decode interference", §11 "Measuring an engine" ([`PRIMER.md`](../../PRIMER.md)).

# %%
import math, os
from servelab import env
from servelab.bench import SLO, Lengths, curve, mixed_requests, random_requests, run_open_loop
from servelab.fake_engine import profile as engine_profile
from servelab.tune import FakeBackend, VLLMBackend, best, grid, max_rate_under_slo, sweep, to_cli_flags, trials_table

REAL = VLLMBackend.available() and os.environ.get("SERVELAB_START_VLLM") == "1"
def backend(**overrides):
    """A fresh engine per config: real vLLM at T1 (opt-in), the fake one otherwise."""
    if REAL:
        return VLLMBackend("Qwen/Qwen2.5-0.5B-Instruct", base_flags={"dtype": "half", "max_model_len": 4096})
    return FakeBackend("t4-qwen2.5-0.5b", **overrides)
LABEL = "MEASURED" if REAL else "SIMULATED"
print(env.describe(), "|", LABEL)
P = engine_profile("t4-qwen2.5-0.5b")
print(P.describe())

# %% [markdown]
# ## Worked example: why batching is nearly free (the step-time model)
#
# The fake engine's step time is the roofline: `overhead + max(FLOPs / FLOP/s, bytes / bandwidth)`.
# For a decode step, bytes = all weights (once, shared by the batch) + every sequence's KV.

# %%
print(f"{'batch':>6} {'step ms':>8} {'tok/s total':>12} {'tok/s per user':>15}")
for b in (1, 8, 32, 64, 128, 256):
    t = P.decode_step_s(b, 1024)
    print(f"{b:>6} {t * 1e3:>8.2f} {b / t:>12,.0f} {1 / t:>15.1f}")

# %% [markdown]
# From 1 to 64 sequences the step slows by about 40% while total throughput grows more than 40x:
# the weights are read once either way (numbers from the simulated profile above). Past that, each sequence's KV reads start to dominate (and, for bigger
# models or longer prompts, compute). That is the whole economic case for continuous batching.
#
# ## Exercise 3.1 — the decode step, and the largest batch that keeps a TPOT target
#
# Implement `decode_step_s(p, batch, context)` from the profile's fields (`overhead_s`,
# `active_params`, `flops`, `weight_bytes`, `kv_bytes_per_token`, `mem_bw`) with the roofline above,
# then `max_batch_for_tpot(p, tpot_s, context)`: the largest batch (up to 4,096) whose decode step
# stays within `tpot_s`.

# %% exercise
def decode_step_s(p, batch: int, context: int) -> float:
    ### BEGIN SOLUTION
    compute = 2 * p.active_params * batch / p.flops
    memory = (p.weight_bytes + batch * context * p.kv_bytes_per_token) / p.mem_bw
    return p.overhead_s + max(compute, memory)
    ### END SOLUTION

def max_batch_for_tpot(p, tpot_s: float, context: int) -> int:
    ### BEGIN SOLUTION
    best_b = 0
    for b in range(1, 4097):
        if decode_step_s(p, b, context) <= tpot_s:
            best_b = b
        else:
            break
    return best_b
    ### END SOLUTION

# %% check
for b, c in ((1, 100), (16, 1024), (256, 4000)):
    assert math.isclose(decode_step_s(P, b, c), P.decode_step_s(b, c))
b12 = max_batch_for_tpot(P, 0.012, 1024)
assert P.decode_step_s(b12, 1024) <= 0.012 < P.decode_step_s(b12 + 1, 1024)
assert max_batch_for_tpot(P, 0.001, 1024) == 0              # below the fixed overhead nothing fits
print(f"✅ at 1,024 tokens of context a 12 ms TPOT target allows a batch of {b12} on this profile")

# %% [markdown]
# ## Worked example: the latency-throughput curve (open loop, 4 s of arrivals per rate)

# %%
engine = backend()
URL = engine.start({})                 # vLLM defaults (or the fake engine's)
rows = []
for rate in (4, 8, 16, 24, 32, 40):
    r = run_open_loop(URL, random_requests(int(rate * 4), Lengths.fixed(512), Lengths.fixed(64), seed=rate),
                      rate=rate, seed=rate)
    s = r.summary(SLO(ttft_ms=300, tpot_ms=15))
    rows.append((rate, s))
    print(f"[{LABEL}] offered {rate:>3}/s: served {s.request_throughput:5.1f}/s  TTFT p99 {s.ttft.p[99]:7.1f} ms  "
          f"TPOT p50 {s.tpot.median:5.1f} ms  SLO met {s.slo_attainment:4.0%}")
print(curve([r for r, _ in rows], [s.tpot.median for _, s in rows], "req/s", "TPOT p50 (ms)"))
engine.stop()

# %% [markdown]
# Past the knee the batch is large and prefills keep joining it: every user's tokens slow down
# (TPOT), and served throughput stops tracking the offered rate. The flag that decides where each
# request waits is next.
#
# ## Exercise 3.2 — the token budget: long prompts versus everyone else's next token
#
# With chunked prefill, a step processes at most `max_num_batched_tokens` tokens: decodes first,
# then a chunk of a waiting prompt. Write `stall_s(p, budget, decode_batch, context, long_prompt)`:
# the duration of a step holding `decode_batch` decode tokens plus a prefill chunk of
# `min(budget - decode_batch, long_prompt)` tokens — that is the inter-token gap every decoding
# request sees during that step (roofline as in 3.1; the chunk's KV reads are its own tokens).
# Predict how ITL p99 for short requests and TTFT for long prompts move as the budget grows, then
# run the sweep.

# %% exercise
def stall_s(p, budget: int, decode_batch: int, context: int, long_prompt: int) -> float:
    ### BEGIN SOLUTION
    chunk = min(budget - decode_batch, long_prompt)
    compute = 2 * p.active_params * (decode_batch + chunk) / p.flops
    memory = (p.weight_bytes + (decode_batch * context + chunk) * p.kv_bytes_per_token) / p.mem_bw
    return p.overhead_s + max(compute, memory)
    ### END SOLUTION

# %% check
stalls = {b: stall_s(P, b, 4, 300, 3000) for b in (256, 1024, 8192)}
assert stalls[256] < stalls[1024] < stalls[8192]
assert stalls[8192] > 10 * P.decode_step_s(4, 300)            # one 3,000-token chunk: a long hiccup for every decoder
assert stall_s(P, 8192, 4, 300, 3000) == stall_s(P, 100_000, 4, 300, 3000)   # the chunk cannot exceed the prompt
budget_trials = sweep(backend(), grid(max_num_batched_tokens=[256, 1024, 8192]),
                      lambda: mixed_requests(30, 10, short_in=256, long_in=3000, output=64, seed=4), rate=6, warmup=0)
itl99 = {t.config["max_num_batched_tokens"]: t.run.summary(tag="short").itl.p[99] for t in budget_trials}
long_ttft = {t.config["max_num_batched_tokens"]: t.run.summary(tag="long").ttft.median for t in budget_trials}
for b in itl99:
    print(f"[{LABEL}] budget {b:>5}: short-request ITL p99 {itl99[b]:6.1f} ms (model: {stalls[b] * 1e3:5.1f} ms), "
          f"long-prompt TTFT p50 {long_ttft[b]:6.1f} ms")
assert itl99[256] < itl99[8192] and long_ttft[256] > long_ttft[8192]
print("✅ a small budget protects everyone's ITL; a big one gets long prompts to their first token sooner")

# %% [markdown]
# ## Exercise 3.3 — choose `max_num_seqs` by goodput, not by throughput
#
# At 12 req/s we sweep the batch cap. Write `best_config(trials, min_attainment)`: among trials whose
# SLO attainment is at least `min_attainment`, return the one with the highest output throughput
# (tokens/s); `None` if no config meets the SLO. (`trial.summary.slo_attainment`,
# `trial.summary.output_throughput`.)

# %%
SLO_B = SLO(ttft_ms=500, tpot_ms=20)
seq_trials = sweep(backend(), grid(max_num_seqs=[2, 8, 32]),
                   lambda: random_requests(40, Lengths.fixed(256), Lengths.fixed(64), seed=5), rate=12, slo=SLO_B, warmup=0)
print(f"[{LABEL}] SLO {SLO_B}")
print(trials_table(seq_trials))
print("vLLM flags for each:", [" ".join(to_cli_flags(t.config)) for t in seq_trials])

# %% exercise
def best_config(trials: list, min_attainment: float = 0.9):
    ### BEGIN SOLUTION
    ok = [t for t in trials if t.summary.slo_attainment >= min_attainment]
    return max(ok, key=lambda t: t.summary.output_throughput) if ok else None
    ### END SOLUTION

# %% check
for a in (0.5, 0.9, 0.99, 1.01):
    mine, ref = best_config(seq_trials, a), best(seq_trials, a)
    assert (mine is None and ref is None) or mine.config == ref.config
choice = best_config(seq_trials, 0.9)
print("✅ chosen:", choice and choice.config, "— the batch cap of 2 had high per-user speed and failed the SLO by queueing")

# %% [markdown]
# ## Exercise 3.4 — the highest rate that still meets the SLO
#
# Capacity is a *rate at an SLO*. Write `max_rate(measure, lo, hi, target, iters)`: bisection in log
# space (`mid = sqrt(lo * hi)`), keeping `lo` meeting the target and `hi` missing it; return the last
# good rate. Return `nan` if `lo` already misses, `hi` if even `hi` meets it.

# %% exercise
def max_rate(measure, lo: float, hi: float, target: float = 0.9, iters: int = 4) -> float:
    ### BEGIN SOLUTION
    if measure(lo) < target:
        return math.nan
    if measure(hi) >= target:
        return hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if measure(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo
    ### END SOLUTION

# %% check
knee = 11.0
r = max_rate(lambda x: 1.0 if x <= knee else 0.0, 1.0, 64.0, iters=10)
assert math.isclose(r, max_rate_under_slo(lambda x: 1.0 if x <= knee else 0.0, 1.0, 64.0, iters=10)[0])
assert knee / 1.01 <= r <= knee and math.isnan(max_rate(lambda x: 0.0, 1.0, 2.0)) and max_rate(lambda x: 1.0, 1.0, 2.0) == 2.0
small = backend()
SMALL = small.start({"max_num_seqs": 8})
def attainment(rate):
    run = run_open_loop(SMALL, random_requests(30, Lengths.fixed(256), Lengths.fixed(64), seed=int(rate * 10)),
                        rate=rate, seed=1)
    a = run.summary(SLO(ttft_ms=300, tpot_ms=20)).slo_attainment
    print(f"   [{LABEL}] {rate:6.2f} req/s -> SLO met by {a:.0%}")
    return a
cap = max_rate(attainment, 4.0, 32.0, 0.9, iters=3)
small.stop()
assert 4.0 <= cap < 32.0
print(f"✅ with --max-num-seqs 8 this engine sustains about {cap:.1f} req/s at the SLO")

# %% [markdown]
# ## Exercise 3.5 — KV blocks and preemption
#
# When the running requests need more KV blocks than exist, the scheduler **preempts** the most
# recently admitted request: frees its blocks, puts it back in the queue, and recomputes its prompt
# (and the tokens it had generated) later — visible as `vllm:num_preemptions` and a long ITL gap.
# Write `blocks_needed(n_concurrent, prompt_len, max_tokens, block_size)`: the blocks that `n`
# requests need to *finish* together. The check runs 12 simultaneous requests with exactly that
# many blocks (`--num-gpu-blocks-override`, vLLM's flag for this experiment) and with a third of it
# (`--max-model-len 1024`, because vLLM refuses to start unless one full-length request fits).

# %% exercise
def blocks_needed(n_concurrent: int, prompt_len: int, max_tokens: int, block_size: int = 16) -> int:
    ### BEGIN SOLUTION
    return n_concurrent * math.ceil((prompt_len + max_tokens) / block_size)
    ### END SOLUTION

# %% check
need = blocks_needed(12, 256, 128)
assert need == 12 * 24 and blocks_needed(1, 17, 0) == 2
pre = {}
for nb in (need, need // 3):
    (t,) = sweep(backend(), [{"num_gpu_blocks_override": nb, "max_model_len": 1024}],
                 lambda: random_requests(12, Lengths.fixed(256), Lengths.fixed(128), seed=6), rate=math.inf, warmup=0)
    pre[nb] = t.snapshot.preemptions
    print(f"[{LABEL}] {nb:4d} blocks: {pre[nb]:.0f} preemptions, E2E p99 {t.summary.e2el.p[99]:7.0f} ms")
assert pre[need] == 0 and pre[need // 3] > 0
print("✅ enough blocks for everyone's full length: no preemption; a third of it: requests are evicted and recomputed")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Decode is bandwidth-bound: one step reads the weights once for the whole batch,
# so we batch as far as the TPOT target allows — `max-num-seqs` is a latency budget, not a
# capacity number. The step token budget, `max-num-batched-tokens`, decides who waits when a long
# prompt arrives: a big budget gives that prompt a fast TTFT and every decoding user a 100 ms
# hiccup; a small one spreads the prefill thin. We give the engine enough KV blocks for the
# concurrency we admit, because preemption means recomputing whole prompts. We pick each setting by
# sweeping it on our own workload and keeping the one with the best goodput at the SLO, and we state
# capacity as the highest rate that still meets the SLO — measured, then confirmed on the real GPU."
#
# **Drill 1.** *Users complain about random pauses mid-answer; average ITL looks fine.* — Look at
# ITL p99/max and at long prompts in the traffic: large prefill chunks share steps with decodes.
# Lower `max-num-batched-tokens` (or cap `long-prefill-token-threshold`), or separate prefill (layer 05).
#
# **Drill 2.** *Why not set `max-num-seqs` to 1,024 everywhere?* — Past the point where KV reads or
# compute dominate, each extra sequence slows every step; TPOT rises and, with too few KV blocks,
# preemptions start. The right cap is the largest one that keeps TPOT within the SLO.
#
# **Drill 3.** *`vllm:num_preemptions_total` is climbing. Three fixes?* — More KV blocks (higher
# `gpu-memory-utilization`, FP8 KV, a smaller `max-model-len` only if requests are really shorter),
# fewer concurrent sequences (`max-num-seqs`), or more GPUs (TP, or another replica).
