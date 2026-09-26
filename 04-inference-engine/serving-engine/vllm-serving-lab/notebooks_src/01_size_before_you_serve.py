# %% [markdown]
# # 01 · Size before you serve: weights, KV blocks, and the number vLLM will print
#
# **Tier:** T0 — pure arithmetic from `config.json` and a GPU datasheet; runs anywhere. Optional T1:
# start a real `vllm serve` (see `deploy/any-gpu/`) and compare its startup log with your prediction.
#
# ## The one-minute version
#
# A GPU running vLLM holds four things: **weights**, the **activation peak** of a profiling
# forward pass, **CUDA graphs and other buffers**, and **KV blocks**. vLLM takes
# `gpu_memory_utilization × total memory`, subtracts the first three, and turns *all* the rest into
# fixed-size blocks (16 tokens each by default). From the model config alone you can predict:
#
# ```
# KV bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes      (per GPU: kv_heads / TP)
# num_blocks         = floor(KV budget / (block_size × KV bytes per token))
# max concurrency    = num_blocks / ceil(max_model_len / block_size)   <- "Maximum concurrency ... Nx"
# ```
#
# After this notebook you can say, before renting anything, whether a model fits a GPU, how many
# requests of a given length it can hold at once, and which knob — `max_model_len`,
# `kv_cache_dtype`, `quantization`, `tensor_parallel_size`, `gpu_memory_utilization` — buys what.
# Concepts: PRIMER §4 "KV cache management revisited" ([`PRIMER.md`](../../PRIMER.md)); the same
# per-token formula in [`00-foundations/gpu-capacity-planning`](../../../../00-foundations/gpu-capacity-planning/PRIMER.md)
# (`capacity.kv_per_token_kb`; a test pins the two together).

# %%
import json, math
from servelab import sizing
from servelab.sizing import GiB, load_config, param_count, size

m = load_config("qwen2.5-0.5b-instruct")          # key fields of the real config.json, bundled
print({k: getattr(m, k) for k in ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim",
                                  "vocab_size", "max_position_embeddings", "tie_word_embeddings")})
p = param_count(m)
print(f"parameters: {p.total:,} total, {p.embedding:,} in embeddings ({p.embedding / p.total:.0%})")
print("bundled configs:", ", ".join(sizing.sample_configs()))

# %% [markdown]
# ## Worked example: the smallest useful chat model on a free T4
#
# `size()` does the whole calculation and prints it in the order vLLM does it. The overhead lines
# are *estimates* (you calibrate them from a real log in exercise 1.5); the block arithmetic is exact.

# %%
t4 = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096, typical_len=1024)
print(t4.summary())

# %% [markdown]
# Roughly 12 GiB of KV for a 1 GB model: a small model on a 16 GB card is *all cache*. The
# "Maximum concurrency" is a worst case — every request at `max_model_len`. Blocks are allocated as
# tokens arrive, so at 1,024-token requests the engine can hold about four times more.
#
# ## Worked example: an 8B model on one L4 (24 GB) — the startup error everyone meets once

# %%
l4 = size("llama-3.1-8b-instruct", "L4")          # default max_model_len = max_position_embeddings = 131072
print(l4.summary())

# %% [markdown]
# vLLM refuses to start when one request of `max_model_len` tokens cannot fit: the model's
# advertised 128K context needs 16 GiB of KV on its own. The fixes are exactly the knobs of this
# notebook. (vLLM v0.30.0 also accepts `--max-model-len -1`, which auto-fits the largest length
# the KV cache holds — checked against its `vllm/config/model.py`.)
#
# ## Worked example: the headline number — 2K-token sessions of an 8B model on one L4
#
# The figure a design review asks for: how many ~2K-token chat sessions does Llama-3.1-8B in bf16
# hold on one L4? Every input is stated, because the answer moves by one or two sessions with them.

# %%
h = size("llama-3.1-8b-instruct", "L4", max_model_len=2048, typical_len=2000)
print(h.summary())
over = sum(h.overhead_bytes.values()) / GiB
print(f"\nAssumptions: {sizing.GPUS['L4'].memory_gib} GiB visible (nvidia-smi's L4 total), gpu_memory_utilization "
      f"{h.gpu_memory_utilization} (vLLM v0.30.0 default), weights {h.weights_bytes / GiB:.2f} GiB (8.03 B params x 2 B), "
      f"~{over:.2f} GiB estimated overhead (activation peak + CUDA graphs + non-torch)")
print(f"-> {h.num_blocks:,} blocks: {h.max_concurrency:.1f} sessions of 2,048 tokens, "
      f"{h.concurrency_at_typical_len:.1f} of 2,000")
cuda_view = size("llama-3.1-8b-instruct", gpu_memory_bytes=int((22.49 - 0.4) * GiB), max_model_len=2048)
print(f"if CUDA sees 0.4 GiB less than nvidia-smi: {cuda_view.num_blocks:,} blocks, {cuda_view.max_concurrency:.1f} sessions")
primer = size("llama-3.1-8b-instruct", gpu_memory_bytes=int(24e9), gpu_memory_utilization=0.9,
              overhead_bytes={"reserve": int(1e9)}, max_model_len=2048, typical_len=2000)
print(f"PRIMER §4's inputs (24e9 B x 0.9 - 1e9 B reserve): {primer.num_blocks:,} blocks, "
      f"{primer.concurrency_at_typical_len:.1f} sessions of 2,000")

# %% [markdown]
# So: **about 18 concurrent 2K-token sessions** (2,363 blocks; 18.5 at 2,048 tokens, 18.9 at
# 2,000), assuming the L4's 22.49 GiB are all visible to CUDA, vLLM v0.30.0's default
# `gpu_memory_utilization` of 0.92 and ~1.1 GiB of estimated overhead. The honest range is 17-19:
# CUDA usually reports a few hundred MiB less than nvidia-smi (verify on your card), which alone
# costs a session. PRIMER §4 sets both input sets side by side: its simulated numbers use the
# core's simpler inputs — 0.9 of 24 GB minus a flat 1 GB reserve (2,164 blocks, **17** sessions) —
# with the same formulas. Neither is a measurement: the startup log's `Available KV cache memory`
# is (exercise 1.5).
#
# ## Exercise 1.1 — KV bytes per token, from a raw `config.json`
#
# Write `kv_bytes_per_token(cfg, kv_bytes=2, tp=1)` for a plain dict as read from `config.json`.
# Two traps real configs set for you: `head_dim` may be given explicitly and differ from
# `hidden_size / num_attention_heads` (Qwen3), and `num_key_value_heads` (GQA) — not
# `num_attention_heads` — is what the cache stores. With tensor parallelism each GPU holds
# `kv_heads / tp` heads, but never fewer than one (heads are replicated when `tp > kv_heads`).

# %% exercise
def kv_bytes_per_token(cfg: dict, kv_bytes: float = 2, tp: int = 1) -> int:
    ### BEGIN SOLUTION
    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    kv_heads = cfg.get("num_key_value_heads") or heads
    return int(2 * cfg["num_hidden_layers"] * max(1, kv_heads // tp) * head_dim * kv_bytes)
    ### END SOLUTION

# %% check
raw = {n: json.loads((sizing.CONFIG_DIR / f"{n}.json").read_text()) for n in sizing.sample_configs()}
assert kv_bytes_per_token(raw["qwen2.5-0.5b-instruct"]) == 12_288
assert kv_bytes_per_token(raw["qwen3-0.6b"]) == 114_688, "Qwen3-0.6B: use the explicit head_dim (128), not 1024/16"
assert kv_bytes_per_token(raw["llama-3.1-8b-instruct"], tp=2) == 65_536
assert kv_bytes_per_token(raw["qwen2.5-0.5b-instruct"], tp=4) == 6_144, "2 KV heads on 4 GPUs: one replicated head each"
for name, cfg in raw.items():
    assert kv_bytes_per_token(cfg, kv_bytes=1) == sizing.kv_bytes_per_token(load_config(name), kv_cache_dtype="fp8"), name
print("✅ kv_bytes_per_token agrees with the library on all", len(raw), "bundled configs")

# %% [markdown]
# ## Exercise 1.2 — from a KV budget to the line vLLM logs
#
# Given the bytes left for KV, return `(num_blocks, max_concurrency, kv_cache_tokens)` exactly as
# vLLM computes them: whole blocks only; a request of `max_model_len` tokens needs
# `ceil(max_model_len / block_size)` blocks (a partial block still costs a whole one); the logged
# token capacity is `int(max_concurrency × max_model_len)`.

# %% exercise
def kv_capacity(kv_budget_bytes: int, kv_per_token: int, max_model_len: int, block_size: int = 16):
    ### BEGIN SOLUTION
    num_blocks = kv_budget_bytes // (block_size * kv_per_token)
    conc = num_blocks / math.ceil(max_model_len / block_size)
    return num_blocks, conc, int(conc * max_model_len)
    ### END SOLUTION

# %% check
assert kv_capacity(1 * GiB, 12_288, 4096) == (5461, 5461 / 256, 87_376)
for budget, model, L in [(3 * GiB, "llama-3.2-1b-instruct", 8000), (7 * GiB, "qwen2.5-7b-instruct", 32768)]:
    r = size(model, "L4", max_model_len=L, kv_budget_bytes=budget)
    assert kv_capacity(budget, r.kv_bytes_per_token, L) == (r.num_blocks, r.max_concurrency, r.kv_cache_tokens), model
print("✅ kv_capacity reproduces vLLM's 'GPU KV cache size / Maximum concurrency' arithmetic")

# %% [markdown]
# ## Exercise 1.3 — pick `max_model_len` for an 8B model on one L4
#
# You serve `llama-3.1-8b-instruct` in bf16 on one L4 with the default utilization, and the design
# requirement is: vLLM must start, and **at least 4 requests of the maximum length** must fit at
# once. Return from `choose_max_model_len()` the largest multiple of 1,024 that satisfies it.
# Invert the block formula yourself: take `num_blocks` from one `size()` report (the KV budget
# does not depend on `max_model_len`, so any length will do), then find the largest `L` with
# `4 × ceil(L / 16) <= num_blocks`. The check compares with `sizing.max_model_len_for`. Then read
# the answer as a product decision: is that context length enough for your workload, or is it
# time for FP8, a bigger GPU, or TP=2?

# %% exercise
def choose_max_model_len() -> int:
    ### BEGIN SOLUTION
    r = size("llama-3.1-8b-instruct", "L4", max_model_len=1024)
    blocks_per_request = r.num_blocks // 4              # 4 x ceil(L/16) <= num_blocks
    return (blocks_per_request * r.block_size // 1024) * 1024
    ### END SOLUTION

chosen_len = choose_max_model_len()
print("chosen max_model_len:", chosen_len)

# %% check
ok = size("llama-3.1-8b-instruct", "L4", max_model_len=chosen_len)
too_far = size("llama-3.1-8b-instruct", "L4", max_model_len=chosen_len + 1024)
assert chosen_len % 1024 == 0 and ok.fits and ok.max_concurrency >= 4, ok.summary()
assert too_far.max_concurrency < 4, "a longer max_model_len would still hold 4 — go higher"
assert chosen_len == sizing.max_model_len_for("llama-3.1-8b-instruct", "L4", concurrency=4) // 1024 * 1024
print(f"✅ max_model_len={chosen_len}: {ok.max_concurrency:.2f} worst-case requests fit ({ok.num_blocks:,} blocks)")

# %% [markdown]
# ## Exercise 1.4 — what FP8 buys, predicted in two pieces
#
# Switching the 8B model to FP8 weights (`--quantization fp8`) *and* FP8 KV (`--kv-cache-dtype fp8`)
# helps concurrency twice: the weights shrink, which grows the KV budget, and each token's KV
# halves. Write `predicted_gain(r_bf16, weights_saved_bytes)` that predicts the ratio of max
# concurrency (fp8/fp8 over bf16/bf16) from the bf16 report and the bytes the weights shrink by —
# without calling `size()` for the FP8 case. The check compares with the full calculation.

# %% exercise
def predicted_gain(r_bf16, weights_saved_bytes: float) -> float:
    ### BEGIN SOLUTION
    tokens_bf16 = r_bf16.kv_budget_bytes / r_bf16.kv_bytes_per_token
    tokens_fp8 = (r_bf16.kv_budget_bytes + weights_saved_bytes) / (r_bf16.kv_bytes_per_token / 2)
    return tokens_fp8 / tokens_bf16
    ### END SOLUTION

# %% check
m8 = load_config("llama-3.1-8b-instruct")
r16 = size(m8, "L4", max_model_len=16384)
r8 = size(m8, "L4", max_model_len=16384, quantization="fp8", kv_cache_dtype="fp8")
saved = sizing.weight_bytes(m8) - sizing.weight_bytes(m8, quantization="fp8")
guess, actual = predicted_gain(r16, saved), r8.max_concurrency / r16.max_concurrency
assert abs(guess / actual - 1) < 0.02, (guess, actual)
print(f"✅ FP8 weights + FP8 KV: {actual:.1f}x the concurrency ({r16.max_concurrency:.2f}x -> {r8.max_concurrency:.2f}x); "
      f"KV halving alone would be 2x")

# %% [markdown]
# ## Exercise 1.5 — calibrate against the log, then re-plan without restarting
#
# The prediction's overhead terms are guesses; the startup log is the truth. Below is an
# **illustrative** excerpt in `vllm serve`'s log format (sample output, not a measurement; with a
# real server, point `LOG_TEXT` at your `vllm.log`). Write `replan(log, max_model_len, kv_per_token)`
# that takes the parsed log and returns the max concurrency vLLM *would* report at a different
# `max_model_len` — KV budget and blocks do not depend on `max_model_len`, so no restart is needed.

# %%
import os, pathlib
from servelab import SAMPLES_DIR
log_path = pathlib.Path(os.environ.get("VLLM_LOG", "vllm.log"))
if log_path.exists():                                   # T1: your own server's log
    LOG_TEXT, source = log_path.read_text(), f"measured ({log_path})"
else:
    LOG_TEXT = (SAMPLES_DIR / "vllm_startup_log.txt").read_text()
    source = "illustrative sample in vLLM's log format (not a measurement)"
log = sizing.parse_startup_log(LOG_TEXT)
print(source)
print(log)
# The utilization the logged server ran with: the log states it (v0.30.0 prints it with the CUDA-graph
# line), else GPU_MEM_UTIL; calibrating a 0.85 server against a 0.92 prediction would inflate the overhead.
util = float(os.environ["GPU_MEM_UTIL"]) if "GPU_MEM_UTIL" in os.environ else None
cal = sizing.calibrate(size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096), log,
                       gpu_memory_utilization=util)
print({k: round(v, 3) for k, v in cal.items()})

# %% exercise
def replan(log: dict, max_model_len: int, kv_per_token: int, block_size: int = 16) -> float:
    ### BEGIN SOLUTION
    blocks = int(round(log["available_kv_gib"] * GiB)) // (block_size * kv_per_token)
    return blocks / math.ceil(max_model_len / block_size)
    ### END SOLUTION

# %% check
same = replan(log, log.get("max_model_len", 4096), 12_288)
if source.startswith("illustrative"):
    assert abs(same - log["max_concurrency"]) < 0.01, (same, log)      # reproduces the logged line
else:
    print(f"your log says {log.get('max_concurrency')}x; replan says {same:.2f}x (12,288 B/token assumes Qwen2.5-0.5B)")
at_8k = replan(log, 8192, 12_288)
ref = size("qwen2.5-0.5b-instruct", "T4", max_model_len=8192,
           kv_budget_bytes=int(round(log["available_kv_gib"] * GiB))).max_concurrency
assert abs(at_8k - ref) < 1e-9 and abs(at_8k - same / 2) < 0.02 * same
print(f"✅ at max_model_len=8192 vLLM would report {at_8k:.2f}x (half of {same:.2f}x: same blocks, twice the length)")

# %% [markdown]
# ## T1: compare with a real engine
#
# On a GPU box (or Colab T4), start vLLM with its log captured and re-run the cells above with
# `VLLM_LOG` pointing at it; the calibrated overhead then makes every later prediction for that
# GPU/model exact:
#
# ```bash
# vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.92 > vllm.log 2>&1 &
# python -m servelab size --model qwen2.5-0.5b-instruct --gpu T4 --dtype half --max-model-len 4096
# ```
#
# If you start it with another utilization (the Colab recipe uses 0.85), the calibration reads it
# from the log line `The current --gpu-memory-utilization=...`, or set `GPU_MEM_UTIL`. On a T4
# (compute capability 7.5) the log also names the attention backend: vLLM v0.30.0 selects
# `TRITON_ATTN` by itself, since its FlashAttention backend needs sm_80+ — no flag or environment
# variable is needed (`VLLM_ATTENTION_BACKEND` no longer exists; see `deploy/any-gpu/`).

# %%
from servelab import env
print(env.describe())
if env.has_gpu():
    print("GPU found: run the command above (or deploy/any-gpu/serve.sh) and set VLLM_LOG=vllm.log")
else:
    print("No GPU here: the prediction above is the T0 result; the same cell reads a real log on a GPU box.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Before choosing hardware I size it from `config.json`. Per token the KV cache
# costs 2 × layers × KV heads × head dim × bytes — 12 KB for Qwen2.5-0.5B, 128 KB for Llama-3.1-8B.
# vLLM reserves `gpu_memory_utilization` of the card, pays weights and overheads, and cuts the rest
# into 16-token blocks. So an 8B model in bf16 on an L4 has about 4.6 GiB of KV: ~37K tokens —
# about 18 concurrent 2K-token sessions at vLLM's default 0.92 utilization, and the default 128K
# context does not even start; at 8K context only four requests fit in the worst case. FP8
# weights plus FP8 KV give ~4.8x the concurrency: the 7 GB the weights shed grows the KV budget
# from 4.6 to 11.1 GiB, and each token's KV halves.
# I then check the prediction against the startup log's 'Available KV cache memory' and 'Maximum
# concurrency' lines and plan other context lengths from the calibrated number."
#
# **Drill 1.** *vLLM says "Maximum concurrency for 32,768 tokens per request: 1.5x". Can it serve
# 10 users?* — Yes, if their requests are short: the figure is a worst case at `max_model_len`.
# Blocks are allocated as tokens arrive; with 3K-token conversations roughly ten times more fit.
# What caps concurrency then is `max_num_seqs` and the SLO, and preemption if lengths grow.
#
# **Drill 2.** *Why is `head_dim` a trap?* — Some configs (Qwen3) set it explicitly and it differs
# from `hidden_size / num_attention_heads`; deriving it under-counts Qwen3-0.6B's KV by 2x.
#
# **Drill 3.** *Tensor parallelism 2 halves the weights per GPU. What does it do to KV per token per
# GPU?* — Halves it too (KV heads are split), unless there are fewer KV heads than GPUs, in which
# case heads are replicated and the saving stops.
