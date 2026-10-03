# %% [markdown]
# # 01 · Size before you serve: weights, KV blocks, and the number vLLM will print
#
# **Tier:** T0. This notebook does only arithmetic from `config.json` and a GPU datasheet, and it runs on any computer. Optional T1:
# start a real `vllm serve` (see `deploy/any-gpu/`) and compare its startup log with your prediction.
#
# ## The one-minute version
#
# A GPU that runs vLLM holds four things:
#
# - the **weights**,
# - the **activation peak** of the forward pass in the profile run,
# - the **CUDA graphs and other buffers**,
# - the **KV blocks**.
#
# vLLM takes
# $\mathtt{gpu\_memory\_utilization} \times{}$ $\text{total memory}$ and subtracts the first three. Then it
# makes *all* of the remainder into fixed-size blocks (16 tokens each by default). From the model config alone,
# you can predict:
#
# $$
# \begin{aligned}
# \text{KV bytes per token} &= 2\ (\text{K and V}) \times \text{layers} \times \mathtt{kv\_heads} \times \mathtt{head\_dim}
#   \times \text{bytes} \\
# &\qquad (\text{per GPU: } \mathtt{kv\_heads} / \mathrm{TP}) \\[4pt]
# \mathtt{num\_blocks} &= \left\lfloor \frac{\text{KV budget}}{\mathtt{block\_size} \times \text{KV bytes per token}} \right\rfloor \\[4pt]
# \text{max concurrency} &= \frac{\mathtt{num\_blocks}}{\lceil \mathtt{max\_model\_len} / \mathtt{block\_size} \rceil} \\
# &\qquad \leftarrow \text{“Maximum concurrency ... Nx”}
# \end{aligned}
# $$
#
# After this notebook, you can tell these things before you rent anything:
#
# - if a model fits a GPU,
# - how many requests of a given length the GPU can hold at the same time,
# - what each knob gives you: `max_model_len`, `kv_cache_dtype`, `quantization`, `tensor_parallel_size`,
#   `gpu_memory_utilization`.
#
# Concepts: PRIMER §4 "KV cache management revisited" ([`PRIMER.md`](../../PRIMER.md)). The same
# per-token formula is in [`00-foundations/gpu-capacity-planning`](../../../../00-foundations/gpu-capacity-planning/PRIMER.md)
# (`capacity.kv_per_token_kb`). A test makes sure that the two formulas agree.

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
# `size()` does all of the calculation. It prints the steps in the same order as vLLM. The overhead lines
# are *estimates*. You calibrate them from a real log in exercise 1.5. The block arithmetic is exact.

# %%
t4 = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096, typical_len=1024)
print(t4.summary())

# %% [markdown]
# A 1 GB model gets approximately 12 GiB of KV. On a 16 GB card, a small model is *all cache*. The
# "Maximum concurrency" is a worst case, with every request at `max_model_len`. vLLM allocates blocks when
# tokens arrive. Thus, with 1,024-token requests, the engine can hold approximately four times more.
#
# ## Worked example: an 8B model on one L4 (24 GB) — the startup error everyone meets once

# %%
l4 = size("llama-3.1-8b-instruct", "L4")          # default max_model_len = max_position_embeddings = 131072
print(l4.summary())

# %% [markdown]
# vLLM does not start when one request of `max_model_len` tokens cannot fit. The model advertises a 128K
# context, and that context alone needs 16 GiB of KV. The knobs of this notebook are the solutions.
# (vLLM v0.30.0 also accepts `--max-model-len -1`. This value sets the length automatically to the largest
# length that the KV cache holds. This statement agrees with its `vllm/config/model.py`.)
#
# ## Worked example: the headline number — 2K-token sessions of an 8B model on one L4
#
# A design review asks for this figure: how many ~2K-token chat sessions does Llama-3.1-8B in bf16
# hold on one L4? The cell gives every input, because a change in the inputs changes the answer by one or two sessions.

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
# Thus, the answer is **about 18 concurrent 2K-token sessions** (2,363 blocks, 18.5 at 2,048 tokens, 18.9 at
# 2,000). This answer has three assumptions:
#
# - CUDA can see all of the L4's 22.49 GiB.
# - `gpu_memory_utilization` is 0.92, the vLLM v0.30.0 default.
# - The estimated overhead is ~1.1 GiB.
#
# The honest range is 17-19. CUDA usually reports a few hundred MiB less than nvidia-smi (verify on your card),
# and this difference alone costs a session.
#
# PRIMER §4 shows both sets of inputs side by side. Its simulated
# numbers use the same formulas with the simpler inputs of the core. These inputs are 0.9 of 24 GB minus a flat
# 1 GB reserve (2,164 blocks, **17** sessions). Neither result is a measurement. The measurement is the
# `Available KV cache memory` line of the startup log (exercise 1.5).
#
# ## Exercise 1.1 — KV bytes per token, from a raw `config.json`
#
# Write `kv_bytes_per_token(cfg, kv_bytes=2, tp=1)` for a plain dict that you read from `config.json`. Real
# configs have two traps for you:
#
# - A config can give `head_dim` explicitly, and this value can be different from
#   $\mathtt{hidden\_size} / \mathtt{num\_attention\_heads}$ (Qwen3).
# - The cache stores `num_key_value_heads` heads (GQA), not `num_attention_heads` heads.
#
# With tensor parallelism, each GPU holds
# $\mathtt{kv\_heads} / \mathtt{tp}$ heads, but never fewer than one. When
# $\mathtt{tp} > \mathtt{kv\_heads}$, vLLM replicates the heads.

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
# The input is the number of bytes that stay available for KV. Return `(num_blocks, max_concurrency, kv_cache_tokens)`
# exactly as vLLM calculates them:
#
# - Count whole blocks only.
# - A request of `max_model_len` tokens needs
#   $\lceil \mathtt{max\_model\_len} / \mathtt{block\_size} \rceil$ blocks. A partial block costs a whole block.
# - The token capacity in the log is
#   $\lfloor \mathtt{max\_concurrency} \times \mathtt{max\_model\_len} \rfloor$.

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
# You serve `llama-3.1-8b-instruct` in bf16 on one L4 with the default utilization. The design
# requirement has two parts: vLLM must start, and **at least 4 requests of the maximum length** must fit at
# the same time. From `choose_max_model_len()`, return the largest multiple of 1,024 that meets the requirement.
# Invert the block formula yourself:
#
# 1. Take `num_blocks` from one `size()` report. The KV budget does not depend on `max_model_len`.
#    Thus, any length is correct for this step.
# 2. Find the largest $L$ with
#    $4 \times \lceil L / 16 \rceil \le \mathtt{num\_blocks}$.
#
# The check compares your result with `sizing.max_model_len_for`. Then read the answer as a product decision.
# Is that context length sufficient for your workload? Or is it time for FP8, a larger GPU, or TP=2?

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
# If you change the 8B model to FP8 weights (`--quantization fp8`) *and* FP8 KV (`--kv-cache-dtype fp8`),
# concurrency increases in two ways:
#
# - The weights become smaller, and thus the KV budget becomes larger.
# - The KV of each token decreases by half.
#
# Write `predicted_gain(r_bf16, weights_saved_bytes)`. It predicts the ratio of max concurrency
# (fp8/fp8 over bf16/bf16) from two inputs. The inputs are the bf16 report and the number of bytes that the weights
# decrease by. Do not call `size()` for the FP8 case. The check compares your result with the full calculation.

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
# The overhead terms of the prediction are guesses. The startup log is the truth. The next cell uses an
# **illustrative** excerpt in the log format of `vllm serve`. This excerpt is sample output, not a measurement.
# With a real server, point `LOG_TEXT` at your `vllm.log`.
#
# Write `replan(log, max_model_len, kv_per_token)`. It takes the parsed log and returns the max concurrency
# that vLLM *will* report at a different `max_model_len`. The KV budget and the blocks do not depend on
# `max_model_len`. Thus, it is not necessary to start the server again.

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
# On a GPU box (or a Colab T4), start vLLM and capture its log. Then run the cells in this notebook again,
# with `VLLM_LOG` set to the log file. After that, the calibrated overhead makes each later prediction for
# that GPU and model exact:
#
# ```bash
# vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.92 > vllm.log 2>&1 &
# python -m servelab size --model qwen2.5-0.5b-instruct --gpu T4 --dtype half --max-model-len 4096
# ```
#
# If you start vLLM with a different utilization (the Colab recipe uses 0.85), the calibration reads it
# from the log line `The current --gpu-memory-utilization=...`. You can also set `GPU_MEM_UTIL`. On a T4
# (compute capability 7.5), the log also names the attention backend. vLLM v0.30.0 selects
# `TRITON_ATTN` automatically, because its FlashAttention backend needs sm_80+. No flag or environment
# variable is necessary. (`VLLM_ATTENTION_BACKEND` no longer exists. See `deploy/any-gpu/`.)

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
# **Two minutes:** "Before I select hardware, I calculate the size from `config.json`. For each token, the KV cache costs
# $2 \times \text{layers} \times \text{KV heads} \times{}$ $\text{head dim} \times \text{bytes}$. This is 12 KB
# for Qwen2.5-0.5B and 128 KB for Llama-3.1-8B. vLLM reserves `gpu_memory_utilization` of the card and pays for the
# weights and the overheads. Then it divides the remainder into 16-token blocks. Thus, an 8B model in bf16 on an L4 has
# about 4.6 GiB of KV, that is ~37K tokens.
#
# "At the vLLM default utilization of 0.92, this is about 18 concurrent 2K-token sessions. The default 128K context
# does not even start. At 8K context, only four requests fit in the worst case.
#
# "FP8 weights and FP8 KV give ~4.8x the concurrency. The weights become 7 GB smaller, and this increases the KV budget
# from 4.6 to 11.1 GiB. Also, the KV of each token decreases by half. Then I compare the prediction with the
# 'Available KV cache memory' and 'Maximum concurrency' lines of the startup log. I plan other context lengths from
# the calibrated number."
#
# **Drill 1.** *vLLM says "Maximum concurrency for 32,768 tokens per request: 1.5x". Can it serve
# 10 users?* Yes, if their requests are short. The figure is a worst case at `max_model_len`.
# vLLM allocates blocks when tokens arrive. With 3K-token conversations, approximately ten times more requests fit.
# Then `max_num_seqs` and the SLO set the limit on concurrency, and also preemption if the lengths increase.
#
# **Drill 2.** *Why is `head_dim` a trap?* Some configs (Qwen3) set it explicitly, and its value is different from
# $\mathtt{hidden\_size} / \mathtt{num\_attention\_heads}$. If you calculate `head_dim` from those values, your count of
# the KV of Qwen3-0.6B is 2x too low.
#
# **Drill 3.** *Tensor parallelism 2 halves the weights per GPU. What does it do to KV per token per
# GPU?* It also decreases it by half, because the GPUs divide the KV heads between them. But if there are fewer KV
# heads than GPUs, vLLM replicates the heads, and the KV per GPU does not decrease more.
