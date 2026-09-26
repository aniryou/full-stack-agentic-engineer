# %% [markdown]
# # 04 · KV-cache quantization in vLLM: twice the sessions, a cheaper long-context step, and its conditions
#
# **Tier:** T0 — sizing reproduces the serving lab's memory model exactly; per-step times come from
# the roofline emulator (**simulated**); the accuracy side is measured on the bundled tiny model with
# each KV dtype emulated. T1 (Ada or newer: L4, RTX 4090, H100) — `vllm serve ... --kv-cache-dtype fp8`;
# with `QUANTLAB_VLLM_LOG` and `QUANTLAB_URL` set, the last code cell reads the capacity and backend
# back from its startup log and `/metrics` and measures decode at two context lengths. Without them it
# parses a bundled log: **sample output in the documented format (illustrative)**.
#
# ## The one-minute version
#
# The KV cache costs `2 x layers x kv_heads x head_dim x bytes` per token, vLLM turns every byte
# left after weights and overheads into blocks of it, and each decode step reads all of it for
# every running sequence. `--kv-cache-dtype fp8` stores K and V in one byte:
#
# * **about 2x the tokens** in the same memory (Llama-3.1-8B on an L4: 2,363 -> 4,727 blocks),
#   independent of how the weights are quantized — the two multiply;
# * a **cheaper decode step at long context**, where the KV read rivals the weight read;
# * **conditions**: a backend that reads FP8 KV on your GPU — none on a T4; FlashInfer on an A100 or
#   L4 (FlashAttention 2 has no FP8 KV, so the flag *changes the attention backend*); FlashAttention 3
#   on an H100 (which also quantizes Q); FlashInfer on B200. Scales default to **1.0** unless the
#   checkpoint carries calibrated `k_scale` / `v_scale`.
#
# Concepts: PRIMER §6 "KV-cache quantization" ([`PRIMER.md`](../../PRIMER.md)); the backend rules are
# vllm-internals §6.3 ([`../../../vllm-internals/vllm-internals-primer.md`](../../../vllm-internals/vllm-internals-primer.md));
# FP8 attention's error sources are the FlashAttention deep dive's §9.4
# ([`../../../flash-attention/flash-attention-deep-dive.md`](../../../flash-attention/flash-attention-deep-dive.md)).

# %%
import json
from quantlab import bench as B, env, evalharness as E, kv, serve, tinymodel as tm
import numpy as np

print(env.describe())
print("Llama-3.1-8B-Instruct on an L4, vLLM v0.30.0 defaults (0.92 utilization; the serving lab's memory model)")
print(kv.table("llama-3.1-8b-instruct", "L4"))
print("\nQwen2.5-1.5B-Instruct on a T4")
print(kv.table("qwen2.5-1.5b-instruct", "T4"))

# %% [markdown]
# The L4 rows are the serving lab's `sizing.size()` numbers to the block (this lab's tests pin them,
# and compare against the serving lab's code when it is in the checkout). FP8 weights free 7 GB for KV;
# FP8 KV halves every token; together an 8B model goes from 19 to 91 two-thousand-token sessions on
# one 24 GB card. On a T4 the FP8 KV rows are arithmetic only: no attention backend there reads it.
#
# ## Exercise 4.1 — KV bytes per token from `config.json`
#
# Write `kv_bytes(config, dtype_bytes)`: `2 x num_hidden_layers x num_key_value_heads x head_dim x
# dtype_bytes`. Use the config's `head_dim` when it has one — Qwen3-0.6B's is 128 although
# `hidden_size / num_attention_heads` is 64.

# %% exercise
def kv_bytes(config, dtype_bytes):
    ### BEGIN SOLUTION
    hd = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * hd * dtype_bytes
    ### END SOLUTION

# %% check
cfgs = {n: json.loads((tm.DATA / f"configs/{n}.json").read_text()) for n in ("llama-3.1-8b-instruct", "qwen2.5-0.5b-instruct", "qwen3-0.6b")}
for n, c in cfgs.items():
    for d, b in (("auto", 2), ("fp8", 1)):
        assert kv_bytes(c, b) == kv.kv_bytes_per_token(kv.load_shape(n), d), (n, d)
print("✅ bytes per token (bf16 / fp8):", {n: (kv_bytes(c, 2), kv_bytes(c, 1)) for n, c in cfgs.items()})

# %% [markdown]
# ## Exercise 4.2 — from a KV budget to blocks and sessions
#
# vLLM allocates `floor(budget / (block_size x bytes_per_token))` blocks; a session of `T` tokens needs
# `ceil(T / block_size)` of them. Write `blocks_and_sessions(budget_bytes, bytes_per_token,
# session_tokens, block_size=16)` returning `(blocks, sessions)` (sessions as a float, like vLLM's
# "Maximum concurrency").

# %% exercise
def blocks_and_sessions(budget_bytes, bytes_per_token, session_tokens, block_size=16):
    ### BEGIN SOLUTION
    blocks = int(budget_bytes // (block_size * bytes_per_token))
    return blocks, blocks / -(-session_tokens // block_size)
    ### END SOLUTION

# %% check
for w in ("bf16", "fp8", "w4a16"):
    for d in ("auto", "fp8"):
        r = kv.size("llama-3.1-8b-instruct", "L4", weights=w, kv_cache_dtype=d)
        blocks, sessions = blocks_and_sessions(r.kv_budget_bytes, r.kv_bytes_per_token, 2000)
        assert blocks == r.num_blocks and abs(sessions - r.sessions(2000)) < 1e-9
r = kv.size("llama-3.1-8b-instruct", "L4", kv_cache_dtype="fp8")
print(f"✅ bf16 weights + fp8 KV on an L4: {r.kv_budget_bytes / 2**30:.2f} GiB of KV -> {r.num_blocks:,} blocks "
      f"-> {r.sessions(2000):.1f} sessions of 2,000 tokens (vllm-internals §4.7: 2,363 -> 4,727 blocks)")

# %% [markdown]
# ## Worked example: which attention backend reads which KV dtype
#
# vLLM walks a per-GPU priority list and keeps the first backend that accepts the KV dtype
# (`flash_attn.py`, `flashinfer.py`, `triton_attn.py` at v0.30.0 / main, Sep 2026 — verify):

# %%
dtypes = ("auto", "fp8", "fp8_e5m2", "int8_per_token_head", "nvfp4")
print(f"{'GPU':11s}" + "".join(f"{d:>22s}" for d in dtypes))
for g in ("T4", "A100-80GB", "L4", "H100-80GB", "B200", "RTXPRO6000"):
    print(f"{g:11s}" + "".join(f"{(kv.attention_backend(g, d).backend or 'refused'):>22s}" for d in dtypes))

# %% [markdown]
# ## Exercise 4.3 — the FP8 KV rule in one function
#
# Encode the `auto` / `fp8` columns: below sm_80 only Triton exists (it reads FP8 only from sm_89, so
# a T4 has no FP8 KV); FlashAttention needs sm_80 and reads FP8 KV only as FA3 on sm_90; FlashInfer
# (sm_80+) reads FP8 and comes first on sm_100. Write `fp8_kv_backend(sm)` returning `"FLASH_ATTN"`,
# `"FLASHINFER"` or `None` for `--kv-cache-dtype fp8`, and `default_backend(sm)` for `auto`.

# %% exercise
def default_backend(sm):
    ### BEGIN SOLUTION
    if sm < 80:
        return "TRITON_ATTN"
    return "FLASHINFER" if 100 <= sm < 110 else "FLASH_ATTN"
    ### END SOLUTION

def fp8_kv_backend(sm):
    ### BEGIN SOLUTION
    if sm < 80:
        return None
    if sm == 90:
        return "FLASH_ATTN"
    return "FLASHINFER"
    ### END SOLUTION

# %% check
for g in ("T4", "A100-80GB", "L4", "H100-80GB", "B200", "RTXPRO6000"):
    sm = serve.gpu(g).sm
    assert (kv.attention_backend(g, "auto").backend or "").startswith(default_backend(sm)), g
    got = kv.attention_backend(g, "fp8").backend
    assert (got.split(" ")[0] if got else None) == fp8_kv_backend(sm), g
print("✅ on an L4 or A100 the fp8 flag moves attention from FlashAttention to FlashInfer — "
      "so a speed change after the flag is not only the KV dtype; benchmark both")

# %% [markdown]
# ## Worked example: what FP8 KV does to a decode step, by context length
#
# FP8 weights on an L4, batch 32 (simulated). The step reads the weights once plus every sequence's
# KV; below some context the weights dominate and the KV dtype barely matters, above it the KV read
# does.

# %%
w8 = {d: B.profile("llama-3.1-8b-instruct", "L4", "fp8", kv_cache_dtype=d) for d in ("auto", "fp8")}
for ctx in (256, 1024, 2048, 4096, 8192):
    t = {d: p.step_time([ctx] * 32) * 1e3 for d, p in w8.items()}
    fits = {d: 32 * -(-ctx // 16) <= p.num_blocks for d, p in w8.items()}
    print(f"  context {ctx:5d}: bf16 KV {t['auto']:6.1f} ms{'' if fits['auto'] else ' (does not fit)':16s} "
          f"fp8 KV {t['fp8']:6.1f} ms{'' if fits['fp8'] else ' (does not fit)':16s} [SIMULATED]")

# %% [markdown]
# ## Exercise 4.4 — where the KV read catches up with the weight read
#
# Write `crossover_context(streamed_weight_bytes, kv_bytes_per_token, batch)`: the context length at
# which `batch x context x kv_bytes_per_token` equals the weights one step streams.

# %% exercise
def crossover_context(streamed_weight_bytes, kv_bytes_per_token, batch):
    ### BEGIN SOLUTION
    return streamed_weight_bytes / (batch * kv_bytes_per_token)
    ### END SOLUTION

# %% check
p16, p8 = w8["auto"], w8["fp8"]
c16 = crossover_context(p16.streamed_bytes, p16.kv_bytes_per_token, 32)
c8 = crossover_context(p8.streamed_bytes, p8.kv_bytes_per_token, 32)
assert round(c16) == 1915 and round(c8) == 3829
kv_part = lambda p, c: 32 * c * p.kv_bytes_per_token / (p.streamed_bytes + 32 * c * p.kv_bytes_per_token)  # noqa: E731
assert abs(kv_part(p16, c16) - 0.5) < 1e-9
print(f"✅ batch 32, FP8 weights: KV = weights at {c16:,.0f} tokens with bf16 KV, {c8:,.0f} with fp8 KV — "
      "past that, long-context decode is a KV problem and the KV dtype is the lever")

# %% [markdown]
# ## Worked example: the accuracy side, and the scale that can ruin it
#
# The tiny model with each KV dtype emulated (K is quantized after RoPE, as vLLM caches it). FP8's
# per-tensor scale divides K and V before rounding; vLLM uses 1.0 unless the checkpoint carries
# calibrated `k_scale`/`v_scale` (llm-compressor's `kv_cache_scheme`: `amax / 448` per layer).

# %%
calib_ids = np.concatenate(tm.make_task("add", 256, 7), axis=1)
ks, vs = kv.calibrate_kv_scales(tm.load(), calib_ids)
ref = tm.load()
cache, rows = {}, {}
cases = {"fp8 e4m3, scale 1.0 (vLLM default)": kv.kv_quantizer("fp8"),
         "fp8 e4m3, calibrated amax/448": kv.kv_quantizer("fp8", ks, vs),
         "fp8 e5m2, scale 1.0": kv.kv_quantizer("fp8_e5m2"),
         "int8 per token-head": kv.kv_quantizer("int8_per_token_head"),
         "int4 per token-head": kv.kv_quantizer("int4_per_token_head"),
         "fp8 e4m3, scale 100x too small": kv.kv_quantizer("fp8", {i: s / 100 for i, s in ks.items()}, {i: s / 100 for i, s in vs.items()}),
         "fp8 e4m3, scale 1000": kv.kv_quantizer("fp8", 1000.0, 1000.0)}
for name, q in cases.items():
    rows[name] = E.mini_eval(ref, ref, n=500, kv_quant=q, ref_cache=cache)
print("measured on the bundled tiny model (T0); K/V amax per layer:",
      {i: round(ks[i] * 448, 1) for i in ks}, "/", {i: round(vs[i] * 448, 1) for i in vs})
print(E.table(rows))

# %% [markdown]
# With `|K|, |V| <= 14` the default scale of 1.0 is as good as a calibrated one: E4M3's relative
# precision is the same in every binade, so the scale only matters at the ends of the range. A
# scale 100x too small saturates everything at 448 x scale (accuracy 0); a scale of 1,000 pushes
# typical values into the subnormals, where precision runs out. Integer KV formats are the opposite:
# their error depends on the scale everywhere, which is why they use dynamic per-token-head scales.
#
# ## Exercise 4.5 — is this FP8 scale safe?
#
# Write `fp8_scale_check(amax, rms, scale)`: return `"saturates"` if `amax / scale > 448`,
# `"underflows"` if `rms / scale < 2**-6` (typical values below E4M3's smallest normal), else `"ok"`.

# %% exercise
def fp8_scale_check(amax, rms, scale):
    ### BEGIN SOLUTION
    if amax / scale > 448:
        return "saturates"
    if rms / scale < 2 ** -6:
        return "underflows"
    return "ok"
    ### END SOLUTION

# %% check
cap = {}
def spy(kind, layer, x):
    cap[(kind, layer)] = x
    return x
ref.forward(calib_ids, kv_quant=spy)
k0 = cap[("k", 0)]
amax, rms = float(np.abs(k0).max()), float(np.sqrt((k0 ** 2).mean()))
verdicts = {s: fp8_scale_check(amax, rms, s) for s in (1.0, ks[0], ks[0] / 100, 1000.0)}
assert verdicts == {1.0: "ok", ks[0]: "ok", ks[0] / 100: "saturates", 1000.0: "underflows"}, verdicts
assert rows["fp8 e4m3, scale 100x too small"]["add"]["accuracy"] < 0.5 < rows["fp8 e4m3, scale 1.0 (vLLM default)"]["add"]["accuracy"]
print(f"✅ layer-0 K: amax {amax:.1f}, rms {rms:.2f} ->", {f"{s:g}": v for s, v in verdicts.items()})

# %% [markdown]
# ## On a real GPU (T1): turn it on and read it back
#
# On an L4 or H100 (not a T4): start `vllm serve` with `--kv-cache-dtype fp8` (the command below),
# keep its log (`... 2>&1 | tee vllm.log`), then set `QUANTLAB_VLLM_LOG=vllm.log` and
# `QUANTLAB_URL=http://127.0.0.1:8000` and run this cell. It checks three things against the
# prediction: the KV dtype and capacity (from the log, and from `/metrics`' `vllm:cache_config_info`),
# the attention backend, and decode speed at a short and a long context. Run it once with the flag and
# once without: the pair is the measurement. With neither variable set it parses a bundled sample log.

# %%
import os, pathlib
p = serve.plan("fp8-online", "L4", kv_cache_dtype="fp8", model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=16384)
print(p.command(), "2>&1 | tee vllm.log")
print(serve.plan("w4a16", "T4", kv_cache_dtype="fp8").notes[-1])
pred = kv.size("llama-3.1-8b-instruct", "L4", weights="fp8", kv_cache_dtype="fp8", max_model_len=16384)
log_path = pathlib.Path(os.environ.get("QUANTLAB_VLLM_LOG") or E.SAMPLES / "vllm_startup_fp8_fp8kv_l4.log")
text = log_path.read_text()
got = serve.parse_startup_log(text)
label = ("[sample output in the documented format (illustrative)]" if text.startswith("# Sample output")
         else f"MEASURED (startup log {log_path})")
print(label, got)
print(f"predicted for Llama-3.1-8B, FP8 weights + FP8 KV on an L4: {pred.kv_tokens:,} KV tokens, backend {pred.backend}; "
      f"the log says {got.get('kv_cache_tokens', 0):,} tokens, backend {got.get('attention_backend')}")
url = env.server_url()
if url:
    kind = B.server_kind(url, env.auth_headers())
    info = serve.parse_cache_config_info(env.get_text(url, "/metrics"))
    print(f"/metrics vllm:cache_config_info ({kind or 'unidentified'} server):",
          {k: info.get(k) for k in ("cache_dtype", "block_size", "num_gpu_blocks")})
    model_id = env.served_model(url)
    for ctx in (512, 4096):          # the KV read grows with context; the weight read does not
        r = B.run_http(url, model_id, users=8, n_requests=16, prompt_len=ctx, output_len=64, headers=env.auth_headers())
        print(f"  context {ctx:5d}: TPOT {r['tpot_ms_mean']:6.1f} ms, TTFT {r['ttft_ms_mean']:7.1f} ms   [{r['source']}]")
    print("Restart the server without --kv-cache-dtype fp8 and run this cell again: the TPOT gap at 4,096 is "
          "the FP8 KV win (plus the FlashAttention -> FlashInfer switch on an L4).")
else:
    print("T0: no server measured. Set QUANTLAB_URL (and QUANTLAB_VLLM_LOG) after starting the command above on an "
          "L4, RTX 4090 or H100.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "The KV cache is the other big tensor, and on a 24 GB card it is the one that caps
# concurrency. FP8 KV halves it: an 8B model on an L4 goes from 2,363 to 4,727 blocks with BF16 weights,
# and to 11,383 with FP8 weights — 19 to 91 two-thousand-token sessions. It also makes long-context decode
# cheaper: at batch 32 the KV read passes the weight read at about 1,900 tokens of context with BF16 KV.
# It needs an attention backend that reads FP8: nothing on T4s; on L4s the flag switches us from
# FlashAttention to FlashInfer, so we benchmark the pair, not just the dtype. We keep the default scale
# of 1.0 only after checking the K/V ranges; if the model has large K values we ship calibrated
# `k_scale`/`v_scale` in the checkpoint."
#
# **Drill 1.** *We turned on `--kv-cache-dtype fp8` on T4s and vLLM refused to start. Why?* — Triton's FP8
# path needs sm_89 and FlashAttention/FlashInfer need sm_80; a T4 is sm_75. Use INT4 weights to free memory instead.
#
# **Drill 2.** *FP8 KV made our short-prompt throughput worse on L4s. How?* — The flag also moved attention
# from FlashAttention 2 to FlashInfer; at short context the KV read is a small part of the step, so the
# backend difference can dominate. Measure with the flag on and off at your real context lengths.
#
# **Drill 3.** *Does FP8 KV break prefix caching?* — No: cached blocks are stored in FP8 and reused as
# they are; a hit returns the same quantized K/V a recompute would write (deterministic scales).
