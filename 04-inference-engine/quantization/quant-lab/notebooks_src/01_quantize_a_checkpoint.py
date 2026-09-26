# %% [markdown]
# # 01 · Quantize a checkpoint: recipes, the compressed-tensors layout, and a loader that checks it
#
# **Tier:** T0 — the bundled tiny Llama (299,648 parameters, two layers, trained on two tasks with
# exact answers) is quantized with this lab's numpy recipes, written as a compressed-tensors
# checkpoint and read back; nothing is downloaded. T1 — the same recipes run with llm-compressor on
# `Qwen/Qwen2.5-0.5B-Instruct` (the script is printed here; it runs when llm-compressor is installed,
# a GPU is present and `QUANTLAB_RUN_T1=1`).
#
# ## The one-minute version
#
# A quantized checkpoint is three things: **codes** (INT4 packed eight to an int32, FP8 bytes, FP4
# nibbles), **scales** (and zero points) that say what the codes mean, and a
# **`quantization_config`** in `config.json` that tells the loader which is which. llm-compressor
# writes all three with `oneshot(model, recipe=...)`; vLLM reads them without a `--quantization` flag.
#
# * `FP8_DYNAMIC` needs no data: FP8 weights with a scale per output channel, activations quantized
#   per token at run time. `W4A16` needs calibration data *if* you want GPTQ or AWQ instead of
#   round-to-nearest (RTN) — and on a model with outlier activations you do.
# * Not everything is quantized: `lm_head`, the embeddings and the norms stay 16-bit
#   (`ignore=["lm_head"]` in every recipe), which is why a real INT4 checkpoint is bigger than
#   params x 4 bits.
# * Groups must divide the layer's input width (`in_features % 128 == 0` for W4A16), or the tools
#   refuse the layer.
#
# Concepts: PRIMER §3 "Granularity and the bits-per-weight budget", §4 "Weight-only post-training
# quantization" and §9 "Producing a checkpoint" ([`PRIMER.md`](../../PRIMER.md)); the formats and error
# metrics this builds on are in the serving-engine primer's §8
# ([`../../../serving-engine/PRIMER.md`](../../../serving-engine/PRIMER.md)).

# %%
import json, pathlib, tempfile
from quantlab import compress as C, env, numerics as N, serve, stio, tinymodel as tm
import numpy as np

print(env.describe())
OUT = pathlib.Path(tempfile.mkdtemp(prefix="quantlab-01-"))
model = tm.load()
print(f"{model.num_params():,} parameters;", {k: model.config[k] for k in ("hidden_size", "intermediate_size",
      "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")})
for task in tm.TASKS:
    p, a = tm.make_task(task, 2, 1000)
    print(f"  {task:8s} e.g. {tm.decode(p[0]):12s} -> {tm.decode(a[0])}   accuracy {model.accuracy(task, 500):.1%} (500 held-out)")

# %% [markdown]
# ## Worked example: what a dense checkpoint is
#
# The safetensors header lists every tensor with its dtype and shape; the bytes follow. Fourteen
# projections (q, k, v, o, gate, up, down in two layers) are what quantization will shrink.

# %%
rows = stio.summary(tm.TINY_DIR / "model.safetensors")
for name, dt, shape, nbytes in rows[:6]:
    print(f"  {name:45s} {dt:5s} {str(shape):12s} {nbytes:>7,} B")
print(f"  ... {len(rows)} tensors, {sum(r[3] for r in rows):,} bytes of data")
linear = sum(model.weights[n + ".weight"].size for n in model.linear_names())
print(f"linear-layer weights: {linear:,} of {model.num_params():,} ({linear / model.num_params():.1%})")

# %% [markdown]
# ## Worked example: FP8_DYNAMIC, written the way llm-compressor writes it
#
# `Recipe("FP8_DYNAMIC")` is `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC",
# ignore=["lm_head"])`. Each projection becomes an `F8_E4M3` `weight` plus a bf16 `weight_scale` of
# shape `[out, 1]` (one per output channel, `amax / 448`); activation scales are computed per token
# at run time, so there is nothing to calibrate.

# %%
rep = C.quantize_checkpoint(tm.TINY_DIR, OUT / "tiny-FP8_DYNAMIC", C.Recipe("FP8_DYNAMIC"))
for name, dt, shape, nbytes in stio.summary(OUT / "tiny-FP8_DYNAMIC/model.safetensors"):
    if "layers.0.self_attn.q_proj" in name:
        print(f"  {name:45s} {dt:8s} {str(shape):10s} {nbytes:>6,} B")
qc = json.loads((OUT / "tiny-FP8_DYNAMIC/config.json").read_text())["quantization_config"]
print(json.dumps({k: qc[k] for k in ("quant_method", "format", "ignore", "quantization_status")}))
print("weights:", json.dumps({k: qc["config_groups"]["group_0"]["weights"][k] for k in ("num_bits", "type", "strategy")}),
      "| input_activations:", json.dumps({k: qc["config_groups"]["group_0"]["input_activations"][k]
                                         for k in ("num_bits", "type", "strategy", "dynamic")}))
print(f"{rep['dense_bytes']:,} -> {rep['bytes']:,} bytes; validate: {C.validate_checkpoint(rep['path']) or 'ok'}")

# %% [markdown]
# ## Exercise 1.1 — pack INT4 codes the way compressed-tensors does
#
# `pack_to_int32` adds 8 to each signed code (-8..7 becomes 0..15) and puts eight of them in one
# 32-bit word, **element 0 in the lowest four bits**. Write `pack_int4(q)` for an integer array whose
# last dimension is a multiple of 8; return `int32` words (view the `uint32` result as `int32` — the
# top nibble often sets the sign bit).

# %% exercise
def pack_int4(q):
    ### BEGIN SOLUTION
    u = (np.asarray(q, dtype=np.int64) + 8).astype(np.uint64)
    u = u.reshape(*u.shape[:-1], -1, 8)
    words = (u << (np.arange(8, dtype=np.uint64) * 4)).sum(-1).astype(np.uint32)
    return words.view(np.int32)
    ### END SOLUTION

# %% check
w = pack_int4(np.array([[-8, -7, 0, 1, 2, 3, 4, 7]]))
assert w.dtype == np.int32 and hex(int(w.view(np.uint32)[0, 0])) == "0xfcba9810", hex(int(w.view(np.uint32)[0, 0]))
q = np.random.default_rng(0).integers(-8, 8, (4, 256))
assert (pack_int4(q) == N.pack_int32(q, 4)).all() and pack_int4(q).shape == (4, 32)
print("✅ [-8, -7, 0, 1, 2, 3, 4, 7] -> 0xfcba9810: the same words compressed-tensors writes")

# %% [markdown]
# ## Exercise 1.2 — predict the size of a W4A16 checkpoint before writing it
#
# W4A16 (groups of 128, symmetric) stores, per projection: packed codes (half a byte per weight), one
# bf16 scale per group of 128 inputs per output row, and `weight_shape` (two int64). Everything that
# is not a projection stays bf16. Write `predict_w4a16_bytes(model, group_size)` — tensor data only,
# no header. Then compare the ratio with the naive "4 bits instead of 16".

# %% exercise
def predict_w4a16_bytes(model, group_size=128):
    ### BEGIN SOLUTION
    lin = sum(model.weights[n + ".weight"].size for n in model.linear_names())
    n_proj = len(model.linear_names())
    rest = model.num_params() - lin
    return lin // 2 + lin // group_size * 2 + n_proj * 16 + rest * 2
    ### END SOLUTION

# %% check
rep4 = C.quantize_checkpoint(tm.TINY_DIR, OUT / "tiny-W4A16", C.Recipe("W4A16"))
actual = C.checkpoint_bytes(rep4["path"])["total"]
assert predict_w4a16_bytes(model) == actual, (predict_w4a16_bytes(model), actual)
dense = C.checkpoint_bytes(tm.TINY_DIR)["total"]
print(f"✅ predicted {predict_w4a16_bytes(model):,} B = written {actual:,} B; {dense / actual:.2f}x smaller, not 4x: "
      f"scales cost {16 / 128:.3f} bits per weight and {model.num_params() - linear:,} parameters stay bf16")

# %% [markdown]
# ## Worked example: round-to-nearest versus GPTQ on a model with outlier channels
#
# This model's projection inputs carry two massive-activation channels (planted after training:
# the RMSNorm weight of those channels is 24x larger and the matching weight columns 24x smaller —
# the function is unchanged). RTN rounds those small columns to almost nothing, and they multiply
# the largest inputs. GPTQ quantizes one column at a time and pushes each column's rounding error
# onto the columns not yet quantized, weighted by the inverse Hessian of the calibration inputs, so
# the error that lands on the big-input channels is compensated elsewhere. Layer error below is
# `||X W_hat^T - X W^T|| / ||X W^T||` on calibration inputs.

# %%
print("outlier channels:", tm.outlier_channels(model))
calib = C.calibration_inputs(model, 128)
runs = {r.name: C.quantize_model(model, r, calib)
        for r in (C.Recipe("W4A16"), C.Recipe("W4A16", group_size=32), C.Recipe("W4A16", ("gptq",)), C.Recipe("W4A16", ("awq",)))}
print(f"{'projection':28s}" + "".join(f"{k:>20s}" for k in runs))
for n in model.linear_names()[:7]:
    print(f"{n.split('layers.')[1]:28s}" + "".join(f"{q.layers[n].out_err:>20.2%}" for q in runs.values()))
for k, q in runs.items():
    print(f"{k:20s} add {q.model().accuracy('add', 500):.1%}   reverse {q.model().accuracy('reverse', 500):.1%}")

# %% [markdown]
# Two things to notice. Smaller groups (g32) barely help the projections whose inputs have the
# outlier channels (q/k/v, gate/up): a group is a run of *inputs within one output row*, and the
# shrunken outlier columns are small relative to their neighbours in every group. The fix is
# per-*input-channel*: GPTQ (error compensation) or AWQ (scale the salient columns up before
# rounding, fold the inverse into the norm).
#
# ## Exercise 1.3 — the symmetric INT4 group quantizer
#
# Implement compressed-tensors' symmetric INT4 grid for `w` shaped `[out, in]`: for each output row
# and each run of `g` inputs, `scale = amax / 7.5`, `code = clip(round(w / scale), -8, 7)`. Return
# `(codes [out, in], scales [out, in / g], w_hat)`.

# %% exercise
def int4_groups(w, g=128):
    ### BEGIN SOLUTION
    rows, cols = w.shape
    b = w.reshape(rows, cols // g, g)
    scales = np.maximum(np.abs(b).max(-1), 1e-12) / 7.5
    codes = np.clip(np.round(b / scales[..., None]), -8, 7)
    return codes.reshape(rows, cols), scales, (codes * scales[..., None]).reshape(rows, cols)
    ### END SOLUTION

# %% check
W = model.weights["model.layers.0.mlp.down_proj.weight"].astype(np.float64)
codes, scales, w_hat = int4_groups(W, 128)
ref = C.rtn(W, C.Recipe("W4A16").scheme_args()["weights"])
assert scales.shape == (128, 2) and codes.min() >= -8 and codes.max() <= 7
assert np.allclose(w_hat, ref[3]) and np.allclose(scales, ref[1])
print(f"✅ your quantizer = the lab's RTN; relative weight error {np.linalg.norm(w_hat - W) / np.linalg.norm(W):.2%} "
      f"at {4 + 16 / 128} bits per weight")

# %% [markdown]
# ## Exercise 1.4 — the loader: from packed words back to weights
#
# A loader never sees `w_hat`. Given the tensors of one projection in a `pack-quantized` checkpoint
# (`weight_packed` int32, `weight_scale` bf16 `[out, in/g]`, `weight_shape`), rebuild the float
# weight. Use `N.unpack_int32(words, 4, cols)` for the unpacking and `.numpy()` to decode bf16.

# %% exercise
def dequant_w4a16(tensors, name):
    ### BEGIN SOLUTION
    rows, cols = (int(v) for v in tensors[name + ".weight_shape"].numpy())
    q = N.unpack_int32(tensors[name + ".weight_packed"].numpy(), 4, cols).astype(np.float64)
    s = tensors[name + ".weight_scale"].numpy().astype(np.float64)
    g = cols // s.shape[1]
    return (q.reshape(rows, -1, g) * s[:, :, None]).reshape(rows, cols)
    ### END SOLUTION

# %% check
T = stio.load(OUT / "tiny-W4A16/model.safetensors")
loaded, _ = C.load_checkpoint(OUT / "tiny-W4A16")
for n in model.linear_names():
    assert np.allclose(dequant_w4a16(T, n), loaded.weights[n + ".weight"], atol=1e-6), n
before, after = runs["W4A16 (rtn)"].model().accuracy("add", 500), loaded.accuracy("add", 500)
assert abs(before - after) <= 0.01                   # only the bf16 rounding of the scales differs
print(f"✅ all {len(model.linear_names())} projections decode; add accuracy {before:.1%} in memory, "
      f"{after:.1%} reloaded from disk (bf16 scales)")

# %% [markdown]
# ## Exercise 1.5 — will the tools accept group size 128 for this model?
#
# llm-compressor errors at initialize, and vLLM's Marlin refuses the layer, when a projection's
# `in_features` is not a multiple of the group size. q/k/v and gate/up read `hidden_size` inputs,
# o_proj reads `num_heads x head_dim`, down_proj reads `intermediate_size`. Write
# `bad_projections(config, g)` returning the sorted names (`"q_proj"`, ... `"down_proj"`) that fail.

# %% exercise
def bad_projections(config, g=128):
    ### BEGIN SOLUTION
    h = config["hidden_size"]
    heads, hd = config["num_attention_heads"], config.get("head_dim") or h // config["num_attention_heads"]
    ins = {"q_proj": h, "k_proj": h, "v_proj": h, "gate_proj": h, "up_proj": h,
           "o_proj": heads * hd, "down_proj": config["intermediate_size"]}
    return sorted(k for k, v in ins.items() if v % g)
    ### END SOLUTION

# %% check
qwen = json.loads((tm.DATA / "configs/qwen2.5-0.5b-instruct.json").read_text())
llama = json.loads((tm.DATA / "configs/llama-3.1-8b-instruct.json").read_text())
assert bad_projections(qwen) == [] and bad_projections(llama) == []            # 896 = 7 x 128, 4,864 = 38 x 128
small = {"hidden_size": 576, "intermediate_size": 1536, "num_attention_heads": 9}   # a SmolLM2-135M-like shape (verify)
assert bad_projections(small) == ["gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
assert bad_projections(small, 64) == []
with_err = None
try:
    C.rtn(np.zeros((8, 576)), C.Recipe("W4A16").scheme_args()["weights"])
except ValueError as e:
    with_err = str(e)
assert "divisible" in with_err
print("✅ 576-wide models need g64 or g32 for INT4 (576 = 4.5 x 128); the lab's quantizer refuses g128 the same way")

# %% [markdown]
# ## On a real model (T1): the same recipes with llm-compressor
#
# The lab renders the llm-compressor 0.14 script for each recipe (run it in its own environment: the
# vLLM docs recommend separate environments for vLLM and llm-compressor). `deploy/any-gpu/compress.sh`
# does exactly this on a GPU box. The result is a directory vLLM serves with no `--quantization` flag;
# `serve.check_checkpoint` says what vLLM will do with it on each GPU.

# %%
for r in (C.Recipe("FP8_DYNAMIC"), C.Recipe("W4A16", ("gptq",))):
    print(C.llmcompressor_script(r, "Qwen/Qwen2.5-0.5B-Instruct"))
cfg = {"quantization_config": C.quantization_config(C.Recipe("FP8_DYNAMIC"))}
for g in ("T4", "L4", "H100-80GB"):
    print(f"FP8_DYNAMIC checkpoint on {g:9s}:", serve.check_checkpoint(cfg, g)[1])
if env.t1_allowed() and env.has("llmcompressor"):
    C.run_llmcompressor(C.Recipe("FP8_DYNAMIC"), "Qwen/Qwen2.5-0.5B-Instruct", str(OUT / "Qwen2.5-0.5B-Instruct-FP8_DYNAMIC"))
    real = json.loads((OUT / "Qwen2.5-0.5B-Instruct-FP8_DYNAMIC/config.json").read_text())
    print("MEASURED: wrote a real checkpoint;", serve.vllm_scheme(real["quantization_config"]))
else:
    print("T0: llm-compressor not run here (needs a GPU, `pip install llmcompressor==0.14.0` and QUANTLAB_RUN_T1=1).")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We ship quantized checkpoints in the compressed-tensors format because vLLM
# auto-detects it: the `quantization_config` names the scheme, the tensors carry codes and scales,
# and nobody has to remember a `--quantization` flag. For FP8 we use `FP8_DYNAMIC`: per-channel
# weight scales and per-token activation scales at run time, no calibration data. For INT4 we use
# GPTQ with calibration data drawn from our own traffic, because round-to-nearest loses the weight
# columns that meet our largest activations — on the lab's model RTN leaves 11% error on the q
# projection and GPTQ under 2%. The LM head and embeddings stay bf16, so the checkpoint is ~3x
# smaller, not 4x, and we check that every projection's input width divides the group size before we
# start."
#
# **Drill 1.** *Why does `FP8_DYNAMIC` need no calibration data but `W4A16` with GPTQ does?* — FP8
# dynamic computes weight scales from the weights and activation scales per token at run time;
# GPTQ's error compensation needs the Hessian `X^T X` of real inputs to each layer.
#
# **Drill 2.** *The INT4 checkpoint of a 0.5B model is only 2.2x smaller. Is the tool broken?* — No:
# Qwen2.5-0.5B ties a 136 M-parameter embedding that stays bf16 (27.6% of the model), plus 16/128 bits
# of scale per weight. Big models approach 3.5-3.9x; small ones do not.
#
# **Drill 3.** *llm-compressor refuses `W4A16` on one model and not another. Why?* — Group size must
# divide `in_features`: 576-wide models fail g128 and need g64/g32 (and vLLM's Marlin only accepts
# -1, 32, 64, 128).
