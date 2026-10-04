# %% [markdown]
# # 01 · Quantize a checkpoint: recipes, the compressed-tensors layout, and a loader that checks it
#
# **Tier:** T0: the numpy recipes of this lab quantize the bundled tiny Llama (299,648 parameters, two layers,
# trained on two tasks with exact answers). The lab writes the result as a compressed-tensors checkpoint and reads
# it back. The notebook downloads nothing. T1: the same recipes run with llm-compressor on
# `Qwen/Qwen2.5-0.5B-Instruct`. The notebook prints the script. The script runs when llm-compressor is installed,
# a GPU is present and `QUANTLAB_RUN_T1=1`.
#
# ## The one-minute version
#
# A quantized checkpoint has three parts. The first part is the **codes**: INT4 packed eight to an int32, FP8
# bytes or FP4 nibbles. The second part is the **scales** (and zero points), which tell what the codes mean. The
# third part is a **`quantization_config`** in `config.json`, which tells the loader which is which. With
# `oneshot(model, recipe=...)`, llm-compressor writes all three. Then vLLM reads them without a `--quantization`
# flag.
#
# * `FP8_DYNAMIC` needs no data. It has FP8 weights with a scale for each output channel, and the engine
#   quantizes the activations per token at run time. `W4A16` needs calibration data *if* you want GPTQ or AWQ
#   instead of round-to-nearest (RTN). On a model with outlier activations, you want GPTQ or AWQ.
# * The recipes do not quantize everything. `lm_head`, the embeddings and the norms stay 16-bit
#   (`ignore=["lm_head"]` in every recipe). Thus a real INT4 checkpoint is larger than params x 4 bits.
# * Groups must divide the input width of the layer (`in_features % 128 == 0` for W4A16). If they do not, the
#   tools refuse the layer.
#
# Concepts: PRIMER §3 "Granularity and the bits-per-weight budget", §4 "Weight-only post-training
# quantization" and §9 "Producing a checkpoint" ([`PRIMER.md`](../../PRIMER.md)). The formats and error metrics
# that this notebook uses are in §8 of the serving-engine primer
# ([`../../../serving-engine/PRIMER.md`](../../../serving-engine/PRIMER.md)).

# %%
import atexit, json, pathlib, tempfile
from quantlab import compress as C, env, numerics as N, serve, stio, tinymodel as tm
import numpy as np

print(env.describe())
OUT = pathlib.Path(tempfile.mkdtemp(prefix="quantlab-01-"))   # removed at the end, or when the kernel exits
atexit.register(C.clean, OUT)
model = tm.load()
print(f"{model.num_params():,} parameters;", {k: model.config[k] for k in ("hidden_size", "intermediate_size",
      "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")})
for task in tm.TASKS:
    p, a = tm.make_task(task, 2, 1000)
    print(f"  {task:8s} e.g. {tm.decode(p[0]):12s} -> {tm.decode(a[0])}   accuracy {model.accuracy(task, 500):.1%} (500 held-out)")

# %% [markdown]
# ## Worked example: what a dense checkpoint is
#
# The safetensors header lists every tensor with its dtype and shape. The bytes come after the header. The
# fourteen projections (q, k, v, o, gate, up, down in two layers) are the tensors that quantization will make
# smaller.

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
# ignore=["lm_head"])`. Each projection becomes an `F8_E4M3` `weight` and a bf16 `weight_scale` of shape
# `[out, 1]` (one for each output channel, $\mathrm{amax}/448$). The engine calculates the activation scales per
# token at run time. Thus there is nothing to calibrate.

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
# `pack_to_int32` adds 8 to each signed code (-8..7 becomes 0..15). It puts eight of these codes in one 32-bit
# word, with **element 0 in the lowest four bits**. Write `pack_int4(q)` for an integer array whose last
# dimension is a multiple of 8. Return `int32` words. View the `uint32` result as `int32`, because the top
# nibble often sets the sign bit.

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
# W4A16 (groups of 128, symmetric) stores these tensors for each projection:
#
# * the packed codes (half a byte for each weight),
# * one bf16 scale for each group of 128 inputs in each output row,
# * `weight_shape` (two int64).
#
# Everything that is not a projection stays bf16. Write `predict_w4a16_bytes(model, group_size)`. Count the
# tensor data only, with no header. Then compare the ratio with the simple "4 bits instead of 16".

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
# The projection inputs of this model carry two massive-activation channels. The lab planted them after
# training. The RMSNorm weight of those channels is 24x larger, and the weight columns that match them are 24x smaller.
# The function stays the same. RTN rounds those small columns to almost nothing, and these columns multiply the
# largest inputs.
#
# GPTQ quantizes one column at a time. It moves the rounding error of each column onto the columns that it did
# not quantize yet. The inverse Hessian of the calibration inputs sets how much error each column receives.
# Thus GPTQ compensates elsewhere for the error that goes to the large-input channels. The layer error in the
# table of the next cell is $\lVert X\hat{W}^\top - XW^\top \rVert / \lVert XW^\top \rVert$ on calibration
# inputs.

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
# Look at two things. First, per layer: on the projections whose inputs carry the outlier channels (q/k/v,
# gate/up), g32 almost does not change the error of RTN. That error is 8-12% at either group size. A group is a run of *inputs
# within one output row*. In every group, the two columns that the lab made smaller are 18-24x smaller than their
# neighbours. Thus they round to zero at any group size.
#
# Second, end to end: g32 does even *worse* than g128 here. This is the opposite of the usual rule (finer groups
# cost bits and give accuracy, PRIMER §3). The next cell finds the cause.
#
# ## Worked example: why finer groups lost accuracy on this model
#
# The next cell does two experiments on the columns of q/k/v/gate/up that meet the outlier channels. The first experiment puts
# their bf16 weights back ("restored"). The second experiment forces every code in those columns to zero ("forced
# to 0"). The cell also counts how many of those codes are *not* zero.

# %%
out_ch = tm.outlier_channels(model)
hot = [n for n in model.linear_names() if n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")]

def outlier_columns(q, restore):
    """The quantized model with the outlier-meeting columns restored to bf16 (True) or forced to zero (False)."""
    w = dict(q.model().weights)
    for n in hot:
        W = w[n + ".weight"].copy()
        W[:, out_ch] = model.weights[n + ".weight"][:, out_ch] if restore else 0.0
        w[n + ".weight"] = W
    return tm.TinyLM(model.config, w)

acc = lambda m: "   ".join(f"{t} {m.accuracy(t, 500):6.1%}" for t in tm.TASKS)  # noqa: E731
for key in ("W4A16 (rtn)", "W4A16-g32 (rtn)"):
    q = runs[key]
    nz = sum(int((q.layers[n].codes[:, out_ch] != 0).sum()) for n in hot)
    total = sum(q.layers[n].codes[:, out_ch].size for n in hot)
    print(f"{key:16s} non-zero codes in those columns: {nz}/{total}")
    print(f"    as quantized  {acc(q.model())}\n    forced to 0   {acc(outlier_columns(q, False))}"
          f"\n    restored      {acc(outlier_columns(q, True))}")

# %% [markdown]
# When you put back two columns out of 128, RTN INT4 becomes lossless at either group size. Thus **all** of the
# loss of RTN on this model has one cause: RTN rounds those two columns to zero. That is, RTN deletes two
# channels of the residual stream from every attention and MLP input. Finer groups do decrease the error of the
# ordinary columns (per layer, in the table of the round-to-nearest versus GPTQ example). But that was never the problem.
#
# At g32, a small number of the small weights are just above half a step of their (smaller) group scale. These
# weights round *up* to one step instead of down to zero. Each of them is still ~90% incorrect, and now its sign
# is opposite to the sign of the deleted weights around it. Each of them multiplies an input ~24x the typical one
# inside q/k, and the error of q/k goes through the softmax. If you force those few codes back to zero, g32
# matches g128 exactly.
#
# Thus the ranking comes from the planted construction only. The massive-activation columns of a real model are
# not 24x smaller than their neighbours. They are usually ordinary (PRIMER §3, "Activation outliers are a
# different problem").
#
# The lesson that transfers is this: when the damage is in a few input channels, group size is the incorrect
# setting to adjust. The correction is per-*input-channel*. One correction is GPTQ (error compensation onto the
# other columns). The other is AWQ (scale the salient columns up, round them, and fold the inverse into the norm).
# Both get back nearly all of the loss at g128 (the table of the round-to-nearest versus GPTQ example).
#
# ## Exercise 1.3 — the symmetric INT4 group quantizer
#
# Write the symmetric INT4 grid of compressed-tensors for a `w` of shape `[out, in]`. For each output row and
# each run of `g` inputs, use $\mathrm{scale} = \mathrm{amax}/7.5$ and
# $\mathrm{code} = \operatorname{clip}(\operatorname{round}(w/\mathrm{scale}), -8, 7)$. Return
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
# A loader never sees `w_hat`. It gets the tensors of one projection in a `pack-quantized` checkpoint:
# `weight_packed` int32, `weight_scale` bf16 `[out, in/g]` and `weight_shape`. From these tensors, make the
# float weight again. Use `N.unpack_int32(words, 4, cols)` to unpack the codes. Use `.numpy()` to decode bf16.

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
# If the `in_features` of a projection is not a multiple of the group size, llm-compressor stops with an error at
# initialize. Marlin in vLLM also refuses the layer. q/k/v and gate/up read `hidden_size` inputs. o_proj reads
# $\mathrm{num\_heads} \times \mathrm{head\_dim}$, and down_proj reads `intermediate_size`. Write
# `bad_projections(config, g)`. It must return the sorted names (`"q_proj"`, ... `"down_proj"`) that fail.

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
# The lab makes the llm-compressor 0.14 script for each recipe. Run the script in its own environment, because
# the vLLM docs recommend separate environments for vLLM and llm-compressor. `deploy/any-gpu/compress.sh` does
# exactly this on a GPU box. The result is a directory that vLLM serves with no `--quantization` flag.
# `serve.check_checkpoint` tells what vLLM will do with the directory on each GPU.

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
if (OUT / "Qwen2.5-0.5B-Instruct-FP8_DYNAMIC").exists():
    atexit.unregister(C.clean)
    print(f"the real checkpoint is in {OUT}; serve it from there, then delete the directory")
else:
    C.clean(OUT)                                     # the tiny checkpoints above lived in a temporary directory

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We ship quantized checkpoints in the compressed-tensors format, because vLLM finds the
# format automatically. The `quantization_config` names the scheme, and the tensors carry the codes and the
# scales. Thus nobody has to remember a `--quantization` flag. For FP8 we use `FP8_DYNAMIC`. It has per-channel
# weight scales and per-token activation scales at run time, and it needs no calibration data.
#
# "For INT4 we use GPTQ with calibration data from our own traffic. The reason is that round-to-nearest loses
# the weight columns that meet our largest activations. On the lab's model, RTN leaves 11% error on the q
# projection, and GPTQ leaves under 2%. The LM head and the embeddings stay bf16. Thus the checkpoint is ~3x
# smaller, not 4x. Before we start, we make sure that the input width of every projection divides the group
# size."
#
# **Drill 1.** *Why does `FP8_DYNAMIC` need no calibration data but `W4A16` with GPTQ does?* FP8 dynamic
# calculates the weight scales from the weights, and it calculates the activation scales per token at run time.
# The error compensation of GPTQ needs the Hessian $X^\top X$ of real inputs to each layer.
#
# **Drill 2.** *The INT4 checkpoint of a 0.5B model is only 2.2x smaller. Is the tool broken?* No. Qwen2.5-0.5B
# ties a 136 M-parameter embedding that stays bf16 (27.6% of the model). Also, there are 16/128 bits of scale for
# each weight. Large models come near 3.5-3.9x, but small models do not.
#
# **Drill 3.** *llm-compressor refuses `W4A16` on one model and not another. Why?* The group size must divide
# `in_features`. 576-wide models fail g128 and need g64/g32. Also, Marlin in vLLM only accepts -1, 32, 64, 128.
