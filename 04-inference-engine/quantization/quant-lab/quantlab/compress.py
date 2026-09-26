"""compress.py — turn a bf16 checkpoint into a compressed-tensors checkpoint, then read it back.

One idea: a quantized checkpoint is three things — low-bit codes, the scales (and zero points)
that say what the codes mean, and a ``quantization_config`` in ``config.json`` that tells the
loader which is which. llm-compressor writes all three with ``oneshot(model, recipe=...)``; this
module does the same for the bundled tiny model with numpy, so the whole path is visible at T0:

    recipe  = Recipe("W4A16", algorithms=("gptq",))            # llm-compressor: GPTQModifier(scheme="W4A16")
    report  = quantize_checkpoint(TINY_DIR, "out/tiny-W4A16", recipe)
    model   = load_checkpoint("out/tiny-W4A16")                 # dequantizes, like a loader
    problems = validate_checkpoint("out/tiny-W4A16")            # [] when the layout is right

The presets mirror compressed-tensors' (``quant_scheme.py``): ``FP8_DYNAMIC`` (FP8 weights per
output channel + dynamic per-token FP8 activations: no calibration data), ``FP8`` (per-tensor
weights + a static activation scale from calibration), ``W8A8`` (INT8 channel + dynamic token),
``W8A16``, ``W4A16`` (INT4, groups of 128, symmetric), ``W4A16_ASYM`` (+ zero points) and
``NVFP4A16`` (FP4 E2M1, an E4M3 scale per 16 plus a global scale). The tensors are named, typed,
shaped and packed as compressed-tensors does (``weight_packed`` int32 with element 0 in the low
bits, ``weight_scale`` ``[out, in/group]``, ``weight_shape``, ``weight_zero_point`` packed along
the output dimension). The algorithms (GPTQ, AWQ, SmoothQuant) are compact versions of the ones
PRIMER §4 "Weight-only post-training quantization" derives; the topic's core teaches them step by
step. ``llmcompressor_script`` renders the same recipe for a real model at T1.
"""
from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fp4, numerics as N, stio
from .tinymodel import TASKS, TinyLM, load, make_task

# ---------------------------------------------------------------------------------------------
# Schemes (compressed-tensors QuantizationArgs, as they appear in config.json)
# ---------------------------------------------------------------------------------------------
def _args(num_bits, type_, strategy, *, symmetric=True, group_size=None, dynamic=False, observer="minmax",
          scale_dtype=None):
    return {"num_bits": num_bits, "type": type_, "symmetric": symmetric, "group_size": group_size,
            "strategy": strategy, "block_structure": None, "dynamic": dynamic, "actorder": None,
            "scale_dtype": scale_dtype, "zp_dtype": None, "observer": None if dynamic is True else observer,
            "observer_kwargs": {}}


SCHEMES = {
    "FP8_DYNAMIC": {"weights": _args(8, "float", "channel"),
                    "input_activations": _args(8, "float", "token", dynamic=True)},
    "FP8": {"weights": _args(8, "float", "tensor"),
            "input_activations": _args(8, "float", "tensor", observer="static_minmax")},
    "W8A8": {"weights": _args(8, "int", "channel"), "input_activations": _args(8, "int", "token", dynamic=True)},
    "W8A16": {"weights": _args(8, "int", "channel"), "input_activations": None},
    "W4A16": {"weights": _args(4, "int", "group", group_size=128), "input_activations": None},
    "W4A16_ASYM": {"weights": _args(4, "int", "group", symmetric=False, group_size=128), "input_activations": None},
    "NVFP4A16": {"weights": _args(4, "float", "tensor_group", group_size=16, scale_dtype="torch.float8_e4m3fn"),
                 "input_activations": None},
    "NVFP4": {"weights": _args(4, "float", "tensor_group", group_size=16, scale_dtype="torch.float8_e4m3fn"),
              "input_activations": _args(4, "float", "tensor_group", group_size=16, dynamic="local",
                                         scale_dtype="torch.float8_e4m3fn", observer="static_minmax")},
}


def compression_format(scheme: dict) -> str:
    """compressed-tensors' choice (``infer_module_format`` priority): int W8A8 -> int-quantized,
    int weight-only -> pack-quantized, float W8A8 -> float-quantized, FP4 groups of 16 -> nvfp4-pack-quantized."""
    w, a = scheme["weights"], scheme["input_activations"]
    if w["type"] == "float" and w["num_bits"] == 4:
        return "nvfp4-pack-quantized"
    if a is not None:
        return "int-quantized" if w["type"] == "int" else "float-quantized"
    return "pack-quantized" if w["type"] == "int" else "naive-quantized"


@dataclass
class Recipe:
    """What to do to which layers. ``algorithms``: ``()`` = round to nearest (RTN), or any of
    ``"awq"`` / ``"smoothquant"`` (transforms, applied first) and ``"gptq"`` (the weight solver)."""
    scheme: str = "W4A16"
    algorithms: tuple = ()
    ignore: tuple = ("lm_head",)
    group_size: int | None = None                # override the preset's (W4A16: 128)
    smoothing_strength: float = 0.8              # SmoothQuantModifier(smoothing_strength=0.8) in the W8A8 example
    dampening_frac: float = 0.01                 # GPTQModifier default
    block_size: int = 128                        # GPTQ lazy-batch block
    num_calibration_samples: int = 128
    awq_grid: int = 20

    def scheme_args(self) -> dict:
        s = json.loads(json.dumps(SCHEMES[self.scheme]))
        if self.group_size is not None and s["weights"]["strategy"] in ("group", "tensor_group"):
            s["weights"]["group_size"] = self.group_size
        return s

    @property
    def needs_calibration(self) -> bool:
        """GPTQ/AWQ/SmoothQuant and static activation scales need data; RTN weights and dynamic activations do not."""
        a = self.scheme_args()["input_activations"]
        return bool(self.algorithms) or (a is not None and a["dynamic"] is not True)

    @property
    def name(self) -> str:
        g = self.scheme_args()["weights"]["group_size"]
        alg = "+".join(self.algorithms) or "rtn"
        return f"{self.scheme}{'-g' + str(g) if self.group_size else ''} ({alg})"


# ---------------------------------------------------------------------------------------------
# Quantizers: scales, codes, dequantization (compressed-tensors' conventions)
# ---------------------------------------------------------------------------------------------
def _qrange(args):
    if args["type"] == "int":
        b = args["num_bits"]
        return -(1 << (b - 1)), (1 << (b - 1)) - 1
    return (-448.0, 448.0) if args["num_bits"] == 8 else (-6.0, 6.0)


def qparams(w, args):
    """Scale (and zero point) for a block ``w [rows, cols]`` -> per-row arrays ``[rows, 1]``.
    Symmetric INT: ``amax / ((qmax - qmin) / 2)`` (INT4: amax/7.5, INT8: amax/127.5); asymmetric:
    ``(max - min) / (qmax - qmin)`` with ``zp = clamp(qmin - min/scale)``; FP8: ``amax / 448``.
    Zero is always representable (min <= 0 <= max)."""
    lo, hi = _qrange(args)
    mn, mx = np.minimum(w.min(1, keepdims=True), 0), np.maximum(w.max(1, keepdims=True), 0)
    if args["symmetric"]:
        scale = np.maximum(np.maximum(-mn, mx), 1e-12) / ((hi - lo) / 2)
        return scale, None
    scale = np.maximum(mx - mn, 1e-12) / (hi - lo)
    return scale, np.clip(np.round(lo - mn / scale), lo, hi)


def quantize_block(w, scale, zp, args):
    lo, hi = _qrange(args)
    if args["type"] == "float":
        return N.minifloat_round(np.clip(w / scale, lo, hi), "e4m3")
    q = np.round(w / scale) + (0 if zp is None else zp)
    return np.clip(q, lo, hi)


def dequantize_block(q, scale, zp):
    return (q - (0 if zp is None else zp)) * scale


def _groups(args, cols):
    s = args["strategy"]
    if s in ("group", "tensor_group"):
        g = args["group_size"]
        if cols % g:
            raise ValueError(f"in_features {cols} is not divisible by group_size {g} "
                             "(llm-compressor rejects this layer at initialize; vLLM's Marlin would too)")
        return g
    return cols


def rtn(w, args):
    """Round-to-nearest. Returns ``(codes [out, in], scale [out, groups], zp or None, w_hat)``."""
    w = np.asarray(w, dtype=np.float64)
    if args["num_bits"] == 4 and args["type"] == "float":
        q = fp4.nvfp4_quantize(w, group=args["group_size"])
        return q.values, q.scales, None, q.dequantize()
    if args["strategy"] == "tensor":
        scale, zp = qparams(w.reshape(1, -1), args)
        q = quantize_block(w, scale, zp, args)
        return q, scale.reshape(1), zp, dequantize_block(q, scale, zp)
    g = _groups(args, w.shape[1])
    blocks = w.reshape(w.shape[0], -1, g)
    scales, zps, codes = [], [], []
    for k in range(blocks.shape[1]):
        s, z = qparams(blocks[:, k], args)
        codes.append(quantize_block(blocks[:, k], s, z, args))
        scales.append(s)
        zps.append(z)
    q = np.concatenate(codes, 1)
    scale = np.concatenate(scales, 1)
    zp = None if zps[0] is None else np.concatenate(zps, 1)
    w_hat = dequantize_block(q.reshape(blocks.shape), scale[:, :, None],
                             None if zp is None else zp[:, :, None]).reshape(w.shape)
    return q, scale, zp, w_hat


def gptq(w, H, args, *, block_size: int = 128, dampening_frac: float = 0.01):
    """GPTQ (Frantar et al.): quantize one input column at a time and push its error onto the
    columns not yet quantized, weighted by the inverse Hessian ``H = 2/n sum x x^T`` of the
    layer's calibration inputs (the reference ``fasterquant`` loop, with lazy block updates).
    Group scales are found on weights already updated by the previous blocks, as in the reference code."""
    W = np.array(w, dtype=np.float64)
    rows, cols = W.shape
    g = _groups(args, cols) if args["strategy"] in ("group", "tensor_group") else cols
    H = np.array(H, dtype=np.float64)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H[np.diag_indices(cols)] += dampening_frac * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T           # upper Cholesky factor of H^-1
    Q = np.zeros_like(W)
    scales, zps = [], []
    scale = zp = None
    if args["strategy"] in ("channel", "tensor"):
        scale, zp = qparams(W if args["strategy"] == "channel" else W.reshape(1, -1), args)
        scales, zps = [scale], [zp]
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W1, Err = W[:, i1:i2].copy(), np.zeros((rows, i2 - i1))
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if args["strategy"] in ("group", "tensor_group") and col % g == 0:
                scale, zp = qparams(W[:, col:col + g], args)     # W's block is updated only after the block
                scales.append(scale)
                zps.append(zp)
            q = quantize_block(W1[:, i:i + 1], scale, zp, args)
            Q[:, col] = q[:, 0]
            err = (W1[:, i] - dequantize_block(q, scale, zp)[:, 0]) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err[:, i] = err
        W[:, i1:i2] = W1
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    scale = np.concatenate(scales, 1) if args["strategy"] != "tensor" else scales[0].reshape(1)
    zp = None if zps[0] is None else np.concatenate(zps, 1)
    gg = g if args["strategy"] in ("group", "tensor_group") else cols
    w_hat = dequantize_block(Q.reshape(rows, -1, gg), np.asarray(scale).reshape(rows if scale.size > 1 else 1, -1, 1),
                             None if zp is None else zp[:, :, None]).reshape(rows, cols)
    return Q, scale, zp, w_hat


# ---------------------------------------------------------------------------------------------
# Transforms that fold a per-channel scale into the previous op (AWQ, SmoothQuant)
# ---------------------------------------------------------------------------------------------
def mappings(model: TinyLM) -> list:
    """``(previous op, [consumers])`` pairs whose shared input channel can be rescaled exactly:
    llm-compressor's default Llama mappings (``v_proj -> o_proj`` is skipped: with grouped-query
    attention the shapes differ)."""
    out = []
    for i in range(model.layers):
        p = f"model.layers.{i}."
        out += [(p + "input_layernorm", [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"]),
                (p + "post_attention_layernorm", [p + "mlp.gate_proj", p + "mlp.up_proj"]),
                (p + "mlp.up_proj", [p + "mlp.down_proj"])]
    return out


def _fold(weights: dict, prev: str, consumers: list, s) -> None:
    """X' = X / s, W' = W * s: the product is unchanged. For a norm, divide its weight; for a
    projection feeding another (up -> down), divide its output rows."""
    w = weights[prev + ".weight"]
    weights[prev + ".weight"] = w / s if w.ndim == 1 else w / s[:, None]
    for c in consumers:
        weights[c + ".weight"] = weights[c + ".weight"] * s[None, :]


def smoothquant(model: TinyLM, calib: dict, alpha: float = 0.8) -> dict:
    """SmoothQuant: ``s_j = max|X_j|^alpha / max|W_j|^(1-alpha)`` per input channel, folded into the
    norm before q/k/v and gate/up. Returns the new weights (the model's function is unchanged)."""
    w = dict(model.weights)
    for prev, cons in mappings(model):
        if not prev.endswith("layernorm"):
            continue
        x_max = np.abs(np.concatenate(calib[cons[0]])).max(0)
        w_max = np.max([np.abs(w[c + ".weight"]).max(0) for c in cons], axis=0)
        s = np.clip(np.maximum(x_max, 1e-5) ** alpha / np.maximum(w_max, 1e-5) ** (1 - alpha), 1e-5, None)
        _fold(w, prev, cons, s)
    return w


def awq(model: TinyLM, calib: dict, args: dict, n_grid: int = 20) -> dict:
    """AWQ: per mapping, search ``s = mean|X|^r`` (r = 0, 1/20, ..., 19/20, normalised by
    sqrt(max * min)) for the one whose quantized output ``(X/s) Q(W s)^T`` is closest to ``X W^T``,
    then fold it in. Salient input channels (large activations) get their weights scaled up so
    rounding hurts them less. (Clipping search and the 'duo' variant are left out.)"""
    w = dict(model.weights)
    for prev, cons in mappings(model):
        X = np.concatenate(calib[cons[0]])[:2048]
        x_mean = np.abs(X).mean(0)
        ref = np.concatenate([X @ w[c + ".weight"].T for c in cons], 1)
        best, best_s = np.inf, np.ones_like(x_mean)
        for k in range(n_grid):
            s = np.clip(x_mean ** (k / n_grid), 1e-4, None)
            s = s / np.sqrt(s.max() * s.min())
            out = np.concatenate([(X / s) @ rtn(w[c + ".weight"] * s[None, :], args)[3].T for c in cons], 1)
            err = float(np.mean((out - ref) ** 2))
            if err < best:
                best, best_s = err, s
        _fold(w, prev, cons, best_s)
    return w


# ---------------------------------------------------------------------------------------------
# Calibration and the whole-model pass
# ---------------------------------------------------------------------------------------------
def calibration_inputs(model: TinyLM, n: int = 128, seed: int = 7) -> dict:
    """Each projection's input rows over ``n`` problems per task (prompt + answer, as a chat
    calibration set would be). Different seeds from the held-out eval (>= 1000)."""
    cap: dict = {}
    for task in TASKS:
        p, a = make_task(task, n, seed)
        model.answer_logits(p, a, capture=cap)
    return cap


@dataclass
class LayerResult:
    name: str
    codes: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray | None
    w_hat: np.ndarray
    out_err: float                 # relative error of the layer output on calibration inputs
    global_scale: float | None = None   # NVFP4 only: the per-tensor multiplier 448*6/amax


@dataclass
class QuantizedModel:
    recipe: Recipe
    base: TinyLM                   # after transforms (AWQ/SmoothQuant fold scales into the weights)
    layers: dict = field(default_factory=dict)
    input_scales: dict = field(default_factory=dict)   # static activation scales (FP8 preset)

    def model(self) -> TinyLM:
        """The dequantized model, with activation fake-quantization for W8A8 schemes."""
        return self.base.with_weights({k + ".weight": v.w_hat for k, v in self.layers.items()})

    def act_quant(self):
        return act_quant_fn(self.recipe.scheme_args()["input_activations"], self.input_scales)


def act_quant_fn(args: dict | None, static_scales: dict | None = None):
    """``(name, x) -> x_hat`` emulating the scheme's activation quantization, or None (weight-only)."""
    if args is None:
        return None

    if args["type"] == "float" and args["num_bits"] == 4:   # NVFP4 activations: E4M3 scale per 16, a global scale
        def f4(name, x):
            flat = x.reshape(-1, x.shape[-1])
            gs = (static_scales or {}).get(name)            # calibrated input_global_scale, else from this batch
            q = fp4.nvfp4_quantize(flat, group=args["group_size"], global_scale=gs)
            return q.dequantize().reshape(x.shape)
        return f4

    def f(name, x):
        if args["strategy"] == "token":            # dynamic: one scale per token (row), at run time
            amax = np.abs(x).max(-1, keepdims=True)
        else:                                      # static per tensor, from calibration
            amax = np.asarray((static_scales or {}).get(name, np.abs(x).max() / 448) * 448.0)
        if args["type"] == "int":
            s = np.maximum(amax, 1e-12) / 127.5
            return np.clip(np.round(x / s), -128, 127) * s
        s = np.maximum(amax, 1e-12) / 448.0
        return N.minifloat_round(np.clip(x / s, -448, 448), "e4m3") * s
    return f


def quantize_model(model: TinyLM, recipe: Recipe, calib: dict | None = None) -> QuantizedModel:
    """Apply the recipe to every projection not in ``recipe.ignore`` (``lm_head``, embeddings and
    norms are never quantized here, as in every llm-compressor example)."""
    s = recipe.scheme_args()
    if calib is None:              # RTN schemes do not need it; it is used here to report each layer's error
        calib = calibration_inputs(model, recipe.num_calibration_samples)
    base = model
    if "smoothquant" in recipe.algorithms:
        base = TinyLM(model.config, smoothquant(model, calib, recipe.smoothing_strength))
    if "awq" in recipe.algorithms:
        base = TinyLM(model.config, awq(model, calib, s["weights"], recipe.awq_grid))
    if base is not model:                          # inputs changed: re-capture on the transformed model
        calib = calibration_inputs(base, recipe.num_calibration_samples)
    qm = QuantizedModel(recipe, base)
    for name in base.linear_names():
        if any(name.endswith(ig) for ig in recipe.ignore):
            continue
        W = base.weights[name + ".weight"]
        if "gptq" in recipe.algorithms and s["weights"]["type"] == "int":
            X = np.concatenate(calib[name])
            codes, scale, zp, w_hat = gptq(W, 2 * X.T @ X / len(X), s["weights"], block_size=recipe.block_size,
                                           dampening_frac=recipe.dampening_frac)
        else:
            codes, scale, zp, w_hat = rtn(W, s["weights"])
        gscale = None
        if s["weights"]["type"] == "float" and s["weights"]["num_bits"] == 4:
            gscale = fp4.nvfp4_quantize(W, group=s["weights"]["group_size"]).global_scale
        X = np.concatenate(calib[name])[:1024]
        ref = X @ W.T
        err = float(np.linalg.norm(X @ w_hat.T - ref) / np.linalg.norm(ref))
        qm.layers[name] = LayerResult(name, codes, np.asarray(scale), zp, w_hat, err, gscale)
    a = s["input_activations"]
    if a is not None and a["dynamic"] is False:            # FP8 static: input_scale = amax / 448
        qm.input_scales = {n: float(np.abs(np.concatenate(calib[n])).max() / 448.0) for n in qm.layers}
    elif a is not None and a["dynamic"] == "local":        # NVFP4: input_global_scale = 448 * 6 / amax
        qm.input_scales = {n: fp4.nvfp4_global_scale(np.abs(np.concatenate(calib[n])).max()) for n in qm.layers}
    return qm


# ---------------------------------------------------------------------------------------------
# Writing and reading the checkpoint
# ---------------------------------------------------------------------------------------------
def quantization_config(recipe: Recipe) -> dict:
    s = recipe.scheme_args()
    fmt = compression_format(s)
    return {"config_groups": {"group_0": {"targets": ["Linear"], "weights": s["weights"],
                                          "input_activations": s["input_activations"], "output_activations": None,
                                          "format": fmt}},
            "format": fmt, "global_compression_ratio": None, "ignore": list(recipe.ignore), "kv_cache_scheme": None,
            "quant_method": "compressed-tensors", "quantization_status": "compressed",
            "sparsity_config": {}, "transform_config": {},
            "version": "quantlab 0.1 (compressed-tensors 0.19 layout)"}


def layer_tensors(name: str, lr: LayerResult, args: dict, fmt: str, input_scale: float | None = None) -> dict:
    """The tensors one quantized Linear becomes, named and typed as compressed-tensors does."""
    out, rows, cols = {}, lr.w_hat.shape[0], lr.w_hat.shape[1]
    if fmt == "pack-quantized":
        out[name + ".weight_packed"] = N.pack_int32(lr.codes.astype(np.int64), args["num_bits"])
        out[name + ".weight_shape"] = np.array([rows, cols], dtype=np.int64)
        if lr.zero_point is not None:           # packed along the output dimension (packed_dim=0)
            out[name + ".weight_zero_point"] = N.pack_int32(lr.zero_point.astype(np.int64).T, args["num_bits"]).T.copy()
    elif fmt == "int-quantized":
        out[name + ".weight"] = lr.codes.astype(np.int8)
    elif fmt == "float-quantized":
        out[name + ".weight"] = stio.Tensor("F8_E4M3", N.fp8_encode(lr.codes, "e4m3"))
    elif fmt == "nvfp4-pack-quantized":
        q = fp4.NVFP4Tensor(lr.codes, lr.scale, lr.global_scale, args["group_size"])
        out.update({name + k: v for k, v in fp4.nvfp4_checkpoint_tensors(q).items()})
        if input_scale is not None:
            out[name + ".input_global_scale"] = stio.Tensor("F32", np.array([input_scale], dtype=np.float32))
        return out
    else:
        raise ValueError(f"format {fmt} not written by this lab")
    out[name + ".weight_scale"] = stio.bf16(np.asarray(lr.scale).reshape(rows, -1) if lr.scale.size > 1
                                            else np.asarray(lr.scale).reshape(1))
    if input_scale is not None:
        out[name + ".input_scale"] = stio.bf16(np.array([input_scale]))
    return out


def save_checkpoint(qm: QuantizedModel, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = qm.recipe.scheme_args()
    fmt = compression_format(s)
    tensors = {}
    for k, v in qm.base.weights.items():
        if k[: -len(".weight")] not in qm.layers:
            tensors[k] = stio.bf16(v)                 # embeddings, norms, lm_head: kept in bf16
    for name, lr in qm.layers.items():
        tensors.update(layer_tensors(name, lr, s["weights"], fmt, qm.input_scales.get(name)))
    stio.save(tensors, out / "model.safetensors", metadata={"format": "pt", "producer": "quantlab.compress"})
    cfg = {**qm.base.config, "quantization_config": quantization_config(qm.recipe)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out


def quantize_checkpoint(src, out_dir, recipe: Recipe) -> dict:
    """Load a dense checkpoint, apply ``recipe``, write the compressed checkpoint; return a report."""
    model = load(src)
    qm = quantize_model(model, recipe)
    path = save_checkpoint(qm, out_dir)
    return {"recipe": recipe.name, "path": str(path), "bytes": (path / "model.safetensors").stat().st_size,
            "dense_bytes": (Path(src) / "model.safetensors").stat().st_size,
            "layer_errors": {k: v.out_err for k, v in qm.layers.items()}, "quantized": qm}


def _dequant_layer(t: dict, name: str, group: dict, fmt: str) -> np.ndarray:
    args = group["weights"]
    if fmt == "pack-quantized":
        rows, cols = (int(v) for v in t[name + ".weight_shape"].numpy())
        q = N.unpack_int32(t[name + ".weight_packed"].numpy(), args["num_bits"], cols).astype(np.float64)
        scale = t[name + ".weight_scale"].numpy().astype(np.float64)
        zp = None
        if name + ".weight_zero_point" in t:
            zp = N.unpack_int32(t[name + ".weight_zero_point"].numpy().T.copy(), args["num_bits"], rows).T.astype(np.float64)
        g = cols // scale.shape[1]
        return dequantize_block(q.reshape(rows, -1, g), scale[:, :, None],
                                None if zp is None else zp[:, :, None]).reshape(rows, cols)
    if fmt in ("int-quantized", "float-quantized", "naive-quantized"):
        q = t[name + ".weight"].numpy().astype(np.float64)
        return q * t[name + ".weight_scale"].numpy().astype(np.float64).reshape(-1, 1) \
            if t[name + ".weight_scale"].data.size > 1 else q * float(t[name + ".weight_scale"].numpy()[0])
    if fmt == "nvfp4-pack-quantized":
        return fp4.nvfp4_from_checkpoint({k[len(name):]: v for k, v in t.items() if k.startswith(name + ".")}).dequantize()
    raise ValueError(f"unsupported format {fmt}")


def load_checkpoint(path) -> tuple:
    """Read a compressed-tensors checkpoint written by this lab (or any checkpoint with the same
    layouts): returns ``(TinyLM with dequantized weights, act_quant hook or None)``."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    qc = cfg.get("quantization_config")
    t = stio.load(path / "model.safetensors")
    dense = {k: v.numpy().astype(np.float64) for k, v in t.items() if k.endswith(".weight") and v.dtype == "BF16"}
    if not qc:
        return TinyLM(cfg, dense), None
    group = qc["config_groups"]["group_0"]
    fmt = group.get("format") or qc["format"]
    layers = sorted({k.rsplit(".", 1)[0] for k in t if k.endswith((".weight_packed", ".weight_scale"))})
    for name in layers:
        dense[name + ".weight"] = _dequant_layer(t, name, group, fmt)
    static = {n: float(t[n + suffix].numpy()[0]) for n in layers for suffix in (".input_scale", ".input_global_scale")
              if n + suffix in t}
    return TinyLM({k: v for k, v in cfg.items() if k != "quantization_config"}, dense), \
        act_quant_fn(group.get("input_activations"), static)


def validate_checkpoint(path) -> list:
    """The checks a loader makes, as a list of problems (empty = consistent): every targeted
    projection has the tensors its format needs with the right dtypes and shapes, ignored modules
    are dense, group sizes divide ``in_features``, ``quant_method`` is compressed-tensors."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    qc, problems = cfg.get("quantization_config") or {}, []
    if qc.get("quant_method") != "compressed-tensors":
        return ["quantization_config.quant_method is not 'compressed-tensors'"]
    t = stio.load(path / "model.safetensors")
    group = qc["config_groups"]["group_0"]
    args, fmt = group["weights"], group.get("format") or qc["format"]
    n_layers, per = int(cfg["num_hidden_layers"]), fmt
    for i in range(n_layers):
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            n = f"model.layers.{i}.{proj}"
            if any(n.endswith(ig) for ig in qc.get("ignore", [])):
                continue
            need = {"pack-quantized": {"weight_packed": "I32", "weight_scale": None, "weight_shape": "I64"},
                    "int-quantized": {"weight": "I8", "weight_scale": None},
                    "float-quantized": {"weight": "F8_E4M3", "weight_scale": None},
                    "nvfp4-pack-quantized": {"weight_packed": "U8", "weight_scale": "F8_E4M3",
                                             "weight_global_scale": "F32"}}[per]
            for suffix, dt in need.items():
                k = f"{n}.{suffix}"
                if k not in t:
                    problems.append(f"{k} missing")
                elif dt and t[k].dtype != dt:
                    problems.append(f"{k} is {t[k].dtype}, expected {dt}")
            if f"{n}.weight" in t and per in ("pack-quantized", "nvfp4-pack-quantized"):
                problems.append(f"{n}.weight present next to weight_packed")
            if per == "pack-quantized" and f"{n}.weight_shape" in t:
                rows, cols = (int(v) for v in t[f"{n}.weight_shape"].numpy())
                g = args["group_size"] if args["strategy"] == "group" else cols
                if cols % g:
                    problems.append(f"{n}: in_features {cols} not divisible by group_size {g}")
                if t[f"{n}.weight_packed"].shape != (rows, cols * args["num_bits"] // 32):
                    problems.append(f"{n}.weight_packed shape {t[f'{n}.weight_packed'].shape}")
                if t[f"{n}.weight_scale"].shape != (rows, cols // g):
                    problems.append(f"{n}.weight_scale shape {t[f'{n}.weight_scale'].shape}, expected {(rows, cols // g)}")
                if (not args["symmetric"]) != (f"{n}.weight_zero_point" in t):
                    problems.append(f"{n}: weight_zero_point must exist iff symmetric is false")
    for ig in qc.get("ignore", []):
        if f"{ig}.weight" in t and t[f"{ig}.weight"].dtype not in ("BF16", "F16", "F32"):
            problems.append(f"{ig} is ignored but not dense")
    return problems


def checkpoint_bytes(path) -> dict:
    """Bytes by role, from the safetensors header alone: codes, scales/zero points, dense tensors."""
    roles = {"codes": 0, "scales": 0, "dense": 0}
    for name, dtype, shape, nbytes in stio.summary(Path(path) / "model.safetensors"):
        if name.endswith((".weight_packed",)) or (name.endswith(".weight") and dtype in ("I8", "F8_E4M3")):
            roles["codes"] += nbytes
        elif any(name.endswith(s) for s in ("_scale", "_zero_point", "_shape", "_global_scale")):
            roles["scales"] += nbytes
        else:
            roles["dense"] += nbytes
    roles["total"] = sum(roles.values())
    return roles


# ---------------------------------------------------------------------------------------------
# T1: the same recipes with llm-compressor / GPTQModel on a real model
# ---------------------------------------------------------------------------------------------
LLMC_DATA = {"W4A16": ("ultrachat_200k", 256, 1024), "W4A16_ASYM": ("ultrachat_200k", 256, 512),
             "W8A8": ("ultrachat_200k", 256, 1024), "FP8": ("ultrachat_200k", 64, 1024), "NVFP4": ("ultrachat_200k", 20, 2048)}


def llmcompressor_script(recipe: Recipe, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", out_dir: str | None = None) -> str:
    """Python source for llm-compressor 0.14 (run it in its own environment, not vLLM's). FP8_DYNAMIC
    and NVFP4A16 need no data; GPTQ/AWQ/SmoothQuant and static activation scales do."""
    out_dir = out_dir or model_id.split("/")[-1] + "-" + recipe.scheme + ("-" + "-".join(recipe.algorithms) if recipe.algorithms else "")
    mods = []
    if "smoothquant" in recipe.algorithms:
        mods.append(f"SmoothQuantModifier(smoothing_strength={recipe.smoothing_strength})")
    if "awq" in recipe.algorithms:
        mods.append("AWQModifier(duo_scaling=\"both\")")
    ignore = list(recipe.ignore)
    if "gptq" in recipe.algorithms:
        mods.append(f"GPTQModifier(targets=\"Linear\", scheme=\"{recipe.scheme}\", ignore={ignore!r})")
    else:
        mods.append(f"QuantizationModifier(targets=\"Linear\", scheme=\"{recipe.scheme}\", ignore={ignore!r})")
    imports = ["from llmcompressor import oneshot"]
    if any(m.startswith("GPTQ") for m in mods):
        imports.append("from llmcompressor.modifiers.gptq import GPTQModifier")
    if any(m.startswith("Quantization") for m in mods):
        imports.append("from llmcompressor.modifiers.quantization import QuantizationModifier")
    if any(m.startswith("AWQ") for m in mods):
        imports.append("from llmcompressor.modifiers.transform.awq import AWQModifier")
    if any(m.startswith("SmoothQuant") for m in mods):
        imports.append("from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier")
    needs_data = recipe.needs_calibration
    ds, n, seqlen = LLMC_DATA.get(recipe.scheme, ("ultrachat_200k", 256, 1024))
    call = [f"# calibration: {n} samples x {seqlen} tokens (the llm-compressor examples use up to 512 x 2,048;",
            "# fewer is faster on a small GPU — check the accuracy, verify the dataset split name)",
            f'oneshot(model=model, recipe=recipe, dataset="{ds}", splits="train_sft[:{n}]",',
            f"        num_calibration_samples={n}, max_seq_length={seqlen})"] if needs_data else \
        ["oneshot(model=model, recipe=recipe)          # no calibration data needed"]
    lines = ["# llm-compressor 0.14 (verify): pip install llmcompressor  -- in an environment without vLLM",
             "from transformers import AutoModelForCausalLM, AutoTokenizer", *imports, "",
             f'MODEL_ID = "{model_id}"',
             'model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")',
             "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)",
             f"recipe = [{', '.join(mods)}]", *call,
             f'model.save_pretrained("{out_dir}", save_compressed=True)',
             f'tokenizer.save_pretrained("{out_dir}")',
             f'print("wrote {out_dir}; serve it with: vllm serve ./{out_dir}")']
    return "\n".join(lines) + "\n"


def gptqmodel_script(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", bits: int = 4, group_size: int = 128) -> str:
    """The GPTQModel equivalent (7.5, verify) — its own kernels and formats; vLLM loads the result as GPTQ."""
    return textwrap.dedent(f"""\
        # GPTQModel 7.5 (verify): pip install -U gptqmodel --no-build-isolation
        from datasets import load_dataset
        from gptqmodel import GPTQModel, QuantizeConfig
        calibration = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
                                   split="train").select(range(256))["text"]
        model = GPTQModel.load("{model_id}", QuantizeConfig(bits={bits}, group_size={group_size}))
        model.quantize(calibration, batch_size=2)
        model.save("{model_id.split('/')[-1]}-gptq-{bits}bit-g{group_size}")
        """)


def run_llmcompressor(recipe: Recipe, model_id: str, out_dir: str) -> str:
    """T1: execute :func:`llmcompressor_script` when llm-compressor and a GPU are present (imports lazily)."""
    import importlib.util
    if importlib.util.find_spec("llmcompressor") is None:
        raise RuntimeError("llmcompressor is not installed: pip install llmcompressor (a separate env from vLLM)")
    src = llmcompressor_script(recipe, model_id, out_dir)
    exec(compile(src, "<llmcompressor recipe>", "exec"), {})   # noqa: S102 — our own generated source
    return out_dir


def clean(path) -> None:
    shutil.rmtree(path, ignore_errors=True)
