"""tinymodel.py — a real (if tiny) Llama-architecture checkpoint to quantize, serve in numpy and grade.

One idea: to *see* what quantization costs you need a model with a task it can fail. The bundled
checkpoint (``data/tiny-adder/``: a Hugging Face-style ``config.json`` + ``model.safetensors`` in
bf16, 299,648 parameters) is a 2-layer Llama — RMSNorm, RoPE, grouped-query attention, SwiGLU, an
untied LM head, the same tensor names as ``LlamaForCausalLM`` — trained by ``tools/train_tiny.py``
on two tasks with exact answers:

* ``add``:     ``457+389=`` -> ``6480`` (the 4-digit sum, least-significant digit first) — carries
  make it fragile, the way arithmetic and long reasoning chains are fragile in large models;
* ``reverse``: ``R381204=`` -> ``402183`` — an attention pattern with wide margins, robust.

After training, two hidden channels were given *massive activations* with a function-preserving
rescaling (the RMSNorm weight of channel j multiplied by c, the matching input column of every
consuming projection divided by c — SmoothQuant run backwards). The model computes the same
function, but its linear-layer inputs now have the outlier channels real LLMs have, so per-tensor
INT8 activations, round-to-nearest INT4 and calibration all behave the way they do at scale
(PRIMER §2 "Number formats" on outliers). ``forward`` is plain numpy; ``act_quant`` and
``kv_quant`` hooks let the lab emulate W8A8 and a quantized KV cache on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import stio

DATA = Path(__file__).parent / "data"
TINY_DIR = DATA / "tiny-adder"

# ---------------------------------------------------------------------------------------------
# Vocabulary and tasks
# ---------------------------------------------------------------------------------------------
VOCAB = [str(d) for d in range(10)] + ["+", "=", "R", "<bos>", "<pad>"]
TOK = {t: i for i, t in enumerate(VOCAB)}
BOS, PAD = TOK["<bos>"], TOK["<pad>"]
TASKS = {"add": 4, "reverse": 6}          # task -> answer length in tokens


def encode(text: str) -> list:
    return [BOS] + [TOK[c] for c in text]


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids if i not in (BOS, PAD))


def make_task(task: str, n: int, seed: int = 0) -> tuple:
    """``(prompts [n, P] int, answers [n, A] int)`` — deterministic for a seed. Held-out evals use
    seeds >= 1000; ``tools/train_tiny.py`` trains on seeds below that (and drops any overlap)."""
    rng = np.random.default_rng([seed, 0 if task == "add" else 1])
    if task == "add":
        a, b = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
        prompts = [encode(f"{x:03d}+{y:03d}=") for x, y in zip(a, b)]
        answers = [[TOK[c] for c in f"{x + y:04d}"[::-1]] for x, y in zip(a, b)]
    elif task == "reverse":
        digits = rng.integers(0, 10, (n, 6))
        prompts = [encode("R" + "".join(map(str, d)) + "=") for d in digits]
        answers = [[TOK[str(v)] for v in d[::-1]] for d in digits]
    else:
        raise KeyError(f"unknown task {task!r}; tasks: {', '.join(TASKS)}")
    return np.array(prompts), np.array(answers)


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    ang = pos[:, None] * inv[None, :]
    cos = np.cos(np.concatenate([ang, ang], -1)).astype(np.float32)
    sin = np.sin(np.concatenate([ang, ang], -1)).astype(np.float32)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + np.concatenate([-x2, x1], -1) * sin


@dataclass
class TinyLM:
    """A decoder-only Llama in numpy. ``weights`` maps Hugging Face names (``model.layers.0.
    self_attn.q_proj.weight``, shape ``[out, in]``, used as ``x @ W.T``) to float32 arrays."""
    config: dict
    weights: dict

    def __post_init__(self):
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in self.weights.items()}

    @property
    def layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    def linear_names(self) -> list:
        """Every quantizable projection, without the ``.weight`` suffix (``lm_head`` excluded)."""
        return [f"model.layers.{i}.{n}" for i in range(self.layers) for n in LINEARS]

    def num_params(self) -> int:
        return int(sum(w.size for w in self.weights.values()))

    def with_weights(self, new: dict) -> "TinyLM":
        return TinyLM(self.config, {**self.weights, **new})

    def forward(self, ids, *, act_quant=None, kv_quant=None, capture: dict | None = None,
                cache: list | None = None) -> np.ndarray:
        """Logits ``[B, T, vocab]`` for token ids ``[B, T]`` (causal, no padding inside a batch).

        ``act_quant(name, x) -> x_hat`` fake-quantizes the input of each projection (W8A8);
        ``kv_quant(kind, layer, x) -> x_hat`` quantizes K (after RoPE, as vLLM caches it) and V as
        they are written to the cache; ``capture`` collects each projection's input rows
        (calibration data); ``cache`` (a list, filled on the first call) makes later calls
        incremental: pass only the new tokens, as a decode step does."""
        c, W = self.config, self.weights
        ids = np.asarray(ids)
        B, T = ids.shape
        H, KV = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
        D = int(c.get("head_dim") or c["hidden_size"] // H)
        eps, theta = float(c.get("rms_norm_eps", 1e-5)), float(c.get("rope_theta", 10000.0))
        past = cache[0]["k"].shape[2] if cache else 0
        pos = np.arange(past, past + T, dtype=np.float64)
        mask = np.triu(np.full((T, past + T), -np.inf, dtype=np.float32), past + 1)

        def lin(name, x):
            if capture is not None:
                capture.setdefault(name, []).append(x.reshape(-1, x.shape[-1]).copy())
            if act_quant is not None:
                x = act_quant(name, x)
            return (x @ W[name + ".weight"].T).astype(np.float32)

        h = W["model.embed_tokens.weight"][ids]
        for i in range(self.layers):
            p = f"model.layers.{i}."
            x = _rms(h, W[p + "input_layernorm.weight"], eps)
            q = lin(p + "self_attn.q_proj", x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = lin(p + "self_attn.k_proj", x).reshape(B, T, KV, D).transpose(0, 2, 1, 3)
            v = lin(p + "self_attn.v_proj", x).reshape(B, T, KV, D).transpose(0, 2, 1, 3)
            q, k = _rope(q, pos, theta), _rope(k, pos, theta)
            if kv_quant is not None:
                k, v = kv_quant("k", i, k), kv_quant("v", i, v)
            if cache is not None:
                if len(cache) <= i:
                    cache.append({"k": k, "v": v})
                else:
                    cache[i]["k"] = k = np.concatenate([cache[i]["k"], k], axis=2)
                    cache[i]["v"] = v = np.concatenate([cache[i]["v"], v], axis=2)
            k, v = np.repeat(k, H // KV, axis=1), np.repeat(v, H // KV, axis=1)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ v
            h = h + lin(p + "self_attn.o_proj", a.transpose(0, 2, 1, 3).reshape(B, T, H * D))
            x = _rms(h, W[p + "post_attention_layernorm.weight"], eps)
            g, u = lin(p + "mlp.gate_proj", x), lin(p + "mlp.up_proj", x)
            h = h + lin(p + "mlp.down_proj", g / (1 + np.exp(-g)) * u)
        h = _rms(h, W["model.norm.weight"], eps)
        return h @ W["lm_head.weight"].T

    # -- tasks ---------------------------------------------------------------------------------
    def answer_logits(self, prompts, answers, **kw) -> np.ndarray:
        """Teacher-forced logits at the answer positions: ``[n, A, vocab]`` (one forward pass)."""
        seq = np.concatenate([prompts, answers[:, :-1]], axis=1)
        return self.forward(seq, **kw)[:, prompts.shape[1] - 1:, :]

    def generate(self, prompts, n_tokens: int, **kw) -> np.ndarray:
        """Greedy decoding with a KV cache: one prefill of the prompts, then one step per token."""
        cache: list = []
        logits = self.forward(np.asarray(prompts), cache=cache, **kw)[:, -1, :]
        out = []
        for _ in range(n_tokens):
            nxt = logits.argmax(-1)
            out.append(nxt)
            if len(out) < n_tokens:
                logits = self.forward(nxt[:, None], cache=cache, **kw)[:, -1, :]
        return np.stack(out, axis=1)

    def accuracy(self, task: str, n: int = 500, seed: int = 1000, **kw) -> float:
        """Exact-match accuracy of greedy answers on ``n`` held-out problems."""
        prompts, answers = make_task(task, n, seed)
        return float((self.generate(prompts, answers.shape[1], **kw) == answers).all(axis=1).mean())


def load_config(path=TINY_DIR) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def load(path=TINY_DIR) -> TinyLM:
    """The dense (unquantized) checkpoint at ``path``, weights decoded to float32. Quantized
    checkpoints are read by :func:`quantlab.compress.load_checkpoint`."""
    path = Path(path)
    cfg = load_config(path)
    if "quantization_config" in cfg:
        raise ValueError(f"{path} is quantized; use quantlab.compress.load_checkpoint()")
    tensors = stio.load(path / "model.safetensors")
    return TinyLM(cfg, {k: t.numpy().astype(np.float32) for k, t in tensors.items()})


def outlier_channels(model: TinyLM, factor: float = 8.0) -> list:
    """Hidden channels whose RMSNorm weight is ``factor`` x the median: the massive-activation
    channels planted after training (see the module docstring)."""
    w = np.abs(model.weights["model.layers.0.input_layernorm.weight"])
    return sorted(int(j) for j in np.nonzero(w > factor * np.median(w))[0])
