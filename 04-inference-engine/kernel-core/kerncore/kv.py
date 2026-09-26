"""The KV cache: what it costs in bytes, and a tiny numpy decoder that shows what it buys.

One idea: a past token's K and V never change, so a decoder computes them once and keeps them.
Decoding with the cache gives the same logits as recomputing the whole prefix every step; the
step then processes one token and *reads* a cache that grows by one token per step, instead of
reprocessing a prefix that grows by one token per step. The price is memory:

    KV bytes = 2 (K and V) x layers x kv_heads x head_dim x tokens x batch x bytes_per_value

Contents
  KiB, MiB, GiB, GB, fmt_bytes   binary units with decimal GB in brackets, as the kv-cache primer prints them
  DTYPE_BYTES, MODELS            bytes per cached value; the attention shapes the primer uses (verify)
  kv_heads_for                   MHA / GQA / MQA: how many K/V heads a layer keeps
  kv_bytes_per_token_per_layer, kv_bytes_per_token, kv_cache_bytes, kv_table
  weight_bytes, sessions_per_gpu, decode_tokens_per_s, decode_intensity, ridge
  random_params, TinyDecoder     a numpy decoder (embedding + positions + attention layers + head)
  generate_naive, generate_cached, per_step_costs

Standard library + numpy. Tier T0: no GPU, no torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------------------------------
# Units: binary for sizes (KiB, GiB), decimal GB in brackets -- the kv-cache primer's convention
# ------------------------------------------------------------------------------------------
KiB, MiB, GiB = 1024, 1024**2, 1024**3
GB = 10**9

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def fmt_bytes(b: float, digits: int = 2) -> str:
    """'1.00 GiB (1.07 GB)' for GiB-sized values; '128 KiB' / '800 KiB' below a MiB."""
    if b < MiB:
        return f"{b / KiB:.0f} KiB" if b % KiB == 0 else f"{b / KiB:.{digits}f} KiB"
    if b < GiB:
        return f"{b / MiB:.{digits}f} MiB ({b / 1e6:.{digits}f} MB)"
    return f"{b / GiB:.{digits}f} GiB ({b / GB:.{digits}f} GB)"


# ------------------------------------------------------------------------------------------
# Shapes. From each model's config.json, as of 2026-09-26 (verify).
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AttnShape:
    name: str
    n_layers: int
    n_heads: int          # query heads
    n_kv_heads: int       # K/V heads (== n_heads for MHA)
    head_dim: int
    params: float         # parameter count, for the weight bytes

    @property
    def group(self) -> int:
        """Query heads per K/V head (1 = MHA, n_heads = MQA)."""
        return self.n_heads // self.n_kv_heads


MODELS = {
    "llama-3-8b": AttnShape("Llama 3 8B", 32, 32, 8, 128, 8.03e9),
    "llama-2-13b": AttnShape("Llama 2 13B", 40, 40, 40, 128, 13.0e9),
}


def kv_heads_for(kind: str, n_heads: int, group: int = 4) -> int:
    """K/V heads a layer keeps: MHA keeps one per query head, GQA one per `group`, MQA one."""
    kind = kind.upper()
    if kind == "MHA":
        return n_heads
    if kind == "GQA":
        if n_heads % group:
            raise ValueError(f"n_heads={n_heads} is not a multiple of group={group}")
        return n_heads // group
    if kind == "MQA":
        return 1
    raise ValueError(kind)


def kv_bytes_per_token_per_layer(n_kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    """One token's K and V in one layer."""
    return 2 * n_kv_heads * head_dim * bytes_per


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    """One token's K and V across every layer: 131,072 bytes (128 KiB) for Llama 3 8B in fp16."""
    return n_layers * kv_bytes_per_token_per_layer(n_kv_heads, head_dim, bytes_per)


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int = 1,
                   bytes_per: int = 2) -> int:
    """The primer's formula: 2 x layers x kv_heads x head_dim x seq_len x batch x bytes_per_value."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per


def kv_table(shape: AttnShape, group: int | None = None, dtypes=("fp16", "fp8")) -> list[dict]:
    """Bytes per token per layer and per token for MHA / GQA / MQA over the same query heads."""
    group = shape.group if group is None or group == 1 else group
    rows = []
    for kind in ("MHA", "GQA", "MQA"):
        kv = kv_heads_for(kind, shape.n_heads, group)
        for dt in dtypes:
            b = DTYPE_BYTES[dt]
            rows.append({"kind": kind, "kv_heads": kv, "dtype": dt,
                         "per_layer": kv_bytes_per_token_per_layer(kv, shape.head_dim, b),
                         "per_token": kv_bytes_per_token(shape.n_layers, kv, shape.head_dim, b)})
    return rows


def weight_bytes(shape: AttnShape, bytes_per: int = 2) -> float:
    return shape.params * bytes_per


def sessions_per_gpu(hbm_bytes: float, weight_bytes_: float, context_tokens: int, per_token_bytes: int,
                     reserve_bytes: float = 0.0) -> int:
    """Sessions of `context_tokens` whose KV fits in the HBM the weights leave (an upper bound: no
    activations, no allocator overhead, no fragmentation -- an engine reserves some of each)."""
    free = hbm_bytes - weight_bytes_ - reserve_bytes
    return max(0, int(free // (context_tokens * per_token_bytes)))


def decode_tokens_per_s(bw_bytes_s: float, weight_bytes_: float, kv_bytes: float) -> float:
    """The primer's approximation: HBM bandwidth / (weight bytes + KV bytes read per step)."""
    return bw_bytes_s / (weight_bytes_ + kv_bytes)


def decode_intensity(group: int, bytes_per: int = 2) -> float:
    """FLOP per byte of decode attention: each cached value (b bytes) feeds g query heads, 2 FLOPs each."""
    return 2.0 * group / bytes_per


def ridge(tflops: float, tb_s: float) -> float:
    """FLOP per byte a kernel needs to keep the math units busy."""
    return tflops / tb_s


# ------------------------------------------------------------------------------------------
# A tiny decoder with a KV cache (float64 numpy, random weights)
# ------------------------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def random_params(vocab: int = 256, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                  n_kv_heads: int | None = None, max_len: int = 1024, seed: int = 0) -> dict:
    """Random weights for TinyDecoder. Scaled so activations stay O(1) through the layers."""
    n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    if d_model % n_heads or n_heads % n_kv_heads:
        raise ValueError("d_model must divide by n_heads, and n_heads by n_kv_heads")
    hd = d_model // n_heads
    rng = np.random.default_rng(seed)
    s = 1.0 / math.sqrt(d_model)
    layers = [{"wq": rng.normal(0, s, (d_model, n_heads * hd)),
               "wk": rng.normal(0, s, (d_model, n_kv_heads * hd)),
               "wv": rng.normal(0, s, (d_model, n_kv_heads * hd)),
               "wo": rng.normal(0, s, (n_heads * hd, d_model))} for _ in range(n_layers)]
    return {"vocab": vocab, "d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "n_layers": n_layers, "head_dim": hd, "max_len": max_len,
            "emb": rng.normal(0, 1, (vocab, d_model)), "pos": rng.normal(0, 0.5, (max_len, d_model)),
            "layers": layers, "head": rng.normal(0, s, (d_model, vocab))}


class TinyDecoder:
    """Embedding + absolute positions -> attention layers (residual) -> output head.

    No MLP and no norm: neither touches the KV cache. `forward(ids, cache)` takes a 1-D array of new
    token ids and the cache of everything before them (a list with one (K, V) pair per layer, each
    shaped (n_kv_heads, tokens, head_dim)), and returns (logits for the new tokens, the grown cache).
    `last_cost` counts the multiply-adds of that call and the bytes of K and V its attention read.
    """

    def __init__(self, params: dict):
        self.p = params
        self.last_cost: dict = {}

    @property
    def cfg(self) -> dict:
        return {k: self.p[k] for k in ("vocab", "d_model", "n_heads", "n_kv_heads", "n_layers", "head_dim")}

    def attend(self, x, lp, cache, past_len):
        """One attention layer. x: (T, C). Returns (y, (K, V), multiply-adds, bytes of K and V attended)."""
        T, C = x.shape
        H, KVH, hd = self.p["n_heads"], self.p["n_kv_heads"], self.p["head_dim"]
        q = (x @ lp["wq"]).reshape(T, H, hd).transpose(1, 0, 2)          # (H, T, hd)
        k = (x @ lp["wk"]).reshape(T, KVH, hd).transpose(1, 0, 2)        # (KVH, T, hd)
        v = (x @ lp["wv"]).reshape(T, KVH, hd).transpose(1, 0, 2)
        if cache is not None:                                           # THE CACHE: glue the new K, V
            k = np.concatenate([cache[0], k], axis=1)                   # onto the past, along the
            v = np.concatenate([cache[1], v], axis=1)                   # sequence axis
        kv_attended = k.nbytes + v.nbytes                               # every K, V this step's queries read
        S = k.shape[1]                                                  # past_len + T keys
        kk = np.repeat(k, H // KVH, axis=0)                             # GQA: query head h reads kv head h // group
        vv = np.repeat(v, H // KVH, axis=0)
        att = q @ kk.transpose(0, 2, 1) / math.sqrt(hd)                 # (H, T, S)
        q_pos = np.arange(past_len, past_len + T)[:, None]              # causal mask by ABSOLUTE position
        k_pos = np.arange(S)[None, :]
        att = np.where(k_pos > q_pos, -np.inf, att)
        y = (softmax(att) @ vv).transpose(1, 0, 2).reshape(T, H * hd) @ lp["wo"]
        macs = T * C * (H + 2 * KVH) * hd + 2 * H * T * S * hd + T * H * hd * C
        return y, (k, v), macs, kv_attended

    def forward(self, ids, cache=None):
        ids = np.atleast_1d(np.asarray(ids))
        T = len(ids)
        past_len = 0 if cache is None else cache[0][0].shape[1]
        if past_len + T > self.p["max_len"]:
            raise ValueError(f"{past_len + T} tokens exceed max_len={self.p['max_len']}")
        x = self.p["emb"][ids] + self.p["pos"][past_len:past_len + T]
        new_cache, macs, kv_attended = [], 0, 0
        for i, lp in enumerate(self.p["layers"]):
            y, kv, m, r = self.attend(x, lp, None if cache is None else cache[i], past_len)
            x = x + y
            new_cache.append(kv)
            macs += m
            kv_attended += r
        logits = x @ self.p["head"]
        macs += T * self.p["d_model"] * self.p["vocab"]
        self.last_cost = {"tokens_in": T, "past_len": past_len, "macs": macs,
                          "kv_bytes_attended": kv_attended}
        return logits, new_cache


def cache_nbytes(cache) -> int:
    return sum(k.nbytes + v.nbytes for k, v in cache)


def generate_naive(model: TinyDecoder, prompt, n_new: int):
    """No cache: every step re-feeds the WHOLE sequence. Returns (ids, last-position logits per step, costs)."""
    ids = list(np.asarray(prompt))
    logits_seen, costs = [], []
    for _ in range(n_new):
        logits, _ = model.forward(np.array(ids))
        costs.append(model.last_cost)
        logits_seen.append(logits[-1])
        ids.append(int(logits[-1].argmax()))
    return np.array(ids), np.array(logits_seen), costs


def generate_cached(model: TinyDecoder, prompt, n_new: int):
    """Prefill the prompt once, then feed ONE token per step with the cache. Same return shape."""
    logits, cache = model.forward(np.asarray(prompt))            # PREFILL
    costs, logits_seen = [model.last_cost], [logits[-1]]
    ids = list(np.asarray(prompt)) + [int(logits[-1].argmax())]
    for _ in range(n_new - 1):
        logits, cache = model.forward(np.array([ids[-1]]), cache)   # DECODE: 1 token + the cache
        costs.append(model.last_cost)
        logits_seen.append(logits[-1])
        ids.append(int(logits[-1].argmax()))
    return np.array(ids), np.array(logits_seen), costs


def compare(model: TinyDecoder, prompt, n_new: int, atol: float = 1e-9) -> dict:
    """Run both paths; the headline check. IDENTICAL means the same token ids AND every step's logits
    within `atol` (float64 matmuls of different shapes may differ in the last bits, never more)."""
    a, la, ca = generate_naive(model, prompt, n_new)
    b, lb, cb = generate_cached(model, prompt, n_new)
    diff = float(np.abs(la - lb).max())
    return {"naive": a, "cached": b, "same_ids": bool(np.array_equal(a, b)), "max_logit_diff": diff,
            "atol": atol, "identical": bool(np.array_equal(a, b) and diff <= atol),
            "naive_costs": ca, "cached_costs": cb}


def token_passes(prompt_len: int, n_new: int) -> tuple[int, int]:
    """Tokens pushed through the model to generate n_new: naive reprocesses the t-token sequence at
    every step; cached pushes the prompt once and then one token per step (counted as P + N)."""
    return sum(range(prompt_len, prompt_len + n_new)), prompt_len + n_new


def per_step_costs(costs: list[dict], key: str = "macs") -> np.ndarray:
    """One number per generated token (the cached path's first entry is the prefill)."""
    return np.array([c[key] for c in costs], dtype=float)
