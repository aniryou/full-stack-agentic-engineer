"""The MoE layer: E expert MLPs, a router that picks k of them per token, and a weighted combine.

The one idea: a mixture-of-experts layer replaces one MLP with E MLPs ("experts") and a small
linear router. Each token is scored against every expert, keeps its top-k, and its output is the
router-weighted sum of those k experts' outputs (plus a shared expert every token uses, if the
model has one). Memory holds all E experts; a token's FLOPs pay for k. With E = k = 1 the layer
is the dense MLP it replaced.

Routers differ in details that change the numbers; `route()` exposes them as options named after
the models that use them (transformers and reference code, Sep 2026):

  mixtral      softmax over E, top-k, renormalise the k weights to sum to 1
  qwen3/olmoe  softmax over E, top-k, renormalise only if `norm_topk_prob` (HF default False)
  deepseek-v3  sigmoid scores; a per-expert bias *chooses* experts but never weights them;
               group-limited top-k; renormalise the k sigmoid scores, then x route_scale 2.5
  gpt-oss      router with a bias; top-k on the logits, softmax over just those k
  llama4       top-1, sigmoid of its score, applied to the expert's *input*; one shared expert

`MoELayer.forward` runs the layer the way fused MoE kernels do: sort the T x k assignments by
expert, one GEMM per expert over a contiguous slice (a grouped GEMM), scatter the weighted rows
back and sum each token's k copies. `forward_dense` is the reference that runs every expert on
every token. `MoEConfig` counts total and active parameters from a model config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores per row, largest first (ties: lower index first)."""
    return np.argsort(-scores, axis=-1, kind="stable")[..., :k]


@dataclass
class Expert:
    """One MLP. Gated (SwiGLU): down(silu(x W_gate) * (x W_up)); plain: down(relu(x W_up))."""
    w_up: np.ndarray                  # [d, ff]
    w_down: np.ndarray                # [ff, d]
    w_gate: np.ndarray | None = None  # [d, ff], None for a plain two-matrix MLP

    @classmethod
    def init(cls, rng: np.random.Generator, d: int, ff: int, gated: bool = True) -> "Expert":
        m = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)
        return cls(m(d, ff), m(ff, d), m(d, ff) if gated else None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.w_up
        if self.w_gate is not None:
            g = x @ self.w_gate
            h = g / (1.0 + np.exp(-g)) * h
        else:
            h = np.maximum(h, 0.0)
        return h @ self.w_down

    @property
    def n_params(self) -> int:
        return sum(w.size for w in (self.w_up, self.w_down, self.w_gate) if w is not None)


@dataclass
class Routing:
    logits: np.ndarray    # [T, E] raw router outputs
    probs: np.ndarray     # [T, E] softmax (or sigmoid) scores: what balance losses read
    idx: np.ndarray       # [T, k] chosen experts, best first
    weights: np.ndarray   # [T, k] combine weights


def limit_groups(scores: np.ndarray, groups: int, keep: int, per_group: int) -> np.ndarray:
    """Group-limited routing (DeepSeek-V3): split E experts into `groups` groups, score each group
    by the sum of its `per_group` best scores, keep the `keep` best groups, mask the rest to -inf.
    With experts laid out group-per-node, a token then talks to at most `keep` nodes."""
    t, e = scores.shape
    g = scores.reshape(t, groups, e // groups)
    group_score = np.sort(g, axis=-1)[..., -per_group:].sum(-1)
    kept = topk(group_score, keep)
    mask = np.ones((t, groups), dtype=bool)
    np.put_along_axis(mask, kept, False, axis=1)
    return np.where(np.repeat(mask, e // groups, axis=1), -np.inf, scores)


def route(logits: np.ndarray, k: int, score: str = "softmax", norm: bool = True, scale: float = 1.0,
          select_bias: np.ndarray | None = None, groups: int = 1, topk_groups: int = 1) -> Routing:
    """Router logits [T, E] -> which k experts, with which weights.

    score: "softmax" | "sigmoid" | "topk_softmax" (gpt-oss/Granite: pick on logits, softmax the k).
    select_bias: added to the scores for *selection only* (DeepSeek's aux-loss-free bias).
    """
    probs = sigmoid(logits) if score == "sigmoid" else softmax(logits)
    choose = logits if score == "topk_softmax" else probs
    if select_bias is not None:
        choose = choose + select_bias
    if groups > 1:
        choose = limit_groups(choose, groups, topk_groups, max(1, k // topk_groups))
    idx = topk(choose, k)
    if score == "topk_softmax":
        w = softmax(np.take_along_axis(logits, idx, axis=1))
    else:
        w = np.take_along_axis(probs, idx, axis=1)       # weights from the UNbiased scores
        if norm:
            w = w / w.sum(axis=1, keepdims=True)
    return Routing(logits, probs, idx, w * scale)


ROUTERS = {   # route() options per family (see the module docstring)
    "mixtral": dict(score="softmax", norm=True),
    "qwen3": dict(score="softmax", norm=False),
    "deepseek-v3": dict(score="sigmoid", norm=True, scale=2.5),
    "gpt-oss": dict(score="topk_softmax"),
    "llama4": dict(score="sigmoid", norm=False),
}


@dataclass
class MoELayer:
    w_router: np.ndarray                         # [d, E]
    experts: list
    k: int
    router: dict = field(default_factory=dict)   # route() options, e.g. ROUTERS["mixtral"]
    router_bias: np.ndarray | None = None        # part of the logits (gpt-oss): weights see it too
    select_bias: np.ndarray | None = None        # selection only (DeepSeek-V3)
    shared: list = field(default_factory=list)   # shared experts: every token, weight 1
    shared_gate: np.ndarray | None = None        # [d]: sigmoid(x . g) scales the shared expert (Qwen2-MoE)
    scale_input: bool = False                    # Llama 4: weight the expert's input, not its output

    @classmethod
    def init(cls, rng, d: int, ff: int, n_experts: int, k: int, n_shared: int = 0, gated: bool = True,
             **router) -> "MoELayer":
        return cls(rng.standard_normal((d, n_experts)) / np.sqrt(d),
                   [Expert.init(rng, d, ff, gated) for _ in range(n_experts)], k, router,
                   shared=[Expert.init(rng, d, ff, gated) for _ in range(n_shared)])

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def route(self, x: np.ndarray) -> Routing:
        logits = x @ self.w_router + (0.0 if self.router_bias is None else self.router_bias)
        return route(logits, self.k, select_bias=self.select_bias, **self.router)

    def _shared(self, x: np.ndarray) -> np.ndarray:
        out = sum((s(x) for s in self.shared), np.zeros_like(x))
        return out * sigmoid(x @ self.shared_gate)[:, None] if self.shared_gate is not None else out

    def forward(self, x: np.ndarray, r: Routing | None = None) -> np.ndarray:
        """Sparse forward: sort assignments by expert, one matmul per expert slice, scatter-add."""
        r = r or self.route(x)
        flat, w = r.idx.reshape(-1), r.weights.reshape(-1)
        order = np.argsort(flat, kind="stable")          # the permutation a fused kernel applies
        counts = np.bincount(flat, minlength=self.n_experts)
        out, start = np.zeros_like(x), 0
        for e, n in enumerate(counts):
            if n == 0:
                continue                                  # an expert nobody chose costs no FLOPs
            rows = order[start:start + n]
            tok, ww = rows // self.k, w[rows][:, None]
            y = self.experts[e](x[tok] * ww) if self.scale_input else self.experts[e](x[tok]) * ww
            np.add.at(out, tok, y)                        # combine: sum each token's k copies
            start += n
        return out + self._shared(x)

    def forward_dense(self, x: np.ndarray, r: Routing | None = None) -> np.ndarray:
        """Reference: every expert on every token, weighted by a dense [T, E] gate (0 off the top-k)."""
        r = r or self.route(x)
        gate = np.zeros((x.shape[0], self.n_experts))
        np.put_along_axis(gate, r.idx, r.weights, axis=1)
        if self.scale_input:
            out = sum(e(x * gate[:, [i]]) for i, e in enumerate(self.experts))
        else:
            out = sum(gate[:, [i]] * e(x) for i, e in enumerate(self.experts))
        return out + self._shared(x)


# ---- counting parameters from a config ------------------------------------------------------

def gqa_params(d: int, heads: int, kv_heads: int, head_dim: int, qkv_bias: bool = False,
               o_bias: bool = False, sinks: bool = False) -> int:
    """W_q, W_k, W_v, W_o (+ biases, + gpt-oss's per-head attention sinks)."""
    q, kv = heads * head_dim, kv_heads * head_dim
    return (2 * d * q + 2 * d * kv + (q + 2 * kv if qkv_bias else 0) + (d if o_bias else 0)
            + (heads if sinks else 0))


def mla_params(d: int, heads: int, q_rank: int, kv_rank: int, nope: int, rope: int, v_dim: int) -> int:
    """DeepSeek multi-head latent attention (inference/model.py `MLA`), incl. its two norms."""
    q = d * q_rank + q_rank + q_rank * heads * (nope + rope)
    kv = d * (kv_rank + rope) + kv_rank + kv_rank * heads * (nope + v_dim)
    return q + kv + heads * v_dim * d


@dataclass(frozen=True)
class MoEConfig:
    """The config.json fields that set memory and compute. n_experts = 0 means a dense model."""
    name: str
    layers: int
    d_model: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    n_experts: int = 0
    top_k: int = 0
    expert_ff: int = 0        # one routed expert's intermediate size (dense model: the MLP's)
    shared_ff: int = 0        # all shared experts together (DeepSeek: n_shared x moe_inter)
    n_shared: int = 0
    dense_layers: int = 0     # layers with a dense MLP instead of MoE (DeepSeek-V3: 3)
    dense_ff: int = 0
    tied: bool = False
    gated: bool = True        # SwiGLU: gate, up, down
    attn: int = 0             # per-layer attention params when not plain GQA (MLA, biases)
    expert_extra: int = 0     # per-expert extras (gpt-oss biases)
    router_extra: int = 0     # per-MoE-layer extras (router bias, shared-expert gate)
    kv_elems: int = 0         # KV elements per token per layer when not 2 x kv_heads x head_dim
    note: str = ""

    @property
    def moe_layers(self) -> int:
        return self.layers - self.dense_layers if self.n_experts else 0

    def attn_params(self) -> int:
        return self.attn or gqa_params(self.d_model, self.heads, self.kv_heads, self.head_dim)

    def _mlp(self, ff: int) -> int:
        return (3 if self.gated else 2) * self.d_model * ff

    def expert_params(self) -> int:
        return self._mlp(self.expert_ff) + self.expert_extra

    def router_params(self) -> int:
        return self.d_model * self.n_experts + self.router_extra

    def moe_layer_params(self, experts: float) -> float:
        """One MoE layer holding (or reading) `experts` routed experts, plus shared and router."""
        return self.attn_params() + experts * self.expert_params() + self._mlp(self.shared_ff) + self.router_params()

    def dense_layer_params(self) -> int:
        return self.attn_params() + self._mlp(self.dense_ff or self.expert_ff)

    def embedding_params(self) -> int:
        return self.vocab * self.d_model * (1 if self.tied else 2)

    def total(self) -> int:
        """Everything that must sit in memory."""
        return int(self.moe_layers * self.moe_layer_params(self.n_experts)
                   + (self.layers - self.moe_layers) * self.dense_layer_params() + self.embedding_params())

    def matmul_active(self) -> int:
        """Params one token multiplies by in the layers (no embedding gather, no LM head)."""
        return int(self.moe_layers * self.moe_layer_params(self.top_k)
                   + (self.layers - self.moe_layers) * self.dense_layer_params())

    def active(self, embeddings: str = "both") -> int:
        """Active params. Conventions differ: "both" (DeepSeek's 37B, roofline.llm.active_params),
        "head" (LM head only: OpenAI's 5.1B for gpt-oss-120b), "none" (matmuls in the layers)."""
        head = self.vocab * self.d_model
        return self.matmul_active() + {"both": head * (1 if self.tied else 2), "head": head, "none": 0}[embeddings]

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> float:
        """KV follows attention, not experts: an MoE caches exactly what its attention caches."""
        return self.layers * (self.kv_elems or 2 * self.kv_heads * self.head_dim) * kv_bytes
