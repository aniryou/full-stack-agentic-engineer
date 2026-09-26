"""Which experts a batch touches -- and why MoE decode wants big batches.

The one idea: one token reads k of E experts per layer, but a decode step reads the *union* of
its tokens' choices. With uniform routing a layer touches E (1 - (1 - k/E)^T) distinct experts
for T tokens -- the closed form `roofline.llm.experts_touched` uses in layer 01 (PRIMER §3.6),
reproduced here and checked against a Monte Carlo draw. So a step's weight bytes climb from
~active at batch 1 toward ~total, while its FLOPs stay proportional to active x batch: the batch
at which decode turns compute-bound grows by about total / active. Skewed (Zipf) routing touches
fewer experts at a given batch but loads the hottest one more.

The decode step below is the same one-kernel roofline as `roofline.llm.decode()`:
  bytes = weights streamed (attention + touched experts + shared + router per layer, LM head,
          one embedding row per token) + KV read and written
  FLOPs = batch x (2 x active matmul params + 2 x LM head + 4 L h dh (context + 1))
  time  = max(FLOPs / peak, bytes / bandwidth)          (bounds, not predictions)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moe import MoEConfig


def experts_touched(n_experts: int, top_k: int, tokens: float) -> float:
    """Expected distinct experts one layer reads for `tokens` tokens, uniform routing.

    P(a given expert is missed by one token) = 1 - k/E exactly (k distinct of E), and tokens are
    independent, so E (1 - (1 - k/E)^T). Mixtral (8, top-2): 2.00 at T=1, 7.92 at T=16.
    """
    if n_experts == 0:
        return 1.0
    return n_experts * (1 - (1 - top_k / n_experts) ** tokens)


def zipf_popularity(n_experts: int, s: float) -> np.ndarray:
    """Routing probability per expert ~ 1 / rank^s (s = 0: uniform). A model of skew, not a measurement."""
    p = 1.0 / np.arange(1, n_experts + 1) ** s
    return p / p.sum()


def sample_routes(n_experts: int, top_k: int, tokens: int, popularity=None, rng=None) -> np.ndarray:
    """[tokens, k] expert ids, k distinct per token, drawn in proportion to `popularity` without
    replacement (Gumbel top-k: the top k of log p + Gumbel noise)."""
    rng = rng or np.random.default_rng(0)
    logp = np.log(popularity if popularity is not None else np.full(n_experts, 1 / n_experts))
    g = logp + rng.gumbel(size=(tokens, n_experts))
    return np.argsort(-g, axis=1)[:, :top_k]


def touched_mc(n_experts: int, top_k: int, tokens: int, s: float = 0.0, trials: int = 200, seed: int = 0):
    """Monte Carlo (simulated): mean distinct experts touched, and mean hottest-expert load / mean load."""
    rng, pop = np.random.default_rng(seed), zipf_popularity(n_experts, s)
    touched, hot = [], []
    for _ in range(trials):
        c = np.bincount(sample_routes(n_experts, top_k, tokens, pop, rng).reshape(-1), minlength=n_experts)
        touched.append((c > 0).sum())
        hot.append(c.max() / (tokens * top_k / n_experts))
    return float(np.mean(touched)), float(np.mean(hot))


@dataclass(frozen=True)
class Device:
    """The three spec-sheet numbers a decode step needs (dense peaks; Sep 2026, verify)."""
    name: str
    tflops: dict          # dense peak by precision, TFLOP/s
    tbs: float            # memory bandwidth, TB/s
    memory_gb: float

    def peak(self, precision: str = "bf16") -> float:
        return self.tflops[precision] * 1e12

    def bandwidth(self) -> float:
        return self.tbs * 1e12

    def ridge(self, precision: str = "bf16") -> float:
        return self.peak(precision) / self.bandwidth()


DEVICES = {   # the same values as layer 01's roofline.specs (verify)
    "t4": Device("NVIDIA T4", {"fp16": 65}, 0.32, 16),
    "l4": Device("NVIDIA L4", {"bf16": 121, "fp16": 121, "fp8": 242.5}, 0.30, 24),
    "a100-80gb": Device("NVIDIA A100 SXM 80GB", {"bf16": 312, "fp16": 312}, 2.039, 80),
    "h100-sxm": Device("NVIDIA H100 SXM", {"bf16": 989.4, "fp8": 1978.9}, 3.35, 80),
    "h200": Device("NVIDIA H200 SXM", {"bf16": 989.4, "fp8": 1978.9}, 4.8, 141),
    "b200": Device("NVIDIA B200 (HGX)", {"bf16": 2250, "fp8": 4500}, 8.0, 180),
}


def streamed_weight_bytes(cfg: MoEConfig, tokens: int, weight_bytes: float = 2,
                          touched: float | None = None) -> float:
    """Weight bytes one step reads: every layer once, but only the experts its tokens hit.
    `touched` overrides the uniform closed form (e.g. a skewed Monte Carlo value)."""
    e = touched if touched is not None else experts_touched(cfg.n_experts, cfg.top_k, tokens)
    layers = cfg.moe_layers * cfg.moe_layer_params(e) + (cfg.layers - cfg.moe_layers) * cfg.dense_layer_params()
    return (layers + cfg.vocab * cfg.d_model + tokens * cfg.d_model) * weight_bytes


@dataclass(frozen=True)
class Step:
    flops: float
    bytes: float
    t_compute: float
    t_memory: float

    @property
    def time(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound(self) -> str:
        return "compute" if self.t_compute >= self.t_memory else "memory"


def decode_step(cfg: MoEConfig, device: Device, batch: int, context: int, weight_bytes: float = 2,
                kv_bytes: float = 2, precision: str = "bf16", touched: float | None = None) -> Step:
    """One decode step for `batch` sequences with `context` cached tokens each (one-kernel roofline)."""
    per_token = (2 * cfg.matmul_active() + 2 * cfg.vocab * cfg.d_model
                 + 4 * cfg.layers * cfg.heads * cfg.head_dim * (context + 1))
    kv = batch * (context + 1) * cfg.kv_bytes_per_token(kv_bytes)
    b = streamed_weight_bytes(cfg, batch, weight_bytes, touched) + kv
    f = batch * per_token
    return Step(f, b, f / device.peak(precision), b / device.bandwidth())


def decode_crossover_batch(cfg: MoEConfig, device: Device, context: int = 0, max_batch: int = 1 << 20,
                           **kw) -> int | None:
    """Smallest batch whose decode step is compute-bound (bisection), or None. On an H200 at c = 0:
    207 for Llama-3.1-8B, 754 for Mixtral-8x7B, 2,055 for Qwen3-30B-A3B (layer 01 PRIMER §3.6)."""
    bound = lambda b: decode_step(cfg, device, b, context, **kw).bound == "compute"
    hi = 1
    while not bound(hi):
        hi *= 2
        if hi > max_batch:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if bound(mid) else (mid, hi)
    return hi


def kv_share(cfg: MoEConfig, batch: int, context: int, weight_bytes: float = 2, kv_bytes: float = 2) -> float:
    """Fraction of a decode step's bytes that are KV cache: lower for an MoE than for a dense model
    of the same active size, because the MoE streams more weight bytes for the same KV."""
    kv = batch * (context + 1) * cfg.kv_bytes_per_token(kv_bytes)
    return kv / (kv + streamed_weight_bytes(cfg, batch, weight_bytes))
