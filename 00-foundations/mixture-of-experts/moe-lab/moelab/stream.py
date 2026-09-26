"""stream.py — a decode step streams the experts its batch touches: predict it, then measure it.

One idea: memory holds every expert, but one decode step only *reads* the experts its tokens are
routed to. With uniform routing a layer touches E(1 - (1 - k/E)^T) distinct experts for T tokens,
so an MoE decodes like its active size at batch 1 and like its total size once T >> E/k — the step
time climbs with batch where a dense model's stays flat, then both flatten (every expert already
streams) until the step turns compute-bound at a batch that scales with total/active.

This module re-implements, standalone, the formulas of layer 01's ``roofline.llm``
(``experts_touched``, ``streamed_weight_bytes``, ``decode``, ``decode_crossover_batch``) generalised
to shared experts, and the tests reproduce its PRIMER §3.6 table (Mixtral-8x7B on an H200: 25.6 GB
and 5.34 ms at batch 1, 754 and 2,055 crossovers) digit for digit. On top of that:

* ``touched_mc`` — Monte Carlo with skewed (Zipf) routing: fewer experts touched, a hotter hottest one;
* ``align_block_size`` — how vLLM's fused MoE kernel lays tokens out (sorted by expert, each
  expert's rows padded to ``BLOCK_SIZE_M``), and the padding it wastes at small batch;
* ``simulate_itl`` — the roofline plus stated efficiencies and a per-step overhead (**simulated**);
* ``measure_itl`` — a closed-loop streaming client: at concurrency B the engine decodes a batch of
  ~B, and the median inter-token latency is the step time (**measured**, T1).
"""
from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .configs import GPU, Model


# ---- which experts a batch touches -----------------------------------------------------------
def experts_touched(n_experts: int, top_k: int, tokens: float) -> float:
    """Expected distinct experts one layer reads for ``tokens`` tokens, uniform routing.

    E (1 - (1 - k/E)^T): each token picks k distinct experts, so a given expert escapes one token
    with probability exactly 1 - k/E. Same formula as ``roofline.llm.experts_touched``; 1.0 for dense.
    """
    if n_experts == 0:
        return 1.0
    return n_experts * (1 - (1 - top_k / n_experts) ** tokens)


def zipf_popularity(n_experts: int, s: float, seed: int | None = 0) -> np.ndarray:
    """Routing probabilities p_e proportional to 1/rank^s, shuffled over experts (s = 0: uniform).
    A *model* of skew for simulation (simulated), not a measurement of any router."""
    p = 1.0 / np.arange(1, n_experts + 1) ** s
    p = p / p.sum()
    if seed is not None:
        p = np.random.default_rng(seed).permutation(p)
    return p


def sample_topk(popularity: np.ndarray, top_k: int, tokens: int, rng: np.random.Generator) -> np.ndarray:
    """``[tokens, top_k]`` distinct expert ids per token, drawn in proportion to ``popularity``
    without replacement (the Gumbel top-k trick: argmax of log p + Gumbel noise, k times)."""
    g = rng.gumbel(size=(tokens, len(popularity)))
    return np.argsort(-(np.log(popularity) + g), axis=1)[:, :top_k]


def touched_mc(n_experts: int, top_k: int, tokens: int, s: float = 0.0, trials: int = 200,
               seed: int = 0) -> tuple[float, float]:
    """(mean distinct experts touched, mean share of assignments on the hottest expert) over
    ``trials`` batches of ``tokens`` tokens with Zipf(s) routing. **Simulated.**"""
    rng = np.random.default_rng(seed)
    pop = zipf_popularity(n_experts, s, seed=seed)
    touched, hot = [], []
    for _ in range(trials):
        ids = sample_topk(pop, top_k, tokens, rng)
        counts = np.bincount(ids.ravel(), minlength=n_experts)
        touched.append((counts > 0).sum())
        hot.append(counts.max() / ids.size)
    return float(np.mean(touched)), float(np.mean(hot))


# ---- bytes and FLOPs of one decode step ---------------------------------------------------------
def streamed_weight_bytes(model: Model, tokens: int, weight_bytes: float = 2, touched: float | None = None) -> float:
    """Weight bytes one step reads: every layer's attention, router and shared expert, the routed
    experts the batch touches, the LM head; the input embedding is a gather of ``tokens`` rows."""
    if model.is_moe:
        e = experts_touched(model.n_experts, model.top_k, tokens) if touched is None else touched
    else:
        e = 1
    per_layer = model.attn_params() + e * model.expert_params() + model.shared_params() + model.router_params()
    return (model.layers * per_layer + model.vocab * model.d_model + tokens * model.d_model) * weight_bytes


def per_token_flops(model: Model, context: int) -> float:
    """2 x active matmul params + 2 x LM head + attention over context + 1 positions."""
    return (2 * model.layers * model.active_layer_params() + 2 * model.vocab * model.d_model
            + 4 * model.layers * model.n_heads * model.head_dim * (context + 1))


@dataclass(frozen=True)
class Step:
    batch: int
    flops: float
    weight_bytes: float
    kv_bytes: float
    t_compute: float
    t_memory: float

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes

    @property
    def time(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound(self) -> str:
        return "compute" if self.t_compute >= self.t_memory else "memory"


def decode_step(model: Model, gpu: GPU, batch: int, context: int, *, weight_bytes: float = 2,
                kv_bytes: float = 2, touched: float | None = None) -> Step:
    """One decode step as one kernel, max(FLOPs / peak, bytes / bandwidth) — ``roofline.llm.decode``.
    Each sequence reads its whole KV cache and writes one entry (context + 1 positions)."""
    kv = batch * (context + 1) * model.kv_bytes_per_token(kv_bytes)
    w = streamed_weight_bytes(model, batch, weight_bytes, touched)
    flops = batch * per_token_flops(model, context)
    return Step(batch, flops, w, kv, flops / (gpu.tflops_16 * 1e12), (w + kv) / (gpu.mem_bw_gbs * 1e9))


def decode_crossover_batch(model: Model, gpu: GPU, context: int = 0, max_batch: int = 1 << 20, **kw) -> int | None:
    """Smallest batch whose one-kernel decode step is compute-bound (bisection), or None."""
    def compute_bound(b):
        return decode_step(model, gpu, b, context, **kw).bound == "compute"
    hi = 1
    while not compute_bound(hi):
        hi *= 2
        if hi > max_batch:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if compute_bound(mid) else (mid, hi)
    return hi


# ---- how the fused MoE kernel lays tokens out ------------------------------------------------------
def align_block_size(topk_ids, block_size: int, num_experts: int):
    """vLLM's ``moe_align_block_size`` semantics in numpy: flatten the T x k assignments, group them
    by expert (ascending id), pad each expert's group to a multiple of ``block_size`` with the pad id
    ``T*k``. Returns (sorted_token_ids, expert_ids per block, num_tokens_post_padded). The Triton
    kernel then runs one ``BLOCK_SIZE_M``-row tile per block against that block's expert weights."""
    flat = np.asarray(topk_ids).ravel()
    pad = flat.size
    n = max(num_experts, int(flat.max()) + 1 if flat.size else 0)
    sorted_ids, expert_ids = [], []
    for e in range(n):
        rows = np.flatnonzero(flat == e).tolist()
        if not rows:
            continue
        padded = -(-len(rows) // block_size) * block_size
        sorted_ids += rows + [pad] * (padded - len(rows))
        expert_ids += [e] * (padded // block_size)
    return np.array(sorted_ids, dtype=np.int64), np.array(expert_ids, dtype=np.int64), len(sorted_ids)


def padding_waste(tokens: int, top_k: int, n_experts: int, block_size: int, s: float = 0.0,
                  trials: int = 50, seed: int = 0) -> float:
    """Mean fraction of the fused kernel's rows that are padding for a batch of ``tokens``
    (simulated routing). High at small batch — harmless there, because the step is memory-bound."""
    rng = np.random.default_rng(seed)
    pop = zipf_popularity(n_experts, s, seed=seed)
    fr = []
    for _ in range(trials):
        ids = sample_topk(pop, top_k, tokens, rng)
        _, _, total = align_block_size(ids, block_size, n_experts)
        fr.append(1 - ids.size / total)
    return float(np.mean(fr))


# ---- simulated ITL: the roofline plus stated inefficiencies ---------------------------------------
@dataclass(frozen=True)
class SimParams:
    """Assumptions that turn a roofline bound into a *simulated* step time. Replace them with
    ``fit_efficiency`` on two measured points from your GPU; until then they are guesses (verify)."""
    mem_eff: float = 0.7            # fraction of datasheet bandwidth the kernels reach
    compute_eff: float = 0.5        # fraction of datasheet FLOP/s (small, untuned MoE GEMMs)
    overhead_s: float = 2.5e-3      # per-step CPU/launch/sampling overhead (CUDA graphs on)


def simulate_itl(model: Model, gpu: GPU, batch: int, context: int, params: SimParams = SimParams(), *,
                 weight_bytes: float = 2, kv_bytes: float = 2, touched: float | None = None) -> float:
    """Seconds per decode step at ``batch`` (**simulated**): overhead + max(compute, memory)."""
    s = decode_step(model, gpu, batch, context, weight_bytes=weight_bytes, kv_bytes=kv_bytes, touched=touched)
    return params.overhead_s + max(s.t_compute / params.compute_eff, s.t_memory / params.mem_eff)


def fit_efficiency(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of ``itl = overhead + bytes / (eff x BW)`` given (predicted memory time at
    100% bandwidth, measured itl) pairs from memory-bound batches: returns (mem_eff, overhead_s)."""
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(1 / slope), float(intercept)


# ---- measured ITL: a closed-loop streaming client (T1) --------------------------------------------
@dataclass
class ItlPoint:
    concurrency: int
    itl_s: list = field(default_factory=list)       # every gap between token-carrying chunks
    ttft_s: list = field(default_factory=list)
    output_tokens: int = 0
    wall_s: float = 0.0
    errors: int = 0
    label: str = "measured"

    @property
    def median_itl_ms(self) -> float:
        return statistics.median(self.itl_s) * 1e3 if self.itl_s else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.wall_s if self.wall_s else 0.0


def _prompt(i: int, n_words: int) -> str:
    # distinct first words per request: no prefix-cache sharing between concurrent requests
    return f"request {i}: " + " ".join(f"w{(i * 7919 + j) % 997}" for j in range(n_words))


def _stream_one(url: str, model: str, prompt: str, max_tokens: int, headers: dict, point: ItlPoint,
                lock: threading.Lock) -> None:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    t0, stamps, n_tok = time.perf_counter(), [], 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    n_tok = chunk["usage"].get("completion_tokens", n_tok)
                if any(c.get("text") for c in chunk.get("choices", [])):
                    stamps.append(time.perf_counter())
    except Exception:  # noqa: BLE001 — count it, keep the others going
        with lock:
            point.errors += 1
        return
    with lock:
        if stamps:
            point.ttft_s.append(stamps[0] - t0)
            point.itl_s += [b - a for a, b in zip(stamps, stamps[1:])]
        point.output_tokens += n_tok or len(stamps)


def measure_itl(url: str, model: str, concurrency: int, *, n_requests: int | None = None, prompt_words: int = 64,
                output_tokens: int = 128, headers: dict | None = None) -> ItlPoint:
    """Closed loop: ``concurrency`` users, each sending the next request when the last finishes.
    With equal output lengths the engine holds ~``concurrency`` sequences in decode, so the median
    ITL is the decode step time at that batch (prefills of new requests show up as the tail)."""
    n_requests = n_requests or 2 * concurrency
    point, lock, nxt = ItlPoint(concurrency), threading.Lock(), iter(range(n_requests))

    def user():
        while True:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            _stream_one(url, model, _prompt(i, prompt_words), output_tokens, headers or {}, point, lock)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=user) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    point.wall_s = time.perf_counter() - t0
    return point


def sweep_itl(url: str, model: str, batches, **kw) -> dict[int, ItlPoint]:
    """``measure_itl`` at each concurrency, with a short warm-up first (CUDA graphs, Triton JIT)."""
    measure_itl(url, model, 1, n_requests=1, output_tokens=16, headers=kw.get("headers"))
    return {b: measure_itl(url, model, b, **kw) for b in batches}
