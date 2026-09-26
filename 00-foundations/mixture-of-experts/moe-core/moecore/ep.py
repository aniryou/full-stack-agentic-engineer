"""Expert parallelism: move tokens to the experts, and wait for the busiest GPU.

The one idea: with EP each of p GPUs holds E/p whole experts. Every MoE layer sends each token's
hidden state to the GPUs that hold its k experts (dispatch, an all-to-all) and brings the k
weighted results back (combine, another all-to-all): at most tokens x k x hidden x bytes per GPU
per direction, of which (p - 1)/p leaves the GPU under uniform routing (layer 02 PRIMER §5.6).
Then every rank waits for the slowest: the layer takes as long as the most loaded rank's expert
GEMMs plus the exchanges, so a hot expert slows the whole step. Replicating hot experts (vLLM's
EPLB, redundant experts) buys balance with HBM.

Links use layer 01's illustrative alpha-beta numbers (roofline.fabric.LINKS): per-direction GB/s
and a per-message latency alpha. Every time printed here is a model -- simulated, not measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moe import MoEConfig
from .touched import Device, experts_touched, streamed_weight_bytes


@dataclass(frozen=True)
class Link:
    name: str
    gbs: float        # per direction, GB/s
    alpha_us: float   # per message, microseconds (illustrative)


LINKS = {
    "nvlink4": Link("NVLink 4 (H100/H200)", 450, 2), "nvlink5": Link("NVLink 5 (Blackwell)", 900, 2),
    "pcie3x16": Link("PCIe Gen3 x16 (T4)", 15.75, 4), "pcie4x16": Link("PCIe Gen4 x16 (L4)", 31.5, 4),
    "ib-ndr": Link("InfiniBand NDR 400 Gb/s", 50, 5), "roce-400g": Link("RoCE 400 GbE", 50, 6),
}


def dispatch_bytes(tokens: int, k: int, hidden: int, elem_bytes: float = 2, scale_block: int = 0,
                   scale_bytes: int = 4) -> float:
    """Bytes one GPU sends per direction if every assignment is remote (the upper bound):
    tokens x k x hidden x bytes. FP8 dispatch adds a `scale_bytes` scale per `scale_block` channels."""
    row = hidden * elem_bytes + (hidden // scale_block * scale_bytes if scale_block else 0)
    return tokens * k * row


def a2a_time(size: float, p: int, link: Link, algo: str = "direct") -> float:
    """All-to-all of `size` bytes per GPU, alpha-beta: pairwise (p-1) alpha + (p-1)/p S/B,
    direct alpha + (p-1)/p S/B (all sends at once: needs all-to-all links, e.g. NVSwitch)."""
    if p < 2:
        return 0.0
    steps = p - 1 if algo == "pairwise" else 1
    return steps * link.alpha_us * 1e-6 + (p - 1) / p * size / (link.gbs * 1e9)


def placement(n_experts: int, ep: int, strategy: str = "linear") -> np.ndarray:
    """Rank holding each expert: "linear" = contiguous blocks (vLLM's default), "round_robin" = e mod p."""
    e = np.arange(n_experts)
    return e // (n_experts // ep) if strategy == "linear" else e % ep


def exchange(idx: np.ndarray, origin: np.ndarray, where: np.ndarray, ep: int) -> np.ndarray:
    """[ep, ep] matrix of rows sent from rank src (where the token's attention ran) to rank dst
    (where its expert lives), one row per assignment; the diagonal stays local."""
    m = np.zeros((ep, ep), dtype=int)
    np.add.at(m, (np.repeat(origin, idx.shape[1]), where[idx.reshape(-1)]), 1)
    return m


@dataclass(frozen=True)
class LayerTime:
    compute: np.ndarray   # [ep] seconds of expert GEMMs per rank
    comm: float           # one all-to-all, seconds (the slowest port)
    total: float          # dispatch + slowest rank's experts + combine

    @property
    def imbalance(self) -> float:
        return float(self.compute.max() / self.compute.mean())


def layer_time(m: np.ndarray, expert_params: float, hidden: int, device: Device, link: Link,
               elem_bytes: float = 2, precision: str = "bf16") -> LayerTime:
    """One MoE layer under EP: each rank computes 2 x expert_params FLOPs per row it receives;
    a send takes alpha + off-diagonal bytes / B per port. The step waits for the slowest rank."""
    compute = m.sum(axis=0) * 2 * expert_params / device.peak(precision)
    off = (m.sum(axis=1) - np.diag(m)) * hidden * elem_bytes
    comm = link.alpha_us * 1e-6 + off.max() / (link.gbs * 1e9)
    return LayerTime(compute, comm, 2 * comm + compute.max())


def rebalance(expert_load: np.ndarray, ep: int, redundant: int = 0):
    """EPLB in miniature: give the `redundant` hottest experts a second copy each (splitting their
    load), then pack all copies onto ranks, heaviest first, onto the least-loaded rank with a free
    slot ((E + R) / ep slots each). Returns per-rank load."""
    load = np.asarray(expert_load, dtype=float)
    hot = set(np.argsort(-load, kind="stable")[:redundant].tolist())
    copies = []
    for e, x in enumerate(load):
        copies += [x / 2, x / 2] if e in hot else [x]
    slots, ranks, used = len(copies) // ep, np.zeros(ep), np.zeros(ep, dtype=int)
    for x in sorted(copies, reverse=True):
        free = np.flatnonzero(used < slots)
        r = free[np.argmin(ranks[free])]
        ranks[r], used[r] = ranks[r] + x, used[r] + 1
    return ranks


def moe_comm(tokens_per_gpu: int, k: int, hidden: int, p: int, link: Link, mode: str,
             elem_bytes: float = 2) -> tuple:
    """(bytes each GPU sends, seconds) for one MoE layer.
    "tp": experts sharded across p GPUs that all hold the same tokens -> one ring all-reduce of
          tokens x hidden: 2(p-1) alpha + 2(p-1)/p S/B.
    "ep": data-parallel attention, whole experts per GPU -> dispatch + combine all-to-alls."""
    if mode == "tp":
        s = tokens_per_gpu * hidden * elem_bytes
        return 2 * (p - 1) / p * s, 2 * (p - 1) * link.alpha_us * 1e-6 + 2 * (p - 1) / p * s / (link.gbs * 1e9)
    s = dispatch_bytes(tokens_per_gpu, k, hidden, elem_bytes)
    return 2 * (p - 1) / p * s, 2 * a2a_time(s, p, link)


def wide_ep_weights(cfg: MoEConfig, ep: int, weight_bytes: float = 1, redundant: int = 0) -> float:
    """Weight bytes per GPU with data-parallel attention + EP: attention, dense layers, shared
    experts, routers and embeddings are replicated on every rank; routed experts are split,
    (E + R) / ep per layer (vLLM's EPLB memory formula)."""
    routed = cfg.moe_layers * cfg.n_experts * cfg.expert_params()
    per_rank = cfg.moe_layers * (cfg.n_experts + redundant) / ep * cfg.expert_params()
    return (cfg.total() - routed + per_rank) * weight_bytes


def decode_on(cfg: MoEConfig, device: Device, batch: int, context: int, n: int, layout: str = "ep",
              link: Link | None = None, weight_bytes: float = 2, kv_bytes: float = 2, precision: str = "bf16") -> dict:
    """A decode step on n GPUs, perfectly balanced (simulated):
    "ep"  data-parallel attention + EP (wide-EP): each rank streams its full copy of the non-expert
          weights, 1/n of the experts the *global* batch touches, and the KV of its batch/n
          sequences; two all-to-alls per MoE layer. For a dense model this is n replicas.
    "tp"  every matrix split n ways, every GPU sees every token; two all-reduces per layer."""
    e = experts_touched(cfg.n_experts, cfg.top_k, batch)
    experts = cfg.moe_layers * e * cfg.expert_params()
    kv = batch / n * (context + 1) * cfg.kv_bytes_per_token(kv_bytes)
    comm = 0.0
    if layout == "ep":
        rest = streamed_weight_bytes(cfg, batch / n, 1, touched=e) - experts   # replicated on every rank
        b = (rest + experts / n) * weight_bytes + kv
        if link is not None and n > 1:
            comm = cfg.moe_layers * moe_comm(-(-batch // n), cfg.top_k, cfg.d_model, n, link, "ep")[1]
    else:
        b = streamed_weight_bytes(cfg, batch, weight_bytes, touched=e) / n + kv
        if link is not None and n > 1:
            comm = 2 * cfg.layers * moe_comm(batch, cfg.top_k, cfg.d_model, n, link, "tp")[1]
    f = batch / n * (2 * cfg.matmul_active() + 2 * cfg.vocab * cfg.d_model
                     + 4 * cfg.layers * cfg.heads * cfg.head_dim * (context + 1))
    t = max(f / device.peak(precision), b / device.bandwidth())
    return dict(bytes_per_gpu=b, flops_per_gpu=f, roofline=t, comm=comm, time=t + comm)
