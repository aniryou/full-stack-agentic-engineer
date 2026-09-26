"""Expert parallelism: move tokens to the experts, and wait for the busiest GPU.

The one idea: with EP each of p GPUs holds E/p whole experts, and tokens travel instead of weights.
With dedicated all-to-all kernels (DeepEP, NIXL-EP, FlashInfer's) every MoE layer sends each token's
hidden state to the GPUs that hold its k experts (dispatch) and brings the k weighted results back
(combine): at most tokens x k x hidden x bytes per GPU per direction, (p - 1)/p of it off the GPU under
uniform routing (layer 02 PRIMER §5.6). vLLM's default exchange is not that: with data-parallel
attention its `allgather_reducescatter` backend all-gathers every rank's tokens and reduce-scatters
the outputs, TP's volume; with DP = 1 (TP + EP) the tokens are already everywhere and one all-reduce
combines the experts' outputs (`moe_comm(mode="tp" | "a2a" | "agrs")`).

Then every rank waits for the slowest, and each rank's expert work is a roofline of its own: its
rows x 2 x expert params / peak against its touched experts x expert bytes / bandwidth. At decode
sizes each expert sees a few rows, every rank reads the weights of all its touched experts, and skew
barely moves the step; at prefill sizes the rows dominate and the rank holding the hot experts sets
the layer time. Replicating hot experts (vLLM's EPLB, redundant experts) buys balance with HBM.

Links: NVLink and NICs use layer 01's illustrative alpha-beta numbers (roofline.fabric.LINKS); PCIe
between GPUs uses the effective NCCL peer-to-peer values the lab assumes (moelab.ep.LINKS), well below
PCIe's line rate (15.75 / 31.5 GB/s per direction for Gen3 / Gen4 x16) -- fit your own with layer 02's
lab. Every time printed here is a model -- simulated, not measured.
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
    "pcie-t4": Link("T4s over PCIe Gen3, NCCL P2P (assumed)", 8, 20),
    "pcie-l4": Link("L4s over PCIe Gen4, NCCL P2P (assumed)", 12, 15),
    "ib-ndr": Link("InfiniBand NDR 400 Gb/s", 50, 5), "roce-400g": Link("RoCE 400 GbE", 50, 6),
}


def dispatch_bytes(tokens: int, k: int, hidden: int, elem_bytes: float = 2, scale_block: int = 0,
                   scale_bytes: int = 4) -> float:
    """Bytes one GPU sends per direction if every assignment is remote (the upper bound):
    tokens x k x hidden x bytes. FP8 dispatch adds a `scale_bytes` scale per `scale_block` channels."""
    row = hidden * elem_bytes + (hidden // scale_block * scale_bytes if scale_block else 0)
    return tokens * k * row


def a2a_time(size: float, p: int, link: Link, algo: str = "direct", per_node: int = 0,
             intra: Link | None = None) -> float:
    """All-to-all of `size` bytes per GPU, alpha-beta: pairwise (p-1) alpha + (p-1)/p S/B,
    direct alpha + (p-1)/p S/B (all sends at once: needs all-to-all links, e.g. NVSwitch).
    With `per_node` GPUs per node on an `intra` link (direct only), the (per_node - 1)/p share for
    peers in the node crosses `intra` while the (p - per_node)/p share crosses `link`, concurrently."""
    if p < 2:
        return 0.0
    if per_node and intra is not None and p > per_node:
        if algo != "direct":
            raise ValueError("two-level all-to-all is modelled for the direct algorithm only")
        return max(intra.alpha_us * 1e-6 + (per_node - 1) / p * size / (intra.gbs * 1e9),
                   link.alpha_us * 1e-6 + (p - per_node) / p * size / (link.gbs * 1e9))
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


def touched_per_rank(idx: np.ndarray, where: np.ndarray, ep: int) -> np.ndarray:
    """[ep] distinct experts each rank reads this layer: an expert with at least one row is streamed
    from HBM once, however many rows it serves; an expert nobody chose is skipped."""
    hit = np.zeros(len(where))
    hit[np.unique(idx)] = 1
    return np.bincount(where, weights=hit, minlength=ep)


@dataclass(frozen=True)
class LayerTime:
    rows: np.ndarray      # [ep] assignments each rank's experts compute
    compute: np.ndarray   # [ep] seconds: rows x 2 x expert params / peak
    memory: np.ndarray    # [ep] seconds: touched experts x expert bytes / bandwidth
    comm: float           # one all-to-all, seconds: alpha + the busiest port's bytes / B
    comm_even: float      # the same exchange spread evenly over the ports

    @property
    def rank(self) -> np.ndarray:
        """Each rank's expert time: its own roofline, max(compute, memory)."""
        return np.maximum(self.compute, self.memory)

    @property
    def total(self) -> float:
        """Dispatch + the slowest rank's experts + combine."""
        return 2 * self.comm + float(self.rank.max())

    @property
    def balanced(self) -> float:
        """The same work spread evenly: every rank at the mean rows and the mean touched experts."""
        return 2 * self.comm_even + max(float(self.compute.mean()), float(self.memory.mean()))

    @property
    def penalty(self) -> float:
        """How much slower this layer is than the balanced one."""
        return self.total / self.balanced

    @property
    def imbalance(self) -> float:
        """Rows on the busiest rank / the mean rank."""
        return float(self.rows.max() / self.rows.mean())

    @property
    def bound(self) -> str:
        r = int(self.rank.argmax())
        return "compute" if self.compute[r] >= self.memory[r] else "memory"


def layer_time(idx: np.ndarray, origin: np.ndarray, where: np.ndarray, ep: int, expert_params: float,
               hidden: int, device: Device, link: Link, weight_bytes: float = 2, elem_bytes: float = 2,
               precision: str = "bf16") -> LayerTime:
    """One MoE layer under EP, from the routes `idx` [T, k] of tokens living on ranks `origin` [T] and
    experts living on ranks `where` [E]. Each rank computes 2 x expert_params FLOPs per row it
    receives and streams each of its touched experts once; the layer waits for the slowest rank.
    A port sends and receives at once, so each all-to-all takes alpha + max(sent, received) / B on
    the busiest rank (the combine is the dispatch transposed: the same ports, reversed)."""
    m = exchange(idx, origin, where, ep)
    rows = m.sum(axis=0)
    compute = rows * 2 * expert_params / device.peak(precision)
    memory = touched_per_rank(idx, where, ep) * expert_params * weight_bytes / device.bandwidth()
    local = np.diag(m)
    port = np.maximum(m.sum(axis=1) - local, rows - local) * hidden * elem_bytes
    even = (m.sum() - local.sum()) / ep * hidden * elem_bytes
    a, bw = link.alpha_us * 1e-6, link.gbs * 1e9
    return LayerTime(rows, compute, memory, a + port.max() / bw, a + even / bw)


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
             elem_bytes: float = 2, dispatch_elem: float | None = None, scale_block: int = 0,
             per_node: int = 0, intra: Link | None = None) -> tuple:
    """(bytes each GPU sends, seconds) for one MoE layer's exchange.
    "tp":   experts sharded across p GPUs that all hold the same `tokens_per_gpu` tokens (also TP + EP
            with DP = 1, where vLLM runs no all-to-all) -> one ring all-reduce of tokens x hidden:
            2(p-1) alpha + 2(p-1)/p S/B.
    "a2a":  data-parallel attention, whole experts per GPU, dedicated all-to-all kernels (DeepEP-class)
            -> dispatch (at `dispatch_elem` bytes, + a 4-byte scale per `scale_block` channels: FP8)
            and combine (at `elem_bytes`) all-to-alls; `per_node`/`intra` split them over two links.
    "agrs": vLLM's default `allgather_reducescatter` with DP > 1 -> ring all-gather of every rank's
            tokens, then a ring reduce-scatter of the outputs: (p-1) alpha + (p-1)/p S/B each, with
            S = p x tokens x hidden x bytes. TP's volume, not 1/p of it."""
    if mode == "tp":
        s = tokens_per_gpu * hidden * elem_bytes
        return 2 * (p - 1) / p * s, 2 * (p - 1) * link.alpha_us * 1e-6 + 2 * (p - 1) / p * s / (link.gbs * 1e9)
    if mode == "agrs":
        s = p * tokens_per_gpu * hidden * elem_bytes
        one = (p - 1) * link.alpha_us * 1e-6 + (p - 1) / p * s / (link.gbs * 1e9)
        return 2 * (p - 1) / p * s, 2 * one if p > 1 else 0.0
    if mode != "a2a":
        raise ValueError(mode)
    d = dispatch_bytes(tokens_per_gpu, k, hidden, elem_bytes if dispatch_elem is None else dispatch_elem, scale_block)
    c = dispatch_bytes(tokens_per_gpu, k, hidden, elem_bytes)
    t = a2a_time(d, p, link, per_node=per_node, intra=intra) + a2a_time(c, p, link, per_node=per_node, intra=intra)
    return (p - 1) / p * (d + c), t


def wide_ep_weights(cfg: MoEConfig, ep: int, weight_bytes: float = 1, redundant: int = 0) -> float:
    """Weight bytes per GPU with data-parallel attention + EP: attention, dense layers, shared
    experts, routers and embeddings are replicated on every rank; routed experts are split,
    (E + R) / ep per layer (vLLM's EPLB memory formula)."""
    routed = cfg.moe_layers * cfg.n_experts * cfg.expert_params()
    per_rank = cfg.moe_layers * (cfg.n_experts + redundant) / ep * cfg.expert_params()
    return (cfg.total() - routed + per_rank) * weight_bytes


def decode_on(cfg: MoEConfig, device: Device, batch: int, context: int, n: int, layout: str = "ep",
              link: Link | None = None, weight_bytes: float = 2, kv_bytes: float = 2, precision: str = "bf16",
              exchange: str = "a2a", **comm) -> dict:
    """A decode step on n GPUs, perfectly balanced (simulated):
    "ep"  data-parallel attention + EP (wide-EP): each rank streams its full copy of the non-expert
          weights, 1/n of the experts the *global* batch touches, and the KV of its batch/n
          sequences; two exchanges per MoE layer (`exchange`: "a2a" kernels or vLLM's default
          "agrs"; `comm` -> moe_comm, e.g. FP8 dispatch or a two-level fabric). For a dense model
          this is n replicas.
    "tp"  every matrix split n ways, every GPU sees every token; two all-reduces per layer."""
    e = experts_touched(cfg.n_experts, cfg.top_k, batch)
    experts = cfg.moe_layers * e * cfg.expert_params()
    kv = batch / n * (context + 1) * cfg.kv_bytes_per_token(kv_bytes)
    t_comm = 0.0
    if layout == "ep":
        rest = streamed_weight_bytes(cfg, batch / n, 1, touched=e) - experts   # replicated on every rank
        b = (rest + experts / n) * weight_bytes + kv
        if link is not None and n > 1:
            t_comm = cfg.moe_layers * moe_comm(-(-batch // n), cfg.top_k, cfg.d_model, n, link, exchange, **comm)[1]
    else:
        b = streamed_weight_bytes(cfg, batch, weight_bytes, touched=e) / n + kv
        if link is not None and n > 1:
            t_comm = 2 * cfg.layers * moe_comm(batch, cfg.top_k, cfg.d_model, n, link, "tp")[1]
    f = batch / n * (2 * cfg.matmul_active() + 2 * cfg.vocab * cfg.d_model
                     + cfg.layers * cfg.attn_flops_per_position() * (context + 1))
    t = max(f / device.peak(precision), b / device.bandwidth())
    return dict(bytes_per_gpu=b, flops_per_gpu=f, roofline=t, comm=t_comm, time=t + t_comm)
