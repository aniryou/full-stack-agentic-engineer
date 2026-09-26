"""Fabrics quantitatively: links, the alpha-beta model, collectives and topology.

The one idea: moving n bytes over a link costs  t = alpha + n / beta  -- a fixed
per-message latency plus a bandwidth term. Small messages (a decode step's
tensor-parallel all-reduces) are latency-bound; large ones (prefill, gradients)
are bandwidth-bound. Per-direction bandwidth per GPU drops ~7x from NVLink 4
(450 GB/s) to PCIe Gen5 x16 (63) and ~9x to a 400 Gb/s NIC (50) -- and a NIC is
about as fast as the PCIe link that feeds it, which is why GPUDirect wants the NIC
on the GPU's PCIe switch. Tensor parallelism's all-reduce cost does not shrink as
the TP degree grows while each GPU's compute does, and past the node every GPU's
share rides a NIC: that is why TP stays inside the NVLink domain, and why cluster
networks give every GPU its own NIC on rails of non-blocking trees.

Bandwidths are theoretical and PER DIRECTION (a ring sends and receives at
once). Latencies (alpha) are illustrative per-step figures that include
software; measure yours with nccl-tests (layer 02) or gpu-bench-lab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .llm import ModelConfig


@dataclass(frozen=True)
class Link:
    name: str
    gbs: float          # GB/s per direction
    alpha_us: float     # per-message latency, microseconds (illustrative)
    tier: str           # scale-up | host | scale-out

    @property
    def beta(self) -> float:
        return self.gbs * 1e9

    @property
    def alpha(self) -> float:
        return self.alpha_us * 1e-6


LINKS = {
    "nvlink3": Link("NVLink 3 (A100): 12 links x 25 GB/s", 300, 2, "scale-up"),
    "nvlink4": Link("NVLink 4 (H100/H200): 18 links x 25 GB/s", 450, 2, "scale-up"),
    "nvlink5": Link("NVLink 5 (Blackwell): 18 links x 50 GB/s", 900, 2, "scale-up"),
    "pcie3x16": Link("PCIe Gen3 x16", 15.75, 4, "host"),
    "pcie4x16": Link("PCIe Gen4 x16", 31.5, 4, "host"),
    "pcie5x16": Link("PCIe Gen5 x16", 63.0, 4, "host"),
    "ib-ndr": Link("InfiniBand NDR, 400 Gb/s per NIC", 50, 5, "scale-out"),
    "ib-xdr": Link("InfiniBand XDR, 800 Gb/s per NIC", 100, 5, "scale-out"),
    "roce-400g": Link("RoCE, 400 GbE per NIC", 50, 6, "scale-out"),
    "eth-100g-tcp": Link("100 GbE with TCP", 12.5, 25, "scale-out"),
}


def transfer_time(n_bytes: float, link: Link) -> float:
    """The alpha-beta model: t = alpha + n / beta."""
    return link.alpha + n_bytes / link.beta


# -- collectives -------------------------------------------------------------------------
def ring_allreduce_time(n_bytes: float, p: int, link: Link) -> float:
    """Reduce-scatter then all-gather around a ring of p ranks, each moving n/p per step.

    2(p-1) alpha + 2 (p-1)/p x n / beta. The bandwidth term is optimal (~2n/beta,
    independent of p); the latency term grows linearly with p.
    """
    if p == 1:
        return 0.0
    return 2 * (p - 1) * link.alpha + 2 * (p - 1) / p * n_bytes / link.beta


def ring_allgather_time(n_bytes: float, p: int, link: Link) -> float:
    """All-gather (or reduce-scatter) of a total of n bytes: (p-1) alpha + (p-1)/p x n / beta."""
    if p == 1:
        return 0.0
    return (p - 1) * link.alpha + (p - 1) / p * n_bytes / link.beta


def recursive_doubling_allreduce_time(n_bytes: float, p: int, link: Link) -> float:
    """log2(p) exchange rounds of the full vector: latency-optimal, bandwidth-poor."""
    return math.ceil(math.log2(p)) * (link.alpha + n_bytes / link.beta) if p > 1 else 0.0


def allreduce_crossover_bytes(p: int, link: Link) -> float:
    """Ring message size where latency and bandwidth terms are equal: n = p alpha beta."""
    return p * link.alpha * link.beta


BUSBW_FACTOR = {  # nccl-tests: busbw = algbw x factor, so it is comparable to link bandwidth
    "allreduce": lambda p: 2 * (p - 1) / p,
    "allgather": lambda p: (p - 1) / p,
    "reducescatter": lambda p: (p - 1) / p,
    "alltoall": lambda p: (p - 1) / p,
    "broadcast": lambda p: 1.0,
    "reduce": lambda p: 1.0,
}


def algbw(n_bytes: float, seconds: float) -> float:
    """What nccl-tests calls algorithm bandwidth: message size / time (bytes/s)."""
    return n_bytes / seconds


def busbw(n_bytes: float, seconds: float, p: int, collective: str = "allreduce") -> float:
    """Bus bandwidth: algbw scaled so a perfect ring reads as the link's bandwidth."""
    return algbw(n_bytes, seconds) * BUSBW_FACTOR[collective](p)


def hierarchical_allreduce_time(n_bytes: float, gpus_per_node: int, nodes: int,
                                intra: Link, inter: Link, nics_per_node: int | None = None) -> float:
    """Reduce-scatter in the node, all-reduce n/g across nodes, all-gather in the node.

    With one NIC per GPU (rails) each GPU's cross-node share n/g rides its own NIC. With
    fewer NICs, g GPUs share them: each gets inter bandwidth x nics / g.
    """
    g = gpus_per_node
    share = min(1.0, (nics_per_node or g) / g)
    nic = Link(inter.name, inter.gbs * share, inter.alpha_us, inter.tier)
    return 2 * ring_allgather_time(n_bytes, g, intra) + ring_allreduce_time(n_bytes / g, nodes, nic)


# -- tensor parallelism: the collective inside every layer --------------------------------
def tp_allreduces_per_step(model: ModelConfig) -> int:
    """Megatron-style TP: one all-reduce after attention's W_o and one after the MLP's W_down."""
    return 2 * model.n_layers


def tp_allreduce_bytes(model: ModelConfig, tokens: int, act_bytes: float = 2) -> float:
    """Each all-reduce sums a [tokens x d_model] activation: 70B, 1 token, bf16 = 16 KiB."""
    return tokens * model.d_model * act_bytes


ALLREDUCE = {"ring": ring_allreduce_time, "recursive-doubling": recursive_doubling_allreduce_time}


def tp_comm_time(model: ModelConfig, tokens: int, tp: int, link: Link,
                 act_bytes: float = 2, algo: str = "ring") -> float:
    """Communication time of one forward step under TP=tp (not overlapped with compute).

    Every ring hop runs over `link`: right for TP inside one NVLink domain. With a NIC
    as `link` it is a flat ring in which every hop crosses the network (one GPU per
    node, or one NIC shared by a whole ring) -- the worst case, not a rail-optimized
    cluster; for that use tp_comm_time_across_nodes().
    """
    return tp_allreduces_per_step(model) * ALLREDUCE[algo](tp_allreduce_bytes(model, tokens, act_bytes), tp, link)


def tp_comm_time_across_nodes(model: ModelConfig, tokens: int, gpus_per_node: int, nodes: int,
                              intra: Link, inter: Link, nics_per_node: int | None = None,
                              act_bytes: float = 2) -> float:
    """TP = gpus_per_node x nodes, each all-reduce modelled as hierarchical.

    On rails NCCL spreads a cross-node all-reduce over every NIC (a ring channel per NIC,
    or its tree algorithm), so each NIC carries about an n/g share; the model: reduce-scatter
    inside each node over `intra`, all-reduce each GPU's n/g share across
    nodes over `inter` (its own NIC with rails; a shared one with fewer NICs), all-gather
    inside the node. 70B, 4K-token prefill, TP=16 over two H100 nodes with 8 x 400 Gb/s
    NICs each: ~75 ms per step, against ~37 ms of compute per GPU.
    """
    n = tp_allreduce_bytes(model, tokens, act_bytes)
    return tp_allreduces_per_step(model) * hierarchical_allreduce_time(n, gpus_per_node, nodes, intra, inter,
                                                                       nics_per_node)


# -- topology ------------------------------------------------------------------------------
def oversubscription(down_ports: int, up_ports: int, down_gbps: float = 400,
                     up_gbps: float = 400) -> float:
    """Leaf downlink bandwidth over uplink bandwidth. 1.0 = non-blocking; 3.0 = '3:1'."""
    return down_ports * down_gbps / (up_ports * up_gbps)


@dataclass(frozen=True)
class LeafSpine:
    endpoints: int
    radix: int
    down_per_leaf: int
    leaves: int
    spines: int

    @property
    def up_per_leaf(self) -> int:
        return self.radix - self.down_per_leaf

    @property
    def oversubscription(self) -> float:
        return oversubscription(self.down_per_leaf, self.up_per_leaf)

    @property
    def max_endpoints(self) -> int:
        """Two tiers top out at radix leaves: radix x down (radix^2/2 when non-blocking)."""
        return self.radix * self.down_per_leaf


def leaf_spine(endpoints: int, radix: int, down_per_leaf: int | None = None) -> LeafSpine:
    """Size a two-tier folded Clos. Default splits each leaf half down, half up (1:1)."""
    d = down_per_leaf or radix // 2
    leaves = math.ceil(endpoints / d)
    if leaves > radix:
        raise ValueError(f"{endpoints} endpoints need {leaves} leaves > radix {radix}: add a third tier")
    return LeafSpine(endpoints, radix, d, leaves, math.ceil(leaves * (radix - d) / radix))


def bisection_gbs(endpoints: int, endpoint_gbps: float, oversub: float = 1.0) -> float:
    """GB/s per direction across the worst half/half cut: N/2 x link / oversubscription."""
    return endpoints / 2 * endpoint_gbps / 8 / max(1.0, oversub)


def switch_hops(a: tuple, b: tuple, nodes_per_leaf: int, rail_optimized: bool = True,
                pxn: bool = False) -> int:
    """Network switch hops between GPUs a = (node, gpu_index) and b in a two-tier fabric.

    Rail-optimized: NIC i of every node plugs into rail-i leaf, so same-index GPUs in a
    leaf group are one hop apart and cross-rail traffic climbs to the spine (3 hops) --
    unless PXN first moves it over NVLink to the local GPU on the destination's rail.
    Top-of-rack: all NICs of a node share a leaf. Same node: 0 (NVLink/PCIe, no switch).
    """
    (na, ga), (nb, gb) = a, b
    if na == nb:
        return 0
    same_group = na // nodes_per_leaf == nb // nodes_per_leaf
    if not rail_optimized:
        return 1 if same_group else 3
    if same_group and (ga == gb or pxn):
        return 1
    return 3


# -- data paths: GPUDirect, staging, NUMA --------------------------------------------------
def staged_transfer_time(n_bytes: float, hops_gbs, chunk_bytes: float | None = None) -> float:
    """Copy n bytes through a chain of hops (GB/s each).

    Store-and-forward (chunk None): sum of n / bw_i. Pipelined in chunks: the slowest
    hop sets the rate and every other hop adds one chunk of fill/drain.
    """
    bws = [g * 1e9 for g in hops_gbs]
    if chunk_bytes is None:
        return sum(n_bytes / bw for bw in bws)
    slow = min(bws)
    return n_bytes / slow + sum(chunk_bytes / bw for bw in bws) - chunk_bytes / slow


TOPO_LEGEND = {  # `nvidia-smi topo -m`, best to worst for GPU<->GPU / GPU<->NIC traffic
    "NV#": "connection traversing a bonded set of # NVLinks",
    "PIX": "at most a single PCIe bridge",
    "PXB": "multiple PCIe bridges, without traversing the PCIe host bridge",
    "PHB": "PCIe and a PCIe host bridge (typically the CPU)",
    "NODE": "PCIe and the interconnect between PCIe host bridges within a NUMA node",
    "SYS": "PCIe and the SMP interconnect between NUMA nodes (e.g. QPI/UPI)",
}
_TOPO_ORDER = ["NV", "PIX", "PXB", "PHB", "NODE", "SYS"]


def topo_rank(code: str) -> tuple:
    """Sort key for topology codes: lower is better; more bonded NVLinks beat fewer."""
    code = code.strip().upper()
    if code.startswith("NV"):
        return (0, -int(code[2:] or 1))
    return (_TOPO_ORDER.index(code), 0)
