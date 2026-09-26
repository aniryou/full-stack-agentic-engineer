"""GPU-to-GPU bandwidth: measure it (T2), predict it from the topology (T0), and price TP with it.

``measure_matrix`` copies a large buffer between every ordered pair of GPUs (optionally both
directions at once) — the same experiment as NVIDIA's ``p2pBandwidthLatencyTest`` and
``nvbandwidth``, which remain the reference tools. ``predict`` turns a parsed
``nvidia-smi topo -m`` into the theoretical per-direction bandwidth of each path, labelled as
a model. Comparing the two is the point: a PCIe pair at a fraction of its link rate, or an
"NV" pair far below the NVLink rate, means something is misconfigured (peer access disabled,
ACS forcing P2P through the root complex, a staged copy).

``ring_allreduce_time`` then prices a tensor-parallel all-reduce with the α-β model (primer
§5): ``2(p−1)·α + 2·(p−1)/p · S/B`` for ``S`` bytes on ``p`` GPUs with per-GPU link bandwidth ``B``.
At decode batch sizes ``S`` is tiny (a few KB per layer), so the α term dominates — which is
why TP stays inside one NVLink domain, where α and B are both at their best.
"""
from __future__ import annotations

from dataclasses import dataclass

from .measure import measure
from .specs import NVLINK_GBS_PER_LINK, NVLINK_GEN, pcie_gbs
from .topo import Topology


def measure_matrix(be, nbytes: int = 256 << 20, bidirectional: bool = False, devices=None,
                   repeats: int = 5, min_time: float = 0.05) -> list:
    """Measure every ordered GPU pair (every unordered pair when ``bidirectional``). T2 only."""
    count = be.describe()["device_count"] if be.is_gpu else 0
    if count < 2:
        raise ValueError("P2P needs the torch backend and at least two visible GPUs")
    devices = list(devices if devices is not None else range(count))
    out = []
    for i in devices:
        for j in devices:
            if i == j or (bidirectional and j < i):
                continue
            op = be.make_p2p(i, j, nbytes, bidirectional=bidirectional)
            out.append(measure(be, op, "p2p.bidir" if bidirectional else "p2p",
                               {"src": i, "dst": j, "nbytes": nbytes}, repeats=repeats, min_time=min_time))
    return out


def size_sweep(be, src: int, dst: int, sizes, repeats: int = 5, min_time: float = 0.02,
               latency: bool = False) -> list:
    """One pair over a range of sizes: feed it to ``transfer.fit`` for α and β of the path.
    ``latency=True`` times one synchronised copy per sample, so the fitted α is a copy's latency
    (what each dependent step of a collective pays) rather than the pipelined issue cost."""
    mode = "latency" if latency else "pipelined"
    return [measure(be, be.make_p2p(src, dst, s, sync_each=latency), "p2p",
                    {"src": src, "dst": dst, "nbytes": s, "mode": mode},
                    repeats=repeats, min_time=min_time) for s in sizes]


@dataclass(frozen=True)
class PathEstimate:
    gbs: float               # theoretical GB/s per direction (a model, not a measurement)
    path: str                # the topo code, e.g. "NV18", "PIX", "SYS"
    why: str
    mode: str = "direct"     # "direct" (peer access), "staged" (through host memory), "upper-bound" (unknown)


def path_bandwidth(code: str, arch: str | None = None, pcie_gen: int = 4, pcie_lanes: int = 16,
                   peer_access: bool | None = None) -> PathEstimate:
    """Theoretical per-direction bandwidth of one ``nvidia-smi topo`` path.

    ``peer_access`` is what ``torch.cuda.can_device_access_peer`` says for the pair (None when it
    is not known, e.g. from a saved topology). With peer access a PCIe copy runs at the link rate;
    without it the driver stages the copy through host memory — two PCIe crossings and a host
    copy — modelled as half the link (measured staged copies often land lower still):

    * ``NV#``: links × the per-link rate of the NVLink generation.
    * ``PIX``/``PXB``: the link rate (peer access through PCIe switches normally works).
    * ``PHB``: the link rate *if* peer access works through the root complex — an upper bound
      when unknown, because many platforms and most VMs disable it; staged when it is off.
    * ``NODE``/``SYS``: staged unless peer access is reported (P2P rarely crosses host bridges or
      the socket link).
    """
    if code.startswith("NV"):
        links = int(code[2:])
        gen = NVLINK_GEN.get(arch or "", 4)
        rate = NVLINK_GBS_PER_LINK[gen]
        assumed = "" if arch in NVLINK_GEN else " (NVLink gen assumed 4; pass arch)"
        return PathEstimate(links * rate, code, f"{links} NVLink{gen} links × {rate:g} GB/s{assumed}")
    if code not in ("PIX", "PXB", "PHB", "NODE", "SYS"):
        raise ValueError(f"unknown path code {code!r}")
    link = pcie_gbs(pcie_gen, pcie_lanes)
    at = f"Gen{pcie_gen} x{pcie_lanes}"
    staged = PathEstimate(link / 2, code, f"no peer access: staged through host memory, ~half of {at} (often less)",
                          "staged")
    if peer_access is False:
        return staged
    if code in ("PIX", "PXB"):
        return PathEstimate(link, code, f"direct P2P through PCIe switch(es) at {at}")
    if code == "PHB":
        if peer_access:
            return PathEstimate(link, code, f"direct P2P through the CPU root complex at {at} (can be slower: measure)")
        return PathEstimate(link, code, f"upper bound {at}, if peer access works through the root complex; "
                                        "often off (most VMs): then staged, ~half or less", "upper-bound")
    if peer_access:
        return PathEstimate(link, code, f"peer access reported across host bridges: at most {at}, usually far "
                                        "less — measure")
    return PathEstimate(link / 2, code, f"P2P rarely crosses host bridges or sockets: staged through host "
                                        f"memory, ~half of {at} (often less)", "staged")


def predict(topo: Topology, arch: str | None = None, pcie_gen: int = 4, pcie_lanes: int = 16,
            peer_access: dict | None = None) -> dict:
    """``{(gpu_a, gpu_b): PathEstimate}`` for every ordered GPU pair of a parsed topology.
    ``peer_access`` optionally maps ``(gpu_a, gpu_b)`` to what the driver reported."""
    peer_access = peer_access or {}
    return {(a, b): path_bandwidth(topo.link(a, b), arch, pcie_gen, pcie_lanes, peer_access.get((a, b)))
            for a in topo.gpus for b in topo.gpus if a != b}


def ring_allreduce_time(nbytes: float, p: int, alpha: float, bw: float) -> float:
    """Ring all-reduce of ``nbytes`` over ``p`` GPUs: ``2(p−1)`` steps, each paying ``alpha`` and
    moving ``nbytes/p`` over a link of ``bw`` bytes/s: ``2(p−1)·α + 2(p−1)/p · nbytes/bw``."""
    if p < 2:
        return 0.0
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * nbytes / bw


def tp_allreduce_per_token(hidden: int, layers: int, batch: int, dtype_bytes: int, p: int, alpha: float,
                           bw: float, allreduces_per_layer: int = 2) -> float:
    """Communication time per decode step for Megatron-style TP: each layer all-reduces the
    ``batch × hidden`` activations twice (after attention, after the MLP)."""
    s = batch * hidden * dtype_bytes
    return layers * allreduces_per_layer * ring_allreduce_time(s, p, alpha, bw)
