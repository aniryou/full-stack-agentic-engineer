"""Sharing one GPU: MIG *partitions* the hardware (fixed shapes, hard isolation), MPS *overlaps*
processes on the SMs (efficient, weakly isolated), time-slicing *takes turns* (simple, no
isolation, and latency multiplies with the number of busy tenants).

MIG geometry follows `nvidia-smi mig -lgipp` (verify for your GPU and driver). The GPU has 7
compute slices and 8 memory slices. Compute slice i sits beside memory slice i. A profile
"Ng.Mgb" takes N compute slices and a fixed number of memory slices, and it may only *start* at
certain slice indices. A layout is valid when the memory ranges don't overlap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MigProfile:
    name: str
    compute: int          # compute slices (of 7)
    mem_slices: int       # memory slices (of 8)
    starts: tuple         # memory-slice indices a GPU instance of this profile may start at

    @property
    def max_count(self) -> int:
        return len(self.starts)


_GEOMETRY = [(1, 1, (0, 1, 2, 3, 4, 5, 6)), (1, 2, (0, 2, 4, 6)), (2, 2, (0, 2, 4)),
             (3, 4, (0, 4)), (4, 4, (0,)), (7, 8, (0,))]


def _profiles(*names):
    return [MigProfile(n, c, m, s) for n, (c, m, s) in zip(names, _GEOMETRY)]


# MIG profiles per GPU (NVIDIA MIG User Guide, "Supported MIG Profiles"; verify, plus the "+me"
# media-extension variants, which are left out here). L4, T4, A10 and RTX 40 have no MIG.
MIG = {
    "A100-40GB": _profiles("1g.5gb", "1g.10gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"),
    "A100-80GB": _profiles("1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"),
    "H100-80GB": _profiles("1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"),
    "H200-141GB": _profiles("1g.18gb", "1g.35gb", "2g.35gb", "3g.71gb", "4g.71gb", "7g.141gb"),
}


def _lookup(gpu: str) -> dict:
    return {p.name: p for p in MIG[gpu]}


def pack(gpu: str, requests) -> list | None:
    """Find a start slice for every requested profile so that they coexist (backtracking,
    biggest first). Returns [(profile, start), ...] sorted by start, or None if impossible."""
    profs = _lookup(gpu)
    items = sorted(requests, key=lambda n: (-profs[n].mem_slices, -profs[n].compute))
    used, placed = [False] * 8, []

    def place(i: int) -> bool:
        if i == len(items):
            return True
        p = profs[items[i]]
        for s in p.starts:
            span = range(s, s + p.mem_slices)
            if not any(used[j] for j in span):
                for j in span:
                    used[j] = True
                placed.append((p.name, s))
                if place(i + 1):
                    return True
                placed.pop()
                for j in span:
                    used[j] = False
        return False

    return sorted(placed, key=lambda x: x[1]) if place(0) else None


def first_fit(gpu: str, arrivals) -> list:
    """Create instances one at a time, in arrival order, each at its first free start slice,
    with no planning ahead. Returns [(profile, start or None)]: None means it did not fit."""
    profs, used, out = _lookup(gpu), [False] * 8, []
    for name in arrivals:
        p = profs[name]
        start = next((s for s in p.starts if not any(used[s:s + p.mem_slices])), None)
        if start is not None:
            used[start:start + p.mem_slices] = [True] * p.mem_slices
        out.append((name, start))
    return out


def layout(gpu: str, placement) -> str:
    """ASCII picture of the 8 memory slices, e.g. '[3g.40gb   ][2g.20][1g][1g]'."""
    profs, cells = _lookup(gpu), ["  .  "] * 8
    for name, start in placement:
        if start is None:
            continue
        span = profs[name].mem_slices
        label = name.split(".")[0]
        for j in range(start, start + span):
            cells[j] = f"{label:^5}" if j == start else "  -  "
    return "|" + "|".join(cells) + "|   (memory slices 0-7; compute slices 0-6 sit under 0-6)"


def timeslice_latency(work_ms: float, tenants: int, quantum_ms: float = 2.0,
                      switch_ms: float = 0.05) -> dict:
    """Time-slicing: the GPU runs one context at a time, round-robin, `quantum_ms` each, paying
    `switch_ms` per context switch. Our request needs `work_ms` of GPU time, and the other
    tenants - 1 contexts always have work. best/mean/worst depend on when in the round we arrive."""
    if tenants <= 1:
        return {"best": work_ms, "mean": work_ms, "worst": work_ms}
    k = math.ceil(work_ms / quantum_ms)                         # quanta we need
    gap = (tenants - 1) * (quantum_ms + switch_ms) + switch_ms  # others' turns between two of ours
    best = work_ms + (k - 1) * gap
    return {"best": best, "mean": best + gap / 2, "worst": best + gap}


def shared_latency(work_ms: float, tenants: int, mode: str, util: float = 1.0,
                   quantum_ms: float = 2.0, switch_ms: float = 0.05) -> float:
    """Idealised latency of one request when `tenants` identical, always-busy tenants share a GPU.
    `util` is the fraction of the whole GPU the request's kernels keep busy when they run alone
    (small-batch decode of a small model is well below 1). SM capacity only: memory bandwidth
    and L2 contention would make MPS and time-slicing worse than this.

    exclusive     work
    time_slicing  one context at a time: mean of timeslice_latency (about tenants x work)
    mps           kernels of all tenants overlap until the SMs are full: work x max(1, tenants x util)
    mig           each tenant owns 7 // tenants compute slices: work x max(1, util / (slices / 7))
    """
    if mode == "exclusive":
        return work_ms
    if mode == "time_slicing":
        return timeslice_latency(work_ms, tenants, quantum_ms, switch_ms)["mean"]
    if mode == "mps":
        return work_ms * max(1.0, tenants * util)
    if mode == "mig":
        share = max(1, 7 // tenants) / 7
        return work_ms * max(1.0, util / share)
    raise ValueError(mode)


# What each mechanism guarantees (NVIDIA MIG / MPS docs and the GPU Operator time-slicing docs; verify).
TRAITS = {
    "mig": {"memory isolation": "yes (own memory slices)", "fault isolation": "yes",
            "performance isolation": "yes (own SMs, L2 slice, memory bandwidth)",
            "granularity": "fixed profiles, up to 7 per GPU", "gpus": "A100, A30, H100, H200, B200 class"},
    "mps": {"memory isolation": "address spaces yes; capacity only if limited per client",
            "fault isolation": "weak: a fatal fault can take down other clients of the server",
            "performance isolation": "partial (optional active-thread % per client)",
            "granularity": "any number of processes", "gpus": "Volta and newer"},
    "time_slicing": {"memory isolation": "no: tenants can OOM each other", "fault isolation": "no guarantee",
                     "performance isolation": "no: latency grows with busy tenants",
                     "granularity": "any number of replicas", "gpus": "any"},
}
