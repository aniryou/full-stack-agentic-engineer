"""What one GPU node can actually give a pod: machine shape minus the kubelet's reservations.

The one idea: a GPU node is a bundle — N GPUs *plus* a fixed amount of CPU and memory. If a
1-GPU pod asks for more than 1/N of the node's allocatable CPU or memory, the node runs out
of CPU before it runs out of GPUs, and the remaining GPUs are *stranded*: paid for, idle, and
unschedulable. ``per_gpu_share`` and ``stranded_gpus`` make that arithmetic explicit.

The allocatable formula is GKE's published node-reservation schedule (verify for your
version); machine shapes are dated Sep 2026 (verify). Other clouds and kubeadm clusters
reserve differently — the stranding arithmetic is the same.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass(frozen=True)
class Machine:
    name: str
    vcpus: int
    memory_gb: float        # as advertised (GB); treated as GiB for the reservation math (verify)
    gpus: int
    gpu_type: str           # GKE accelerator label value
    note: str = ""


# Machine shapes (Sep 2026, verify on the GCE machine-types page before relying on them).
CATALOG: dict[str, Machine] = {m.name: m for m in [
    Machine("g2-standard-4", 4, 16, 1, "nvidia-l4"),
    Machine("g2-standard-8", 8, 32, 1, "nvidia-l4"),
    Machine("g2-standard-12", 12, 48, 1, "nvidia-l4"),
    Machine("g2-standard-16", 16, 64, 1, "nvidia-l4"),
    Machine("g2-standard-24", 24, 96, 2, "nvidia-l4"),
    Machine("g2-standard-32", 32, 128, 1, "nvidia-l4"),
    Machine("g2-standard-48", 48, 192, 4, "nvidia-l4"),
    Machine("g2-standard-96", 96, 384, 8, "nvidia-l4"),
    Machine("n1-standard-8-t4", 8, 30, 1, "nvidia-tesla-t4", "N1 + one attached T4"),
    Machine("a2-highgpu-1g", 12, 85, 1, "nvidia-tesla-a100"),
    Machine("a2-highgpu-8g", 96, 680, 8, "nvidia-tesla-a100"),
    Machine("a2-ultragpu-1g", 12, 170, 1, "nvidia-a100-80gb"),
    Machine("a3-highgpu-8g", 208, 1872, 8, "nvidia-h100-80gb"),
    Machine("a3-ultragpu-8g", 224, 2952, 8, "nvidia-h200-141gb"),
    Machine("a4-highgpu-8g", 224, 3968, 8, "nvidia-b200"),
]}


def gke_reserved_cpu(vcpus: float) -> float:
    """CPU (cores) GKE reserves for system daemons: 6% of the 1st core, 1% of the 2nd,
    0.5% of cores 3-4, 0.25% of every core above 4 (verify)."""
    tiers = [(1, 0.06), (1, 0.01), (2, 0.005), (math.inf, 0.0025)]
    left, total = vcpus, 0.0
    for size, rate in tiers:
        take = min(left, size)
        total += take * rate
        left -= take
        if left <= 0:
            break
    return total


def gke_reserved_memory_gib(memory_gib: float) -> float:
    """Memory (GiB) GKE reserves: 255 MiB below 1 GiB; else 25% of the first 4 GiB, 20% of the
    next 4, 10% of the next 8, 6% of the next 112, 2% above 128 GiB; plus a 100 MiB eviction
    threshold (verify)."""
    if memory_gib < 1:
        return 255 / 1024 + 100 / 1024
    tiers = [(4, 0.25), (4, 0.20), (8, 0.10), (112, 0.06), (math.inf, 0.02)]
    left, total = memory_gib, 0.0
    for size, rate in tiers:
        take = min(left, size)
        total += take * rate
        left -= take
        if left <= 0:
            break
    return total + 100 / 1024


@dataclass(frozen=True)
class Allocatable:
    cpu: float          # cores
    memory_gib: float
    gpus: int


def allocatable(machine: Machine | str, *, daemonset_cpu: float = 0.0, daemonset_mem_gib: float = 0.0) -> Allocatable:
    """Node allocatable after GKE reservations, optionally minus DaemonSet requests."""
    m = CATALOG[machine] if isinstance(machine, str) else machine
    cpu = m.vcpus - gke_reserved_cpu(m.vcpus) - daemonset_cpu
    mem = m.memory_gb - gke_reserved_memory_gib(m.memory_gb) - daemonset_mem_gib
    return Allocatable(cpu=round(cpu, 4), memory_gib=round(mem, 4), gpus=m.gpus)


def per_gpu_share(machine: Machine | str, **kw) -> Allocatable:
    """The CPU and memory a 1-GPU pod can ask for without stranding the node's other GPUs."""
    a = allocatable(machine, **kw)
    return Allocatable(cpu=round(a.cpu / a.gpus, 4), memory_gib=round(a.memory_gib / a.gpus, 4), gpus=1)


def pods_per_node(alloc: Allocatable, *, pod_cpu: float, pod_mem_gib: float, pod_gpus: int) -> int:
    """How many identical pods fit on an empty node: the minimum over every resource."""
    limits = [math.floor(alloc.cpu / pod_cpu) if pod_cpu > 0 else math.inf,
              math.floor(alloc.memory_gib / pod_mem_gib) if pod_mem_gib > 0 else math.inf,
              alloc.gpus // pod_gpus if pod_gpus > 0 else math.inf]
    n = min(limits)
    return int(n) if n != math.inf else 0


def stranded_gpus(alloc: Allocatable, *, pod_cpu: float, pod_mem_gib: float, pod_gpus: int) -> int:
    """GPUs left idle on a node packed with these pods because CPU or memory ran out first."""
    n = pods_per_node(alloc, pod_cpu=pod_cpu, pod_mem_gib=pod_mem_gib, pod_gpus=pod_gpus)
    return alloc.gpus - n * pod_gpus


def parse_cpu(q: str | int | float) -> float:
    """Kubernetes CPU quantity -> cores ('500m' -> 0.5, '2' -> 2.0)."""
    s = str(q)
    return float(s[:-1]) / 1000 if s.endswith("m") else float(s)


_MEM_UNITS = {"Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3, "Ti": 1024 ** 4,
              "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_memory_gib(q: str | int | float) -> float:
    """Kubernetes memory quantity -> GiB ('64Mi' -> 0.0625, '16Gi' -> 16)."""
    s = str(q)
    for unit in sorted(_MEM_UNITS, key=len, reverse=True):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * _MEM_UNITS[unit] / GIB
    return float(s) / GIB
