"""Reading ``nvidia-smi topo -m``: which GPUs share NVLink, a PCIe switch, a NIC, a NUMA node.

The matrix names the *path* between every pair of devices, best to worst:

    NV#   a bonded set of # NVLinks (through NVSwitch on HGX boards: every pair shows NV18 on H100)
    PIX   at most one PCIe bridge (same PCIe switch)
    PXB   several PCIe bridges, but not the host bridge
    PHB   through a PCIe host bridge (the CPU's root complex)
    NODE  across PCIe host bridges within one NUMA node
    SYS   across the inter-socket link (QPI/UPI/Infinity Fabric) between NUMA nodes

Three placement decisions come straight out of it (primer §5, "Fabrics quantitatively"):
put a tensor-parallel group where every pair is NV# (``best_group``); give each GPU the NIC on
its own PCIe switch for GPUDirect RDMA (``nearest_nic``); and pin the process that feeds a GPU
to the CPU cores of its NUMA node (``cpu_list``).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import resources
from itertools import combinations

LEGEND = {
    "X": "self",
    "NV#": "bonded set of # NVLinks",
    "PIX": "at most a single PCIe bridge",
    "PXB": "multiple PCIe bridges (not the host bridge)",
    "PHB": "a PCIe host bridge (typically the CPU)",
    "NODE": "PCIe + interconnect between host bridges within a NUMA node",
    "SYS": "PCIe + the SMP interconnect between NUMA nodes (QPI/UPI)",
}
_CLASS = {"SYS": 0, "NODE": 1, "PHB": 2, "PXB": 3, "PIX": 4}
EXTRA_COLUMNS = ("CPU Affinity", "NUMA Affinity", "GPU NUMA ID")


def link_rank(code: str) -> tuple:
    """Sortable quality of a path: ``(class, links)``; higher is better. NV18 > NV12 > PIX > ... > SYS."""
    code = code.strip()
    if code == "X":
        return (9, 0)
    m = re.fullmatch(r"NV(\d+)", code)
    if m:
        return (5, int(m.group(1)))
    return (_CLASS.get(code, -1), 0)


def _cells(line: str) -> list:
    """Split a matrix line on tabs or runs of 2+ spaces (copy-pasted output loses its tabs)."""
    return [c.strip() for c in re.split(r"\t|\s{2,}", line.strip()) if c.strip()]


@dataclass
class Topology:
    columns: list
    matrix: dict                          # row label -> {column label: code}
    cpu_affinity: dict = field(default_factory=dict)
    numa_affinity: dict = field(default_factory=dict)
    nic_names: dict = field(default_factory=dict)

    @property
    def gpus(self) -> list:
        return [c for c in self.columns if re.fullmatch(r"GPU\d+", c)]

    @property
    def nics(self) -> list:
        return [c for c in self.columns if c not in self.gpus and c not in EXTRA_COLUMNS]

    def link(self, a: str, b: str) -> str:
        return self.matrix[a][b]

    def nvlinks(self, a: str, b: str) -> int:
        m = re.fullmatch(r"NV(\d+)", self.link(a, b))
        return int(m.group(1)) if m else 0

    def gpu_pairs(self) -> list:
        return [(a, b, self.link(a, b)) for a, b in combinations(self.gpus, 2)]

    def summary(self) -> str:
        lines = [f"{len(self.gpus)} GPUs, {len(self.nics)} NICs"]
        codes = sorted({c for _, _, c in self.gpu_pairs()}, key=link_rank, reverse=True)
        lines.append("GPU-GPU paths present: " + ", ".join(codes))
        for g in self.gpus:
            nic = nearest_nic(self, g)
            nic_s = f"{nic[0]} ({self.nic_names.get(nic[0], nic[0])}, {nic[1]})" if nic else "none"
            lines.append(f"  {g}: NUMA {self.numa_affinity.get(g, '?')}, CPUs {self.cpu_affinity.get(g, '?')}, "
                         f"nearest NIC {nic_s}")
        return "\n".join(lines)


def parse(text: str) -> Topology:
    """Parse the text printed by ``nvidia-smi topo -m`` (tabs or spaces; with or without legends)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "GPU0" in _cells(l)), None)
    if start is None:
        raise ValueError("no header row with GPU0 found — is this `nvidia-smi topo -m` output?")
    columns = _cells(lines[start])
    matrix, cpu, numa, nics = {}, {}, {}, {}
    for line in lines[start + 1:]:
        cells = _cells(line)
        if not cells or cells[0].rstrip(":") in ("Legend", "NIC Legend"):
            break
        label, values = cells[0], cells[1:]
        row = dict(zip(columns, values))
        matrix[label] = {c: row[c] for c in columns if c not in EXTRA_COLUMNS and c in row}
        if row.get("CPU Affinity"):
            cpu[label] = row["CPU Affinity"]
        if row.get("NUMA Affinity"):
            numa[label] = row["NUMA Affinity"]
    for line in lines:
        m = re.match(r"^\s*(NIC\d+)\s*:\s*(\S+)\s*$", line)
        if m:
            nics[m.group(1)] = m.group(2)
    return Topology(columns, matrix, cpu, numa, nics)


def parse_cpu_list(spec: str) -> list:
    """``"0-3,8-11"`` -> ``[0, 1, 2, 3, 8, 9, 10, 11]``; ``"N/A"`` -> ``[]``."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or part.upper() == "N/A":
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def cpu_list(topo: Topology, gpu: str) -> list:
    """CPU cores local to ``gpu`` (pin its data-loading / serving process here)."""
    return parse_cpu_list(topo.cpu_affinity.get(gpu, ""))


def best_group(topo: Topology, k: int) -> tuple:
    """The ``k`` GPUs whose *worst* pairwise path is best (ties: better overall, then lower ids)."""
    if k > len(topo.gpus):
        raise ValueError(f"asked for {k} GPUs, topology has {len(topo.gpus)}")
    if k == 1:
        return (topo.gpus[0],)

    def score(group):
        ranks = sorted(link_rank(topo.link(a, b)) for a, b in combinations(group, 2))
        return (ranks[0], ranks)

    return max(combinations(topo.gpus, k), key=score)


def nearest_nic(topo: Topology, gpu: str):
    """``(nic, path)`` for the NIC with the best path to ``gpu``, or None if there are no NICs."""
    if not topo.nics:
        return None
    nic = max(topo.nics, key=lambda n: link_rank(topo.link(gpu, n)))
    return nic, topo.link(gpu, nic)


# -- fixtures and the real thing ----------------------------------------------------------------------
FIXTURES = {
    "hgx-h100-8gpu": "topo_hgx_h100_8gpu.txt",
    "pcie-4gpu-2socket": "topo_pcie_4gpu_2socket.txt",
    "a2-highgpu-2g": "topo_a2_highgpu_2g.txt",
    "kaggle-2xt4": "topo_2x_t4_pcie.txt",
}


def load_fixture(name: str) -> str:
    """Sample output in the documented format (illustrative) — not a measurement of any machine."""
    return resources.files("gpubench").joinpath("fixtures").joinpath(FIXTURES.get(name, name)).read_text()


def run_nvidia_smi() -> str | None:
    """The local machine's ``nvidia-smi topo -m``, or None when there is no NVIDIA driver."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30,
                              check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
