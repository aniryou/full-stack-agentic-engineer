"""A small, dated spec table: the numbers you hold a measurement up against.

Spec sheets quote *peaks*: every SM at boost clock issuing the widest tensor-core instruction
every cycle — and, when there is an asterisk, with 2:4 structured sparsity, which doubles the
figure and which dense inference never sees. This table stores **dense** peaks only
(``float64`` is the FP64 tensor-core rate where one exists, since DGEMM uses it). A GEMM
that reaches 70–85% of the dense peak is healthy; 100% is not a target, it is a ceiling.

The table is vendored (this lab does not import the roofline core) and small on purpose:
the GPUs a learner is likely to rent or get free. Every figure is from the vendor datasheet
as read in 2026-09 — treat each as ``(verify)`` before relying on it, and add your own GPU
by appending a ``GpuSpec``. The primer's §1 ("Spec-sheet literacy") explains every column.

Also here: an estimate of a CPU's peak from its vector width (so the T0 path has a "spec"
to compare against), and the theoretical rates of PCIe and NVLink links.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SPEC_DATE = "2026-09 (verify against vendor datasheets)"


@dataclass(frozen=True)
class GpuSpec:
    key: str
    name: str
    arch: str
    compute_capability: str
    memory_gb: float
    memory_type: str
    mem_bw_gbs: float                  # GB/s, 1 GB = 1e9 bytes
    dense_tflops: dict = field(default_factory=dict)   # dtype -> dense peak TFLOP/s (no sparsity)
    tdp_w: float = 0.0
    host_link: str = ""                # e.g. "PCIe Gen4 x16"
    nvlink_links: int = 0              # NVLink links per GPU (0 = none)
    match: tuple = ()                  # regexes tried against the device name
    note: str = ""

    def peak_flops(self, dtype: str) -> float | None:
        """Dense peak in FLOP/s for a dtype, or None if the GPU has no fast path for it."""
        from .accounting import canonical_dtype
        tf = self.dense_tflops.get(canonical_dtype(dtype))
        return None if tf is None else tf * 1e12

    @property
    def mem_bw(self) -> float:
        """Memory bandwidth in bytes/s."""
        return self.mem_bw_gbs * 1e9

    def ridge(self, dtype: str) -> float | None:
        """FLOPs per byte needed to be compute-bound at this dtype (primer §2)."""
        p = self.peak_flops(dtype)
        return None if p is None else p / self.mem_bw

    @property
    def nvlink_gbs_per_direction(self) -> float:
        return self.nvlink_links * NVLINK_GBS_PER_LINK.get(NVLINK_GEN.get(self.arch, 0), 0.0)


# fmt: off
GPUS = [
    GpuSpec("t4", "NVIDIA T4", "turing", "7.5", 16, "GDDR6", 320,
            {"float32": 8.1, "float16": 65.0}, 70, "PCIe Gen3 x16", 0, (r"\bT4\b",),
            "no BF16/TF32/FP8 tensor cores; 70 W cap makes sustained clocks lower than boost"),
    GpuSpec("p100", "NVIDIA P100 PCIe 16GB", "pascal", "6.0", 16, "HBM2", 732,
            {"float64": 4.7, "float32": 9.3, "float16": 18.7}, 250, "PCIe Gen3 x16", 0, (r"P100",),
            "no tensor cores (Kaggle offers it)"),
    GpuSpec("l4", "NVIDIA L4", "ada", "8.9", 24, "GDDR6", 300,
            {"float32": 30.3, "tf32": 60.0, "float16": 121.0, "bfloat16": 121.0, "float8_e4m3fn": 242.0},
            72, "PCIe Gen4 x16", 0, (r"\bL4\b",), "GCP g2-*; Cloud Run GPU"),
    GpuSpec("a100-40gb", "NVIDIA A100 40GB", "ampere", "8.0", 40, "HBM2", 1555,
            {"float64": 19.5, "float32": 19.5, "tf32": 156.0, "float16": 312.0, "bfloat16": 312.0},
            400, "PCIe Gen4 x16", 12, (r"A100.*40GB",), "GCP a2-highgpu-*; SXM4 400 W, PCIe 250 W"),
    GpuSpec("a100-80gb-pcie", "NVIDIA A100 80GB PCIe", "ampere", "8.0", 80, "HBM2e", 1935,
            {"float64": 19.5, "float32": 19.5, "tf32": 156.0, "float16": 312.0, "bfloat16": 312.0},
            300, "PCIe Gen4 x16", 0, (r"A100.*80GB.*PCIe", r"A100-PCIE-80GB"), ""),
    GpuSpec("a100-80gb", "NVIDIA A100 80GB SXM4", "ampere", "8.0", 80, "HBM2e", 2039,
            {"float64": 19.5, "float32": 19.5, "tf32": 156.0, "float16": 312.0, "bfloat16": 312.0},
            400, "PCIe Gen4 x16", 12, (r"A100.*80GB",), "GCP a2-ultragpu-*"),
    GpuSpec("h100-pcie", "NVIDIA H100 PCIe", "hopper", "9.0", 80, "HBM2e", 2000,
            {"float64": 51.2, "float32": 51.2, "tf32": 378.0, "float16": 756.0, "bfloat16": 756.0, "float8_e4m3fn": 1513.0},
            350, "PCIe Gen5 x16", 0, (r"H100.*PCIe",), ""),
    GpuSpec("h100-sxm", "NVIDIA H100 SXM5 80GB", "hopper", "9.0", 80, "HBM3", 3350,
            {"float64": 67.0, "float32": 67.0, "tf32": 495.0, "float16": 989.0, "bfloat16": 989.0, "float8_e4m3fn": 1979.0},
            700, "PCIe Gen5 x16", 18, (r"H100.*(HBM3|SXM)",), "GCP a3-*; reports as 'NVIDIA H100 80GB HBM3'"),
    GpuSpec("h200", "NVIDIA H200 SXM 141GB", "hopper", "9.0", 141, "HBM3e", 4800,
            {"float64": 67.0, "float32": 67.0, "tf32": 495.0, "float16": 989.0, "bfloat16": 989.0, "float8_e4m3fn": 1979.0},
            700, "PCIe Gen5 x16", 18, (r"(?<!G)H200",), "H100 compute, more and faster memory; GCP a3-ultragpu"),
    GpuSpec("b200", "NVIDIA B200 (HGX)", "blackwell", "10.0", 180, "HBM3e", 8000,
            {"tf32": 1100.0, "float16": 2250.0, "bfloat16": 2250.0, "float8_e4m3fn": 4500.0},
            1000, "PCIe Gen5 x16", 18, (r"(?<!G)B200",), "FP4 dense ~9,000 TFLOP/s; GCP a4-highgpu-8g"),
    GpuSpec("rtx4090", "NVIDIA GeForce RTX 4090", "ada", "8.9", 24, "GDDR6X", 1008,
            {"float32": 82.6, "tf32": 82.6, "float16": 165.2, "bfloat16": 165.2},
            450, "PCIe Gen4 x16", 0, (r"RTX 4090",),
            "fp16/bf16 figure is with FP32 accumulate (what PyTorch uses); GeForce halves it vs FP16 accumulate"),
]
# fmt: on


def lookup(device_name: str) -> GpuSpec | None:
    """Find the spec entry for a device name as reported by ``nvidia-smi`` / ``torch``."""
    for spec in GPUS:
        if any(re.search(p, device_name or "", flags=re.I) for p in spec.match):
            return spec
    return None


def get(key: str) -> GpuSpec:
    for spec in GPUS:
        if spec.key == key:
            return spec
    raise KeyError(f"no spec {key!r}; known: {[s.key for s in GPUS]}")


# -- CPUs: a peak you can compute from the instruction set ---------------------------------------
def simd_bits_from_flags(flags) -> int:
    """Widest vector unit advertised by /proc/cpuinfo flags (x86) or 128 for Arm NEON/ASIMD."""
    flags = set(flags or ())
    if "avx512f" in flags:
        return 512
    if "avx2" in flags or "avx" in flags:
        return 256
    return 128


def cpu_peak_flops(cores: int, ghz: float, simd_bits: int, fma_units: int = 2, dtype_bytes: int = 8) -> float:
    """Peak FLOP/s = cores × clock × lanes × 2 (an FMA is a multiply and an add) × FMA units.

    ``lanes = simd_bits / (8·dtype_bytes)``: AVX-512 holds 8 fp64 or 16 fp32 lanes. Most
    server x86 cores have two FMA units; some have one — and heavy AVX-512 code often runs
    below the base clock. So this is an *estimate* (verify), not a spec.
    """
    lanes = simd_bits // (8 * dtype_bytes)
    return cores * ghz * 1e9 * lanes * 2 * fma_units


# -- links ------------------------------------------------------------------------------------------
# PCIe generation -> (GT/s per lane, line-encoding efficiency)
PCIE = {1: (2.5, 8 / 10), 2: (5.0, 8 / 10), 3: (8.0, 128 / 130), 4: (16.0, 128 / 130),
        5: (32.0, 128 / 130), 6: (64.0, 242 / 256)}   # Gen6: PAM4 + FLIT framing (verify)


def pcie_gbs(gen: int, lanes: int = 16) -> float:
    """Theoretical PCIe bandwidth per direction, GB/s: GT/s × lanes × encoding ÷ 8 bits.

    Gen3 x16 ≈ 15.8, Gen4 x16 ≈ 31.5, Gen5 x16 ≈ 63.0. Packet headers and flow control take
    another ~10–20%, so a good pinned host→device copy lands at roughly 80–90% of this.
    """
    gts, eff = PCIE[int(gen)]
    return gts * lanes * eff / 8


NVLINK_GEN = {"pascal": 1, "volta": 2, "ampere": 3, "hopper": 4, "blackwell": 5}
NVLINK_GBS_PER_LINK = {1: 20.0, 2: 25.0, 3: 25.0, 4: 25.0, 5: 50.0}   # GB/s per link, per direction


def nvlink_gbs(n_links: int, gen: int) -> float:
    """NVLink bandwidth per direction: links × per-link rate. H100: 18 × 25 = 450 GB/s each way
    (NVIDIA quotes 900 GB/s: both directions added)."""
    return n_links * NVLINK_GBS_PER_LINK[gen]
