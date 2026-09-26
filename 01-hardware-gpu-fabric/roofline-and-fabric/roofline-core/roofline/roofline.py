"""The roofline model: which ceiling -- compute or memory -- limits a kernel.

The one idea: a kernel that performs F FLOPs while moving B bytes cannot finish
faster than max(F / peak, B / bandwidth). F / B is its arithmetic intensity I;
the device's ridge point peak / bandwidth is where the two ceilings meet.

    attainable(I) = min(peak, I x bandwidth)            [FLOP/s]

Below the ridge you are memory-bound (move fewer bytes: fuse, tile, quantize,
batch); above it you are compute-bound (use a narrower precision, buy FLOPs).

Byte counts here are *compulsory* traffic: every operand read once, every result
written once, as a perfectly tiled and fused kernel would do. Real kernels move
more (``tiled_gemm_bytes``) and reach a fraction of both ceilings -- measure
yours with gpu-bench-lab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .specs import Device


@dataclass(frozen=True)
class Kernel:
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes


@dataclass(frozen=True)
class Timing:
    kernel: Kernel
    t_compute: float      # seconds at the compute ceiling
    t_memory: float       # seconds at the memory ceiling

    @property
    def time(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound(self) -> str:
        return "compute" if self.t_compute >= self.t_memory else "memory"

    @property
    def achieved(self) -> float:
        """FLOP/s actually delivered at the roofline time."""
        return self.kernel.flops / self.time


def ridge_point(device: Device, precision: str = "bf16") -> float:
    """FLOP/byte where memory time equals compute time. H100 bf16: 989.4e12 / 3.35e12 = 295."""
    return device.peak(precision) / device.bandwidth()


def attainable(intensity: float, device: Device, precision: str = "bf16") -> float:
    """The roofline: min(peak, I x bandwidth), in FLOP/s."""
    return min(device.peak(precision), intensity * device.bandwidth())


def time_kernel(kernel: Kernel, device: Device, precision: str = "bf16",
                compute_eff: float = 1.0, memory_eff: float = 1.0) -> Timing:
    """Roofline time of one kernel. Efficiencies < 1 model achievable fractions of peak."""
    return Timing(kernel,
                  kernel.flops / (device.peak(precision) * compute_eff),
                  kernel.bytes / (device.bandwidth() * memory_eff))


# -- kernels: FLOPs and compulsory bytes ------------------------------------------
def elementwise(n: int, bytes_per_el: float = 2, n_in: int = 2, n_out: int = 1,
                flops_per_el: float = 1, name: str = "elementwise") -> Kernel:
    """z = x + y over n elements: n FLOPs, (2 + 1) x n x b bytes -> 1/6 FLOP/B at bf16."""
    return Kernel(name, flops_per_el * n, (n_in + n_out) * n * bytes_per_el)


def reduction(n: int, bytes_per_el: float = 4, name: str = "reduction") -> Kernel:
    """sum(x) over n elements: ~n adds, n x b bytes read -> 1/4 FLOP/B at fp32."""
    return Kernel(name, n, n * bytes_per_el)


def gemm(m: int, n: int, k: int, bytes_per_el: float = 2, name: str | None = None) -> Kernel:
    """C[m,n] = A[m,k] @ B[k,n]: 2mnk FLOPs over (mk + kn + mn) x b bytes.

    Square n: intensity = 2n / (3b) -- 1,365 FLOP/B at n = 4096, bf16.
    m = 1 (one token through a weight matrix): intensity ~ 2 / b = 1 FLOP/B at bf16.
    """
    return Kernel(name or f"gemm {m}x{n}x{k}", 2 * m * n * k, (m * k + k * n + m * n) * bytes_per_el)


# -- tiling: why a bigger on-chip tile raises intensity ---------------------------------
def tile_intensity(tile_m: int, tile_n: int, bytes_per_el: float = 2) -> float:
    """FLOP per byte fetched when a Tm x Tn output tile streams its A and B panels once.

    Each k-step loads (Tm + Tn) elements and does 2 Tm Tn FLOPs:
    I = 2 Tm Tn / ((Tm + Tn) b). 128x128 bf16 -> 64; 128x256 -> 85; independent of k.
    """
    return 2 * tile_m * tile_n / ((tile_m + tile_n) * bytes_per_el)


def tiled_gemm_bytes(m: int, n: int, k: int, tile_m: int, tile_n: int,
                     bytes_per_el: float = 2) -> float:
    """Bytes fetched from the next level down by a tiled GEMM with no reuse between tiles.

    Every output tile re-reads its A panel (Tm x k) and B panel (k x Tn):
    bytes = (mnk (1/Tn + 1/Tm) + mn) b. Tiles of 1x1 give the naive 2 loads per FMA.
    """
    return (m * n * k * (1 / tile_n + 1 / tile_m) + m * n) * bytes_per_el


def tile_smem_bytes(tile_m: int, tile_n: int, tile_k: int, bytes_per_el: float = 2,
                    stages: int = 1) -> float:
    """Shared memory to hold `stages` pipelined A and B tiles: stages x (Tm + Tn) x Tk x b."""
    return stages * (tile_m + tile_n) * tile_k * bytes_per_el


def fusion_bytes(n: int, n_ops: int, bytes_per_el: float = 2, fused: bool = True) -> float:
    """HBM traffic of a chain of n_ops elementwise ops on n elements (one in, one out each).

    Unfused, every op round-trips through HBM: 2 n b n_ops. Fused: 2 n b once.
    """
    return 2 * n * bytes_per_el * (1 if fused else n_ops)


# -- a dependency-free picture ----------------------------------------------------------
def ascii_roofline(device: Device, kernels=(), precision: str = "bf16",
                   width: int = 64, height: int = 14, i_range=(0.1, 10_000)) -> str:
    """Log-log roofline as text. Each kernel is drawn at its attainable FLOP/s."""
    lo_i, hi_i = i_range
    peak = device.peak(precision)
    lo_p, hi_p = attainable(lo_i, device, precision) / 2, peak * 2

    def col(i):
        return round((math.log10(i) - math.log10(lo_i)) / (math.log10(hi_i) - math.log10(lo_i)) * (width - 1))

    def row(p):
        return height - 1 - round((math.log10(p) - math.log10(lo_p)) / (math.log10(hi_p) - math.log10(lo_p)) * (height - 1))

    grid = [[" "] * width for _ in range(height)]
    for c in range(width):
        i = 10 ** (math.log10(lo_i) + c / (width - 1) * (math.log10(hi_i) - math.log10(lo_i)))
        p = attainable(i, device, precision)
        grid[row(p)][c] = "=" if p >= peak else "/"
    legend = []
    for idx, k in enumerate(kernels):
        mark = chr(ord("A") + idx)
        i = min(max(k.intensity, lo_i), hi_i)
        grid[row(attainable(i, device, precision))][col(i)] = mark
        t = time_kernel(k, device, precision)
        legend.append(f"  {mark}: {k.name:<28} I = {k.intensity:9.2f} FLOP/B  -> {t.bound}-bound, "
                      f"{t.achieved / 1e12:8.2f} TFLOP/s")
    title = (f"{device.name} {precision}: peak {peak / 1e12:,.0f} TFLOP/s, "
             f"{device.memory_tbs:g} TB/s, ridge {ridge_point(device, precision):.0f} FLOP/B")
    axis = f"  I = {lo_i:g} FLOP/B" + " " * (width - 30) + f"I = {hi_i:,.0f}"
    return "\n".join([title] + ["|" + "".join(r) for r in grid] + ["+" + "-" * width, axis] + legend)
