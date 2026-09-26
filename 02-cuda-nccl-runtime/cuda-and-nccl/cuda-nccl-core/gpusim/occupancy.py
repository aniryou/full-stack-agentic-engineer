"""Occupancy: an SM hides memory latency by switching among resident warps, and how many warps can
be resident is capped by whichever per-SM resource a block uses up first: warp slots, block
slots, registers or shared memory.

`occupancy()` follows the arithmetic of NVIDIA's occupancy calculator (`cuda_occupancy.h`):
registers are allocated per warp in 256-register units and split across the SM's 4
sub-partitions; shared memory is allocated in 128-byte units (256 before CC 8.0) plus 1 KB the
driver reserves per block (CC 8.0+). It assumes the largest shared-memory carveout; the real
answer for a compiled kernel is `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
`bytes_in_flight()` is Little's law, the reason occupancy matters at all.
"""
from __future__ import annotations

from dataclasses import dataclass

KB = 1024


@dataclass(frozen=True)
class SM:
    name: str
    max_threads: int            # resident threads per SM
    max_blocks: int             # resident blocks per SM
    smem: int                   # max shared memory per SM (bytes, largest carveout)
    smem_per_block: int         # max shared memory one block may use (opt-in above 48 KB)
    reserved_smem: int          # driver-reserved shared memory per block
    smem_unit: int              # shared-memory allocation granularity
    regs: int = 64 * KB         # 32-bit registers per SM
    regs_per_block: int = 64 * KB
    max_regs_per_thread: int = 255
    reg_unit: int = 256         # registers are allocated per warp in units of 256
    sub_partitions: int = 4     # each SM is 4 sub-partitions, each with its own register file slice
    max_threads_per_block: int = 1024

    @property
    def max_warps(self) -> int:
        return self.max_threads // 32


# Per-compute-capability limits: "Technical Specifications per Compute Capability", CUDA C++
# Programming Guide (checked Sep 2026; verify new architectures before relying on them).
SMS = {
    "7.0": SM("Volta (V100)", 2048, 32, 96 * KB, 96 * KB, 0, 256),
    "7.5": SM("Turing (T4)", 1024, 16, 64 * KB, 64 * KB, 0, 256),
    "8.0": SM("Ampere (A100)", 2048, 32, 164 * KB, 163 * KB, 1 * KB, 128),
    "8.6": SM("Ampere (A10G, A40, RTX 30)", 1536, 16, 100 * KB, 99 * KB, 1 * KB, 128),
    "8.9": SM("Ada (L4, L40S, RTX 40)", 1536, 24, 100 * KB, 99 * KB, 1 * KB, 128),
    "9.0": SM("Hopper (H100, H200)", 2048, 32, 228 * KB, 227 * KB, 1 * KB, 128),
    "10.0": SM("Blackwell (B200, GB200)", 2048, 32, 228 * KB, 227 * KB, 1 * KB, 128),
}

# Spec-sheet values used by the worked examples (verify; the dated catalogue is layer 01's job).
DEVICES = {
    "T4": {"cc": "7.5", "sms": 40, "hbm_Bps": 320e9},
    "L4": {"cc": "8.9", "sms": 58, "hbm_Bps": 300e9},
    "A100-80GB": {"cc": "8.0", "sms": 108, "hbm_Bps": 2.039e12},
    "H100-SXM": {"cc": "9.0", "sms": 132, "hbm_Bps": 3.35e12},
}


def _round_up(x: int, unit: int) -> int:
    return -(-x // unit) * unit


@dataclass
class Occupancy:
    blocks_per_sm: int
    warps_per_block: int
    max_warps: int
    limits: dict        # blocks per SM each resource allows

    @property
    def active_warps(self) -> int:
        return self.blocks_per_sm * self.warps_per_block

    @property
    def occupancy(self) -> float:
        return self.active_warps / self.max_warps

    @property
    def limiter(self) -> str:
        """The resource(s) that set blocks_per_sm, e.g. 'registers' or 'warps+shared_mem'."""
        return "+".join(k for k, v in self.limits.items() if v == self.blocks_per_sm)


def occupancy(threads_per_block: int, regs_per_thread: int = 32, smem_per_block: int = 0,
              cc: str = "8.9") -> Occupancy:
    """Resident blocks and warps per SM for one kernel launch configuration."""
    sm = SMS[cc]
    if not 0 < threads_per_block <= sm.max_threads_per_block:
        raise ValueError(f"threads_per_block must be 1..{sm.max_threads_per_block}")
    wpb = -(-threads_per_block // 32)                    # warps are allocated whole
    limits = {"warps": sm.max_warps // wpb, "blocks": sm.max_blocks}

    regs_per_warp = _round_up(regs_per_thread * 32, sm.reg_unit)
    if regs_per_thread > sm.max_regs_per_thread or \
            regs_per_warp * _round_up(wpb, sm.sub_partitions) > sm.regs_per_block:
        limits["registers"] = 0                          # the launch fails outright
    elif regs_per_warp:
        warps_per_partition = (sm.regs // sm.sub_partitions) // regs_per_warp
        limits["registers"] = warps_per_partition * sm.sub_partitions // wpb
    else:
        limits["registers"] = sm.max_blocks

    if smem_per_block > sm.smem_per_block:
        limits["shared_mem"] = 0
    else:
        need = _round_up(smem_per_block + sm.reserved_smem, sm.smem_unit)
        limits["shared_mem"] = sm.smem // need if need else sm.max_blocks

    return Occupancy(min(limits.values()), wpb, sm.max_warps, limits)


def waves(grid_blocks: int, n_sms: int, blocks_per_sm: int) -> dict:
    """A grid runs in waves of n_sms x blocks_per_sm blocks; a partial last wave idles SMs."""
    per_wave = n_sms * blocks_per_sm
    w = -(-grid_blocks // per_wave)
    return {"waves": w, "efficiency": grid_blocks / (w * per_wave)}


def bytes_in_flight(bandwidth_Bps: float, latency_s: float) -> float:
    """Little's law: to sustain `bandwidth` with `latency` per request, this many bytes must be
    outstanding at every instant, spread over all SMs."""
    return bandwidth_Bps * latency_s


def loads_in_flight_per_thread(device: str, latency_s: float, bytes_per_load: int = 4,
                               threads_per_sm: int | None = None) -> float:
    """Independent loads each resident thread must keep outstanding to saturate DRAM."""
    d = DEVICES[device]
    threads = threads_per_sm or SMS[d["cc"]].max_threads
    per_sm = bytes_in_flight(d["hbm_Bps"], latency_s) / d["sms"]
    return per_sm / (threads * bytes_per_load)
