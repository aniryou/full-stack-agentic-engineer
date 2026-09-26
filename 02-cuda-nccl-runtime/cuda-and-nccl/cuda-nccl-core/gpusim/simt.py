"""SIMT: a warp is the unit of work, so performance is decided by what 32 threads do *together*.

Three rules, one function each. The numbers are NVIDIA's documented model (CUDA C++
Programming Guide / Best Practices Guide); Nsight Compute's counters are the ground truth.

* Coalescing: a warp's global-memory request is served in 32-byte *sectors* (four per
  128-byte cache line). On compute capability 6.0+ the number of distinct sectors the active
  lanes touch is the number of transactions. `coalescing()` counts them.
* Bank conflicts: shared memory has 32 banks, each 4 bytes wide, and successive 32-bit words
  map to successive banks. Lanes reading *different* words in one bank serialize; lanes
  reading the *same* word get a broadcast. `bank_conflicts()` counts the passes (wavefronts).
* Divergence: the 32 lanes share one instruction stream, so a warp issues every path any of
  its active lanes takes, one after the other. `divergence()` and `loop_divergence()` cost it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WARP = 32         # threads per warp (every NVIDIA GPU so far)
SECTOR = 32       # bytes: the unit L2 and DRAM traffic is counted in
LINE = 128        # bytes per L1/L2 cache line (= 4 sectors)
BANKS = 32        # shared-memory banks
BANK_BYTES = 4    # each bank serves one 32-bit word per cycle


def warp_addresses(stride: int = 1, offset: int = 0, elem_bytes: int = 4, base: int = 0,
                   n_threads: int = WARP) -> np.ndarray:
    """Byte addresses of `a[offset + stride * lane]` for lanes 0..n_threads-1 (`a` at `base`)."""
    lane = np.arange(n_threads, dtype=np.int64)
    return base + (offset + stride * lane) * elem_bytes


def _touched_bytes(addresses, elem_bytes: int, active) -> np.ndarray:
    a = np.asarray(addresses, dtype=np.int64).ravel()
    if active is not None:
        a = a[np.asarray(active, dtype=bool).ravel()]
    return np.unique((a[:, None] + np.arange(elem_bytes)).ravel())


@dataclass
class Coalescing:
    sectors: int        # distinct 32-byte sectors = memory transactions for this warp request
    lines: int          # distinct 128-byte lines
    useful_bytes: int   # distinct bytes the active lanes asked for

    @property
    def moved_bytes(self) -> int:
        return self.sectors * SECTOR

    @property
    def efficiency(self) -> float:
        """Useful bytes / moved bytes: 1.0 means every byte fetched was used by some lane."""
        return self.useful_bytes / self.moved_bytes if self.sectors else 1.0


def coalescing(addresses, elem_bytes: int = 4, active=None) -> Coalescing:
    """Sectors and lines one warp-wide global load/store touches. `active` masks lanes off
    (a bounds check or a branch): inactive lanes generate no memory traffic."""
    b = _touched_bytes(addresses, elem_bytes, active)
    sectors = np.unique(b // SECTOR)
    lines = np.unique(sectors // (LINE // SECTOR))
    return Coalescing(int(sectors.size), int(lines.size), int(b.size))


@dataclass
class BankAccess:
    wavefronts: int   # shared-memory passes this warp request takes
    ideal: int        # passes with no conflicts (one per phase)
    degree: int       # worst n-way conflict in any phase (1 = conflict-free)


def bank_conflicts(addresses, elem_bytes: int = 4, active=None) -> BankAccess:
    """One warp-wide shared-memory access. A 32-bit word lives in bank (addr // 4) % 32. In each
    phase the cost is the largest number of *distinct* words that land in one bank (the same word
    is a broadcast and costs nothing extra). Accesses of 4 bytes or less are one phase of 32 lanes;
    8-byte accesses are served per half-warp and 16-byte per quarter-warp (the usual model)."""
    a = np.asarray(addresses, dtype=np.int64).ravel()
    mask = np.ones(a.size, bool) if active is None else np.asarray(active, bool).ravel()
    lanes = WARP if elem_bytes <= BANK_BYTES else WARP * BANK_BYTES // elem_bytes
    offsets = np.arange(0, max(elem_bytes, BANK_BYTES), BANK_BYTES)
    wavefronts = ideal = degree = 0
    for p0 in range(0, a.size, lanes):
        sel = a[p0:p0 + lanes][mask[p0:p0 + lanes]]
        if sel.size == 0:
            continue
        words = np.unique(((sel[:, None] + offsets) // BANK_BYTES).ravel())
        worst = int(np.bincount(words % BANKS, minlength=BANKS).max())
        wavefronts, ideal, degree = wavefronts + worst, ideal + 1, max(degree, worst)
    return BankAccess(wavefronts, ideal, degree)


@dataclass
class Divergence:
    issued: int     # warp-instructions issued
    lane_ops: int   # lane-instructions that did useful work

    @property
    def efficiency(self) -> float:
        """Useful lane-instructions / (issued x 32): the SIMT efficiency."""
        return self.lane_ops / (self.issued * WARP) if self.issued else 1.0


def divergence(paths, cost: dict) -> Divergence:
    """`paths[t]` is the branch thread t takes and `cost[path]` its instruction count. Each warp
    issues every distinct path its lanes take, serially (SIMT serialization)."""
    p = np.asarray(paths).ravel().tolist()
    issued = lane_ops = 0
    for w0 in range(0, len(p), WARP):
        warp = p[w0:w0 + WARP]
        issued += sum(cost[x] for x in set(warp))
        lane_ops += sum(cost[x] for x in warp)
    return Divergence(issued, lane_ops)


def loop_divergence(trip_counts, body_cost: int = 1) -> Divergence:
    """A loop whose trip count differs per thread: each warp runs until its slowest lane is done."""
    t = np.asarray(trip_counts, dtype=np.int64).ravel()
    w = np.concatenate([t, np.zeros((-t.size) % WARP, np.int64)]).reshape(-1, WARP)
    return Divergence(int(w.max(axis=1).sum()) * body_cost, int(t.sum()) * body_cost)
