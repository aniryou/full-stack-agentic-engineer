"""Watch the memory system from inside the simulator: which bytes does each warp ask for?

The one idea: global memory is served in **32-byte sectors**, and the unit of work is a **warp
request** — the same load or store instruction executed by the 32 lanes of a warp. If the lanes'
addresses fall in 4 consecutive sectors (32 lanes x 4 bytes = 128 bytes), the request is
*coalesced*; if they fall in 32 different sectors, the hardware moves 8x the useful bytes.

Because the simulator runs kernels as ordinary Python, we can hand a kernel a ``TracedArray``
(an object with ``shape``/``size``/``__getitem__``/``__setitem__``) instead of a NumPy array, and it
records *every element access* with the simulated thread that made it. Grouping those records into
warp requests — same block, same warp (``thread_id // 32``), same array, same k-th access by each lane
— gives what Nsight Compute reports on real hardware as sectors per request
(``l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`` / ``l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum``;
metric names (verify) for your ncu version).

Assumptions, stated so the numbers are not over-read:
* arrays are C-contiguous and start 256-byte aligned (``cudaMalloc`` guarantees this);
* a warp's lanes execute each traced access in the same order (true for these kernels on sizes
  that are multiples of the tile; heavy divergence would mis-group requests);
* counts are *requests to the memory system*, before any L1/L2 cache hit — not DRAM traffic.

Shared memory is not traced; :func:`bank_conflict_ways` answers the shared-memory question for a
list of word addresses.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import SIMULATOR


@dataclass(frozen=True)
class ArrayStats:
    array: str
    kind: str  # "R" or "W"
    accesses: int  # thread-level element accesses
    requests: int  # warp-level requests
    sectors: int  # 32-byte sectors touched, summed over requests
    ideal_sectors: int  # sectors needed if each request's distinct elements were contiguous
    itemsize: int

    @property
    def sectors_per_request(self) -> float:
        return self.sectors / self.requests if self.requests else 0.0

    @property
    def efficiency(self) -> float:
        """Useful fraction of the bytes moved (1.0 = perfectly coalesced)."""
        return self.ideal_sectors / self.sectors if self.sectors else 1.0

    @property
    def bytes_moved(self) -> int:
        return self.sectors * 32


class TracedArray:
    """A NumPy array wrapper that logs each element access made by a simulated CUDA thread."""

    def __init__(self, trace: "Trace", name: str, data: np.ndarray):
        if not isinstance(data, np.ndarray) or not data.flags.c_contiguous:
            raise TypeError("TracedArray wraps C-contiguous NumPy arrays")
        self._trace, self.name, self._data = trace, name, data
        self.shape, self.size, self.ndim, self.dtype = data.shape, data.size, data.ndim, data.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def _record(self, idx, kind: str) -> None:
        idx = idx if isinstance(idx, tuple) else (idx,)
        flat = int(np.ravel_multi_index(tuple(int(i) for i in idx), self.shape))  # raises if out of bounds
        self._trace.record(self.name, kind, flat * self.dtype.itemsize, self.dtype.itemsize)

    def __getitem__(self, idx):
        self._record(idx, "R")
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._record(idx, "W")
        self._data[idx] = value


class Trace:
    """Collects accesses and turns them into per-array coalescing statistics."""

    def __init__(self, warp_size: int = 32, sector_bytes: int = 32):
        self.warp_size, self.sector_bytes = warp_size, sector_bytes
        self.records: list[tuple] = []  # (launch, block, thread_id, array, kind, seq, byte_addr)
        self.launches = 0
        self._seq: dict[tuple, int] = {}
        self._itemsize: dict[str, int] = {}

    def wrap(self, name: str, data: np.ndarray) -> TracedArray:
        return TracedArray(self, name, data)

    def record(self, array: str, kind: str, byte_addr: int, itemsize: int) -> None:
        t = threading.current_thread()
        if not hasattr(t, "thread_id"):
            raise RuntimeError("TracedArray accessed outside a simulated CUDA thread")
        b = t.blockIdx
        block = (b.x, b.y, b.z)
        key = (self.launches, block, t.thread_id, array, kind)
        seq = self._seq.get(key, 0)  # only this simulated thread touches this key
        self._seq[key] = seq + 1
        self._itemsize[array] = itemsize
        self.records.append((self.launches, block, t.thread_id, array, kind, seq, byte_addr))

    # -- analysis ----------------------------------------------------------------------------
    def _requests(self):
        groups: dict[tuple, list[int]] = defaultdict(list)
        for launch, block, tid, array, kind, seq, addr in self.records:
            groups[(array, kind, launch, block, tid // self.warp_size, seq)].append(addr)
        return groups

    def stats(self) -> list[ArrayStats]:
        agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for (array, kind, *_), addrs in self._requests().items():
            item = self._itemsize[array]
            distinct = set(addrs)
            a = agg[(array, kind)]
            a[0] += len(addrs)
            a[1] += 1
            a[2] += len({x // self.sector_bytes for x in distinct})
            a[3] += math.ceil(len(distinct) * item / self.sector_bytes)
        return [ArrayStats(array, kind, acc, req, sec, ideal, self._itemsize[array])
                for (array, kind), (acc, req, sec, ideal) in sorted(agg.items())]

    def get(self, array: str, kind: str) -> ArrayStats:
        for s in self.stats():
            if s.array == array and s.kind == kind:
                return s
        raise KeyError(f"no {kind!r} accesses to {array!r} recorded")

    def accesses(self) -> dict[tuple[str, str], int]:
        return {(s.array, s.kind): s.accesses for s in self.stats()}

    def warp_request(self, array: str, kind: str, block=(0, 0, 0), warp: int = 0, seq: int = 0,
                     launch: int = 1) -> list[int]:
        """Element indices one warp request touched, in lane order — to *look at* a pattern."""
        rows = sorted((tid, addr) for n, b, tid, a, k, s, addr in self.records
                      if n == launch and a == array and k == kind and b == tuple(block)
                      and tid // self.warp_size == warp and s == seq)
        item = self._itemsize.get(array, 4)
        return [addr // item for _, addr in rows]

    def table(self) -> str:
        head = f"{'array':>8} {'op':>2} {'accesses':>9} {'requests':>9} {'sectors':>8} {'sect/req':>8} {'efficiency':>10}"
        lines = [head, "-" * len(head)]
        for s in self.stats():
            lines.append(f"{s.array:>8} {s.kind:>2} {s.accesses:>9} {s.requests:>9} {s.sectors:>8} "
                         f"{s.sectors_per_request:>8.1f} {s.efficiency:>10.1%}")
        return "\n".join(lines)


def trace_launch(kernel, grid, block, *args, names: list[str] | None = None, warp_size: int = 32,
                 trace: Trace | None = None) -> Trace:
    """Launch ``kernel[grid, block](*args)`` in the simulator with every NumPy array argument traced.

    Arrays are updated in place (outputs appear in the arrays you passed). Scalars pass through.
    Pass ``trace=`` to accumulate several launches (e.g. the four kernels of an unfused softmax)
    into one set of statistics; give the same array the same name in each launch.
    """
    if not SIMULATOR:
        raise RuntimeError("tracing needs the CUDA simulator (NUMBA_ENABLE_CUDASIM=1 before importing "
                           "numba). On a real GPU, use Nsight Compute's sectors/requests metrics instead.")
    trace = trace if trace is not None else Trace(warp_size=warp_size)
    trace.launches += 1
    arrays = [i for i, a in enumerate(args) if isinstance(a, np.ndarray)]
    names = list(names) if names is not None else [f"arg{i}" for i in arrays]
    if len(names) != len(arrays):
        raise ValueError(f"{len(arrays)} array arguments but {len(names)} names")
    wrapped = list(args)
    for i, name in zip(arrays, names):
        wrapped[i] = trace.wrap(name, args[i])  # C-contiguous required, so in-place writes are visible
    kernel[grid, block](*wrapped)
    return trace


def bank_conflict_ways(word_indices, banks: int = 32) -> int:
    """Degree of the shared-memory bank conflict for one warp access.

    Shared memory has ``banks`` banks of 4-byte words; word ``w`` lives in bank ``w % banks``. Lanes
    reading the *same* word are served by one broadcast; *distinct* words in the same bank are
    served one after another. Returns the worst bank's count of distinct words (1 = conflict-free).
    """
    by_bank: dict[int, set] = defaultdict(set)
    for w in word_indices:
        by_bank[int(w) % banks].add(int(w))
    return max((len(s) for s in by_bank.values()), default=0)
