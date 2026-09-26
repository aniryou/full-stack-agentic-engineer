"""Storage and cold start: how long until a new replica serves its first token.

The one idea: cold start is a pipeline -- provision the machine, pull the image,
fetch the weights, copy them into GPU memory, initialise the engine -- and the
weights term is bytes / (bandwidth of the slowest tier on the path). Parallel
streams raise fetch bandwidth up to the NIC or disk cap; streaming weights
straight into GPU memory overlaps fetch with the host-to-device copy, so the
time approaches max(fetch, copy) instead of their sum; a cache moves the fetch
to a faster tier.

The tier bandwidths below are planning ASSUMPTIONS, not measurements. Replace
them with yours: gpu-bench-lab notebook 04 measures disk and host-to-device.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fabric import staged_transfer_time

GIB = 1024 ** 3

ASSUMED_GBS = {                      # GB/s, sustained, one reader -- illustrative (verify)
    "internet-download": 0.1,        # model hub over the public internet; varies widely
    "object-store-1-stream": 0.1,    # one HTTP range stream from an object store
    "network-disk": 1.2,             # network block device on a mid-size VM
    "local-nvme": 7.0,               # one PCIe Gen4 NVMe drive, sequential read
    "page-cache": 20.0,              # a warm file in host DRAM
    "h2d-pageable": 12.0,            # host -> GPU from pageable memory (bounce-buffered)
    "h2d-pinned-pcie4": 25.0,        # host -> GPU, pinned, ~80% of PCIe Gen4 x16
    "h2d-pinned-pcie5": 50.0,        # host -> GPU, pinned, ~80% of PCIe Gen5 x16
    "nic-100g": 12.5,                # the VM's network cap bounds any parallel fetch
}


def checkpoint_bytes(params: float, weight_bytes: float = 2) -> float:
    """Bytes on disk: params x bytes per param. Llama-3.1-70B in bf16: 141 GB."""
    return params * weight_bytes


def read_time(n_bytes: float, gbs: float) -> float:
    return n_bytes / (gbs * 1e9)


def parallel_gbs(streams: int, per_stream_gbs: float, cap_gbs: float) -> float:
    """Parallel range reads add up -- until the NIC, disk or service cap."""
    return min(streams * per_stream_gbs, cap_gbs)


def load_time(n_bytes: float, fetch_gbs: float, h2d_gbs: float, gpus: int = 1,
              streamed: bool = False, chunk_bytes: float = 0.25 * GIB) -> float:
    """Fetch the checkpoint once per node, then copy each GPU's shard over its own link.

    Sequential: n / fetch + n / (gpus x h2d). Streamed: chunks flow fetch -> GPU, so the
    slower hop sets the rate (fabric.staged_transfer_time) -- the idea behind model
    streamers and safetensors loaders that read straight into device memory.
    """
    return staged_transfer_time(n_bytes, [fetch_gbs, h2d_gbs * gpus],
                                chunk_bytes if streamed else None)


@dataclass(frozen=True)
class ColdStart:
    stages: tuple        # ((name, seconds), ...) in order

    @property
    def total(self) -> float:
        return sum(s for _, s in self.stages)

    def table(self) -> str:
        w = max(len(n) for n, _ in self.stages)
        rows = [f"{n:<{w}}  {s:8.1f} s  {s / self.total:6.1%}" for n, s in self.stages]
        return "\n".join(rows + [f"{'total':<{w}}  {self.total:8.1f} s"])


def cold_start(n_bytes: float, *, fetch_gbs: float, h2d_gbs: float, gpus: int = 1,
               provision_s: float = 0.0, image_s: float = 0.0, init_s: float = 0.0,
               streamed: bool = False) -> ColdStart:
    """Time from 'scale up' to 'ready', stage by stage (inputs are yours, not measured)."""
    if streamed:
        weights = (("weights, streamed to GPU", load_time(n_bytes, fetch_gbs, h2d_gbs, gpus, True)),)
    else:
        weights = (("fetch weights", read_time(n_bytes, fetch_gbs)),
                   ("copy to GPU", read_time(n_bytes, h2d_gbs * gpus)))
    return ColdStart((("provision node", provision_s), ("pull image", image_s))
                     + weights + (("engine init + warm-up", init_s),))
