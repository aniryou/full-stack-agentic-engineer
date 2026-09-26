"""gpusim: the GPU software substrate, simulated on a CPU.

    from gpusim import simt, occupancy, tiling, collectives, compat, sharing, health

Seven small modules, numpy only, one idea each. Read them in notebook order:

    simt         a warp is the unit: coalescing into 32-byte sectors, bank conflicts, divergence
    occupancy    resident warps per SM, what limits them, Little's law
    tiling       bytes and launches: tiled GEMM traffic, fused softmax, CUDA Graphs
    collectives  ring/tree/direct/in-switch collectives on simulated ranks; alpha-beta; busbw
    compat       driver <-> CUDA runtime <-> compute capability, and the error you would see
    sharing      MIG placement, time-slicing and MPS latency
    health       GPU util vs SM active, throttle reasons, XID triage

Everything here is a *model* of documented NVIDIA behaviour, and its output is simulated. The
lab next door (`cuda-nccl-lab`, package `gpurt`) measures the real thing on a GPU.
"""
from . import collectives, compat, health, occupancy, sharing, simt, tiling

__all__ = ["collectives", "compat", "health", "occupancy", "sharing", "simt", "tiling"]
__version__ = "0.1.0"
