"""Collectives you can measure: nccl-tests' bandwidth definitions, the α-β fit, and three transports.

The one idea: a collective's cost is ``α + factor(op, n) · S / B`` — a latency term and a bandwidth
term — and busbw is the number that lets you compare a measurement with the link's peak whatever the
op and the number of ranks (primer §5). Modules:

    busbw      algbw/busbw factors and buffer sizing, exactly as nccl-tests defines them
    alphabeta  fit t(S) = α + S/B; half-performance size; ring cost formulas
    semantics  what each collective computes (reference results for the #wrong check)
    sweep      the benchmark loop shared by every backend, and a process launcher
    pipes      a real ring all-reduce over OS pipes (T0, any machine, no torch)
    bench      torch.distributed: gloo on CPU (T0) or NCCL on GPUs (T1/T2); torch imported lazily

Nothing here imports torch at import time.
"""

from .alphabeta import AlphaBeta, fit, fit_rows, ring_allreduce_time
from .busbw import OPS, algbw, bus_factor, plan
from .sweep import Row, default_sizes, format_table

# Note: the *function* busbw() lives in gpurt.dist.busbw; it is not re-exported here because the
# name would shadow the submodule itself.
__all__ = ["OPS", "AlphaBeta", "Row", "algbw", "bus_factor", "default_sizes", "fit", "fit_rows",
           "format_table", "plan", "ring_allreduce_time"]
