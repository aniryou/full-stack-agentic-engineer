"""nccl-tests' bandwidth definitions and the α-β model, pinned to hand-computed numbers."""
import numpy as np
import pytest

from gpurt.dist import alphabeta as ab
from gpurt.dist import busbw as bw


@pytest.mark.parametrize("op,n,factor", [
    ("all_reduce", 8, 1.75), ("all_reduce", 2, 1.0), ("all_gather", 4, 0.75),
    ("reduce_scatter", 8, 0.875), ("alltoall", 2, 0.5), ("broadcast", 8, 1.0), ("sendrecv", 2, 1.0),
])
def test_bus_factors(op, n, factor):
    assert bw.bus_factor(op, n) == factor


def test_algbw_and_busbw_hand_computed():
    assert bw.algbw(1e9, 0.01) == pytest.approx(100.0)  # 1 GB in 10 ms
    assert bw.busbw(1e9, 0.01, "all_reduce", 8) == pytest.approx(175.0)
    assert bw.bytes_sent_per_rank("all_reduce", 1 << 30, 4) == 1.5 * (1 << 30)  # 2(n-1)/n = 1.5


def test_plan_sizes_buffers_like_nccl_tests():
    # nccl-tests v2.20.0 all_gather.cu: base = (count/nranks) & ~(16/eltSize - 1)
    p = bw.plan("all_gather", 4000, 3)  # 1000 floats -> 333 per rank -> 332 (a 16-byte multiple)
    assert (p.send_count, p.recv_count, p.param_count, p.size) == (332, 996, 332, 3984)
    p = bw.plan("reduce_scatter", 4096, 4)
    assert (p.send_count, p.recv_count, p.param_count, p.size) == (1024, 256, 256, 4096)
    p = bw.plan("alltoall", 4000, 3)
    assert (p.send_count, p.recv_count, p.param_count, p.size) == (996, 996, 332, 3984)
    assert bw.plan("all_reduce", 4096, 8).param_count == 1024
    assert bw.plan("all_reduce", 4000, 3).param_count == 1000  # no split, no rounding
    assert bw.plan("all_gather", 8, 2).param_count == 0  # 1 float per rank < 16 B: nccl-tests' size 0
    assert bw.per_rank_count(1000, 3, itemsize=8) == 332 and bw.per_rank_count(1001, 3, itemsize=8) == 332
    assert bw.per_rank_count(999, 3, itemsize=2) == 328  # 333 halves -> multiple of 8


def test_op_names_are_canonicalised():
    assert bw.canonical("allreduce") == bw.canonical("all_reduce_perf") == "all_reduce"
    with pytest.raises(ValueError):
        bw.canonical("gossip")


def test_fit_recovers_the_generating_parameters():
    sizes = [2 ** k for k in range(3, 31)]
    exact = ab.fit(sizes, ab.synthetic_times(sizes, 20e-6, 10e9))
    assert exact.alpha_s == pytest.approx(20e-6, rel=1e-6) and exact.bw_Bps == pytest.approx(10e9, rel=1e-6)
    noisy = ab.fit(sizes, ab.synthetic_times(sizes, 20e-6, 10e9, noise=0.05, seed=3))
    assert noisy.alpha_s == pytest.approx(20e-6, rel=0.15) and noisy.bw_Bps == pytest.approx(10e9, rel=0.1)


def test_half_bandwidth_size_and_regimes():
    m = ab.AlphaBeta(20e-6, 10e9)
    assert m.n_half == pytest.approx(200_000)  # α·B
    assert m.algbw_gbps(m.n_half) == pytest.approx(5.0)  # half of 10 GB/s
    assert m.regime(1_000) == "latency-bound" and m.regime(10_000_000) == "bandwidth-bound"


def test_ring_allreduce_cost_hand_computed():
    # 2(n-1)α + 2(n-1)/n · S/B = 14 x 5 µs + 1.75 x 1e9 / 100e9 s
    assert ab.ring_allreduce_time(1e9, 8, 5e-6, 100e9) == pytest.approx(0.01757)


def test_fit_clamps_negative_latency_to_zero():
    sizes = np.array([1e6, 2e6, 4e6])
    m = ab.fit(sizes, sizes / 5e9 * np.array([0.98, 1.0, 1.01]))  # super-linear: intercept < 0
    assert m.alpha_s == 0.0 and m.bw_Bps == pytest.approx(5e9, rel=0.03)
