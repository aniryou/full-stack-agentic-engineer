"""Coalescing and reuse, measured from the kernels themselves in the simulator."""
import numpy as np

from gpurt.kernels import matmul as mm
from gpurt.kernels import softmax as sm
from gpurt.kernels import traffic
from gpurt.kernels import transpose as tr
from gpurt.kernels.trace import Trace, bank_conflict_ways, trace_launch


def _transpose_trace(kernel, n=64):
    A = np.arange(n * n, dtype=np.float32).reshape(n, n)
    out = np.zeros_like(A)
    grid, block = tr.launch_config(n, n)
    t = trace_launch(kernel, grid, block, A, out, names=["in", "out"])
    assert np.array_equal(out, A.T)
    return t


def test_naive_transpose_writes_are_uncoalesced():
    t = _transpose_trace(tr.transpose_naive)
    assert t.get("in", "R").sectors_per_request == 4.0  # 32 lanes x 4 B = 4 sectors
    w = t.get("out", "W")
    assert w.sectors_per_request == 32.0 and w.efficiency == 0.125  # one sector per lane
    assert t.warp_request("out", "W")[:3] == [0, 64, 128]  # lanes stride by a row


def test_tiled_transpose_coalesces_both_sides():
    t = _transpose_trace(tr.transpose_tiled)
    assert t.get("in", "R").sectors_per_request == 4.0
    assert t.get("out", "W").sectors_per_request == 4.0 and t.get("out", "W").efficiency == 1.0


def test_tiling_divides_matmul_global_loads_by_the_tile():
    M = N = K = 32
    A, B = np.ones((M, K), np.float32), np.ones((K, N), np.float32)
    for kernel, tile in ((mm.matmul_naive, 1), (mm.make_matmul_tiled(8), 8)):
        C = np.zeros((M, N), np.float32)
        grid, block = mm.launch_config(M, N, 16 if tile == 1 else tile)
        acc = trace_launch(kernel, grid, block, A, B, C, names=["A", "B", "C"]).accesses()
        assert acc[("A", "R")] + acc[("B", "R")] == traffic.matmul_global_loads(M, N, K, tile)
        assert acc[("C", "W")] == M * N and np.all(C == K)


def test_fused_softmax_moves_half_the_bytes_of_unfused():
    rows, cols, threads = 4, 64, 32
    x = np.random.default_rng(1).standard_normal((rows, cols)).astype(np.float32)
    fused_out = np.zeros_like(x)
    fused = trace_launch(sm.make_softmax_fused(threads), rows, threads, x, fused_out, names=["x", "out"]).accesses()

    t = Trace()  # four launches, one set of statistics
    m, e = np.zeros(rows, np.float32), np.zeros_like(x)
    s, out = np.zeros(rows, np.float32), np.zeros_like(x)
    row_max, row_sum = sm.make_row_reductions(threads)
    grid, block = sm.elementwise_grid(rows, cols)
    trace_launch(row_max, rows, threads, x, m, names=["x", "m"], trace=t)
    trace_launch(sm.sub_exp, grid, block, x, m, e, names=["x", "m", "e"], trace=t)
    trace_launch(row_sum, rows, threads, e, s, names=["e", "s"], trace=t)
    trace_launch(sm.divide_rows, grid, block, e, s, out, names=["e", "s", "out"], trace=t)
    unfused = t.accesses()

    ref = sm.softmax_reference(x)
    np.testing.assert_allclose(fused_out, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    per_elem = lambda acc, keys: sum(acc.get(k, 0) for k in keys) / (rows * cols)  # noqa: E731
    big = [("x", "R"), ("e", "R"), ("e", "W"), ("out", "W")]
    assert per_elem(fused, big) == 3 and per_elem(unfused, big) == 6  # matches traffic.SOFTMAX_ACCESSES
    assert traffic.softmax_bytes(rows, cols, "unfused") == 2 * traffic.softmax_bytes(rows, cols, "fused")


def test_bank_conflicts_of_a_tile_column_read():
    assert bank_conflict_ways([lane * 32 for lane in range(32)]) == 32  # 32-wide float tile: same bank
    assert bank_conflict_ways([lane * 33 for lane in range(32)]) == 1  # padded to 33: all banks differ
    assert bank_conflict_ways([7] * 32) == 1  # same word: a broadcast, no conflict
