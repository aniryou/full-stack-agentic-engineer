"""Bytes and launches: tiling, fusion and CUDA Graphs, pinned to hand-computed values."""
import numpy as np
import pytest

from gpusim import tiling


def test_gemm_traffic_formulas_4096_bf16():
    n = 4096
    assert tiling.gemm_traffic(n, n, n)["bytes"] == 3 * n * n * 2                        # compulsory
    assert tiling.gemm_traffic(n, n, n, 1, 1)["bytes"] == (2 * n ** 3 + n * n) * 2       # naive
    t = tiling.gemm_traffic(n, n, n, 128, 128)
    assert t["bytes"] == (2 * n * n * (n // 128) + n * n) * 2                            # 2.18 GB
    assert t["intensity"] == pytest.approx(2 * n ** 3 / t["bytes"])


def test_tiled_matmul_matches_numpy_and_the_traffic_formula():
    rng = np.random.default_rng(0)
    A = rng.integers(-3, 4, (70, 50)).astype(np.float64)       # uneven shapes: partial edge tiles
    B = rng.integers(-3, 4, (50, 90)).astype(np.float64)
    C, counted = tiling.tiled_matmul(A, B, BM=32, BN=32, BK=16)
    assert np.array_equal(C, A @ B)
    f = tiling.gemm_traffic(70, 90, 50, 32, 32)
    assert counted == {"loads": f["a_elems"] + f["b_elems"], "stores": f["c_elems"]}


def test_one_tile_covering_the_output_is_the_compulsory_traffic():
    assert tiling.gemm_traffic(256, 512, 64, 256, 512) == tiling.gemm_traffic(256, 512, 64)


def test_tile_shared_memory():
    assert tiling.tile_smem_bytes(128, 128, 32, dtype_bytes=2, stages=3) == 48 * 1024


def test_softmax_traffic_unfused_vs_fused_vs_online():
    R = C = 4096
    assert tiling.softmax_traffic(R, C, 2, "unfused") == {"bytes": (8 * R * C + 4 * R) * 2, "kernels": 5}
    assert tiling.softmax_traffic(R, C, 2, "fused") == {"bytes": 2 * R * C * 2, "kernels": 1}
    assert tiling.softmax_traffic(R, C, 2, "online")["bytes"] == 3 * R * C * 2


def test_online_softmax_is_exact_and_overflow_safe():
    x = np.random.default_rng(1).normal(size=(4, 1000)) * 300      # exp(300) would overflow float32
    y = tiling.softmax_online(x, block=64)
    assert np.allclose(y, tiling.softmax_ref(x), rtol=1e-12, atol=0)
    assert np.allclose(y.sum(axis=-1), 1.0)


def test_elementwise_fusion_saves_bytes_and_launches():
    un, fu = tiling.elementwise_traffic(10 ** 6, 4), tiling.elementwise_traffic(10 ** 6, 4, fused=True)
    assert un["bytes"] == 4 * fu["bytes"] and (un["kernels"], fu["kernels"]) == (4, 1)


def test_short_kernels_are_launch_bound_and_graphs_fix_it():
    eager = tiling.step_time([3.0] * 384, launch_us=10.0)
    graph = tiling.step_time([3.0] * 384, launch_us=10.0, graph=True, graph_launch_us=10.0)
    assert eager["time_us"] == 384 * 10 + 3 and graph["time_us"] == 10 + 384 * 3
    long_kernels = tiling.step_time([8.0] * 384, launch_us=5.0)          # GPU-bound: graphs barely help
    assert long_kernels["time_us"] == 5 + 384 * 8
