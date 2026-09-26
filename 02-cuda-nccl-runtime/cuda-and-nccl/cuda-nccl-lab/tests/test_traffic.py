"""The byte/FLOP model, pinned to hand-computed values."""
import pytest

from gpurt.kernels import traffic as t


def test_elementwise_and_transpose_bytes():
    assert t.elementwise_bytes("vec_add", 1 << 20) == 12 * (1 << 20)  # 2 reads + 1 write of 4 B
    assert t.elementwise_bytes("copy", 1000, itemsize=2) == 4000
    assert t.transpose_bytes(4096, 4096) == 134_217_728  # 2 x 16 Mi elements x 4 B


def test_matmul_counts_and_intensity():
    assert t.matmul_flops(1024, 1024, 1024) == 2 ** 31
    assert t.matmul_min_bytes(1024, 1024, 1024) == 12_582_912
    assert t.arithmetic_intensity(2 ** 31, 12_582_912) == pytest.approx(170.67, abs=0.01)
    assert t.matmul_global_loads(64, 64, 64, tile=16) == 2 * 64 * 64 * 4


def test_machine_balance_is_the_ridge_point():
    assert t.machine_balance(t.GPUS["T4"]) == pytest.approx(25.3125)  # 8.1 TFLOP/s / 320 GB/s
    assert t.machine_balance(t.GPUS["L4"]) == pytest.approx(101.0)  # 30.3 TFLOP/s / 300 GB/s


def test_softmax_fusion_halves_traffic_and_bandwidth_math():
    assert t.softmax_bytes(4096, 4096, "unfused") == 402_653_184  # 24 B per element
    assert t.softmax_bytes(4096, 4096, "fused") == 201_326_592  # 12 B per element
    assert t.effective_gbps(12e6, 1e-3) == pytest.approx(12.0)
    assert t.predicted_seconds(320e9 * 0.8, t.GPUS["T4"], efficiency=0.8) == pytest.approx(1.0)
