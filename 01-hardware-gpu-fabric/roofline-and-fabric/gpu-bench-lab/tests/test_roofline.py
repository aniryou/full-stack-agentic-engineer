"""The roofline model, the spec table, and the numbers the notebooks ask you to predict."""
import pytest

from gpubench import roofline, specs
from gpubench.accounting import gemm_cost, gemm_intensity, stream_cost
from gpubench.measure import Measurement
from gpubench.timing import Timing


def test_attainable_ridge_and_bound():
    assert roofline.attainable(1, 100, 10) == 10       # memory-bound: intensity × bandwidth
    assert roofline.attainable(100, 100, 10) == 100    # compute-bound: the flat roof
    assert roofline.ridge(989e12, 3.35e12) == pytest.approx(295.22, abs=0.01)
    assert roofline.bound(10, 100, 10) == "compute" and roofline.bound(9.9, 100, 10) == "memory"


def test_h100_bf16_decode_is_memory_bound_and_big_gemm_is_not():
    r = roofline.spec_roofline(specs.get("h100-sxm"), "bf16")
    assert r.bound(gemm_intensity(1, 8192, 8192, 2)) == "memory"
    assert r.bound(gemm_intensity(4096, 4096, 4096, 2)) == "compute"


def test_decode_batch_crossover_on_h100_is_319():
    ridge = specs.get("h100-sxm").ridge("bf16")
    b = next(m for m in range(1, 5000) if gemm_intensity(m, 8192, 8192, 2) >= ridge)
    assert b == 319          # the batch at which a d=8192 projection becomes compute-bound


def _m(op, cost, seconds, **params):
    return Measurement(op, params, cost, Timing((seconds,)), "numpy", "cpu")


def test_measured_roofline_takes_the_best_of_each_roof():
    g = [_m("gemm", gemm_cost(64, 64, 64, 8), 1e-5, dtype="float64"),
         _m("gemm", gemm_cost(512, 512, 512, 8), 1e-3, dtype="float64")]
    s = [_m("stream.add", stream_cost("add", 10**6, 8), 2.4e-3), _m("stream.copy", stream_cost("copy", 10**6, 8), 1.6e-3)]
    r = roofline.measured_roofline(g, s, "float64")
    assert r.peak_flops == pytest.approx(2 * 512 ** 3 / 1e-3)
    assert r.peak_bw == pytest.approx(10e9)          # both kernels moved 10 GB/s
    assert r.efficiency(r.peak_flops, 1e6) == pytest.approx(1.0)


def test_ascii_plot_draws_roofs_points_and_legend():
    r = roofline.Roofline("toy", 100e9, 10e9)
    art = roofline.ascii_plot([r], [("o", 2.0, 15e9), ("x", 50.0, 90e9)])
    assert "o" in art and "x" in art and "#" in art and "ridge 10.0 FLOP/B" in art


def test_spec_lookup_and_links():
    assert specs.lookup("NVIDIA H100 80GB HBM3").key == "h100-sxm"
    assert specs.lookup("NVIDIA H100 PCIe").key == "h100-pcie"
    assert specs.lookup("Tesla T4").key == "t4" and specs.lookup("NVIDIA L4").key == "l4"
    assert specs.lookup("NVIDIA L40S") is None and specs.lookup("NVIDIA GH200 480GB") is None
    assert specs.lookup("NVIDIA A100-SXM4-80GB").key == "a100-80gb"
    assert specs.pcie_gbs(3) == pytest.approx(15.754, abs=1e-3)
    assert specs.pcie_gbs(4) == pytest.approx(31.508, abs=1e-3)
    assert specs.pcie_gbs(5, 8) == pytest.approx(31.508, abs=1e-3)
    assert specs.nvlink_gbs(18, 4) == 450 and specs.get("h100-sxm").nvlink_gbs_per_direction == 450


def test_cpu_peak_estimate():
    assert specs.cpu_peak_flops(4, 2.8, 512, 2, 8) == pytest.approx(358.4e9)   # 4 × 2.8G × 8 lanes × 2 × 2
    assert specs.cpu_peak_flops(4, 2.8, 512, 2, 4) == pytest.approx(716.8e9)
    assert specs.simd_bits_from_flags({"avx2", "fma"}) == 256 and specs.simd_bits_from_flags({"asimd"}) == 128


def test_spec_table_is_internally_consistent():
    for s in specs.GPUS:
        assert s.mem_bw_gbs > 0 and s.memory_gb > 0 and s.match
        for dtype, tf in s.dense_tflops.items():
            assert tf > 0, (s.key, dtype)
        if "float16" in s.dense_tflops and "float32" in s.dense_tflops:
            assert s.dense_tflops["float16"] >= s.dense_tflops["float32"]
