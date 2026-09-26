"""Spec-sheet literacy and the roofline: formulas pinned to hand-computed values."""
import pytest

from roofline import specs
from roofline import roofline as rl

H100 = specs.get("h100-sxm")


# -- specs -------------------------------------------------------------------------
def test_catalogue_is_dated_and_complete():
    assert specs.AS_OF == "2026-09"
    for key in ["t4", "l4", "a100-80gb", "h100-sxm", "h200", "b200", "gb200", "gb300", "rtx-pro-6000",
                "mi300x", "mi325x", "mi355x", "tpu-v5e", "tpu-v6e", "tpu-v7"]:
        d = specs.DEVICES[key]
        assert d.memory_gb > 0 and d.memory_tbs > 0
        assert d.supports("bf16") or d.supports("fp16")


def test_dense_is_half_the_sparse_headline():
    # H100 datasheet: "1,979 TFLOPS*" bf16 and "3,958 TFLOPS*" fp8, * = with sparsity
    assert specs.from_sparse(1979) == pytest.approx(H100.tflops["bf16"], abs=0.2)
    assert specs.from_sparse(3958) == pytest.approx(H100.tflops["fp8"], abs=0.2)
    assert specs.from_sparse(485) == pytest.approx(specs.get("l4").tflops["fp8"])


def test_peak_is_units_times_work_per_clock_times_clock():
    assert specs.peak_from_clock(108, 2048, 1.41) == pytest.approx(311.9, abs=0.05)   # A100
    assert specs.peak_from_clock(132, 4096, 1.83) == pytest.approx(989.4, abs=0.05)   # H100
    for key, (sms, per_clk, ghz) in specs.CLOCKS.items():
        assert specs.peak_from_clock(sms, per_clk, ghz) == pytest.approx(specs.DEVICES[key].tflops["fp16"], rel=0.005)


def test_missing_precision_is_an_error_not_a_zero():
    with pytest.raises(KeyError, match="no bf16"):
        specs.get("t4").peak("bf16")          # Turing has fp16 tensor cores, not bf16


def test_link_marketing_is_bidirectional():
    assert H100.scaleup_gbs == 900 and H100.scaleup_gbs_per_dir == 450
    assert specs.get("l4").scaleup_gbs_per_dir is None          # PCIe only


def test_get_by_fragment_and_table():
    assert specs.get("MI300X").vendor == "AMD"
    with pytest.raises(KeyError):
        specs.get("a100")                                        # ambiguous: 40GB and 80GB
    assert "| NVIDIA H100 SXM | 80 | 3.35 | 989 | 1,979 | – | 295 |" in specs.table()
    fp8 = specs.table(precision="fp8")                                     # falls back where there is no fp8
    assert "| NVIDIA H100 SXM | 80 | 3.35 | 1,979 | 1,979 | – | 591 |" in fp8
    assert "| Google TPU v5e | 16 | 0.819 | 197 | – | – | 241 |" in fp8   # bf16: no fp8, no fp16
    assert "| NVIDIA T4 | 16 | 0.32 | 65 | – | – | 203 |" in fp8           # fp16: no fp8, no bf16
    assert "| Google TPU v6e (Trillium) | 32 | 1.64 | 918 |" in specs.table(precision="fp16")


# -- roofline ------------------------------------------------------------------------
def test_ridge_points():
    assert H100.ridge("bf16") == pytest.approx(989.4 / 3.35)                 # 295.3
    assert H100.ridge("fp8") == pytest.approx(1978.9 / 3.35)                 # 590.7
    assert specs.get("l4").ridge("bf16") == pytest.approx(121 / 0.30)        # 403.3
    assert specs.get("t4").ridge("fp16") == pytest.approx(65 / 0.32)         # 203.1


def test_attainable_is_the_min_of_two_roofs():
    assert rl.attainable(1.0, H100) == pytest.approx(3.35e12)                # on the slope
    assert rl.attainable(10_000, H100) == pytest.approx(989.4e12)            # on the flat
    assert rl.attainable(H100.ridge(), H100) == pytest.approx(989.4e12)      # the corner


def test_kernel_intensities():
    assert rl.elementwise(1000, 2).intensity == pytest.approx(1 / 6)       # z = x + y, bf16
    assert rl.reduction(1000, 4).intensity == pytest.approx(1 / 4)         # sum, fp32
    g = rl.gemm(4096, 4096, 4096, 2)
    assert g.flops == 2 * 4096 ** 3
    assert g.intensity == pytest.approx(2 * 4096 / (3 * 2))                # 1365.3
    assert rl.gemm(1, 4096, 4096, 2).intensity == pytest.approx(1.0, abs=0.001)   # GEMV ~ 2/b


def test_time_kernel_classifies_the_bound():
    t = rl.time_kernel(rl.gemm(8192, 8192, 8192), H100)
    assert t.bound == "compute" and t.time == pytest.approx(2 * 8192 ** 3 / 989.4e12)   # 1.11 ms
    v = rl.time_kernel(rl.elementwise(1 << 26, 2), H100)
    assert v.bound == "memory" and v.time == pytest.approx(6 * (1 << 26) / 3.35e12)
    assert v.achieved == pytest.approx(3.35e12 / 6)


def test_tiling_raises_intensity_with_tile_size():
    assert rl.tile_intensity(128, 128) == pytest.approx(64.0)               # 2*128*128/(256*2)
    assert rl.tile_intensity(128, 256) == pytest.approx(85.33, abs=0.01)
    assert rl.tile_intensity(256, 256) == pytest.approx(128.0)
    n = 4096
    tiled = rl.tiled_gemm_bytes(n, n, n, 128, 128)
    assert tiled == pytest.approx((n ** 3 * 2 / 128 + n * n) * 2)
    assert tiled / rl.gemm(n, n, n).bytes == pytest.approx(21.67, abs=0.01)
    assert rl.gemm(n, n, n).flops / rl.tiled_gemm_bytes(n, n, n, 1, 1) == pytest.approx(0.5, abs=0.001)
    assert rl.tile_smem_bytes(128, 256, 64, 2, stages=4) == 196_608       # 192 KiB < 228 KB SMEM


def test_fusion_divides_traffic_by_the_number_of_ops():
    n = 4096 * 8192
    assert rl.fusion_bytes(n, 4, 2, fused=False) == 4 * rl.fusion_bytes(n, 4, 2, fused=True)
    assert rl.fusion_bytes(n, 4, 2, fused=True) == 2 * n * 2               # 134 MB
    # a residual add reads a second full tensor, fused or not: 9 passes vs 3, fusion saves 2/3
    assert rl.fusion_bytes(n, 4, 2, fused=False, extra_inputs=1) == 9 * n * 2
    assert rl.fusion_bytes(n, 4, 2, fused=True, extra_inputs=1) == 3 * n * 2


def test_ascii_roofline_renders_every_kernel():
    art = rl.ascii_roofline(H100, [rl.gemm(1, 4096, 4096), rl.gemm(4096, 4096, 4096)])
    assert "ridge 295" in art and "A:" in art and "B:" in art and "compute-bound" in art
