"""The byte and FLOP accounting, pinned to hand-computed values. If one of these changes,
every number the lab reports changes with it — so each formula has its own test."""
import pytest

from gpubench.accounting import (NUMPY_TRIAD_PASSES, STREAM, canonical_dtype, chain_cost, dtype_bytes,
                                 gemm_cost, gemm_intensity, passes_cost, stream_cost, transfer_cost,
                                 write_allocate_bytes)


def test_gemm_small_by_hand():
    c = gemm_cost(2, 2, 2, "float32")          # 8 multiply-adds; read 4+4 floats, write 4 floats
    assert c.flops == 16
    assert (c.bytes_read, c.bytes_written, c.bytes) == (32, 16, 48)
    assert c.intensity == pytest.approx(1 / 3)


def test_gemm_4096_bf16_is_n_over_3():
    c = gemm_cost(4096, 4096, 4096, "bf16")
    assert c.flops == 137_438_953_472            # 2·4096³
    assert c.bytes == 100_663_296                # 3·4096²·2
    assert c.intensity == pytest.approx(4096 / 3)   # square GEMM: 2n³/(3n²·b) = n/(1.5·b)


def test_gemm_rectangular_fp8_in_bf16_out_and_accumulate():
    c = gemm_cost(8, 16, 32, 1, out_bytes=2)
    assert c.flops == 8192
    assert c.bytes_read == (8 * 32 + 32 * 16) * 1 == 768
    assert c.bytes_written == 8 * 16 * 2 == 256
    acc = gemm_cost(8, 16, 32, 2, accumulate=True)   # C += A·B also reads C
    assert acc.bytes_read == (8 * 32 + 32 * 16) * 2 + 8 * 16 * 2


def test_decode_gemv_is_about_one_flop_per_byte_at_bf16():
    i = gemm_intensity(1, 8192, 8192, 2)
    assert i == pytest.approx(2 * 8192 * 8192 / ((8192 + 8192 * 8192 + 8192) * 2))
    assert 0.99 < i < 1.0


def test_stream_bytes_and_flops_per_element_match_mccalpin():
    n = 1_000_000
    assert {k: stream_cost(k, n, 8).bytes / n for k in STREAM} == {"copy": 16, "scale": 16, "add": 24, "triad": 24}
    assert {k: stream_cost(k, n, 8).flops / n for k in STREAM} == {"copy": 0, "scale": 1, "add": 1, "triad": 2}


def test_write_allocate_adds_one_read_per_array_written():
    n = 1000
    assert stream_cost("copy", n, 8).bytes + write_allocate_bytes("copy", n, 8) == 24 * n
    assert stream_cost("triad", n, 8).bytes + write_allocate_bytes("triad", n, 8) == 32 * n


def test_numpy_triad_is_charged_for_both_passes():
    n = 1000
    two_pass = passes_cost(n, 8, NUMPY_TRIAD_PASSES)     # a = q·c (read c, write a); a = a + b (read a, b; write a)
    assert two_pass.bytes == 5 * n * 8 and two_pass.flops == 2 * n
    assert stream_cost("triad", n, 8).bytes == 3 * n * 8


def test_fusion_removes_bytes_not_flops():
    unfused, fused = chain_cost(1000, 8, 8, fused=False), chain_cost(1000, 8, 8, fused=True)
    assert unfused.flops == fused.flops == 8000
    assert unfused.bytes == 2 * 8 * 1000 * 8 and fused.bytes == 2 * 1000 * 8


def test_transfer_counts_delivered_bytes_once():
    t = transfer_cost(1 << 30)
    assert (t.bytes, t.flops, t.basis) == (1 << 30, 0, "transfer")
    assert transfer_cost(100, directions=2).bytes == 200


def test_dtype_names_and_mixing_bases():
    assert canonical_dtype("bf16") == "bfloat16" and canonical_dtype("torch.float16") == "float16"
    assert dtype_bytes("tf32") == 4 and dtype_bytes("fp8") == 1
    with pytest.raises(ValueError):
        canonical_dtype("float7")
    with pytest.raises(ValueError):
        stream_cost("copy", 10, 8) + transfer_cost(10)
