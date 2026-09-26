"""The numpy backend computes what it claims and charges exactly the accounted bytes."""
import numpy as np
import pytest

from gpubench import gemm, membw, transfer
from gpubench.backends.numpy_backend import NumpyBackend, split

be = NumpyBackend()


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_gemm_op_computes_the_product(dtype):
    op = be.make_gemm(24, 16, 8, dtype)
    op.fn()
    assert op.verify()
    assert op.cost.flops == 2 * 24 * 16 * 8
    assert op.cost.bytes == (24 * 8 + 8 * 16 + 24 * 16) * np.dtype(dtype).itemsize


def test_run_gemm_measurement_carries_cost_and_rates():
    m = gemm.run_gemm(be, 32, 32, 32, "float32", repeats=2, min_time=0.001, check=True)
    assert m.params == {"m": 32, "n": 32, "k": 32, "dtype": "float32"}
    assert m.cost.flops == 65_536 and m.cost.bytes == 3 * 32 * 32 * 4
    assert m.flops_per_s() == pytest.approx(m.cost.flops / m.timing.best)
    assert m.flops_per_s("best") >= m.flops_per_s("median")


@pytest.mark.parametrize("kernel", ["copy", "scale", "add", "triad"])
@pytest.mark.parametrize("threads", [1, 3])
def test_stream_kernels_compute_what_they_claim(kernel, threads):
    op = be.make_stream(kernel, 1001, threads=threads)
    op.fn()
    assert op.verify()


def test_stream_ops_carry_both_byte_conventions():
    add, triad = be.make_stream("add", 1000), be.make_stream("triad", 1000)
    assert add.cost.bytes == add.extras["stream_bytes"] == 24_000
    assert triad.cost.bytes == 40_000 and triad.extras["stream_bytes"] == 24_000   # two passes moved, three counted


def test_split_covers_every_element_once():
    parts = split(10, 3)
    covered = [i for s in parts for i in range(10)[s]]
    assert covered == list(range(10)) and len(parts) == 3
    assert len(split(2, 8)) == 2


def test_fused_chain_matches_unfused_and_moves_k_times_fewer_bytes():
    u, f = be.make_chain(5000, k=8, fused=False), be.make_chain(5000, k=8, fused=True, block_elems=512)
    u.fn(), f.fn()
    assert u.verify() and f.verify()
    assert u.cost.flops == f.cost.flops and u.cost.bytes == 8 * f.cost.bytes


def test_memcpy_and_ladder_ops():
    op = be.make_memcpy(4096)
    op.fn()
    assert op.verify() and op.cost.bytes == 4096 and op.extras["stream_bytes"] == 8192
    lad = be.make_scale_inplace(8192)
    assert lad.cost.bytes == 2 * 8192 and lad.extras["working_set_bytes"] == 8192


def test_small_suites_produce_labelled_measurements():
    ms = membw.stream_suite(be, n=1 << 14, repeats=2, min_time=0.001)
    assert [m.op for m in ms] == ["stream.copy", "stream.scale", "stream.add", "stream.triad"]
    ms = transfer.memcpy_sweep(be, [4096, 65536], repeats=2, min_time=0.001)
    assert [m.params["nbytes"] for m in ms] == [4096, 65536] and all(m.cost.basis == "transfer" for m in ms)
    u, f = membw.fusion(be, n=1 << 14, k=4, block_elems=1024, repeats=2, min_time=0.001)
    assert (u.op, f.op) == ("chain.unfused", "chain.fused")
