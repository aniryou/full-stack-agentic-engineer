"""The Numba kernels, run in the CUDA simulator and checked against NumPy."""
import numpy as np
import pytest

from gpurt.kernels import SIMULATOR, blocks_for, cuda
from gpurt.kernels import elementwise as ew
from gpurt.kernels import matmul as mm
from gpurt.kernels import reduction as red
from gpurt.kernels import softmax as sm
from gpurt.kernels import transpose as tr

rng = np.random.default_rng(0)


def test_running_in_the_simulator():
    assert SIMULATOR


def test_launch_config_rounds_up_to_whole_blocks():
    assert [blocks_for(n, 256) for n in (0, 1, 1000, 1024, 1025)] == [1, 1, 4, 4, 5]


def test_vector_add_with_a_partial_last_block():
    a, b = rng.random(300, dtype=np.float32), rng.random(300, dtype=np.float32)
    np.testing.assert_array_equal(ew.run_vec_add(a, b, threads=128), a + b)  # 384 threads for 300 elements


def test_simulator_catches_a_missing_bounds_check():
    @cuda.jit
    def add_no_guard(a, b, out):
        i = cuda.grid(1)
        out[i] = a[i] + b[i]  # the tail threads of the last block index past the end

    a = np.ones(100, np.float32)
    with pytest.raises(IndexError):
        add_no_guard[1, 128](a, a, np.zeros_like(a))


def test_grid_stride_loop_covers_more_elements_than_threads():
    x, y = rng.random(1000, dtype=np.float32), rng.random(1000, dtype=np.float32)
    np.testing.assert_allclose(ew.run_saxpy(2.5, x, y, blocks=2, threads=32), 2.5 * x + y, rtol=1e-6)
    np.testing.assert_array_equal(ew.run_copy(x, blocks=2, threads=32), x)


def test_reduction_two_pass_and_atomic_agree_with_numpy():
    x = rng.random(2000, dtype=np.float32)
    want = float(x.astype(np.float64).sum())
    assert red.run_sum(x, threads=32) == pytest.approx(want, rel=1e-5)
    assert red.run_sum(x, threads=32, atomic=True) == pytest.approx(want, rel=1e-5)


def test_two_pass_launch_count_hand_computed():
    # 2**24 elements, 512 per block: 32768 partials -> 64 -> 1  => 3 launches
    assert red.launches_for_two_pass(2 ** 24, 256) == 3
    assert red.launches_for_two_pass(512, 256) == 1


@pytest.mark.parametrize("variant,tile", [("naive", 16), ("tiled", 8)])
def test_matmul_matches_numpy_on_ragged_shapes(variant, tile):
    A, B = rng.random((20, 24), dtype=np.float32), rng.random((24, 17), dtype=np.float32)
    np.testing.assert_allclose(mm.run_matmul(A, B, variant, tile), A @ B, rtol=1e-5, atol=1e-5)


def test_transposes_on_a_non_square_matrix():
    X = rng.random((40, 70), dtype=np.float32)
    np.testing.assert_array_equal(tr.run_transpose(X, "copy"), X)
    for variant, pad in (("naive", 1), ("tiled", 1), ("tiled", 0)):
        np.testing.assert_array_equal(tr.run_transpose(X, variant, pad), X.T)


@pytest.mark.parametrize("fused", [True, False])
def test_softmax_matches_numpy_and_rows_sum_to_one(fused):
    S = (rng.standard_normal((5, 50)) * 4).astype(np.float32)
    got = sm.run_softmax(S, fused=fused, threads=16)
    np.testing.assert_allclose(got, sm.softmax_reference(S), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-5)


def test_fused_softmax_with_fewer_columns_than_threads():
    S = rng.standard_normal((3, 5)).astype(np.float32)  # most threads see no element: (-inf, 0) pairs
    np.testing.assert_allclose(sm.run_softmax(S, True, threads=16), sm.softmax_reference(S), rtol=1e-5, atol=1e-6)


def test_block_sizes_must_be_powers_of_two():
    with pytest.raises(ValueError):
        red.make_block_sum(96)
    with pytest.raises(ValueError):
        sm.make_softmax_fused(100)


def test_triton_kernels_are_optional_and_lazy():
    import importlib.util

    from gpurt.kernels import triton_kernels

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
        assert triton_kernels.available() is False  # notebook 02 then prints what it would run
    assert "triton" not in triton_kernels.__dict__ and callable(triton_kernels.softmax)  # nothing imported yet
