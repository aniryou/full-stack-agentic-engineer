"""Environment detection and the launch-overhead model."""
import importlib.util

import pytest

from gpurt import env
from gpurt.launch import LaunchModel


def test_tiers():
    assert [env.tier(n) for n in (0, 1, 2, 8)] == ["T0", "T1", "T2", "T2"]


def test_numba_mode_is_fixed_once_numba_cuda_is_imported():
    import gpurt.kernels  # noqa: F401  (imports numba.cuda in simulator mode under the test conftest)

    assert env.ensure_numba_mode() == "simulator"
    with pytest.raises(RuntimeError):
        env.ensure_numba_mode(simulator=False)


def test_summary_keys():
    assert {"tier", "gpus", "numba_mode", "torch_installed", "platform"} <= set(env.summary())


def test_launch_model_hand_computed():
    m = LaunchModel(launch_us=5.0, graph_launch_us=10.0, node_gap_us=1.0)
    assert m.eager_us(100, 2.0) == 500.0  # launch-bound: 100 x max(2, 5)
    assert m.graph_us(100, 2.0) == 310.0  # 10 + 100 x (2 + 1)
    assert m.speedup(100, 2.0) == pytest.approx(500 / 310)
    assert m.launch_bound(2.0) and not m.launch_bound(50.0)
    assert m.gpu_idle_fraction(2.0) == pytest.approx(0.6)
    assert m.eager_us(100, 50.0) == 5000.0  # compute-bound: the CPU keeps ahead


def test_torch_paths_are_optional():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed: gloo/NCCL sweeps and CUDA Graphs are T1 paths")
    from gpurt.dist import bench

    rows = bench.run("gloo", "all_reduce", 2, sizes=[64, 4096], iters=2, warmup=1)
    assert all(r.wrong == 0 for r in rows)
