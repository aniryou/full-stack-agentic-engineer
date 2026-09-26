"""Environment detection and the launch-overhead model."""
import importlib.util
import os
import pathlib
import subprocess
import sys

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


def test_a_gpu_numba_cannot_use_falls_back_to_the_simulator():
    def boom():
        raise AssertionError("the probe must not run")

    mode, why = env.choose_mode(None, 1, probe=lambda: (False, "NVVM missing"))
    assert mode == "simulator" and "NVVM missing" in why and "numba-cuda" in why
    assert env.choose_mode(None, 2, probe=lambda: (True, "ok")) == ("cuda", None)
    assert env.choose_mode(None, 0, probe=boom) == ("simulator", None)  # no GPU: nothing to probe
    assert env.choose_mode("0", 1, probe=boom) == ("cuda", None)  # an explicit choice wins
    assert env.choose_mode("1", 1, probe=boom) == ("simulator", None)


def test_fallback_reaches_gpurt_kernels_in_a_fresh_process():
    """What a Colab GPU runtime without numba-cuda sees: a visible GPU, an unusable CUDA target."""
    code = ("import gpurt.env as e; e.gpu_count = lambda: 1; "
            "e.numba_cuda_usable = lambda **k: (False, 'numba-cuda not installed'); "
            "import gpurt.kernels as K, numpy as np; from gpurt.kernels import elementwise as ew; "
            "x = np.arange(100, dtype=np.float32); assert K.SIMULATOR and (ew.run_vec_add(x, x) == 2 * x).all(); "
            "print(e.describe())")
    root = pathlib.Path(__file__).resolve().parents[1]
    environ = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=environ, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "fell back to its simulator" in out.stdout and "numba-cuda not installed" in out.stdout


def test_the_real_probe_answers_without_touching_this_process():
    ok, why = env.numba_cuda_usable(timeout=120)
    assert isinstance(ok, bool) and why
    assert env.numba_mode() in ("simulator", "unset")  # this process is still in simulator mode


def test_summary_keys():
    assert {"tier", "gpus", "numba_mode", "torch_installed", "platform"} <= set(env.summary())


def test_launch_model_hand_computed():
    m = LaunchModel(launch_us=5.0, graph_launch_us=10.0, node_gap_us=1.0)
    assert m.eager_us(100, 2.0) == 500.0  # launch-bound: 100 x max(2 + 1, 5)
    assert m.graph_us(100, 2.0) == 310.0  # 10 + 100 x (2 + 1)
    assert m.speedup(100, 2.0) == pytest.approx(500 / 310)
    assert m.launch_bound(2.0) and not m.launch_bound(50.0)
    assert m.gpu_idle_fraction(2.0) == pytest.approx(0.6)
    assert m.eager_us(100, 50.0) == 5100.0  # compute-bound: the CPU keeps ahead ...
    assert m.graph_us(100, 50.0) == 5110.0  # ... so a graph buys nothing


def test_launch_model_defaults_reproduce_primer_4_2():
    m = LaunchModel()  # L = 5 µs, G = 10 µs, no GPU-side gap: primer §4.2's assumptions
    assert (m.eager_us(384, 2.0), m.graph_us(384, 2.0)) == (1920.0, 778.0)  # primer: 1,922 vs 778 µs, 2.5x
    assert (m.eager_us(384, 20.0), m.graph_us(384, 20.0)) == (7680.0, 7690.0)  # primer: 7,685 vs 7,690
    gap = LaunchModel(node_gap_us=1.0)
    assert gap.graph_us(384, 2.0) == 1162.0 and gap.speedup(384, 2.0) == pytest.approx(1.652, abs=1e-3)


def test_torch_paths_are_optional():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed: gloo/NCCL sweeps and CUDA Graphs are T1 paths")
    from gpurt.dist import bench

    rows = bench.run("gloo", "all_reduce", 2, sizes=[64, 4096], iters=2, warmup=1)
    assert all(r.wrong == 0 for r in rows)
