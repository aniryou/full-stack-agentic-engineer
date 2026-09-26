"""Every kernel compiles to PTX for a real GPU (sm_75, sm_89) — checked only where numba-cuda is installed.

The rest of the suite runs the kernels in Numba's simulator, which never compiles them. This test runs
``tools/check_ptx.py`` in a fresh process (the real CUDA target must be chosen before numba.cuda is
imported, and this process is already in simulator mode). No GPU is needed, only numba-cuda and its NVVM.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("numba_cuda") is None,
                    reason="numba-cuda not installed (pip install -e '.[gpu]'): PTX compilation not checked")
def test_every_kernel_compiles_to_ptx_for_t4_and_l4():
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "check_ptx.py")], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    if out.returncode == 2:
        pytest.skip(out.stdout.strip())
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "failures: 0" in out.stdout
