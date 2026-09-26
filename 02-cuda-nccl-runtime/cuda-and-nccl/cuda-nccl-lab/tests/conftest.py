"""Tests are offline and GPU-free: force Numba's CUDA simulator before anything imports numba.cuda."""
import os
import pathlib
import sys

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

FIXTURES = ROOT / "gpurt" / "fixtures"


@pytest.fixture(autouse=True)
def _fast_simulator():
    """Shorter GIL switch interval: barrier-heavy simulated kernels run several times faster."""
    from gpurt.env import fast_simulator

    with fast_simulator():
        yield


@pytest.fixture
def fixture_text():
    return lambda name: (FIXTURES / name).read_text()
