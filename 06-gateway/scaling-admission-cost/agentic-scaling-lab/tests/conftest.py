"""Shared fixtures: every test runs with the backend switch unset, so an exported ``SCALELAB_BACKEND`` or
``SCALELAB_PROVIDER`` (the notebooks' switch) cannot change what a test measures. A test that exercises the
switch sets the variable itself with ``monkeypatch.setenv``.
"""

import pytest

from scalelab.sim import BACKEND_ENV, PROVIDER_ENV


@pytest.fixture(autouse=True)
def _no_backend_switch_from_the_environment(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.delenv(PROVIDER_ENV, raising=False)
