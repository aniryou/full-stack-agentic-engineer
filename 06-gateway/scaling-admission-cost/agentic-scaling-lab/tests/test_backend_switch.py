"""The backend switch: which model backend and provider a simulation runs, and that the default is unchanged.

``make_setup(mode, provider=...)`` picks them; without arguments ``SCALELAB_BACKEND`` / ``SCALELAB_PROVIDER``
do (that is how a notebook is switched without editing it); unset, it is the Google Cloud anchor scenario.
"""

import pytest

from scalelab import capacity, mistral
from scalelab.model import FakeModel, HostedBackend, HybridBackend, ServerPool, cost_per_call, fake_model
from scalelab.sim import BACKEND_ENV, PROVIDER_ENV, make_setup, resolve_backend


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.delenv(PROVIDER_ENV, raising=False)


def test_the_default_is_the_gemini_pool_with_the_primers_cap():
    s = make_setup()
    assert (s.mode, s.provider, s.server) == ("hosted", "gemini", None)
    assert s.standard.model == "gemini-3.5-flash" and s.lite.model == "gemini-3.5-flash-lite"
    assert isinstance(s.standard.backend, HostedBackend) and s.standard.pool is s.standard.backend.pool
    assert s.standard.pool.tpm == 2_000_000 and s.lite.pool.tpm == 4_000_000
    assert s.admission.cfg.max_inflight == 85 and s.admission.cfg.saturation_degrade == float("inf")
    assert (s.standard.backend.ttft_median_s, s.standard.backend.tokens_per_s) == (0.7, 200.0)   # the fake Gemini's latency


@pytest.mark.parametrize("env, backend_type", [("local", ServerPool), ("hybrid", HybridBackend), ("HOSTED", HostedBackend)])
def test_the_environment_switches_the_backend(monkeypatch, env, backend_type):
    monkeypatch.setenv(BACKEND_ENV, env)
    s = make_setup()
    assert isinstance(s.standard.backend, backend_type) and s.mode == env.lower()
    assert s.provider == ("gemini" if env == "HOSTED" else "mistral")      # a fleet serves open-weight Mistral models
    assert (s.server is not None) == (env != "HOSTED")


def test_the_environment_switches_the_provider(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV, "mistral")
    s = make_setup(pool_tpm=20_000_000)
    assert s.standard.model == "mistral-small-2603" and s.lite.model == "ministral-8b-2512"
    derived = mistral.plan(mistral.Scenario(tpm_limit=20_000_000))["concurrency"]["max_inflight_for_limit"]
    assert s.admission.cfg.max_inflight == int(derived) == 174          # the cap the capacity model derives


def test_an_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "local")
    monkeypatch.setenv(PROVIDER_ENV, "mistral")
    assert resolve_backend("hosted", "gemini") == ("hosted", "gemini")
    assert make_setup("hosted", provider="gemini").standard.model == "gemini-3.5-flash"
    assert make_setup(max_inflight=30).admission.cfg.max_inflight == 30


def test_bad_switches_fail_loudly(monkeypatch):
    with pytest.raises(ValueError, match="unknown backend"):
        make_setup("gpu")
    with pytest.raises(ValueError, match="open-weight Mistral"):
        make_setup("local", provider="gemini")
    monkeypatch.setenv(PROVIDER_ENV, "other")
    with pytest.raises(ValueError, match=PROVIDER_ENV):
        make_setup()
    with pytest.raises(ValueError, match="unknown provider"):
        fake_model("other")


def test_each_call_is_priced_from_its_providers_table():
    assert cost_per_call("gemini-3.5-flash", 5000, 350, 2700) == capacity.cost_per_call("gemini-3.5-flash", 5000, 350, 2700)
    assert cost_per_call("mistral-small-2603", 5000, 200, 2700) == mistral.cost_per_call("mistral-small-2603", 5000, 200, 2700)
    with pytest.raises(KeyError):
        cost_per_call("no-such-model", 1, 1)
    assert FakeModel().model == "gemini-3.5-flash" and fake_model("mistral").model == "mistral-small-2603"
