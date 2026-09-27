"""The T1 code paths of notebooks 01-04, run offline against a vLLM stand-in (a fake provider serving `lab/llm` with
vLLM's /health, /metrics names, 16-token prefix cache and cache_salt), including a vLLM that is down or dies mid-run."""
import pytest

from gwlab import env, t1
from gwlab.fakes import FakeSpec
from gwlab.stack import LocalStack

FAST = dict(ttft_s=0.01, prefill_s_per_token=0.0, itl_s=0.001, output_tokens=8)
FALLBACK = FakeSpec(name="acme", **FAST)
SYSTEM = "You are the support agent for a large company. Follow the policy exactly and cite the article you used. " * 4


@pytest.fixture
def vllm():
    """A stand-in `vllm serve ... --served-model-name lab/llm` (no --api-key), on its own port."""
    spec = FakeSpec(name="acme", models={"lab/llm": {"context_window": 4096}}, keys=("",), **FAST)
    s = LocalStack(config="lab", fakes={"acme": spec}).start()
    s.vllm_url = s.fake_url("acme")
    s.kill = lambda: s.run(s.fakes["acme"].stop())
    try:
        yield s
    finally:
        s.stop()


def test_t1_is_detected_only_when_vllm_answers(vllm, monkeypatch):
    monkeypatch.setenv(env.VLLM_ENV, vllm.vllm_url)
    assert env.vllm_url(quiet=True) == vllm.vllm_url
    monkeypatch.setenv(env.VLLM_ENV, "http://127.0.0.1:9")                  # set, but nothing listens
    assert env.vllm_url(quiet=True) is None and env.describe()["vllm_url"] is None


def test_results_are_labelled_by_what_served_them(vllm):
    rows = t1.first_requests(vllm.vllm_url, ["What does a gateway own?"], FALLBACK)
    assert rows[0][0] == "MEASURED" and rows[0][1] == "local/llm" and rows[0][3]["completion_tokens"] == 8
    vllm.kill()
    rows = t1.first_requests(vllm.vllm_url, ["What does a gateway own?"], FALLBACK)
    assert rows[0][0].startswith("SIMULATED") and rows[0][1] == "acme/fast"     # never printed as a measurement


def test_prefix_cache_rule_and_the_salt(vllm):
    rows = t1.prefix_cache(vllm.vllm_url, SYSTEM, FALLBACK)
    assert [r[1] for r in rows] == ["MEASURED"] * 3
    t1.check_prefix_rows(rows)
    assert rows[0][3] == 0 and rows[1][3] > 0 and rows[2][3] == 0
    again = t1.prefix_cache(vllm.vllm_url, SYSTEM, FALLBACK)                     # a second run: the nonce keeps it cold
    assert again[0][3] == 0
    unsalted = t1.prefix_cache(vllm.vllm_url, SYSTEM, FALLBACK, {"providers": {"local": {"cache_salt": False}}})
    with pytest.raises(AssertionError, match="cache_salt"):
        t1.check_prefix_rows(unsalted)                                           # team-b hit team-a's blocks


def test_stopping_vllm_mid_run_moves_the_chain_to_the_fallback(vllm):
    res = t1.stop_midrun(vllm.vllm_url, vllm.kill, FALLBACK, rate=20, n=60, stop_after_s=1.0,
                         breaker={"failure_threshold": 3, "recovery_timeout_s": 30})
    line, k = res["line"], res["line"].count("v")
    assert k >= 5 and line == "v" * k + "a" * (60 - k)                 # vLLM until the stop, the fallback after; none lost
    assert res["breaker_opens"] == 1 and 3 <= res["reached_dead"] <= 6  # a few connection errors open the breaker


def test_ledger_reconciles_with_vllms_counters(vllm):
    out = t1.reconcile(vllm.vllm_url, n=8, fallback=FALLBACK)
    rec = out["reconcile"]["local"]
    assert out["served_by_vllm"] == 8 and rec["prompt_gap"] == 0 and rec["completion_gap"] == 0
    vllm.kill()
    assert t1.reconcile(vllm.vllm_url, n=2, fallback=FALLBACK) is None          # /metrics down: skipped, not a crash
