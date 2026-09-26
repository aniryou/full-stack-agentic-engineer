"""The tiny model and the eval: calibration matters, and the eval says by how much."""
import numpy as np
import pytest

from quantcore import TinyModel, eval as E, quantize_model


@pytest.fixture(scope="module")
def setup():
    m = TinyModel()
    X, y = m.sample(4000, "test")
    Xc, _ = m.sample(256, "calib")
    return m, X, y, Xc, m.forward(X)


def test_deterministic_and_accurate(setup):
    m, X, y, _, ref = setup
    np.testing.assert_array_equal(TinyModel().forward(X), ref)
    assert 0.85 < (ref.argmax(1) == y).mean() < 0.95


def test_activation_outliers_come_from_the_norm_gains(setup):
    m, _, _, Xc, _ = setup
    a = np.abs(m.calibration_inputs(Xc)["blocks.0.up"]).max(0)
    assert np.sort(a)[-4:].min() > 15 * np.median(a)


def test_calibration_recovers_what_rtn_loses(setup):
    m, X, y, Xc, ref = setup
    r = {meth: E.compare(ref, quantize_model(m, meth, 4, 32, calib=Xc).forward(X), y) for meth in ("rtn", "gptq", "awq")}
    assert r["gptq"]["acc"] > r["rtn"]["acc"] + 0.03 and r["gptq"]["kl"] < r["rtn"]["kl"] / 3
    assert r["awq"]["kl"] < r["rtn"]["kl"]
    int3 = {meth: E.compare(ref, quantize_model(m, meth, 3, 32, calib=Xc).forward(X), y)["acc"] for meth in ("rtn", "awq")}
    assert int3["awq"] > int3["rtn"] + 0.03


def test_the_head_stays_in_high_precision(setup):
    m, X, y, _, ref = setup
    head4 = E.compare(ref, quantize_model(m, "rtn", 4, None, targets=["head"]).forward(X), y)
    head8 = E.compare(ref, quantize_model(m, "rtn", 8, None, targets=["head"]).forward(X), y)
    assert head4["acc_ref"] - head4["acc"] > 2 * head4["stderr"] and head8["kl"] < 0.001


def test_flips_explain_the_delta(setup):
    m, X, y, _, ref = setup
    r = E.compare(ref, quantize_model(m, "rtn", 4, 32).forward(X), y)
    assert r["lost"] - r["gained"] == round((r["acc_ref"] - r["acc"]) * len(y)) and r["gained"] > 0
    assert not E.within_budget(r) and E.within_budget(E.compare(ref, ref, y))


def test_stderr_matches_lm_eval():
    assert round(E.accuracy_stderr(0.768, 250), 4) == 0.0268       # vLLM's FP8 gsm8k example: 0.768 +- 0.0268


def test_a_difference_has_a_wider_error_bar_than_a_score():
    """Unpaired: se_diff = sqrt(se_a^2 + se_b^2) = sqrt(2) x one score's se at equal p (the lab's evalharness.compare)."""
    assert E.diff_stderr(0.768, 0.768, 250) == pytest.approx(np.sqrt(2) * E.accuracy_stderr(0.768, 250))
    assert round(2 * E.diff_stderr(0.768, 0.768, 250), 3) == 0.076                  # 7.6 points on 250 items
    n = next(n for n in range(1000, 100000) if 2 * E.diff_stderr(0.77, 0.77, n) <= 0.01)
    assert n == 14169                                                             # 2 x 4 x 0.1771 / 1e-4 + 1
    assert E.paired_z(10, 10) == 0 and E.paired_z(0, 0) == 0 and round(E.paired_z(64, 36), 1) == -2.8


def test_paired_comparison_sees_what_the_unpaired_one_cannot(setup):
    m, X, y, Xc, ref = setup
    r = E.compare(ref, quantize_model(m, "awq+gptq", 4, 32, calib=Xc).forward(X), y)
    assert r["n"] == len(y) and r["diff_stderr"] > r["stderr"]
    drop = r["acc_ref"] - r["acc"]
    assert drop < 2 * r["diff_stderr"] and E.within_budget(r, max_kl=0.05)      # unpaired: within noise
    assert abs(r["paired_z"]) > 2                                                 # paired: the flips say it is real


def test_kl_and_perplexity_basics():
    z = np.random.default_rng(0).standard_normal((10, 16))
    assert E.kl(z, z) == pytest.approx(0.0, abs=1e-12)
    assert E.perplexity(np.zeros((10, 16)), np.arange(10)) == pytest.approx(16.0)
