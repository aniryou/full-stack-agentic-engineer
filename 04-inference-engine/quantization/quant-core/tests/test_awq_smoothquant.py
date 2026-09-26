"""AWQ's scale search and SmoothQuant's migration: exact reparameterisations that move error around."""
import numpy as np
import pytest

from quantcore import TinyModel, awq, granularity as G, smoothquant as S
from quantcore.gptq import layer_loss, rtn


@pytest.fixture(scope="module")
def model_and_calib():
    m = TinyModel()
    X, _ = m.sample(256, "calib")
    return m, m.calibration_inputs(X)


def test_alpha_zero_is_rtn(model_and_calib):
    m, cap = model_and_calib
    W, X = m.weights["blocks.0.up"], cap["blocks.0.up"]
    _, _, losses = awq.search_scale(W, X, bits=4, group_size=32, symmetric=True)
    assert losses[0] == pytest.approx(layer_loss(W, rtn(W, 4, 32).w_hat, X))


def test_awq_protects_the_layer_with_outlier_channels(model_and_calib):
    m, cap = model_and_calib
    for name in ("blocks.0.up", "blocks.1.up"):                  # inputs: RMSNorm with 25-40x gains
        W, X = m.weights[name], cap[name]
        q, s = awq.awq(W, X, bits=4, group_size=32, symmetric=True)
        assert G.output_error(X / s, W * s, q.w_hat) < 0.85 * G.output_error(X, W, rtn(W, 4, 32).w_hat)
        assert s[np.argmax(np.abs(X).mean(0))] == s.max()         # the largest activation channel is scaled most


def test_fold_is_exact(model_and_calib):
    m, _ = model_and_calib
    X, _ = m.sample(64)
    s = np.random.default_rng(0).uniform(0.5, 2.0, 64)
    s2 = np.random.default_rng(1).uniform(0.5, 2.0, 256)
    folded = m.with_weights({**m.fold("blocks.0.up", s)}).with_weights(m.with_weights(m.fold("blocks.0.up", s)).fold("blocks.1.down", s2))
    np.testing.assert_allclose(folded.forward(X), m.forward(X), atol=1e-9)


def test_smoothquant_is_exact_and_helps_w8a8(model_and_calib):
    m, cap = model_and_calib
    X, W = cap["blocks.0.up"], m.weights["blocks.0.up"]
    Xs, Ws, s = S.smooth(X, W, 0.5)
    np.testing.assert_allclose(Xs @ Ws.T, X @ W.T, atol=1e-9)
    sweep = S.alpha_sweep(X, W)
    assert sweep[0.5] < S.w8a8_error(X, W) / 2                   # 2.7x less output error on this layer
    best = min(sweep, key=sweep.get)
    assert 0 < best < 1                                          # neither extreme: split the difficulty


def test_smoothquant_scales_formula():
    s = S.smooth_scales(np.array([16.0, 1.0]), np.array([1.0, 4.0]), alpha=0.5)
    np.testing.assert_allclose(s, [4.0, 0.5])
