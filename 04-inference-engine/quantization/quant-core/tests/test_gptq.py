"""GPTQ: the Hessian, the OBS update, and when it reduces to RTN."""
import numpy as np

from quantcore import gptq as Q
from quantcore.formats import dequantize_int, int_scale, quantize_int


def _correlated(n=512, d=64, seed=0):
    """Inputs that live near an 8-dimensional subspace (fixed mixing), and a weight to quantize."""
    mix = np.random.default_rng(100).standard_normal((8, d))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8)) @ mix + 0.1 * rng.standard_normal((n, d))
    return X, np.random.default_rng(200).standard_normal((32, d))


def test_hessian_is_two_over_n_xtx():
    X = np.arange(12.0).reshape(4, 3)
    np.testing.assert_allclose(Q.hessian(X), 2 / 4 * sum(np.outer(x, x) for x in X))


def test_uncorrelated_inputs_make_gptq_rtn():
    """With H diagonal, the inverse's Cholesky factor has no off-diagonal terms: nothing to compensate."""
    W = np.random.default_rng(1).standard_normal((16, 64))
    X = np.repeat(np.eye(64) * np.linspace(0.5, 3, 64), 4, axis=0)     # orthogonal columns, unequal scales
    for g in (None, 32):
        np.testing.assert_allclose(Q.gptq(W, X, bits=4, group_size=g).w_hat, Q.rtn(W, 4, g).w_hat, atol=1e-12)


def test_gptq_beats_rtn_on_correlated_inputs_in_and_out_of_sample():
    X, W = _correlated()
    X_test = _correlated(seed=9)[0]                                      # same structure, new rows
    r, g = Q.rtn(W, 3, 32).w_hat, Q.gptq(W, X, bits=3, group_size=32).w_hat
    assert Q.layer_loss(W, g, X) < 0.5 * Q.layer_loss(W, r, X)
    assert Q.layer_loss(W, g, X_test) < 0.5 * Q.layer_loss(W, r, X_test)


def _reference_blocked(W, H, bits, blocksize, percdamp=0.01):
    """A direct transcription of IST-DASLab gptq.py fasterquant (per-channel, symmetric 'full' grid, lazy batch)."""
    W = W.copy()
    n = W.shape[1]
    H = H.copy()
    H[np.diag_indices(n)] += percdamp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T
    scale = int_scale(np.abs(W).max(1), bits, "full")
    Qm = np.zeros_like(W)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1, Err1, Hinv1 = W[:, i1:i2].copy(), np.zeros((W.shape[0], i2 - i1)), Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = dequantize_int(quantize_int(w, scale, bits, None, "full"), scale)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= np.outer(err, Hinv1[i, i:])
            Qm[:, i1 + i], Err1[:, i] = q, err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Qm


def test_matches_the_blocked_reference():
    X, W = _correlated(d=48)
    ref = _reference_blocked(W, Q.hessian(X), 4, blocksize=16)
    np.testing.assert_allclose(Q.gptq(W, X, bits=4, group_size=None).w_hat, ref, atol=1e-9)


def test_actorder_and_dead_inputs():
    X, W = _correlated()
    X[:, 5] = 0.0                                                        # an input that never fires
    q = Q.gptq(W, X, bits=4, group_size=32, actorder=True)
    assert np.all(q.w_hat[:, 5] == 0) and q.scale.shape == (32, 2)
    assert Q.layer_loss(W, q.w_hat, X) < Q.layer_loss(W, Q.rtn(W, 4, 32).w_hat, X)


def test_asymmetric_groups_store_zero_points():
    X, W = _correlated()
    q = Q.gptq(W, X, bits=4, group_size=32, symmetric=False)
    assert q.zero.shape == (32, 2) and np.all((q.zero >= 0) & (q.zero <= 15))
