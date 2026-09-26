"""Granularity: scale shapes, outliers in rows vs columns, activation scales."""
import numpy as np
import pytest

from quantcore import granularity as G


def test_scale_shapes_follow_compressed_tensors():
    W = np.random.default_rng(0).standard_normal((256, 512))
    assert G.quantize(W, "int8", "tensor").scale.shape == (1,)
    assert G.quantize(W, "int8", "channel").scale.shape == (256, 1)
    assert G.quantize(W, "int4", "group", group_size=128).scale.shape == (256, 4)
    q = G.quantize(W, "fp8", "block", block=(128, 128))
    assert q.scale.shape == (2, 4) and q.w_hat.shape == W.shape


def test_block_scales_are_per_tile():
    W = np.random.default_rng(1).standard_normal((256, 256))
    W[:128, :128] *= 50                                         # one hot tile
    e = lambda fmt, gran: G.error(W[128:], G.fake_quant(W, fmt=fmt, granularity=gran)[128:])["rel"]
    assert G.quantize(W, "fp8", "block").scale[0, 0] > 20 * G.quantize(W, "fp8", "block").scale[1, 1]
    assert e("int8", "block") < e("int8", "tensor") / 10               # an integer grid needs the local scale...
    assert e("fp8", "tensor") < 1.1 * e("fp8", "block")                # ...a float grid's range absorbs a 50x tile


def test_group_size_must_divide_in_features():
    with pytest.raises(ValueError, match="group_size"):
        G.quantize(np.zeros((8, 576)), "int4", "group", group_size=128)    # SmolLM2-style 576 % 128 = 64


def test_full_convention_uses_the_extra_code():
    W = np.random.default_rng(2).standard_normal((128, 256))
    r = G.error(W, G.fake_quant(W, fmt="int4", granularity="group", group_size=32))["rel"]
    f = G.error(W, G.fake_quant(W, fmt="int4", granularity="group", group_size=32, convention="full"))["rel"]
    assert f < r


def test_per_channel_does_not_isolate_an_outlier_input_column():
    W = np.random.default_rng(0).standard_normal((256, 512)) * 0.02
    W[:, 7] *= 20
    rest = [c for c in range(512) if c != 7]
    others = {g: G.error(W[:, rest], G.fake_quant(W, fmt="int4", granularity="channel" if g is None else "group",
                                                  group_size=g or 128, convention="full")[:, rest])["rel"]
              for g in (None, 128, 32)}
    assert others[None] > 0.5 and others[128] < others[None] and others[32] < 0.2


def test_row_errors_expose_what_the_aggregate_hides():
    W = np.random.default_rng(1).standard_normal((32, 64)) * 0.02
    W[3] *= 100
    rows = G.row_errors(W, G.fake_quant(W, fmt="int8", granularity="tensor"))
    assert G.error(W, G.fake_quant(W, fmt="int8", granularity="tensor"))["rel"] < 0.05 and np.median(rows) > 0.5


def test_dynamic_per_token_vs_static_activation_scales():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 32))
    X[:, 4] *= 40                                                # an outlier activation channel
    normal = [c for c in range(32) if c != 4]
    tok = G.quantize_activations(X, "int8", "token")
    ten = G.quantize_activations(X, "int8", "tensor")
    assert G.error(X[:, normal], tok[:, normal])["rel"] < G.error(X[:, normal], ten[:, normal])["rel"]
    fp8 = G.quantize_activations(X, "fp8", "token")              # a float grid keeps small values' relative precision
    assert G.error(X[:, normal], fp8[:, normal])["rel"] < G.error(X[:, normal], tok[:, normal])["rel"]
    clipped = G.quantize_activations(X * 3, "int8", static_amax=np.abs(X).max())
    assert np.abs(clipped).max() <= np.abs(X).max() + 1e-9      # a static scale saturates what calibration never saw


def test_argmax_agreement():
    a = np.array([[1.0, 2.0], [3.0, 0.0]])
    assert G.argmax_agreement(a, a) == 1.0 and G.argmax_agreement(a, -a) == 0.0


def test_output_error_follows_the_activation_not_the_weight():
    """Two identical weight columns, one meeting 30x larger inputs: that channel owns ~900x the output error."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 2))
    W[:, 1] = W[:, 0]
    X = rng.standard_normal((500, 2)) * [1.0, 30.0]
    e = G.output_error_by_input(X, W, G.fake_quant(W, fmt="int4", granularity="channel"))
    assert 700 < e[1] / e[0] < 1100
