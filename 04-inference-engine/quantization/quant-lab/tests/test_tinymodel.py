"""The bundled tiny checkpoint: a real Llama layout, a task it solves, and planted outlier channels."""
import sys
from pathlib import Path

import numpy as np
import pytest

from quantlab import stio, tinymodel as tm


@pytest.fixture(scope="module")
def model():
    return tm.load()


def test_checkpoint_layout_is_llama(model):
    rows = stio.summary(tm.TINY_DIR / "model.safetensors")
    names = {n for n, *_ in rows}
    assert "model.layers.1.mlp.down_proj.weight" in names and "lm_head.weight" in names
    assert all(dt == "BF16" for _, dt, *_ in rows)
    assert model.num_params() == 299_648 and len(model.linear_names()) == 14
    assert model.config["model_type"] == "llama" and model.config["hidden_size"] % 128 == 0   # W4A16 g128 is legal


def test_it_solves_both_tasks(model):
    assert model.accuracy("add", 300) >= 0.99 and model.accuracy("reverse", 300) >= 0.99
    p, a = tm.make_task("add", 3, 1000)
    assert tm.decode(p[0]).endswith("=") and a.shape == (3, 4)


def test_cached_generation_equals_full_recompute(model):
    p, _ = tm.make_task("reverse", 20, 1001)
    seq = p
    for _ in range(6):
        seq = np.concatenate([seq, model.forward(seq)[:, -1].argmax(-1)[:, None]], 1)
    assert (seq[:, p.shape[1]:] == model.generate(p, 6)).all()


def test_planted_outliers_are_massive_activations(model):
    chans = tm.outlier_channels(model)
    assert len(chans) == 2
    cap = {}
    p, a = tm.make_task("add", 64, 7)
    model.answer_logits(p, a, capture=cap)
    x = np.abs(np.concatenate(cap["model.layers.0.self_attn.q_proj"]))
    others = np.delete(x.mean(0), chans)
    assert x.mean(0)[chans].min() > 8 * np.median(others)                      # the inputs carry them


def test_numpy_forward_matches_torch_when_available(model):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import train_tiny
    t = train_tiny.build_torch_model(torch)
    t.load_state_dict({k: torch.from_numpy(v.astype(np.float32)) for k, v in model.weights.items()})
    ids = tm.make_task("add", 4, 1000)[0]
    ref = t(torch.from_numpy(ids).long()).detach().numpy()
    np.testing.assert_allclose(model.forward(ids), ref, atol=2e-3)
