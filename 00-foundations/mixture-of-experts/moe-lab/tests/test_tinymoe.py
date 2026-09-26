"""The tiny MoE: the toy task, the statistics, the recorded curves, and (with torch) the layer and a short run."""
import math

import numpy as np
import pytest

from moelab import env
from moelab.tinymoe import ToyTask, load_bundled, load_stats, specialisation, table

HAS_TORCH = env.has_torch()


def test_toy_task_follows_its_rules():
    t = ToyTask()
    x, dom = t.batch(np.random.default_rng(0), 500)
    assert x.shape == (500, t.seq_len) and (x[:, 0] == dom).all() and (x[:, 1:] >= t.n_domains).all()
    rules = t.rules()
    cur, nxt = x[:, 1:-1] - t.n_domains, x[:, 2:] - t.n_domains
    follow = (rules[dom[:, None], cur] == nxt).mean()
    assert follow == pytest.approx(1 - t.noise + t.noise / t.n_content, abs=0.02)


def test_bayes_loss_by_hand():
    t = ToyTask(n_content=4, noise=0.2)
    p_rule, p_other = 0.8 + 0.05, 0.05
    assert t.bayes_loss() == pytest.approx(-(p_rule * math.log(p_rule) + 3 * p_other * math.log(p_other)))


def test_load_stats_and_specialisation():
    s = load_stats([0.5, 0.5, 0, 0])
    assert s["max_over_mean"] == 2.0 and s["dead"] == 2 and s["entropy"] == pytest.approx(0.5)
    assert specialisation(np.eye(3) * 7) == pytest.approx(1.0)
    assert specialisation(np.ones((3, 5))) == pytest.approx(0.0, abs=1e-12)


def test_bundled_curves_tell_the_story():
    doc = load_bundled()
    assert "recorded" in doc["_label"]
    runs = {(r["config"]["balance"], r["config"]["seed"]): r for r in doc["runs"]}
    assert len(runs) == 9
    for seed in (0, 1, 2):
        none = load_stats(runs[("none", seed)]["load"][-1])
        assert none["max_over_mean"] > 4 and none["dead"] >= 3           # collapse: a few experts carry it all
        for b in ("aux", "bias"):
            bal = runs[(b, seed)]
            assert load_stats(bal["load"][-1])["max_over_mean"] < 1.3 and load_stats(bal["load"][-1])["dead"] == 0
            assert bal["ce"][-1] < runs[("none", seed)]["ce"][-1]         # and collapse costs loss
            # a healthy router keys on the token in front of it far more than on the domain
            assert specialisation(bal["token_expert"]) > 5 * specialisation(bal["domain_expert"])
    for r in runs.values():
        assert r["config"]["common"] > 0
        assert len(r["load"]) == len(r["steps"]) and np.isclose(sum(r["load"][-1]), 1.0)
    assert "by token" in table(doc["runs"])


needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (T0 numpy path)")


@needs_torch
def test_layer_equals_dense_masked_compute():
    import torch
    from moelab.tinymoe.model import TinyMoE
    torch.manual_seed(0)
    layer = TinyMoE(16, 4, 2, 8, shared_ff=8)
    x = torch.randn(30, 16)
    with torch.no_grad():
        y = layer(x)
        logits, w, idx = layer.router(x)
        dense = torch.zeros_like(x)
        for e in range(4):
            gate = ((idx == e) * w).sum(-1, keepdim=True)
            dense += layer.expert(e, x) * gate
        g, u = layer.shared["gate_up"](x).chunk(2, -1)
        dense += layer.shared["down"](torch.nn.functional.silu(g) * u)
    assert torch.allclose(y, dense, atol=1e-5)
    total, active = layer.params()
    assert total - active == 2 * 3 * 16 * 8                                  # E - k experts idle per token


@needs_torch
def test_router_layout_bias_and_renormalisation():
    import torch
    from moelab.tinymoe.model import TinyTopKRouter
    r = TinyTopKRouter(4, 4, 2)
    x = torch.randn(10, 4)
    logits, w, idx = r(x)
    assert torch.allclose(w.sum(-1), torch.ones(10))                         # renormalised at k = 2
    r.bias[:] = torch.tensor([10.0, 10.0, 0, 0])                            # the bias forces the choice ...
    _, w2, idx2 = r(x)
    assert (idx2.sort(-1).values == torch.tensor([0, 1])).all()
    p = logits.softmax(-1)
    chosen = p.gather(1, idx2)
    assert torch.allclose(w2, chosen / chosen.sum(-1, keepdim=True), atol=1e-6)      # ... not the weights
    top1 = TinyTopKRouter(4, 4, 1)
    _, w1, _ = top1(x)
    assert (w1 < 1).all()                                                    # Switch: raw probability at k = 1


@needs_torch
def test_aux_loss_values():
    import torch
    from moelab.tinymoe.model import router_z_loss, switch_aux_loss
    logits = torch.zeros(8, 4)
    assert float(switch_aux_loss(logits, torch.arange(8).reshape(8, 1) % 4, 4)) == pytest.approx(1.0)
    assert float(switch_aux_loss(logits, torch.tensor([[0, 1], [2, 3]] * 4), 4)) == pytest.approx(2.0)
    assert float(router_z_loss(logits)) == pytest.approx(math.log(4) ** 2)


@needs_torch
def test_a_short_training_run_records_everything():
    from moelab.tinymoe.train import TrainConfig, train
    run, model = train(TrainConfig(balance="bias", steps=100, log_every=10, eval_batch=32))
    assert run.steps == list(range(10, 101, 10)) and len(run.load) == 10
    assert run.ce[-1] < 0.8 * run.ce[0]                                       # it learns (3.9 -> 2.8 nats)
    assert np.asarray(run.domain_expert).shape == (8, 8) and np.asarray(run.token_expert).shape == (48, 8)
    assert float(model.moes[0].router.bias.abs().sum()) > 0                 # the bias moved
    assert "specialisation" not in run.summary() and "by token" in run.summary()


@needs_torch
def test_live_training_collapses_without_balancing_and_balances_with_it():
    """The bundled curves' story, from the code: 300 steps each, ~5 s on one CPU thread."""
    from moelab.tinymoe.train import TrainConfig, train
    none, _ = train(TrainConfig(balance="none", steps=300, eval_batch=32))
    bias, _ = train(TrainConfig(balance="bias", steps=300, eval_batch=32))
    s_none, s_bias = load_stats(none.final_load), load_stats(bias.final_load)
    assert s_none["max_over_mean"] > s_bias["max_over_mean"] + 0.4
    assert s_none["max_over_mean"] > 4 and s_none["dead"] >= 3 and s_bias["max_over_mean"] < 1.3
    assert none.ce[-1] > bias.ce[-1]


def test_the_cli_refuses_cleanly_without_torch(monkeypatch):
    import subprocess
    import sys
    env_ = {**__import__("os").environ, "MOELAB_NO_TORCH": "1"}
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "moelab.tinymoe", "--steps", "5"], capture_output=True, text=True,
                       env=env_, cwd=root)
    assert r.returncode == 1 and "torch is not available" in r.stderr and "load_bundled" in r.stderr
    assert "Traceback" not in r.stderr


def test_torch_parts_are_lazy(monkeypatch):
    import moelab.tinymoe as tm
    monkeypatch.setenv("MOELAB_NO_TORCH", "1")
    with pytest.raises(ImportError, match="torch is not available"):
        tm.__getattr__("train")
