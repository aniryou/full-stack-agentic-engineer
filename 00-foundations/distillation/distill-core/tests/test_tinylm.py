"""The tiny model family: exact gradients, sampling, the capacity gap, pruning."""
import numpy as np

from distillcore import ModLang, TinyLM, eval as E, losses as L, train
from distillcore.tinylm import fit_language


def test_backward_matches_finite_differences():
    m, rng = TinyLM(7, 5, 3, seed=0), np.random.default_rng(0)
    ctx, W = rng.integers(0, 7, (6, 2)), rng.normal(size=(6, 7))
    z, cache = m.forward(ctx)
    g = m.backward(cache, W)                          # gradient of Σ <W, logits>
    for k, a in m.p.items():
        idx = tuple(rng.integers(0, s) for s in a.shape)
        old = a[idx]
        a[idx] = old + 1e-6
        up = (m.logits(ctx) * W).sum()
        a[idx] = old - 1e-6
        dn = (m.logits(ctx) * W).sum()
        a[idx] = old
        assert abs((up - dn) / 2e-6 - g[k][idx]) < 1e-6, k


def test_grad_logprob_is_the_gradient_of_weighted_token_logprobs():
    m, rng = TinyLM(5, 4, 3, seed=2), np.random.default_rng(1)
    seqs = m.sample(rng.integers(0, 5, (4, 2)), 3, rng)
    w = rng.normal(size=(4, 3))
    g = m.grad_logprob(seqs, w)
    k, idx = "W1", (1, 2)
    old = m.p[k][idx]
    m.p[k][idx] = old + 1e-6
    up = (w * m.token_logprobs(seqs)).sum()
    m.p[k][idx] = old - 1e-6
    dn = (w * m.token_logprobs(seqs)).sum()
    m.p[k][idx] = old
    assert abs((up - dn) / 2e-6 - g[k][idx]) < 1e-6
    per_seq = m.grad_logprob(seqs, w[:, 0])           # one weight per sequence broadcasts to its tokens
    assert np.allclose(per_seq["b2"], m.grad_logprob(seqs, np.repeat(w[:, :1], 3, 1))["b2"])


def test_greedy_sampling_is_argmax_and_logprobs_have_one_entry_per_token():
    m = TinyLM(11, 8, seed=3)
    s = m.sample(np.array([[1, 2], [3, 4]]), 5, None, T=0.0)
    ctx, y = m.positions(s)
    assert s.shape == (2, 7) and np.all(m.logits(ctx).argmax(1) == y)
    assert m.token_logprobs(s).shape == (2, 5) and np.all(m.token_logprobs(s) <= 0)


def test_width_sets_capacity_a_teacher_holds_the_language_a_narrow_student_cannot(lang, teacher):
    t = E.vs_truth(teacher, lang)
    assert t["kl"] < 0.001 and t["rule_acc"] == 1.0 and teacher.n_params == 1891
    s8, s4 = E.vs_truth(fit_language(lang, 8), lang), E.vs_truth(fit_language(lang, 4), lang)
    assert 0.25 < s8["kl"] < 0.4 and s8["rule_acc"] < 0.95        # 0.321 nats, 93.4%: the capacity gap
    assert s4["kl"] > s8["kl"] and s4["rule_acc"] < 0.7


def test_prune_width_keeps_the_most_active_units(lang, teacher):
    C = lang.contexts()
    p = teacher.prune_width(C, 16)
    _, (_, _, h) = teacher.forward(C)
    kept = np.sort(np.argsort(-np.abs(h).mean(0))[:16])
    assert p.H == 16 and p.n_params == TinyLM(11, 16).n_params
    assert np.allclose(p.p["W2"], teacher.p["W2"][kept]) and np.allclose(p.p["E"], teacher.p["E"])


def test_adam_training_lowers_the_loss():
    lang = ModLang(5, 0.2)
    C = lang.contexts()
    P = lang.true_probs(C)
    h = train(TinyLM(5, 8, seed=0), C, lambda z, i: L.soft_ce(z, P[i]), 200)
    assert h[-1] < 0.8 * h[0]
