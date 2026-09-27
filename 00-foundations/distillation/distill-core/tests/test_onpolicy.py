"""On-policy distillation as policy gradient: the per-token and sequence identities, checked exactly (primer §4)."""
import numpy as np

from distillcore import ModLang, TinyLM, onpolicy as op
from distillcore.tinylm import fit_language


def test_per_token_reward_equals_the_reverse_kl_gradient_in_expectation():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    for v in (np.array([0.2, -0.1, 0.4, 1.0]), np.zeros(4), np.array([3.0, 0.0, -2.0, 1.0])):
        est, analytic = op.token_pg_identity(p, v)
        assert np.allclose(est, analytic, atol=1e-12)


def _world():
    lang = ModLang(5, 0.2)
    return fit_language(lang, 16, steps=500), TinyLM(5, 4, 3, seed=1), np.array([1, 2]), 3


def _directional(g, dirs):
    return sum(float((g[k] * dirs[k]).sum()) for k in g)


def test_sequence_reinforce_expectation_is_minus_grad_kl_by_enumeration():
    teacher, student, prompt, n = _world()
    kl, ps, seqs = op.exact_seq_kl(student, teacher, prompt, n)
    assert len(seqs) == 125 and abs(ps.sum() - 1) < 1e-12 and kl > 0
    dirs = {k: np.random.default_rng(0).normal(size=a.shape) for k, a in student.p.items()}
    sp, sm = student.copy(), student.copy()
    for k in dirs:
        sp.p[k] += 1e-5 * dirs[k]
        sm.p[k] -= 1e-5 * dirs[k]
    fd = -(op.exact_seq_kl(sp, teacher, prompt, n)[0] - op.exact_seq_kl(sm, teacher, prompt, n)[0]) / 2e-5
    assert abs(_directional(op.exact_pg_grad(student, teacher, prompt, n), dirs) - fd) < 1e-6


def _flat(g):
    return np.concatenate([g[k].ravel() for k in sorted(g)])


def test_sampled_estimate_is_unbiased_and_the_per_token_form_is_biased_but_quieter():
    teacher, student, prompt, n = _world()
    exact_seq = _flat(op.exact_pg_grad(student, teacher, prompt, n))
    exact_tok = _flat(op.exact_pg_grad(student, teacher, prompt, n, per_token=True))
    bias = np.linalg.norm(exact_tok - exact_seq) / np.linalg.norm(exact_seq)
    assert 0.25 < bias < 0.4                                       # 31% of the gradient's norm: ignores later states
    rng, seq, tok = np.random.default_rng(0), [], []
    for _ in range(200):
        s = student.sample(np.tile(prompt, (64, 1)), n, rng)
        seq.append(_flat(op.pg_grad(student, teacher, s)))
        tok.append(_flat(op.pg_grad(student, teacher, s, per_token=True)))
    seq, tok = np.array(seq), np.array(tok)
    assert np.linalg.norm(seq.mean(0) - exact_seq) / np.linalg.norm(exact_seq) < 0.15   # MC noise only
    assert tok.var(0).sum() < 0.7 * seq.var(0).sum()               # 1.86 vs 3.15: lower variance


def test_advantages_follow_the_batch_mean_baseline_and_the_1_over_n_average():
    s_lp = np.log(np.array([[0.5, 0.5], [0.2, 0.9], [0.4, 0.4]]))
    t_lp = np.log(np.array([[0.5, 0.5], [0.4, 0.9], [0.1, 0.4]]))
    R = (t_lp - s_lp).sum(1)                                       # 0, ln 2, ln 1/4
    w = op.advantages(s_lp, t_lp)
    assert np.allclose(w[:, 0], (R - R.mean()) / 3) and np.allclose(w[:, 0], w[:, 1])
    assert np.allclose(op.advantages(s_lp, t_lp, per_token=True), (t_lp - s_lp) / 3)


def test_on_policy_compute_per_prompt():
    g = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False)     # GRPO, G = 16
    d = op.flops_per_prompt(8e9, 32e9, 4096, samples=4)                            # 4 samples, teacher scores
    assert g["total"] == 16 * 8 * 8e9 * 4096 and d["score"] == 4 * 2 * 32e9 * 4096 and g["total"] / d["total"] == 2.0
