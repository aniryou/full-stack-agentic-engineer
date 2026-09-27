"""Sequence-level distillation: the data pipeline's bookkeeping and exposure bias (primer §3)."""
import numpy as np

from distillcore import TinyLM, onpolicy as op, seqkd


def test_pipeline_bookkeeping_greedy_samples_are_duplicates(lang, teacher, bench):
    prompts = bench["prompts"]
    _, s0 = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=0.0)
    assert s0 == {"generated": 160, "verified": 160, "unique": 20, "tokens_paid": 1920, "tokens_used": 240}


def test_pipeline_bookkeeping_sampled_data_mostly_fails_the_verifier(lang, teacher, bench):
    data, s1 = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=1.0)
    assert s1["generated"] == 160 and s1["verified"] == 16 and s1["unique"] == 11 and s1["tokens_used"] == 132
    assert lang.verify(data).all() and len(np.unique(data, axis=0)) == len(data)
    assert abs(0.8 ** 12 - 0.0687) < 1e-4                          # a perfect teacher passes 6.9% of 12-token samples


def test_sft_on_greedy_data_collapses_the_students_diversity(lang, teacher, bench):
    s = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(s, bench["greedy"], 400)
    ctx, _ = s.positions(bench["greedy"])
    ent = lambda m: float(-(m.probs(ctx) * np.log(m.probs(ctx))).sum(1).mean())
    assert ent(s) < 0.02 and ent(teacher) > 0.6
    assert seqkd.own_accuracy(s, lang, bench["prompts"], 12, np.random.default_rng(5))[-1] > 0.99   # never slips


def test_exposure_bias_teacher_prefixes_vs_own_prefixes(lang, bench):
    eb = seqkd.exposure_bias(bench["kd"], lang, bench["greedy"], bench["prompts"], np.random.default_rng(5))
    assert np.all(eb["teacher_prefixes"] == 1.0)
    own = eb["own_prefixes"]
    assert own[0] == 1.0 and own[3] < 0.85 and own[11] < 0.85 and eb["gap_last"] > 0.15


def test_on_policy_distillation_removes_it(lang, teacher, bench):
    s = bench["kd"].copy()
    op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=1.0, beta=0.0, data=bench["greedy"], seed=1)
    own = seqkd.own_accuracy(s, lang, bench["prompts"], 12, np.random.default_rng(5))
    assert own[3] > 0.98 and own[11] > 0.97
