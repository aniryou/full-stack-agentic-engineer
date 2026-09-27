"""Sequence-level distillation: the data pipeline's bookkeeping and exposure bias (primer §3)."""
import numpy as np

from distillcore import TinyLM, onpolicy as op, seqkd
from tests.pins import near


def test_pipeline_bookkeeping_greedy_samples_are_duplicates(lang, teacher, bench):
    prompts = bench["prompts"]
    _, s0 = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=0.0)
    assert s0 == {"generated": 160, "verified": 160, "unique": 20, "tokens_paid": 1920, "tokens_used": 240}


def test_pipeline_bookkeeping_sampled_data_mostly_fails_the_verifier(lang, teacher, bench):
    """The reference run keeps 16 of 160 samples, 11 of them unique; another CPU's kernel samples a slightly
    different teacher and lands within a few of that (tests/pins.py)."""
    data, s1 = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=1.0)
    assert s1["generated"] == 160 and s1["tokens_paid"] == 1920 and s1["tokens_used"] == 12 * s1["unique"]
    near(s1["verified"], 16, 4)
    near(s1["unique"], 11, 4)
    assert s1["verified"] <= 0.15 * s1["generated"]                # most of what the teacher wrote is thrown away
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


def test_exposure_bias_compounds_per_output_not_per_position(lang, teacher, bench):
    """Per position the KD student settles near 0.78 after a few tokens; the chance its whole output is right so far
    falls every position (0.650 at 4 and 0.190 at 12 in the reference run; other CPUs' kernels land within 0.03 and
    0.04 of that, tests/pins.py). The teacher stays right, and samples the rule's ~0.8 throughout."""
    eb = seqkd.exposure_bias(bench["kd"], lang, bench["greedy"], bench["prompts"], np.random.default_rng(5))
    own, allr, onrule = eb["own_prefixes"], eb["all_right_so_far"], eb["sampled_on_rule"]
    assert np.ptp(own[3:]) < 0.05                                  # flat per position after the first slips
    assert np.all(np.diff(allr) < 0)                               # compounding per output
    near(allr[3], 0.650, 0.03)
    near(allr[11], 0.190, 0.04)
    assert np.allclose(allr[0], own[0]) and np.all(allr <= own + 1e-12)
    t = seqkd.exposure_bias(teacher, lang, bench["greedy"], bench["prompts"], np.random.default_rng(5))
    assert np.all(t["all_right_so_far"] == 1.0) and np.all(np.abs(t["sampled_on_rule"] - 0.8) < 0.03)
    assert onrule[3:].max() < t["sampled_on_rule"].min() - 0.1     # the student slips more often than the teacher


def test_on_policy_distillation_removes_it(lang, teacher, bench):
    """From 0.788 / 0.774 on its own prefixes (positions 4 / 12) to 0.994 / 0.984 in the reference run, and never
    below 0.986 / 0.948 on the kernels measured (tests/pins.py)."""
    s = bench["kd"].copy()
    op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=1.0, beta=0.0, data=bench["greedy"], seed=1)
    own = seqkd.own_accuracy(s, lang, bench["prompts"], 12, np.random.default_rng(5))
    assert own[3] > 0.96 and own[11] > 0.90
