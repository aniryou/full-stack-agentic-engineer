"""Distilling reasoning traces on the ThinkTask-style toy (primer §5)."""
import numpy as np

from distillcore import ThinkToy, reasoning as R


def test_fit_is_the_empirical_stopping_rate_with_a_prior():
    p = R.LengthPolicy.fit([0, 1, 1, 2, 2, 2], max_think=4, prior=1.0)
    h = 1 / (1 + np.exp(-p.theta))
    assert np.allclose(h, [1.5 / 7, 2.5 / 6, 3.5 / 4, 0.5])       # (stops + ½) / (reached + 1)
    assert abs(p.length_distribution().sum() - 1) < 1e-12
    big = R.LengthPolicy.fit(np.repeat([0, 1, 2], [100, 200, 300]), max_think=4, prior=1.0)
    assert np.allclose(big.length_distribution()[:3], [1 / 6, 2 / 6, 3 / 6], atol=0.01)   # the prior washes out


def test_the_untrained_student_answers_at_once_half_the_time():
    task = ThinkToy(0.8, 0.1, 32)
    e = R.LengthPolicy(32).expected(task)
    assert abs(e["length"] - 1.0) < 1e-6 and round(e["accuracy"], 3) == 0.273   # 1 − 0.8·0.5/(1 − 0.45)
    g = R.LengthPolicy.geometric(8).expected(task)
    assert abs(g["length"] - 8 * (1 - (8 / 9) ** 32)) < 1e-9       # a geometric thinker, truncated at 32


def test_the_rl_teacher_and_a_student_distilled_from_its_traces(think):
    task, teacher = think
    t = teacher.expected(task)
    assert round(t["accuracy"], 3) == 0.893 and round(t["length"], 1) == 19.9
    L, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
    s = R.distil(L, ok).expected(task)
    assert abs(s["accuracy"] - t["accuracy"]) < 0.01 and abs(s["length"] - t["length"]) < 0.5


def test_filters_trade_accuracy_for_tokens(think):
    task, teacher = think
    L, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
    allt = R.distil(L, ok).expected(task)
    right = R.distil(L, ok, keep="correct").expected(task)
    short = R.distil(L, ok, keep="correct", max_len=16).expected(task)
    assert right["length"] > allt["length"] and right["accuracy"] >= allt["accuracy"]   # correct traces are longer
    assert short["length"] < 13 and short["accuracy"] < 0.8 and short["tokens_per_correct"] < allt["tokens_per_correct"]


def test_traces_beat_rl_at_equal_samples(think):
    task, teacher = think
    L, ok = R.traces(teacher, task, np.random.default_rng(2), 1000)
    distilled = R.distil(L, ok).expected(task)["accuracy"]
    rl = R.reinforce(task, np.random.default_rng(3), 1000).expected(task)["accuracy"]
    assert distilled > 0.88 and rl < 0.5                          # a trace carries the behaviour; a reward, one bit


def test_knowledge_does_not_transfer(think):
    task, teacher = think
    L, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
    s = R.distil(L, ok)
    weak = s.expected(task, q=0.05)
    assert abs(weak["length"] - s.expected(task)["length"]) < 1e-9 and round(weak["accuracy"], 3) == 0.706
    assert task.optimal_length(0.01, q=0.05) > task.optimal_length(0.01)   # a weaker solver should think longer


def test_distil_leaves_the_callers_arrays_alone(think):
    task, teacher = think
    L, ok = R.traces(teacher, task, np.random.default_rng(1), 500)
    before = ok.copy()
    for keep, ml in (("correct", None), ("correct", 12), ("all", 8)):
        R.distil(L, ok, keep=keep, max_len=ml)
    assert np.array_equal(ok, before)
