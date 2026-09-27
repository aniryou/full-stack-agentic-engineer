"""Forward vs reverse KL on a bimodal teacher, the JSD in between, and TV (primer §2)."""
import math

import numpy as np

from distillcore import divergences as D


def test_basics():
    p, q = np.array([0.5, 0.3, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.4])
    assert math.isclose(D.tv(p, q), 0.4) and math.isclose(1 - D.tv(p, q), np.minimum(p, q).sum())
    assert D.kl(p, p) == 0 and D.kl(p, q) > 0 and D.kl(q, p) != D.kl(p, q)
    assert D.jsd(p, q, 0) == D.kl(p, q) and D.jsd(p, q, 1) == D.kl(q, p) and D.jsd(p, q, 0.5) <= math.log(2)
    assert D.kl(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == math.inf


def test_forward_kl_covers_both_modes_and_puts_mass_between_them():
    p = D.bimodal()
    f = D.fit_bump(p, "forward")
    assert f["mu"] == 5.0 and f["s"] == 8.6 and round(f["value"], 4) == 0.6425
    assert round(f["mass_where_teacher_is_empty"], 3) == 0.454 and f["teacher_mass_uncovered"] == 0


def test_reverse_kl_commits_to_one_mode():
    p = D.bimodal()
    r = D.fit_bump(p, "reverse")
    assert r["mu"] == 2.0 and r["s"] == 0.7 and abs(r["value"] - math.log(2)) < 1e-3
    assert round(r["mass_where_teacher_is_empty"], 3) == 0.019 and r["teacher_mass_uncovered"] > 0.5


def test_jsd_flips_from_covering_to_seeking_between_beta_06_and_07():
    p = D.bimodal()
    assert D.fit_bump(p, "jsd", 0.6)["mu"] == 5.0
    assert D.fit_bump(p, "jsd", 0.7)["mu"] == 2.0
