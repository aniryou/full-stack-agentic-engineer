"""Which divergence the student minimises decides what it does when it cannot copy the teacher.

The one idea: a student too small to represent the teacher must choose what to get wrong. Forward
KL(p ‖ q) — the teacher's view, what SFT and classic KD minimise — charges the student wherever the teacher
has mass and it has none, so it spreads to cover every mode and, when it is unimodal, puts mass *between*
them: samples the teacher would never produce. Reverse KL(q ‖ p) — the student's view, what on-policy
distillation minimises — charges it for mass where the teacher has none, so it commits to the modes it can
fit and drops the rest. The generalised JSD(β) interpolates. The acceptance rate of a draft, Σ min(p, q),
is 1 − TV(p, q) (§7). Worked here with a bimodal teacher and a one-bump student family on 11 tokens.
"""
from __future__ import annotations

import numpy as np


def kl(p, q) -> float:
    """KL(p ‖ q) in nats, 0·log 0 = 0; infinite if q = 0 where p > 0."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    with np.errstate(divide="ignore"):
        return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


def jsd(p, q, beta: float = 0.5) -> float:
    """Generalised JSD with TRL's convention (p = teacher, q = student): β·KL(p ‖ m) + (1 − β)·KL(q ‖ m),
    m = β·p + (1 − β)·q; β = 0 is taken as KL(p ‖ q), β = 1 as KL(q ‖ p), as `generalized_jsd_loss` does."""
    if beta == 0:
        return kl(p, q)
    if beta == 1:
        return kl(q, p)
    m = beta * np.asarray(p, float) + (1 - beta) * np.asarray(q, float)
    return beta * kl(p, m) + (1 - beta) * kl(q, m)


def tv(p, q) -> float:
    """Total variation ½·Σ|p − q|. A draft q is accepted against a target p with probability 1 − TV."""
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def bump(mu: float, s: float, n: int = 11) -> np.ndarray:
    """A discretised Gaussian on tokens 0..n−1: the one-mode student family."""
    x = np.arange(n)
    w = np.exp(-0.5 * ((x - mu) / s) ** 2)
    return w / w.sum()


def bimodal(modes=(2, 8), s: float = 0.7, n: int = 11, weights=(0.5, 0.5)) -> np.ndarray:
    """A two-mode teacher: a mixture of bumps (a prompt with two good continuations)."""
    return sum(w * bump(m, s, n) for w, m in zip(weights, modes))


def fit_bump(p, divergence: str = "forward", beta: float = 0.5, mus=None, ss=None) -> dict:
    """Best one-bump student for teacher p under a divergence, by grid search over (μ, s).

    divergence: "forward" KL(p ‖ q), "reverse" KL(q ‖ p), "jsd" (with β). Returns the fit and what it
    means for generation: the student's mass on the teacher's near-empty tokens (p < 0.01) — where
    forward KL's compromise puts it — and the teacher's mass the student leaves uncovered.
    """
    n = len(p)
    mus = np.round(np.linspace(0, n - 1, 201), 3) if mus is None else mus
    ss = np.round(np.arange(0.3, 12.01, 0.1), 2) if ss is None else ss
    div = {"forward": lambda q: kl(p, q), "reverse": lambda q: kl(q, p), "jsd": lambda q: jsd(p, q, beta)}[divergence]
    best = min(((div(bump(m, s, n)), m, s) for m in mus for s in ss), key=lambda t: round(t[0], 9))   # ties: lowest μ
    q = bump(best[1], best[2], n)
    low = np.asarray(p) < 0.01
    return {"mu": float(best[1]), "s": float(best[2]), "value": float(best[0]), "q": q,
            "mass_where_teacher_is_empty": float(q[low].sum()),
            "teacher_mass_uncovered": float(np.asarray(p)[q < 0.01].sum()), "tv": tv(p, q)}
