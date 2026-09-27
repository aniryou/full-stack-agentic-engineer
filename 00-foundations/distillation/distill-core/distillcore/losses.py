"""Distillation losses and their gradients on the student's logits, in closed form.

The one idea: every distillation loss is a divergence between the teacher's distribution p and the
student's q = softmax(v / T), and its gradient on the student's logits is small and exact. Soft-target
cross-entropy and KL(p_T ‖ q_T) both give (q_T − p_T) / T; Hinton's loss multiplies the KL by T² so that
gradient stays O(1) as T grows, and as T → ∞ it becomes matching centred logits (MSE). The hard label is
the same loss with p one-hot — which is why a soft target is worth more per example: a sampled label adds
1 − Σp² of noise the teacher's distribution does not (`label_noise`). `gkd` is TRL's generalised JSD(β):
β = 0 is forward KL(teacher ‖ student), β = 1 reverse KL(student ‖ teacher).

Every function takes logits of shape (N, V), averages over the N rows, and returns (loss, dlogits).
"""
from __future__ import annotations

import numpy as np


def log_softmax(z, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def softmax(z, T: float = 1.0) -> np.ndarray:
    return np.exp(log_softmax(z, T))


def kl(p, q) -> np.ndarray:
    """KL(p ‖ q) per row, in nats, with 0·log 0 = 0."""
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0).sum(-1)


def hard_ce(v, labels):
    """Cross-entropy with one-hot targets: −log q(y); gradient q − onehot(y)."""
    lq, n = log_softmax(v), len(labels)
    g = np.exp(lq)
    g[np.arange(n), labels] -= 1
    return float(-lq[np.arange(n), labels].mean()), g / n


def soft_ce(v, p, T: float = 1.0):
    """−Σ p log q_T with q_T = softmax(v / T); gradient (q_T − p) / T."""
    lq = log_softmax(v, T)
    return float(-(p * lq).sum(-1).mean()), (np.exp(lq) - p) / (T * len(p))


def kd(v, z, T: float = 1.0, scale: bool = True):
    """Hinton's soft term: T²·KL(p_T ‖ q_T), p_T = softmax(z / T) from the teacher's logits z.
    Gradient T·(q_T − p_T); without the T² (`scale=False`) it is (q_T − p_T)/T and fades as 1/T²."""
    p, lq = softmax(z, T), log_softmax(v, T)
    s = T * T if scale else 1.0
    return float(s * kl(p, np.exp(lq)).mean()), s * (np.exp(lq) - p) / (T * len(p))


def hinton(v, z, labels, T: float = 2.0, alpha: float = 0.5):
    """α·T²·KL(p_T ‖ q_T) + (1 − α)·CE(y, q_1): the classic distillation loss."""
    ls, gs = kd(v, z, T)
    lh, gh = hard_ce(v, labels)
    return alpha * ls + (1 - alpha) * lh, alpha * gs + (1 - alpha) * gh


def logit_mse(v, z):
    """(1/2V)·‖(v − v̄) − (z − z̄)‖²: what T²·KL becomes as T → ∞ (logit matching on centred logits)."""
    v, z = np.asarray(v, float), np.asarray(z, float)
    d = (v - v.mean(-1, keepdims=True)) - (z - z.mean(-1, keepdims=True))
    V = v.shape[-1]
    return float((d ** 2).sum(-1).mean() / (2 * V)), d / (V * len(d))


def label_noise(p) -> np.ndarray:
    """E‖onehot(y) − p‖² for y ~ p: the gradient noise a sampled hard label adds per example, 1 − Σp²."""
    p = np.atleast_2d(p)
    return 1.0 - (p ** 2).sum(-1)


def gkd(v, p, beta: float = 0.5):
    """TRL's `generalized_jsd_loss` at T = 1 between teacher probabilities p and student logits v.

    β = 0: KL(p ‖ q) (forward, mode-covering); β = 1: KL(q ‖ p) (reverse, mode-seeking); otherwise
    β·KL(p ‖ m) + (1 − β)·KL(q ‖ m) with m = β·p + (1 − β)·q. The gradient comes from ∂/∂q_j, pushed
    through the softmax: ∂/∂v_i = q_i·(g_i − Σ_j q_j g_j), with g = log(q/p) + 1 for reverse KL and
    g = (1 − β)·log(q/m) for the JSD (the "+1" cancels in the softmax projection).
    """
    p = np.atleast_2d(p)
    q = softmax(v)
    n = len(q)
    if beta == 0:
        return float(kl(p, q).mean()), (q - p) / n
    with np.errstate(divide="ignore"):
        lp = np.log(p)
    if beta == 1:
        g = np.log(q) - lp
        loss = kl(q, p)
    else:
        m = beta * p + (1 - beta) * q
        g = (1 - beta) * (np.log(q) - np.log(m))
        loss = beta * kl(p, m) + (1 - beta) * kl(q, m)
    return float(loss.mean()), q * (g - (q * g).sum(-1, keepdims=True)) / n
