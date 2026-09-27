"""losses.py — the distillation losses in torch, shared by the tiny transformers and the Hugging Face trainer.

One idea: every logit-level distillation loss compares two next-token distributions at the same positions
and differs only in *which* divergence and *whose* samples. This file holds the three the lab trains with,
written so their definitions can be read against PRIMER §2 "Soft targets, temperature and the choice of
divergence" and §4 "On-policy distillation":

    kd_loss            Hinton's loss: α · T² · KL(p_T^teacher ‖ q_T^student) + (1 − α) · CE(labels, q_1)
                       (the gradient of the soft term on the student's logits is T · (q_T − p_T); the T² keeps
                       it the same size as T grows — ModelOpt's ``LogitsDistillationLoss`` multiplies by T²)
    generalized_jsd    GKD's divergence with TRL's convention (``trl.experimental.gkd``, TRL 1.14.0):
                       β = 0 → KL(teacher ‖ student) (forward, mode-covering), β = 1 → KL(student ‖ teacher)
                       (reverse, mode-seeking), in between β·KL(p‖m) + (1−β)·KL(q‖m), m = β·p + (1−β)·q;
                       TRL computes it at T = 1 (its ``temperature`` only sets sampling) and has no T²
    token_rewards      the on-policy view: r_t = log π_teacher(y_t | y_<t) − log π_student(y_t | y_<t) on the
                       student's own sample — the dense per-token reward whose policy gradient is −∇ reverse KL

All three take ``(N, V)`` logits (rows = positions that count) or ``(B, T, V)`` with a mask. ``chunk`` rows at
a time bounds the memory of the float32 softmaxes: at a 151,936-token vocabulary one fp32 row block of 4,096
positions is 2.49 GB, which is why TRL's trainers chunk and why a T4 OOMs without it (PRIMER §10).
Importing this module imports torch; the rest of ``distillab`` never needs it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rows(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """(B, T, V) + (B, T) mask → (N, V) rows that count; (N, V) passes through."""
    if logits.dim() == 2:
        return logits if mask is None else logits[mask.bool()]
    flat = logits.reshape(-1, logits.shape[-1])
    return flat if mask is None else flat[mask.reshape(-1).bool()]


def _f(x: torch.Tensor) -> torch.Tensor:
    """float32 at least (16-bit logits are upcast); float64 stays float64."""
    return x if x.dtype == torch.float64 else x.float()


def soft_kl_rows(teacher: torch.Tensor, student: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-row KL(softmax(teacher / T) ‖ softmax(student / T)) in nats, in float32 (float64 if given)."""
    lp = F.log_softmax(_f(teacher) / temperature, dim=-1)
    lq = F.log_softmax(_f(student) / temperature, dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor | None = None,
            mask: torch.Tensor | None = None, temperature: float = 2.0, alpha: float = 0.5,
            chunk: int | None = None) -> torch.Tensor:
    """α · T² · mean KL(p_T ‖ q_T) + (1 − α) · mean CE(labels, softmax(student)) over the rows in ``mask``.

    ``alpha`` is the weight on the soft (teacher) term: 1.0 = pure distillation, 0.0 = plain SFT.
    ``labels`` (same leading shape as the logits, ints) are needed when ``alpha < 1``. The teacher's logits
    carry no gradient. ``chunk`` computes the soft term ``chunk`` rows at a time (same value, less memory)."""
    s, t = _rows(student_logits, mask), _rows(teacher_logits.detach(), mask)
    n = max(1, s.shape[0])
    soft = s.new_zeros((), dtype=_f(s).dtype)
    if alpha > 0:
        step = chunk or s.shape[0] or 1
        for i in range(0, s.shape[0], step):
            soft = soft + soft_kl_rows(t[i:i + step], s[i:i + step], temperature).sum()
        soft = soft / n * temperature ** 2
    if alpha >= 1:
        return soft
    if labels is None:
        raise ValueError("alpha < 1 mixes in the hard-label loss: pass labels")
    y = labels.reshape(-1) if labels.dim() > 1 else labels
    if mask is not None and y.shape[0] != s.shape[0]:
        y = y[mask.reshape(-1).bool()]
    hard = F.cross_entropy(_f(s), y, reduction="mean")
    return alpha * soft + (1 - alpha) * hard


def generalized_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, beta: float = 0.5,
                    mask: torch.Tensor | None = None, temperature: float = 1.0) -> torch.Tensor:
    """GKD's generalised Jensen-Shannon divergence, TRL 1.14.0's convention, summed over the vocabulary and
    averaged over rows: β = 0 → KL(teacher ‖ student); β = 1 → KL(student ‖ teacher); else
    β·KL(teacher ‖ m) + (1 − β)·KL(student ‖ m) with m = β·p_teacher + (1 − β)·q_student."""
    s = F.log_softmax(_f(_rows(student_logits, mask)) / temperature, dim=-1)
    t = F.log_softmax(_f(_rows(teacher_logits.detach(), mask)) / temperature, dim=-1)
    if beta == 0:
        per = (t.exp() * (t - s)).sum(-1)
    elif beta == 1:
        per = (s.exp() * (s - t)).sum(-1)
    else:
        m = torch.logsumexp(torch.stack([s + math.log(1 - beta), t + math.log(beta)]), dim=0)
        per = beta * (t.exp() * (t - m)).sum(-1) + (1 - beta) * (s.exp() * (s - m)).sum(-1)
    return per.mean() if per.numel() else per.sum()


def token_rewards(student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor) -> torch.Tensor:
    """r_t = log π_teacher(y_t) − log π_student(y_t) for the tokens the student sampled. Their sum over a
    sequence is minus that sequence's log-ratio, whose expectation is −KL(student ‖ teacher) over sequences."""
    return teacher_logprobs - student_logprobs
