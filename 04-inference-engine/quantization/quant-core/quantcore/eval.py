"""eval.py - measure the accuracy you pay against the model you had, with error bars.

The one idea: compare distributions before strings. Mean KL(p_ref || p_quant) and top-1 agreement over
many positions see damage that a task score averages away; the task score (here, the toy task's
accuracy) is what a user feels, and it moves in both directions - quantization flips some right answers
wrong and some wrong answers right, so the net change understates the churn. A difference smaller than
about two standard errors is noise: lm-eval reports stderr = sample stddev / sqrt(n), which for an
accuracy p over n items is sqrt(p (1 - p) / (n - 1)) - 250 GSM8K items at 76.8% carry +-2.7 points.
"""
from __future__ import annotations

import numpy as np

from .granularity import argmax_agreement


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, float)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kl(ref_logits, test_logits) -> float:
    """Mean KL(p_ref || p_test) in nats per position."""
    lp, lq = log_softmax(ref_logits), log_softmax(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def perplexity(logits, labels) -> float:
    """exp(mean negative log-likelihood of the labels)."""
    lp = log_softmax(logits)
    return float(np.exp(-lp[np.arange(len(labels)), labels].mean()))


def accuracy_stderr(p: float, n: int) -> float:
    """lm-eval's `mean_stderr` for a 0/1 metric: sample stddev / sqrt(n) = sqrt(p (1 - p) / (n - 1))."""
    return float(np.sqrt(p * (1 - p) / (n - 1)))


def compare(ref_logits, test_logits, labels=None) -> dict:
    """Distribution distance and, with labels, task accuracy both ways plus the flips behind the delta."""
    out = {"kl": kl(ref_logits, test_logits), "top1": argmax_agreement(ref_logits, test_logits)}
    if labels is not None:
        r, t = np.argmax(ref_logits, -1) == labels, np.argmax(test_logits, -1) == labels
        n = len(labels)
        out.update(acc_ref=float(r.mean()), acc=float(t.mean()), lost=int((r & ~t).sum()),
                   gained=int((~r & t).sum()), stderr=accuracy_stderr(float(t.mean()), n),
                   ppl=perplexity(test_logits, labels))
    return out


def within_budget(report: dict, max_kl: float = 0.01, max_drop: float | None = None) -> bool:
    """A budget stated before measuring: KL below `max_kl` and an accuracy drop no larger than
    `max_drop` (default: two standard errors, the smallest drop this eval can distinguish from noise)."""
    drop = report["acc_ref"] - report["acc"]
    limit = 2 * report["stderr"] if max_drop is None else max_drop
    return report["kl"] <= max_kl and drop <= limit
