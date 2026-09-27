"""Measuring a student: agreement with the teacher, accuracy on the task, and where the gap hides.

The one idea: agreement and accuracy answer different questions. Teacher–student agreement — mean
KL(p_teacher ‖ p_student), top-1 agreement, top-k overlap — needs no labels and uses the same definitions as
`quantcore.eval` (a quantized model is a student too). Task accuracy is what users feel, and it needs an
interval: Wilson's, as `memcore.harness.wilson_interval` and the 07 platform lab compute it. Report both
per slice — common against rare inputs, short against long outputs — because a student's gap hides in the
tail; and remember a student can beat its teacher on the task while agreeing with it less.
"""
from __future__ import annotations

import math

import numpy as np

from .losses import log_softmax


def kl(ref_logits, test_logits) -> float:
    """Mean KL(p_ref ‖ p_test) in nats per position (quantcore.eval.kl)."""
    lp, lq = log_softmax(ref_logits), log_softmax(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def argmax_agreement(ref_logits, test_logits) -> float:
    """Fraction of positions whose top-1 token is the same (quantcore.granularity.argmax_agreement)."""
    return float((np.argmax(ref_logits, -1) == np.argmax(test_logits, -1)).mean())


def topk_overlap(ref_logits, test_logits, k: int = 3) -> float:
    """Mean |top-k(ref) ∩ top-k(test)| / k: agreement on the plausible set, not only the favourite."""
    a = np.argsort(-np.asarray(ref_logits), -1)[..., :k]
    b = np.argsort(-np.asarray(test_logits), -1)[..., :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a.reshape(-1, k), b.reshape(-1, k))]))


def vs_truth(model, lang) -> dict:
    """A toy luxury: score a model against the true language over every context — mean KL(truth ‖ model)
    and rule accuracy (its top token is the main rule's)."""
    C = lang.contexts()
    P, lq = lang.true_probs(C), log_softmax(model.logits(C))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(P > 0, P * (np.log(P) - lq), 0.0).sum(1)
    wrong = lq.argmax(1) != lang.main(C)
    q = np.exp(lq)
    right_q = q[np.arange(len(C)), lang.main(C)]
    return {"kl": float(k.mean()), "rule_acc": float((~wrong).mean()),
            # where it is wrong: how sure it is of its wrong token, and how much it gives the right one
            "wrong_top_q": float(q.max(1)[wrong].mean()) if wrong.any() else 0.0,
            "wrong_right_q": float(right_q[wrong].mean()) if wrong.any() else 0.0}


def accuracy_stderr(p: float, n: int) -> float:
    """lm-eval's standard error for a 0/1 metric: sqrt(p(1 − p)/(n − 1)) (quantcore.eval.accuracy_stderr)."""
    return math.sqrt(p * (1 - p) / (n - 1))


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval: sane at 0/n and n/n, where p ± 1.96·SE is not."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = passes / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_z(lost: int, gained: int) -> float:
    """McNemar on the same items: z = (gained − lost) / sqrt(gained + lost) (quantcore.eval.paired_z)."""
    return 0.0 if lost + gained == 0 else (gained - lost) / math.sqrt(lost + gained)


def compare(teacher_correct, student_correct) -> dict:
    """Accuracy both ways on the same items, the flips behind the difference, and intervals."""
    t, s = np.asarray(teacher_correct, bool), np.asarray(student_correct, bool)
    n, lost, gained = len(t), int((t & ~s).sum()), int((~t & s).sum())
    return {"n": n, "teacher": float(t.mean()), "student": float(s.mean()), "gap": float(t.mean() - s.mean()),
            "teacher_ci": wilson_interval(int(t.sum()), n), "student_ci": wilson_interval(int(s.sum()), n),
            "lost": lost, "gained": gained, "paired_z": paired_z(lost, gained)}


def capability_gap(slices: dict) -> list[dict]:
    """One row per slice: {name: (teacher_correct, student_correct)} → compare() rows, largest gap first."""
    rows = [{"slice": name, **compare(t, s)} for name, (t, s) in slices.items()]
    return sorted(rows, key=lambda r: -r["gap"])
