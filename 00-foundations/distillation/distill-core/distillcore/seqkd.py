"""Sequence-level distillation: generate with the teacher, filter, and fine-tune the student on the text.

The one idea: SeqKD is ordinary SFT whose data the teacher wrote. It needs only samples from the teacher —
no logits — which is why it works through any API, and it is how the R1 distills were made. The pipeline
is generate (n samples per prompt at a temperature) → verify → deduplicate → SFT, and every stage has a
price: tokens the verifier throws away were still paid for, and samples at T = 0 are duplicates. Its flaw is
exposure bias: the student learns on the teacher's prefixes and decodes on its own, so one early slip puts it
in a context no training example covered, and errors compound with length (`exposure_bias`).
"""
from __future__ import annotations

import numpy as np

from .losses import hard_ce
from .tinylm import TinyLM, train


def teacher_data(teacher: TinyLM, prompts, n_per_prompt: int, n_new: int, rng, T: float = 1.0) -> np.ndarray:
    """n samples per prompt from the teacher at temperature T (T = 0: greedy, every sample identical)."""
    return teacher.sample(np.repeat(np.asarray(prompts), n_per_prompt, axis=0), n_new, rng, T)


def keep_verified(lang, seqs) -> np.ndarray:
    """Rejection sampling: keep the samples the verifier accepts."""
    return seqs[lang.verify(seqs)]


def dedupe(seqs) -> np.ndarray:
    """Drop exact duplicates (order preserved)."""
    _, first = np.unique(seqs, axis=0, return_index=True)
    return seqs[np.sort(first)]


def pipeline(teacher: TinyLM, lang, prompts, n_per_prompt: int, n_new: int, rng, T: float = 1.0,
             verify: bool = True) -> tuple[np.ndarray, dict]:
    """generate → verify → dedupe, with the bookkeeping: tokens generated (paid) and tokens kept (used)."""
    raw = teacher_data(teacher, prompts, n_per_prompt, n_new, rng, T)
    kept = keep_verified(lang, raw) if verify else raw
    uniq = dedupe(kept)
    stats = {"generated": len(raw), "verified": len(kept), "unique": len(uniq),
             "tokens_paid": raw.size - raw.shape[0] * lang.order, "tokens_used": uniq.size - uniq.shape[0] * lang.order}
    return uniq, stats


def sft(student: TinyLM, seqs, steps: int = 400, lr: float = 0.02, batch: int | None = 256, seed: int = 0):
    """Maximum likelihood on the teacher's tokens: hard-label cross-entropy at every generated position."""
    ctx, y = student.positions(seqs)
    return train(student, ctx, lambda z, idx: hard_ce(z, y[idx]), steps, lr, batch, seed)


def tf_accuracy(model: TinyLM, lang, seqs) -> np.ndarray:
    """Teacher forcing: per position, is the model's greedy token right given the prefix in `seqs`?"""
    ctx, _ = model.positions(seqs)
    ok = model.logits(ctx).argmax(1) == lang.main(ctx)
    return ok.reshape(len(seqs), -1).mean(0)


def own_accuracy(model: TinyLM, lang, prompts, n_new: int, rng, T: float = 1.0, n: int = 2000) -> np.ndarray:
    """Free running: the model samples its own prefix at temperature T; per position, is its greedy token
    right given that prefix? The same question as `tf_accuracy`, asked on the prefixes it will really see."""
    p = np.asarray(prompts)
    seqs = model.sample(p[rng.integers(0, len(p), n)], n_new, rng, T)
    return tf_accuracy(model, lang, seqs)


def exposure_bias(model: TinyLM, lang, teacher_seqs, prompts, rng, T: float = 1.0) -> dict:
    """Per-position accuracy on teacher prefixes against the model's own prefixes, and the gap at the end."""
    n_new = teacher_seqs.shape[1] - lang.order
    tf, own = tf_accuracy(model, lang, teacher_seqs), own_accuracy(model, lang, prompts, n_new, rng, T)
    return {"teacher_prefixes": tf, "own_prefixes": own, "gap_last": float(tf[-1] - own[-1])}
