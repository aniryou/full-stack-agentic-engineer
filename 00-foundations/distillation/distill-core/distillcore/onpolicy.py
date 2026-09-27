"""On-policy distillation: the student samples, the teacher scores every token, the student moves.

The one idea: train on the prefixes the student will actually produce. GKD samples a continuation from the
student (with probability λ; otherwise it uses a dataset or teacher sequence), asks the teacher for its full
next-token distribution at every position, and minimises a divergence there — reverse KL (β = 1), forward KL
(β = 0) or the JSD in between (`losses.gkd`). Seen as RL, reverse KL is a dense reward: every token earns
r_t = log π_T(y_t | y_<t) − log π_S(y_t | y_<t), where GRPO's verifier gives one number per sequence, and
the same REINFORCE machinery applies — exactly `rlcore.pg.reinforce_grad` with the teacher as the
reference, β = 1 and no task reward (`pg_grad`). The exact expectation is checked by enumeration
(`exact_seq_kl`), so the estimator and the loss can be compared number for number.
"""
from __future__ import annotations

import numpy as np

from .losses import gkd, log_softmax
from .tasks import ModLang
from .tinylm import Adam, TinyLM


def gkd_train(student: TinyLM, teacher: TinyLM, prompts, n_new: int, steps: int = 300, lam: float = 1.0,
              beta: float = 1.0, data=None, batch: int = 64, lr: float = 0.02, T_sample: float = 1.0,
              seed: int = 0) -> list[float]:
    """TRL's GKD loop: each step, with probability λ the student samples the continuations (on-policy),
    otherwise a batch of `data` sequences is used (teacher-written: supervised KD); then the JSD(β) between
    the teacher's and the student's full distributions at every generated position is minimised."""
    rng, opt, hist = np.random.default_rng(seed), Adam(student, lr), []
    prompts = np.asarray(prompts)
    for _ in range(steps):
        if data is None or rng.random() < lam:
            seqs = student.sample(prompts[rng.integers(0, len(prompts), batch)], n_new, rng, T_sample)
        else:
            seqs = data[rng.integers(0, len(data), batch)]
        ctx, _ = student.positions(seqs)
        z, cache = student.forward(ctx)
        loss, dz = gkd(z, teacher.probs(ctx), beta)
        opt.step(student.backward(cache, dz))
        hist.append(loss)
    return hist


def token_rewards(student: TinyLM, teacher: TinyLM, seqs) -> np.ndarray:
    """r_t = log π_T(y_t | ·) − log π_S(y_t | ·) for every generated token: the dense reward. (N, n_new)"""
    return teacher.token_logprobs(seqs) - student.token_logprobs(seqs)


def advantages(student_logps, teacher_logps, baseline: str = "mean", per_token: bool = False) -> np.ndarray:
    """Per-token weights on ∇ log π_S, already divided by the batch size N as rlcore averages.

    Sequence form (default): every token of sequence i gets A_i = R_i − mean(R), R_i = Σ_t r_t — what
    `rlcore.pg.reinforce_grad(policy, trajs, "mean", ref=teacher, beta=1.0)` computes with zero task reward.
    Per-token form: token t gets r_t alone (Tinker's `kl_discount_factor=0`): lower variance, but it ignores
    how a token changes the states after it, so it is biased for the sequence KL.
    """
    r = np.asarray(teacher_logps, float) - np.asarray(student_logps, float)
    if per_token:
        return r / len(r)
    R = r.sum(1)
    A = R - R.mean() if baseline == "mean" else R
    return np.repeat((A / len(r))[:, None], r.shape[1], 1)


def pg_grad(student: TinyLM, teacher: TinyLM, seqs, baseline: str = "mean", per_token: bool = False) -> dict:
    """REINFORCE estimate of −∇ KL(π_S ‖ π_T) from the student's own samples — an ascent direction."""
    w = advantages(student.token_logprobs(seqs), teacher.token_logprobs(seqs), baseline, per_token)
    return student.grad_logprob(seqs, w)


def exact_seq_kl(student: TinyLM, teacher: TinyLM, prompt, n_new: int) -> tuple[float, np.ndarray, np.ndarray]:
    """KL(π_S ‖ π_T) over every continuation of one prompt, by enumeration: (KL, π_S(y), all sequences)."""
    seqs = ModLang(student.V).continuations(prompt, n_new)
    ls, lt = student.token_logprobs(seqs).sum(1), teacher.token_logprobs(seqs).sum(1)
    ps = np.exp(ls)
    return float(ps @ (ls - lt)), ps, seqs


def exact_pg_grad(student: TinyLM, teacher: TinyLM, prompt, n_new: int, per_token: bool = False) -> dict:
    """The estimator's expectation, computed over every continuation: E_y[R(y)·∇ log π_S(y)], which equals
    −∇ KL(π_S ‖ π_T); with per_token=True, E_y[Σ_t r_t·∇ log π_S(y_t | ·)], which does not."""
    _, ps, seqs = exact_seq_kl(student, teacher, prompt, n_new)
    r = token_rewards(student, teacher, seqs)
    return student.grad_logprob(seqs, ps[:, None] * r if per_token else ps * r.sum(1))


def token_pg_identity(p, v) -> tuple[np.ndarray, np.ndarray]:
    """At one position: Σ_y q(y)·(log q(y) − log p(y))·(onehot(y) − q), the expected score-function
    estimate, and the analytic gradient of KL(q ‖ p) on the logits v. They are equal: the sampled per-token
    reward and GKD's β = 1 loss are the same gradient in expectation."""
    lq = log_softmax(v)[0] if np.ndim(v) == 2 else log_softmax(v)
    q, lp = np.exp(lq), np.log(np.asarray(p, float).reshape(-1))
    est = sum(q[y] * (lq[y] - lp[y]) * (np.eye(len(q))[y] - q) for y in range(len(q)))
    return est, gkd(np.atleast_2d(v), np.atleast_2d(p), beta=1.0)[1][0]


def flops_per_prompt(student_params: float, teacher_params: float, tokens: int, samples: int = 1,
                     teacher_scores: bool = True, reference_params: float = 0.0) -> dict:
    """One prompt's compute: the student generates `samples` completions of `tokens` (2·N_S per token, and
    decode-bound in practice), trains on them (6·N_S), and the teacher scores every token in one
    prefill-shaped forward pass (2·N_T) — no teacher generation. GRPO: samples = G, a verifier instead of the
    teacher, and, with a KL penalty, a reference model's forward pass (`reference_params`, 2·N_ref per token).
    Per token that is 8·N_S + 2·N_T against GRPO's 8·N_S (+ 2·N_ref): a big teacher makes each on-policy step
    dearer, so its saving has to come from needing fewer steps."""
    gen, trn = 2 * student_params * tokens * samples, 6 * student_params * tokens * samples
    score = 2 * teacher_params * tokens * samples if teacher_scores else 0.0
    ref = 2 * reference_params * tokens * samples
    total = gen + trn + score + ref
    return {"generate": gen, "train": trn, "score": score, "reference": ref, "total": total,
            "per_token": total / (tokens * samples)}
