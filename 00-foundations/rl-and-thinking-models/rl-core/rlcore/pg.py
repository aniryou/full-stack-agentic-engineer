"""Policy gradients over token sequences: sample, score, reweight.

The one idea: ∇ E_π[R] = E_π[(R − b)·∇log π(y)] for any baseline b that does not depend on y (REINFORCE):
push up completions that scored above the baseline, push down the rest. The baseline changes the variance,
never the expectation. A KL penalty to a frozen reference makes the objective E[R] − β·KL(π ‖ π_ref), whose
optimum is π* ∝ π_ref·exp(R/β) (`kl_optimal()`): RL reweights what the reference already does.
"""
from __future__ import annotations

import numpy as np

from .policy import Policy


def advantages(rewards, baseline: str = "mean") -> np.ndarray:
    """R − b for one batch. "none": b = 0; "mean": b = batch mean; "loo": b = mean of the *other* samples
    (leave-one-out, unbiased for every sample — RLOO)."""
    r = np.asarray(rewards, dtype=float)
    if baseline == "none":
        return r.copy()
    if baseline == "mean":
        return r - r.mean()
    if baseline == "loo":
        return r - (r.sum() - r) / (len(r) - 1)
    raise ValueError(baseline)


def reinforce_grad(policy: Policy, trajs, baseline: str = "mean", ref: Policy | None = None,
                   beta: float = 0.0, entropy_coef: float = 0.0) -> np.ndarray:
    """The REINFORCE estimate of ∇ (E[R] − β·KL(π‖π_ref) + entropy bonus), averaged over the batch.

    The KL penalty enters as reward shaping, R − β(log π(y) − log π_ref(y)) — the PPO-RLHF way — which is
    exactly the gradient of −β·KL because E[∇ log π] = 0.
    """
    r = np.array([t.reward for t in trajs], dtype=float)
    if ref is not None and beta > 0:
        r = r - beta * np.array([policy.token_logprobs(t).sum() - ref.token_logprobs(t).sum() for t in trajs])
    if entropy_coef > 0:                    # entropy bonus: −log π(y) is an unbiased sample of the entropy
        r = r - entropy_coef * np.array([policy.token_logprobs(t).sum() for t in trajs])
    A = advantages(r, baseline)
    return sum(a * policy.grad_logprob(t) for a, t in zip(A, trajs)) / len(trajs)


def grad_variance(policy: Policy, task, rng, baseline: str, batch: int = 8, trials: int = 200,
                  offset: float = 0.0) -> float:
    """Total variance (sum over parameters) of the batch gradient estimate — what a baseline reduces.
    `offset` adds a constant to every reward: no information, but without a baseline it is all noise."""
    gs = []
    for _ in range(trials):
        trajs = policy.sample(task, rng, batch)
        for t in trajs:
            t.reward += offset
        gs.append(reinforce_grad(policy, trajs, baseline).ravel())
    return float(np.array(gs).var(axis=0).sum())


def train_reinforce(policy: Policy, task, rng, steps: int = 200, batch: int = 16, lr: float = 1.0,
                    baseline: str = "mean", ref: Policy | None = None, beta: float = 0.0, log_every: int = 10,
                    entropy_coef: float = 0.0):
    """Plain on-policy REINFORCE; returns a history of (step, mean reward, P(correct), mean length)."""
    hist = []
    for step in range(steps):
        trajs = policy.sample(task, rng, batch, prompt=step % task.n_prompts)
        if step % log_every == 0 or step == steps - 1:
            hist.append((step, np.mean([t.reward for t in trajs]), np.mean([t.info["correct"] for t in trajs]),
                         np.mean([t.info["length"] for t in trajs])))
        policy.step(reinforce_grad(policy, trajs, baseline, ref, beta, entropy_coef), lr)
    return hist


def sft_step(policy: Policy, trajs, lr: float = 1.0) -> float:
    """Supervised fine-tuning: one ascent step on the mean log-likelihood of the given completions.
    Returns the mean negative log-likelihood (per completion) before the step."""
    nll = -np.mean([policy.token_logprobs(t).sum() for t in trajs])
    policy.step(sum(policy.grad_logprob(t) for t in trajs) / len(trajs), lr)
    return float(nll)


# -- exact views for SeqTask (the whole sequence space is enumerable) ---------------------------------
def expected(policy: Policy, task, fn) -> float:
    """E_π[fn(y)] by enumerating every sequence."""
    seqs = task.all_sequences()
    return float(np.dot(policy.sequence_probs(task, seqs), [fn(s) for s in seqs]))


def kl_seq(p: Policy, q: Policy, task) -> float:
    """KL(π_p ‖ π_q) over whole sequences, exactly."""
    seqs = task.all_sequences()
    lp, lq = np.log(p.sequence_probs(task, seqs)), np.log(q.sequence_probs(task, seqs))
    return float(np.dot(np.exp(lp), lp - lq))


def kl_optimal(ref_probs, rewards, beta: float):
    """The maximiser of E[R] − β·KL(π ‖ π_ref) over all distributions: π* = π_ref·exp(R/β) / Z.

    Returns (π*, E_π*[R], KL(π*‖π_ref)); the KL is E_π*[R]/β − log Z, because log(π*/π_ref) = R/β − log Z.
    """
    ref_probs, rewards = np.asarray(ref_probs, float), np.asarray(rewards, float)
    w = ref_probs * np.exp((rewards - rewards.max()) / beta)
    Z = w.sum()
    pi = w / Z
    er = float(np.dot(pi, rewards))
    log_z = np.log(Z) + rewards.max() / beta
    return pi, er, float(er / beta - log_z)
