"""GRPO: G samples per prompt, rewards normalised within the group, no critic — plus the DAPO fixes.

The one idea: the group *is* the baseline. For each prompt sample G completions, score them with a
verifier, and use A_i = (r_i − mean(r)) / (std(r) + 1e-4) for every token of completion i; a group that is all
right or all wrong teaches nothing (A = 0). Each token's loss is PPO's clipped surrogate
−min(ρA, clip(ρ, 1−ε_low, 1+ε_high)·A) with ρ = π/π_old, plus β·k3, the low-variance KL estimator
π_ref/π − log(π_ref/π) − 1. How per-token losses are averaged (`loss_type`) decides whether long completions
are under-weighted — the length bias DAPO and Dr. GRPO remove. The names and defaults follow TRL's
`GRPOConfig` (v1.14.0) so a setting here maps one-to-one onto a real run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import Policy


def group_advantages(rewards, scale: str = "group", eps: float = 1e-4) -> np.ndarray:
    """rewards: (n_groups, G). r − group mean, divided by std + eps unless scale == "none".
    std uses Bessel's correction, as TRL's does; scale == "batch" divides by the std of the whole batch."""
    r = np.atleast_2d(np.asarray(rewards, dtype=float))
    a = r - r.mean(axis=1, keepdims=True)
    if scale == "group":
        a = a / (r.std(axis=1, ddof=1, keepdims=True) + eps)
    elif scale == "batch":
        a = a / (r.std(ddof=1) + eps)
    elif scale != "none":
        raise ValueError(scale)
    return a


def k1(logp, ref_logp):
    """log(π/π_ref): unbiased for KL(π‖π_ref) under samples from π, but noisy and often negative."""
    return np.asarray(logp) - np.asarray(ref_logp)


def k3(logp, ref_logp):
    """π_ref/π − log(π_ref/π) − 1: unbiased, never negative, low variance (Schulman 2020; GRPO, TRL)."""
    d = np.asarray(ref_logp) - np.asarray(logp)
    return np.exp(d) - d - 1.0


def clipped_surrogate(ratio, adv, eps_low: float = 0.2, eps_high: float | None = None):
    """PPO's per-token objective min(ρA, clip(ρ, 1−ε_low, 1+ε_high)·A) and dObjective/dρ.

    The derivative is A where the unclipped term is the minimum and 0 where clipping binds: a token whose
    ratio has already moved past the clip in the direction the advantage wants stops receiving gradient.
    """
    eps_high = eps_low if eps_high is None else eps_high
    ratio, adv = np.asarray(ratio, float), np.asarray(adv, float)
    unclipped, clipped = ratio * adv, np.clip(ratio, 1 - eps_low, 1 + eps_high) * adv
    obj = np.minimum(unclipped, clipped)
    dobj = np.where(unclipped <= clipped, adv, 0.0)
    return obj, dobj


def token_weights(lengths, loss_type: str = "dapo", max_len: int | None = None) -> list[np.ndarray]:
    """The weight each completion's tokens get when per-token losses are averaged into one number.

    "grpo":    mean over each completion's tokens, then over completions: 1 / (G·|o_i|) — long completions'
               tokens count less (the length bias).
    "dapo":    token-level: every token in the batch counts the same, 1 / Σ|o_i| (TRL's default).
    "dr_grpo": a constant, 1 / (G·max_len), as in Dr. GRPO.
    """
    lengths = [int(n) for n in lengths]
    G = len(lengths)
    if loss_type == "grpo":
        return [np.full(n, 1.0 / (G * max(n, 1))) for n in lengths]
    if loss_type == "dapo":
        total = max(sum(lengths), 1)
        return [np.full(n, 1.0 / total) for n in lengths]
    if loss_type == "dr_grpo":
        return [np.full(n, 1.0 / (G * max_len)) for n in lengths]
    raise ValueError(loss_type)


def soft_overlong_penalty(length, max_len: int, cache: int) -> float:
    """DAPO's length-aware penalty: 0 up to max_len − cache, then linear down to −1 at max_len, −1 beyond."""
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return ((max_len - cache) - length) / cache
    return -1.0


@dataclass
class GRPOConfig:
    """The TRL `GRPOConfig` fields that change the algorithm, with TRL v1.14.0's defaults (verify)."""
    num_generations: int = 8           # G
    beta: float = 0.0                  # KL coefficient; 0 means no reference model at all
    epsilon: float = 0.2               # ε_low
    epsilon_high: float | None = None  # None → ε_low; DAPO's clip-higher uses 0.28
    loss_type: str = "dapo"            # "grpo" | "dapo" | "dr_grpo"
    scale_rewards: str = "group"       # "group" | "batch" | "none" (Dr. GRPO)
    num_iterations: int = 1            # μ: gradient steps per batch of rollouts; 1 → ρ ≡ 1, the clip never binds
    mask_truncated_completions: bool = False   # DAPO overlong filtering
    dynamic_sampling: bool = False     # DAPO: resample groups whose rewards are all equal (not a TRL flag)
    overlong_cache: int = 0            # >0: add soft_overlong_penalty(length, max_len, overlong_cache)
    lr: float = 1.0


def grpo_step(policy: Policy, task, rng, cfg: GRPOConfig, prompts=(0,), ref: Policy | None = None,
              max_len: int | None = None) -> dict:
    """One GRPO update: sample G completions for each prompt with π_old, then μ gradient steps.

    Returns statistics over everything generated: mean reward, P(correct), mean length, truncation rate, the
    fraction of groups with zero reward std (they carry no gradient), the groups generated (dynamic sampling
    pays for its full batches in extra rollouts), and the fraction of tokens whose ratio was clipped.
    """
    max_len = max_len or getattr(task, "max_think", None) or getattr(task, "length", None)
    old = policy.copy()
    groups, rewards, seen, generated = [], [], [], 0
    queue = list(prompts)
    while queue and generated < 4 * len(prompts):      # dynamic sampling over-samples, at most 4x
        p = queue.pop(0)
        trajs = old.sample(task, rng, cfg.num_generations, prompt=p)
        generated += 1
        r = np.array([t.reward for t in trajs], dtype=float)
        if cfg.overlong_cache:
            r = r + np.array([soft_overlong_penalty(t.length, max_len, cfg.overlong_cache) for t in trajs])
        seen.append((trajs, r))
        if cfg.dynamic_sampling and r.std() == 0:
            queue.append(int(rng.integers(task.n_prompts)))   # uninformative: draw another prompt instead
            continue
        groups.append(trajs)
        rewards.append(r)
    flat = [t for g, _ in seen for t in g]
    stats = {"reward": float(np.mean([t.reward for t in flat])),
             "correct": float(np.mean([t.info["correct"] for t in flat])),
             "length": float(np.mean([t.info["length"] for t in flat])),
             "truncated": float(np.mean([t.info["truncated"] for t in flat])),
             "frac_reward_zero_std": float(np.mean([r.std() == 0 for _, r in seen])),
             "groups_generated": generated}
    if not groups:
        stats["clipped"] = 0.0
        return stats
    A = group_advantages(np.array(rewards), cfg.scale_rewards)
    batch = [(t, A[i, j]) for i, g in enumerate(groups) for j, t in enumerate(g)
             if not (cfg.mask_truncated_completions and t.info["truncated"])]
    if not batch:
        stats["clipped"] = 0.0
        return stats
    weights = token_weights([t.length for t, _ in batch], cfg.loss_type, max_len)
    old_lp = [old.token_logprobs(t) for t, _ in batch]
    ref_lp = [ref.token_logprobs(t) for t, _ in batch] if (ref is not None and cfg.beta > 0) else None
    clipped = total = 0
    for _ in range(cfg.num_iterations):
        grad = np.zeros_like(policy.theta)
        for k, ((t, a), w) in enumerate(zip(batch, weights)):
            lp = policy.token_logprobs(t)
            ratio = np.exp(lp - old_lp[k])
            _, dobj = clipped_surrogate(ratio, a, cfg.epsilon, cfg.epsilon_high)
            coef = dobj * ratio                       # d(ρA)/dθ = A·ρ·∇log π where unclipped
            if ref_lp is not None:                    # d(β·k3)/dθ = β(1 − π_ref/π)·∇log π, subtracted (it is a loss)
                coef = coef - cfg.beta * (1.0 - np.exp(ref_lp[k] - lp))
            grad += policy.grad_logprob(t, coef * w)
            clipped += int(((dobj == 0) & (a != 0)).sum())
            total += t.length
        policy.step(grad, cfg.lr)
    stats["clipped"] = clipped / max(total, 1)
    return stats


def train_grpo(policy: Policy, task, rng, cfg: GRPOConfig, steps: int = 200, prompts=(0,),
               ref: Policy | None = None, log_every: int = 10, batch_prompts: int | None = None) -> list[dict]:
    """Repeated grpo_step on `prompts`, or on `batch_prompts` prompts drawn from the task's dataset each step;
    returns the logged statistics with their step numbers."""
    hist = []
    for step in range(steps):
        ps = [int(x) for x in rng.integers(task.n_prompts, size=batch_prompts)] if batch_prompts else prompts
        s = grpo_step(policy, task, rng, cfg, ps, ref)
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, **s})
    return hist


def expected_update(policy: Policy, task, rng, loss_type: str, G: int = 8, n_groups: int = 1000,
                    scale: str = "group", prompt: int = 0) -> np.ndarray:
    """The average single-step GRPO gradient (μ = 1, β = 0) over many sampled groups at a fixed policy —
    its direction is what the aggregation choice biases."""
    g = np.zeros_like(policy.theta)
    for _ in range(n_groups):
        trajs = policy.sample(task, rng, G, prompt=prompt)
        adv = group_advantages([[t.reward for t in trajs]], scale)[0]
        for t, a, w in zip(trajs, adv, token_weights([t.length for t in trajs], loss_type, getattr(task, "max_think", None))):
            g += policy.grad_logprob(t, a * w)
    return g / n_groups
