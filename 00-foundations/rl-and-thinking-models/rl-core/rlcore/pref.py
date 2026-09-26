"""Learning from preferences: Bradley–Terry reward models, PPO's pieces in brief, and DPO.

The one idea: a preference "A over B" is a noisy comparison of rewards, P(A ≻ B) = σ(r(A) − r(B))
(Bradley–Terry), so a reward model is a logistic regression on pairs. Maximising E[r] − β·KL(π‖π_ref) has the
closed-form optimum π* ∝ π_ref·exp(r/β); invert it and r = β·log(π*/π_ref) + const. Substitute that into the
Bradley–Terry likelihood and the constant cancels: DPO trains the policy directly on pairs with the loss
−log σ(β[(log π(y⁺) − log π_ref(y⁺)) − (log π(y⁻) − log π_ref(y⁻))]). No reward model, no sampling — and so
nothing is learned outside the pairs' support. An annotator who likes long answers teaches the reward model
to like length, and a policy optimised against it pads (`length_biased_prefs`).
"""
from __future__ import annotations

import numpy as np

from .policy import Policy, softmax


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def bt_prob(r_a, r_b):
    """Bradley–Terry: P(a ≻ b) = σ(r_a − r_b). Only differences matter (add a constant to every r: no change)."""
    return sigmoid(np.asarray(r_a) - np.asarray(r_b))


def fit_bradley_terry(x_chosen, x_rejected, l2: float = 1e-3, steps: int = 500, lr: float = 1.0) -> np.ndarray:
    """Fit a linear reward r(y) = w·x(y) by maximising Σ log σ(w·(x⁺ − x⁻)) − l2·|w|².

    With one-hot features this is a per-response reward table (centre it: BT cannot see a shift)."""
    d = np.asarray(x_chosen, float) - np.asarray(x_rejected, float)
    w = np.zeros(d.shape[1])
    for _ in range(steps):
        p = sigmoid(d @ w)
        w += lr * ((d * (1 - p)[:, None]).mean(axis=0) - 2 * l2 * w)
    return w


def bt_nll(w, x_chosen, x_rejected) -> float:
    """Mean −log σ(r⁺ − r⁻): the reward model's training loss (ln 2 = 0.693 at chance)."""
    d = np.asarray(x_chosen, float) - np.asarray(x_rejected, float)
    return float(np.mean(np.log1p(np.exp(-(d @ w)))))


# -- DPO -------------------------------------------------------------------------------------------------
def dpo_loss(chosen_logratio, rejected_logratio, beta: float = 0.1):
    """TRL's "sigmoid" loss: −log σ(β·(log π/π_ref(y⁺) − log π/π_ref(y⁻))); log-probs summed over tokens."""
    m = beta * (np.asarray(chosen_logratio, float) - np.asarray(rejected_logratio, float))
    return np.log1p(np.exp(-m))


def ipo_loss(chosen_logratio, rejected_logratio, beta: float = 0.1):
    """IPO: a squared loss toward a fixed margin 1/(2β) instead of the logistic, so the margin cannot run
    to infinity on deterministic preferences (TRL averages the log-ratios per token for this loss)."""
    return (np.asarray(chosen_logratio, float) - np.asarray(rejected_logratio, float) - 1 / (2 * beta)) ** 2


def implicit_reward(policy: Policy, ref: Policy, task, seq, beta: float) -> float:
    """DPO's implicit reward β·log(π(y)/π_ref(y)) — TRL logs it as rewards/chosen and rewards/rejected."""
    return beta * (policy.seq_logprob(task, seq) - ref.seq_logprob(task, seq))


def encode_pairs(task, pairs) -> dict:
    """Flatten (chosen, rejected) sequences into token arrays once, so each DPO step is a few numpy calls."""
    out = {}
    for key, idx in (("chosen", 0), ("rejected", 1)):
        trajs = [task.as_trajectory(pair[idx]) for pair in pairs]
        out[key] = (np.concatenate([t.states for t in trajs]), np.concatenate([t.actions for t in trajs]),
                    np.repeat(np.arange(len(trajs)), [t.length for t in trajs]))
    out["n"] = len(pairs)
    return out


def _seq_logps(theta, enc, n):
    logp = np.log(softmax(theta))
    S, A, seg = enc
    return np.bincount(seg, logp[S, A], minlength=n)


def dpo_step(policy: Policy, ref: Policy, data: dict, beta: float = 0.1, lr: float = 1.0) -> dict:
    """One gradient step on the mean DPO loss over pairs encoded by `encode_pairs`.

    ∂loss/∂θ = −β·σ(−m)·(∇log π(y⁺) − ∇log π(y⁻)), m = β(Δ⁺ − Δ⁻), Δ = log π(y) − log π_ref(y): pairs the
    policy already ranks confidently (large m) contribute little; mis-ranked ones contribute up to β.
    Returns TRL's metric names: loss, rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins.
    """
    n, P = data["n"], policy.probs()
    dc = _seq_logps(policy.theta, data["chosen"], n) - _seq_logps(ref.theta, data["chosen"], n)
    dr = _seq_logps(policy.theta, data["rejected"], n) - _seq_logps(ref.theta, data["rejected"], n)
    m = beta * (dc - dr)
    coef = beta * sigmoid(-m) / n                 # ascent weight on log π(y⁺); the same, negated, on y⁻
    grad = np.zeros_like(policy.theta)
    for key, sign in (("chosen", 1.0), ("rejected", -1.0)):
        S, A, seg = data[key]
        c = sign * coef[seg]
        np.add.at(grad, (S, A), c)                # Σ c·onehot(a) − c·π(·|s): the closed-form softmax gradient
        np.add.at(grad, S, -c[:, None] * P[S])
    policy.step(grad, lr)
    rc, rr = beta * dc, beta * dr
    return {"loss": float(np.mean(np.log1p(np.exp(-m)))), "rewards/chosen": float(rc.mean()),
            "rewards/rejected": float(rr.mean()), "rewards/accuracies": float(np.mean(rc > rr)),
            "rewards/margins": float(np.mean(rc - rr))}


# -- PPO's value side, in brief --------------------------------------------------------------------------
def gae(rewards, values, gamma: float = 1.0, lam: float = 0.95) -> np.ndarray:
    """Generalised advantage estimation over one completion's tokens.

    δ_t = r_t + γ·V(s_{t+1}) − V(s_t) (V after the last token = 0); A_t = δ_t + γλ·A_{t+1}.
    λ = 1 gives Monte-Carlo returns minus V (unbiased, noisy); λ = 0 the one-step TD error (biased, quiet).
    RLHF puts the reward-model score on the last token and −β·log(π/π_ref) on every token.
    """
    r, v = np.asarray(rewards, float), np.asarray(values, float)
    v_next = np.append(v[1:], 0.0)
    delta = r + gamma * v_next - v
    adv, run = np.zeros_like(r), 0.0
    for t in range(len(r) - 1, -1, -1):
        run = delta[t] + gamma * lam * run
        adv[t] = run
    return adv


# -- a length-biased annotator ---------------------------------------------------------------------------
def response_catalogue(n_contents: int = 40, paddings=(0, 100, 200, 400, 800), base_len: int = 150,
                       padding_cost: float = 0.3, seed: int = 0) -> dict:
    """Candidate answers to one prompt: each of n_contents answers (content quality c ~ N(0, 1)) comes with
    extra filler tokens p ∈ paddings. True quality q = c − padding_cost·p/100 (filler hurts a little);
    length = base_len + p. Returns arrays c, p, quality, length."""
    rng = np.random.default_rng(seed)
    c = np.repeat(rng.standard_normal(n_contents), len(paddings))
    p = np.tile(np.asarray(paddings, float), n_contents)
    return {"content": c, "padding": p, "quality": c - padding_cost * p / 100, "length": base_len + p}


def length_biased_prefs(cat: dict, n_pairs: int, length_weight: float, ref_probs=None, seed: int = 0):
    """Pairs drawn from π_ref, labelled by an annotator with P(a ≻ b) = σ(Δquality + length_weight·Δlength/100).
    Returns index arrays (chosen, rejected)."""
    rng = np.random.default_rng(seed)
    n = len(cat["quality"])
    probs = np.full(n, 1 / n) if ref_probs is None else np.asarray(ref_probs, float)
    a, b = rng.choice(n, n_pairs, p=probs), rng.choice(n, n_pairs, p=probs)
    logit = (cat["quality"][a] - cat["quality"][b]) + length_weight * (cat["length"][a] - cat["length"][b]) / 100
    a_wins = rng.random(n_pairs) < sigmoid(logit)
    return np.where(a_wins, a, b), np.where(a_wins, b, a)
