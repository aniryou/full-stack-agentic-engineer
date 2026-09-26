"""A policy is a table of softmaxes, π(a | s) = softmax(θ[s])_a — a language model with the network removed.

The one idea: for a softmax the gradient of a log-probability has a closed form,
    ∂ log π(a | s) / ∂θ[s, b] = 1[a = b] − π(b | s),
and a completion's log-probability is the sum of its tokens' (TRL sums them the same way). Everything RL
does to a model — REINFORCE, PPO, DPO, GRPO — is a weighted sum of these per-token gradients; the methods
differ only in the weights. `copy()` makes the explicit reference and old-policy copies those methods keep
beside the one being trained.
"""
from __future__ import annotations

import numpy as np

from .tasks import Trajectory


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class Policy:
    def __init__(self, n_states: int, n_actions: int, theta: np.ndarray | None = None):
        self.theta = np.zeros((n_states, n_actions)) if theta is None else np.array(theta, dtype=float)

    @classmethod
    def for_task(cls, task, theta=None) -> "Policy":
        return cls(task.n_states, task.n_actions, theta)

    def copy(self) -> "Policy":
        """A frozen snapshot: the reference model (π_ref) or the rollout policy (π_old)."""
        return Policy(*self.theta.shape, self.theta.copy())

    def probs(self) -> np.ndarray:
        return softmax(self.theta)

    # -- sampling -------------------------------------------------------------------------------------
    def sample(self, task, rng, n: int = 1, prompt: int = 0) -> list[Trajectory]:
        """n completions for one prompt, token by token, each scored by the task's verifier."""
        P, out = self.probs(), []
        for _ in range(n):
            states, actions = [], []
            while not task.done(prompt, actions):
                s = task.state(prompt, actions)
                a = min(int(np.searchsorted(np.cumsum(P[s]), rng.random(), side="right")), P.shape[1] - 1)
                states.append(s)
                actions.append(a)
            reward, info = task.score(prompt, actions, rng)
            out.append(Trajectory(prompt, states, actions, reward, info))
        return out

    # -- log-probabilities and their gradients --------------------------------------------------------
    def token_logprobs(self, traj: Trajectory) -> np.ndarray:
        """log π(a_t | s_t) for every token of a completion (what an engine returns as logprobs)."""
        logp = np.log(self.probs())
        return logp[traj.states, traj.actions]

    def seq_logprob(self, task, actions, prompt: int = 0) -> float:
        """log π(y) = Σ_t log π(a_t | s_t), recomputing the states from the tokens."""
        logp = np.log(self.probs())
        return float(sum(logp[task.state(prompt, actions[:t]), a] for t, a in enumerate(actions)))

    def grad_logprob(self, traj: Trajectory, weights=None) -> np.ndarray:
        """∇θ Σ_t w_t log π(a_t | s_t), in closed form: row s_t gets w_t · (onehot(a_t) − π(· | s_t))."""
        P = self.probs()
        w = np.ones(traj.length) if weights is None else np.broadcast_to(np.asarray(weights, float), (traj.length,))
        g = np.zeros_like(self.theta)
        rows = -P[traj.states] * w[:, None]
        rows[np.arange(traj.length), traj.actions] += w
        np.add.at(g, traj.states, rows)
        return g

    # -- whole-space views for small tasks ------------------------------------------------------------
    def sequence_probs(self, task, seqs=None, prompt: int = 0) -> np.ndarray:
        """π(y) for every sequence of a SeqTask (or the ones given): exact, by enumeration."""
        seqs = task.all_sequences() if seqs is None else seqs
        return np.exp([self.seq_logprob(task, s, prompt) for s in seqs])

    def stop_probs(self, task, prompt: int = 0) -> np.ndarray:
        """ThinkTask: P(answer now | t thinking tokens so far), t = 0..max_think−1."""
        base = prompt * (task.max_think + 1)
        return self.probs()[base: base + task.max_think, 1]

    def step(self, grad: np.ndarray, lr: float) -> None:
        """Gradient ascent on the objective whose gradient is `grad`."""
        self.theta += lr * grad
