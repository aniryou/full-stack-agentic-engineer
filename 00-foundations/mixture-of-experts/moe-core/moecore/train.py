"""A tiny MoE trained with hand-written gradients: collapse without balancing, specialisation with it.

The one idea: the router is trained by the same loss as the experts, and that loss rewards
sending a token to whichever expert is *already* good -- so the expert that wins early keeps
winning, and an expert that gets no tokens never trains and never gets a chance. Here hidden
states share a large common direction (as transformer hidden states do), so at initialisation one
or two experts top most tokens. Without a balancing term the router locks that in or narrows it
further: one to three of four experts carry every token and the rest are dead weight. The Switch
aux loss, or a DeepSeek-style selection bias, spreads the tokens; the experts specialise one
cluster each and the loss drops.

Task: tokens come from C clusters; cluster c's target is its own linear map of the cluster part,
y = (x - offset) A_c. One linear expert cannot fit C maps; E >= C experts can, if the router
sends each cluster to its own expert. Model: linear experts (hidden=0) or one-hidden-layer ReLU
experts, a linear router, top-k with Switch-style weights -- the chosen experts' softmax
probabilities, not renormalised, so the router gets a gradient from the task loss even at k = 1.

Gradients (per step, T tokens, r_t = (y_hat_t - y_t) / T):
  y_hat_t        = sum_j p[t, e_j] * f_{e_j}(x_t)
  dL/dp[t, e_j]  = <r_t, f_{e_j}(x_t)>                (task loss, chosen experts only)
                 + alpha * E * f_e / T                 (aux loss, every expert; routing.switch_aux_grad)
  dL/dlogits     = p * (g - sum_e g_e p_e)             (softmax backward, row by row)
  expert e sees only its own tokens, each scaled by its weight p[t, e].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .moe import softmax, topk
from .routing import load, switch_aux_grad, switch_aux_loss, update_bias


@dataclass
class Task:
    x: np.ndarray        # [N, d_in]
    y: np.ndarray        # [N, d_out]
    cluster: np.ndarray  # [N] which map produced y


def make_task(n: int = 512, clusters: int = 4, d_in: int = 8, d_out: int = 4, common: float = 6.0,
              spread: float = 1.0, seed: int = 0) -> Task:
    """Clustered inputs plus a shared offset `common` along one direction (anisotropic hidden
    states); cluster c's target is its own linear map of the cluster part, y = (x - offset) A_c."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((clusters, d_in)) * spread
    maps = rng.standard_normal((clusters, d_in, d_out)) / np.sqrt(d_in)
    offset = np.zeros(d_in)
    offset[0] = common
    c = rng.integers(clusters, size=n)
    local = centres[c] + 0.3 * rng.standard_normal((n, d_in))
    return Task(local + offset, np.einsum("ni,nio->no", local, maps[c]), c)


@dataclass
class History:
    loss: list = field(default_factory=list)      # task loss (MSE / 2) per step
    aux: list = field(default_factory=list)       # Switch aux loss (hf convention) per step
    share: list = field(default_factory=list)     # [E] share of assignments per step
    placement: np.ndarray | None = None           # [C, E] tokens of cluster c sent to expert e, at the end

    @property
    def final_share(self) -> np.ndarray:
        return self.share[-1]


class Adam:
    """Adam, in place. (Plain SGD also collapses without balancing, but converges too slowly on this
    badly scaled toy to show what balance buys.)"""
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.n = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]

    def step(self, grads):
        self.n += 1
        for q, g, m, v in zip(self.params, grads, self.m, self.v):
            m += (1 - self.b1) * (g - m)
            v += (1 - self.b2) * (g * g - v)
            q -= self.lr * (m / (1 - self.b1 ** self.n)) / (np.sqrt(v / (1 - self.b2 ** self.n)) + self.eps)


def loss_and_grads(x, y, w_r, w1, w2, k: int = 1, alpha: float = 0.0, bias=None):
    """Forward and hand-written backward for one full batch. Returns (task loss, aux loss, probs,
    chosen idx, [dW_router, dW1, dW2]) for L = task + alpha * aux. w2 empty (size 0) = linear experts."""
    t, e, hidden = x.shape[0], w1.shape[0], w2.size > 0
    p = softmax(x @ w_r)
    idx = topk(p if bias is None else p + bias, k)   # a bias picks; the weights below never see it
    w = np.take_along_axis(p, idx, axis=1)
    out, parts = np.zeros_like(y), []
    for j in range(e):
        tok, slot = np.nonzero(idx == j)                 # expert j sees only its own tokens
        h = x[tok] @ w1[j]
        f = np.maximum(h, 0) @ w2[j] if hidden else h
        out[tok] += w[tok, slot][:, None] * f
        parts.append((j, tok, slot, h, f))
    r = (out - y) / t                                    # dL/dy_hat for L = mean 0.5 ||y_hat - y||^2
    g, d1, d2 = np.zeros_like(p), np.zeros_like(w1), np.zeros_like(w2)
    for j, tok, slot, h, f in parts:
        g[tok, j] = np.einsum("no,no->n", r[tok], f)     # dL/dp for the chosen experts
        dy = w[tok, slot][:, None] * r[tok]
        if hidden:
            d2[j] = np.maximum(h, 0).T @ dy
            d1[j] = x[tok].T @ ((dy @ w2[j].T) * (h > 0))
        else:
            d1[j] = x[tok].T @ dy
    if alpha:
        g = g + alpha * switch_aux_grad(p, idx)
    dlogits = p * (g - (g * p).sum(axis=1, keepdims=True))   # softmax backward
    task = float(0.5 * np.sum((out - y) ** 2) / t)
    return task, switch_aux_loss(p, idx), p, idx, [x.T @ dlogits, d1, d2]


def train(task: Task, n_experts: int = 4, k: int = 1, balance: str = "none", steps: int = 300,
          lr: float = 0.02, alpha: float = 0.1, bias_rate: float = 0.01, hidden: int = 0,
          seed: int = 1) -> History:
    """Full-batch Adam. balance: "none" | "aux" (alpha x Switch loss) | "bias" (DeepSeek-style)."""
    rng = np.random.default_rng(seed)
    x, y = task.x, task.y
    d_in, d_out, e = x.shape[1], y.shape[1], n_experts
    w_r = rng.standard_normal((d_in, e)) * 0.1
    w1 = rng.standard_normal((e, d_in, hidden or d_out)) * (1 / np.sqrt(d_in) if hidden else 0.01)
    w2 = rng.standard_normal((e, hidden, d_out)) * 0.01 if hidden else np.zeros((e, 0, 0))
    opt, bias, hist = Adam([w_r, w1, w2], lr), np.zeros(e), History()
    for _ in range(steps):
        loss, aux, p, idx, grads = loss_and_grads(x, y, w_r, w1, w2, k, alpha if balance == "aux" else 0.0,
                                                  bias if balance == "bias" else None)
        opt.step(grads)
        counts = load(idx, e)
        if balance == "bias":
            bias = update_bias(bias, counts, bias_rate)
        hist.loss.append(loss)
        hist.aux.append(aux)
        hist.share.append(counts / counts.sum())
    clusters = int(task.cluster.max()) + 1
    hist.placement = np.zeros((clusters, e), dtype=int)
    np.add.at(hist.placement, (np.repeat(task.cluster, k), idx.reshape(-1)), 1)
    return hist
