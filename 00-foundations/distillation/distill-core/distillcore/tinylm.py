"""A tiny next-token model family with manual gradients: the teacher is wide, the student narrow.

The one idea: a language model is a function from a context to logits, trained by pushing a gradient on
those logits back through the network. Every distillation method in this package differs only in the
gradient it hands to `backward()` — hard labels, soft targets at a temperature, a divergence on the
student's own samples — so one small model serves all of them. Architecture: embed the last two tokens,
concatenate, one tanh hidden layer of width H, logits over V. Width is the capacity knob: a teacher with
H = 64 holds ModLang's 121-entry rule table; a student with H = 8 cannot, which is the teacher–student gap.
"""
from __future__ import annotations

import numpy as np

from .losses import log_softmax, soft_ce, softmax


class TinyLM:
    def __init__(self, V: int = 11, H: int = 64, d: int = 8, order: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V, self.H, self.d, self.order = V, H, d, order
        self.p = {"E": rng.normal(0, 1.0, (V, d)),
                  "W1": rng.normal(0, 1 / np.sqrt(order * d), (order * d, H)), "b1": np.zeros(H),
                  "W2": rng.normal(0, 1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}

    @property
    def n_params(self) -> int:
        return sum(a.size for a in self.p.values())

    def copy(self) -> "TinyLM":
        m = TinyLM.__new__(TinyLM)
        m.V, m.H, m.d, m.order = self.V, self.H, self.d, self.order
        m.p = {k: a.copy() for k, a in self.p.items()}
        return m

    # -- forward and backward ---------------------------------------------------------------------------
    def forward(self, ctxs):
        """Logits (N, V) for contexts (N, order), plus what backward() needs."""
        c = np.asarray(ctxs, dtype=int).reshape(-1, self.order)
        x = self.p["E"][c].reshape(len(c), -1)
        h = np.tanh(x @ self.p["W1"] + self.p["b1"])
        return h @ self.p["W2"] + self.p["b2"], (c, x, h)

    def logits(self, ctxs) -> np.ndarray:
        return self.forward(ctxs)[0]

    def probs(self, ctxs, T: float = 1.0) -> np.ndarray:
        return softmax(self.logits(ctxs), T)

    def backward(self, cache, dz) -> dict:
        """Gradients of Σ_rows <dz, logits> with respect to every parameter (dz = dLoss/dlogits)."""
        c, x, h = cache
        da = (dz @ self.p["W2"].T) * (1 - h * h)
        dE = np.zeros_like(self.p["E"])
        np.add.at(dE, c, (da @ self.p["W1"].T).reshape(len(c), self.order, self.d))
        return {"E": dE, "W1": x.T @ da, "b1": da.sum(0), "W2": h.T @ dz, "b2": dz.sum(0)}

    # -- sequences ----------------------------------------------------------------------------------------
    def sample(self, prompts, n_new: int, rng, T: float = 1.0) -> np.ndarray:
        """Continue each prompt token by token at temperature T (T = 0: greedy)."""
        seqs = np.array(prompts, dtype=int)
        for _ in range(n_new):
            z = self.logits(seqs[:, -self.order:])
            if T == 0:
                nxt = z.argmax(1)
            else:
                nxt = (softmax(z, T).cumsum(1) > rng.random((len(z), 1))).argmax(1)
            seqs = np.column_stack([seqs, nxt])
        return seqs

    def positions(self, seqs):
        """Every generated position of every sequence as (contexts (N·n, order), targets (N·n,))."""
        s = np.asarray(seqs)
        ctx = np.stack([s[:, t - self.order:t] for t in range(self.order, s.shape[1])], 1)
        return ctx.reshape(-1, self.order), s[:, self.order:].reshape(-1)

    def token_logprobs(self, seqs) -> np.ndarray:
        """log π(y_t | context) for every generated token, shape (N, n_new) — what an engine returns."""
        ctx, y = self.positions(seqs)
        lp = log_softmax(self.logits(ctx))[np.arange(len(y)), y]
        return lp.reshape(len(seqs), -1)

    def grad_logprob(self, seqs, weights) -> dict:
        """∇θ Σ_i Σ_t w_{i,t}·log π(y_{i,t} | context): row (i, t) of dlogits is w·(onehot(y) − q).
        weights: a scalar, one per sequence (N,), or one per token (N, n_new)."""
        seqs = np.asarray(seqs)
        ctx, y = self.positions(seqs)
        z, cache = self.forward(ctx)
        w = np.asarray(weights, float)
        w = w[:, None] if w.ndim == 1 else w
        w = np.broadcast_to(w, (len(seqs), seqs.shape[1] - self.order)).reshape(-1)
        dz = -softmax(z) * w[:, None]
        dz[np.arange(len(y)), y] += w
        return self.backward(cache, dz)

    def prune_width(self, ctxs, keep: int) -> "TinyLM":
        """Minitron-style width pruning: keep the `keep` hidden units with the largest mean |activation| on
        calibration contexts; slice W1, b1 and W2. The result is a narrower model to distil into."""
        _, (_, _, h) = self.forward(ctxs)
        idx = np.sort(np.argsort(-np.abs(h).mean(0))[:keep])
        m = self.copy()
        m.H = keep
        m.p["W1"], m.p["b1"], m.p["W2"] = self.p["W1"][:, idx].copy(), self.p["b1"][idx].copy(), self.p["W2"][idx].copy()
        return m


class Adam:
    """Adam with the usual defaults, over a TinyLM's parameter dict (descent on a loss)."""

    def __init__(self, model: TinyLM, lr: float = 0.01, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps, self.t = model, lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(a) for k, a in model.p.items()}
        self.v = {k: np.zeros_like(a) for k, a in model.p.items()}

    def step(self, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh, vh = self.m[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            self.model.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model: TinyLM, ctxs, loss_fn, steps: int = 300, lr: float = 0.02, batch: int | None = None,
          seed: int = 0) -> list[float]:
    """Minimise loss_fn(logits, idx) -> (loss, dlogits) over contexts ctxs; idx picks the rows of each
    minibatch (all rows when batch is None), so the loss can look up its own targets. Returns the losses."""
    ctxs = np.asarray(ctxs)
    opt, rng, hist = Adam(model, lr), np.random.default_rng(seed), []
    for _ in range(steps):
        idx = np.arange(len(ctxs)) if batch is None else rng.integers(0, len(ctxs), batch)
        z, cache = model.forward(ctxs[idx])
        loss, dz = loss_fn(z, idx)
        opt.step(model.backward(cache, dz))
        hist.append(loss)
    return hist


def fit_language(lang, H: int = 64, steps: int = 1500, lr: float = 0.02, seed: int = 0, d: int = 8) -> TinyLM:
    """A model that has learned `lang`: soft-target cross-entropy to the true next-token distribution over
    every context. With H = 64 this is the teacher (KL to the truth ≈ 0.0004 nats); narrower models fall short."""
    C = lang.contexts()
    P = lang.true_probs(C)
    m = TinyLM(lang.V, H, d, lang.order, seed)
    train(m, C, lambda z, idx: soft_ce(z, P[idx]), steps, lr)
    return m
