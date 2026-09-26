"""tinymodel.py - a small model whose activations have outlier channels, so calibration matters.

The one idea: quantization damage depends on the activations the weights meet, so the test model
needs LLM-like activations. TinyModel is a residual MLP stack: each block applies RMSNorm with a gain
that is 25-40x larger on a few channels (the "massive activations" of real LLMs come from exactly
such gains), then an up-projection, ReLU and a down-projection added back to the residual stream; a
final RMSNorm and a classifier head read the result. The task is synthetic but real: classify noisy
points drawn around 16 class prototypes. The hidden weights are seeded random features; the head is
fitted in closed form (ridge regression, then a temperature), so the model is deterministic, needs no download and builds
in milliseconds. The four hidden linears are what gets quantized; the head, like an LLM's LM head,
stays in high precision unless you say otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def rmsnorm(x, gain=None, eps: float = 1e-6):
    y = x / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps)
    return y if gain is None else y * gain


@dataclass
class TinyModel:
    d: int = 64
    hidden: int = 256
    n_blocks: int = 2
    n_classes: int = 16
    noise: float = 1.6
    seed: int = 0
    weights: dict = field(default_factory=dict)     # name -> (out, in) array; gains stored beside them

    def __post_init__(self):
        if self.weights:
            return
        rng = np.random.default_rng(self.seed)
        self.prototypes = rng.standard_normal((self.n_classes, self.d))
        for b in range(self.n_blocks):
            gain = np.ones(self.d)
            gain[rng.choice(self.d, 4, replace=False)] = rng.uniform(25, 40, 4)   # outlier channels
            self.weights[f"blocks.{b}.norm"] = gain
            self.weights[f"blocks.{b}.up"] = rng.standard_normal((self.hidden, self.d)) / np.sqrt(self.d)
            self.weights[f"blocks.{b}.down"] = rng.standard_normal((self.d, self.hidden)) / np.sqrt(self.hidden)
        X, y = self.sample(4096, split="train")
        F = rmsnorm(self.forward(X, features=True))
        Y = np.eye(self.n_classes)[y]
        head = np.linalg.solve(F.T @ F + 1.0 * np.eye(self.d), F.T @ Y).T       # ridge, closed form
        taus = 2.0 ** np.arange(0, 8, 0.25)                                        # then a temperature, so
        nll = [-log_softmax(F @ head.T * t)[np.arange(len(y)), y].mean() for t in taus]   # the probabilities
        self.weights["head"] = head * taus[int(np.argmin(nll))]                    # (and KL) mean something

    def sample(self, n: int, split: str = "test"):
        """n labelled inputs; the split seeds the noise, so train, calib and test never share points."""
        rng = np.random.default_rng([self.seed, {"train": 1, "calib": 2, "test": 3}[split]])
        y = rng.integers(0, self.n_classes, n)
        return self.prototypes[y] + self.noise * rng.standard_normal((n, self.d)), y

    def linears(self) -> list[str]:
        """The layers a weight-quantization recipe targets (targets=['Linear'], ignore=['head'])."""
        return [f"blocks.{b}.{p}" for b in range(self.n_blocks) for p in ("up", "down")]

    def forward(self, X, features: bool = False, capture: dict | None = None, act_quant=None, weights=None):
        """Logits for inputs X (n, d). `capture` collects each linear's input (for calibration);
        `act_quant(name, x)` replaces a linear's input (to emulate W8A8); `weights` overrides some."""
        w = {**self.weights, **(weights or {})}

        def linear(name, x):
            if capture is not None:
                capture.setdefault(name, []).append(x)
            if act_quant is not None:
                x = act_quant(name, x)
            return x @ w[name].T

        x = np.asarray(X, float)
        for b in range(self.n_blocks):
            h = rmsnorm(x, w[f"blocks.{b}.norm"])
            x = x + linear(f"blocks.{b}.down", np.maximum(linear(f"blocks.{b}.up", h), 0.0))
        return x if features else linear("head", rmsnorm(x))

    def with_weights(self, new: dict) -> "TinyModel":
        """A copy with some weights (or gains) replaced - a quantized or smoothed model."""
        m = TinyModel(self.d, self.hidden, self.n_blocks, self.n_classes, self.noise, self.seed,
                      {**self.weights, **new})
        m.prototypes = self.prototypes
        return m

    def calibration_inputs(self, X, weights=None) -> dict:
        """Each linear's input activations over X: what GPTQ's Hessian and AWQ's statistic are built from."""
        cap: dict = {}
        self.forward(X, capture=cap, weights=weights)
        return {k: np.concatenate(v) for k, v in cap.items()}

    def fold(self, name: str, s) -> dict:
        """Absorb a per-input-channel scale s into the model without changing its function: the linear's
        weight columns are multiplied by s and whatever produces its input is divided by s - the norm gain
        for an up-projection, the up-projection's rows for a down-projection (ReLU commutes with s > 0)."""
        b = name.split(".")[1]
        new = {name: self.weights[name] * s}
        if name.endswith(".up"):
            new[f"blocks.{b}.norm"] = self.weights[f"blocks.{b}.norm"] / s
        else:
            new[f"blocks.{b}.up"] = self.weights[f"blocks.{b}.up"] / s[:, None]
        return new


def quantize_model(model: TinyModel, method: str = "rtn", bits: int = 4, group_size: int | None = 32,
                   calib=None, symmetric: bool = True, targets: list | None = None) -> TinyModel:
    """Weight-only INT`bits` quantization of `targets` (default: the hidden linears, head ignored).
    'rtn': round to nearest, no data. 'gptq': each layer calibrated on the inputs produced by the layers
    already quantized (llm-compressor's sequential pipeline). 'awq': a transform, then RTN - search and
    fold each layer's scale on the still-full-precision model, then round everything (AWQModifier
    followed by QuantizationModifier). 'awq+gptq': the same transform, then GPTQ (AWQModifier + GPTQModifier)."""
    from .awq import search_scale
    from .gptq import gptq, rtn

    targets = targets or model.linears()
    if method in ("awq", "awq+gptq"):
        m = model
        for name in targets:
            X = m.calibration_inputs(calib)[name]
            s, _, _ = search_scale(m.weights[name], X, bits, group_size, symmetric=symmetric)
            m = m.with_weights(m.fold(name, s))
        return quantize_model(m, "rtn" if method == "awq" else "gptq", bits, group_size, calib, symmetric, targets)
    q = model
    for name in targets:
        if method == "rtn":
            w_hat = rtn(q.weights[name], bits, group_size, symmetric).w_hat
        elif method == "gptq":
            w_hat = gptq(q.weights[name], q.calibration_inputs(calib)[name], bits=bits,
                         group_size=group_size, symmetric=symmetric).w_hat
        else:
            raise ValueError(f"method must be 'rtn', 'gptq', 'awq' or 'awq+gptq', not {method!r}")
        q = q.with_weights({name: w_hat})
    return q
