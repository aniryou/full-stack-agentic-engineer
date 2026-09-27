"""A draft model for speculative decoding is a student whose only metric is acceptance.

The one idea: the target accepts a drafted token with probability Σ_v min(p(v), q(v)) = 1 − TV(p, q), so a
draft is good exactly when its distribution is close to the target's *on the target's own text* — which is
what distilling from the target optimises. One target pass then yields (1 − α^(k+1)) / (1 − α) tokens, and
the speedup is that over (k·c + 1), c = a draft step's cost in target steps. These are
`minengine.spec.acceptance_rate`, `expected_tokens`, `speedup` and `best_k`, restated. A bigger draft
accepts more and costs more: the best draft size is where the product turns over.
"""
from __future__ import annotations

import numpy as np

from .losses import kl


def acceptance_rate(p, q) -> float:
    """α = Σ_v min(p(v), q(v)) for one position (minengine.spec.acceptance_rate)."""
    return float(np.minimum(p, q).sum())


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens per target pass: 1 + α + … + α^k = (1 − α^(k+1)) / (1 − α)."""
    return k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    """Wall-clock gain when one draft step costs c target steps: E[tokens] / (k·c + 1)."""
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha: float, c: float, k_max: int = 16) -> int:
    return max(range(1, k_max + 1), key=lambda k: speedup(alpha, k, c))


def greedy_acceptance(p, q) -> float:
    """vLLM drafts greedily by default: the draft proposes argmax q, accepted with probability p(argmax q)."""
    return float(np.asarray(p)[int(np.argmax(q))])


def vllm_view(alpha: float, k: int) -> dict:
    """What vLLM's spec-decode counters show for an i.i.d. per-token α: mean acceptance length (bonus token
    included) = expected_tokens; per-position acceptance α^(i+1); 'draft acceptance rate' = (E − 1)/k ≠ α."""
    E = expected_tokens(alpha, k)
    return {"mean_acceptance_length": E, "per_position": [alpha ** (i + 1) for i in range(k)],
            "draft_acceptance_rate": (E - 1) / k}


def draft_cost(draft_params: float, target_params: float, overhead: float = 0.0) -> float:
    """c for memory-bound decode: a step streams the weights, so c ≈ draft bytes / target bytes (+ overhead)."""
    return draft_params / target_params + overhead


def acceptance_on_text(target, draft, seqs) -> dict:
    """Score a draft where it will be used — on the target's own samples: mean α = Σ min(p, q) per position,
    greedy acceptance p(argmax q), KL(p ‖ q), and TV (α = 1 − TV exactly)."""
    ctx, _ = target.positions(seqs)
    P, Q = target.probs(ctx), draft.probs(ctx)
    a = np.minimum(P, Q).sum(1)
    return {"alpha": float(a.mean()), "greedy": float(P[np.arange(len(P)), Q.argmax(1)].mean()),
            "kl": float(kl(P, Q).mean()), "tv": float(0.5 * np.abs(P - Q).sum(1).mean())}
