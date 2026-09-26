"""spec.py - speculative decoding: guess k tokens cheaply, verify them in ONE target pass, lose nothing.

The one idea: decode is memory-bound, so a target forward pass over k+1 positions costs about the
same as over one. A cheap proposer drafts x1..xk; the target scores all k+1 positions at once; then
(Leviathan et al. 2023; Chen et al. 2023):
    accept draft token x ~ q with probability min(1, p(x) / q(x));
    at the first rejection, sample a "recovered" token from norm(max(0, p - q)) and stop;
    if all k are accepted, sample a "bonus" token from the target's next distribution.
P(emit y) = q(y) min(1, p(y)/q(y)) + (1 - sum_x min(p(x), q(x))) * residual(y) = p(y): the output
follows the target distribution EXACTLY - speculation changes speed, never quality. With per-token
acceptance rate alpha = sum min(p, q) = 1 - TV(p, q), one pass yields (1 - alpha^(k+1)) / (1 - alpha)
tokens on average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import VOCAB, softmax


def acceptance_rate(p, q) -> float:
    """alpha = sum_x min(p(x), q(x)): the chance a draft token from q survives verification against p."""
    return float(np.minimum(p, q).sum())


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens emitted per target pass: 1 + alpha + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)."""
    return k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    """Wall-clock gain when one draft step costs c target steps: E[tokens] / (k c + 1)."""
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha: float, c: float, k_max: int = 16) -> int:
    return max(range(1, k_max + 1), key=lambda k: speedup(alpha, k, c))


def verify(p, q, draft, rng) -> list[int]:
    """p: (k+1, V) target distributions at each draft position plus one; q: (k, V) the draft's;
    draft: the k proposed tokens. Returns the accepted prefix plus one recovered or bonus token."""
    out = []
    for i, x in enumerate(draft):
        if rng.random() < min(1.0, p[i, x] / q[i, x]):
            out.append(int(x))                                        # accepted
            continue
        residual = np.maximum(p[i] - q[i], 0.0)
        out.append(int(rng.choice(len(residual), p=residual / residual.sum())))   # recovered
        return out
    out.append(int(rng.choice(p.shape[1], p=p[len(draft)])))              # bonus
    return out


def ngram_propose(tokens, k: int, n: int = 3) -> list[int]:
    """Prompt lookup: find the latest earlier occurrence of the last n tokens and propose the k
    tokens that followed it. Free to compute; great when the output copies the context (code edits,
    quoting tool results), useless otherwise. As a draft distribution it is one-hot."""
    if len(tokens) <= n:
        return []
    tail = list(tokens[-n:])
    for s in range(len(tokens) - n - 1, -1, -1):
        if list(tokens[s:s + n]) == tail:
            return list(tokens[s + n:s + n + k])
    return []


def lm_probs(model, temperature: float = 1.0):
    """tokens -> (len(tokens), V) next-token distributions at every position, from ONE forward pass.
    Temperature 0 gives one-hot argmax rows, which turns verification into 'accept iff equal'."""
    def fn(tokens):
        logits = model.forward_dense(tokens)
        if temperature == 0:
            return np.eye(logits.shape[1])[logits.argmax(1)]
        return softmax(logits / temperature)
    return fn


@dataclass
class SpecStats:
    passes: int = 0          # target forward passes
    proposed: int = 0        # draft tokens proposed
    accepted: int = 0        # draft tokens accepted
    emitted: int = 0         # tokens produced (accepted + recovered + bonus)

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_pass(self) -> float:
        return self.emitted / self.passes if self.passes else 0.0


def speculative_generate(target_probs, prompt, max_new_tokens: int, k: int = 4, draft_probs=None,
                         ngram: int = 3, seed: int = 0):
    """Generate with speculation. target_probs(tokens) -> (T, V) distributions at every position
    (one pass); draft_probs(tokens) -> the draft's (T, V) likewise, called k times per round
    (autoregressive drafting); if draft_probs is None, draft by n-gram prompt lookup instead.
    `seed` may be an int or a numpy Generator."""
    rng, out, st = np.random.default_rng(seed), list(prompt), SpecStats()
    while len(out) - len(prompt) < max_new_tokens:
        if draft_probs is None:
            draft = ngram_propose(out, k, ngram)
            q = np.eye(VOCAB)[draft] if draft else np.zeros((0, VOCAB))
        else:
            draft, q = [], []
            for _ in range(k):
                qi = draft_probs(out + draft)[-1]
                draft.append(int(rng.choice(len(qi), p=qi)))
                q.append(qi)
            q = np.array(q)
        p = target_probs(out + draft)[len(out) - 1:]                  # k+1 rows from one target pass
        emitted = verify(p, q, draft, rng)
        st.passes += 1
        st.proposed += len(draft)
        st.accepted += len(emitted) - 1
        st.emitted += len(emitted)
        out += emitted
    return out[len(prompt):len(prompt) + max_new_tokens], st
