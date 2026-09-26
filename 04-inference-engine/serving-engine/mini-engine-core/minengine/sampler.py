"""sampler.py - logits in, one token out: the pipeline every engine runs after the forward pass.

The one idea: sampling settings never change the model; they reshape its output distribution.
The order (as in vLLM V1's Sampler): structured-output mask -> penalties -> greedy if
temperature == 0 -> temperature -> min-p -> top-k -> top-p -> draw with the request's own seeded
generator. Every stage either rescales logits or sets some to -inf. Logprobs are reported from the
RAW logits (vLLM's default `raw_logprobs`), not from the reshaped distribution.

Structured output is the same mechanism with a grammar behind it: an FSM says which tokens are
legal in its current state, everything else is masked to -inf before sampling, and the FSM
advances on the token that was drawn. The model never "knows" the schema; it is fenced in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import EOS, VOCAB

GREEDY_EPS = 1e-5            # temperatures below this mean greedy, as in vLLM


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0          # 0 -> greedy (argmax)
    top_k: int = 0                    # 0 -> off
    top_p: float = 1.0                # 1 -> off
    min_p: float = 0.0                # 0 -> off
    repetition_penalty: float = 1.0   # >1 discourages tokens seen in prompt or output
    presence_penalty: float = 0.0     # subtract once if the token appeared in the output
    frequency_penalty: float = 0.0    # subtract per occurrence in the output
    seed: int | None = None           # per-request generator: reproducible whatever else is batched
    stop: tuple = ()                  # stop strings (checked on the decoded text)
    stop_token_ids: tuple = ()
    ignore_eos: bool = False
    logprobs: int = 0                 # return this many top (raw) logprobs per generated token
    fsm: object = None                # structured output: start() / allowed(state) / advance(state, tok)


def log_softmax(x):
    m = np.max(x)
    return x - m - np.log(np.exp(x - m).sum())


def probs(x):
    return np.exp(log_softmax(x))


def apply_penalties(logits, p: SamplingParams, prompt_ids=(), output_ids=()):
    """Repetition penalty (prompt + output, HF/vLLM style: divide positive logits, multiply
    negative ones); presence and frequency penalties (output only, OpenAI style)."""
    x = np.array(logits, float)
    if p.repetition_penalty != 1.0:
        seen = np.unique(np.asarray(list(prompt_ids) + list(output_ids), int))
        x[seen] = np.where(x[seen] > 0, x[seen] / p.repetition_penalty, x[seen] * p.repetition_penalty)
    if len(output_ids) and (p.presence_penalty or p.frequency_penalty):
        counts = np.bincount(np.asarray(output_ids, int), minlength=len(x))
        x -= p.frequency_penalty * counts + p.presence_penalty * (counts > 0)
    return x


def min_p_filter(x, min_p):
    """Keep tokens whose probability is at least min_p x the top token's probability."""
    if min_p <= 0:
        return x
    pr = probs(x)
    return np.where(pr >= min_p * pr.max(), x, -np.inf)


def top_k_filter(x, k):
    """Keep the k largest logits (ties at the k-th value are kept too)."""
    if k <= 0 or k >= len(x):
        return x
    return np.where(x >= np.sort(x)[-k], x, -np.inf)


def top_p_filter(x, p):
    """Nucleus: keep the smallest set of top tokens whose total probability reaches p."""
    if p >= 1.0:
        return x
    order = np.argsort(-x, kind="stable")
    pr = probs(x)[order]
    keep = np.zeros(len(x), bool)
    keep[order[(np.cumsum(pr) - pr) < p]] = True      # mass strictly before a token < p -> keep it
    return np.where(keep, x, -np.inf)


def process_logits(logits, p: SamplingParams, prompt_ids=(), output_ids=(), allowed=None):
    """Everything between the model and the draw. Returns the logits the token is drawn from."""
    x = np.asarray(logits, float)
    if allowed is not None:
        x = np.where(allowed, x, -np.inf)
    x = apply_penalties(x, p, prompt_ids, output_ids)
    if p.temperature < GREEDY_EPS:
        return x
    x = x / p.temperature
    return top_p_filter(top_k_filter(min_p_filter(x, p.min_p), p.top_k), p.top_p)


def sample(logits, p: SamplingParams, rng, prompt_ids=(), output_ids=(), allowed=None):
    """Returns (token, raw logprob of that token, {token: raw logprob} for the top p.logprobs)."""
    x = process_logits(logits, p, prompt_ids, output_ids, allowed)
    tok = int(np.argmax(x)) if p.temperature < GREEDY_EPS else int(rng.choice(len(x), p=probs(x)))
    raw = log_softmax(np.asarray(logits, float))
    top = {int(i): float(raw[i]) for i in np.argsort(-raw)[:p.logprobs]} if p.logprobs else None
    return tok, float(raw[tok]), top


class ChoiceFSM:
    """Structured output as a token mask: the output must be exactly one of `choices`.
    State = the bytes emitted so far. Byte tokens make the grammar walk trivial; with a BPE
    vocabulary each token spans several characters and the engine must precompute, per grammar
    state, which of ~100k tokens keep the text legal (what xgrammar / llguidance optimise)."""

    def __init__(self, choices):
        self.choices = [c.encode() for c in choices]

    def start(self):
        return b""

    def allowed(self, state: bytes) -> np.ndarray:
        mask = np.zeros(VOCAB, bool)
        for c in self.choices:
            if c.startswith(state):
                mask[c[len(state)] if len(c) > len(state) else EOS] = True
        return mask

    def advance(self, state: bytes, tok: int) -> bytes:
        return state + bytes([tok]) if tok < 256 else state
