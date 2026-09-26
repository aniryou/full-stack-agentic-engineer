"""model.py — a tiny decoder-only transformer, sampling with log-probabilities, and sequence scoring.

One idea: an RL step for a language model needs exactly two things from the model — *sample*
completions (recording the log-probability of every sampled token) and *score* given completions
(recompute those log-probabilities with gradients). Everything else in GRPO is bookkeeping on
those numbers. This file is the torch half; it mirrors ``00-foundations/transformers/lessons/03_tiny_gpt.py``
(same blocks, GPT-2 initialisation) at a size that trains on a laptop CPU in seconds.

Importing this module imports torch (lazily, from ``thinklab.tinyrl.train``); the rest of
``thinklab`` never needs it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import EOS, PAD


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, max_len: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, max_len: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, max_len)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Token + position embeddings, ``n_layers`` pre-LN blocks, tied output head."""

    def __init__(self, vocab: int, d: int = 64, n_layers: int = 2, n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, max_len) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def sample(model: TinyGPT, prompts: torch.Tensor, max_new: int, temperature: float = 1.0,
           generator: torch.Generator | None = None):
    """Sample completions for a batch of equal-length prompts.

    Returns ``(completions, logprobs, mask)``, each ``(B, max_new)``: the sampled tokens (PAD after
    ``<eos>``), the sampler's log-probability of each sampled token (the "old policy" numbers an
    engine such as vLLM would return), and 1.0 where a token was really generated. No KV cache:
    the tiny model recomputes the prefix every step, which is cheap at this size and keeps the
    code short (an engine would not)."""
    model.eval()
    B = prompts.shape[0]
    seq = prompts.clone()
    done = torch.zeros(B, dtype=torch.bool, device=prompts.device)
    toks, lps, mask = [], [], []
    for _ in range(max_new):
        logits = model(seq)[:, -1] / max(temperature, 1e-6)
        logp = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(logp.float().exp(), 1, generator=generator).squeeze(1)
        lp = logp.gather(1, nxt[:, None]).squeeze(1)
        live = ~done
        nxt = torch.where(live, nxt, torch.full_like(nxt, PAD))
        toks.append(nxt)
        lps.append(torch.where(live, lp.float(), torch.zeros_like(lp.float())))
        mask.append(live.float())
        done = done | (nxt == EOS)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        if bool(done.all()):
            break
    pad = max_new - len(toks)
    comp = torch.stack(toks, 1)
    lp = torch.stack(lps, 1)
    m = torch.stack(mask, 1)
    if pad:
        comp = F.pad(comp, (0, pad), value=PAD)
        lp = F.pad(lp, (0, pad))
        m = F.pad(m, (0, pad))
    return comp, lp, m


def token_logprobs(model: TinyGPT, prompts: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    """Log-probability of each completion token under ``model`` (with gradients): ``(B, T_c)``.
    Position t of the concatenated sequence predicts token t + 1, so the completion's scores are
    read from the positions starting one before it."""
    seq = torch.cat([prompts, completions], dim=1)
    logits = model(seq[:, :-1])[:, prompts.shape[1] - 1:]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(2, completions.clamp(max=logits.shape[-1] - 1)[:, :, None]).squeeze(2)
