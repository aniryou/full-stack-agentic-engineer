"""model.py — a tiny decoder-only transformer: sample with log-probabilities, and score with full logits.

One idea: distillation needs three things from a language model, and only three — *sample* completions
(SeqKD's teacher, GKD's student), the *log-probability* of each sampled token (the on-policy reward), and the
*full next-token distribution* at every completion position (logit KD, GKD's divergence, a draft's acceptance
Σ min(p, q)). ``sample`` returns the first two, ``completion_logits`` the third. The blocks mirror
``00-foundations/transformers/lessons/03_tiny_gpt.py`` (pre-LN, GPT-2 initialisation) at a size that trains
on a laptop CPU in seconds; teacher and student differ only in width and depth.

Importing this module imports torch; ``distillab.tinylm.task`` does not.
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


class TinyLM(nn.Module):
    """Token + position embeddings, ``n_layers`` pre-LN blocks, a tied output head."""

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
def sample(model: TinyLM, prompts: torch.Tensor, max_new: int, temperature: float = 1.0,
           generator: torch.Generator | None = None):
    """Sample completions for a batch of equal-length prompts: ``(completions, logprobs, mask)``, each
    ``(B, max_new)`` — the tokens (PAD after ``<eos>``), the sampler's log-probability of each sampled token
    at temperature 1 (what an engine returns as ``logprobs``), and 1.0 where a token was really generated.
    ``temperature`` 0 samples greedily. No KV cache: the tiny model recomputes the prefix every step."""
    was_training = model.training
    model.eval()
    B = prompts.shape[0]
    seq = prompts.clone()
    done = torch.zeros(B, dtype=torch.bool, device=prompts.device)
    toks, lps, mask = [], [], []
    for _ in range(max_new):
        logits = model(seq)[:, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=generator).squeeze(1)
        lp = logp.gather(1, nxt[:, None]).squeeze(1)
        live = ~done
        nxt = torch.where(live, nxt, torch.full_like(nxt, PAD))
        toks.append(nxt)
        lps.append(torch.where(live, lp, torch.zeros_like(lp)))
        mask.append(live.float())
        done = done | (nxt == EOS)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        if bool(done.all()):
            break
    pad = max_new - len(toks)
    comp, lp, m = torch.stack(toks, 1), torch.stack(lps, 1), torch.stack(mask, 1)
    if pad:
        comp, lp, m = F.pad(comp, (0, pad), value=PAD), F.pad(lp, (0, pad)), F.pad(m, (0, pad))
    model.train(was_training)
    return comp, lp, m


def completion_logits(model: TinyLM, prompts: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    """The logits that predict each completion token: ``(B, T_c, V)``. Position t of the concatenated
    sequence predicts token t + 1, so the rows start one position before the completion."""
    seq = torch.cat([prompts, completions], dim=1)
    return model(seq[:, :-1])[:, prompts.shape[1] - 1:]


def token_logprobs(model: TinyLM, prompts: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    """Log-probability of each completion token under ``model`` (with gradients): ``(B, T_c)``."""
    logp = F.log_softmax(completion_logits(model, prompts, completions).float(), dim=-1)
    return logp.gather(2, completions.clamp(max=logp.shape[-1] - 1)[:, :, None]).squeeze(2)
