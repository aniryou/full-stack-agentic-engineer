"""train.py — SFT warm-up, then GRPO with the verifier, on the tiny transformer (torch, CPU is fine).

One idea: GRPO is "sample a group, score it, push up what beat the group's average". Per step:

    1. sample G completions for each of B prompts from the current policy (the rollout; an engine's job at scale)
    2. score each with the verifier: r ∈ {0, 1}
    3. advantage A_i = (r_i − mean(r_group)) / (std(r_group) + 1e-4)   — no critic, the group is the baseline
    4. loss per token = −min(ρ·A, clip(ρ, 1−ε_low, 1+ε_high)·A) + β·k3(π_ref, π_θ),  ρ = π_θ / π_old
    5. average over tokens ("dapo": ÷ all active tokens in the batch), step the optimizer

The names and defaults follow TRL's ``GRPOConfig`` (v1.14.0): ``num_generations``, ``beta``
(0.0 there: no reference model), ``epsilon`` / ``epsilon_high``, ``num_iterations`` (μ; with 1 the
ratio is exactly 1 and the clip never binds), ``loss_type`` (``grpo`` / ``dr_grpo`` / ``dapo``) and
``scale_rewards``. Before RL a short SFT warm-up teaches the *format* from demonstrations with every
scratchpad length 0..K equally often — the "cold start" of the R1 recipe (PRIMER §5) — so RL starts
from a policy that sometimes thinks and sometimes does not, and has to discover which pays.

Everything returned is what *this run* measured on *this machine*: curves, timings and the
accuracy-by-scratchpad-length table. ``thinklab.tinyrl.curves`` holds a recorded copy for
machines without torch (labelled illustrative).
"""
from __future__ import annotations

import copy
import platform
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass

import torch

from .model import TinyGPT, sample, token_logprobs
from .task import VOCAB, DigitSum, parse, reward, sft_batch


@dataclass
class TinyRLConfig:
    k: int = 6                      # digits per problem (the "depth" of the problem)
    base: int = 10                  # the answer is the sum mod base
    d: int = 64
    n_layers: int = 2
    n_heads: int = 4
    sft_steps: int = 400
    sft_batch: int = 128
    sft_lr: float = 3e-3
    rl_steps: int = 60
    prompts_per_step: int = 32      # B
    num_generations: int = 8        # G (TRL default 8)
    rl_lr: float = 1e-3
    beta: float = 0.0               # KL weight to the SFT reference (TRL default 0.0; R1 used 0.001)
    epsilon: float = 0.2            # ε_low
    epsilon_high: float = 0.2       # ε_high (DAPO "clip-higher": 0.28)
    num_iterations: int = 1         # μ: optimizer passes per rollout batch
    loss_type: str = "dapo"         # "grpo" | "dr_grpo" | "dapo"
    scale_rewards: str = "group"    # "group" | "batch" | "none" (Dr. GRPO)
    temperature: float = 1.0
    eval_every: int = 5
    eval_prompts: int = 256
    seed: int = 0
    threads: int = 2                # torch CPU threads (the machine may be shared)


def _tensor(rows):
    return torch.tensor(rows, dtype=torch.long)


def sft(model: TinyGPT, task: DigitSum, cfg: TinyRLConfig, rng: random.Random, log=print) -> list:
    """Cross-entropy on completion tokens only (the prompt is given, not learned)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sft_lr)
    curve = []
    model.train()
    for step in range(cfg.sft_steps + 1):
        batch = sft_batch(task, cfg.sft_batch, rng)
        prompts = _tensor([p.prompt for p, _ in batch])
        comps = [c for _, c in batch]
        width = task.max_completion
        mask = _tensor([[1] * len(c) + [0] * (width - len(c)) for c in comps]).float()
        comps = _tensor([c + [0] * (width - len(c)) for c in comps])
        lp = token_logprobs(model, prompts, comps)
        loss = -(lp * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.sft_steps // 8) == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4)})
            log(f"  sft step {step:4d}  loss {loss.item():.3f}")
    return curve


def group_advantages(rewards: torch.Tensor, groups: int, scale: str = "group") -> torch.Tensor:
    """(r − group mean) / (std + 1e-4), std with Bessel's correction — TRL's ``_compute_advantages``."""
    r = rewards.view(groups, -1)
    adv = r - r.mean(1, keepdim=True)
    if scale == "group":
        adv = adv / (r.std(1, unbiased=True, keepdim=True) + 1e-4)
    elif scale == "batch":
        adv = adv / (rewards.std(unbiased=True) + 1e-4)
    elif scale != "none":
        raise ValueError(scale)
    return adv.reshape(-1)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, cfg: TinyRLConfig, max_len: int):
    """The per-token clipped surrogate (+ β·k3) and the three aggregations TRL names."""
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    clipped = torch.clamp(ratio, 1 - cfg.epsilon, 1 + cfg.epsilon_high)
    per_token = -torch.min(ratio * a, clipped * a)
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1                                         # k3 estimator of KL(π_θ ‖ π_ref)
    if cfg.beta:
        per_token = per_token + cfg.beta * kl
    if cfg.loss_type == "grpo":                                       # mean per sequence, then over sequences
        loss = ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    elif cfg.loss_type == "dr_grpo":                                  # ÷ a constant: B·G·max_completion_length
        loss = (per_token * mask).sum() / (per_token.shape[0] * max_len)
    elif cfg.loss_type == "dapo":                                     # ÷ active tokens in the batch
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    else:
        raise ValueError(cfg.loss_type)
    stats = {"kl": ((kl * mask).sum() / mask.sum()).item(),
             "clip_frac": ((((ratio < 1 - cfg.epsilon) & (a < 0)) | ((ratio > 1 + cfg.epsilon_high) & (a > 0)))
                           .float() * mask).sum().item() / mask.sum().item()}
    return loss, stats


@torch.no_grad()
def evaluate(model: TinyGPT, task: DigitSum, n: int, gen: torch.Generator, rng: random.Random,
             temperature: float = 1.0) -> dict:
    """Sample one completion per problem; accuracy overall and by scratchpad length."""
    probs = [task.sample(rng) for _ in range(n)]
    comps, _, _ = sample(model, _tensor([p.prompt for p in probs]), task.max_completion, temperature, gen)
    by_len, hits, lengths, fmt = Counter(), Counter(), [], 0
    for p, c in zip(probs, comps.tolist()):
        r = parse(c)
        if r["ok"]:
            fmt += 1
            by_len[r["scratch"]] += 1
            hits[r["scratch"]] += reward(p, c) == 1.0
        lengths.append(r["length"])
    acc = sum(hits.values()) / n
    return {"accuracy": round(acc, 4), "format_ok": round(fmt / n, 4),
            "mean_length": round(sum(lengths) / n, 3),
            "scratch_hist": {j: by_len[j] for j in range(task.k + 1)},
            "acc_by_scratch": {j: (round(hits[j] / by_len[j], 3) if by_len[j] else None) for j in range(task.k + 1)}}


def grpo(model: TinyGPT, ref: TinyGPT, task: DigitSum, cfg: TinyRLConfig, gen: torch.Generator,
         rng: random.Random, log=print) -> list:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.rl_lr)
    curve, G, B = [], cfg.num_generations, cfg.prompts_per_step
    for step in range(cfg.rl_steps + 1):
        probs = [task.sample(rng) for _ in range(B)]
        prompts = _tensor([p.prompt for p in probs for _ in range(G)])
        comps, old_lp, mask = sample(model, prompts, task.max_completion, cfg.temperature, gen)   # 1. rollout
        rewards = torch.tensor([reward(p, c) for p, c in                                          # 2. verify
                                zip([p for p in probs for _ in range(G)], comps.tolist())])
        adv = group_advantages(rewards, B, cfg.scale_rewards)                                     # 3. advantages
        zero_std = (rewards.view(B, G).std(1) == 0).float().mean().item()
        with torch.no_grad():
            ref_lp = token_logprobs(ref, prompts, comps)
        model.train()
        for _ in range(cfg.num_iterations):                                                      # 4-5. update
            lp = token_logprobs(model, prompts, comps)
            loss, st = grpo_loss(lp, old_lp, ref_lp, adv, mask, cfg, task.max_completion)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % cfg.eval_every == 0 or step == cfg.rl_steps:
            row = {"step": step, "reward": round(rewards.mean().item(), 4),
                   "length": round(mask.sum(1).mean().item(), 3), "frac_zero_std": round(zero_std, 3),
                   "kl": round(st["kl"], 5), "clip_frac": round(st["clip_frac"], 4)}
            curve.append(row)
            log(f"  rl step {step:4d}  reward {row['reward']:.3f}  length {row['length']:.2f}  "
                f"zero-std groups {row['frac_zero_std']:.2f}  kl {row['kl']:.4f}")
    return curve


def run(cfg: TinyRLConfig | None = None, log=print) -> dict:
    """SFT warm-up → evaluate → GRPO → evaluate. Returns everything measured (JSON-serialisable)."""
    cfg = cfg or TinyRLConfig()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng, gen = random.Random(cfg.seed), torch.Generator().manual_seed(cfg.seed)
    task = DigitSum(cfg.k, cfg.base)
    model = TinyGPT(VOCAB, cfg.d, cfg.n_layers, cfg.n_heads, task.seq_len)
    t0 = time.perf_counter()
    log(f"tiny transformer: {model.num_params():,} parameters; task: last digit of a {cfg.k}-digit sum")
    sft_curve = sft(model, task, cfg, rng, log)
    t1 = time.perf_counter()
    before = evaluate(model, task, cfg.eval_prompts, gen, random.Random(cfg.seed + 1), cfg.temperature)
    log(f"after SFT: accuracy {before['accuracy']:.3f}, mean completion {before['mean_length']:.2f} tokens")
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rl_curve = grpo(model, ref, task, cfg, gen, rng, log)
    t2 = time.perf_counter()
    after = evaluate(model, task, cfg.eval_prompts, gen, random.Random(cfg.seed + 1), cfg.temperature)
    log(f"after GRPO: accuracy {after['accuracy']:.3f}, mean completion {after['mean_length']:.2f} tokens")
    return {"source": "measured", "config": asdict(cfg), "params": model.num_params(),
            "sft": sft_curve, "rl": rl_curve, "before": before, "after": after,
            "timing_s": {"sft": round(t1 - t0, 1), "rl": round(t2 - t1, 1)},
            "machine": f"{platform.machine()} / torch {torch.__version__} / {cfg.threads} CPU threads"}
