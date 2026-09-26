"""rollout.py — the bookkeeping around an RL step when an inference engine generates the rollouts.

One idea: in RL for LLMs the *rollout* — generating G completions for each of B prompts — is an
inference workload, and the trainer only consumes its outputs. So a GRPO step is: an engine (vLLM,
SGLang; here the tiny model or the simulated one) returns tokens **and the sampler's log-probs**; a
verifier scores them; the trainer recomputes log-probs with its own copy of the weights, and the
two copies never agree exactly (different kernels, batch shapes, precision), which an importance
ratio corrects. Then the weights go back to the engine. What this module computes, in plain Python:

    group_advantages      (r − mean) / (std + 1e-4) per prompt, Bessel std — TRL's _compute_advantages
    frac_zero_std         share of groups whose rewards are all equal: no gradient from them (TRL logs it)
    dynamic_sampling      DAPO: drop those groups and sample more prompts until the batch is full
    is_ratios             train–inference mismatch correction, TRL's four modes (default "sequence_mask", C_max 3.0)
    soft_overlong         DAPO's length penalty in the last L_cache tokens before L_max
    clipped_token_loss    −min(ρA, clip(ρ, 1−ε_low, 1+ε_high)A) per token
    aggregate             "grpo" / "dr_grpo" / "dapo" normalisations (who gets weight: short or long answers)
    rollout_phase         a synchronous rollout batch waits for its longest completion: GPU idle share
    weight_sync_bytes     what goes back to the engine each step (full vs delta sync)

``vllm_rollouts`` / ``one_grpo_step`` are the T1 path (vLLM's offline ``LLM`` API + transformers,
imported lazily); ``trl_grpo_config`` returns the TRL ``GRPOConfig`` kwargs this lab uses on a T4.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Rollout:
    prompt_id: str
    tokens: int                               # completion length
    reward: float
    sampler_logps: list = field(default_factory=list)    # per token, from the engine that generated it
    trainer_logps: list = field(default_factory=list)    # per token, recomputed by the trainer ("old" policy)
    truncated: bool = False


def group_by_prompt(rollouts: list) -> dict:
    groups = defaultdict(list)
    for r in rollouts:
        groups[r.prompt_id].append(r)
    return dict(groups)


def _std(xs: list) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0          # Bessel's correction, as torch.std / TRL nanstd


def group_advantages(rewards: list, scale: str = "group", eps: float = 1e-4, batch_std: float | None = None) -> list:
    """Advantages for one group. ``scale``: "group" (default), "batch" (pass ``batch_std``) or "none" (Dr. GRPO)."""
    mean = statistics.fmean(rewards)
    adv = [r - mean for r in rewards]
    if scale == "group":
        s = _std(rewards)
        return [a / (s + eps) for a in adv]
    if scale == "batch":
        return [a / ((batch_std or 0.0) + eps) for a in adv]
    if scale == "none":
        return adv
    raise ValueError(scale)


def frac_zero_std(groups: dict) -> float:
    return sum(_std([r.reward for r in g]) == 0 for g in groups.values()) / max(1, len(groups))


def dynamic_sampling(groups: dict) -> dict:
    """DAPO's filter: keep only groups with mixed outcomes (all-right or all-wrong groups carry no signal)."""
    return {k: g for k, g in groups.items() if _std([r.reward for r in g]) > 0}


def k3(logp: float, ref_logp: float) -> float:
    """Schulman's k3 estimator of KL(π_θ ‖ π_ref) from one sample: e^d − d − 1 with d = log π_ref − log π_θ.
    Always ≥ 0, unbiased, low variance; the per-token KL term of GRPO."""
    d = ref_logp - logp
    return math.exp(d) - d - 1


def clipped_token_loss(logp: float, old_logp: float, adv: float, eps_low: float = 0.2, eps_high: float = 0.2) -> float:
    ratio = math.exp(logp - old_logp)
    return -min(ratio * adv, min(max(ratio, 1 - eps_low), 1 + eps_high) * adv)


def aggregate(per_token: list, kind: str = "dapo", max_completion_length: int | None = None) -> float:
    """Reduce per-token losses (a list of per-sequence lists) to one number, as TRL's ``loss_type`` does.

    grpo:    mean over each sequence's tokens, then mean over sequences — a long sequence's tokens each
             weigh less (the length bias Dr. GRPO and DAPO point out)
    dr_grpo: sum over all tokens ÷ (sequences × max_completion_length) — a constant normaliser
    dapo:    sum over all tokens ÷ number of tokens — every token weighs the same"""
    if kind == "grpo":
        return statistics.fmean(sum(s) / max(1, len(s)) for s in per_token)
    if kind == "dr_grpo":
        return sum(map(sum, per_token)) / (len(per_token) * max_completion_length)
    if kind == "dapo":
        return sum(map(sum, per_token)) / max(1, sum(map(len, per_token)))
    raise ValueError(kind)


def is_ratios(sampler_logps: list, trainer_logps: list, mode: str = "sequence_mask", c_max: float | None = 3.0,
              c_min: float | None = None) -> list:
    """Per-token importance weights π_trainer / π_sampler for one sequence, TRL's four modes
    (``vllm_importance_sampling_mode``): *_truncate clips the ratio to [c_min, c_max], *_mask zeroes it
    outside; sequence_* uses one ratio for the whole sequence (exp of the summed log-diff)."""
    diffs = [t - s for s, t in zip(sampler_logps, trainer_logps)]
    lo, hi = (c_min if c_min is not None else -math.inf), (c_max if c_max is not None else math.inf)
    if mode.startswith("sequence"):
        ratio = math.exp(sum(diffs))
        if mode == "sequence_truncate":
            ratio = min(max(ratio, lo), hi)
        elif not lo <= ratio <= hi:
            ratio = 0.0
        return [ratio] * len(diffs)
    out = []
    for d in diffs:
        r = math.exp(d)
        out.append(min(max(r, lo), hi) if mode == "token_truncate" else (r if lo <= r <= hi else 0.0))
    return out


def soft_overlong(length: int, max_len: int, cache: int) -> float:
    """DAPO eq. 13: 0 up to L_max − L_cache, then a linear ramp to −1 at L_max, −1 beyond."""
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return ((max_len - cache) - length) / cache
    return -1.0


def rollout_phase(lengths: list, step_s_at_batch, overlap_training_s: float = 0.0) -> dict:
    """A synchronous rollout batch: all sequences start together, the batch shrinks as they finish,
    and the phase ends with the *longest*. ``step_s_at_batch(b)`` is the decode step time with b
    sequences running. Returns the phase time, the share of sequence-slots idle while stragglers
    finish, and the time if the next batch's rollouts overlapped the ``overlap_training_s`` of training
    (one-step-off-policy: rollouts of step k+1 run while step k trains)."""
    order = sorted(lengths)
    t, prev, busy_slots = 0.0, 0, 0
    n = len(order)
    for i, L in enumerate(order):
        running = n - i
        steps = L - prev
        t += steps * step_s_at_batch(running)
        busy_slots += steps * running
        prev = L
    idle = 1 - busy_slots / (n * max(order)) if order else 0.0
    return {"phase_s": t, "idle_share": idle, "longest": max(order), "mean": statistics.fmean(order),
            "sync_step_s": t + overlap_training_s, "overlapped_step_s": max(t, overlap_training_s)}


def weight_sync_bytes(params: float, bytes_per_param: float = 2.0, changed_fraction: float | None = None) -> float:
    """Bytes sent from trainer to engine per step: every weight, or only the changed ones (delta sync;
    verl reports ~1-3% of BF16 weight bytes change step over step, so a delta is far smaller)."""
    full = params * bytes_per_param
    return full if changed_fraction is None else full * changed_fraction


def trl_grpo_config(gpu: str = "T4", model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> dict:
    """``GRPOConfig`` kwargs for one small GRPO run with vLLM colocated (TRL 1.14.0 names; values for a
    15 GB T4 are a starting point, not a measurement (verify they fit with your versions)."""
    cfg = {"output_dir": f"grpo-{model.split('/')[-1]}", "num_generations": 8, "max_completion_length": 256,
           "per_device_train_batch_size": 8, "gradient_accumulation_steps": 1, "learning_rate": 1e-6,
           "beta": 0.0, "epsilon": 0.2, "epsilon_high": 0.28, "loss_type": "dapo", "scale_rewards": "group",
           "mask_truncated_completions": True, "temperature": 1.0, "num_iterations": 1,
           "use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": 0.3,
           "gradient_checkpointing": True, "logging_steps": 1, "max_steps": 20, "report_to": "none"}
    if gpu.upper() in ("T4", "V100", "P100"):
        cfg.update(bf16=False, fp16=True)       # GRPOConfig turns bf16 on unless fp16 is set; a T4 has no bf16
    return cfg


# --- T1: the engine as rollout generator (lazy imports; needs a GPU, vllm and transformers) --------
def vllm_rollouts(model: str, prompts: list, n: int = 8, max_tokens: int = 256, temperature: float = 1.0,
                  gpu_memory_utilization: float = 0.3, dtype: str = "auto", seed: int = 0) -> tuple:
    """Generate ``n`` completions per chat prompt with vLLM's offline API and return
    ``(llm, [[(text, token_ids, sampler_logps)] per prompt])``. ``logprobs=0`` returns the sampled
    token's log-prob at every position."""
    from vllm import LLM, SamplingParams       # noqa: PLC0415 — T1 only
    llm = LLM(model=model, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization, max_model_len=2048, seed=seed)
    sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, logprobs=0, seed=seed)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False)
    result = []
    for o in outs:
        group = []
        for c in o.outputs:
            lps = [pos[t].logprob for pos, t in zip(c.logprobs, c.token_ids)]
            group.append((c.text, list(c.token_ids), lps, list(o.prompt_token_ids)))
        result.append(group)
    return llm, result


def one_grpo_step(model: str = "Qwen/Qwen2.5-0.5B-Instruct", n_prompts: int = 8, n: int = 8, max_tokens: int = 256,
                  lr: float = 1e-6, seed: int = 0, log=print) -> dict:
    """T1: one GRPO step with vLLM generating the rollouts and transformers computing the loss.

    Rollout (vLLM) → verify → advantages → trainer log-probs (the "old" policy) → mismatch vs the
    sampler → clipped loss × IS weight → one SGD step. The weights are *not* pushed back to vLLM here
    (TRL's colocate/server modes do that); the step reports where the time and memory went."""
    import time

    import torch                                                      # noqa: PLC0415
    from transformers import AutoModelForCausalLM                     # noqa: PLC0415

    from .thinking.evalset import make_evalset, verify
    probs = make_evalset(n_prompts, seed=seed, difficulties=(1, 2))
    t0 = time.perf_counter()
    llm, groups = vllm_rollouts(model, [p.prompt for p in probs], n=n, max_tokens=max_tokens, seed=seed,
                                dtype="half" if not torch.cuda.is_bf16_supported() else "auto")
    t1 = time.perf_counter()
    rollouts = []
    for p, g in zip(probs, groups):
        for text, ids, lps, prompt_ids in g:
            rollouts.append((p.id, float(verify(p, text.split("</think>")[-1])), ids, lps, prompt_ids))
    by = group_by_prompt([Rollout(pid, len(ids), r) for pid, r, ids, _, _ in rollouts])
    adv = {}
    for pid, g in by.items():
        for r, a in zip(g, group_advantages([x.reward for x in g])):
            adv.setdefault(pid, []).append(a)
    policy = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).cuda()
    policy.gradient_checkpointing_enable()
    opt = torch.optim.SGD(policy.parameters(), lr=lr)                 # no optimizer state: fits next to vLLM on 15 GB (verify)
    total, mismatch, used = 0.0, [], {k: 0 for k in adv}
    opt.zero_grad()
    for pid, r, ids, lps, prompt_ids in rollouts:
        a = adv[pid][used[pid]]
        used[pid] += 1
        if a == 0 or not ids:
            continue
        seq = torch.tensor([prompt_ids + ids], device="cuda")
        logits = policy(seq).logits[0, len(prompt_ids) - 1:-1].float()
        logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ids, device="cuda")[:, None]).squeeze(1)
        diff = (logp.detach() - torch.tensor(lps, device="cuda"))
        mismatch.append(diff.abs().mean().item())
        w = torch.exp(diff.sum()).clamp(max=3.0)                       # sequence-level truncated IS
        loss = -(torch.exp(logp - logp.detach()) * a * w).sum() / (len(rollouts) * max_tokens)   # dr_grpo normaliser
        loss.backward()
        total += loss.item()
    opt.step()
    t2 = time.perf_counter()
    rewards = [r for _, r, *_ in rollouts]
    res = {"rollouts": len(rollouts), "mean_reward": statistics.fmean(rewards), "frac_zero_std": frac_zero_std(by),
           "mean_completion_tokens": statistics.fmean(len(x[2]) for x in rollouts),
           "sampler_trainer_abs_logp_diff": statistics.fmean(mismatch) if mismatch else 0.0,
           "rollout_s": t1 - t0, "train_s": t2 - t1, "loss": total,
           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    log(res)
    return res
