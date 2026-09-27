"""draft.py — a distilled draft for speculative decoding: a student whose metric is acceptance.

One idea: a draft model is a student of its target, graded on one number — how often the target accepts what
it proposes (PRIMER §7 "A distilled draft for speculative decoding"; the mechanism is serving-engine PRIMER §7):

    α = Σ_v min(p(v), q(v)) = 1 − TV(p, q)          per-position acceptance (the draft samples from q)
    α_greedy = p(argmax q)                          when the draft proposes its argmax (vLLM 0.30.0's default,
                                                    ``draft_sample_method="greedy"``)
    E[tokens per target pass] = (1 − α^(k+1)) / (1 − α)
    speedup = E / (k·c + 1)                         c = one draft step's cost in target steps

The same formulas as ``minengine.spec`` (``acceptance_rate``, ``expected_tokens``, ``speedup``, ``best_k``),
re-implemented because labs are standalone; ``tests/test_repo_numbers.py`` checks them against that module on
the serving primer's own example (α = 0.6). A draft distilled on the target's own outputs matches the target's
distribution better than an off-the-shelf small model trained on other data, so its α is higher — measured here
on the tiny models (T0) and with vLLM's spec-decode counters (T1), whose "acceptance rate" is *not* α: it is
accepted/drafted = (E − 1)/k, while α is the per-position rate at position 0 (see :func:`vllm_views`).

``speculative_config`` builds and checks the JSON for ``vllm serve --speculative-config`` at v0.30.0 (fields from
``vllm/config/speculative.py`` at that tag; verify when you move versions).
"""
from __future__ import annotations

import json
import random

# SpeculativeConfig fields at vLLM v0.30.0 that this lab uses or checks (vllm_config_speculative.py; verify)
SPEC_FIELDS = {"method", "model", "num_speculative_tokens", "draft_tensor_parallel_size", "quantization",
               "max_model_len", "prompt_lookup_max", "prompt_lookup_min", "parallel_drafting",
               "use_heterogeneous_vocab", "rejection_sample_method", "draft_sample_method",
               "synthetic_acceptance_rates", "synthetic_acceptance_length", "kv_cache_dtype",
               "num_speculative_tokens_per_batch_size"}
METHODS = {"ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class", "eagle", "eagle3", "mtp",
           "ngram_gpu", "dflash", "extract_hidden_states"}


def acceptance_rate(p, q) -> float:
    """α = Σ min(p, q): the chance a token sampled from the draft q survives verification against p."""
    return float(sum(min(a, b) for a, b in zip(p, q)))


def greedy_acceptance(p, q) -> float:
    """The draft proposes argmax q; the target keeps it with probability p(argmax q)."""
    i = max(range(len(q)), key=lambda j: q[j])
    return float(p[i])


def expected_tokens(alpha: float, k: int) -> float:
    """1 + α + … + α^k = (1 − α^(k+1)) / (1 − α): tokens one verify pass emits (k + 1 at α = 1)."""
    return k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    """Tokens per unit of time against plain decoding, when a draft step costs ``c`` target steps."""
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha: float, c: float, k_max: int = 16) -> int:
    return max(range(1, k_max + 1), key=lambda k: speedup(alpha, k, c))


def vllm_views(alpha: float, k: int) -> dict:
    """What vLLM's spec-decode metrics report for an i.i.d. per-token acceptance α with k drafts: the mean
    acceptance length (1 + accepted/drafts) = E, the "Avg Draft acceptance rate" (accepted/drafted) = (E − 1)/k,
    and the per-position rates α^(i+1) (position i counts only when 0..i were all accepted)."""
    e = expected_tokens(alpha, k)
    return {"mean_acceptance_length": e, "acceptance_rate": (e - 1) / k,
            "per_position": [alpha ** (i + 1) for i in range(k)]}


def cost_ratio(draft_params: float, target_params: float) -> float:
    """c for a memory-bound decode step: the draft streams draft_params of weights where the target streams
    target_params (a first approximation — the draft's KV reads and its kernel launches add to it)."""
    return draft_params / target_params


def simulate_counters(alphas: list, k: int, drafts: int, seed: int = 0) -> dict:
    """vLLM's four spec-decode counters after ``drafts`` verify steps, when position i is accepted with
    probability ``alphas[i]`` given 0..i−1 were (a conditional rate per position; pass [α]*k for i.i.d.)."""
    rng = random.Random(seed)
    acc, per = 0, [0] * k
    for _ in range(drafts):
        for i in range(k):
            if rng.random() >= alphas[min(i, len(alphas) - 1)]:
                break
            acc += 1
            per[i] += 1
    return {"drafts": drafts, "draft_tokens": drafts * k, "accepted": acc, "per_position": per}


def counters_text(c: dict, model: str = "Qwen/Qwen3-4B") -> str:
    """Render counters as vLLM's ``/metrics`` exposition (names from vLLM v0.30.0's SpecDecodingProm)."""
    lab = f'engine="0",model_name="{model}"'
    lines = [f"vllm:spec_decode_num_drafts_total{{{lab}}} {float(c['drafts'])}",
             f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {float(c['draft_tokens'])}",
             f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {float(c['accepted'])}"]
    lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{lab},position="{i}"}} {float(v)}'
              for i, v in enumerate(c["per_position"])]
    return "\n".join(lines) + "\n"


# --- vLLM configuration -----------------------------------------------------------------------------------

def speculative_config(model: str | None, k: int = 4, method: str = "draft_model", **extra) -> dict:
    """The ``--speculative-config`` dict, checked against v0.30.0's fields. Raises on the mistakes the fact
    sheet lists: ``tensor_parallel_size`` inside it (vLLM wants ``draft_tensor_parallel_size``), a
    ``speculative_token_tree`` (no such field), an unknown method, k < 1."""
    if "tensor_parallel_size" in extra:
        raise ValueError("'tensor_parallel_size' is not a valid argument in the speculative config; "
                         "pass 'draft_tensor_parallel_size' (1 or the target's)")
    unknown = set(extra) - SPEC_FIELDS
    if unknown:
        raise ValueError(f"not SpeculativeConfig fields at vLLM v0.30.0: {sorted(unknown)}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if k < 1:
        raise ValueError("num_speculative_tokens must be > 0")
    cfg = {"method": method, "num_speculative_tokens": k}
    if model is not None:
        cfg["model"] = model
    cfg.update(extra)
    return cfg


def serve_args(target: str, spec: dict | None, *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.9,
               dtype: str = "auto", port: int = 8000) -> list:
    """``vllm serve`` arguments for a target with (or without) a draft."""
    args = ["vllm", "serve", target, "--dtype", dtype, "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization), "--port", str(port)]
    if spec:
        args += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
    return args


def check_vocab(target_config: dict, draft_config: dict) -> tuple:
    """vLLM's ``draft_model`` check (and TRL GKD's) compares ``vocab_size`` from the configs, not the tokenizers:
    Qwen3-0.6B → Qwen3-4B pass (151,936 both); Qwen2.5-0.5B (151,936) → Qwen2.5-7B (152,064) fails although the
    token ids agree. Returns ``(ok, message)``."""
    t, d = target_config.get("vocab_size"), draft_config.get("vocab_size")
    if t == d:
        return True, f"vocab_size {t} on both"
    return False, (f"Target and draft model should have the same vocabulary size ({t} vs {d}); "
                   "use_heterogeneous_vocab (greedy drafts only) or a draft from the target's family and size class")


def topk_overlap_acceptance(target_top: list, draft_top: list) -> float:
    """A lower bound on α from two top-k log-prob lists at one position (all an API returns): Σ min(p, q) over
    the tokens present in both lists. The mass outside the lists can only add to it."""
    import math
    p = {t: math.exp(lp) for t, lp in target_top}
    q = {t: math.exp(lp) for t, lp in draft_top}
    return sum(min(p[t], q[t]) for t in p.keys() & q.keys())


# --- the tiny models (torch, lazy) --------------------------------------------------------------------------

def tiny_acceptance(draft, target, teacher_seqs) -> list:
    """Per completion position of the target's own samples: mean α = Σ min(p, q) and mean greedy acceptance
    p(argmax q) of ``draft`` against ``target`` (two ``distillab.tinylm`` models). Needs torch."""
    import torch
    import torch.nn.functional as F

    from .tinylm.model import completion_logits
    P, C, M = teacher_seqs
    with torch.no_grad():
        p = F.softmax(completion_logits(target, P, C).float(), -1)
        q = F.softmax(completion_logits(draft, P, C).float(), -1)
    a = torch.minimum(p, q).sum(-1)
    g = p.gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1)
    n = M.sum(0).clamp(min=1)
    return [{"position": i, "alpha": round(((a * M).sum(0) / n)[i].item(), 4),
             "alpha_greedy": round(((g * M).sum(0) / n)[i].item(), 4), "sequences": int(M.sum(0)[i].item())}
            for i in range(C.shape[1])]
