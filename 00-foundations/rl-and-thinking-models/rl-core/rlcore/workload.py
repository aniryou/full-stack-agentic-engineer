"""What thinking does to serving: output-heavy, decode-bound, heavy-tailed and hungry for KV cache.

The one idea: a request that thinks for L tokens holds a growing KV for L decode steps — P·L + L(L+1)/2
KV-token-steps, quadratic in output — so ten times the output is ~10× the concurrency *and* ~2× the KV per
session: memory, not FLOPs, sets the GPU count, and a tight ITL SLO caps the batch before HBM does. The sizing
formulas are the capacity primer's (00-foundations/gpu-capacity-planning/capacity.py), restated so this
package stands alone; tests/test_workload.py reproduces that primer's numbers. Added: the batch capped by HBM
and ITL (capacity.decode_aggregate caps neither), heavy tails, budgets, prefix reuse, cost per correct answer.
"""
from __future__ import annotations

import itertools
import math
import statistics
from collections import namedtuple
from dataclasses import dataclass

BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


GPU = namedtuple("GPU", "name hbm_gb bw_tb_s bf16_tflops fp8_tflops")   # GB, TB/s, dense TFLOP/s


@dataclass
class Model:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    active_b: float | None = None

    def __post_init__(self):
        self.active_b = self.params_b if self.active_b is None else self.active_b


# Datasheet-level numbers (verify). H100 and Mistral Small are the capacity primer's; T4 has no FP8.
GPUS = {"H100": GPU("H100", 80, 3.35, 990, 1979), "L4": GPU("L4", 24, 0.30, 121, 242),
        "T4": GPU("T4", 16, 0.32, 65, 65)}
MISTRAL_SMALL = Model("Mistral Small 3 (24B dense)", 24, 40, 8, 128)
QWEN3_0_6B = Model("Qwen3-0.6B", 0.596, 28, 8, 128)


# -- the capacity primer's formulas, same units (weights in 1e9 bytes, KV in KB/1024², as capacity.py) --
def weight_gb(model: Model, dtype: str = "bf16") -> float:
    return model.params_b * BYTES[dtype]


def kv_per_token_kb(model: Model, dtype: str = "bf16") -> float:
    return 2 * model.layers * model.kv_heads * model.head_dim * BYTES[dtype] / 1024


def kv_per_session_gb(model: Model, context_tokens: float, dtype: str = "bf16") -> float:
    return kv_per_token_kb(model, dtype) * context_tokens / (1024 * 1024)


def _tflops(gpu: GPU, dtype: str) -> float:
    return gpu.fp8_tflops if dtype in ("fp8", "int8") else gpu.bf16_tflops


def ttft_s(active_b: float, prompt_tokens: int, gpu: GPU, dtype: str = "fp8", mfu: float = 0.5) -> float:
    return 2 * active_b * 1e9 * prompt_tokens / (_tflops(gpu, dtype) * 1e12 * mfu)


def prefill_tok_s(active_b: float, gpu: GPU, dtype: str = "fp8", mfu: float = 0.5) -> float:
    return _tflops(gpu, dtype) * 1e12 * mfu / (2 * active_b * 1e9)


def request_duration_s(active_b, in_tokens, out_tokens, gpu, tpot_ms=40, dtype="fp8", mfu=0.5) -> float:
    return ttft_s(active_b, in_tokens, gpu, dtype, mfu) + out_tokens * tpot_ms / 1000


def sessions_per_gpu(model: Model, gpu: GPU, context_tokens: float, dtype: str = "bf16", overhead: float = 0.10) -> float:
    """(usable HBM − weights) / KV per session — capacity.max_concurrent_sessions on usable_hbm_gb."""
    return (gpu.hbm_gb * (1 - overhead) - weight_gb(model, dtype)) / kv_per_session_gb(model, context_tokens, dtype)


def decode_tok_s(model: Model, gpu: GPU, batch: int, context_tokens: float, dtype: str = "bf16") -> float:
    """capacity.decode_aggregate's aggregate: one step streams the weights once plus every session's KV."""
    step_s = (weight_gb(model, dtype) + batch * kv_per_session_gb(model, context_tokens, dtype)) / (gpu.bw_tb_s * 1000)
    return batch / step_s


# -- added: the step on the roofline, and a batch capped by HBM and by the ITL SLO ----------------------
def decode_step_s(model, gpu, batch, context_tokens, dtype="bf16", mfu=0.5) -> float:
    """max(bytes / bandwidth, FLOPs / (peak · mfu)) for one decode step of `batch` sequences."""
    t_mem = (weight_gb(model, dtype) + batch * kv_per_session_gb(model, context_tokens, dtype)) / (gpu.bw_tb_s * 1000)
    t_compute = batch * 2 * model.active_b * 1e9 / (_tflops(gpu, dtype) * 1e12 * mfu)
    return max(t_mem, t_compute)


def max_batch_for_itl(model, gpu, context_tokens, itl_s, dtype="bf16", mfu=0.5) -> int:
    """The largest batch whose decode step still meets the ITL SLO (0 if even batch 1 misses it)."""
    b = 0
    while decode_step_s(model, gpu, b + 1, context_tokens, dtype, mfu) <= itl_s and b < 100_000:
        b += 1
    return b


def plan(model: Model, gpu: GPU, rps: float, in_tokens: int, out_tokens: int, tpot_ms: float = 40,
         dtype: str = "fp8", mfu: float = 0.5) -> dict:
    """The capacity primer's sizing (Little's law, then GPUs per constraint) with the batch per GPU capped by
    HBM and by the ITL SLO. Durations assume every token takes the SLO's TPOT, as the primer does."""
    duration = request_duration_s(model.active_b, in_tokens, out_tokens, gpu, tpot_ms, dtype, mfu)
    conc = rps * duration
    ctx = in_tokens + out_tokens // 2                 # KV holds the prompt plus half the output, on average
    mem = sessions_per_gpu(model, gpu, ctx, dtype)
    itl = max_batch_for_itl(model, gpu, ctx, tpot_ms / 1000, dtype, mfu)
    cap = min(int(mem), itl)
    batch = max(min(cap, int(conc)), 1)
    per_gpu = batch / decode_step_s(model, gpu, batch, ctx, dtype, mfu)
    need = {"memory": conc / mem, "itl_slots": conc / max(itl, 1e-9),
            "decode": rps * out_tokens / per_gpu, "prefill": rps * in_tokens / prefill_tok_s(model.active_b, gpu, dtype, mfu)}
    binding = max(need, key=need.get)
    return {"duration_s": duration, "concurrency": conc, "avg_ctx": ctx, "sessions_per_gpu": mem,
            "itl_batch": itl, "batch": batch, "decode_tok_s_per_gpu": per_gpu, "gpus": need,
            "binding": binding, "gpus_needed": math.ceil(need[binding] - 1e-9)}


def rl_step_time(model, gpu, n_gpus, prompt_tokens, out_lengths, dtype="bf16", mfu=0.5, mfu_train=0.4,
                 scoring_passes: int = 2) -> dict:
    """One synchronous RL step on n_gpus: every rollout decodes at once (rollouts split evenly across the
    GPUs, each engine holding its share), and the step lasts until the *longest* completion ends; then one
    forward + backward pass (6·N FLOPs per token) and `scoring_passes` log-prob passes (π_old, π_ref: 2·N
    each) over every token. Returns generation and training seconds and how busy the generation batch stayed."""
    lengths = sorted(int(x) for x in out_lengths)
    per_gpu, gen, done, alive_steps = len(lengths) / n_gpus, 0.0, 0, 0.0
    for t in range(lengths[-1]):
        while lengths[done] <= t:
            done += 1
        alive = (len(lengths) - done) / n_gpus
        gen += decode_step_s(model, gpu, max(alive, 1e-9), prompt_tokens + t, dtype, mfu)
        alive_steps += alive
    tokens = len(lengths) * prompt_tokens + sum(lengths)
    train = (6 + 2 * scoring_passes) * model.active_b * 1e9 * tokens / (_tflops(gpu, dtype) * 1e12 * mfu_train * n_gpus)
    return {"generate_s": gen, "train_s": train, "rollout_fraction": gen / (gen + train),
            "batch_occupancy": alive_steps / (lengths[-1] * per_gpu)}


# -- per-request shape --------------------------------------------------------------------------------
def kv_token_steps(prompt_tokens: int, out_tokens: int) -> int:
    """Σ over decode steps of the tokens held in KV: P·L + L(L+1)/2 — the memory × time one request costs."""
    return prompt_tokens * out_tokens + out_tokens * (out_tokens + 1) // 2


_N = statistics.NormalDist()


def lognormal_quantile(median: float, sigma: float, p: float) -> float:
    return median * math.exp(sigma * _N.inv_cdf(p))


def lognormal_mean(median: float, sigma: float) -> float:
    return median * math.exp(sigma ** 2 / 2)


def budget_outcome(median: float, sigma: float, answer_tokens: int = 300, e0: float = 1.0,
                   max_tokens: int | None = None, budget: int | None = None) -> dict:
    """Each question needs L_req ~ lognormal(median, sigma) thinking tokens and the model stops when done.
    max_tokens (counts thinking + answer) truncates: no answer. budget forcing ends the thinking at `budget`
    and answers anyway (right with 1 − e0 when uncracked). Returns accuracy and mean output tokens."""
    mu = math.log(median)
    cut = budget if budget is not None else (max_tokens - answer_tokens if max_tokens is not None else None)
    if cut is None:
        return {"accuracy": 1.0, "tokens": lognormal_mean(median, sigma) + answer_tokens, "truncated": 0.0}
    done = _N.cdf((math.log(cut) - mu) / sigma)          # P(L_req ≤ cut)
    think = lognormal_mean(median, sigma) * _N.cdf((math.log(cut) - mu - sigma ** 2) / sigma) + cut * (1 - done)
    if budget is not None:                            # forced: think min(L_req, budget), then answer
        return {"accuracy": done + (1 - e0) * (1 - done), "tokens": think + answer_tokens, "truncated": 0.0}
    # truncated requests think on through all of max_tokens (= cut + answer_tokens), so the mean is the same
    return {"accuracy": done, "tokens": think + answer_tokens, "truncated": 1 - done}


def turn_prefills(system: int, turns, keep_thinking: bool = False, block: int = 16, header: int = 4):
    """Per turn (prompt tokens, tokens served from the prefix cache) for a chat whose earlier turns' thinking
    is dropped by the template (keep_thinking=False) or kept (inside one tool loop). turns: (user, think,
    answer) token counts. Hits are full blocks only and stop before the last prompt token (04 PRIMER §5)."""
    ids = itertools.count()
    seg = lambda n: [next(ids) for _ in range(n)]
    hist, held, out = seg(system), [], []
    for user, think, answer in turns:
        prompt = hist + seg(user + header)            # the user message, then the assistant header
        common = next((i for i, (a, b) in enumerate(zip(held, prompt)) if a != b), min(len(held), len(prompt)))
        out.append((len(prompt), min(common, len(prompt) - 1) // block * block))
        t, a = seg(think), seg(answer)
        held = prompt + t + a                         # what the engine computed KV for this turn
        hist = prompt + (t if keep_thinking else []) + a
    return out


def api_cost(input_tokens: int, output_tokens: int, cached_tokens: int = 0, usd_in=1.50, usd_out=9.00,
             usd_cached=0.15) -> float:
    """Dollars per call at per-1M-token prices (the 06 scaling lab's cost_per_call; its default row, verify)."""
    return ((input_tokens - cached_tokens) * usd_in + cached_tokens * usd_cached + output_tokens * usd_out) / 1e6


def cost_per_correct(cost_per_request: float, accuracy: float) -> float:
    """What a right answer costs: wrong answers are paid for too."""
    return cost_per_request / accuracy
