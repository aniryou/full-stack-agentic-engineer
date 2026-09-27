"""The economics of a student: a fixed cost to make it, a lower cost per token to run it, a break-even.

The one idea: decode streams the weights and the KV cache every step, so a student with a twentieth of the
parameters and a ninth of the KV per token fits a far bigger batch under the same latency budget and costs
far less per token ($/M = $/GPU-h ÷ (tokens/s × 3600) × 10⁶, `roofline.cost.cost_per_million_tokens`).
Against that saving stands a one-off bill: the teacher's tokens at their price, plus the student's training,
≈ 6·N·D FLOPs (transformer primer §6.2) at some MFU. Break-even is that bill over the saving per token; a
cascade (student first, teacher when needed) sits between the two. Compare like with like: a 32B teacher on one
80 GB GPU has almost no room left for KV and cannot batch, so price it on the tensor-parallel group you would
really run (`tp_group`). The decode step below restates `roofline.llm.decode` for dense models, and the memory
helpers restate `capacity.py`, so the numbers match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    """The config.json fields that set bytes and FLOPs (dense decoder, SwiGLU MLP)."""
    name: str
    layers: int
    d: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    tied: bool = False

    def matmul_params(self) -> int:
        attn = 2 * self.d * self.heads * self.head_dim + 2 * self.d * self.kv_heads * self.head_dim
        return self.layers * (attn + 3 * self.d * self.ffn)

    def params(self) -> int:
        return self.matmul_params() + self.vocab * self.d * (1 if self.tied else 2)

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bytes


# Published configs (verify against each model's config.json; counts exclude norms and biases, as roofline.llm).
SHAPES = {s.name: s for s in [
    Shape("qwen2.5-0.5b", 24, 896, 14, 2, 64, 4864, 151_936, tied=True),
    Shape("qwen2.5-1.5b", 28, 1536, 12, 2, 128, 8960, 151_936, tied=True),
    Shape("qwen2.5-32b", 64, 5120, 40, 8, 128, 27648, 152_064),
    Shape("qwen3-0.6b", 28, 1024, 16, 8, 128, 3072, 151_936, tied=True),
    Shape("qwen3-4b", 36, 2560, 32, 8, 128, 9728, 151_936, tied=True),
    Shape("llama-3.1-8b", 32, 4096, 32, 8, 128, 14336, 128_256),
]}


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    tb_s: float
    tflops: dict


# roofline.specs' figures (dense peaks, verify): a T4 has no bf16 path, so pass precision="fp16" there.
GPUS = {"t4": GPU("T4", 16, 0.32, {"fp16": 65}), "l4": GPU("L4", 24, 0.30, {"fp16": 121, "bf16": 121, "fp8": 242.5}),
        "h100": GPU("H100 SXM", 80, 3.35, {"fp16": 989.4, "bf16": 989.4, "fp8": 1978.9})}


# -- decode on the roofline ---------------------------------------------------------------------------------
def decode_step(m: Shape, g: GPU, batch: int, context: int, wbytes: float = 2, kvbytes: float = 2,
                precision: str = "bf16") -> float:
    """Seconds for one decode step (ideal bound): max(FLOPs / peak, bytes / bandwidth)."""
    lm_head = m.vocab * m.d
    nbytes = (m.matmul_params() + lm_head + batch * m.d) * wbytes + batch * (context + 1) * m.kv_bytes_per_token(kvbytes)
    flops = batch * (2 * m.matmul_params() + 2 * lm_head + 4 * m.layers * m.heads * m.head_dim * (context + 1))
    return max(flops / (g.tflops[precision] * 1e12), nbytes / (g.tb_s * 1e12))


def max_batch(m: Shape, g: GPU, context: int, wbytes: float = 2, kvbytes: float = 2, reserve: float = 0.10) -> int:
    """Sequences of `context` tokens whose KV fits beside the weights, with 10% headroom."""
    free = g.memory_gb * 1e9 * (1 - reserve) - m.params() * wbytes
    return max(0, math.floor(free / (context * m.kv_bytes_per_token(kvbytes))))


def best_batch(m: Shape, g: GPU, context: int, itl_s: float, **kw) -> int:
    """Largest batch whose step fits the ITL budget and memory (0 if even batch 1 misses the budget)."""
    mem = {k: v for k, v in kw.items() if k in ("wbytes", "kvbytes")}
    best = 0
    for b in range(1, max_batch(m, g, context, **mem) + 1):
        if decode_step(m, g, b, context, **kw) > itl_s:
            break
        best = b
    return best


def cost_per_million_tokens(price_per_gpu_hour: float, tokens_per_s: float, utilisation: float = 1.0,
                            n_gpus: int = 1) -> float:
    """$/M tokens = $/GPU-h × GPUs / (tokens/s × 3600 × utilisation) × 10⁶ (roofline.cost)."""
    return price_per_gpu_hour * n_gpus / (tokens_per_s * 3600 * utilisation) * 1e6


def tp_group(g: GPU, n: int) -> GPU:
    """n GPUs as one ideal tensor-parallel device: n× the memory, bandwidth and FLOPs, the all-reduces not counted
    (the roofline primer §5.3 prices them with α-β) — an upper bound, like everything here. Its memory matches
    `roofline.llm.max_batch_by_memory(..., n_devices=n)`."""
    return g if n == 1 else GPU(f"{n}×{g.name}", g.memory_gb * n, g.tb_s * n, {k: v * n for k, v in g.tflops.items()})


def serving(m: Shape, g: GPU, price_per_gpu_hour: float, context: int, itl_s: float, n_gpus: int = 1, **kw) -> dict:
    """The batch the ITL budget allows, its step, throughput and $/M output tokens on `n_gpus` GPUs (ideal TP)."""
    grp = tp_group(g, n_gpus)
    b = best_batch(m, grp, context, itl_s, **kw)
    if b == 0:
        return {"batch": 0, "step_s": decode_step(m, grp, 1, context, **kw), "tok_s": 0.0, "usd_per_m": math.inf,
                "n_gpus": n_gpus}
    t = decode_step(m, grp, b, context, **kw)
    return {"batch": b, "step_s": t, "tok_s": b / t, "n_gpus": n_gpus,
            "usd_per_m": cost_per_million_tokens(price_per_gpu_hour, b / t, n_gpus=n_gpus)}


# -- the capacity primer's memory view (capacity.py, GB = 1e9 bytes) ----------------------------------------
def weight_gb(params: float, bytes_per_param: float = 2) -> float:
    return params * bytes_per_param / 1e9


def kv_per_token_kb(layers: int, kv_heads: int, head_dim: int, bytes_per: float = 2) -> float:
    return 2 * layers * kv_heads * head_dim * bytes_per / 1e3


def sessions_per_gpu(hbm_gb: float, weights_gb: float, kv_kb: float, context: int, overhead: float = 0.10) -> float:
    """capacity.max_concurrent_sessions: (usable HBM − weights) / KV per session."""
    return (hbm_gb * (1 - overhead) - weights_gb) / (kv_kb * context / 1e6)


def decode_tok_s_single(weights_gb: float, tb_s: float) -> float:
    """capacity.decode_tok_s_single: one stream is bounded by the weight read, 1000 / (GB / (TB/s·1000) · 1000)."""
    return tb_s * 1000 / weights_gb


# -- the fixed cost of making a student ----------------------------------------------------------------------
PRICES = {"gemini-3.5-flash": (1.50, 9.00, 0.15), "gemini-3.5-flash-lite": (0.30, 2.50, 0.03)}   # $/M in/out/cached, 2026-09-05 (verify)


def api_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for one call at list prices (the 06 scaling lab's cost_per_call)."""
    i, o, c = PRICES[model]
    return ((input_tokens - cached_tokens) * i + cached_tokens * c + output_tokens * o) / 1e6


def training_flops(params: float, tokens: float, per_token: float = 6) -> float:
    """≈ 6·N·D: 2 per parameter per token forward, 4 backward (transformer primer §6.2). A frozen teacher's
    forward pass over the same tokens adds 2·N_T·D (logit KD, on-policy scoring)."""
    return per_token * params * tokens


def gpu_hours(flops: float, g: GPU, mfu: float = 0.4, precision: str = "bf16") -> float:
    return flops / (g.tflops[precision] * 1e12 * mfu) / 3600


def fixed_cost(prompts: int, samples: int, tokens_per_sample: int, teacher_usd_per_m: float, student_params: float,
               g: GPU, gpu_price: float, mfu: float = 0.4, epochs: float = 1.0) -> dict:
    """Teacher generation (tokens × price) + student SFT (6·N·D at an MFU) — the one-off bill."""
    tokens = prompts * samples * tokens_per_sample
    flops = training_flops(student_params, tokens * epochs)
    hours = gpu_hours(flops, g, mfu)
    gen, train = tokens / 1e6 * teacher_usd_per_m, hours * gpu_price
    return {"tokens": tokens, "generation_usd": gen, "train_flops": flops, "gpu_hours": hours,
            "train_usd": train, "total_usd": gen + train}


def break_even(fixed_usd: float, teacher_usd_per_m: float, student_usd_per_m: float, tokens_per_day: float) -> dict:
    """Tokens (and days at a daily volume) after which the student has paid for itself."""
    saving = teacher_usd_per_m - student_usd_per_m
    tokens = math.inf if saving <= 0 else fixed_usd / saving * 1e6
    return {"tokens": tokens, "days": tokens / tokens_per_day, "saving_per_day": saving * tokens_per_day / 1e6}


def cascade(cost_s: float, cost_t: float, acc_s: tuple, acc_t: tuple, hard: float, catch: float = 1.0,
            false_alarm: float = 0.0, student_first: bool = True) -> dict:
    """Route by difficulty: acc_s, acc_t = (accuracy on easy, accuracy on hard); `hard` is the hard share.
    A gate sends `catch` of hard and `false_alarm` of easy requests to the teacher. student_first: the student
    always runs and the gate escalates (its cost is paid twice); otherwise a router decides up front."""
    to_t = hard * catch + (1 - hard) * false_alarm
    cost = (cost_s if student_first else (1 - to_t) * cost_s) + to_t * cost_t
    acc = (hard * (catch * acc_t[1] + (1 - catch) * acc_s[1])
           + (1 - hard) * (false_alarm * acc_t[0] + (1 - false_alarm) * acc_s[0]))
    return {"to_teacher": to_t, "cost": cost, "accuracy": acc, "cost_per_correct": cost / acc}
