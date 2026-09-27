"""cost.py — is the student worth it? Serving cost per million tokens, the fixed cost of distilling, break-even.

One idea: distillation is a fixed cost paid once (the teacher's generation tokens plus the student's training
GPU-hours) against a saving per token served for as long as the student is in use (PRIMER §9 "The economics of a
student"). Every piece is a formula the repo already has:

    decode step        max(FLOPs / peak, bytes / bandwidth), bytes = weights streamed once + every sequence's KV
                       (01 PRIMER §3 "LLM inference on the roofline"; ``roofline.llm.decode``)
    $/M tokens         price × GPUs / (tokens/s × 3600 × utilisation) × 1e6   (01 PRIMER §8; ``roofline.cost``)
    training FLOPs     6 · N · D  (2 per parameter per token forward, 4 backward — the transformer primer §6)
    teacher scoring    2 · N_teacher · D for every token a logit-KD or GKD step shows the teacher
    break-even         fixed cost / (teacher $/M − student $/M) → million tokens → days at a daily volume
    cascade            student for easy queries, teacher for the rest: blended $/M and accuracy

The roofline functions are re-implemented for Hugging Face ``config.json`` shapes (labs are standalone) and
``tests/test_repo_numbers.py`` checks them against ``roofline.llm`` and ``roofline.cost``. They are *bounds* at
ideal bandwidth, labelled PREDICTED; at T1 :func:`from_metrics` replaces the throughput with what vLLM's
``/metrics`` measured. GPU figures are the roofline core's (dense peaks, verify); prices are COMPUTE.md's,
dated 2026-09-26 (verify).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources

from . import metrics as M


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float          # as marketed (roofline.specs convention)
    bw_tbs: float             # memory bandwidth, TB/s
    tflops: dict              # dense peak TFLOP/s by precision
    price_hr: float           # USD per GPU-hour, on-demand, COMPUTE.md 2026-09-26 (verify)
    spot_hr: float | None = None


GPUS = {   # the roofline core's specs (roofline/specs.py: t4, l4, h100-sxm); prices from COMPUTE.md (verify)
    "T4": GPU("T4", 16, 0.32, {"fp32": 8.1, "fp16": 65}, 0.35),
    "L4": GPU("L4", 24, 0.30, {"fp16": 121, "bf16": 121, "fp8": 242.5}, 0.70),
    "H100": GPU("H100", 80, 3.35, {"fp16": 989.4, "bf16": 989.4, "fp8": 1978.9}, 11.0, 3.7),
}


@dataclass(frozen=True)
class Shape:
    """The config.json fields that set a decoder's FLOPs and bytes (norms and biases ignored, as roofline.llm)."""
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tied: bool = False

    @classmethod
    def from_hf(cls, cfg: dict, name: str | None = None) -> "Shape":
        heads = int(cfg["num_attention_heads"])
        return cls(name or cfg.get("_name_or_path", "model"), int(cfg["num_hidden_layers"]), int(cfg["hidden_size"]),
                   heads, int(cfg.get("num_key_value_heads") or heads),
                   int(cfg.get("head_dim") or cfg["hidden_size"] // heads), int(cfg["intermediate_size"]),
                   int(cfg["vocab_size"]), bool(cfg.get("tie_word_embeddings", False)))

    def attn(self) -> int:
        return 2 * self.d_model * self.n_heads * self.head_dim + 2 * self.d_model * self.n_kv_heads * self.head_dim

    def matmul_params(self) -> int:
        return self.n_layers * (self.attn() + 3 * self.d_model * self.d_ff)

    def params(self) -> int:
        return self.matmul_params() + self.vocab * self.d_model * (1 if self.tied else 2)

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * kv_bytes


def load_config(name: str) -> dict:
    """A bundled config.json (key fields; verify): qwen2.5-0.5b-instruct, qwen2.5-1.5b-instruct, qwen2.5-7b-instruct,
    qwen2.5-32b-instruct, qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, deepseek-r1-distill-qwen-1.5b."""
    return json.loads(resources.files("distillab.assets").joinpath("configs", f"{name}.json").read_text())


def shape(name: str) -> Shape:
    return Shape.from_hf(load_config(name), name)


def decode_step(m: Shape, gpu: GPU, batch: int, context: int, *, weight_bytes: float = 2, kv_bytes: float = 2,
                precision: str = "bf16") -> float:
    """Seconds for one decode step of ``batch`` sequences at ``context`` cached tokens (ideal roofline bound)."""
    if precision not in gpu.tflops:
        raise KeyError(f"{gpu.name} has no {precision} path (it has: {', '.join(gpu.tflops)})")
    streamed = (m.matmul_params() + m.vocab * m.d_model + batch * m.d_model) * weight_bytes
    kv = batch * (context + 1) * m.kv_bytes_per_token(kv_bytes)
    flops = batch * (2 * m.matmul_params() + 2 * m.vocab * m.d_model
                     + 4 * m.n_layers * m.n_heads * m.head_dim * (context + 1))
    return max(flops / (gpu.tflops[precision] * 1e12), (streamed + kv) / (gpu.bw_tbs * 1e12))


def max_batch_by_memory(m: Shape, gpu: GPU, context: int, *, weight_bytes: float = 2, kv_bytes: float = 2,
                        reserve: float = 0.10) -> int:
    free = gpu.memory_gb * 1e9 * (1 - reserve) - m.params() * weight_bytes
    return max(0, math.floor(free / (context * m.kv_bytes_per_token(kv_bytes))))


def best_batch(m: Shape, gpu: GPU, context: int, itl_s: float, **kw) -> int:
    """Largest batch whose decode step fits the ITL budget and memory (0 if even batch 1 misses the ITL)."""
    cap = max_batch_by_memory(m, gpu, context, **{k: v for k, v in kw.items() if k in ("weight_bytes", "kv_bytes")})
    best = 0
    for b in range(1, cap + 1):
        if decode_step(m, gpu, b, context, **kw) > itl_s:
            break
        best = b
    return best


def cost_per_million_tokens(price_per_gpu_hour: float, tokens_per_s: float, utilisation: float = 1.0,
                            n_gpus: int = 1) -> float:
    """$ per 1e6 tokens: price × GPUs / (tokens/s × 3600 × utilisation) × 1e6 (``roofline.cost``)."""
    if tokens_per_s <= 0:
        raise ValueError("no throughput: the model does not meet the ITL at any batch on this GPU")
    return price_per_gpu_hour * n_gpus / (tokens_per_s * 3600 * utilisation) * 1e6


def tp_group(gpu: GPU, n: int) -> GPU:
    """``n`` GPUs as one ideal tensor-parallel device: n× memory, bandwidth and FLOPs, all-reduces not counted (01
    PRIMER §5.3 prices them). The per-GPU price stays per GPU; :func:`serving` multiplies by ``n``."""
    if n == 1:
        return gpu
    return GPU(f"{n}x{gpu.name}", gpu.memory_gb * n, gpu.bw_tbs * n, {k: v * n for k, v in gpu.tflops.items()},
               gpu.price_hr, gpu.spot_hr)


def serving(m: Shape, gpu: GPU, *, context: int = 2048, itl_s: float = 0.030, utilisation: float = 1.0,
            price: float | None = None, precision: str = "bf16", n_gpus: int = 1) -> dict:
    """The serving row of one model on ``n_gpus`` (ideal TP): the batch under an ITL (capped by memory), tokens/s
    and $/M (PREDICTED). Price a big teacher on the GPUs you would really give it: a 32B on one 80 GB card has
    almost no room for KV and cannot batch, which flatters any student compared with it."""
    grp = tp_group(gpu, n_gpus)
    b = best_batch(m, grp, context, itl_s, precision=precision)
    if b == 0:
        return {"model": m.name, "gpu": grp.name, "batch": 0, "step ms": round(1e3 * decode_step(m, grp, 1, context, precision=precision), 2),
                "tok/s": 0.0, "$/M": math.inf, "note": "misses the ITL at batch 1"}
    step = decode_step(m, grp, b, context, precision=precision)
    tps = b / step
    return {"model": m.name, "gpu": grp.name, "batch": b, "step ms": round(step * 1e3, 2), "tok/s": round(tps),
            "$/M": cost_per_million_tokens(price if price is not None else gpu.price_hr, tps, utilisation, n_gpus)}


# --- the fixed cost of distilling ------------------------------------------------------------------------

def train_flops(n_params: float, tokens: float) -> float:
    """6 · N · D: 2 per parameter per token forward, 4 backward (transformer primer §6)."""
    return 6 * n_params * tokens


def teacher_flops(n_teacher: float, tokens: float) -> float:
    """2 · N · D: one forward pass of a frozen teacher over the tokens (logit KD, GKD scoring)."""
    return 2 * n_teacher * tokens


def gpu_hours(flops: float, gpu: GPU, mfu: float = 0.4, precision: str = "bf16") -> float:
    return flops / (gpu.tflops[precision] * 1e12 * mfu) / 3600


def fixed_cost(*, teacher_tokens: float, teacher_price_per_m: float, student_params: float, train_tokens: float,
               gpu: GPU, mfu: float = 0.4, gpu_price: float | None = None, teacher_params: float = 0.0,
               teacher_forward_tokens: float = 0.0, precision: str = "bf16") -> dict:
    """What distilling costs once: the teacher's generated tokens at their price, plus training the student
    (6·N·D) and any teacher forward passes during training (2·N_T·D; logit KD and GKD) on ``gpu`` at ``mfu``."""
    price = gpu.price_hr if gpu_price is None else gpu_price
    gen = teacher_tokens * teacher_price_per_m / 1e6
    fl = train_flops(student_params, train_tokens) + teacher_flops(teacher_params, teacher_forward_tokens)
    hours = gpu_hours(fl, gpu, mfu, precision)
    return {"teacher generation $": gen, "training FLOPs": fl, "training GPU-hours": hours,
            "training $": hours * price, "total $": gen + hours * price}


def break_even(fixed: float, teacher_per_m: float, student_per_m: float, tokens_per_day: float) -> dict:
    """Million tokens (and days at ``tokens_per_day``) after which the student has paid for itself."""
    saving = teacher_per_m - student_per_m
    if saving <= 0:
        return {"million tokens": math.inf, "days": math.inf}
    m = fixed / saving
    return {"million tokens": m, "days": m * 1e6 / tokens_per_day}


def cascade(route_to_teacher: float, student_per_m: float, teacher_per_m: float, *, acc_student_kept: float,
            acc_teacher_routed: float, student_first: bool = True) -> dict:
    """A share of queries goes to the teacher. ``student_first``: every query runs on the student and the
    routed share runs again on the teacher (escalation after an answer, e.g. on low confidence); otherwise a
    router decides before either runs. Accuracies are those of each model on the queries it answers."""
    f = route_to_teacher
    per_m = student_per_m + f * teacher_per_m if student_first else (1 - f) * student_per_m + f * teacher_per_m
    acc = (1 - f) * acc_student_kept + f * acc_teacher_routed
    return {"routed to teacher": f, "$/M": per_m, "accuracy": acc, "$/M per unit accuracy": per_m / acc if acc else math.inf}


def cost_per_correct(cost_per_query: float, accuracy: float) -> float:
    """What one right answer costs: failures are paid for too (rl-and-thinking-models PRIMER §7)."""
    return math.inf if accuracy <= 0 else cost_per_query / accuracy


def from_metrics(before: M.Scrape, after: M.Scrape, price_per_gpu_hour: float, *, seconds: float | None = None,
                 utilisation: float = 1.0, n_gpus: int = 1) -> dict:
    """T1: $/M from the output tokens vLLM counted between two scrapes (MEASURED throughput)."""
    t = M.throughput(before, after, seconds)
    return {**t, "$/M": cost_per_million_tokens(price_per_gpu_hour, t["output_tok_s"], utilisation, n_gpus)}
