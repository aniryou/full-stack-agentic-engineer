"""
capacity.py — the core LLM-serving capacity math, as ~10 small functions.

Everything about "how many GPUs do I need?" reduces to:
  - two CONSTRAINTS:  memory  and  bandwidth/compute
  - two SLOs:         TTFT (prefill)  and  TPOT (decode)

Each function below is one line of the mental model. No numpy, no classes
beyond three tiny records. Units are stated in every docstring so you can
check them by hand.

Convention for units:
  * memory in GB (10^9 bytes), bandwidth in TB/s (10^12 bytes/s)
    -> 1 TB/s = 1000 GB/s, so seconds = GB / (TB_s * 1000)
  * params in "B" (billions), so a 24B model is params_b = 24
  * FLOPs are raw; TFLOPS = 10^12 FLOP/s
"""

from dataclasses import dataclass

# --- bytes per parameter / per KV element, by number format ------------------
BYTES = {
    "bf16": 2.0,   # ships-as default. treat as "full quality"
    "fp16": 2.0,
    "fp8":  1.0,   # ~free on Hopper+, ~2x speed, tiny quality hit
    "int8": 1.0,
    "int4": 0.5,   # weights-only usually (AWQ/GPTQ)
    "nvfp4": 0.5,  # Blackwell-native 4-bit float, better quality than int4
}


@dataclass
class GPU:
    name: str
    hbm_gb: float          # memory SIZE -> what fits
    bw_tb_s: float         # memory SPEED -> how fast tokens stream out
    bf16_tflops: float     # dense compute -> prefill / high-batch decode
    fp8_tflops: float


# Round numbers worth keeping in your head.
# (Blackwell / MI300X compute are approximate — read the current spec sheet.)
GPUS = {
    "H100":   GPU("H100",   80,  3.35,  990, 1979),
    "H200":   GPU("H200",  141,  4.80,  990, 1979),
    "B200":   GPU("B200",  180,  8.00, 2250, 4500),
    "MI300X": GPU("MI300X",192,  5.30, 1300, 2600),
}


@dataclass
class ModelSpec:
    name: str
    params_b: float        # for a dense model, sizes BOTH memory and compute
    layers: int
    kv_heads: int          # GQA: KV heads (NOT query heads) — this is the lever
    head_dim: int
    active_b: float = None  # MoE: params used per token. dense -> == params_b

    def __post_init__(self):
        if self.active_b is None:
            self.active_b = self.params_b


# Two Mistral models used throughout the primer.
MISTRAL_SMALL = ModelSpec("Mistral Small 3 (24B dense)",
                          params_b=24, layers=40, kv_heads=8, head_dim=128)
MISTRAL_LARGE = ModelSpec("Mistral Large 3 (675B MoE)",
                          params_b=675, layers=88, kv_heads=8, head_dim=128,
                          active_b=41)


# =============================================================================
# 1. WEIGHTS — does it fit?   memory = params * bytes/param
# =============================================================================
def weight_memory_gb(params_b, dtype="bf16"):
    """GB to hold the weights. 24B bf16 -> 48 GB; fp8 -> 24; int4 -> 12."""
    return params_b * BYTES[dtype]


def usable_hbm_gb(gpu, overhead=0.10):
    """Not all HBM is yours — CUDA context, activations, fragmentation.
    ~10% is a safe planning haircut."""
    return gpu.hbm_gb * (1 - overhead)


# =============================================================================
# 2. KV CACHE — how many concurrent users fit? (the number most people miss)
#    per token = 2(K,V) * layers * kv_heads * head_dim * bytes
# =============================================================================
def kv_per_token_kb(model, dtype="bf16"):
    """KB of KV cache per token, per sequence.
    Mistral Small: 2*40*8*128*2 = 163840 B = 160 KB (bf16); 80 KB (fp8).
    Note it uses kv_heads, so GQA (8 vs 32 query heads) already saves 4x."""
    bytes_ = 2 * model.layers * model.kv_heads * model.head_dim * BYTES[dtype]
    return bytes_ / 1024


def kv_per_session_gb(model, context_tokens, dtype="bf16"):
    """GB of KV for ONE conversation of `context_tokens` length."""
    return kv_per_token_kb(model, dtype) * context_tokens / (1024 * 1024)


def max_concurrent_sessions(spare_gb, model, context_tokens, dtype="bf16"):
    """How many simultaneous conversations the leftover HBM holds.
    THIS is what caps concurrency, and it's why fp8 is a 4x concurrency lever
    before it is ever a speed lever."""
    per = kv_per_session_gb(model, context_tokens, dtype)
    return spare_gb / per


# =============================================================================
# 3. DECODE — one token at a time, BANDWIDTH-bound (not FLOP-bound)
#    each step streams all weights (+ the batch's KV) through HBM
# =============================================================================
def decode_step_ms(bytes_gb, gpu):
    """ms to stream `bytes_gb` once from HBM. time = GB / (TB/s * 1000)."""
    return bytes_gb / (gpu.bw_tb_s * 1000) * 1000


def decode_tok_s_single(weight_gb, gpu):
    """Single-stream ceiling: 1 / step_time. 48 GB on H100 -> ~70 tok/s.
    More compute does NOT raise this — only more bandwidth or fewer bytes."""
    return 1000 / decode_step_ms(weight_gb, gpu)


def decode_aggregate(model, gpu, batch, context_tokens, dtype="bf16"):
    """Batched decode: weights are read ONCE per step for the whole batch,
    so throughput scales with batch until you run out of memory (or hit the
    compute roofline ~ batch 300 on H100). Returns (agg_tok_s, per_user_tok_s)."""
    w = weight_memory_gb(model.params_b, dtype)          # MoE caveat: see primer
    kv = batch * kv_per_session_gb(model, context_tokens, dtype)
    step_ms = decode_step_ms(w + kv, gpu)
    agg = batch / (step_ms / 1000)
    return agg, agg / batch


def roofline_batch(gpu):
    """Batch at which decode flips from bandwidth-bound to compute-bound.
    = arithmetic intensity of the GPU = FLOPs per byte. H100 ~ 295."""
    return (gpu.bf16_tflops * 1e12) / (gpu.bw_tb_s * 1e12)


# =============================================================================
# 4. PREFILL — read the whole prompt in parallel, COMPUTE-bound
#    FLOPs ~ 2 * params * prompt_tokens
# =============================================================================
def prefill_flops(active_b, prompt_tokens):
    """For MoE use ACTIVE params here — prefill costs like a dense `active_b`."""
    return 2 * active_b * 1e9 * prompt_tokens


def ttft_s(active_b, prompt_tokens, gpu, dtype="fp8", mfu=0.5):
    """Time to first token. 24B * 2K prompt on H100 ~ 0.2 s.
    32K RAG prompt ~ 2-3 s -> prefix caching is the big win for RAG/agents."""
    tflops = gpu.fp8_tflops if dtype in ("fp8", "int8") else gpu.bf16_tflops
    return prefill_flops(active_b, prompt_tokens) / (tflops * 1e12 * mfu)


def prefill_tok_s(active_b, gpu, dtype="fp8", mfu=0.5):
    """Prefill tokens/sec one GPU can chew = compute_budget / flops_per_token."""
    tflops = gpu.fp8_tflops if dtype in ("fp8", "int8") else gpu.bf16_tflops
    return (tflops * 1e12 * mfu) / (2 * active_b * 1e9)


# =============================================================================
# 5. WORKLOAD -> concurrency  (Little's Law turns "N users" into "K live convos")
# =============================================================================
def request_duration_s(active_b, in_tokens, out_tokens, gpu, tpot_ms=40,
                       dtype="fp8", mfu=0.5):
    """How long one request lives = prefill + generation."""
    return ttft_s(active_b, in_tokens, gpu, dtype, mfu) + out_tokens * tpot_ms / 1000


def concurrency(rps, duration_s):
    """Little's Law: live requests = arrival rate * time each one lasts."""
    return rps * duration_s


def provision(raw_gpus, util=0.7, redundancy=1):
    """Never run at 100%. Size for headroom, then add N+redundancy spares."""
    import math
    active = math.ceil(raw_gpus / util)
    return active + redundancy
