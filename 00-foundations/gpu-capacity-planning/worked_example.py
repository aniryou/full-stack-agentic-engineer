"""
worked_example.py — run the whole primer end to end and print the numbers.

    python worked_example.py

Two scenarios:
  A. Mistral Small 3 (24B) on one H100 — fits, so the question is concurrency.
  B. A Singapore bank, on-prem, internal assistant — how many GPUs?
  C. Mistral Large 3 (675B MoE) — when one GPU (or one node) won't do.
"""

import math
from capacity import (
    BYTES, GPUS, MISTRAL_SMALL as SMALL, MISTRAL_LARGE as LARGE,
    weight_memory_gb, usable_hbm_gb, kv_per_token_kb, kv_per_session_gb,
    max_concurrent_sessions, decode_tok_s_single, decode_aggregate,
    roofline_batch, ttft_s, prefill_tok_s, request_duration_s,
    concurrency, provision,
)

H100 = GPUS["H100"]
H200 = GPUS["H200"]
line = lambda: print("-" * 68)


# ---------------------------------------------------------------- A ----------
print("=" * 68)
print("A. MISTRAL SMALL 3 (24B dense) ON ONE H100")
print("=" * 68)

for dt in ("bf16", "fp8", "int4"):
    print(f"  weights @ {dt:<5}: {weight_memory_gb(SMALL.params_b, dt):5.0f} GB")

usable = usable_hbm_gb(H100)
print(f"  usable HBM (H100, -10%): {usable:.0f} GB")
for dt in ("bf16", "fp8"):
    spare = usable - weight_memory_gb(SMALL.params_b, dt)
    print(f"  spare after weights @ {dt:<4}: {spare:5.0f} GB  <- this, not 80, sizes concurrency")

line()
print("  KV cache (the concurrency tax):")
for dt in ("bf16", "fp8"):
    print(f"    {dt}: {kv_per_token_kb(SMALL, dt):.0f} KB/token   "
          f"| 8K conv = {kv_per_session_gb(SMALL, 8000, dt):.2f} GB")

print("  concurrent 8K sessions on one H100:")
for dt in ("bf16", "fp8"):
    spare = usable - weight_memory_gb(SMALL.params_b, dt)
    n = max_concurrent_sessions(spare, SMALL, 8000, dt)
    print(f"    {dt}: ~{n:4.0f} sessions   (fp8 is a 4x concurrency lever)")

line()
print("  decode speed (bandwidth-bound):")
for dt in ("bf16", "fp8"):
    w = weight_memory_gb(SMALL.params_b, dt)
    print(f"    {dt}: single-stream ceiling ~{decode_tok_s_single(w, H100):.0f} tok/s")
agg, per = decode_aggregate(SMALL, H100, batch=32, context_tokens=4000, dtype="bf16")
print(f"    batch 32 @ 4K bf16: {agg:.0f} tok/s aggregate, {per:.0f} tok/s per user")
print(f"    decode stays bandwidth-bound until ~batch {roofline_batch(H100):.0f}")

line()
print("  prefill (compute-bound) -> TTFT:")
for pt in (2000, 8000, 32000):
    print(f"    {pt:>6}-token prompt: TTFT ~{ttft_s(SMALL.active_b, pt, H100):.2f} s")


# ---------------------------------------------------------------- B ----------
print()
print("=" * 68)
print("B. SINGAPORE BANK — internal assistant, on-prem (data residency)")
print("=" * 68)
# The workload's inputs (state them before sizing anything):
STAFF        = 10_000
PEAK_ACTIVE  = 0.10          # fraction active at peak
REQ_PER_MIN  = 0.5           # each active user: ~1 request / 2 min
IN_TOK       = 1500          # RAG-ish prompt
OUT_TOK      = 300
AVG_CTX      = IN_TOK + OUT_TOK // 2   # KV holds prompt + generated-so-far
TPOT_MS      = 40            # SLO: >=25 tok/s per user
DTYPE        = "fp8"         # default on Hopper+

rps = STAFF * PEAK_ACTIVE * REQ_PER_MIN / 60
print(f"  peak RPS = {STAFF} * {PEAK_ACTIVE} * {REQ_PER_MIN}/min "
      f"= {rps:.1f} req/s")

dur = request_duration_s(SMALL.active_b, IN_TOK, OUT_TOK, H100, TPOT_MS, DTYPE)
conc = concurrency(rps, dur)
print(f"  request lifetime = TTFT + {OUT_TOK}*{TPOT_MS}ms = {dur:.1f} s")
print(f"  concurrent sessions (Little's Law) = {rps:.1f} * {dur:.1f} = ~{conc:.0f}")

line()
print("  GPUs required, per constraint (H100):")
usable = usable_hbm_gb(H100)

# memory constraint, both dtypes (to show the lever)
for dt in ("bf16", "fp8"):
    spare = usable - weight_memory_gb(SMALL.params_b, dt)
    per_gpu = max_concurrent_sessions(spare, SMALL, AVG_CTX, dt)
    g = conc / per_gpu
    print(f"    memory @ {dt:<4}: {per_gpu:4.0f} sessions/GPU -> {g:4.1f} GPU  "
          f"({math.ceil(g)} rounded)")

# decode-throughput constraint
decode_demand = rps * OUT_TOK
per_gpu_decode, _ = decode_aggregate(SMALL, H100, batch=int(conc), context_tokens=AVG_CTX, dtype=DTYPE)
g_dec = decode_demand / per_gpu_decode
print(f"    decode tput  : need {decode_demand:.0f} tok/s, "
      f"~{per_gpu_decode:.0f}/GPU -> {g_dec:.1f} GPU")

# prefill-throughput constraint
prefill_demand = rps * IN_TOK
per_gpu_prefill = prefill_tok_s(SMALL.active_b, H100, DTYPE)
g_pre = prefill_demand / per_gpu_prefill
print(f"    prefill tput : need {prefill_demand:.0f} tok/s, "
      f"~{per_gpu_prefill:.0f}/GPU -> {g_pre:.1f} GPU")

line()
raw_fp8 = max(math.ceil(conc / max_concurrent_sessions(usable - weight_memory_gb(SMALL.params_b, "fp8"), SMALL, AVG_CTX, "fp8")),
              math.ceil(g_dec), math.ceil(g_pre))
raw_bf16 = math.ceil(conc / max_concurrent_sessions(usable - weight_memory_gb(SMALL.params_b, "bf16"), SMALL, AVG_CTX, "bf16"))
print(f"  raw requirement: {raw_fp8} GPU in fp8  |  {raw_bf16} GPU in bf16")
print(f"    -> the driver is KV-cache MEMORY; fp8 is the biggest single lever")
print(f"  provisioned (conservative 70% util + N+1): {provision(raw_fp8)} GPU (fp8)")
print(f"  RECOMMENDATION: run fp8. Raw need is <1 GPU, so 2 (1 active + 1 spare)")
print(f"    meets today's SLO; the {provision(raw_fp8)} above is the conservative floor.")
print(f"    Buy one 8-GPU HGX node -> room for embed+rerank models and growth.")


# ---------------------------------------------------------------- C ----------
print()
print("=" * 68)
print("C. MISTRAL LARGE 3 (675B MoE, 41B active) — sizing splits in two")
print("=" * 68)
print("  MEMORY is sized by TOTAL params (all experts live in HBM):")
for dt in ("fp8", "nvfp4"):
    w = weight_memory_gb(LARGE.params_b, dt)
    print(f"    weights @ {dt:<5}: {w:4.0f} GB")
print(f"    8x H100 usable = {8*usable_hbm_gb(H100):.0f} GB  -> fp8 weights DO NOT fit once KV is added")
print(f"    8x H200 usable = {8*usable_hbm_gb(H200):.0f} GB  -> fits, ~{8*usable_hbm_gb(H200)-weight_memory_gb(LARGE.params_b,'fp8'):.0f} GB left for KV")

line()
print("  COMPUTE is sized by ACTIVE params:")
print(f"    prefill costs like a {LARGE.active_b:.0f}B dense model, not 675B")
print(f"    TTFT, 2K prompt, 8x H200: ~{ttft_s(LARGE.active_b, 2000, H200)/8:.2f} s (compute split 8 ways)")

line()
print("  DECODE — the MoE trap:")
print("    low batch  : streams ~41B active -> fast")
w_total = weight_memory_gb(LARGE.params_b, "fp8")
agg_bw = 8 * H200.bw_tb_s          # 8 GPUs, tensor-parallel, aggregate HBM bw
step_ms = w_total / (agg_bw * 1000) * 1000
print(f"    high batch : ~every expert hit each step -> stream ~{w_total:.0f} GB")
print(f"                 / (8 x {H200.bw_tb_s} TB/s) = ~{step_ms:.0f} ms/step floor")
print("    => MoE wants BIG batches + expert parallelism to be worth it.")
print()
print("  Parallelism rule: smallest tensor-parallel degree that fits weights+KV")
print("  (TP inside a node over NVLink @450 GB/s each way; never TP across IB @~50 GB/s each way, ~9x less),")
print("  then add replicas for throughput; pipeline across nodes only if forced.")
