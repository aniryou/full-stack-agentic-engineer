# GPU Capacity Planning for LLM Deployment — A Primer

Everything reduces to **two constraints** and **two SLOs**. Master these and
GPU count, batch size, parallelism, and quantization all fall out.

| | What it limits | What sets it |
|---|---|---|
| **Constraint: Memory (HBM size)** | *What fits* — weights + KV cache | params × bytes, KV cache growth |
| **Constraint: Bandwidth / Compute** | *How fast* tokens come out | HBM speed (decode), FLOPS (prefill) |
| **SLO: TTFT** | Time to first token | **Prefill** — compute-bound |
| **SLO: TPOT** | Time per output token | **Decode** — bandwidth-bound |

The single biggest idea: **prefill is compute-bound, decode is
bandwidth-bound.** They fail for different reasons and are fixed with different
levers. Never reason about them together.

---

## The formulas (all of `capacity.py` in one page)

**1. Weights — does it fit?**
```
memory = params × bytes/param        bf16=2, fp8=1, int4=0.5
```
24B → 48 GB (bf16), 24 (fp8), 12 (int4). Add ~10% for CUDA context and
activations. On an 80 GB H100 the **spare after weights**, not the 80, is what
decides how many users you serve.

**2. KV cache — how many concurrent users fit?** *(the number most people miss)*
```
KV/token = 2(K,V) × layers × kv_heads × head_dim × bytes
```
Every token of every live conversation holds this in HBM. Mistral Small:
`2×40×8×128×2 = 160 KB/token` (bf16), 80 KB (fp8).
- 8K conversation ≈ 1.2 GB; 32K ≈ 5 GB; 128K ≈ 20 GB
- `concurrent sessions = spare_HBM ÷ KV_per_session`
- **GQA is why this works**: it uses `kv_heads` (8), not query heads (32) — a
  built-in 4× cut. Mistral 7B popularised it.
- **Quantization is a concurrency lever before it is a speed lever**: fp8 both
  halves weights (more spare) *and* halves KV → ~4× the sessions.

**3. Decode — bandwidth-bound.** Each token streams all weights (plus the
batch's KV) through HBM.
```
time/token ≈ bytes_read ÷ HBM_bandwidth
```
48 GB ÷ 3.35 TB/s ≈ 14 ms → **~70 tok/s single-stream ceiling** on H100. More
compute does *not* help — only more bandwidth or fewer bytes (quantize).
**Batching is nearly free**: weights are read once per step for the whole
batch, so throughput scales with batch until you hit memory or the compute
roofline (~batch 300 on H100 = FLOPS ÷ bandwidth).

**4. Prefill — compute-bound.**
```
FLOPs ≈ 2 × params × prompt_tokens        TTFT = FLOPs ÷ (peak_FLOPS × MFU)
```
24B × 2K prompt ≈ 100 TFLOP → ~0.1–0.2 s. A 32K RAG prompt ≈ 1.6 PFLOP → seconds.
RAG and agents are prefill-dominated, so **prefix caching** (shared system
prompts, repeated documents) is the biggest single win there.

**5. Workload → GPUs.** Get from the customer: **peak concurrent users** (not
headcount), avg input/output tokens, TTFT + TPOT targets, availability, growth.
```
concurrency = RPS × request_duration        (Little's Law)
request_duration ≈ TTFT + output_tokens × TPOT
```
Then compute GPUs required by *each* constraint separately and take the max:
memory (sessions ÷ per-GPU), decode throughput, prefill throughput. Finally
apply utilisation headroom (~60–70%) and add **N+1** spares.

---

## Worked example — Singapore bank, on-prem (data residency)

10,000 staff, 10% active at peak, ~1 request / 2 min → **~8 RPS**.
Avg 1,500 in / 300 out. Target ≥25 tok/s/user (TPOT ≤ 40 ms).

- **Concurrency**: duration ≈ 0.1 s + 300×40 ms ≈ 12 s → 8 × 12 ≈ **~100 live sessions**
- **Memory (the driver)**: bf16 → ~95 sessions/GPU → **2 GPUs**; fp8 → ~380/GPU
  → **1 GPU**. *This is the lesson: the binding constraint is KV-cache memory,
  and fp8 is the biggest lever on it.*
- **Decode throughput**: need 8×300 = 2,500 tok/s; one H100 does ~9,000 → <1 GPU
- **Prefill throughput**: need 8×1,500 = 12,500 tok/s; one H100 does ~20,000 → <1 GPU
- **Answer**: run fp8. Raw need <1 GPU → **2× H100 (1 active + 1 spare)** meets
  today's SLO. Provision a full **8-GPU HGX node** for the embedding model,
  reranker, and a year of growth.
- **Cost check**: $/M tokens = GPU-hour price ÷ tokens/hour (1,500 tok/s ≈ 5.4M/hr).

> The two sizing paths — memory (~sessions/GPU) and throughput (tok/s/GPU) —
> should roughly agree. Say that out loud; it shows you know
> both constraints bind.

---

## When one GPU (or one node) won't do — Mistral Large 3 (675B MoE, 41B active)

MoE splits the sizing in two, and people get both wrong:

- **Memory is sized by TOTAL params** (all experts live in HBM): ~675 GB in
  fp8. That does **not** fit 8×H100 (576 GB usable) once KV is added. Floor is
  **8×H200** (~1 TB) or a Blackwell node; NVFP4 (~340 GB) is Blackwell's native path.
- **Compute is sized by ACTIVE params**: prefill costs like a 41B dense model.
  But **decode is only 41B-like at low batch** — at high batch nearly every
  expert is hit every step, so you stream ~675 GB/step → ~18 ms floor across
  8×H200. **MoE wants big batches and expert parallelism** to pay off.

Why a batch reads nearly every expert, and how to size memory by total, prefill by active and decode by the bytes a
step streams: [MoE primer §5](../mixture-of-experts/PRIMER.md#5-moe-at-inference-which-experts-a-step-touches) and
[§7](../mixture-of-experts/PRIMER.md#7-sizing-and-cost) (`moecore` reproduces this section's 17.6 ms floor).

**Parallelism rule of thumb:**
1. Use the **smallest tensor-parallel (TP)** degree that fits weights + KV.
2. TP only *inside* a node — it needs constant chatter over **NVLink (~900 GB/s)**;
   never TP across **InfiniBand (~50 GB/s)**, ~18× slower.
3. Scale throughput with **replicas** (data parallelism), each a full copy.
4. **Pipeline-parallel (PP) across nodes** only when a single node can't hold it.
5. **Expert-parallel (EP)** spreads MoE experts across GPUs.

---

## Numbers to carry in your head

| GPU | HBM | Bandwidth | BF16 / FP8 TFLOPS |
|---|---|---|---|
| H100 | 80 GB | 3.35 TB/s | 990 / 1,979 |
| H200 | 141 GB | 4.8 TB/s | 990 / 1,979 |
| B200 | ~180 GB | ~8 TB/s | ~2,250 / ~4,500 (native FP4) |
| MI300X | 192 GB | 5.3 TB/s | (AMD; matters when NVIDIA supply is tight) |

*(Blackwell/MI300X compute are approximate — check current spec sheets.)*

- bytes/param: **bf16 = 2, fp8 = 1, int4 = 0.5**
- single-stream tok/s ≈ **bandwidth ÷ weight_bytes**
- good chat UX: **TTFT < 1 s, ≥ 20–30 tok/s/user**
- **concurrency = RPS × duration**; plan at 60–70% util, N+1
- decode goes compute-bound at **batch ≈ FLOPS ÷ bandwidth** (~300 on H100)

## Decisions worth defending

- **Dense vs MoE** for the workload (quality needed vs hardware you can get).
- **FP8 by default** on Hopper+; **INT4 only where quality was validated on the
  customer's own evals.**
- **Prefix caching** for RAG/agents; **speculative decoding** for latency at low batch.
- **Thinking models** change the output length, not the formulas: with 2,700 thinking tokens before a 300-token
  answer, the bank example needs about 18× the GPUs for KV memory at the SLO's TPOT
  ([RL and thinking-models primer §7](../rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)).
- **Chunked prefill** (interleave long prompts with decode) vs **disaggregated
  prefill/decode pools** (separate GPU fleets, each sized for its bottleneck)
  once you're past a few nodes.
- For an APAC partner deployment the real constraint is usually **data residency +
  in-country GPU availability** → "which Mistral model gives the quality you need
  on the hardware you can actually get?"
