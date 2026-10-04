# GPU Capacity Planning for LLM Deployment — A Primer

Capacity planning depends on **two constraints** and **two SLOs** only. Learn these
well. Then the GPU count, the batch size, the parallelism and the quantization all come
from them.

| | What it limits | What sets it |
|---|---|---|
| **Constraint: Memory (HBM size)** | *What fits*: weights + KV cache | params × bytes, KV cache growth |
| **Constraint: Bandwidth / Compute** | *How fast* tokens come out | HBM speed (decode), FLOPS (prefill) |
| **SLO: TTFT** | Time to first token | **Prefill**, compute-bound |
| **SLO: TPOT** | Time per output token | **Decode**, bandwidth-bound |

The most important idea is this: **prefill is compute-bound, decode is
bandwidth-bound.** They fail for different reasons, and different levers repair
them. Never think about the two together.

---

## The formulas (all of `capacity.py` in one page)

**Units.** From here to the end of this primer, GB means 10⁹ bytes. This unit applies
to the weights, the HBM and the KV cache alike. Thus you can add and subtract them.

This primer uses the marketed size of the HBM. The marketed size of an H100 is "80 GB",
but the H100 has 80 GiB (85.9 × 10⁹ bytes). Thus a plan with 80 × 10⁹ is about 7%
conservative. Layer 01's
[roofline primer §1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#1-spec-sheet-literacy)
makes the same choice.

**1. Weights: does it fit?**

$$
\text{memory} = \text{params} \times \text{bytes/param} \qquad \text{bf16} = 2,\ \text{fp8} = 1,\ \text{int4} = 0.5
$$

A 24B model needs 48 GB (bf16), 24 (fp8) or 12 (int4). Add ~10% for the CUDA context
and the activations. On an 80 GB H100, the **spare after weights** decides how many
users you serve. The 80 does not decide it.

**2. KV cache: how many concurrent users fit?** *(most people do not include this number)*

$$
\text{KV/token} = 2\,(K,V) \times \text{layers} \times \text{kv_heads} \times \text{head_dim} \times \text{bytes}
$$

Each token of each live conversation keeps this quantity in HBM. For Mistral Small, the value is
`2×40×8×128×2 = 163,840 bytes ≈ 164 kB/token` (bf16), and 82 kB (fp8).

- An 8K conversation ≈ 1.3 GB, a 32K conversation ≈ 5.2 GB, and a 128K conversation ≈ 21 GB.
- $\text{concurrent sessions} = \frac{\text{spare_HBM}}{\text{KV_per_session}}$
- **GQA is why this works**: the formula uses `kv_heads` (8), not the query heads (32).
  The use of 8 heads in place of 32 is a built-in 4× decrease. Mistral 7B made GQA popular.
- **Quantization is a concurrency lever before it is a speed lever**: fp8 divides the
  weights by two (more spare) *and* divides the KV by two. The result is ~4× the sessions.

**3. Decode: bandwidth-bound.** For each token, the GPU reads all the weights (plus the
KV of the batch) from HBM.

$$
\text{time/token} \approx \frac{\text{bytes_read}}{\text{HBM_bandwidth}}
$$

48 GB ÷ 3.35 TB/s ≈ 14 ms. This gives a **~70 tok/s single-stream ceiling** on H100.
More compute does *not* help. Only more bandwidth or fewer bytes (quantize) help.

**Batching is nearly free**: the GPU reads the weights one time per step for the full
batch. Thus throughput increases in proportion to the batch until you reach the memory limit or the
compute roofline (~batch 300 on H100 = FLOPS ÷ bandwidth).

**4. Prefill: compute-bound.**

$$
\begin{aligned}
\text{FLOPs} &\approx \underbrace{2 \times \text{params} \times S}_{\text{weight GEMMs}} + \underbrace{2 \times \text{layers} \times \text{q_heads} \times \text{head_dim} \times S^2}_{\text{causal attention } (QK^\top \text{ and } AV)} \\
&\qquad (S = \text{prompt tokens}) \\[8pt]
\text{TTFT} &= \frac{\text{FLOPs}}{\text{peak_FLOPS} \times \text{MFU}}
\end{aligned}
$$

The attention term increases as $S^2$. Thus it is important only for long prompts.
Mistral Small has 40 layers and 32 query heads × 128. For this model, the attention term adds 1.4% at 2K
tokens, 5.6% at 8K, 22% at 32K and 90% at 128K (`capacity.attention_flops`).

A 24B model with a 2K prompt needs ≈ 100 TFLOP. At 50% MFU in fp8, this takes ~0.1 s.
A 32K RAG prompt needs ≈ 1.5 PFLOP for the weights plus 0.35 for attention, a total of
≈ 1.9 PFLOP. This takes ~1.9 s. The bank example in the next section does not include
attention: at 1,500 tokens, attention is 1%.

RAG and agents are prefill-dominated. Thus **prefix caching** (shared system prompts,
repeated documents) is the largest single gain for them.

**5. From the workload to GPUs.** First, find the exact workload. It has these parts:

- the **peak concurrent users** (not the headcount)
- the average input/output tokens
- the TTFT and TPOT targets
- the availability
- the growth

$$
\begin{aligned}
\text{concurrency} &= \text{RPS} \times \text{request_duration} \qquad \text{(Little's Law)} \\
\text{request_duration} &\approx \text{TTFT} + \text{output_tokens} \times \text{TPOT}
\end{aligned}
$$

Then calculate the GPUs that *each* constraint needs, one constraint at a time. Take the
maximum. The constraints are memory (sessions ÷ per-GPU), decode throughput and prefill
throughput. Last, apply utilisation headroom (~60–70%) and add **N+1** spares.

---

## Worked example — Singapore bank, on-prem (data residency)

The bank has 10,000 staff. At peak, 10% are active, and each sends ~1 request / 2 min.
This gives **~8 RPS**. On average, a request has 1,500 tokens in and 300 tokens out. The
target is ≥25 tok/s/user (TPOT ≤ 40 ms).

- **Concurrency**: the duration is ≈ 0.1 s + 300×40 ms ≈ 12 s. Thus 8 × 12 ≈ **~100 live
  sessions**.
- **Memory (the driver)**: in bf16, one GPU holds ~89 sessions. Thus you need **2 GPUs**.
  In fp8, one GPU holds ~355. Thus you need **1 GPU**. *This is the lesson: the binding
  constraint is KV-cache memory, and fp8 is the largest lever on it.*
- **Decode throughput**: the need is 8×300 = 2,500 tok/s. One H100 does ~9,000. Thus
  the need is <1 GPU.
- **Prefill throughput**: the need is 8×1,500 = 12,500 tok/s. One H100 does ~20,000.
  Thus the need is <1 GPU.
- **Answer**: run fp8. The raw need is <1 GPU. Thus **2× H100 (1 active + 1 spare)**
  meets today's SLO. Get a full **8-GPU HGX node** for the embedding model, the
  reranker and a year of growth.
- **Cost check**: $/M tokens = GPU-hour price ÷ tokens/hour (1,500 tok/s ≈ 5.4M/hr).

> The two paths of the size calculation are memory (~sessions/GPU) and throughput (tok/s/GPU). Expect
> their results to be approximately the same. Show both in a design review. If the
> difference between them is large, one of the inputs is incorrect.

---

## When one GPU (or one node) won't do — Mistral Large 3 (675B MoE, 41B active)

MoE divides the size calculation into two parts, and people make errors in both:

- **The TOTAL params set the memory** (all experts are in HBM): ~675 GB in fp8. When
  you add the KV, this does **not** fit in 8×H100 (576 GB usable). The minimum is
  **8×H200** (~1 TB) or a Blackwell node. NVFP4 (~340 GB) is the native path of Blackwell.
- **The ACTIVE params set the compute**: prefill costs the same as a 41B dense model.
  But **decode is only 41B-like at low batch**. At high batch, each step uses almost
  every expert. Thus each step reads ~675 GB, which gives a floor of ~18 ms across
  8×H200. For MoE to give a benefit, it needs **large batches and expert parallelism**.

[MoE primer §5](../mixture-of-experts/PRIMER.md#5-moe-at-inference-which-experts-a-step-touches) and
[§7](../mixture-of-experts/PRIMER.md#7-sizing-and-cost) tell why a batch reads almost every
expert. They also tell how to size memory by the total, prefill by the active and decode by
the bytes that a step reads. `moecore` reproduces the 17.6 ms floor of this section.

**The general rule for parallelism:**

1. Use the **smallest tensor-parallel (TP)** degree in which the weights + KV fit.
2. Use TP only *inside* a node. TP needs continuous traffic over **NVLink (H100: 450 GB/s
   each way, marketed as 900 GB/s for both directions)**. Never use TP across **InfiniBand
   (a 400 Gb/s NIC: 50 GB/s each way)**. That NIC gives ~9× less per direction than NVLink
   ([roofline primer §5.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#51-the-link-ladder)).
3. Scale throughput with **replicas** (data parallelism). Each replica is a full copy.
4. Use **pipeline-parallel (PP) across nodes** only when a single node cannot hold
   the model.
5. **Expert-parallel (EP)** divides the MoE experts across GPUs.

---

## Numbers to carry in your head

| GPU | HBM | Bandwidth | BF16 / FP8 TFLOPS |
|---|---|---|---|
| H100 | 80 GB | 3.35 TB/s | 990 / 1,979 |
| H200 | 141 GB | 4.8 TB/s | 990 / 1,979 |
| B200 | ~180 GB | ~8 TB/s | ~2,250 / ~4,500 (native FP4) |
| MI300X | 192 GB | 5.3 TB/s | (AMD, important when the NVIDIA supply is low) |

*(The Blackwell/MI300X compute numbers are approximate. Read the current spec sheets.)*

- bytes/param: **bf16 = 2, fp8 = 1, int4 = 0.5**
- single-stream tok/s ≈ **bandwidth ÷ weight_bytes**
- good chat UX: **TTFT < 1 s, ≥ 20–30 tok/s/user**
- **concurrency = RPS × duration**. Plan at 60–70% util, with N+1.
- decode becomes compute-bound at **batch ≈ FLOPS ÷ bandwidth** (~300 on H100)

## Decisions worth defending

- **Dense against MoE** for the workload (the quality that you need against the hardware
  that you can get).
- **FP8 by default** on Hopper+. **INT4 only where the workload's own evals showed that
  the quality is sufficient.**
- **Prefix caching** for RAG/agents, and **speculative decoding** for latency at low batch.
- **Thinking models** change the output length, not the formulas. For example, the model
  writes 2,700 thinking tokens before a 300-token answer. In that case, the bank example needs
  about 18× the GPUs for KV memory at the TPOT of the SLO
  ([RL and thinking-models primer §7](../rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)).
- **Chunked prefill** (interleave long prompts with decode) against **disaggregated
  prefill/decode pools** (separate GPU fleets, each sized for its bottleneck). Make this
  decision when you have more than a few nodes.
- When data must stay in one country, the binding constraint is usually **in-country
  GPU availability**. The question then becomes: "which model gives the quality you need
  on the hardware you can actually get there?"
