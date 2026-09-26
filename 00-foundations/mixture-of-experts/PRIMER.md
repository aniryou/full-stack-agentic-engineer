# Mixture-of-experts models: the router, the experts, and what sparsity does to serving

A mixture-of-experts (MoE) model replaces each transformer block's MLP with many expert MLPs and a small router
that sends every token to a few of them. This primer explains the layer from first principles, how routers are
trained to share the work, which experts a batch of tokens actually reads at inference time, how MoE runs on GPUs
(fused kernels, expert parallelism, all-to-alls, offloading, quantized experts), and how to size and cost a
deployment. Every formula has a worked number computed by the package `moecore` in [`moe-core/`](moe-core/README.md)
(standard library + numpy; the function is named next to the number, and `moe-core/tests/test_primer_numbers.py`
fails if the text and the code disagree). The lab in [`moe-lab/`](moe-lab/README.md) runs the same ideas on real GPUs. The transformer
itself is in the [transformer primer](../transformers/docs/transformer-primer.md); memory and TTFT/TPOT sizing in
the [capacity primer](../gpu-capacity-planning/PRIMER.md); the roofline in
[layer 01](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). This primer links to those rather than
repeating them.

---

## The one-minute version

- **An MoE layer is E expert MLPs plus a router.** The router scores each token against every expert; the token
  keeps its top-k and its output is the weighted sum of those k experts' outputs (plus a shared expert, in models
  that have one). All E experts live in memory; each token computes only k.
- **Three parameter counts size three things.** HBM holds the *total* (Mixtral-8x7B: 46.70B; DeepSeek-V3: 671.03B).
  Prefill FLOPs follow the *active* count (12.88B; 37.55B). Decode time follows the bytes a step *streams*, which
  start near active at batch 1 and approach total at serving batches.
- **Routers collapse unless balanced.** Training rewards the expert that is already good, so a few experts take
  every token. An auxiliary loss, capacity limits, or DeepSeek-V3's selection-only bias spread the work.
- **A batch reads the union of its tokens' experts:** E(1 − (1 − k/E)^T) per layer, the formula layer 01 uses. So
  Mixtral decodes like a 13B model at batch 1 and streams nearly all 93 GB of its weights by batch 16, and decode turns
  compute-bound only at a batch about total ÷ active (more precisely, weights streamed ÷ weights multiplied) times
  the dense one: 754 for Mixtral and 2,055 for Qwen3-30B-A3B on an H200, against 207 for a dense 8B. MoE wants big
  batches.
- **Big batches mean expert parallelism.** Each GPU holds some experts; each MoE layer exchanges tokens (with
  DeepEP-class kernels, two all-to-alls: dispatch tokens, combine results) and waits for its busiest GPU — in
  prefill the GPU with the most rows, in decode the busiest link. Data-parallel attention plus EP ("wide-EP")
  spreads the experts thin and frees memory for KV.
- **MoE does not change attention or the KV cache.** Prefix caching and KV sizing work exactly as for the dense
  model with the same attention; at long context the KV cache dominates again.

---

## 1. Why sparsity

**Parameters are knowledge; FLOPs are cost.** A dense transformer multiplies every weight by every token, so its
parameter count sets both what it can store and what each token costs. Loss keeps falling as parameters grow, but a
dense model pays for every parameter on every token. MoE breaks that link: store E times as many MLP weights, but
route each token through only k of them. The transformer primer's
[§9 table](../transformers/docs/transformer-primer.md#9-modern-variants-and-why-each-exists) puts it in one row:
"parameters grow ~E× while compute per token barely moves. More knowledge per FLOP, at the cost of memory and
serving complexity."

**The scaling-law view.** At a fixed active size, more total parameters buy lower loss. Moonshot's Kimi K2 report
(§2.3) fixes the activated parameters (8 routed + 1 shared expert) and varies the number of experts: at equal
validation loss, sparsity 48 (384 experts, 8 active) needs 1.69×, 1.39× and 1.15× fewer FLOPs than sparsity 8,
16 and 32 — diminishing returns, but returns. Most of the 2026 frontier open-weight models are MoE, from 26B with
~4B active (Gemma 4) to 2.8T with 104B active (Kimi K3); see the
[open-weight primer §4](../model-landscape/open-weight-llms-primer.md#4-the-families-vendor-by-vendor) (verify).

**Memory by total, FLOPs by active.** The capacity primer works this through for Mistral Large 3 (675B total, 41B
active; section "When one GPU (or one node) won't do" in the [capacity primer](../gpu-capacity-planning/PRIMER.md)).
In FP8 its weights are ~675 GB — more than 8×H100's 576 GB of usable HBM before any KV cache. Prefill costs like a 41B dense model:
a 2,048-token prompt is 2 × 41e9 × 2,048 = 1.68 × 10¹⁴ FLOPs (`sizing.prefill_flops()`, the same formula as
`capacity.prefill_flops()`), 16.5× less than a dense 675B model would need. Decode at large batch streams every
expert every step: 675 GB ÷ (8 × 4.8 TB/s) = **17.6 ms** per step across 8 H200s (`sizing.decode_floor()`), the
"~18 ms floor" the capacity primer quotes.

| What | Sized by | Mixtral-8x7B | Qwen3-30B-A3B | DeepSeek-V3 |
|---|---|---|---|---|
| HBM for weights | total parameters | 46.70B → 93.4 GB bf16 | 30.53B → 61.1 GB bf16 | 671.03B → 671.0 GB FP8 |
| compute per token (FLOPs = 2 × this) | active parameters | 12.88B | 3.35B | 37.55B |
| total ÷ active | how much the batch must grow | 3.6 | 9.1 | 17.9 |
| KV per token (bf16) | attention only | 131,072 B | 98,304 B | 70,272 B (MLA latent) |

(`MoEConfig.total()`, `.active()`, `.kv_bytes_per_token()`, `sizing.weight_bytes()`; configs in `sizing.MODELS`.)

**What MoE does not change: attention and the KV cache.** Experts replace the MLP; attention is untouched. Mixtral's
attention has Llama-3.1-8B's shape (32 query heads, 8 KV heads of 128, 32 layers), so both cache exactly
2 × 32 × 8 × 128 × 2 = 131,072 bytes per token. Paged KV, prefix caching and KV quantization
([`04-inference-engine/kv-cache`](../../04-inference-engine/kv-cache/), [`paged-attention`](../../04-inference-engine/paged-attention/))
apply unchanged; what changes is how large a share of each step the KV reads are (§5).

---

## 2. The MoE layer

### 2.1 Experts and the router

```
                  x (one token, d)
                  │
        ┌─────────┴───────────────────────────┐
        │ router: logits = x · W_r   (d × E)   │      shared expert(s)
        │ scores = softmax or sigmoid          │      ┌──────────────┐
        │ keep top-k, weights w₁..w_k          │      │ MLP_s(x)     │ every token, weight 1
        └──┬──────────┬───────────────────────┘      └──────┬───────┘
           │ w₁       │ w₂          (k = 2 of E = 8)         │
        ┌──▼───┐   ┌──▼───┐   ┌──────┐ ┌──────┐            │
        │ E₃   │   │ E₆   │   │ E₀   │ │ ...  │  unchosen: no FLOPs, still in HBM
        └──┬───┘   └──┬───┘   └──────┘ └──────┘            │
           └────┬─────┘                                    │
                ▼                                          ▼
          y = w₁·E₃(x) + w₂·E₆(x)            +          MLP_s(x)
```

Each expert is an ordinary gated MLP (SwiGLU: down(silu(x·W_gate) ⊙ x·W_up)), `moe.Expert`. The router is one
linear layer, d × E parameters — 32,768 for Mixtral, negligible. With E = k = 1 the layer is exactly the dense MLP
(`moe-core` notebook 01, exercise 1.2).

### 2.2 Router variants, as implemented

The families differ in how scores become weights, and the difference is visible in outputs. The same eight logits
[2.0, 1.2, 0.4, 0.3, −0.5, −1.0, −1.1, −2.0] through each router (`moe.route()` with `moe.ROUTERS`):

| Family (source) | Scores | Selection | Weights | Worked weights (k = 2) |
|---|---|---|---|---|
| Mixtral (`MixtralTopKRouter`) | softmax over E | top-k of the probabilities | renormalised to sum to 1 | [0.69, 0.31] |
| OLMoE; Qwen2/Qwen3-MoE at transformers' default (`norm_topk_prob` False) | softmax over E | top-k | raw probabilities | [0.493, 0.221], sum 0.714 |
| Qwen2-MoE / Qwen1.5-MoE | softmax | top-k | raw; plus a shared expert × sigmoid(x · g) | — |
| DeepSeek-V3 (`Gate`) | sigmoid | top-k of score + per-expert bias, inside the best 4 of 8 groups | *unbiased* scores, renormalised, × 2.5 (`route_scale`) | [1.335, 1.165], sum 2.5 |
| gpt-oss, Granite | router with a bias (gpt-oss) | top-k of the logits | softmax over just the k logits | [0.69, 0.31] |
| Llama 4 | sigmoid | top-1 | its sigmoid score scales the expert's **input** | [0.881] (k = 1) |

Two details carry lessons. First, softmax over the top-k *logits* (gpt-oss) equals Mixtral's renormalised softmax —
the exponent ratios are the same — so the family difference there is the router bias, not the arithmetic. Second,
DeepSeek-V3's bias only **chooses**: with a bias of +1.0 on the last expert (logit −2.0), it enters the top-2 on its
biased score, but its weight is its unbiased sigmoid, renormalised: 0.298 of the 2.5 (`moe-core` notebook 01). That
separation is what makes auxiliary-loss-free balancing work (§3.4). transformers' `norm_topk_prob` defaults to False
for Qwen2/Qwen3/OLMoE and OLMoE uses False (`ROUTERS["olmoe"]`); the released Qwen3 MoE configs are reported to set
it true (verify), which makes Qwen3 weight like Mixtral (`ROUTERS["qwen3-moe"]`). Read the checkpoint's config.

### 2.3 Shared experts and granularity

A **shared expert** runs on every token with weight 1: DeepSeek-V3 has 1 shared + 256 routed experts; Llama 4 one
shared expert; Qwen1.5-MoE-A2.7B a shared expert four routed experts wide (5,632 = 4 × 1,408), gated by
sigmoid(x · g), so each token uses 4 routed + 4 shared-sized units of 64. The shared expert holds what every token
needs, so routed experts are free to specialise. Qwen3 dropped it ("Unlike Qwen2.5-MoE, the Qwen3-MoE design
excludes shared experts", Qwen3 report §2).

**Granularity** is the size of an expert. Coarse: Mixtral's 8 experts of width 14,336, top-2. Fine: DeepSeek-V3's
256 of width 2,048, top-8 (plus shared). At equal parameters and equal FLOPs, fine experts give a token vastly more
combinations: splitting Mixtral's 8 × 14,336 into 64 × 1,792 with top-16 keeps FLOPs and parameters identical but
raises the number of expert sets from 28 to 4.89 × 10¹⁴ (`math.comb`, notebook 01 exercise 1.5) — the DeepSeekMoE
argument. The serving price comes in §5: fine-grained experts per token touch more of the model per batch, and each
expert sees fewer rows.

### 2.4 Counting total and active parameters from a config

`MoEConfig` (`moecore/moe.py`) counts matmul weights, embeddings and biases (norms, < 0.01%, are ignored) from the
config fields, read from the model code under the upstream repos (transformers, deepseek-v3, gpt-oss, llama-models).

**Mixtral-8x7B** (32 layers, d 4,096, GQA 32/8 × 128, experts of width 14,336, E 8, k 2, vocab 32,000):

```
one expert     3 × 4,096 × 14,336                       = 176,160,768
attention      2 × 4,096 × 4,096 + 2 × 4,096 × 1,024     =  41,943,040
router         4,096 × 8                                =      32,768
total   32 × (attention + 8 experts + router) + 2 × 32,000 × 4,096 = 46,702,526,464   (46.70B)
active  32 × (attention + 2 experts + router) + 2 × 32,000 × 4,096 = 12,879,659,008   (12.88B)
```

"8x7B" double-counts: attention and the embeddings are shared, so eight 7B models would be ~56B.

**DeepSeek-V3** (61 layers of which 3 dense, d 7,168, MLA with 128 heads, 256 routed experts of width 2,048 + 1
shared, top-8, vocab 129,280): MLA per layer is 187,107,328 parameters (`moe.mla_params()`), one expert 44,040,192,
the dense layers' MLP width 18,432. Total **671.03B** and active **37.55B** — the published 671B / 37B. The Hugging
Face checkpoint is 685B: 671B plus a 14B multi-token-prediction module that serving loads only for speculative
decoding (verify).

**Which "active"?** Published active counts use different embedding accounting (`MoEConfig.active(embeddings=)`):

| Model | total | active, both tables | LM head only | matmuls only | published |
|---|---|---|---|---|---|
| Mixtral-8x7B | 46.70B | 12.88B | 12.75B | 12.62B | 12.9B |
| Qwen3-30B-A3B | 30.53B | 3.35B | 3.04B | 2.73B | 3.3B |
| Qwen3-235B-A22B (verify d, I) | 235.09B | 22.19B | 21.57B | 20.95B | 22B |
| DeepSeek-V3 | 671.03B | 37.55B | 36.62B | 35.70B | 37B |
| gpt-oss-120b | 116.83B | 5.71B | **5.13B** | 4.55B | 5.1B |
| gpt-oss-20b (24 layers derived, verify) | 20.91B | 4.19B | **3.61B** | 3.03B | 3.6B |
| Llama 4 Scout (text) | 107.77B | 17.17B | 16.14B | 15.10B | 17B / 109B incl. vision |
| Llama 4 Maverick (text; interleave verify) | 400.71B | 17.18B | 16.15B | 15.12B | 17B / 400B |
| Qwen1.5-MoE-A2.7B | 14.32B | 2.69B | 2.38B | 2.07B | 2.7B |
| OLMoE-1B-7B | 6.92B | 1.28B | 1.18B | 1.08B | 1.3B / 6.9B |

OpenAI's 5.1B for gpt-oss-120b counts the LM head only; DeepSeek's 37B and layer 01's `roofline.llm.active_params()`
count both tables. A 201K-token vocabulary at d = 2,880 is 0.58B per table — 11% of gpt-oss-120b's active count —
so name the convention before comparing.

### 2.5 How the layer runs

Running every expert on every token and zeroing the unchosen outputs is correct and E/k times too expensive
(`MoELayer.forward_dense()`, the reference). Engines do what `MoELayer.forward()` does: flatten the T × k
assignments, **sort them by expert**, run each expert once over its contiguous slice — a *grouped GEMM* — then
scatter the weighted rows back and sum each token's k copies. A test pins the two equal to 10⁻¹² for all five
router families. The sort is the heart of every fused MoE kernel (§6.1).

---

## 3. Routing and load balance

### 3.1 Router collapse

The router is trained by the task loss, and the task loss rewards routing a token to the expert that is *already*
good at it. The expert that wins early gets the gradient, improves, and wins more; an expert that gets no tokens
never trains and never gets a chance. Left alone, a layer **collapses** onto a few experts — the others are dead
weight in HBM and the model is a smaller dense model with extra memory.

`moecore.train` shows it with hand-written gradients (checked against finite differences): four clusters of tokens,
each with its own target map; four linear experts; a top-1 router with Switch-style weights. Hidden states share a
large common direction (as transformer hidden states do), so at step 0 two experts top most tokens, split 55/45.
Without balancing, one expert carries 95% of the tokens by the end and the task loss stays at 0.186. Over ten task
seeds, one to three of four experts end up carrying everything.

### 3.2 The Switch auxiliary loss and the z-loss

Switch Transformer (and GShard before it) adds

```
L_aux = α · E · Σ_e f_e · P_e      f_e = share of the batch's assignments routed to e
                                   P_e = mean router probability of e over the batch
```

which is smallest when both f and P are uniform. Only P carries a gradient (f comes out of a top-k), and
∂L_aux/∂p[t, e] = α · E · f_e / T pushes down the probability of loaded experts (`routing.switch_aux_grad()`).
**Two normalisations exist.** transformers' `load_balancing_loss_func` counts all k assignments (Σ f = k), so a
perfectly uniform router scores **k**; Megatron-LM and MegaBlocks divide by k, so uniform scores **1**
(`routing.switch_aux_loss(convention=)`). With E = 8, k = 2:

| Routing | hf convention | Megatron convention |
|---|---|---|
| uniform | 2.00 | 1.00 |
| two experts take everything | 8.00 | 4.00 |

The same coefficient therefore means a k-times different push. Coefficients in the wild: Mixtral's config 0.001,
OLMoE 0.01, Megatron's recommended starting value 1e-2 (Switch's paper used 0.01, verify). In the toy (§3.1), α =
0.1 balances every seed and lowers the task loss to 0.057 at seed 6; α = 0.01 is too weak and still collapses (notebook
02, exercise 2.4). Too strong trades quality for balance: the loss insists on equal counts even when the data is
not balanced, and the toy's aux run splits two unequal clusters across experts.

The **router z-loss**, mean over tokens of logsumexp(logits)², keeps router logits small so a bf16 softmax stays
accurate (`routing.z_loss()`): eight zero logits give (ln 8)² = 4.324, and adding 20 to every logit leaves the
softmax unchanged but raises it to 488. OLMoE trained with weight 0.001; MegaBlocks calls 1e-3 "a reasonable value".

### 3.3 Capacity factor and token dropping vs dropless

A fixed-shape training kernel wants each expert to process the same number of rows. The **capacity** is
`int(factor · k · T / E)` rows per expert (MegaBlocks' `expert_capacity`; `routing.capacity()`); assignments beyond it
are **dropped** — the token skips that expert and rides the residual connection (Megatron drops the tail of the
batch or the lowest-probability assignments; `routing.apply_capacity()`). For a skewed batch of 256 tokens, 8
experts, top-2 (hottest expert 191 assignments against a mean of 64):

| Capacity factor | Rows per expert | Assignments dropped |
|---|---|---|
| 1.00 | 64 | 32.8% |
| 1.25 | 80 | 26.6% |
| 2.00 | 128 | 12.3% |

Nothing is dropped until the factor passes the hottest load over the mean (191 / 64 = 2.98; 3.00 in steps of
0.25). **Dropless** MoE (MegaBlocks' dMoE, "removing the capacity_factor hyperparameter altogether"; OLMoE trained dropless) reformulates the expert
computation as block-sparse matmuls so every assignment is kept. Inference engines are dropless — a dropped token
changes the answer — and pay with padding instead: vLLM's `moe_align_block_size` sorts the T·k slots by expert and
pads each expert's segment to a multiple of the kernel's `BLOCK_SIZE_M` with a pad id (`routing.align_block_size()`
reproduces its docstring example). Here: 512 assignments become 576 rows at block 16, 12.5% padding.

### 3.4 Auxiliary-loss-free balancing and its sequence-level complement

DeepSeek-V3 moved balancing out of the loss. Each expert has a bias added to its score **for selection only**
(§2.2: the combine weight uses the unbiased score), and after each step the bias moves by a fixed rate against the
load — Megatron's `get_updated_expert_bias`:

```python
update_direction = torch.sign(total_tokens - tokens_per_expert * num_experts)   # under-loaded +, over-loaded −
expert_bias      = expert_bias + update_direction * expert_bias_update_rate     # 1e-3, "same as DeepSeekV3"
```

No gradient touches the router, so the task gradient stays clean. On a frozen, skewed router (load
[416, 237, 61, …] over 8 experts, top-2) 400 bias steps of 0.002 bring max/mean load to 1.03 (`routing.update_bias()`,
notebook 02 exercise 2.3). In the toy, bias balancing reaches the lowest loss of the three runs (0.036 at seed 6) with
the cleanest one-cluster-per-expert placement.

Batch-level balance can hide per-sequence collapse: four sequences that each send every token to a different expert
look perfect to the batch loss (1.0, Megatron convention) and collapsed to a per-sequence one (4.0 = E;
`routing.sequence_aux_loss()`, Megatron's `seq_aux_loss`). DeepSeek-V3 keeps a small sequence-wise balance loss
alongside the bias (α = 0.0001 in its report, verify).

### 3.5 Expert-choice routing

Flip the choice: each expert picks its top-C tokens from the batch (`routing.expert_choice()`). Load is perfectly
balanced by construction, but a token may get zero experts or many, and its experts depend on the other tokens in
the batch — not causal, and not deterministic per request, which rules it out for autoregressive decode. OLMoE
reports that expert choice was not better than dropless token choice in their experiments.

### 3.6 Group-limited (node-limited) routing for EP

With experts spread over nodes, a token routed to 8 experts on 8 different nodes costs 8 cross-node transfers.
DeepSeek-V3 splits its 256 experts into 8 groups of 32, scores each group by the sum of its top-2 biased scores, keeps
the best 4 groups, and picks the top-8 inside them (`moe.limit_groups()`); with one group per node a token talks to
at most 4 nodes. Kimi K2 dropped expert grouping (384 experts, no groups, per its report's comparison table).

### 3.7 Balance at inference: hot experts and domain skew

Training balances experts *on average over the training mix*. A serving workload is not the training mix: a
code-heavy tenant, one language, or one long document favours a few experts. Model the skew as a Zipf popularity
(a model, not a measurement; `touched.touched_mc()`, simulated): for Qwen3-30B-A3B at 16 tokens, s = 1.0 touches
58.0 experts instead of 82.4 and loads the hottest one 13.4× the mean. On one GPU skew *helps* (fewer bytes; §5).
Across GPUs it hurts where the rows set the time: in prefill, and in compute-bound decode at very large batches,
the step waits for the GPU holding the hot experts; in ordinary memory-bound decode every GPU streams its touched
experts whatever their rows, and the skew shows up in the exchange instead (§6.3). Measuring it needs a real
router: the lab's [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb) records per-token expert
choices with router hooks or vLLM's `--enable-return-routed-experts`.

---

## 4. Training MoE in brief

**Communication.** With expert parallelism every MoE layer runs a dispatch and a combine all-to-all in the forward
pass and their transposes in the backward pass — four all-to-alls per layer per step, on top of data-parallel
gradient reductions. Training frameworks (Megatron-LM, MegaBlocks, DeepSpeed) overlap them with compute; the
collective's cost model is layer 02's ([PRIMER §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#5-collectives)).

**Loss terms.** Language-modelling loss + α · aux (§3.2, per micro-batch or per sequence) + a z-loss (§3.2), or the
bias update instead of α (§3.4). Router computations run in fp32 (DeepSeek-V3 keeps its bias in fp32; transformers
upcasts router logits before the softmax).

**Upcycling.** Start from a trained dense model: copy its MLP into every expert, add a fresh router, keep training.
OLMoE's repo documents "sparse upcycling" (e.g. OLMo-1B into an 8-expert MoE) with a conversion script. Upcycled
experts start identical, so balancing and noise are what make them diverge.

**MoE + MLA vs MoE + GQA — the attention side.** MoE shrinks the per-token cost of the MLP; it does nothing for the
KV cache. DeepSeek-V3 pairs MoE with multi-head latent attention, caching a 512 + 64-wide latent per layer:
(512 + 64) × 61 × 2 = 70,272 bytes per token. Qwen3-235B-A22B pairs MoE with GQA (4 KV heads of 128, 94 layers):
192,512 bytes per token (`MoEConfig.kv_bytes_per_token()`). The
[transformer primer §9](../transformers/docs/transformer-primer.md#9-modern-variants-and-why-each-exists) lists both
variants; at long context this choice matters more than the expert count (§7).

**The toy, honestly.** `moecore.train` is full-batch Adam on 512 tokens, 300 steps, a quarter of a second per run.
Plain SGD also collapses without balancing but converges too slowly on this badly scaled toy to show what balance
buys; the torch version with a real tiny transformer is the lab's
[`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb).

---

## 5. MoE at inference: which experts a step touches

Layer 01 states the core result in
[PRIMER §3.6](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#36-moe-which-weights-a-step-streams) with
`roofline.llm.experts_touched()`: its table of experts touched, step bytes and step time for Mixtral-8x7B and
Qwen3-30B-A3B at batches 1–64, and the decode crossover batches. `moecore.touched` reproduces all of it
(`tests/test_touched.py` pins the table to the byte and checks that layer 01's primer still prints it); this
section derives the formula and extends it to fine-grained experts, skew, rows per expert and the KV share.

**The union.** One token misses a given expert with probability 1 − k/E (it picks k distinct experts of E); T
independent tokens miss it with (1 − k/E)^T. So one layer touches

```
experts_touched(E, k, T) = E · (1 − (1 − k/E)^T)
```

distinct experts (`touched.experts_touched()`; a Monte Carlo draw, `touched.touched_mc()`, agrees within 2%).
Every touched expert's weights are streamed from HBM once per step, whatever the number of rows it serves.

The two ends of layer 01's table (1K context, H200; `touched.decode_step()`, the same one-kernel roofline as
`roofline.llm.decode()`): at batch 1 Mixtral reads 2.00 of 8 experts per layer, 25,631,531,008 bytes of which
25,497,182,208 are weights, in 5.34 ms; at batch 64 it reads 8.00 of 8, 101.7 GB, in 21.20 ms. Qwen3 reads 8.0 and
125.9 of 128. All memory-bound: these are bounds, not measurements. Mixtral touches 7.5 of
its 8 experts per layer from batch 10, Qwen3 120 of 128 from batch 43 (notebook 03, exercise 3.2). DeepSeek-V3 (256,
top-8) touches 8.0, 57.4, 163.3, 251.6 and 255.9 experts at 1, 8, 32, 128 and 256 tokens: fine granularity keeps
the saving to larger batches, but by typical decode batches every expert is read every step.

**Skewed vs uniform.** Skew (§3.7) touches fewer experts: Qwen3 at batch 16 under Zipf s = 1.0 reads 58.0 experts
per layer instead of 82.4, a 1.36× faster step on one GPU (simulated; notebook 03, exercise 3.6). Uniform routing is
the conservative assumption for bytes; skew is the conservative assumption for expert-parallel balance.

**The decode crossover.** FLOPs follow active × batch while the bytes approach total, so the batch at which a
decode step turns compute-bound grows with the ratio of weights streamed to weights multiplied. Layer 01's
bisection (c = 0, H200, ridge 206; `touched.decode_crossover_batch()` reproduces it) gives 207 for Llama-3.1-8B,
754 for Mixtral-8x7B and 2,055 for Qwen3-30B-A3B. The ratio that predicts them leaves out the input embedding — a
gather, neither streamed nor multiplied: streamed ÷ multiplied is 3.7 for Mixtral and 9.9 for Qwen3, against
total ÷ active of 3.6 and 9.1 (Qwen3's large vocabulary at small d is why its two ratios differ). Predicting
Qwen3's crossover from the dense one, 207 × 9.9 ≈ 2,057, is within 1% of the bisection; 207 × 9.1 would be 8% low
(notebook 03, exercise 3.4).

**Why: each expert sees B·k/E of the batch.** Inside the step, an expert's GEMM has as many rows as tokens routed to
it — on average B·k/E — and its arithmetic intensity is about that row count. At batch 256 that is 64 rows for
Mixtral, 16 for Qwen3 and 8 for DeepSeek-V3; reaching the H200's ridge of 206 needs batches of 824, 3,296 and 6,592
tokens. **That is why MoE wants big batches**, and why expert parallelism — which pools many GPUs' tokens at each
expert — is how large MoE models are served (§6.5).

**KV unchanged, so the KV/weights ratio shifts.** An MoE step streams more weight bytes than a dense model with the
same active count, for the same KV. At batch 64, 1K context, the KV cache is 8.5% of a Mixtral step but 36.4% of a
Llama-3.1-8B step; at 32K context, 74.7% against 94.8% (`touched.kv_share()`). The MoE stays weight-bound longer; at
long context KV takes over for both.

**Prefix caching unchanged.** Cached prefix blocks hold K and V, which attention produces; a prefix hit skips the
same prefill FLOPs for an MoE as for a dense model (the prefix-caching mechanics are in the
[serving-engine primer §5](../../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching)).

---

## 6. Running MoE on GPUs

### 6.1 Fused MoE kernels

A naive MoE launches one small GEMM per expert per layer. Fused kernels do the §2.5 sort on the GPU and run all
experts in one launch:

1. **Align.** vLLM's `moe_align_block_size(topk_ids, block_size, num_experts, expert_map)` flattens the T·k
   assignments, sorts them by expert and pads each expert's segment to a multiple of `BLOCK_SIZE_M` (§3.3); with EP,
   experts that live on other ranks get id −1 and their blocks are skipped.
2. **Grouped GEMM.** `fused_moe_kernel` (Triton) walks the sorted ids: each program computes one block of rows for
   one expert, GEMM 1 on the gate and up projections (`w1`), then the activation, then GEMM 2 on `w2`.
3. **Combine.** `moe_sum` adds each token's k weighted rows.

Block sizes, warps and stages are tuned per shape: vLLM ships JSON files named
`E={E},N={N},device_name={GPU},dtype=...,block_shape=[...].json` (N = the expert's intermediate size per shard),
mapping batch size M to a kernel config; `benchmarks/kernels/benchmark_moe.py` generates them. There are none for T4,
L4 or A10 (checked at vLLM main `a4eb3f2`, verify for v0.30.0), so on those GPUs expect the warning "Using default MoE
config. Performance might be sub-optimal!" For unquantized weights on CUDA, vLLM prefers FlashInfer TRT-LLM and
CUTLASS backends, then Triton (on SM90, Triton first).

### 6.2 Expert parallelism: dispatch and combine

With **expert parallelism** (EP), each of p GPUs holds E/p whole experts. Every MoE layer moves tokens, not weights:
a **dispatch** all-to-all sends each token's hidden state to the ranks holding its k experts, and a **combine**
all-to-all brings the k weighted results back. Per GPU and direction that is at most

```
bytes = tokens × k × hidden × bytes per element        (every assignment remote: the upper bound)
```

of which (p − 1)/p leaves the GPU under uniform routing (`ep.dispatch_bytes()`). Layer 02's
[§5.6](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#56-how-inference-uses-collectives) prices a Mixtral-like
layer (hidden 4,096, top-2) at 256 tokens per GPU: 4 MiB per direction, **22 µs pairwise or 10 µs direct** on 8 GPUs
with α = 2 µs and 450 GB/s (`ep.a2a_time()` reproduces `gpusim.collectives.model_time("all_to_all", ...)`). Per MoE
layer on 8 GPUs, direct all-to-all (simulated, layer 01's illustrative links):

| Tokens per GPU | Link | Mixtral dispatch (BF16) | DeepSeek-V3 dispatch (FP8 + scales) | DeepSeek-V3 combine (BF16) |
|---|---|---|---|---|
| 8 (decode) | NVLink 4 | 0.12 MiB, 2.3 µs | 0.45 MiB, 2.9 µs | 0.88 MiB, 3.8 µs |
| 8 (decode) | 400 Gb/s IB | 0.12 MiB, 7.3 µs | 0.45 MiB, 13.3 µs | 0.88 MiB, 21.1 µs |
| 4,096 (prefill) | NVLink 4 | 64 MiB, 132.5 µs | 231 MiB, 473.0 µs | 448 MiB, 915.4 µs |
| 4,096 (prefill) | 400 Gb/s IB | 64 MiB, 1,179.4 µs | 231 MiB, 4,243.9 µs | 448 MiB, 8,225.8 µs |

Decode is latency-bound (α is most of it); prefill is bandwidth-bound, and across 400 Gb/s NICs roughly 9× slower
than on NVLink — the deployment primer's rule, "tensor and expert parallelism go inside the NVLink domain"
([gpu-deployment §4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu)),
in numbers. That is why **DeepEP**, DeepSeek's EP library, ships two kernel families: *normal* (high-throughput,
prefill; NVLink then RDMA forwarding within a node) and *low-latency* (decode; pure RDMA, CUDA-graph compatible; V1
offered a hook-based overlap that uses no SMs). The formula explains its published numbers: EP8 low-latency dispatch of 128 tokens × top-8 × (7,168 FP8 bytes + 7,168/128 × 4
bytes of scales) = 7,569,408 B in 77 µs is **98.3 GB/s** (reported 98); the BF16 combine, 14,680,064 B in 114 µs, is
128.8 GB/s (reported 127). DeepEP V2 needs SM90 (Hopper) or newer with NVLink and RDMA — not a T4 or L4 (verify).

### 6.3 The slowest rank, and rebalancing

All ranks wait for each other at every exchange, so a layer takes as long as the slowest rank's expert work plus
the exchanges (`ep.layer_time()`). Each rank's expert work is a roofline of its own: its rows × 2 × expert
parameters ÷ peak against its touched experts × expert bytes ÷ bandwidth (`ep.touched_per_rank()`). Qwen3-30B-A3B's
128 experts on 8 H100s (16 per GPU), NVLink, all-to-all kernels, simulated:

| Tokens per GPU | Routing | Placement | Rows on the busiest rank ÷ mean | Slowest rank bound by | Layer time | vs balanced |
|---|---|---|---|---|---|---|
| 128 (decode) | uniform | linear | 1.04 | weight reads | 65.9 µs | 1.01× |
| 128 (decode) | Zipf s = 1.0 | linear (contiguous blocks, vLLM's default) | 1.63 | weight reads | 75.4 µs | 1.15× |
| 4,096 (prefill chunk) | uniform | linear | 1.01 | FLOPs | 844.0 µs | 1.01× |
| 4,096 (prefill chunk) | Zipf s = 1.0 | linear | 1.65 | FLOPs | 1,378.2 µs | 1.64× |
| 4,096 (prefill chunk) | Zipf s = 1.0 | round-robin (not applied to this model; below) | 1.46 | FLOPs | 1,225.9 µs | 1.46× |

**Decode: the skew moves to the exchange.** At 128 tokens per GPU each expert sees about 64 rows, far below the
H100's ridge of 295, so every rank spends 45.1 µs streaming its 16 experts' weights whatever the skew; the hot
rank's extra rows (16.0 µs of FLOPs against a mean of 9.8) hide under that read. What the skew costs in decode is
the exchange: the hot rank's port receives the dispatch and sends the combine for 1.5× the mean traffic (15.2 µs per
all-to-all against 10.2), so the layer is 1.15× slower. **Prefill: the rows set the time.** At 4,096 tokens per
GPU the expert GEMMs are compute-bound, and the busiest rank's 1.65× rows make the layer 1.64× slower than a
balanced one — on every layer of every prefill chunk (notebook 04, exercise 4.4). Decode behaves the same way once
its batch pushes the experts past the ridge (§5).

**Placement.** vLLM's `--expert-placement-strategy round_robin` puts expert e on rank e mod p instead of in
contiguous blocks, which here would move the hot spot (1.65 → 1.46 rows) without removing it. vLLM v0.30.0 applies
it only to models with more than one expert group (DeepSeek-V3's 8), no redundant experts and EPLB off — with
all-to-all kernels, only on the DeepEP low-latency or NIXL-EP backends — and otherwise logs a warning and falls back
to linear (`determine_expert_placement_strategy()` in `fused_moe/expert_map_manager.py`, v0.30.0 and main `a4eb3f2`;
verify for your version). For Qwen3-30B-A3B, which has no expert groups, the round-robin row is a placement vLLM
would not apply.

**EPLB** (`--enable-eplb`, window 1,000 steps, rebalance every 3,000 by default) measures load and re-places
experts, replicating the hottest (`num_redundant_experts`). A greedy version (`ep.rebalance()`) on the skewed decode
batch: re-placement alone takes the busiest rank from 1.63 to 1.08 max/mean; eight redundant copies take it to 1.01.
Replicas cost HBM: vLLM's EP guide gives ~2.4 GB for one redundant DeepSeek-V3 expert per rank, which is 58 MoE
layers × 44,040,192 B in FP8 = 2.38 GiB (`ep.wide_ep_weights(redundant=)`).

### 6.4 TP vs EP for experts, and the hybrid

Without EP, experts are just MLPs and can be **tensor-parallel**: every GPU holds 1/p of every expert, all GPUs see
all tokens, and an all-reduce restores each layer's output (serving-engine
[§9](../../04-inference-engine/serving-engine/PRIMER.md#9-parallelism-inside-the-engine)). With EP each GPU holds
whole experts and ships only assignments. For a Mixtral MoE layer at batch 64 on 8 GPUs (`ep.moe_comm()`, simulated):
TP's ring all-reduce sends 896 KiB per GPU in 30.0 µs; EP with data-parallel attention and dedicated all-to-all
kernels (DeepEP, NIXL-EP, FlashInfer's; `mode="a2a"`) sends 224 KiB per GPU in 4.5 µs. TP also splits each
expert's GEMM p ways, making already-thin GEMMs thinner. The **hybrid** is the usual shape of a large deployment:
attention replicated (or tensor-parallel inside each data-parallel group), experts expert-parallel across all the
GPUs.

vLLM's semantics (v0.30.0): `--enable-expert-parallel` makes the MoE layers expert-parallel over **EP = TP × DP**
ranks — EP size is not a flag of its own. Attention is replicated across DP ranks when `--tensor-parallel-size 1`
and TP-sharded within each DP group when TP > 1. **Without** the flag, MoE layers run tensor-parallel over the same
TP × DP group. `--all2all-backend` picks the exchange (default `allgather_reducescatter`; `deepep_low_latency` for
decode and `deepep_high_throughput` for prefill across nodes; `pplx` and `naive` were removed; there is no
`VLLM_ALL2ALL_BACKEND` environment variable) (verify against your version).

**What vLLM actually puts on the wire.** All-to-all kernels run only with EP and DP > 1 (or prefill context or
sequence parallelism; `FusedMoEParallelConfig.use_all2all_kernels`, `fused_moe/config.py`, v0.30.0), which leaves
three cases:

- **TP × EP with DP = 1** (`--tensor-parallel-size 2 --enable-expert-parallel`): after attention every GPU already
  holds every token, so there is no all-to-all at all. Each GPU runs its own experts on the tokens routed to them
  and one all-reduce sums the outputs — TP's traffic (`mode="tp"`), EP's memory layout.
- **DP > 1 with the default `allgather_reducescatter`**: every rank all-gathers every rank's tokens (tokens ×
  hidden, not × k) and reduce-scatters the outputs back, (p − 1) × tokens per GPU × hidden × bytes each way — here
  the same 896 KiB and 30.0 µs as TP (`mode="agrs"`).
- **DP > 1 with an all-to-all backend** (`deepep_low_latency`, `deepep_high_throughput`, `nixl_ep`, …): the
  dispatch and combine of §6.2, the 224 KiB. DeepEP needs SM90 or newer with NVLink and RDMA, so not a T4 or L4.

So EP's traffic saving needs data-parallel attention *and* all-to-all kernels; on two PCIe GPUs what
`--enable-expert-parallel` changes is the memory layout and the balance, not the bytes on the wire (the lab's
[`04_expert_parallelism_on_two_gpus`](moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb) measures exactly
that). SGLang and TensorRT-LLM serve large MoE models with the same two layouts under different flags; which engine
to pick is the serving-engine primer's [§12](../../04-inference-engine/serving-engine/PRIMER.md#12-engines-and-where-to-run-them).

### 6.5 Data-parallel attention + EP (wide-EP)

Large deployments combine **data-parallel attention** — every rank runs attention for its own requests with its
own KV cache — with **EP** for the experts: "wide-EP" ([layer 05 §8](../../05-orchestrator/serving-orchestration/PRIMER.md#8-large-moe-topologies-wide-ep-in-brief)).
Attention, dense layers, shared experts, routers and embeddings are replicated on every rank; only routed experts
shrink with EP (`ep.wide_ep_weights()`). DeepSeek-V3 in FP8 on H200s (141 GB):

| EP degree | Weights per GPU | Share of HBM | 4K-token sequences that fit |
|---|---|---|---|
| 8 | 98.9 GB | 70% | 776 |
| 16 | 58.0 GB | 41% | 3,824 |
| 32 | 37.6 GB | 27% | 9,920 |
| 64 | 27.3 GB | 19% | 22,080 |

(`sizing.sessions(layout="ep")`.) 17.1B parameters are replicated on every rank; by EP 64 they are most of each
GPU's weights (17.1 of 27.3 GB). Two consequences. First, DP ranks are not independent: forward passes are aligned, and an idle rank runs
**dummy forward passes** while any rank has work (vLLM's DP coordinator). Second, EP 16 means two 8-GPU nodes, so
half of each GPU's all-to-all traffic crosses the scale-out network. DeepSeek-V3 at batch 512, 4K context, EP 16,
FP8 dispatch and BF16 combine: 0.9 ms of all-to-alls per step if all 16 GPUs shared one NVLink domain (an
NVL72-class rack), 3.8 ms as two 8-GPU nodes — 7 peers over NVLink, 8 over one 400 Gb/s InfiniBand NIC per GPU —
which is 22% of a 17.6 ms step with no overlap (`ep.decode_on(per_node=8, intra=...)`, simulated). Putting every
peer behind the NIC, the upper bound, gives 6.6 ms. Hence NVL72-class racks, DeepEP's RDMA kernels, and two-batch
overlap (vLLM `--enable-dbo`, SGLang TBO).
llm-d's wide-EP guide runs DeepSeek-R1 on 32 H200 or B200 GPUs as 16-way DP prefill plus 16-way DP decode; SGLang's
single-node form is `--tp 8 --dp-size 8 --ep 8 --enable-dp-attention --moe-a2a-backend deepep` (verify).

### 6.6 Expert offloading for small GPUs

When the experts do not fit, keep some in CPU memory:

- **vLLM** `--cpu-offload-gb N` puts N GiB of weights per GPU in CPU memory and reads them through unified virtual
  addressing during each forward pass; `--cpu-offload-params experts` restricts it to expert weights (v0.30.0). The
  docs say offloaded weights are loaded "on the fly in each model forward pass" — count all of them per step as the
  upper bound (verify what a given kernel actually touches).
- **llama.cpp** `--cpu-moe` keeps all expert weights in CPU memory and `--n-cpu-moe N` the first N layers'
  experts. In small-batch decode the CPU computes those experts, so mostly activations cross PCIe. For large
  batches (prompt processing) llama.cpp by default copies the host-resident weights to the GPU and computes there
  (`--op-offload`, default on; `--no-op-offload` keeps the work on the CPU), and then the weights do cross PCIe
  (from `common/arg.cpp` and the buffer-override semantics; verify).

The PCIe bill is brutal. Budget one small GPU the way vLLM does: 0.92 of the 15.0 GiB a T4 reports, minus ~1.5 GiB
of activations and CUDA graphs (verify), is 12.3 GiB (`sizing.kv_room_gib()`, the same budget as the lab's
`moelab.offload.fit`). OLMoE-1B-7B in fp16 is 13.8 GB = 12.9 GiB of weights, so on a 16 GB T4 it has no room for KV
at all. The smallest offload that holds four 4K-token sequences is 3.0 GiB (`sizing.min_offload_gib()`, rounded up
to 0.5 GiB; the lab prints `--cpu-offload-gb 3`). Streaming it over PCIe Gen3 (~12 GB/s effective, verify) adds
~268 ms to every step, against a 26 ms step at batch 4 — about 11× slower (`sizing.offload_step_s()`,
`touched.decode_step()`, simulated). A 4-bit checkpoint is usually the better trade: with 4-bit experts OLMoE is
4.4 GB and leaves room for 67,377 tokens of KV on the same T4; so is computing the offloaded experts on the CPU.
The lab's [`05_moe_on_a_small_gpu`](moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb) reads these numbers back from
vLLM's start-up log and measures the step.

### 6.7 Quantized experts

Experts are most of an MoE's bytes, so they are what quantization targets. The formats (MXFP4 among them) are in
the quantization primer's [§2](../../04-inference-engine/quantization/PRIMER.md#2-number-formats), why routers stay
in 16-bit in its [§5](../../04-inference-engine/quantization/PRIMER.md#5-weight-and-activation-quantization), and
calibrating rarely routed experts in its [§8](../../04-inference-engine/quantization/PRIMER.md#8-measuring-the-accuracy-you-pay)
(llm-compressor's `moe_calibrate_all_experts`, and ignoring `mlp.gate`). The MoE-specific numbers:

- **gpt-oss ships MXFP4 experts** (4.25 bits per weight) and everything else in bf16. That makes gpt-oss-120b
  **65.2 GB** (one 80 GB GPU) and gpt-oss-20b **13.8 GB** (`sizing.weight_bytes(expert_bits=4.25)`). vLLM's MXFP4
  path requires compute capability 8.0 and bf16 activations — L4 and RTX 4090 yes, T4 no.
- **DeepSeek-V3 ships FP8** weights with 128 × 128 block scales, activations quantized per 128 channels.

---

## 7. Sizing and cost

The recipe (`moecore.sizing`, notebook 05):

1. **Memory by total + KV.** Weights at the chosen precision (`sizing.weight_bytes()`) plus the batch's KV cache
   (`MoEConfig.kv_bytes_per_token()` × context × sequences): 10% headroom on nominal GB for fleet sizing
   (`sizing.sessions()`, layer 01's round budget); on one small GPU, where the margin is the question, vLLM's own
   budget (`sizing.kv_room_gib()`, §6.6).
2. **Prefill by active.** 2 × active × tokens (`sizing.prefill_flops()`), divided by an achieved fraction of the peak.
3. **Decode by the bytes streamed at the batch.** Touched experts + everything else + KV (§5), per GPU in the chosen
   layout (`ep.decode_on()`), plus two exchanges per MoE layer (all-to-alls with DeepEP-class kernels, vLLM's
   default all-gather and reduce-scatter otherwise; §6.4).
4. **GPUs and EP degree.** The smallest count that holds weights + KV and meets the ITL target (`sizing.plan()`).
5. **Cost.** GPUs × $/GPU-hour ÷ tokens per second (`sizing.usd_per_mtok()`).

**Memory, worked.** Qwen3-30B-A3B is a 30B model for memory: 61.1 GB in bf16, 30.5 GB in FP8, 18.5 GB with 4.25-bit
experts and the rest in bf16 — only the last fits a 24 GB L4. vLLM's budget (0.92 × 22.49 GiB − 1.5 GiB) then leaves
2.0 GiB for KV: 21,593 tokens, five sequences of 4K (`sizing.kv_tokens()`; the lab's `python -m moelab fit` prints
the same). The round 10%-headroom budget would say 7; on one small GPU the overheads are the whole margin. The 3B
active count only helps the FLOPs.

**Prefill, worked.** An 8,192-token prompt costs a dense Llama-3.1-70B 5.48× the FLOPs it costs Mixtral-8x7B; at
50% of an H100's bf16 peak Mixtral prefills it in 427 ms (simulated). Prefill is where the FLOP saving shows in full.

**Plans, worked** (4K context; prices are illustrative placeholders per GPU-hour, see [`COMPUTE.md`](../../COMPUTE.md);
all times simulated):

| Case | Batch | ITL target | GPUs | Step | Tokens/s | $/M tokens at the placeholder price |
|---|---|---|---|---|---|---|
| Mixtral-8x7B, bf16, H100, DP attention + EP (all-to-all kernels) | 64 | 50 ms | 2 | 19.6 ms | 3,259 | 0.51 at $3/GPU-h |
| Llama-3.1-70B, bf16, H100, tensor parallel | 64 | 50 ms | 4 | 19.3 ms | 3,322 | 1.00 at $3/GPU-h |
| DeepSeek-V3, FP8, H200, wide-EP (all-to-all kernels, FP8 dispatch) | 256 | 50 ms | 8 | 23.2 ms | 11,046 | 0.60 at $3/GPU-h |
| Qwen3-30B-A3B, FP8, L4, DP attention + EP (vLLM's default `allgather_reducescatter` over PCIe) | 32 | 60 ms | 4 | 41.0 ms | 780 | 1.00 at $0.7/GPU-h |

The H100 and H200 EP rows assume all-to-all kernels (DeepEP-class, which need Hopper with NVLink and RDMA); the L4
row uses vLLM's default exchange, since DeepEP does not run on an L4, over an assumed PCIe peer-to-peer link of
12 GB/s and α = 15 µs (`ep.LINKS["pcie-l4"]`, the lab's `pcie-2xL4` value; fit your own with layer 02's lab). At
batch 64 Mixtral costs about half as much per token as the dense 70B; whether the two are of "similar quality"
depends on versions and on your evals (verify) — the method, not the verdict, is the point. DeepSeek-V3's step
(23.2 ms) sits above its all-experts floor (17.5 ms, `sizing.decode_floor()`) by each rank's KV reads, its replicated
weights and 0.9 ms of all-to-alls.

**Long context, worked.** Mixtral on 2 H100s: at 4K context 88 sequences fit and batch 16 runs at 1,024 tokens/s;
at 32K only 10 fit (497 tokens/s at batch 10); at 128K, 2 (166 tokens/s). The KV cache, not the experts, sets both
the memory and the step once context is long — MoE's advantage shrinks with the batch that fits.

**The families, dated.** Configs as data in `sizing.MODELS` (read from upstream code on 2026-09-26; "verify" where a
field was derived), plus the 2026 models the [open-weight primer §4](../model-landscape/open-weight-llms-primer.md#4-the-families-vendor-by-vendor)
reports (no configs here: numbers cited, not recomputed, all verify):

| Model | Routed E | k | Shared | d | Total / active | Source |
|---|---|---|---|---|---|---|
| Mixtral-8x7B | 8 | 2 | 0 | 4,096 | 46.70B / 12.88B | config |
| OLMoE-1B-7B | 64 | 8 | 0 | 2,048 | 6.92B / 1.28B | config; expert width derived (verify, 2026-09-26) |
| Qwen1.5-MoE-A2.7B | 60 | 4 | 1 (4× wide) | 2,048 | 14.32B / 2.69B | config |
| Qwen3-30B-A3B | 128 | 8 | 0 | 2,048 | 30.53B / 3.35B | config |
| Qwen3-235B-A22B | 128 | 8 | 0 | 4,096 | 235.09B / 22.19B | report; d and expert width 1,536 derived (verify, 2026-09-26) |
| gpt-oss-20b / 120b | 32 / 128 | 4 | 0 | 2,880 | 20.91B / 3.61B; 116.83B / 5.13B (head only) | reference code; 20b's 24 layers derived (verify, 2026-09-26) |
| Llama 4 Scout / Maverick | 16 / 128 | 1 | 1 | 5,120 | 107.77B / 17.17B; 400.71B / 17.18B (text) | Scout: config; Maverick: card 17B / 400B, MoE interleave and dense width 16,384 derived (verify, 2026-09-26) |
| DeepSeek-V3 | 256 | 8 | 1 | 7,168 | 671.03B / 37.55B | config |
| Kimi K2 | 384 | 8 | 1 | 7,168 | 1.04T / 32.6B | report (README: 1T / 32B) |
| Mistral Large 3 | — | — | — | — | 675B / 41B | capacity and open-weight primers (verify) |
| DeepSeek V4-Pro / V4-Flash | — | — | — | — | 1.6T / ~49B; 284B / ~13B | open-weight primer (verify) |
| Qwen3.8-2.4T-A95B | 512 | 10 | 1 | — | 2.4T / ~95B | open-weight primer (verify) |
| Kimi K3 | 896 | 16 | yes | — | 2.8T / 104B | open-weight primer (verify) |
| MiniMax M3; Nemotron 3 Nano / Super / Ultra | — | — | — | — | 428B / ~23B; ~31.6B / 3.2B, ~120B / 12B, ~550B / ~55B | open-weight primer (verify) |
| Gemma 4 26B MoE | — | — | — | — | 26B / ~4B | open-weight primer (verify) |

---

## 8. In a design review: failure modes

| Failure mode | What you see | The number behind it | What to do |
|---|---|---|---|
| **MoE at batch 1** | a 47B model's memory for 13B-class decode speed; at small batches above 1, slower than a dense model of the active size | Mixtral on an H200 at batch 1: 5.34 ms, the same as a dense model of its 12.9B active size (`sizing.dense_equivalent()`), but 93 GB of weights resident against 26 GB; at batch 4 and 16 it streams 65.1 and 94.4 GB against 26.0 and 27.6 GB, 2.5× and 3.4× slower | serve MoE where traffic gives big batches; for low, bursty traffic a dense model of the active size is cheaper |
| **EP across a slow fabric** | tokens/s flat or falling as GPUs are added past one node | DeepSeek-V3, EP 16 as two 8-GPU nodes on 400 Gb/s IB: all-to-alls 3.8 ms per step (22% of the step) against 0.9 ms in one NVLink domain | keep EP inside the NVLink domain; DeepEP RDMA kernels; overlap (DBO/TBO); larger scale-up domains |
| **Hot experts** | in prefill, one GPU at 100% while others idle and the step tracks the busiest rank; in decode, one GPU's link the busiest | Zipf-skewed Qwen3 on 8 H100s: 1.65× rows on the busiest rank make a prefill layer 1.64× slower; in decode the weight reads stay balanced and the hot rank's port makes the layer 1.15× slower | EPLB with redundant experts (~2.4 GiB each for DeepSeek-V3 FP8), placement strategy (round-robin only for grouped models), balance per tenant |
| **MoE on one 24 GB GPU** | out of memory at load, or no KV room | Qwen3-30B-A3B: 61.1 GB bf16, 30.5 GB FP8, 18.5 GB with 4-bit experts (5 sequences of 4K at vLLM's defaults) | quantize the experts; offload only if you can afford PCIe per step (OLMoE fp16 on a T4: 3 GiB offloaded, ~11× slower at batch 4) |
| **Long context where KV dominates** | batch that fits collapses; the MoE's cost advantage shrinks | Mixtral on 2 H100s: 88 sequences at 4K, 10 at 32K, 2 at 128K | KV quantization, MLA-style attention, prefix caching, P/D disaggregation (layer 05) |
| **Quantizing experts** | quality drops on some domains only | rarely routed experts get too few calibration tokens; the router is sensitive | keep gates in high precision; calibrate every expert (llm-compressor `load_context()`); evaluate per domain |
| **Dense vs MoE for a workload** | the "cheaper" model is more expensive in production | at batch 64: Mixtral $0.51 vs dense 70B $1.00 per M tokens (placeholder prices); at batch 1 the MoE only matches a dense model of its active size, on 3.6× the memory | decide with the plan: traffic (batch), hardware you can get (HBM, NVLink), context length, quality on your evals |

---

## 9. Where to run it

Every concept in this primer is learnable at T0; hardware makes the numbers real. Prices and how to get each option
are in [`COMPUTE.md`](../../COMPUTE.md).

| To learn | T0 (laptop / Colab CPU, $0) | T1 (one small GPU) | T2 (two or more GPUs) | T3 (GCP, optional) |
|---|---|---|---|---|
| the layer, routers, parameter counts (§2) | [`moe-core` notebook 01](moe-core/notebooks/01_the_moe_layer.ipynb) | router hooks on a small open MoE (lab [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb)) | — | — |
| balance and collapse (§3–4) | [`moe-core` notebook 02](moe-core/notebooks/02_routing_and_load_balance.ipynb) (numpy); lab [`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb) (torch on CPU) | the same, faster | — | — |
| experts touched, decode vs batch (§5) | [`moe-core` notebook 03](moe-core/notebooks/03_which_experts_a_batch_touches.ipynb) (simulated) | lab [`03_batch_vs_weight_stream`](moe-lab/notebooks/03_batch_vs_weight_stream.ipynb): vLLM step time vs batch, MoE vs dense | — | — |
| EP and all-to-all (§6.2–6.5) | [`moe-core` notebook 04](moe-core/notebooks/04_expert_parallelism_and_all_to_all.ipynb) (simulated) | — | lab [`04_expert_parallelism_on_two_gpus`](moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb): Kaggle's free 2×T4 (PCIe), `--enable-expert-parallel` vs TP | the 02 lab's `l4x2` pool (2 × L4, PCIe) with the MoE lab's [`deploy/gke/`](moe-lab/deploy/gke/README.md) manifests |
| sizing, offload, INT4 experts (§6.6–7) | [`moe-core` notebook 05](moe-core/notebooks/05_sizing_and_cost.ipynb) | lab [`05_moe_on_a_small_gpu`](moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb) | — | — |

**What the two-GPU runs show.** On two PCIe GPUs vLLM runs no dispatch/combine all-to-all (§6.4): TP × EP with
DP = 1 all-reduces, and DP = 2 uses `allgather_reducescatter`. So the T2 and T3 EP runs compare memory layout
(half the experts per GPU vs half of every expert), balance across the two GPUs, and the all-reduce vs
all-gather/reduce-scatter traffic; dispatch and combine appear only with DP > 1 and DeepEP-class backends on
Hopper-class GPUs.

**Candidate models for T1** (ids and fits are verify): `allenai/OLMoE-1B-7B-0924` (6.9B; fp16 on a 24 GB GPU, or a
T4 with `--cpu-offload-gb`), `Qwen/Qwen1.5-MoE-A2.7B` in INT4 (fits a T4), `Qwen/Qwen3-30B-A3B` in INT4 on a 24 GB GPU,
and the small granite-3 MoE models (`ibm-granite/granite-3.0-1b-a400m-*`, `3b-a800m-*`). Free: Colab's T4, Kaggle's
T4 or 2×T4; rented: a 24 GB card on RunPod or Vast.ai (containers) or Lambda (VMs), or a GCP L4 Spot VM. A T4 has
no bf16 (`--dtype half`) and no FP8, and cannot run gpt-oss's MXFP4 path; an L4 or RTX 4090 can. On **GCP**, the cuda-and-nccl lab's
Terraform ([`02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/`](../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/))
creates a Spot `l4x2` pool (`g2-standard-24`, 2 × L4, PCIe, scales from zero) that the MoE lab's GKE manifests
target with `--enable-expert-parallel`; no new Terraform. DeepEP and wide-EP need Hopper-class GPUs with NVLink and
RDMA — a rented 8-GPU H100/H200 box (RunPod, Vast.ai, Lambda) or GCP A3 shapes, a T2/T3 session measured in hours
and dollars.

---

## In a design review

**The two-minute walkthrough.** "An MoE layer swaps the MLP for E expert MLPs and a router that sends each token to
its top-k. That separates what the model stores from what each token costs: Mixtral holds 46.7B parameters and runs
12.9B per token; DeepSeek-V3 holds 671B and runs about 37B. Training has to impose balance — an auxiliary loss, or
DeepSeek-V3's selection-only bias — or the router collapses onto a few experts. At inference, three numbers size
the deployment: memory by the *total* plus KV, prefill by the *active*, and decode by what a step *streams*. A
decode step reads every expert any token in the batch picked, E(1 − (1 − k/E)^T) per layer, so an MoE is a small
model at batch 1 and streams nearly all its weights by batch 16–64, and it turns compute-bound only at batches
about total ÷ active (more precisely, weights streamed ÷ weights multiplied) times a dense model's — 754 for
Mixtral on an H200. So we serve MoE at large batch with expert parallelism: each GPU owns some experts, each layer
exchanges tokens (two all-to-alls with DeepEP-class kernels), and data-parallel attention keeps each rank's KV
local. We keep EP inside the NVLink domain, rebalance hot experts with a few redundant copies, and
quantize experts but not the router. The KV cache is whatever the attention says; at long context it dominates
again. For low-traffic workloads on one GPU, a dense model of the active size is usually the better choice."

**Drill questions**

1. *"Qwen3-30B-A3B is 3B active, so it runs on my 24 GB GPU."* — Memory follows the total: 61.1 GB in bf16, 30.5 GB
   in FP8. Only 4-bit experts (18.5 GB) fit, leaving room for about 5 sequences of 4K tokens at vLLM's defaults
   (21,593 tokens of KV). The 3B helps FLOPs, not memory.
2. *Why does decode turn compute-bound at batch 754 for Mixtral but 207 for Llama-3.1-8B on an H200?* — The step's
   weight bytes approach the total (all experts touched) while FLOPs follow the active parameters; the ratio of
   streamed to multiplied weights is 3.7, and each expert sees only B·k/E of the batch.
3. *Your aux loss reads 2.0 on a perfectly balanced router. Is something wrong?* — No: transformers' convention
   counts all k assignments, so uniform routing scores k (2 for top-2); Megatron's divides by k and scores 1. Check
   the convention before comparing coefficients.
4. *Why does DeepSeek-V3's balancing bias not change the model's outputs directly?* — It is added to the scores
   only to choose experts; the combine weights come from the unbiased scores. It steers load without biasing the
   gradient of the task loss.
5. *EP = 16 across two nodes is slower per token than EP = 8 in one. Why?* — Half of each GPU's all-to-all traffic
   now crosses the network (for DeepSeek-V3 at batch 512: 3.8 ms of all-to-alls per step across two nodes on
   400 Gb/s vs 0.9 ms in one NVLink domain), the replicated attention weights are a bigger share of each GPU's
   bytes, and the busiest rank still sets the step.
6. *A hot expert keeps one GPU at 100% during prefill. Three fixes and their costs?* — EPLB-style replication of
   hot experts (~2.4 GiB of HBM per redundant DeepSeek-V3 expert per GPU), a different placement (moves the hot spot,
   may not remove it, and vLLM applies round-robin only to grouped models), or routing work by tenant or domain
   across replicas (layer 05) — and re-measure, since skew is a property of the workload. In memory-bound decode the
   same skew shows up on the hot GPU's link rather than its GEMMs.

---

## Glossary

- **Active parameters** — parameters one token multiplies by; published counts differ on whether embedding tables
  are included (§2.4).
- **All-to-all** — collective in which rank r's chunk j goes to rank j; MoE's dispatch and combine.
- **Auxiliary (balance) loss** — E · Σ f_e · P_e, added to the training loss to spread tokens over experts.
- **Capacity factor** — multiplier on the average rows per expert (k · T / E) that caps each expert's work; overflow
  tokens are dropped.
- **Combine** — the all-to-all that returns expert outputs to the tokens' home ranks, weighted and summed.
- **DeepEP** — DeepSeek's expert-parallel communication library: high-throughput and low-latency all-to-all kernels.
- **Dispatch** — the all-to-all that sends tokens' hidden states to the ranks holding their experts.
- **Dropless** — MoE that processes every assignment (no capacity), padding blocks instead; all inference engines.
- **EPLB** — expert-parallel load balancer: measures expert load and re-places or replicates experts.
- **Expert** — one of the E MLPs in an MoE layer.
- **Expert choice** — routing where each expert picks its top tokens instead of tokens picking experts.
- **Expert parallelism (EP)** — placing different experts on different GPUs and moving tokens to them.
- **Fine-grained experts** — many narrow experts with a larger k, at the same parameters and FLOPs as few wide ones.
- **Fused MoE kernel** — one launch that sorts assignments by expert and runs all experts as a grouped GEMM.
- **Group-limited routing** — restricting a token's experts to a few groups (nodes) to bound cross-node traffic.
- **Grouped GEMM** — many independent matrix multiplies of different row counts in one kernel.
- **Hot expert** — an expert that receives far more than the mean load for a given workload.
- **MXFP4** — 4-bit floating-point values with a shared 8-bit scale per 32 values (4.25 bits per weight).
- **Router (gate)** — the linear layer that scores experts for each token.
- **Router collapse** — a few experts receiving nearly all tokens because routing reinforces early winners.
- **Shared expert** — an expert every token uses with weight 1, beside the routed ones.
- **Top-k** — the k highest-scoring experts a token is routed to.
- **Upcycling** — initialising an MoE from a trained dense model by copying its MLP into every expert.
- **Wide-EP** — data-parallel attention plus expert parallelism across many GPUs.
- **z-loss** — mean of logsumexp(router logits)²; keeps logits small for numerical stability.

---

## Sources

Papers and reports (arXiv ids as cited by the upstream repos; papers were not readable from this build environment,
so numbers taken only from them are marked verify):

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017), arXiv:1701.06538.
- Lepikhin et al., *GShard* (2020), arXiv:2006.16668. Fedus, Zoph, Shazeer, *Switch Transformers* (2021), arXiv:2101.03961.
- Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (2022, router z-loss), arXiv:2202.08906.
- Zhou et al., *Mixture-of-Experts with Expert Choice Routing* (2022), arXiv:2202.09368.
- Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (2022), arXiv:2211.15841; <https://github.com/databricks/megablocks>.
- Jiang et al., *Mixtral of Experts* (2024), arXiv:2401.04088. Dai et al., *DeepSeekMoE* (2024), arXiv:2401.06066.
- Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (2024), arXiv:2408.15664.
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024), arXiv:2412.19437; <https://github.com/deepseek-ai/DeepSeek-V3> (`inference/model.py`, `inference/configs/config_671B.json`).
- Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models* (2024), arXiv:2409.02060; <https://github.com/allenai/OLMoE>.
- Qwen Team, *Qwen3 Technical Report* (2025), arXiv:2505.09388. Kimi Team, *Kimi K2* technical report, <https://github.com/MoonshotAI/Kimi-K2>.
- OpenAI, gpt-oss model card and reference code, <https://github.com/openai/gpt-oss>. Meta, Llama 4 model card and reference code, <https://github.com/meta-llama/llama-models> (`models/llama4/moe.py`).

Code and docs read for this primer (September 2026):

- Hugging Face transformers (`models/{mixtral,qwen2_moe,qwen3_moe,olmoe,granitemoe,deepseek_v3,llama4,gpt_oss}`), <https://github.com/huggingface/transformers>.
- Megatron-LM `megatron/core/transformer/moe/moe_utils.py` (Switch loss, z-loss, `get_updated_expert_bias`), <https://github.com/NVIDIA/Megatron-LM>.
- vLLM v0.30.0: `vllm/model_executor/layers/fused_moe/` (`moe_align_block_size`, `fused_moe`, tuned configs), `vllm/config/parallel.py`,
  `vllm/config/offload.py`, `docs/serving/expert_parallel_deployment.md`, `docs/serving/data_parallel_deployment.md`, <https://github.com/vllm-project/vllm>.
- DeepEP, <https://github.com/deepseek-ai/DeepEP> (README and `docs/legacy.md` for the V1 numbers).
- SGLang docs, `advanced_features/expert_parallelism.mdx` and `dp_dpa_smg_guide.mdx`, <https://github.com/sgl-project/sglang>.
- llama.cpp `common/arg.cpp` (`--cpu-moe`, `--n-cpu-moe`, `--op-offload`), <https://github.com/ggml-org/llama.cpp>.
- llm-compressor `examples/quantizing_moe/`, <https://github.com/vllm-project/llm-compressor>.
- llm-d wide-EP guide (via layer 05), <https://github.com/llm-d/llm-d>.

Repo material this primer builds on: [transformer primer](../transformers/docs/transformer-primer.md) §9;
[capacity primer](../gpu-capacity-planning/PRIMER.md) and `capacity.py`;
[roofline-and-fabric PRIMER §3.6](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#36-moe-which-weights-a-step-streams)
and `roofline-core/roofline/llm.py`; [gpu-deployment §4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu);
[cuda-and-nccl PRIMER §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#5-collectives) and `cuda-nccl-core/gpusim/collectives.py`;
[serving-engine PRIMER §9 and §12](../../04-inference-engine/serving-engine/PRIMER.md#12-engines-and-where-to-run-them);
[serving-orchestration PRIMER §8](../../05-orchestrator/serving-orchestration/PRIMER.md#8-large-moe-topologies-wide-ep-in-brief);
[quantization PRIMER §2, §5, §8](../../04-inference-engine/quantization/PRIMER.md#2-number-formats);
[open-weight primer §4](../model-landscape/open-weight-llms-primer.md#4-the-families-vendor-by-vendor).

---

## Verify list

Dated 2026-09-26. Re-check before relying on any of these.

- **vLLM v0.30.0** (the repo's pinned version; flags re-checked against the v0.30.0 tag): `--enable-expert-parallel`;
  EP size = TP × DP; `--all2all-backend` default `allgather_reducescatter`, `pplx` and `naive` removed, no
  `VLLM_ALL2ALL_BACKEND` variable; `--enable-eplb` with `EPLBConfig` defaults window 1,000, step interval 3,000,
  0 redundant experts; `--expert-placement-strategy linear | round_robin`; `--enable-dbo`; `--cpu-offload-gb`,
  `--cpu-offload-params`; `--enable-return-routed-experts`. MXFP4 minimum compute capability 8.0 with bf16
  activations.
- **fused_moe tuned configs**: none for T4, L4 or A10 at vLLM main `a4eb3f2`; check the v0.30.0 wheel.
- **DeepEP**: V2 requires SM90+, CUDA ≥ 12.3, PyTorch ≥ 2.10, NVLink intranode and RDMA internode; `deepep_v2` in vLLM
  needs NCCL ≥ 2.30.4. Published V1 numbers are H800 with ConnectX-7 400 Gb/s.
- **SGLang** flags (`--moe-a2a-backend`, `--enable-dp-attention`, `--ep`) as of `3ed56a3`.
- **Model configs** derived rather than read: Qwen3-235B-A22B d and expert width; Llama 4 Maverick's MoE-every-2nd-layer
  interleave and dense width 16,384; gpt-oss-20b's 24 layers; OLMoE's expert width; the granite-3.0 MoE configs; Kimi
  K2's attention ranks. Released Qwen3 MoE configs' `norm_topk_prob`.
- **Model ids for T1**: `allenai/OLMoE-1B-7B-0924(-Instruct)`, `Qwen/Qwen1.5-MoE-A2.7B(-Chat)` and its GPTQ-Int4,
  `Qwen/Qwen3-30B-A3B-GPTQ-Int4`, `Qwen/Qwen3-30B-A3B-FP8`, `ibm-granite/granite-3.0-{1b-a400m,3b-a800m}-instruct`.
- **2026 model numbers** (Kimi K3, DeepSeek V4, Qwen3.8, MiniMax M3, Nemotron 3, Gemma 4, Mistral Large 3) — from the
  open-weight primer, dated there.
- **Values from papers not readable here**: Switch's α = 0.01 and capacity factors 1.0–1.25; ST-MoE's z-loss 1e-3;
  DeepSeek-V3's sequence-wise α = 0.0001 and bias-rate schedule.
- **Hardware and links**: GPU peaks and bandwidths (layer 01's catalogue); link α-β values are illustrative; PCIe
  host-to-device bandwidth ~25 GB/s (Gen4 x16) and ~12 GB/s (Gen3 x16, T4); GPU-to-GPU NCCL over PCIe assumed
  8 GB/s with α = 20 µs (T4s) and 12 GB/s with α = 15 µs (L4s), the lab's values — fit your own with layer 02's lab.
  Memory the driver reports (T4 15.0 GiB, L4 22.49 GiB) and the ~1.5 GiB activation and CUDA-graph overhead in
  `sizing.kv_room_gib()`.
- **Expert placement**: `round_robin` honoured only with more than one expert group, no redundant experts, EPLB off
  and (with all-to-all kernels) the DeepEP low-latency or NIXL-EP backend — read in v0.30.0 and main `a4eb3f2`.
- **llama.cpp**: whether `--cpu-moe` experts are computed on the CPU in decode and copied to the GPU for large
  batches (`--op-offload`, default on).
- **Prices**: every $/GPU-hour here is a placeholder; current prices are in [`COMPUTE.md`](../../COMPUTE.md).
- **GCP**: the cuda-and-nccl lab's `l4x2` pool (`g2-standard-24`, 2 × L4) and GKE's L4 Spot availability.
