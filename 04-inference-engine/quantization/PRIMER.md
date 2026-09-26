# Quantization for inference: number formats, calibration, kernels and the accuracy you pay

*A primer for layer 04. Snapshot: September 2026. Product facts are dated and marked (verify); every formula has a
worked number and names the function in [`quant-core/`](quant-core/) (package `quantcore`) that computes it, and
`quant-core/tests/test_primer_numbers.py` recomputes every such number and fails if this file stops saying it.
Latencies from `quantcore.cost` are **simulated** — a roofline model, not a measurement.*

This primer is the deep dive behind one section of the serving-engine primer. That section,
[serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization), surveys the formats, scale granularity,
weight-only versus W8A8, FP8 KV and what each buys for Llama-3.1-8B on an L4; `minengine.quant` computes it. This
primer starts where that survey stops. It works through the number formats down to their bit patterns, including
the Blackwell block formats. It covers the calibration algorithms that make 4 bits usable (GPTQ, AWQ, SmoothQuant),
activation and KV-cache quantization below 8 bits, the kernels and why a format's speed depends on the GPU
generation, and how to measure the accuracy you pay. It also covers how a quantized checkpoint is produced and
loaded by vLLM, and how to choose a scheme. Everything is learnable at tier T0 with [`quant-core/`](quant-core/);
[`quant-lab/`](quant-lab/) produces real checkpoints with llm-compressor, serves them with vLLM and measures them
(T1), and points at the serving lab's Cloud Run and GKE deploys for T3.

---

## The one-minute version

Quantization stores numbers on a coarser **grid** with a **scale**: `x ≈ code × scale`. It speeds up only what is
bound by the bytes or FLOPs it removes. **Decode** streams every weight each step, so fewer weight bytes are faster
tokens: weight-only INT4 (**W4A16**) decodes an 8B model ~3× faster at batch 1. **Prefill** is compute-bound.
Weight-only kernels do 16-bit math on dequantized weights, so only formats the tensor cores multiply natively make
it faster: **FP8 W8A8** on Ada, Hopper and Blackwell, **INT8 W8A8** on Turing to Hopper, **FP4 W4A4** on
Blackwell. The **KV cache** is the third lever: FP8 KV halves it and doubles the sessions per GPU.

Accuracy is decided by **granularity** and **outliers**. A scale is set by the largest value that shares it, so
INT4 needs groups of 32–128 plus a calibration method. **GPTQ** lets later columns absorb each column's rounding
error through the inverse Hessian of the calibration inputs. **AWQ** scales up the weights that meet large
activations. **SmoothQuant** moves activation outliers into the weights so INT8 activations survive. Floating
grids (FP8 E4M3, NVFP4) tolerate outliers better than integer grids because their error is relative. Keep the LM
head, embeddings, norms, softmax and MoE routers in 16-bit.

Checkpoints come from llm-compressor as compressed-tensors; vLLM detects the format and picks a kernel per GPU
generation. An FP8 checkpoint on an A100 runs weight-only. Measure the damage as KL and top-1 agreement first,
then task accuracy **with its standard error**, against a budget written down before you look.

---

## 1. Why quantize, and what it can and cannot speed up

**The roofline argument.** A kernel takes `max(FLOPs ÷ peak, bytes ÷ bandwidth)`
([layer 01 PRIMER §2](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#2-the-roofline-model)), and an
LLM step's arithmetic intensity is roughly the number of tokens in it
([§3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#3-llm-inference-on-the-roofline)). So:

| What a step is bound by | When | What quantization must cut | Schemes that cut it |
|---|---|---|---|
| weight bytes | decode at small batch | bytes per weight | W4A16, W8A16, FP8/INT8 W8A8, NVFP4 |
| KV bytes | decode at long context × large batch | bytes per KV element | FP8 KV, INT4/INT2 KV |
| FLOPs | prefill; decode past the ridge | the precision the tensor cores multiply in | FP8/INT8 W8A8, FP4 W4A4 — **not** weight-only |
| memory capacity | always, for concurrency | weight + KV bytes | all of them |

Layer 01 §3.5 prices this for Llama-3.1-8B on an H100: FP8 is exactly 2.00× in both regimes (it halves every byte
and doubles the peak); W4A16 is 3.80× at batch 1 but 1.54× at batch 64, because half the bytes there are KV cache
it did not touch. The same arithmetic for one linear layer shows where weight-only stops paying. Take Llama-3.1-8B's
`down_proj` (K = 14,336, N = 4,096) with M tokens in the step: FLOPs `2MKN`, bytes `K·N·w + M·(K·a + 2N)`. The layer
turns compute-bound above (`cost.crossover_tokens()`):

```
M* = (K·N·w / BW) / (2·K·N / peak − (K·a + 2N) / BW)

L4 (121 TFLOP/s bf16, 242.5 fp8, 0.30 TB/s):   BF16 462 tokens    FP8 W8A8 478    W4A16 119
```

FP8 halves the bytes and doubles the peak, so its crossover barely moves. W4A16 cuts the bytes 3.9× and keeps the
BF16 peak, so it turns compute-bound early, at ~120 tokens per step, while BF16 stays byte-bound until ~460. Up to
~120 tokens W4A16 keeps the whole byte ratio: vllm-internals §8.1's table (reproduced by `cost.gemm_time()` in
`tests/test_repo_numbers.py`) shows W4A16 at 102 µs against BF16's 392 µs for one token. Between ~120 and ~460 its
speedup shrinks, because W4A16 is paying for BF16 math while BF16 is still paying for bytes: 1.7× at 256 tokens,
nothing at 462 and beyond (the two are equal at 2,048). So decode batches of a few hundred still gain from INT4,
and long prefill chunks do not. Real kernels lose sooner, because dequantizing is not free (§4).

**Tensor-core throughput by precision** is the other half: L4 121 → 242.5 TFLOP/s from BF16 to FP8; H100 989.4 →
1,978.9; B200 2,250 → 4,500 → 9,000 at FP4 (dense, `roofline.specs`, verify). Every halving of precision doubles
the rate, but only on parts that have that datapath ([gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md#4-tensor-cores-the-biggest-change-since-you-last-looked)).

**What it cannot speed up.** Attention over the KV cache is not a weight GEMM: weight quantization leaves it alone,
and only KV quantization shrinks its bytes. The step's fixed overheads (launches, sampling, scheduling) do not
shrink. A 16-bit LM head is read every step: for Llama-3.1-8B it is 1.05 GB of the 4.65 GB an INT4 decode step
streams (`cost.Model.streamed_bytes()`; the checkpoint is 5.70 GB with the input embedding, which is only
gathered). serving-engine §8's L4 table shows the net effect — 3× faster single-stream decode, not 3.9×.

**The accuracy budget.** Every scheme costs some accuracy, and the cost depends on the model, the task and the
recipe. The rest of this primer is how to keep it small (§3–§6), how to measure it (§8), and how to trade it
against speed and memory (§10).

## 2. Number formats

**Integers.** `code = clip(round(x / scale) + zero, qmin, qmax)`, `x̂ = (code − zero) × scale`
(`formats.quantize_int()`, `dequantize_int()`). Three conventions are in use, and the repo has all three:

| Convention | INT4 codes | Scale | Used by |
|---|---|---|---|
| symmetric, restricted | −7…7 | amax / 7 | `minengine.quant`, SmoothQuant's fake-quant (`/127`) |
| symmetric, full | −8…7 | amax / 7.5 | GPTQ's `quant.py`, compressed-tensors (`_calculate_range`), vLLM `uint4b8` |
| asymmetric | 0…15, zero point | (max − min) / 15 | AWQ (`zero_point=True`), KIVI, `W4A16_ASYM` |

`formats.int_range()` and `int_scale()` implement all three. The full convention uses the 16th code, so its step
is 7/7.5 of the restricted one: 0.24 dB better on Gaussian INT4 g32 weights. Asymmetric widens the range to
include 0 and covers [min, max] instead of [−amax, amax]. For `[−0.3, 0, 0.05, 0.4, 1.2]` it gives scale 0.1 and
zero point 3, with 0 exact (`formats.asym_params()`). It halves the step for one-sided data such as ReLU outputs.

**Floating point.** Sign, `e` exponent bits with a bias, `m` mantissa bits. Within one power of two the step is
`2^(exponent − m)`, so the **relative** error is at most `2^−(m+1)` at any magnitude. Below the smallest normal
exponent the spacing stays fixed (subnormals). `formats.FloatFormat.grid()` enumerates every value from the bit
patterns, and `formats.to_float()` rounds to it (ties to even, saturating); on 100,000 values it is bit-identical to
torch's `float8_e4m3fn` and `float8_e5m2` casts (notebook 01).

| Format | Bits (s-e-m), bias | Largest | Smallest normal | Smallest subnormal | Max relative error | Normal range |
|---|---|---|---|---|---|---|
| FP8 **E4M3** (`fn`) | 1-4-3, 7 | **448** | 2⁻⁶ | 2⁻⁹ | 6.25% | 2^14.8 |
| FP8 **E5M2** | 1-5-2, 15 | **57,344** | 2⁻¹⁴ | 2⁻¹⁶ | 12.5% | 2^29.8 |
| FP4 **E2M1** | 1-2-1, 1 | 6 | 1 | 0.5 | — (8 magnitudes) | 2^2.6 |
| BF16 | 1-8-7 | ~3.4 × 10³⁸ | 2⁻¹²⁶ | 2⁻¹³³ | 0.39% | fp32's |
| FP16 | 1-5-10 | 65,504 | 2⁻¹⁴ | 2⁻²⁴ | 0.05% | 2^30 |

E4M3 has no infinities: S.1111.111 is NaN, so its top value is 1.75 × 2⁸ = 448. That leaves 254 finite codes and
253 distinct values. E5M2 keeps IEEE's infinities and has half the precision. Inference uses **E4M3** for weights,
activations and the KV cache (vLLM's FP8 linear method accepts only `float8_e4m3fn`, verify); E5M2 is mostly a
gradient format. On N(0,1) values E4M3 rounds with a 2.2% median and 5.9% maximum relative error, E5M2 with 4.3%
and 11.1% (notebook 01). E2M1's whole grid is {0, 0.5, 1, 1.5, 2, 3, 4, 6}. vLLM's reference rounding uses
thresholds 0.25/0.75/1.25/1.75/2.5/3.5/5, which is round-half-to-even, as `to_float` does. BF16 against FP16 for
attention is FlashAttention deep dive §9.1: bf16 has range, fp16 has precision. A T4 has no bf16 at all.

**Block formats** give a 4-bit float grid a local scale:

- **MXFP4** (OCP Microscaling). E2M1 elements and one **E8M0** scale (a pure power of two, stored as exponent +
  127) per **32** values. The OCP rule is `shared exponent = floor(log2(block amax)) − 2`, where 2 = floor(log2 6)
  (`formats.mxfp4()`). A block whose amax is 5 gets exponent 0 (byte 127); one whose amax is 0.75 gets −3 (byte
  124). Because the scale floors to a power of two, the block max lands in [4, 8) on the E2M1 grid: above 6 it is
  clipped, and near 4 the top codes go unused. Producers differ here: compressed-tensors, which writes
  llm-compressor's MXFP4 checkpoints, first rounds amax to a power of two, **up** when its mantissa is ≥ 1.75
  (`round_to_power_2`), so its block max lands in [3.5, 7) (`formats.mxfp4(rule="compressed-tensors")`, and the
  lab's `fp4.mxfp4_scale_exponent()`). A block whose amax is 7.5 gets exponent 0 and clips to 6 under the OCP rule,
  exponent 1 and 8 under compressed-tensors'. That clips 28% of Gaussian blocks' maxima instead of 43%, and
  wastes a little range just under each power of two. gpt-oss ships its MoE weights in MXFP4.
- **NVFP4**. E2M1 elements, one **FP8 E4M3** scale per **16**, and one FP32 scale per tensor. The tensor scale is a
  multiplier `g = 448 × 6 / amax(tensor)`. Each block's scale is `E4M3(g × block_amax / 6)`, and
  `x̂ = element × block_scale / g` (`formats.nvfp4()`, matching compressed-tensors' `generate_gparam` and vLLM's
  `ref_nvfp4_quant`).

Which grid wins depends on the data (`granularity.error()`, relative error, seed 0, 256×512 weights):

| 4-bit scheme | Bits per weight | Gaussian | Heavy-tailed (Student-t, 3 d.o.f.) |
|---|---|---|---|
| INT4 g32, fp16 scale (full convention) | 4.5 | 0.0951 | 0.1433 |
| FP4 E2M1 g32, fp16 scale | 4.5 | 0.1010 | 0.1061 |
| MXFP4 (OCP exponent) | 4.25 | 0.1145 | 0.1404 |
| MXFP4 (compressed-tensors exponent) | 4.25 | 0.1123 | 0.1373 |
| NVFP4 | 4.5 | 0.0953 | 0.0909 |

On Gaussian data an evenly spaced grid is as good as a float grid. On heavy tails the float grid wins, because most
values are small and E2M1 spends its codes near zero. MXFP4's power-of-two scale costs it against NVFP4's E4M3
scale at every distribution.

**The error model: ~6 dB per bit.** Rounding noise has power `step² / 12`, and with `step = 2·amax / 2^b` the
signal-to-quantization-noise ratio is (`formats.sqnr_rule_db()`):

```
SQNR ≈ 6.02·b + 4.77 − 20·log10(amax / rms)          a sine (crest √2): the textbook 6.02·b + 1.76
```

For 128×256 Gaussian weights per channel (crest factor 3.07), the rule predicts 43.2 dB at INT8 and the grid
measures 43.1; at INT4 it predicts 19.1 and the grid measures 18.4. The model holds from 4 bits up and
over-predicts at 2–3 bits, where the noise is no longer busy. serving-engine §8 measured the same slope: 43.0 dB
per channel at INT8 against 17.9 at INT4.

**Outliers dominate.** The last term is the crest factor. Every 10× of amax over rms costs 20 dB, more than three
bits. A 4,096-value row of rms 0.02 has a crest factor of 3.9. Add one value of 0.8 and it becomes 33.9, and INT8
per-channel SQNR falls from 41.1 dB to 22.2 dB (the rule predicts 22.3): one value costs **3.1 bits** (notebook 01,
exercise 1.3). LLM weights and, far more, LLM activations have such values in fixed channels. That is the problem
§3–§5 solve.

## 3. Granularity and the bits-per-weight budget

**Who shares a scale.** `granularity.quantize()` implements each option for `W[out, in]` (the torch and checkpoint
layout) and returns scales in compressed-tensors' shapes:

| Granularity | Scale shape | Isolates | Typical use |
|---|---|---|---|
| per tensor | (1,) | nothing | FP8 weights and static activations |
| per channel (output row) | (out, 1) | an outlier row | INT8 and FP8 weights |
| per group of g inputs | (out, in / g) | an outlier within g inputs | INT4 (g = 128, 64, 32), FP4 (16, 32) |
| per token | (tokens, 1) | an outlier token | dynamic INT8/FP8 activations |
| per block r × c | (out / r, in / c) | an outlier tile | DeepSeek-V3 FP8 (128 × 128) |

serving-engine §8's six-scheme table (INT8 per tensor to INT4 g32 on one Gaussian weight) is the survey. quantcore
reproduces all six rows from the same seed (`tests/test_repo_numbers.py`), and so do its outlier-row and
SmoothQuant numbers; they are not repeated here.

**Rows are easy, columns are not.** Per-channel scales isolate an outlier *row*, but an outlier *input column* of
W is in every row and sets every row's scale. Take a 256×512 weight with one column 20× larger, quantized to INT4
(full convention). The error on the **other** columns is 0.612 per channel, 0.320 with groups of 128, 0.236 with 64
and 0.176 with 32 (notebook 02). Groups confine the damage but do not remove it. The other remedies for a *weight*
outlier also act on W: GPTQ lets the columns not yet rounded compensate for its rounding, and a rotation spreads
it over every column (both §4). AWQ does not apply, because its scales come from activation magnitudes.

**Activation outliers are a different problem.** The weight columns that meet a large activation channel are
usually ordinary. In `quantcore.TinyModel`'s first up-projection, the columns behind its four outlier channels
have absmax 0.353–0.392 against a median column of 0.375, so no weight granularity singles them out. Yet they
cause 98.4% of that layer's INT4 g32 output error (`granularity.output_error_by_input()`), because a column's
rounding error reaches the output multiplied by its input. That is the case AWQ fixes on the weight side (§4) and
SmoothQuant or a float grid on the activation side (below, and §5).

**Activations.** They are quantized at run time, per token (dynamic, one max-reduction per token) or with one
static per-tensor scale fixed at calibration. LLM activations have a few channels that are large in *every* token.
In `quantcore.TinyModel` they come from RMSNorm gains of 25–40×: the first up-projection's input has four
channels with absmax 76–120 against a median channel absmax of 3.0. Per-token INT8 gives 1.4% error on that
tensor as a whole, but **11.1%** on the 60 ordinary channels, because each token's step is set by the outlier. Per
tensor gives 27.5% on them, and per-token FP8 **2.7%**, because a float grid keeps small values' relative precision
(notebook 02). Static scales saturate whatever calibration did not see. For the same layer's INT8 output on
held-out inputs, the calibration max gives 3.53% error. The 99.99th percentile gives 3.22%, and the 99th percentile
clips the outlier channels themselves and gives 24.5%. Dynamic per-token scales give **1.51%**
(`w8a8.w8a8_matmul()`). This is why `FP8_DYNAMIC` and INT8 `W8A8` recipes quantize activations dynamically per
token and need no calibration data for them.

**The bits-per-weight budget.** Scales and zero points are stored too (`formats.bits_per_weight()`):

```
bits per weight = element bits + (scale bits + zero-point bits) / group size  [+ tensor-scale bits / numel]

INT8 per channel, 4,096 inputs, fp16 scale       8.004
INT4 g128 symmetric, fp16 scale                 4.125      <- compressed-tensors W4A16; minengine.quant's convention
INT4 g128, fp16 scale + 4-bit zero point         4.15625    <- AWQ, GPTQ-format; servelab.sizing's ("4.16")
INT4 g32, fp16 scale / + zero point              4.5 / 4.625
MXFP4 (E8M0 per 32)                              4.25
NVFP4 (E4M3 per 16, FP32 per tensor)             4.5
FP8, FP32 scale per 128 × 128 block              8.002
```

Both INT4 figures in the repo are right for their assumptions. Symmetric compressed-tensors W4A16 stores no zero
point. The IST GPTQ reference quantizes asymmetrically by default, and GPTQ-format checkpoints (AutoGPTQ,
GPTQModel) always store packed `qzeros`, so they cost ~4.16 bits (vllm-internals §8.1's figure). The whole model is
never 16/4.125 smaller either:
recipes keep the embedding and LM head in 16-bit (`cost.Model.weight_bytes()`):

| Model | BF16 | FP8 | INT4 g128 (4.125) | Ratio | Kept 16-bit |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (tied embedding) | 0.988 GB | 0.630 GB | 0.457 GB | 2.16× | 27.6% of parameters |
| Llama-3.1-8B (untied) | 16.06 GB | 9.08 GB | 5.70 GB | 2.82× | 13.1% |
| Llama-3.1-70B | 141.1 GB | 72.7 GB | 39.5 GB | 3.57× | 3.0% |

Group sizes are constrained by the kernels. `in_features` must divide by the group size (llm-compressor errors at
initialization, and Marlin rejects it). Marlin supports groups of −1 (per channel), 32, 64 and 128, and Machete −1,
64 and 128 (vLLM source, verify). A 576-wide model cannot use g128.

## 4. Weight-only post-training quantization

**Round to nearest (RTN)** needs no data. Each weight goes to the nearest code of its group (`gptq.rtn()`). On
the tiny model, INT8 is free, INT4 g32 costs 6.6 points (90.3% → 83.7%) and INT3 22 points (notebook 03).
Larger models tolerate RTN better. The GPTQ paper's LLaMA-7B goes from 5.68 to 6.29 WikiText-2 perplexity at 4-bit
RTN and to 25.5 at 3-bit (README, verify).

**GPTQ** minimises each layer's output error, `‖X Wᵀ − X Qᵀ‖²` over calibration tokens X, not the weight error.
The curvature of that objective is the Hessian (`gptq.hessian()`):

```
H = 2/n · Σ_t x_t x_tᵀ                                    (in × in, from calibration activations)
H += 0.01 · mean(diag H) · I                              damping, `percdamp`
U = chol(H⁻¹)ᵀ                                            upper Cholesky factor, H⁻¹ = UᵀU
for each input column j:                                  (gptq.gptq)
    q_j = quantize(w_j)                                   on the group's grid
    e   = (w_j − q_j) / U[j, j]
    W[:, j+1:] −= e ⊗ U[j, j+1:]                          the Optimal Brain Surgeon update
```

Each column's rounding error is pushed onto the columns not yet rounded, in the directions the inputs actually
use. Details that matter in practice:

- **Group scales.** `gptq.gptq()` computes each group's scale when the loop reaches the group, on the
  already-updated weights. llm-compressor's `GPTQModifier` fixes them up front, from the weight observer on the
  original weights. Either way the codes, not the scales, carry the compensation.
- **Act-order** visits columns by decreasing `H_jj`. The most-used inputs go first, while the most columns remain
  to absorb their error. With *static groups* (parameters fixed up front) the checkpoint layout does not change.
- Inputs that never fire (`H_jj = 0`) are zeroed.
- The reference implementation's "lazy batch" defers updates beyond a 128-column block for GPU efficiency. Its
  OBS updates are the same, and so is its result per channel or when groups start on block boundaries. A group
  that starts mid-block takes its scale from weights that miss the block's in-flight updates, a slightly different
  choice. `tests/test_gptq.py` checks `gptq.gptq()` against a transcription of it in both cases.
- With uncorrelated inputs H is diagonal, U has no off-diagonal terms and GPTQ *is* RTN (tested).

GPTQ is strongest where inputs are correlated. The tiny model's down-projections read a ReLU of a 64-dim stream
spread over 256 dims, so 99% of their H lies in 13–37 directions. GPTQ cuts their output error 5–8× (0.1000 →
0.0174, 0.0197 on held-out data). On the model, INT4 g32 recovers to **89.7%** (KL 0.259 → 0.048) and INT3 to
**84.4%** (notebook 03, `tinymodel.quantize_model()`). Real layers are less redundant than this toy's, so expect
smaller gains. At 3-bit the GPTQ paper reports 8.07 perplexity against RTN's 25.54 on LLaMA-7B (verify).

**AWQ** protects the weights that meet large activations. A weight's rounding error reaches the output multiplied
by its input, so the input channels with large activations own most of the output error. AWQ multiplies those
weight columns by `s > 1` before rounding and divides the activations by `s`, folding `1/s` into the preceding
RMSNorm gain (or the previous linear's rows), so `(X/s)(W·s)ᵀ = X Wᵀ` exactly (`tinymodel.TinyModel.fold()`).
The scale is searched, not solved (`awq.search_scale()`, following llm-awq's `auto_scale.py`):

```
s = mean|x|^α,  s /= sqrt(max s · min s),  α ∈ {0, 1/20, …, 19/20}
loss(α) = mean((X/s) · Q(W·s)ᵀ − X Wᵀ)²       keep the best α; α = 0 is RTN, so AWQ never loses on calibration data
```

On the tiny model's first up-projection the loss curve is U-shaped with its minimum at α = 0.30, 0.58 of RTN's
loss. Too little scaling leaves the salient channels coarse. Too much makes them set the group scale for everyone.
AWQ cuts the up-projections' output error ~25% (0.0939 → 0.0715) and does little for the down-projections, which
have no dominant channels. On the up-projections alone at INT3 it beats GPTQ: 85.9% against 84.7%. The two
compose, AWQ's scales first and then GPTQ's rounding (llm-compressor: an `AWQModifier` then a `GPTQModifier`),
for the lowest KL of all: 0.0307 at INT4 g32. llm-awq also searches a per-group **clipping** threshold (amax ×
1.00 down to 0.55) and skips q/k projections (verify). llm-compressor's "duo" variant divides by `w_mean^(1−α)`.

**What calibration data does and does not do.** It supplies H (GPTQ) or activation statistics (AWQ, SmoothQuant). It
teaches the model nothing, and it cannot rescue a format that is too coarse. On the tiny model, GPTQ INT3 gets 78.4%
with 16 samples, 82.4% with 64, 84.4% with 256 and 85.0% with 1,024. Three other 256-sample draws give 83.4–84.6%:
past a few hundred samples, which samples you drew matters as much as how many. With 256 samples from only 2 of the
16 classes it still gets 83.0%, and with 256 samples of pure noise 84.5%. Outlier channels and input correlations
are properties of the weights and norm gains, and any input reveals them. On real models the text still matters:
chat templates, languages and long contexts shift activation statistics. Calibrate on data that looks like your
traffic. The recipes use 512 sequences × 2,048 tokens (GPTQ), 128–512 × 512 (AWQ) and 512 × 512 (SmoothQuant) (§9,
verify).

**Rotations** (QuaRot, SpinQuant, QuIP) multiply W and X by an orthogonal matrix, often a Hadamard, which leaves
`X Wᵀ` unchanged and spreads an outlier's energy over all channels. The FlashAttention deep dive §9.4 measures it
on attention: a random Hadamard takes INT4 error on a K with one 20× channel from 52% to 24%. llm-compressor ships
SpinQuant and QuIP transforms; QuaRot's details are (verify).

**Kernels: dequantize on the fly.** A W4A16 kernel reads packed 4-bit weights and their group scales, dequantizes
them to FP16/BF16 **in registers**, and feeds the ordinary 16-bit tensor-core MMA with FP32 accumulation
([vllm-internals §8.1](../vllm-internals/vllm-internals-primer.md#81-how-a-quantization-method-is-chosen)). It moves
a quarter of the bytes and does exactly the BF16 math:

- **Marlin**. FP16×INT4, "close to ideal (4x) speedups up to batchsizes of 16-32 tokens". With g128 scales the
  byte ratio is 16/4.125 = **3.88×** (the README's "optimal 3.87x"). vLLM's port runs from SM75, though the
  original README says SM80 (verify).
- **Machete**. vLLM's CUTLASS mixed-input GEMM, Hopper only.
- **ExLlama**. SM60+ (verify).

vLLM tries its mixed-precision kernels in the order CutlassW4A8, Machete, Marlin, Conch, Exllama, TritonW4A16,
Humming and logs `Selected <kernel> for <module>`. The consequence is §1's crossover: W4A16 wins decode and ties
or loses prefill. NVIDIA measured weight-only NVFP4 slower than BF16 in 10 of 12 GEMM shapes on Blackwell, because
of the dequantize-to-BF16 fallback, while W4A4 beat BF16 in 9 of 12 (ModelOpt, 2026-09-16, verify).

## 5. Weight-and-activation quantization

**The epilogue.** A W8A8 GEMM multiplies 8-bit codes and accumulates in a wide register. The scales factor out of
the sum, so they are applied once per output (`w8a8.w8a8_matmul()`):

```
y[t, j] = s_x[t] · s_w[j] · Σ_k qx[t, k] · qw[j, k]          INT8: INT32 accumulator; FP8: FP32 (but see below)
```

Emulated with integer codes and integer accumulation, it equals the fake-quantized product to within 10⁻¹⁴
(notebook 04). Three consequences follow:

- The activation scale may vary per token and the weight scale per output channel, but **neither may vary along
  k**, the reduction axis. That is why SmoothQuant must move per-input-channel variation into the weights rather
  than scale it.
- The worst-case INT8 accumulator for a reduction of length K is 127² × K. It stays below 2³¹ up to K = 133,144,
  far above any hidden size.
- Block formats need one partial sum per 128-wide k block, rescaled by that block's two scales before it is added
  (`w8a8.block_fp8_matmul()`, the DeepGEMM and CUTLASS block-scaled pattern).
- FP8 "FP32 accumulation" is not quite that inside the tensor core. The DeepSeek-V3 report found Hopper's FP8 MMA
  keeps about 14 bits of accumulator precision, so its GEMMs promote each 128-element partial sum to FP32
  registers. That is a second reason for the 128-wide k blocks (DeepSeek-V3 report §3.3, verify).

**INT8 W8A8 with SmoothQuant.** INT8 per-token activations are wrecked by outlier channels (§3). SmoothQuant
divides activation channel j by `s_j` and multiplies weight column j by `s_j`, then folds `1/s` into the preceding
norm (`smoothquant.smooth_scales()`):

```
s_j = max|X_j|^α / max|W_j|^(1−α)           α = 0.5 default; tuned: Llama-3-8B 0.85, Mistral/Mixtral 0.8 (verify)
```

On the tiny model's up-projection, INT8 W8A8 output error falls from 1.52% to **0.57%** at α = 0.5 (the sweep is
U-shaped between α = 0 and 1). Over the whole model, KL falls 2.7× (0.00296 → 0.00110), for free at run time. The
serving-engine example — one activation channel 60× larger, 6.6× less output error — is reproduced in
`tests/test_repo_numbers.py`. llm-compressor's INT8 recipe is `SmoothQuantModifier(smoothing_strength=0.8)`
followed by `GPTQModifier(scheme="W8A8")`.

**FP8 W8A8.** Weights are E4M3 per tensor, per channel or per 128 × 128 block. Activations are E4M3 per token,
dynamic, or per tensor, static. DeepSeek-V3 ships `weight_block_size [128, 128]` with FP32 `weight_scale_inv`
per block and quantizes activations per token per 128 channels on the fly (`act_quant(x, block_size=128)`).

FP8's float grid is why it usually needs no smoothing. The tiny model's outlier channels leave FP8 per-token
activations at 2.7% error on the ordinary channels, against 11.1% for INT8. But three mantissa bits are coarser
than INT8's seven for well-scaled values: over the whole model FP8 W8A8 has 6× the KL of INT8 W8A8 (0.0186 against
0.00296), with accuracy within noise.

Block scales matter little for FP8 at ordinary ranges. With a weight tile 30× hotter than the rest, per-tensor,
per-channel and 128 × 128-block FP8 all give 3.7–3.9% output error, while INT8 per token and channel gives 0.9%
(notebook 04). They earn their place when ranges exceed E4M3's 2^14.8, in training, and because a kernel that
already tiles by 128 can apply them for free. There is no CUTLASS block-FP8 kernel for SM89 (vLLM source, verify).

**FP4 W4A4 (Blackwell).** NVFP4 weights, plus activations quantized per 16 at run time with a calibrated global
scale (`dynamic="local"`; llm-compressor calibrates it with 20 samples). The tensor cores multiply E2M1 directly at
twice the FP8 rate. Turing and Ampere had INT4 tensor cores, but no production serving stack ran LLMs on them; NVFP4
is the first 4-bit format vLLM runs natively on the tensor cores, so the first that speeds up prefill in production
serving. Below SM100, vLLM runs NVFP4 checkpoints weight-only (verify). For Llama-3.1-8B on a B200, the roofline
model gives a 1,800-token prefill of 21 ms in BF16, 12 ms in FP8 and 7 ms in NVFP4 (`cost.table()`, SIMULATED,
verify Blackwell figures).

**W4A4's accuracy risk is the activations.** Sixteen activations share one E4M3 scale, set by their largest.
An outlier channel 30× the typical value sets that scale at 30/6 = 5 typical values per unit of the E2M1 grid.
A typical neighbour then lands at 0.2, below the 0.25 that rounds up to E2M1's smallest step, so it becomes zero.
The tiny model's first up-projection has its four outlier channels in three of its four 16-channel blocks. With
NVFP4 activations, its ordinary channels carry 55.4% error, and 48% of them become zero; the block with no outlier
carries 9.7% (`formats.nvfp4()`, notebook 04). The model has 84.5% accuracy with NVFP4 weights only and 80.8% with
W4A4 (full precision: 90.3%). The mitigations are the ones for INT8 activations, pushed harder:

- **Smoothing** (SmoothQuant, or AWQ-style scales folded into the norm). At α = 0.5 the ordinary channels drop to
  14.5% error and the model recovers to 84.1%.
- **Hadamard rotations**, which spread an outlier over its block (llm-compressor's SpinQuant and QuIP transforms,
  §4).
- **Quantization-aware distillation** (§7).

Gate W4A4 on an eval. In the lab's notebook 05 a model that is 100% accurate falls to 42% and 51% on its two
tasks with NVFP4 W4A4, and gets 99–100% back with SmoothQuant.

**Which layers stay in high precision.** Recipes target the transformer blocks' linears and `ignore=["lm_head"]`:

- **LM head.** Its errors land on the logits with no later layer to average them. Quantizing only the tiny model's
  head to INT4 costs 4.0 points, against 6.0 for all four hidden linears (notebook 04).
- **Embedding.** A gather, not a GEMM: quantizing it saves memory but no compute.
- **Norms and the attention softmax.** Tiny, and range-sensitive in exactly the way low precision handles worst.
- **MoE routers** (`mlp.gate`). A flipped top-k choice is a discrete error.

Attention itself stays in BF16 unless the kernel quantizes it: FA3's FP8 path also quantizes Q, and "FP8
attention" can mean an FP8 KV cache with BF16 math or FP8 matrix multiplies, with different error budgets
([FlashAttention deep dive §9.4](../flash-attention/flash-attention-deep-dive.md#94-fp8-error-sources-and-mitigations)).

## 6. KV-cache quantization

The KV cache is an activation stored for later: quantized once when written, dequantized at every read. At long
context and large batch it is most of what decode streams ([layer 01 §3.4](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#34-kv-reads-cap-decode-intensity)).
Halving it buys two things (`kvquant.kv_bits_per_element()`, `cost.Model.kv_bytes_per_token()`, `cost.sessions()`).

**Concurrency.** For Llama-3.1-8B, `2 × layers × kv_heads × head_dim × bytes` is 131,072 B per token in BF16 and
65,536 in FP8. Sub-8-bit schemes store a scale and a minimum per group: 4-bit with groups of 32 costs 5 bits per
element (40,960 B per token) and 2-bit costs 3 (24,576). On an L4 with FP8 weights (the core's round memory inputs)
the 2,000-token sessions go 43 → 87 → 140 → 234. At vLLM's defaults, `servelab.sizing` gives BF16 weights 2,363 KV
blocks and 4,727 with an FP8 cache (vllm-internals §4.7).

**Faster long-context decode.** For Llama-3.1-8B with FP8 weights on an H100 at batch 32 and 8,000 tokens of
context, FP8 KV takes a decode step from 17.5 ms to 11.3 ms (`cost.step_cost()`, SIMULATED).

**FP8 KV and its scale.** vLLM's `--kv-cache-dtype` accepts `fp8` (= `fp8_e4m3`) and `fp8_e5m2`, plus per-token-head
dynamic types (`int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head`) and NVFP4 on SM100 (verify). The
static scales `k_scale` and `v_scale` come from the checkpoint and are **1.0 otherwise**. Per-attention-head
scales work only with the FlashAttention backend. On one synthetic decode head (`kvquant.synthetic_qkv()`: keys
with four outlier channels of magnitude ~12, values with a shared mean), the attention-output error is:

| Cache | Attention-output error (`kvquant.attention_error()`) |
|---|---|
| FP8 K and V, calibrated per-tensor scales | 0.71% |
| … keys only / values only | 0.64% / 0.28% |
| FP8, scale 1.0 (vLLM's default), values of order 1 | 0.81% |
| FP8, scale 1.0, values ×10⁻³ | 4.9% (subnormals and zeros) — calibrated: 0.28% |
| FP8, scale 1.0, values ×10³ | 76% (saturation at 448) — calibrated: 0.28% |

Key errors cost more than value errors because they pass through the softmax's exponential, while value errors
are averaged. An uncalibrated scale is harmless for values of order 1 and wrong far from it. llm-compressor's
`kv_cache_scheme` writes calibrated scales; per-head scales need its per-head recipe.

**Below 8 bits: KIVI.** Keys have fixed outlier channels; values do not. So KIVI quantizes **keys per channel**
(a scale and minimum per channel per group of tokens) and **values per token** (per group of channels). Both are
asymmetric. The newest tokens stay 16-bit until a group fills. On the same head, with groups of 32
(`kvquant.kivi()`), 4-bit keys per channel give 1.16% error against 2.31% per token, and 2-bit 6.9% against 9.5%.
The KIVI README reports 2.6× less peak memory and up to 4× larger batches (verify). vLLM's sub-8-bit KV types at
this snapshot are INT4 per token-head with dynamic scales, NVFP4 (E4M3 scales per 16, SM100 only) and the
TurboQuant variants (`turboquant_k8v4`, `turboquant_4bit_nc`, …). None is KIVI's per-channel-key scheme (verify):
check which one a system implements before you trust its keys at 4 bits.

**Kernel conditions** decide whether you can use it at all. They are worked out in
[vllm-internals §6.3](../vllm-internals/vllm-internals-primer.md#63-how-a-backend-is-chosen) and in each attention
backend's checks (`vllm/v1/attention/backends/`):

- **T4.** No FP8 KV cache in any backend: Triton's FP8 path needs SM89, FlashInfer SM80, FlashAttention SM80.
- **A100 and L4.** `--kv-cache-dtype fp8` moves attention from FlashAttention 2 to FlashInfer. A throughput change
  is therefore not only the KV dtype.
- **H100.** FA3 handles FP8 KV, and also quantizes Q.

**Prefix caching with a quantized cache** works unchanged. Blocks are named by their tokens
([serving-engine §5](../serving-engine/PRIMER.md#5-prefix-caching)) and reused as stored. A later request
dequantizes a cached block exactly as the one that wrote it, because static scales are per layer and dynamic
per-token scales are stored with the block. A scheme whose scale depended on the reading request would break
sharing.

## 7. Quantization-aware training and QLoRA in brief

**QAT.** Training can see the quantizer. The forward pass uses the fake-quantized weight `Q(w)` (and activations).
The backward pass treats rounding as the identity inside the clipping range, the **straight-through estimator**, so
the weights learn to sit where rounding hurts least. It can recover much of what PTQ loses at 4 bits and below, at
the cost of a training run (measure it per model). Quantization-aware distillation (QAD) trains the quantized model
to match the full-precision model's outputs rather than labels. NVIDIA's NVFP4 W4A4 note reports 500 QAD iterations
recovering an instruction-following benchmark that lost 2.6 points after PTQ, with a 67 → 22 GiB checkpoint
(ModelOpt, 2026-09-16, verify). This is the same "cheap post-training step on top of a big model" economics as the
post-training stages in [rl-and-thinking-models §1](../../00-foundations/rl-and-thinking-models/PRIMER.md#1-from-pretraining-to-post-training).

**QLoRA is a training recipe, not a serving format.** It freezes the base model in **NF4**, a 4-bit format whose
16 levels are normal-distribution quantiles, with a scale per block of 64 (QLoRA paper, verify). It trains LoRA
adapters in BF16 on top. NF4 with an FP32 scale per 64 costs 4.5 bits per weight. With "double quantization" —
8-bit block scales with one FP32 per 256 blocks — it costs 4.127 (`formats.bits_per_weight(4, 64, scale_bits=…)`).
transformers loads it with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True)`. To serve the result, merge the adapter into BF16 weights and quantize with a
serving scheme (§9). vLLM serves bitsandbytes checkpoints only through the out-of-tree `vllm-bnb-plugin` at this
snapshot (verify).

## 8. Measuring the accuracy you pay

**Three levels, cheapest first.**

- **Distribution distance** against the unquantized model on the same inputs: mean `KL(p_ref ‖ p_quant)` per
  position and top-1 agreement (`eval.kl()`, `granularity.argmax_agreement()`). It needs no labels and sees damage
  that a task score averages away.
- **Perplexity** on held-out text (`eval.perplexity()`). It is cheap, but it misses failures on generation-heavy
  tasks.
- **Task accuracy** on evals like your traffic. This is what users feel, and it moves in both directions.

On the tiny model, INT4 RTN has KL 0.259, top-1 agreement 87.5% and accuracy 83.7%: it lost 341 right answers and
gained 79. GPTQ has KL 0.048, 95.5% and 89.7%, losing 85 and gaining 61 (`eval.compare()`). Net accuracy
understates the churn. Greedy text is the most brittle metric of all: in serving-engine §8 an INT8 model with
99.8% top-1 agreement diverges from the BF16 greedy text after 13 tokens.

**Error bars.** lm-eval reports `stderr = sample stddev / √n`, which for a 0/1 metric is (`eval.accuracy_stderr()`):

```
stderr = sqrt(p · (1 − p) / (n − 1))        vLLM's FP8 example: 250 GSM8K items at 76.8% → ±0.0268
```

That is the error bar of one score. A drop is the difference of two scores, and compared unpaired its standard
error is `sqrt(se_ref² + se_quant²)`, √2 larger (`eval.diff_stderr()`). A drop smaller than about two of those is
noise. 250 items cannot see a drop smaller than ~7.6 points, and resolving 1 point at 77% takes 14,169 items per
model.

Quantized and reference models answer the **same** items, so compare them paired. Only the items that flipped
carry information, and McNemar's test on them is `z = (gained − lost) / sqrt(gained + lost)` (`eval.paired_z()`,
the lab's notebook 03). On the tiny model, AWQ + GPTQ INT4 drops 0.9 points on 4,000 items: inside the unpaired
bar of ±1.4, but 85 lost against 51 gained gives z = −2.9, a small loss that is real.

**lm-evaluation-harness** (0.4.13, verify) is the standard runner:

```bash
pip install "lm_eval[vllm]"
lm_eval --model vllm --model_args pretrained=$MODEL,add_bos_token=True,gpu_memory_utilization=0.8 \
        --tasks gsm8k --num_fewshot 5 --batch_size auto            # against a running server:
lm_eval --model local-completions --model_args model=$MODEL,base_url=http://HOST:8000/v1/completions --tasks gsm8k
```

Pass `add_bos_token=True` when comparing quantized models: the vLLM docs note they can be sensitive to it.
`--limit` is "for testing only". GSM8K defaults to 5-shot `exact_match`; MMLU is 57 subtasks scored by `acc`.

**Failure modes to test for explicitly:**

- **Small models.** Fewer parameters share the error; the tiny model loses 6.6 points at INT4 RTN.
- **MoE experts.** Rarely routed experts see little calibration data (llm-compressor's `moe_calibrate_all_experts`,
  on by default, sends every calibration token through every expert). Routers must stay 16-bit
  ([mixture-of-experts §6.7](../../00-foundations/mixture-of-experts/PRIMER.md#67-quantized-experts)).
- **Long context.** KV errors accumulate over thousands of positions, and outlier tokens set per-tensor scales.
- **Long generations and thinking models.** A flipped near-tie changes everything after it, and a reasoning
  trace of thousands of tokens gives it thousands of chances; evaluate with the full generation length.
- **Multilingual inputs.** Calibration in one language under-represents others' activation statistics.
- **Tool calling and structured output.** An argument that is wrong by one token is wrong; test exact-match on
  your own tool schemas.

**Setting a budget.** Write it down before measuring, in the form "KL ≤ x, an accuracy drop of at most y points,
and within two standard errors of the difference on our eval" (`eval.within_budget()`), and say whether the test
is paired. On the tiny model, with KL ≤ 0.05, INT8 RTN and INT4 GPTQ pass and INT4 RTN fails (notebook 05). The
recipe is part of the scheme.

## 9. Producing a checkpoint

**llm-compressor** (0.14.0, verify) applies a *recipe* of modifiers with `oneshot()` and saves a
**compressed-tensors** checkpoint that vLLM loads without flags. The example recipes, all `targets="Linear"`,
`ignore=["lm_head"]`:

| Scheme | Recipe | Calibration |
|---|---|---|
| FP8 dynamic (W8A8, per-channel weights, per-token activations) | `QuantizationModifier(scheme="FP8_DYNAMIC")` | **none** |
| FP8 block (DeepSeek-V3 style) | `QuantizationModifier(scheme="FP8_BLOCK")` | none |
| W4A16 GPTQ (INT4 g128, symmetric) | `GPTQModifier(scheme="W4A16")` | 512 × 2,048 tokens |
| W4A16 AWQ (asymmetric) | `AWQModifier(duo_scaling="both")` + `QuantizationModifier(scheme="W4A16_ASYM")` | 256 × 512 |
| W8A8 INT8 | `SmoothQuantModifier(smoothing_strength=0.8)` + `GPTQModifier(scheme="W8A8")` | 512 × 2,048 |
| NVFP4 (W4A4) | `QuantizationModifier(scheme="NVFP4")`; add smoothing or a rotation transform if activations have outlier channels (§5) | 20 samples (global activation scales) |
| FP8 KV cache | `kv_cache_scheme: {num_bits: 8, type: float, strategy: tensor, dynamic: false}` | 512 × 2,048 |

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]))
model.save_pretrained("Qwen2.5-0.5B-Instruct-FP8-dynamic", save_compressed=True)
```

MoE models must also ignore their routers (`"re:.*mlp.gate$"`). Install llm-compressor and vLLM in **separate
environments** (vLLM docs). The lab's notebook 01 runs these recipes at T1 on a 0.5B model and, at T0, writes the
same layout from its own code.

**The format.** `config.json` carries a `quantization_config`:

- `quant_method: "compressed-tensors"`, plus `format`.
- `config_groups`, each with `targets`, `weights` and `input_activations`, which are `QuantizationArgs`:
  `num_bits`, `type`, `symmetric`, `strategy`, `group_size`, `block_structure`, `dynamic`, `actorder`.
- `ignore`, `kv_cache_scheme`, and `quantization_status: "compressed"`.

Per linear, a pack-quantized INT4 weight is stored as `weight_packed` (int32), `weight_scale` (`(out, in/g)`),
`weight_shape`, and `weight_zero_point` if asymmetric. An NVFP4 weight adds `weight_global_scale`, and static
activations add `input_scale`. Attention layers carry `k_scale` and `v_scale`. The packing is
(`formats.pack_int4()`, `formats.pack_fp4()`):

```
INT4: codes −8…7 + 8, eight per int32, element 0 in the lowest 4 bits: [−8,−7,0,1,2,3,4,7] → 0xfcba9810
      a [4096, 896] INT4 weight → weight_packed [4096, 112]
FP4:  4-bit code = index in {0, .5, 1, 1.5, 2, 3, 4, 6} | sign << 3, two per byte, first in the low nibble
      [0.5, −6, 1.5, 0] → 0xf1, 0x03
```

**Other producers.**

- **GPTQModel** (7.5.0): `GPTQModel.load(id, QuantizeConfig(bits=4, group_size=128))`, then `.quantize(data)` and
  `.save()`; Turing and newer.
- **AutoAWQ**: deprecated, its role taken over by llm-compressor.
- **NVIDIA ModelOpt**: `mtq.FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG`, `INT8_SMOOTHQUANT_CFG`; vLLM serves its exports
  with `quantization="modelopt"` or `"modelopt_fp4"`.

Pre-quantized checkpoints exist on the Hub under the model vendors' and RedHatAI/nm-testing names (verify each id
before relying on it).

**Loading in vLLM** ([vllm-internals §8.1–8.3](../vllm-internals/vllm-internals-primer.md#8-quantization-and-weight-loading)):

- **Detection.** vLLM reads `quantization_config.quant_method` and picks the method. A `--quantization` that
  disagrees with the checkpoint raises an error, so omit the flag for pre-quantized models.
- **Online FP8.** To quantize a BF16 checkpoint at load time use `--quantization fp8_per_tensor`. Plain
  `--quantization fp8` still does it at v0.30.0 but raises at `main` (vLLM's `Fp8Config`; the serving lab's
  notebook 05 uses the old form).
- **Minimum capability.** The load fails when the GPU is below the method's minimum ("Minimum capability: …").

| Kernel / path | Minimum | T4 (7.5) | A100 (8.0) | L4, 4090 (8.9) | H100 (9.0) | B200 (10.0) |
|---|---|---|---|---|---|---|
| W4A16 Marlin (GPTQ, AWQ, compressed-tensors) | SM75 | yes | yes | yes | Machete | yes (Machete is SM90-only) |
| FP8 weight-only (Marlin FP8) | SM75 | yes | yes | native FP8 instead | native FP8 instead | native FP8 instead |
| FP8 W8A8 (CUTLASS; SM89 needs CUDA ≥ 12.4) | SM89 | runs W8A16 | runs W8A16 | yes (no block-FP8 CUTLASS) | yes + block, DeepGEMM | yes |
| INT8 W8A8 (CUTLASS) | SM75, < SM100 | yes | yes | yes | yes | **no** |
| NVFP4 W4A4 (CUTLASS/FlashInfer, CUDA ≥ 12.8) | SM100 | W4A16 | W4A16 | W4A16 | W4A16 | yes |
| FP8 KV cache | SM80 backends | **no** | FlashInfer | FlashInfer | FA3 | yes |

`quantcore.cost.supported()` encodes this table (vLLM 0.30.0 and `main`, from each kernel's `get_min_capability`
and the scheme dispatch in `compressed_tensors.py`; verify on your version); vLLM's docs table disagrees with the
code on the INT4 floor, so trust the code and the log line. A T4 needs `--dtype half`.

**The CPU and consumer path: llama.cpp GGUF.** Block formats with an fp16 scale per 32 (`q4_0` 18 B per 32 = 4.5
bits, `q8_0` 8.5 — `formats.bits_per_weight(4, 32)` and `(8, 32)`), and "k-quants" with 256-value super-blocks.
Llama-3.1-8B at `Q4_K_M` is 4.89 bits per weight, 4.58 GiB (llama.cpp README, verify).

```bash
./build/bin/llama-quantize in-bf16.gguf out-Q4_K_M.gguf Q4_K_M
```

vLLM reads GGUF only through the out-of-tree `vllm-gguf-plugin`, "highly experimental" (verify).

## 10. Choosing a scheme

`cost.table()` prices every scheme for one GPU and model. It says what each scheme runs as, the weights, decode
at batch 1 and 32, a 1,800-token prefill and the 2,000-token sessions, all SIMULATED with serving-engine §8's
assumptions (80% bandwidth, 60% FLOPs, 2 ms per step). With `minengine.perf`'s L4 it reproduces that section's
table to the digit. `cost.choose()` returns the least aggressive runnable scheme that meets every target, in a
starting order by typical accuracy cost: BF16, FP8 weight-only, FP8 W8A8, INT8 W8A8, INT4 W4A16, NVFP4. Measure
and reorder for your model. The decision:

| GPU generation | Decode-heavy, memory-bound | Prefill-heavy | Concurrency-bound | Notes |
|---|---|---|---|---|
| **Turing** (T4, 16 GB, fp16 only) | W4A16 (Marlin) | INT8 W8A8 + SmoothQuant | W4A16 (no FP8 KV) | an 8B model fits only at 8 or 4 bits |
| **Ampere** (A100) | W4A16 or FP8 weight-only | INT8 W8A8 | + FP8 KV (FlashInfer) | FP8 checkpoints run W8A16 |
| **Ada** (L4, RTX 4090) | W4A16 or FP8 | **FP8 W8A8** | + FP8 KV | CUTLASS FP8 needs CUDA ≥ 12.4; no block-FP8 CUTLASS |
| **Hopper** (H100, H200) | W4A16 (Machete) or FP8 | FP8 W8A8 (block or per-channel) | + FP8 KV (FA3) | the FP8 default |
| **Blackwell** (B200, RTX PRO 6000) | NVFP4 | **NVFP4 W4A4** (with smoothing or rotation, gated on an eval), FP8 | + FP8 KV (NVFP4 KV on B200/SM100 only) | no INT8 W8A8; W4A16 is Marlin, not Machete |

Worked through `cost.table()` and `cost.choose()` (notebook 05):

- **L4, Llama-3.1-8B, ≥ 48 sessions and a 1,800-token prefill ≤ 300 ms.** FP8 W8A8 with FP8 KV: 87 sessions,
  181 ms. This is serving-engine drill 6's answer, now computed by `choose`.
- **Free T4, Llama-3.1-8B, ≥ 20 sessions and a prefill ≤ 700 ms.** W4A16: 16-bit weights do not fit at all, 8-bit
  weights leave room for 16 sessions, INT4 for 29.
- **One H100, Llama-3.1-70B.** BF16 (141.1 GB) does not fit. FP8 (72.7 GB) fits, but with no useful concurrency.
  With the core's round memory inputs (0.9 × 80 GB − 1 GB, `cost.kv_blocks()`) it leaves no room for a single 4K
  session. At vLLM's defaults on the 79.65 GiB an H100 reports, the lab's `kv.size()` leaves room for 2 such sessions
  with a BF16 KV cache or 4 with FP8. INT4 (39.5 GB) serves 48 of them with FP8 KV (54 in the lab's model). FP8 means
  two GPUs with tensor parallelism, or an H200.
- **B200, Llama-3.1-8B.** NVFP4 W4A4 is the only 4-bit option that cuts prefill FLOPs (verify), and it needs the
  activation mitigations of §5.

**Cost per token** ([layer 01 §8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens)):
`$/M = $/GPU-hr ÷ (tokens/s × 3600 × utilisation) × 10⁶` (`cost.cost_per_million()`). Take an L4 at ~$0.70/hr
(verify, [`COMPUTE.md`](../../COMPUTE.md)) with every session it can hold decoding. BF16 fits 17 sessions, 230
tokens/s, **$0.844/M** at full utilisation. FP8 weights plus FP8 KV fit 87 sessions, 1,469 tokens/s, **$0.132/M**
(SIMULATED). That is 6.4× cheaper, not 2×: fewer bytes per step, and a 5× larger batch sharing each step.

This agrees with serving-engine §8's knob table. `kv_cache_dtype fp8` buys 2× KV capacity and faster long-context
decode. `quantization` buys memory and decode speed, and prefill speed only for FP8 (or INT8, FP4) W8A8. Both cost
accuracy and depend on kernel availability per GPU.

**Where to run it** — concept by concept, on GCP and elsewhere (prices and obtainability in [`COMPUTE.md`](../../COMPUTE.md)):

| To learn | T0 (laptop / Colab CPU) | Non-GCP GPU (T1) | GCP (T3) |
|---|---|---|---|
| §2–§6 mechanics, §10 decisions | `quant-core` notebooks 01–05 | — | — |
| producing checkpoints (§9) | the lab's numpy/torch path on a bundled tiny model | llm-compressor on any 16–24 GB GPU; a 0.5B model fits a free T4 (verify) | a `g2-standard-4` L4 VM (Spot) |
| INT4 serving, INT8 W8A8 | the lab's fake server (simulated) | Colab/Kaggle T4 (free, `--dtype half`; no FP8 compute or FP8 KV) | L4 via the serving lab's Cloud Run or GKE deploy |
| FP8 W8A8 and FP8 KV | emulation in `quantcore` | RTX 4090 on RunPod/Vast (~$0.3–0.4/hr, verify), L4 | L4 (G2, Cloud Run), H100 (A3) |
| FP4 / NVFP4 | `formats.nvfp4()`, `cost.table()` | a rented B200 or RTX PRO 6000 (verify) | A4 (B200), G4 / Cloud Run (RTX PRO 6000, verify) |

The lab's `deploy/any-gpu/` has `docker run` recipes per GPU generation and scheme. For GCP it points at
[`serving-engine/vllm-serving-lab/deploy/gcp/`](../serving-engine/vllm-serving-lab/deploy/gcp/) with a quantized
model; there is no new Terraform.

---

## In a design review

**The two-minute walkthrough.** "We quantize for three different reasons and pick the scheme per reason. Decode is a
weight read, so fewer weight bytes are faster tokens — weight-only INT4 is ~3× at batch 1 for an 8B model. But its
kernels do BF16 math on dequantized weights: on an L4 the GEMM hits that ceiling at ~120 tokens per step, its edge
shrinks from there (1.7× at 256), and by ~460 it is no faster than BF16, so long prefill chunks gain nothing.
Prefill gets faster only with formats the tensor cores multiply natively: FP8 W8A8 on Ada and Hopper, INT8 W8A8 on
older parts, NVFP4 on Blackwell. The KV cache is the third lever: FP8 KV halves it, and on a 24 GB card that doubles
the sessions, which does more for cost per token than speed does — 6× cheaper for an 8B model on an L4 with FP8
weights and KV, simulated. We checked what each checkpoint runs as on our GPUs: an FP8 checkpoint on an A100 is
weight-only, and NVFP4 is W4A4 only on Blackwell. Accuracy comes from granularity and calibration. We use groups of
128 for INT4 with GPTQ or AWQ, dynamic per-token FP8 activations, calibrated KV scales, and we keep the LM head,
embeddings, norms and routers in 16-bit. We gate on KL against the BF16 model and on task evals with their standard
errors, against a budget we wrote down first."

**Drill questions.**

1. *Why does INT4 weight-only speed up decode ~3× but not prefill at all?* — Decode streams the weights once per
   step, so it is bound by bytes and INT4 cuts them 3.9× (less the 16-bit LM head, KV and overhead). Prefill is
   compute-bound, and W4A16 kernels dequantize to BF16 before the MMA — same FLOPs. On an L4 the down_proj GEMM
   turns compute-bound at ~120 tokens per step in W4A16 and ~460 in BF16; between the two, INT4's edge shrinks
   from 3.9× to nothing.
2. *Our FP8 checkpoint runs on A100s and the H100 benchmark doesn't transfer. Why?* — A100s have no FP8 tensor
   cores: vLLM runs the checkpoint as weight-only FP8 through Marlin. The memory and decode-byte savings remain,
   but prefill runs at the BF16 rate. INT8 W8A8 with SmoothQuant is the Ampere prefill lever. Also, FP8 KV on an
   A100 switches attention to FlashInfer.
3. *INT4 RTN lost 5 points on our model. What next, before giving up on 4 bits?* — Calibrate. GPTQ, act-order,
   groups of 128 (or 64/32 if the kernel allows), and AWQ first for layers whose inputs have outlier channels. On
   our toy, GPTQ took INT4 from −6.6 to −0.6 points. Then check the LM head and routers are excluded, and that
   calibration data looks like traffic.
4. *The per-token INT8 activation error is 1.4%. Are we fine?* — Look per channel. With a few outlier channels
   30× larger, the ordinary channels carry ~11% error while the aggregate hides it. Use SmoothQuant (α 0.5–0.85)
   or FP8, whose relative grid keeps them at ~3%.
5. *Is FP8 KV safe to turn on?* — Usually, if the scales fit. It halves KV bytes (2× sessions) at <1% attention
   error on well-scaled heads. With the default scale 1.0, values far below 1 flush to subnormals (5% error in our
   example) and values above 448 saturate. Calibrate `k_scale`/`v_scale`, check the backend (none on a T4), and
   run a long-context eval.
6. *We need the 70B model on one H100. What are the options?* — BF16 (141 GB) does not fit. FP8 (72.7 GB) fits
   but leaves room for only 2–4 sessions of 4K (none at the core's round inputs), which is no useful concurrency.
   INT4 W4A16 (39.5 GB) fits with ~50 such sessions and FP8 KV. It costs
   accuracy to be measured, and prefill runs at BF16 speed. The alternatives are two H100s with tensor parallelism
   in FP8, or an H200 (141 GB).

---

## Glossary

| Term | Meaning |
|---|---|
| **Scale / zero point** | `x ≈ (code − zero) × scale`; symmetric formats have no zero point |
| **Granularity** | how many values share one scale: tensor, channel, group, token, block |
| **bpw** | bits per weight including scales and zero points (INT4 g128: 4.125 or 4.156) |
| **E4M3 / E5M2** | FP8 with 4 exponent + 3 mantissa bits (max 448) or 5 + 2 (max 57,344) |
| **E2M1** | FP4: magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| **E8M0** | an 8-bit power-of-two scale (MX formats) |
| **MXFP4 / NVFP4** | E2M1 with an E8M0 scale per 32 / an E4M3 scale per 16 plus an FP32 per-tensor scale |
| **W4A16, W8A8, W4A4** | weight bits / activation bits; "A16" means activations stay 16-bit (weight-only) |
| **RTN** | round to nearest, no calibration |
| **GPTQ** | column-by-column rounding with inverse-Hessian error compensation (Optimal Brain Surgeon) |
| **AWQ** | activation-aware scaling of salient weight columns before rounding, folded into the previous layer |
| **SmoothQuant** | migrating activation outliers into weights with per-channel scales `s = max|X|^α / max|W|^(1−α)` |
| **Act-order** | GPTQ visiting columns by decreasing Hessian diagonal |
| **Calibration data** | sample inputs used to compute H, activation statistics or static scales |
| **Dynamic / static scales** | computed per token at run time / fixed from calibration |
| **Epilogue** | the GEMM's final step that applies scales (and bias, activation) to the accumulator |
| **Dequantize-on-the-fly** | weight-only kernels (Marlin, Machete) expanding 4-bit weights to 16-bit in registers |
| **k_scale / v_scale** | per-layer FP8 KV scales in a checkpoint; 1.0 if absent |
| **KIVI** | sub-8-bit KV: keys per channel, values per token, a full-precision residual window |
| **Crest factor** | amax / rms: every 10× costs 20 dB of SQNR |
| **SQNR** | signal-to-quantization-noise ratio, ~6 dB per bit |
| **STE** | straight-through estimator: treat rounding as identity in the backward pass (QAT) |
| **NF4** | QLoRA's 4-bit normal-quantile format — a training format |
| **compressed-tensors** | the checkpoint format llm-compressor writes and vLLM reads |
| **KL / top-1 agreement** | distribution distance and argmax match between the quantized and reference models |

## Sources

Papers:

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023;
  code IST-DASLab/gptq (`gptq.py: fasterquant`, `quant.py`).
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024; code
  mit-han-lab/llm-awq (`awq/quantize/auto_scale.py`, `auto_clip.py`).
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML
  2023; code mit-han-lab/smoothquant (`smoothquant/smooth.py`).
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024; code jy-yuan/KIVI.
- Dettmers et al., *LLM.int8()* (2022) and *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).
- Micikevicius et al., *FP8 Formats for Deep Learning* (2022).
- Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0* (2023).
- Ashkboos et al., *QuaRot* (2024); Liu et al., *SpinQuant* (2024).
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024), and deepseek-ai/DeepSeek-V3 `README_WEIGHTS.md`,
  `inference/kernel.py`.
- Frantar et al., *Marlin* (IST-DASLab/marlin README).

Code and docs, read on 2026-09-26 (vLLM v0.30.0 and `main@a4eb3f25`, llm-compressor `c6fb66c`,
compressed-tensors `47f7d42`, lm-eval `d6de816`, GPTQModel `3b2e435`):

- vllm-project/vllm: `vllm/model_executor/layers/quantization/`, `kernels/linear/`, `config/model.py`,
  `config/cache.py`, `v1/attention/backends/`, `docs/features/quantization/`; v0.30.0 and `main`.
- vllm-project/llm-compressor: `examples/`, `modifiers/`.
- neuralmagic/compressed-tensors: `quant_scheme.py`, `quant_args.py`, `compressors/`.
- ModelCloud/GPTQModel; NVIDIA/TensorRT-Model-Optimizer (ModelOpt) docs.
- EleutherAI/lm-evaluation-harness (`lm_eval/api/metrics.py: mean_stderr`).
- ggml-org/llama.cpp `ggml-common.h` and the quantize README; openai/gpt-oss (`weights.py`, MXFP4).

In this repo:

- [serving-engine PRIMER §8](../serving-engine/PRIMER.md) and `minengine.quant`; `servelab.sizing`.
- [vllm-internals §6.3, §8](../vllm-internals/vllm-internals-primer.md) and the
  [FlashAttention deep dive §9](../flash-attention/flash-attention-deep-dive.md).
- [layer 01 PRIMER §1–3, §8](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) and
  [gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md).
- [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) (bytes per parameter).

## Verify list

Dated 2026-09-26; each item was read from the sources above, and can change.

- vLLM **0.30.0** (PyPI) and `main@a4eb3f25`: the `QuantizationMethods` list; `--quantization fp8` quantizes a BF16
  checkpoint online at 0.30.0 but raises at `main` (use `fp8_per_tensor`); bitsandbytes and GGUF are out-of-tree
  plugins (`vllm-bnb-plugin` 0.0.3, `vllm-gguf-plugin` 0.0.5).
- vLLM minimum capabilities: Marlin SM75; Machete SM90 only; CUTLASS FP8 SM89 (CUDA ≥ 12.4) and SM90+; CUTLASS
  block FP8 SM90+ (none on SM89); NVFP4 W4A4 SM100–129 with CUDA ≥ 12.8; INT8 W8A8 not on compute capability ≥
  10.0; MXFP4 minimum SM80; FP8 KV unavailable on SM75.
- vLLM KV-cache dtypes (`CacheDType`) including `fp8`, `fp8_e5m2`, `*_per_token_head`, `nvfp4` (SM100 family only),
  `turboquant_*`; `k_scale`/`v_scale`
  default 1.0; per-head scales only with FlashAttention; FP8 KV on L4/A100 selects FlashInfer, on H100 FA3.
- llm-compressor **0.14.0**, compressed-tensors **0.19.0**, lm-eval **0.4.13**, GPTQModel **7.5.0**, ModelOpt
  **0.47.0**, AutoAWQ 0.2.9 (deprecated): recipe names, defaults (GPTQ `dampening_frac` 0.01, `block_size` 128; AWQ
  `n_grid` 20; SmoothQuant 0.5 default), calibration sizes in the examples.
- compressed-tensors tensor names, scale shapes and INT4/FP4 packing order; NVFP4 global scale as a multiplier
  (448 × 6 / amax); MXFP4 scale exponent rounded up at a mantissa ≥ 1.75 (`round_to_power_2`); llm-compressor's
  GPTQ group scales taken from the weight observer before the loop.
- DeepSeek-V3 report §3.3: Hopper FP8 MMA accumulation of about 14 bits, promotion to FP32 every 128 elements.
- Reported accuracy figures quoted from sources: GPTQ README LLaMA perplexities; vLLM's FP8 GSM8K example
  (0.768 ± 0.0268 on 250 items); SmoothQuant tuned α per model; KIVI's memory and batch claims; ModelOpt's NVFP4
  QAD note (2026-09-16).
- GPU figures (dense TFLOP/s, bandwidth, memory) from `roofline.specs` and `quantcore.cost.GPUS`: T4, A100, L4, H100,
  B200, RTX PRO 6000; RTX 4090 figures are from `servelab.GPUS` and unverified.
- Model configs: Llama-3.1-8B/70B, Qwen2.5-0.5B/1.5B (parameters, layers, heads, vocab, tied embeddings).
- Prices: L4 ~$0.70/hr and H100 ~$11/GPU-hr on demand in us-central1; RTX 4090 ~$0.3–0.4/hr on Vast/RunPod.
- llama.cpp k-quant sizes (Q4_K_M 4.89 bpw on Llama-3.1-8B); Hub ids of pre-quantized checkpoints.
