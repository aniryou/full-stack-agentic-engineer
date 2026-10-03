# Quantization for inference: number formats, calibration, kernels and the accuracy you pay

*A primer for layer 04. Snapshot: September 2026. Each product fact has a date and the mark (verify). Each formula
has a worked number, and it names the function in [`quant-core/`](quant-core/) (package `quantcore`) that
calculates it. `quant-core/tests/test_primer_numbers.py` calculates each such number again, and it fails if this
file no longer gives that number. The latencies from `quantcore.cost` are **simulated** values from a roofline
model, not measurements.*

This primer is the deep dive behind one section of the serving-engine primer. That section is
[serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization). It gives a survey of the formats, scale granularity,
weight-only against W8A8, FP8 KV, and the gain of each for Llama-3.1-8B on an L4.
`minengine.quant` calculates it.

This primer starts where that survey stops. It examines the number formats down to their bit patterns, and it
includes the Blackwell block formats. It also covers these subjects:

- the calibration algorithms that make 4 bits usable (GPTQ, AWQ, SmoothQuant),
- activation and KV-cache quantization below 8 bits,
- the kernels, and why the speed of a format depends on the GPU generation,
- how to measure the accuracy that you pay,
- how you produce a quantized checkpoint, and how vLLM loads it,
- how to select a scheme.

You can learn everything at tier T0 with [`quant-core/`](quant-core/). [`quant-lab/`](quant-lab/) produces real
checkpoints with llm-compressor, serves them with vLLM and measures them (T1). For T3, it points to the Cloud Run
and GKE deploys of the serving lab.

---

## The one-minute version

Quantization stores numbers on a coarser **grid** with a **scale**: $x \approx \text{code} \times \text{scale}$. It
makes work faster only where the bytes or the FLOPs that it removes are the limit.

- **Decode** streams every weight in each step. Thus fewer weight bytes give faster tokens. Weight-only INT4
  (**W4A16**) decodes an 8B model ~3× faster at batch 1.
- **Prefill** is compute-bound. Weight-only kernels do 16-bit math on dequantized weights. Thus prefill becomes
  faster only with the formats that the tensor cores multiply natively. These are **FP8 W8A8** on Ada, Hopper
  and Blackwell, **INT8 W8A8** on Turing to Hopper, and **FP4 W4A4** on Blackwell.
- The **KV cache** is the third lever. FP8 KV halves it and doubles the sessions per GPU.

The **granularity** and the **outliers** decide the accuracy. The largest value that shares a scale sets that
scale. Thus INT4 needs groups of 32–128 and a calibration method.

- **GPTQ** lets later columns absorb the rounding error of each column, through the inverse Hessian of the
  calibration inputs.
- **AWQ** scales up the weights that meet large activations.
- **SmoothQuant** moves activation outliers into the weights, so that INT8 activations survive.
- Float grids (FP8 E4M3, NVFP4) tolerate outliers better than integer grids, because their error is relative.
- Keep the LM head, embeddings, norms, softmax and MoE routers in 16-bit.

Checkpoints come from llm-compressor as compressed-tensors. vLLM detects the format and selects a kernel for each
GPU generation. An FP8 checkpoint on an A100 runs weight-only. Measure the damage first as KL and top-1 agreement.
Then measure the task accuracy **with its standard error**. Compare both against a budget that you write down
before you look.

---

## 1. Why quantize, and what it can and cannot speed up

**The roofline argument.** A kernel takes $\max(\text{FLOPs} / \text{peak}, \text{bytes} / \text{bandwidth})$
([layer 01 PRIMER §2](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#2-the-roofline-model)). The
arithmetic intensity of an LLM step is approximately the number of tokens in it
([§3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#3-llm-inference-on-the-roofline)). Thus:

| What a step is bound by | When | What quantization must cut | Schemes that cut it |
|---|---|---|---|
| weight bytes | decode at small batch | bytes per weight | W4A16, W8A16, FP8/INT8 W8A8, NVFP4 |
| KV bytes | decode at long context × large batch | bytes per KV element | FP8 KV, INT4/INT2 KV |
| FLOPs | prefill, decode past the ridge | the precision that the tensor cores multiply in | FP8/INT8 W8A8, FP4 W4A4, **not** weight-only |
| memory capacity | always, for concurrency | weight + KV bytes | all of them |

Layer 01 §3.5 calculates this for Llama-3.1-8B on an H100. FP8 is exactly 2.00× in both regimes, because it halves
every byte and doubles the peak. W4A16 is 3.80× at batch 1 but 1.54× at batch 64. At batch 64, half of the bytes
are KV cache, and W4A16 does not change them.

The same arithmetic for one linear layer shows where weight-only no longer gives a gain. Take the `down_proj` of
Llama-3.1-8B ($K$ = 14,336, $N$ = 4,096) with $M$ tokens in the step. Its FLOPs are $2\,MKN$, and its bytes are
$K \cdot N \cdot w + M \cdot (K \cdot a + 2N)$. The layer becomes compute-bound above this value
(`cost.crossover_tokens()`):


$$
M^* = \frac{K \cdot N \cdot w \,/\, \mathrm{BW}}{2 \cdot K \cdot N \,/\, \text{peak} - (K \cdot a + 2N) \,/\, \mathrm{BW}}
$$

```
L4 (121 TFLOP/s bf16, 242.5 fp8, 0.30 TB/s):   BF16 462 tokens    FP8 W8A8 478    W4A16 119
```

FP8 halves the bytes and doubles the peak. Thus its crossover almost does not move. W4A16 makes the bytes 3.9×
smaller and keeps the BF16 peak. Thus it becomes compute-bound early, at ~120 tokens per step. BF16 stays
byte-bound until ~460.

Up to ~120 tokens, W4A16 keeps the full byte ratio. The table in vllm-internals §8.1 shows this, and
`cost.gemm_time()` in `tests/test_repo_numbers.py` calculates that table again. It shows W4A16 at 102 µs against
BF16's 392 µs for one token. Between ~120 and ~460 tokens, the speedup of W4A16 decreases. In that range, W4A16
pays for BF16 math, but BF16 still pays for bytes. The speedup is 1.7× at 256 tokens, and nothing at 462 and
beyond (the two are equal at 2,048).

Thus decode batches of a few hundred still get a gain from INT4, but long prefill chunks do not. Real kernels lose
the gain sooner, because the dequantization is not free (§4).

**Tensor-core throughput by precision** is the other half. From BF16 to FP8, an L4 goes from 121 to 242.5 TFLOP/s,
and an H100 from 989.4 to 1,978.9. A B200 goes from 2,250 to 4,500, and to 9,000 at FP4 (dense, `roofline.specs`,
verify). Each time the precision halves, the rate doubles. But this is true only on parts that have that datapath
([gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md#4-tensor-cores-the-biggest-change-since-you-last-looked)).

**What it cannot speed up.** Attention over the KV cache is not a weight GEMM. Weight quantization does not change
it, and only KV quantization decreases its bytes. The constant overheads of the step (launches, sampling, scheduling)
do not decrease.

Each step reads a 16-bit LM head. For Llama-3.1-8B it is 1.05 GB of the 4.65 GB an INT4 decode
step streams (`cost.Model.streamed_bytes()`; the checkpoint is 5.70 GB with the input embedding). The step only
gathers from the input embedding. The L4 table in serving-engine §8 shows the net effect: single-stream decode is
3× faster, not 3.9×.

**The accuracy budget.** Each scheme costs some accuracy. The cost depends on the model, the task and the recipe.
The rest of this primer shows these things:

- how to keep the cost small (§3–§6),
- how to measure it (§8),
- how to trade it against speed and memory (§10).

## 2. Number formats

### Integer and floating-point grids

**Integers.**

$$
\begin{aligned}
\text{code} &= \operatorname{clip}(\operatorname{round}(x / \text{scale}) + \text{zero},\ q_{\min},\ q_{\max}), \\
\hat{x} &= (\text{code} - \text{zero}) \times \text{scale}
\end{aligned}
$$

(`formats.quantize_int()`, `dequantize_int()`). Three conventions are in use, and the repo has all three:

| Convention | INT4 codes | Scale | Used by |
|---|---|---|---|
| symmetric, restricted | −7…7 | amax / 7 | `minengine.quant`, SmoothQuant's fake-quant (`/127`) |
| symmetric, full | −8…7 | amax / 7.5 | GPTQ's `quant.py`, compressed-tensors (`_calculate_range`), vLLM `uint4b8` |
| asymmetric | 0…15, zero point | (max − min) / 15 | AWQ (`zero_point=True`), KIVI, `W4A16_ASYM` |

`formats.int_range()` and `int_scale()` implement all three. The full convention uses the 16th code. Thus its step
is 7/7.5 of the restricted step, and the result is 0.24 dB better on Gaussian INT4 g32 weights. The asymmetric
convention widens the range to include 0, and it covers [min, max] instead of [−amax, amax]. For
`[−0.3, 0, 0.05, 0.4, 1.2]` it gives scale 0.1 and zero point 3, and 0 is exact (`formats.asym_params()`). It
halves the step for one-sided data, for example ReLU outputs.

**Floating point.** A float has a sign, $e$ exponent bits with a bias, and $m$ mantissa bits. Within one power of
two, the step is $2^{\text{exponent} - m}$, so the **relative** error is at most $2^{-(m+1)}$ at any magnitude.
Below the smallest normal exponent, the distance between values stays constant (subnormals).
`formats.FloatFormat.grid()` lists every value from the bit patterns, and `formats.to_float()` rounds to that grid
(ties to even, with saturation). On 100,000 values, `to_float` is bit-identical to the `float8_e4m3fn` and
`float8_e5m2` casts of torch (notebook 01).

| Format | Bits (s-e-m), bias | Largest | Smallest normal | Smallest subnormal | Max relative error | Normal range |
|---|---|---|---|---|---|---|
| FP8 **E4M3** (`fn`) | 1-4-3, 7 | **448** | 2⁻⁶ | 2⁻⁹ | 6.25% | 2^14.8 |
| FP8 **E5M2** | 1-5-2, 15 | **57,344** | 2⁻¹⁴ | 2⁻¹⁶ | 12.5% | 2^29.8 |
| FP4 **E2M1** | 1-2-1, 1 | 6 | 1 | 0.5 | not applicable (8 magnitudes) | 2^2.6 |
| BF16 | 1-8-7 | ~3.4 × 10³⁸ | 2⁻¹²⁶ | 2⁻¹³³ | 0.39% | fp32's |
| FP16 | 1-5-10 | 65,504 | 2⁻¹⁴ | 2⁻²⁴ | 0.05% | 2^30 |

E4M3 has no infinities. S.1111.111 is NaN, so its top value is 1.75 × 2⁸ = 448. That leaves 254 finite codes and
253 distinct values. E5M2 keeps the IEEE infinities and has half the precision. Inference uses **E4M3** for
weights, activations and the KV cache (vLLM's FP8 linear method accepts only `float8_e4m3fn`, verify). E5M2 is
mostly a gradient format.

On $\mathcal{N}(0,1)$ values, E4M3 rounds with a 2.2% median and 5.9% maximum relative error, E5M2 with 4.3% and
11.1% (notebook 01).

The full grid of E2M1 is {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The reference rounding of vLLM uses the thresholds
0.25/0.75/1.25/1.75/2.5/3.5/5. That is round-half-to-even, the same as `to_float`. For BF16 against FP16 in
attention, see FlashAttention deep dive §9.1: bf16 has range, and fp16 has precision. A T4 has no bf16 at all.

### MXFP4 and NVFP4

**Block formats** give a 4-bit float grid a local scale:

- **MXFP4** (OCP Microscaling). It has E2M1 elements and one **E8M0** scale per **32** values. The scale is a pure
  power of two, and the format stores it as exponent + 127. The OCP rule is

    $$
    \text{shared exponent} = \lfloor \log_2(\text{block amax}) \rfloor - 2
    $$

    where $2 = \lfloor \log_2 6 \rfloor$ (`formats.mxfp4()`). A block whose amax is 5 gets exponent 0 (byte 127).
    A block whose amax is 0.75 gets −3 (byte 124). The scale floors to a power of two. Thus the block max is in
    [4, 8) on the E2M1 grid. Above 6, the format clips the value, and near 4 the top codes stay unused.

    The producers differ here. compressed-tensors writes the MXFP4 checkpoints of llm-compressor. It first rounds
    amax to a power of two, and it rounds **up** when the mantissa is ≥ 1.75 (`round_to_power_2`). Thus its block
    max is in [3.5, 7) (`formats.mxfp4(rule="compressed-tensors")`, and the lab's `fp4.mxfp4_scale_exponent()`).

    A block whose amax is 7.5 gets exponent 0 under the OCP rule, and it clips to 6. Under the compressed-tensors
    rule, it gets exponent 1 and goes to 8. That clips 28% of Gaussian blocks' maxima instead of 43%. It also
    wastes a small part of the range just under each power of two. gpt-oss ships its MoE weights in MXFP4.

- **NVFP4**. It has E2M1 elements, one **FP8 E4M3** scale per **16**, and one FP32 scale per tensor. The tensor
  scale is a multiplier $g = 448 \times 6 / \operatorname{amax}(\text{tensor})$. The scale of each block is
  $\operatorname{E4M3}(g \times \text{block_amax} / 6)$, and $\hat{x} = \text{element} \times \text{block_scale} / g$
  (`formats.nvfp4()`). This is the same as `generate_gparam` in compressed-tensors and `ref_nvfp4_quant` in vLLM.

The grid that wins depends on the data (`granularity.error()`, relative error, seed 0, 256×512 weights):

| 4-bit scheme | Bits per weight | Gaussian | Heavy-tailed (Student-t, 3 d.o.f.) |
|---|---|---|---|
| INT4 g32, fp16 scale (full convention) | 4.5 | 0.0951 | 0.1433 |
| FP4 E2M1 g32, fp16 scale | 4.5 | 0.1010 | 0.1061 |
| MXFP4 (OCP exponent) | 4.25 | 0.1145 | 0.1404 |
| MXFP4 (compressed-tensors exponent) | 4.25 | 0.1123 | 0.1373 |
| NVFP4 | 4.5 | 0.0953 | 0.0909 |

On Gaussian data, an evenly spaced grid is as good as a float grid. On heavy tails, the float grid wins, because
most values are small and E2M1 puts its codes near zero. At every distribution, the power-of-two scale of
MXFP4 gives it a larger error than the E4M3 scale gives NVFP4.

### Quantization noise and outliers

**The error model: ~6 dB per bit.** Rounding noise has the power $\text{step}^2 / 12$. With
$\text{step} = 2 \cdot \text{amax} / 2^b$, the signal-to-quantization-noise ratio is (`formats.sqnr_rule_db()`):

$$
\begin{aligned}
\mathrm{SQNR} &\approx 6.02\,b + 4.77 - 20 \log_{10}(\text{amax} / \text{rms}) \\
&\text{a sine (crest } \sqrt{2}\text{): the textbook } 6.02\,b + 1.76
\end{aligned}
$$

Take 128×256 Gaussian weights with one scale per channel. Here (crest factor 3.07), the rule predicts 43.2 dB at
INT8 and the grid measures 43.1; at INT4 it predicts 19.1 and the grid measures 18.4. The model is valid from 4
bits up. At 2–3 bits, where the noise is no longer busy, it predicts too high. In serving-engine §8, the measured
slope is the same: 43.0 dB per channel at INT8 against 17.9 at INT4.

**Outliers dominate.** The last term is the crest factor, and each 10× of amax over rms costs 20 dB, which is more
than three bits. A 4,096-value row of rms 0.02 has a crest factor of 3.9. Add one value of 0.8, and it becomes 33.9.
Then INT8 per-channel SQNR falls from 41.1 dB to 22.2 dB (the rule predicts 22.3). One value costs **3.1 bits**
(notebook 01, exercise 1.3).

LLM weights have such values, always in the same channels. LLM activations have many more of them. §3–§5 solve that
problem.

## 3. Granularity and the bits-per-weight budget

**Who shares a scale.** `granularity.quantize()` implements each option for `W[out, in]` (the torch and checkpoint
layout). It returns the scales in the shapes of compressed-tensors:

| Granularity | Scale shape | Isolates | Typical use |
|---|---|---|---|
| per tensor | (1,) | nothing | FP8 weights and static activations |
| per channel (output row) | (out, 1) | an outlier row | INT8 and FP8 weights |
| per group of g inputs | (out, in / g) | an outlier within g inputs | INT4 (g = 128, 64, 32), FP4 (16, 32) |
| per token | (tokens, 1) | an outlier token | dynamic INT8/FP8 activations |
| per block r × c | (out / r, in / c) | an outlier tile | DeepSeek-V3 FP8 (128 × 128) |

The six-scheme table in serving-engine §8 (INT8 per tensor to INT4 g32 on one Gaussian weight) is the survey.
The `quantcore` package calculates all six rows again from the same seed (`tests/test_repo_numbers.py`). It also
calculates the outlier-row and SmoothQuant numbers of that section again. This primer does not repeat them.

**Rows are easy, columns are not.** Per-channel scales isolate an outlier *row*. But an outlier *input column* of W
is in every row, and it sets the scale of every row. Take a 256×512 weight with one column 20× larger, and quantize
it to INT4 (full convention). The error on the **other** columns is 0.612 per channel, 0.320 with groups of 128,
0.236 with 64 and 0.176 with 32 (notebook 02). Groups limit the damage, but they do not remove it.

The other remedies for a *weight* outlier also act on W. GPTQ lets the columns that are not yet rounded compensate
for the rounding of the outlier column. A rotation spreads the outlier over every column (both §4). AWQ does not
apply, because its scales come from activation magnitudes.

**Activation outliers are a different problem.** The weight columns that meet a large activation channel are
usually ordinary. In the first up-projection of `quantcore.TinyModel`, the columns behind its four outlier channels
have absmax 0.353–0.392 against a median column of 0.375. Thus no weight granularity isolates them. But they cause
98.4% of that layer's INT4 g32 output error (`granularity.output_error_by_input()`). The reason is that the
rounding error of a column reaches the output multiplied by its input.

AWQ corrects this case on the weight side (§4). SmoothQuant or a float grid correct it on the activation side (the
next paragraph, and §5).

**Activations.** The engine quantizes them at run time. It uses one scale per token (dynamic, one max-reduction per
token), or one static per-tensor scale that calibration sets. LLM activations have a few channels that are large in
*every* token. In `quantcore.TinyModel`, they come from RMSNorm gains of 25–40×. The input of the first
up-projection has four channels with absmax 76–120 against a median channel absmax of 3.0.

Per-token INT8 gives 1.4% error on that tensor as a whole, but **11.1%** on the 60 ordinary channels. The reason is
that the outlier sets the step of each token. Per tensor gives 27.5% on them, and per-token FP8 **2.7%**, because
a float grid keeps the relative precision of small values (notebook 02).

Static scales saturate all values that calibration did not see. On the INT8 output of the same layer on held-out
inputs, the calibration max gives 3.53% error. The 99.99th percentile gives 3.22%. The 99th percentile clips the
outlier channels themselves and gives 24.5%. Dynamic per-token scales give **1.51%** (`w8a8.w8a8_matmul()`).

For this reason, the `FP8_DYNAMIC` and INT8 `W8A8` recipes quantize activations dynamically per token. They need no
calibration data for the activations.

**The bits-per-weight budget.** The checkpoint also stores the scales and the zero points
(`formats.bits_per_weight()`):

$$
\text{bits per weight} = \text{element bits} + \frac{\text{scale bits} + \text{zero-point bits}}{\text{group size}}
\ \left[ + \frac{\text{tensor-scale bits}}{\text{numel}} \right]
$$

```
INT8 per channel, 4,096 inputs, fp16 scale       8.004
INT4 g128 symmetric, fp16 scale                 4.125      <- compressed-tensors W4A16; minengine.quant's convention
INT4 g128, fp16 scale + 4-bit zero point         4.15625    <- AWQ, GPTQ-format; servelab.sizing's ("4.16")
INT4 g32, fp16 scale / + zero point              4.5 / 4.625
MXFP4 (E8M0 per 32)                              4.25
NVFP4 (E4M3 per 16, FP32 per tensor)             4.5
FP8, FP32 scale per 128 × 128 block              8.002
```

Both INT4 figures in the repo are correct for their assumptions. Symmetric compressed-tensors W4A16 stores no zero
point. The IST GPTQ reference quantizes asymmetrically by default. GPTQ-format checkpoints (AutoGPTQ, GPTQModel)
always store packed `qzeros`. Thus they cost ~4.16 bits (the figure in vllm-internals §8.1).

Also, the full model is never 16/4.125 smaller. The recipes keep the embedding and the LM head in 16-bit
(`cost.Model.weight_bytes()`):

| Model | BF16 | FP8 | INT4 g128 (4.125) | Ratio | Kept 16-bit |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (tied embedding) | 0.988 GB | 0.630 GB | 0.457 GB | 2.16× | 27.6% of parameters |
| Llama-3.1-8B (untied) | 16.06 GB | 9.08 GB | 5.70 GB | 2.82× | 13.1% |
| Llama-3.1-70B | 141.1 GB | 72.7 GB | 39.5 GB | 3.57× | 3.0% |

The kernels limit the group sizes. The group size must divide `in_features` (llm-compressor gives an error at
initialization, and Marlin rejects such a layer). Marlin supports groups of −1 (per channel), 32, 64 and 128, and
Machete supports −1, 64 and 128 (vLLM source, verify). A 576-wide model cannot use g128.

## 4. Weight-only post-training quantization

### RTN and GPTQ

**Round to nearest (RTN)** needs no data. Each weight goes to the nearest code of its group (`gptq.rtn()`). On
the tiny model, INT8 is free, INT4 g32 costs 6.6 points (90.3% → 83.7%) and INT3 22 points (notebook 03).
Larger models tolerate RTN better. In the GPTQ paper, LLaMA-7B goes from 5.68 to 6.29 WikiText-2 perplexity at
4-bit RTN, and to 25.5 at 3-bit (README, verify).

**GPTQ** minimises the output error of each layer, $\lVert X W^\top - X Q^\top \rVert^2$ over calibration tokens
$X$, not the weight error. The curvature of that objective is the Hessian (`gptq.hessian()`):

$$
\begin{aligned}
& H = \frac{2}{n} \sum_t x_t x_t^\top && \text{(in × in, from calibration activations)} \\
& H \mathrel{+}= 0.01 \cdot \operatorname{mean}(\operatorname{diag} H) \cdot I && \text{damping, } \texttt{percdamp} \\
& U = \operatorname{chol}(H^{-1})^\top && \text{upper Cholesky factor, } H^{-1} = U^\top U \\
& \text{for each input column } j\text{:} && (\texttt{gptq.gptq}) \\
& \qquad q_j = \operatorname{quantize}(w_j) && \text{on the group's grid} \\
& \qquad e = (w_j - q_j) \,/\, U[j, j] \\
& \qquad W[{:}, j{+}1{:}] \mathrel{-}= e \otimes U[j, j{+}1{:}] && \text{the Optimal Brain Surgeon update}
\end{aligned}
$$

GPTQ pushes the rounding error of each column onto the columns that are not yet rounded, in the directions that the
inputs actually use. These details are important in practice:

- **Group scales.** `gptq.gptq()` calculates the scale of each group when the loop gets to the group, on the
  weights that it already updated. The `GPTQModifier` of llm-compressor sets them at the start, from the weight
  observer on the original weights. In both cases, the codes carry the compensation, not the scales.
- **Act-order** visits the columns from the largest $H_{jj}$ to the smallest. The most-used inputs go first, while
  the most columns stay to absorb their error. With *static groups* (parameters set at the start), the checkpoint
  layout does not change.
- GPTQ sets the inputs that never fire ($H_{jj} = 0$) to zero.
- The "lazy batch" of the reference implementation delays the updates beyond a 128-column block, for GPU
  efficiency. Its OBS updates are the same. Its result is also the same per channel, or when groups start on block
  boundaries. A group that starts mid-block takes its scale from weights that do not have the in-flight
  updates of the block yet. That is a slightly different choice. `tests/test_gptq.py` compares
  `gptq.gptq()` with a transcription of the reference in both cases.
- With uncorrelated inputs, H is diagonal, and U has no off-diagonal terms. Then GPTQ *is* RTN (a test examines
  this).

GPTQ is strongest where the inputs have correlations. The down-projections of the tiny model read a ReLU of a 64-dim
stream, spread over 256 dims. Thus 99% of their H lies in 13–37 directions. GPTQ makes their output error 5–8×
smaller (0.1000 → 0.0174, 0.0197 on held-out data). On the model, INT4 g32 recovers to **89.7%** (KL 0.259 → 0.048)
and INT3 to **84.4%** (notebook 03, `tinymodel.quantize_model()`).

Real layers are less redundant than the layers of this toy model. Thus expect smaller gains. At 3-bit, the GPTQ
paper reports 8.07 perplexity against RTN's 25.54 on LLaMA-7B (verify).

### AWQ, calibration data and rotations

**AWQ** protects the weights that meet large activations. The rounding error of a weight reaches the output
multiplied by its input. Thus the input channels with large activations own most of the output error.

AWQ multiplies those weight columns by ${s > 1}$ before it rounds them, and it divides the activations by $s$. It
folds ${1/s}$ into the previous RMSNorm gain (or the rows of the previous linear). Thus
$(X/s)(W \cdot s)^\top = X W^\top$ exactly (`tinymodel.TinyModel.fold()`). AWQ searches for the scale, and it does
not solve for it (`awq.search_scale()`, as in llm-awq's `auto_scale.py`):

$$
\begin{aligned}
s &= \operatorname{mean} \lvert x \rvert^{\alpha}, \quad s \mathrel{/}= \sqrt{\max s \cdot \min s}, \quad
\alpha \in \lbrace 0, \tfrac{1}{20}, \ldots, \tfrac{19}{20} \rbrace \\
\operatorname{loss}(\alpha) &= \operatorname{mean}\bigl((X/s) \cdot Q(W \cdot s)^\top - X W^\top\bigr)^2
\end{aligned}
$$

Keep the best $\alpha$. $\alpha = 0$ is RTN, so AWQ never loses on calibration data.

On the first up-projection of the tiny model, the loss curve is U-shaped, with its minimum at α = 0.30, 0.58 of
RTN's loss. With too small a scale, the salient channels stay coarse. With too large a scale, they set the group
scale for the whole group. AWQ decreases the output error of the up-projections by ~25% (0.0939 → 0.0715). It does
not do much for the down-projections, which have no dominant channels.

On the up-projections alone at INT3, it beats GPTQ: 85.9% against 84.7%. The two methods compose: the AWQ scales go
first, then the GPTQ rounding (llm-compressor: an `AWQModifier`, then a `GPTQModifier`). They combine for the lowest
KL of all: 0.0307 at INT4 g32. The llm-awq code also searches for a per-group **clipping** threshold (amax × 1.00 down to
0.55), and it skips the q/k projections (verify). The "duo" variant of llm-compressor divides by
$\text{w_mean}^{1-\alpha}$.

**What calibration data does and does not do.** It supplies H (GPTQ) or activation statistics (AWQ, SmoothQuant). It
teaches the model nothing. It cannot save a format that is too coarse.

On the tiny model, GPTQ INT3 gets 78.4% with 16 samples, 82.4% with 64, 84.4% with 256 and 85.0% with 1,024. Three
other 256-sample draws give 83.4–84.6%. This shows that past a few hundred samples, the samples that you drew are as
important as their number. With 256 samples from only 2 of the 16 classes, it still gets 83.0%, and with 256
samples of pure noise 84.5%. Outlier channels and input correlations are properties of the weights and the norm
gains. Any input shows them.

On real models, the text is still important. Chat templates, languages and long contexts change the activation
statistics. Calibrate on data that looks like your traffic. The recipes use 512 sequences × 2,048 tokens (GPTQ),
128–512 × 512 (AWQ) and 512 × 512 (SmoothQuant) (§9, verify).

**Rotations** (QuaRot, SpinQuant, QuIP) multiply W and X by an orthogonal matrix, often a Hadamard. This does not
change $X W^\top$, and it spreads the energy of an outlier over all channels. The FlashAttention deep dive §9.4
measures this on attention. There, a random Hadamard takes the INT4 error on a K with one 20× channel from 52% to
24%. llm-compressor ships SpinQuant and QuIP transforms. The details of QuaRot are (verify).

### Weight-only kernels

**Kernels: dequantize on the fly.** A W4A16 kernel reads packed 4-bit weights and their group scales. It
dequantizes them to FP16/BF16 **in registers**. Then it sends them to the ordinary 16-bit tensor-core MMA with FP32
accumulation ([vllm-internals §8.1](../vllm-internals/vllm-internals-primer.md#81-how-a-quantization-method-is-chosen)).
It moves a quarter of the bytes and does exactly the BF16 math:

- **Marlin**. FP16×INT4, with "close to ideal (4x) speedups up to batchsizes of 16-32 tokens". With g128 scales,
  the byte ratio is 16/4.125 = **3.88×** (the README's "optimal 3.87x"). The vLLM port runs from SM75, but the
  original README says SM80 (verify).
- **Machete**. The CUTLASS mixed-input GEMM of vLLM, for Hopper only.
- **ExLlama**. SM60+ (verify).

vLLM tries its mixed-precision kernels in this order: CutlassW4A8, Machete, Marlin, Conch, Exllama, TritonW4A16,
Humming. It logs `Selected <kernel> for <module>`. The result is the crossover of §1: W4A16 wins decode, and it
ties or loses prefill. NVIDIA measured weight-only NVFP4 slower than BF16 in 10 of 12 GEMM shapes on Blackwell,
because of the dequantize-to-BF16 fallback. In the same measurement, W4A4 beat BF16 in 9 of 12 (ModelOpt,
2026-09-16, verify).

## 5. Weight-and-activation quantization

**The epilogue.** A W8A8 GEMM multiplies 8-bit codes and accumulates the products in a wide register. The scales
factor out of the sum. Thus the kernel applies them one time per output (`w8a8.w8a8_matmul()`):

$$
y[t, j] = s_x[t] \cdot s_w[j] \cdot \sum_k \mathit{qx}[t, k] \cdot \mathit{qw}[j, k]
$$

INT8 uses an INT32 accumulator. FP8 uses FP32 (but see the last item of the list that comes next).

An emulation with integer codes and integer accumulation gives the fake-quantized product to within 10⁻¹⁴
(notebook 04). This has three consequences:

- The activation scale can change per token, and the weight scale per output channel. But **neither can change
  along k**, the reduction axis. That is why SmoothQuant must move per-input-channel variation into the weights,
  and must not scale it.
- The worst-case INT8 accumulator for a reduction of length K is 127² × K. It stays below 2³¹ up to K = 133,144,
  which is far above any hidden size.
- Block formats need one partial sum per 128-wide k block. The kernel rescales each partial sum by the two scales
  of its block before the addition (`w8a8.block_fp8_matmul()`, the DeepGEMM and CUTLASS block-scaled pattern).
- FP8 "FP32 accumulation" is not exactly that inside the tensor core. The DeepSeek-V3 report found that the FP8
  MMA of Hopper keeps about 14 bits of accumulator precision. Thus the DeepSeek-V3 GEMMs promote each 128-element
  partial sum to FP32 registers. That is a second reason for the 128-wide k blocks (DeepSeek-V3 report §3.3,
  verify).

### INT8, FP8 and FP4 schemes

**INT8 W8A8 with SmoothQuant.** Outlier channels badly damage INT8 per-token activations (§3). SmoothQuant divides
activation channel $j$ by $s_j$ and multiplies weight column $j$ by $s_j$. Then it folds ${1/s}$ into the previous
norm (`smoothquant.smooth_scales()`):

$$
s_j = \frac{\max \lvert X_j \rvert^{\alpha}}{\max \lvert W_j \rvert^{1-\alpha}}
$$

The default is $\alpha = 0.5$. The adjusted values are 0.85 for Llama-3-8B and 0.8 for Mistral/Mixtral (verify).

On the up-projection of the tiny model, INT8 W8A8 output error falls from 1.52% to **0.57%** at α = 0.5. The sweep
is U-shaped between α = 0 and 1. Over the full model, KL falls 2.7× (0.00296 → 0.00110), at no cost at run time.
`tests/test_repo_numbers.py` calculates the serving-engine example again: one activation channel 60× larger, and
6.6× less output error. The INT8 recipe of llm-compressor is `SmoothQuantModifier(smoothing_strength=0.8)`, and
then `GPTQModifier(scheme="W8A8")`.

**FP8 W8A8.** The weights are E4M3 per tensor, per channel or per 128 × 128 block. The activations are E4M3 per
token (dynamic) or per tensor (static). DeepSeek-V3 ships `weight_block_size [128, 128]` with an FP32
`weight_scale_inv` per block. It quantizes activations per token per 128 channels on the fly
(`act_quant(x, block_size=128)`).

Because FP8 has a float grid, it usually needs no smoothing. With the outlier channels of the tiny model, FP8
per-token activations have 2.7% error on the ordinary channels, against 11.1% for INT8. But for well-scaled
values, three mantissa bits are coarser than the seven of INT8. Over the full model, FP8 W8A8 has 6× the KL of
INT8 W8A8 (0.0186 against 0.00296), with accuracy within noise.

Block scales have a small effect on FP8 at ordinary ranges. Take a weight tile 30× larger than the rest.
Per-tensor, per-channel and 128 × 128-block FP8 all give 3.7–3.9% output error, while INT8 per token and channel
gives 0.9% (notebook 04). Block scales are useful when ranges exceed E4M3's $2^{14.8}$, in training. They are also
useful because a kernel that already tiles by 128 can apply them at no cost. There is no CUTLASS block-FP8 kernel
for SM89 (vLLM source, verify).

**FP4 W4A4 (Blackwell).** The scheme has NVFP4 weights. The kernel quantizes the activations per 16 at run time,
with a calibrated global scale (`dynamic="local"`). llm-compressor calibrates that scale with 20 samples. The
tensor cores multiply E2M1 directly at twice the FP8 rate.

Turing and Ampere had INT4 tensor cores, but no production serving stack ran LLMs on them. NVFP4 is the first
4-bit format that vLLM runs natively on the tensor cores. Thus it is the first 4-bit format that makes prefill
faster in production serving. Below SM100, vLLM runs NVFP4 checkpoints weight-only (verify). The roofline model
gives these values for Llama-3.1-8B on a B200 (`cost.table()`, SIMULATED, verify Blackwell figures). It gives a
1,800-token prefill of 21 ms in BF16, 12 ms in FP8 and 7 ms in NVFP4.

**W4A4's accuracy risk is the activations.** Sixteen activations share one E4M3 scale, and the largest of them sets
that scale. An outlier channel 30× the typical value sets that scale at 30/6 = 5 typical values per unit of the
E2M1 grid. Then a typical neighbour is at 0.2 on that grid. That is below the 0.25 that rounds up to the smallest
step of E2M1, so the neighbour becomes zero.

The first up-projection of the tiny model has its four outlier channels in three of its four 16-channel blocks.
With NVFP4 activations, its ordinary channels carry 55.4% error, and 48% of them become zero; the block with no
outlier carries 9.7% (`formats.nvfp4()`, notebook 04). The model has 84.5% accuracy with NVFP4 weights only and
80.8% with W4A4 (full precision: 90.3%). The mitigations are the same as for INT8 activations, but used more
strongly:

- **Smoothing** (SmoothQuant, or AWQ-style scales folded into the norm). At $\alpha = 0.5$, the ordinary channels
  drop to 14.5% error, and the model recovers to 84.1%.
- **Hadamard rotations**. They spread an outlier over its block (the SpinQuant and QuIP transforms of
  llm-compressor, §4).
- **Quantization-aware distillation** (§7).

Use W4A4 only after it passes an eval. In the lab's notebook 05, a model that is 100% accurate decreases to 42%
and 51% on its two tasks with NVFP4 W4A4. With SmoothQuant, it gets 99–100% back.

### Layers kept in 16-bit

**Which layers stay in high precision.** The recipes target the linears of the transformer blocks, and they use
`ignore=["lm_head"]`:

- **LM head.** Its errors go directly to the logits, and no later layer averages them. INT4 quantization of only
  the head of the tiny model costs 4.0 points, against 6.0 for all four hidden linears (notebook 04).
- **Embedding.** It is a gather, not a GEMM. Its quantization saves memory but no compute.
- **Norms and the attention softmax.** They are small. Their sensitivity to range is exactly the property that low
  precision handles worst.
- **MoE routers** (`mlp.gate`). A flipped top-k choice is a discrete error.

Attention itself stays in BF16 if the kernel does not quantize it. The FP8 path of FA3 also quantizes Q. "FP8
attention" can mean an FP8 KV cache with BF16 math, or FP8 matrix multiplies. The two have different error budgets
([FlashAttention deep dive §9.4](../flash-attention/flash-attention-deep-dive.md#94-fp8-error-sources-and-mitigations)).

## 6. KV-cache quantization

The KV cache is an activation that the engine stores for later. The engine quantizes it one time when it writes
it, and dequantizes it at every read. At long context and large batch, it is most of what decode streams
([layer 01 §3.4](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#34-kv-reads-cap-decode-intensity)).
When you halve it, you get two gains (`kvquant.kv_bits_per_element()`, `cost.Model.kv_bytes_per_token()`,
`cost.sessions()`).

**Concurrency.** For Llama-3.1-8B,

$$
2 \times \text{layers} \times \text{kv_heads} \times \text{head_dim} \times \text{bytes}
$$

is 131,072 B per token in BF16 and 65,536 in FP8. Sub-8-bit schemes store a scale and a minimum per group. 4-bit
with groups of 32 costs 5 bits per element (40,960 B per token), and 2-bit costs 3 (24,576). Take an L4 with FP8
weights (the core's round memory inputs). There, the 2,000-token sessions go 43 → 87 → 140 → 234. At vLLM's
defaults, `servelab.sizing` gives BF16 weights 2,363 KV blocks, and 4,727 with an FP8 cache (vllm-internals §4.7).

**Faster long-context decode.** Take Llama-3.1-8B with FP8 weights on an H100, at batch 32 and 8,000 tokens of
context. FP8 KV takes a decode step from 17.5 ms to 11.3 ms (`cost.step_cost()`, SIMULATED).

**FP8 KV and its scale.** vLLM's `--kv-cache-dtype` accepts `fp8` (= `fp8_e4m3`) and `fp8_e5m2`, plus
per-token-head dynamic types (`int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head`) and NVFP4 on
SM100 (verify). The static scales `k_scale` and `v_scale` come from the checkpoint and are **1.0 otherwise**.
Per-attention-head scales work only with the FlashAttention backend. On one synthetic decode head
(`kvquant.synthetic_qkv()`: keys with four outlier channels of magnitude ~12, values with a shared mean), the
attention-output error is:

| Cache | Attention-output error (`kvquant.attention_error()`) |
|---|---|
| FP8 K and V, calibrated per-tensor scales | 0.71% |
| … keys only / values only | 0.64% / 0.28% |
| FP8, scale 1.0 (vLLM's default), values of order 1 | 0.81% |
| FP8, scale 1.0, values ×10⁻³ | 4.9% (subnormals and zeros), but calibrated: 0.28% |
| FP8, scale 1.0, values ×10³ | 76% (saturation at 448), but calibrated: 0.28% |

Key errors cost more than value errors, because they go through the exponential of the softmax. The attention
averages the value errors. An uncalibrated scale causes no damage for values of order 1, but it is incorrect far
from 1. The `kv_cache_scheme` of llm-compressor writes calibrated scales. Per-head scales need its per-head recipe.

**Below 8 bits: KIVI.** Keys have outlier channels that stay the same, but values do not. Thus KIVI quantizes **keys per
channel** (a scale and minimum per channel per group of tokens) and **values per token** (per group of channels).
Both are asymmetric. The newest tokens stay 16-bit until a group is full.

On the same head, with groups of 32 (`kvquant.kivi()`), 4-bit keys per channel give 1.16% error against 2.31% per
token, and 2-bit 6.9% against 9.5%. The KIVI README reports 2.6× less peak memory and up to 4× larger batches
(verify).

At this snapshot, vLLM has these sub-8-bit KV types:

- INT4 per token-head with dynamic scales,
- NVFP4 (E4M3 scales per 16, SM100 only),
- the TurboQuant variants (`turboquant_k8v4`, `turboquant_4bit_nc`, …).

None of them is the per-channel-key scheme of KIVI (verify). Before you trust the keys of a system at 4 bits, find
out which scheme it implements.

**Kernel conditions** decide if you can use it at all. They are in
[vllm-internals §6.3](../vllm-internals/vllm-internals-primer.md#63-how-a-backend-is-chosen) and in the checks of
each attention backend (`vllm/v1/attention/backends/`):

- **T4.** No backend has an FP8 KV cache. The FP8 path of Triton needs SM89, FlashInfer needs SM80, and
  FlashAttention needs SM80.
- **A100 and L4.** `--kv-cache-dtype fp8` moves attention from FlashAttention 2 to FlashInfer. Thus a change in
  throughput does not come only from the KV dtype.
- **H100.** FA3 handles FP8 KV, and it also quantizes Q.

**Prefix caching with a quantized cache** works with no change. The engine names blocks by their tokens
([serving-engine §5](../serving-engine/PRIMER.md#5-prefix-caching)), and it uses them again in their stored form.
A later request dequantizes a cached block exactly as the request that wrote it. The reason is that static scales
are per layer, and the engine stores dynamic per-token scales with the block. If the scale of a scheme depends on
the request that reads a block, that scheme cannot share blocks between requests.

## 7. Quantization-aware training and QLoRA in brief

**QAT.** Training can see the quantizer. The forward pass uses the fake-quantized weight ${Q(w)}$ (and
activations). The backward pass treats rounding as the identity inside the clipping range (the **straight-through
estimator**). Thus the weights learn to go where rounding causes the least damage. QAT can recover much of what PTQ
loses at 4 bits and below, at the cost of a training run (measure it per model).

Quantization-aware distillation (QAD) trains the quantized model to match the outputs of the full-precision model,
not labels. The full-precision model is the teacher in the loss of
[distillation §2](../../00-foundations/distillation/PRIMER.md#2-soft-targets-temperature-and-the-choice-of-divergence).
NVIDIA's NVFP4 W4A4 note reports the result of 500 QAD iterations. They recovered an instruction-following
benchmark that lost 2.6 points after PTQ, with a checkpoint that went from 67 to 22 GiB (ModelOpt, 2026-09-16,
verify). This has the same economics as the post-training stages in
[rl-and-thinking-models §1](../../00-foundations/rl-and-thinking-models/PRIMER.md#1-from-pretraining-to-post-training):
a low-cost post-training step on top of a large model.

**QLoRA is a training recipe, not a serving format.** It freezes the base model in **NF4**: 4 bits, 16 levels that
are normal-distribution quantiles, and a scale per block of 64 (QLoRA paper, verify). QLoRA trains LoRA
adapters in BF16 on top. NF4 with an FP32 scale per 64 costs 4.5 bits per weight. With "double quantization"
(8-bit block scales with one FP32 per 256 blocks), it costs 4.127 (`formats.bits_per_weight(4, 64, scale_bits=…)`).

The transformers library loads it with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True)`. To serve the result, merge the adapter into BF16 weights. Then quantize them with
a serving scheme (§9). vLLM serves bitsandbytes checkpoints only through the out-of-tree `vllm-bnb-plugin` at this
snapshot (verify).

## 8. Measuring the accuracy you pay

**Three levels, cheapest first.**

- **Distribution distance** against the unquantized model on the same inputs: the mean
  $\mathrm{KL}(p_{\text{ref}} \parallel p_{\text{quant}})$ per position, and the top-1 agreement (`eval.kl()`,
  `granularity.argmax_agreement()`). It needs no labels. It sees damage that a task score hides in its average.
- **Perplexity** on held-out text (`eval.perplexity()`). It has a low cost, but it does not see failures on
  generation-heavy tasks.
- **Task accuracy** on evals like your traffic. This is what users feel, and it moves in both directions.

On the tiny model, INT4 RTN has KL 0.259, top-1 agreement 87.5% and accuracy 83.7%: it lost 341 right answers and
gained 79. GPTQ has KL 0.048, 95.5% and 89.7%, losing 85 and gaining 61 (`eval.compare()`). The net accuracy makes
the number of changed answers look smaller than it is. Greedy text is the most brittle metric of all. In
serving-engine §8, an INT8 model with 99.8% top-1 agreement diverges from the BF16 greedy text after 13 tokens.

**Error bars.** lm-eval reports $\text{stderr} = \text{sample stddev} / \sqrt{n}$. For a 0/1 metric, this is
(`eval.accuracy_stderr()`):

$$
\text{stderr} = \sqrt{\frac{p \cdot (1 - p)}{n - 1}}
$$

In the FP8 example of vLLM, 250 GSM8K items at 76.8% give ±0.0268.

That is the error bar of one score. A drop is the difference of two scores. In an unpaired comparison, its
standard error is $\sqrt{\mathrm{se}_{\text{ref}}^2 + \mathrm{se}_{\text{quant}}^2}$, $\sqrt{2}$ larger
(`eval.diff_stderr()`). A drop smaller than about two of those is noise. 250 items cannot see a drop smaller than
~7.6 points. To resolve 1 point at 77%, the eval takes 14,169 items per model.

The quantized and reference models answer the **same** items. Thus use a paired comparison. Only the items that
flipped carry information. McNemar's test on them is
$z = (\text{gained} - \text{lost}) / \sqrt{\text{gained} + \text{lost}}$ (`eval.paired_z()`, the lab's notebook 03).

Here is an example on the tiny model. AWQ + GPTQ INT4 drops 0.9 points on 4,000 items: inside the unpaired bar of
±1.4, but 85 lost against 51 gained gives z = −2.9. That is a small loss, but it is real.

**lm-evaluation-harness** (0.4.13, verify) is the standard runner:

```bash
pip install "lm_eval[vllm]"
lm_eval --model vllm --model_args pretrained=$MODEL,add_bos_token=True,gpu_memory_utilization=0.8 \
        --tasks gsm8k --num_fewshot 5 --batch_size auto            # against a running server:
lm_eval --model local-completions --model_args model=$MODEL,base_url=http://HOST:8000/v1/completions --tasks gsm8k
```

When you compare quantized models, pass `add_bos_token=True`. The vLLM docs note that they can be sensitive to it.
`--limit` is "for testing only". The default for GSM8K is 5-shot `exact_match`. MMLU is 57 subtasks with the
score `acc`.

**Failure modes to test for explicitly:**

- **Small models.** Fewer parameters share the error. The tiny model loses 6.6 points at INT4 RTN.
- **MoE experts.** Rarely routed experts see only a small quantity of calibration data. The
  `moe_calibrate_all_experts` option of llm-compressor is on by default, and it sends every calibration token
  through every expert. Routers must stay 16-bit
  ([mixture-of-experts §6.7](../../00-foundations/mixture-of-experts/PRIMER.md#67-quantized-experts)).
- **Long context.** KV errors accumulate over thousands of positions, and outlier tokens set per-tensor scales.
- **Long generations and thinking models.** A flipped near-tie changes everything after it. A reasoning trace of
  thousands of tokens gives it thousands of chances. Evaluate with the full generation length.
- **Multilingual inputs.** Calibration in one language does not fully represent the activation statistics of
  other languages.
- **Tool calling and structured output.** An argument that is incorrect by one token is incorrect. Do an
  exact-match test on your own tool schemas.

**Setting a budget.** Write the budget down before you measure. Use the form "$\mathrm{KL} \le x$, an accuracy drop
of at most $y$ points, and within two standard errors of the difference on our eval" (`eval.within_budget()`). Also
say if you use a paired test. On the tiny model, with $\mathrm{KL} \le 0.05$, INT8 RTN and INT4 GPTQ pass, and INT4
RTN fails (notebook 05). The recipe is part of the scheme.

## 9. Producing a checkpoint

**llm-compressor** (0.14.0, verify) applies a *recipe* of modifiers with `oneshot()`. It saves a
**compressed-tensors** checkpoint that vLLM loads without flags. The example recipes all use `targets="Linear"` and
`ignore=["lm_head"]`:

| Scheme | Recipe | Calibration |
|---|---|---|
| FP8 dynamic (W8A8, per-channel weights, per-token activations) | `QuantizationModifier(scheme="FP8_DYNAMIC")` | **none** |
| FP8 block (DeepSeek-V3 style) | `QuantizationModifier(scheme="FP8_BLOCK")` | none |
| W4A16 GPTQ (INT4 g128, symmetric) | `GPTQModifier(scheme="W4A16")` | 512 × 2,048 tokens |
| W4A16 AWQ (asymmetric) | `AWQModifier(duo_scaling="both")` + `QuantizationModifier(scheme="W4A16_ASYM")` | 256 × 512 |
| W8A8 INT8 | `SmoothQuantModifier(smoothing_strength=0.8)` + `GPTQModifier(scheme="W8A8")` | 512 × 2,048 |
| NVFP4 (W4A4) | `QuantizationModifier(scheme="NVFP4")`. If the activations have outlier channels, add smoothing or a rotation transform (§5). | 20 samples (global activation scales) |
| FP8 KV cache | `kv_cache_scheme: {num_bits: 8, type: float, strategy: tensor, dynamic: false}` | 512 × 2,048 |

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]))
model.save_pretrained("Qwen2.5-0.5B-Instruct-FP8-dynamic", save_compressed=True)
```

MoE models must also ignore their routers (`"re:.*mlp.gate$"`). Install llm-compressor and vLLM in **separate
environments** (vLLM docs). The lab's notebook 01 runs these recipes at T1 on a 0.5B model. At T0, it writes the
same layout from its own code.

**The format.** `config.json` carries a `quantization_config`:

- `quant_method: "compressed-tensors"`, plus `format`.
- `config_groups`, each with `targets`, `weights` and `input_activations`, which are `QuantizationArgs`:
  `num_bits`, `type`, `symmetric`, `strategy`, `group_size`, `block_structure`, `dynamic`, `actorder`.
- `ignore`, `kv_cache_scheme`, and `quantization_status: "compressed"`.

For each linear, the checkpoint stores a pack-quantized INT4 weight as `weight_packed` (int32), `weight_scale`
(`(out, in/g)`), `weight_shape`, and `weight_zero_point` if the scheme is asymmetric. An NVFP4 weight adds
`weight_global_scale`, and static activations add `input_scale`. Attention layers carry `k_scale` and `v_scale`.
The checkpoint packs the codes in this layout (`formats.pack_int4()`, `formats.pack_fp4()`):

```
INT4: codes −8…7 + 8, eight per int32, element 0 in the lowest 4 bits: [−8,−7,0,1,2,3,4,7] → 0xfcba9810
      a [4096, 896] INT4 weight → weight_packed [4096, 112]
FP4:  4-bit code = index in {0, .5, 1, 1.5, 2, 3, 4, 6} | sign << 3, two per byte, first in the low nibble
      [0.5, −6, 1.5, 0] → 0xf1, 0x03
```

**Other producers.**

- **GPTQModel** (7.5.0): `GPTQModel.load(id, QuantizeConfig(bits=4, group_size=128))`, then `.quantize(data)` and
  `.save()`. It supports Turing and newer.
- **AutoAWQ**: deprecated. llm-compressor took over its role.
- **NVIDIA ModelOpt**: `mtq.FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG`, `INT8_SMOOTHQUANT_CFG`. vLLM serves its exports
  with `quantization="modelopt"` or `"modelopt_fp4"`.

Pre-quantized checkpoints are on the Hub, under the names of the model vendors and RedHatAI/nm-testing (verify
each id before you rely on it).

**Loading in vLLM** ([vllm-internals §8.1–8.3](../vllm-internals/vllm-internals-primer.md#8-quantization-and-weight-loading)):

- **Detection.** vLLM reads `quantization_config.quant_method` and selects the method. A `--quantization` that
  does not agree with the checkpoint raises an error. Thus, for pre-quantized models, do not use the flag.
- **Online FP8.** To quantize a BF16 checkpoint at load time, use `--quantization fp8_per_tensor`. Plain
  `--quantization fp8` still does it at v0.30.0, but it raises an error at `main` (vLLM's `Fp8Config`). The serving
  lab's notebook 05 uses the old form.
- **Minimum capability.** The load fails when the GPU is below the minimum of the method ("Minimum capability: …").

| Kernel / path | Minimum | T4 (7.5) | A100 (8.0) | L4, 4090 (8.9) | H100 (9.0) | B200 (10.0) |
|---|---|---|---|---|---|---|
| W4A16 Marlin (GPTQ, AWQ, compressed-tensors) | SM75 | yes | yes | yes | Machete | yes (Machete is SM90-only) |
| FP8 weight-only (Marlin FP8) | SM75 | yes | yes | native FP8 instead | native FP8 instead | native FP8 instead |
| FP8 W8A8 (CUTLASS, SM89 needs CUDA ≥ 12.4) | SM89 | runs W8A16 | runs W8A16 | yes (no block-FP8 CUTLASS) | yes + block, DeepGEMM | yes |
| INT8 W8A8 (CUTLASS) | SM75, < SM100 | yes | yes | yes | yes | **no** |
| NVFP4 W4A4 (CUTLASS/FlashInfer, CUDA ≥ 12.8) | SM100 | W4A16 | W4A16 | W4A16 | W4A16 | yes |
| FP8 KV cache | SM80 backends | **no** | FlashInfer | FlashInfer | FA3 | yes |

`quantcore.cost.supported()` encodes this table. It covers vLLM 0.30.0 and `main`, from the `get_min_capability` of
each kernel and the scheme dispatch in `compressed_tensors.py` (verify on your version). The docs table of vLLM
does not agree with the code on the INT4 floor. Thus trust the code and the log line. A T4 needs `--dtype half`.

**The CPU and consumer path: llama.cpp GGUF.** GGUF has block formats with an fp16 scale per 32 (`q4_0` 18 B per
32 = 4.5 bits, `q8_0` 8.5, which `formats.bits_per_weight(4, 32)` and `(8, 32)` calculate). It also has "k-quants"
with 256-value super-blocks. Llama-3.1-8B at `Q4_K_M` is 4.89 bits per weight, 4.58 GiB (llama.cpp README,
verify).

```bash
./build/bin/llama-quantize in-bf16.gguf out-Q4_K_M.gguf Q4_K_M
```

vLLM reads GGUF only through the out-of-tree `vllm-gguf-plugin`, which is "highly experimental" (verify).

## 10. Choosing a scheme

`cost.table()` calculates the cost of every scheme for one GPU and model. It gives what each scheme runs as, the
weights, decode at batch 1 and 32, a 1,800-token prefill and the 2,000-token sessions. All of these values are
SIMULATED with the assumptions of serving-engine §8 (80% bandwidth, 60% FLOPs, 2 ms per step). With the L4 of
`minengine.perf`, it gives the table of that section again, to the digit.

`cost.choose()` returns the least aggressive runnable scheme that meets every target. It uses an initial order by
typical accuracy cost: BF16, FP8 weight-only, FP8 W8A8, INT8 W8A8, INT4 W4A16, NVFP4. Measure the schemes for your
model. Then change the order for your model. The decision:

| GPU generation | Decode-heavy, memory-bound | Prefill-heavy | Concurrency-bound | Notes |
|---|---|---|---|---|
| **Turing** (T4, 16 GB, fp16 only) | W4A16 (Marlin) | INT8 W8A8 + SmoothQuant | W4A16 (no FP8 KV) | an 8B model fits only at 8 or 4 bits |
| **Ampere** (A100) | W4A16 or FP8 weight-only | INT8 W8A8 | + FP8 KV (FlashInfer) | FP8 checkpoints run W8A16 |
| **Ada** (L4, RTX 4090) | W4A16 or FP8 | **FP8 W8A8** | + FP8 KV | CUTLASS FP8 needs CUDA ≥ 12.4. There is no block-FP8 CUTLASS. |
| **Hopper** (H100, H200) | W4A16 (Machete) or FP8 | FP8 W8A8 (block or per-channel) | + FP8 KV (FA3) | the FP8 default |
| **Blackwell** (B200, RTX PRO 6000) | NVFP4 | **NVFP4 W4A4** (with smoothing or rotation, only after it passes an eval), FP8 | + FP8 KV (NVFP4 KV on B200/SM100 only) | There is no INT8 W8A8. W4A16 is Marlin, not Machete. |

Notebook 05 works through these examples with `cost.table()` and `cost.choose()`:

- **L4, Llama-3.1-8B, ≥ 48 sessions and a 1,800-token prefill ≤ 300 ms.** FP8 W8A8 with FP8 KV: 87 sessions,
  181 ms. This is the answer to serving-engine drill 6, and now `choose` calculates it.
- **Free T4, Llama-3.1-8B, ≥ 20 sessions and a prefill ≤ 700 ms.** W4A16: 16-bit weights do not fit at all. 8-bit
  weights leave room for 16 sessions, INT4 for 29.
- **One H100, Llama-3.1-70B.** BF16 (141.1 GB) does not fit. FP8 (72.7 GB) fits, but with no useful concurrency.
  With the core's round memory inputs (0.9 × 80 GB − 1 GB, `cost.kv_blocks()`), it leaves no room for a single 4K
  session.

  An H100 reports 79.65 GiB. At vLLM's defaults on that memory, the lab's `kv.size()` leaves room for 2 such
  sessions with a BF16 KV cache, or 4 with FP8. INT4 (39.5 GB) serves 48 of them with FP8 KV (54 in the lab's
  model). FP8 means two GPUs with tensor parallelism, or an H200.
- **B200, Llama-3.1-8B.** NVFP4 W4A4 is the only 4-bit option that decreases the prefill FLOPs (verify). It needs
  the activation mitigations of §5.

**Cost per token** ([layer 01 §8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens)):
`$/M = $/GPU-hr ÷ (tokens/s × 3600 × utilisation) × 10⁶` (`cost.cost_per_million()`). Take an L4 at ~$0.70/hr
(verify, [`COMPUTE.md`](../../COMPUTE.md)), with every session that it can hold in decode. BF16 fits 17 sessions,
230 tokens/s, **$0.844/M** at full utilisation. FP8 weights plus FP8 KV fit 87 sessions, 1,469 tokens/s,
**$0.132/M** (SIMULATED). That is 6.4× cheaper, not 2×. The reasons are fewer bytes per step, and a 5× larger batch
that shares each step.

This agrees with the knob table of serving-engine §8. `kv_cache_dtype fp8` gives 2× KV capacity and faster
long-context decode. `quantization` gives memory and decode speed. It gives prefill speed only for FP8 (or INT8,
FP4) W8A8. Both cost accuracy, and both depend on the kernels that are available for each GPU.

**Where to run it**, concept by concept, on GCP and elsewhere (prices and obtainability are in [`COMPUTE.md`](../../COMPUTE.md)):

| To learn | T0 (laptop / Colab CPU) | Non-GCP GPU (T1) | GCP (T3) |
|---|---|---|---|
| §2–§6 mechanics, §10 decisions | `quant-core` notebooks 01–05 | — | — |
| checkpoint production (§9) | the lab's numpy/torch path on a bundled small model | llm-compressor on any 16–24 GB GPU. A 0.5B model fits a free T4 (verify). | a `g2-standard-4` L4 VM (Spot) |
| INT4 serving, INT8 W8A8 | the lab's fake server (simulated) | Colab/Kaggle T4 (free, `--dtype half`, no FP8 compute or FP8 KV) | L4 via the serving lab's Cloud Run or GKE deploy |
| FP8 W8A8 and FP8 KV | emulation in `quantcore` | RTX 4090 on RunPod/Vast (~$0.3–0.4/hr, verify), L4 | L4 (G2, Cloud Run), H100 (A3) |
| FP4 / NVFP4 | `formats.nvfp4()`, `cost.table()` | a rented B200 or RTX PRO 6000 (verify) | A4 (B200), G4 / Cloud Run (RTX PRO 6000, verify) |

The lab's `deploy/any-gpu/` has `docker run` recipes for each GPU generation and scheme. For GCP, it points to
[`serving-engine/vllm-serving-lab/deploy/gcp/`](../serving-engine/vllm-serving-lab/deploy/gcp/) with a quantized
model. There is no new Terraform.

---

## In a design review

**The two-minute walkthrough.** "We quantize for three different reasons, and we select the scheme for each reason.
Decode is a weight read, so fewer weight bytes give faster tokens: weight-only INT4 is ~3× at batch 1 for an 8B
model. But its kernels do BF16 math on dequantized weights. On an L4, the GEMM gets to that ceiling at ~120 tokens
per step, and from there its gain decreases (1.7× at 256). At ~460 it is no faster than BF16, so long prefill
chunks gain nothing.

"Prefill becomes faster only with formats that the tensor cores multiply natively. These are FP8 W8A8 on Ada and
Hopper, INT8 W8A8 on older parts, and NVFP4 on Blackwell. The KV cache is the third lever: FP8 KV halves it, and on
a 24 GB card that doubles the sessions. For the cost per token, this does more than speed does. The cost is 6× lower
for an 8B model on an L4 with FP8 weights and KV, simulated. We examined what each checkpoint runs as on our GPUs:
an FP8 checkpoint on an A100 is weight-only, and NVFP4 is W4A4 only on Blackwell.

"Accuracy comes from granularity and calibration. We use groups of 128 for INT4 with GPTQ or AWQ, dynamic
per-token FP8 activations and calibrated KV scales. We keep the LM head, embeddings, norms and routers in 16-bit.
Our gate is KL against the BF16 model and task evals with their standard errors. We compare both against a budget
that we wrote down first."

**Drill questions.**

1. *Why does INT4 weight-only speed up decode ~3× but not prefill at all?* Decode streams the weights one time per
   step. Thus the bytes are its limit, and INT4 makes them 3.9× smaller (less the 16-bit LM head, KV and overhead).

   Prefill is compute-bound. W4A16 kernels dequantize to BF16 before the MMA, so the FLOPs are the same. On an L4,
   the down_proj GEMM becomes compute-bound at ~120 tokens per step in W4A16, and at ~460 in BF16. Between the
   two, the gain of INT4 decreases from 3.9× to nothing.
2. *Our FP8 checkpoint runs on A100s and the H100 benchmark does not transfer. Why?* A100s have no FP8 tensor
   cores. Thus vLLM runs the checkpoint as weight-only FP8 through Marlin.

   The memory and decode-byte savings stay, but prefill runs at the BF16 rate. INT8 W8A8 with SmoothQuant is the
   prefill lever on Ampere. Also, FP8 KV on an A100 moves attention to FlashInfer.
3. *INT4 RTN lost 5 points on our model. What next, before we give up on 4 bits?* Calibrate. Use GPTQ, act-order
   and groups of 128 (or 64/32 if the kernel permits them). For layers whose inputs have outlier channels, use AWQ
   first. On our toy model, GPTQ took INT4 from −6.6 to −0.6 points.

   Then make sure that the recipe excludes the LM head and the routers. Also make sure that the calibration data
   looks like the traffic.
4. *The per-token INT8 activation error is 1.4%. Are we fine?* Look per channel. With a few outlier channels 30×
   larger, the ordinary channels carry ~11% error, but the aggregate hides it. Use SmoothQuant ($\alpha$ 0.5–0.85)
   or FP8. The relative grid of FP8 keeps them at ~3%.
5. *Is FP8 KV safe to turn on?* It is usually safe, if the scales fit. It halves the KV bytes (2× sessions) at <1%
   attention error on well-scaled heads. With the default scale 1.0, values far below 1 flush to subnormals (5%
   error in our example). Values above 448 saturate.

   Calibrate `k_scale`/`v_scale`. Examine the backend (none on a T4). Then run a long-context eval.
6. *We need the 70B model on one H100. What are the options?* BF16 (141 GB) does not fit. FP8 (72.7 GB) fits, but
   it leaves room for only 2–4 sessions of 4K (none at the core's round inputs). That is no useful concurrency.

   INT4 W4A16 (39.5 GB) fits with ~50 such sessions and FP8 KV. It costs accuracy, and you must measure how much.
   Prefill also runs at BF16 speed. The alternatives are two H100s with tensor parallelism in FP8, or an H200
   (141 GB).

---

## Glossary

| Term | Meaning |
|---|---|
| **Scale / zero point** | $x \approx (\text{code} - \text{zero}) \times \text{scale}$. Symmetric formats have no zero point. |
| **Granularity** | how many values share one scale: tensor, channel, group, token, block |
| **bpw** | bits per weight, which includes scales and zero points (INT4 g128: 4.125 or 4.156) |
| **E4M3 / E5M2** | FP8 with 4 exponent + 3 mantissa bits (max 448) or 5 + 2 (max 57,344) |
| **E2M1** | FP4: magnitudes \{0, 0.5, 1, 1.5, 2, 3, 4, 6\} |
| **E8M0** | an 8-bit power-of-two scale (MX formats) |
| **MXFP4 / NVFP4** | E2M1 with an E8M0 scale per 32 / an E4M3 scale per 16 plus an FP32 per-tensor scale |
| **W4A16, W8A8, W4A4** | weight bits / activation bits. "A16" means that activations stay 16-bit (weight-only). |
| **RTN** | round to nearest, no calibration |
| **GPTQ** | column-by-column rounding with inverse-Hessian error compensation (Optimal Brain Surgeon) |
| **AWQ** | activation-aware scales on salient weight columns before rounding, folded into the previous layer |
| **SmoothQuant** | moves activation outliers into the weights with per-channel scales $s = \max \lvert X \rvert^{\alpha} / \max \lvert W \rvert^{1-\alpha}$ |
| **Act-order** | GPTQ visits the columns from the largest Hessian diagonal to the smallest |
| **Calibration data** | sample inputs used to calculate H, activation statistics or static scales |
| **Dynamic / static scales** | calculated per token at run time / set from calibration |
| **Epilogue** | the GEMM's final step that applies scales (and bias, activation) to the accumulator |
| **Dequantize-on-the-fly** | weight-only kernels (Marlin, Machete) expand 4-bit weights to 16-bit in registers |
| **k_scale / v_scale** | per-layer FP8 KV scales in a checkpoint. They are 1.0 if absent. |
| **KIVI** | sub-8-bit KV: keys per channel, values per token, a full-precision residual window |
| **Crest factor** | amax / rms: every 10× costs 20 dB of SQNR |
| **SQNR** | signal-to-quantization-noise ratio, ~6 dB per bit |
| **STE** | straight-through estimator: treat rounding as identity in the backward pass (QAT) |
| **NF4** | QLoRA's 4-bit normal-quantile format, a training format |
| **compressed-tensors** | the checkpoint format that llm-compressor writes and vLLM reads |
| **KL / top-1 agreement** | distribution distance and argmax match between the quantized and reference models |

## Sources

Papers:

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023.
  Code: IST-DASLab/gptq (`gptq.py: fasterquant`, `quant.py`).
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024. Code:
  mit-han-lab/llm-awq (`awq/quantize/auto_scale.py`, `auto_clip.py`).
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML
  2023. Code: mit-han-lab/smoothquant (`smoothquant/smooth.py`).
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024. Code: jy-yuan/KIVI.
- Dettmers et al., *LLM.int8()* (2022) and *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).
- Micikevicius et al., *FP8 Formats for Deep Learning* (2022).
- Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0* (2023).
- Ashkboos et al., *QuaRot* (2024), and Liu et al., *SpinQuant* (2024).
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024), and deepseek-ai/DeepSeek-V3 `README_WEIGHTS.md`,
  `inference/kernel.py`.
- Frantar et al., *Marlin* (IST-DASLab/marlin README).

Code and docs, read on 2026-09-26 (vLLM v0.30.0 and `main@a4eb3f25`, llm-compressor `c6fb66c`,
compressed-tensors `47f7d42`, lm-eval `d6de816`, GPTQModel `3b2e435`):

- vllm-project/vllm: `vllm/model_executor/layers/quantization/`, `kernels/linear/`, `config/model.py`,
  `config/cache.py`, `v1/attention/backends/`, `docs/features/quantization/`, at v0.30.0 and `main`.
- vllm-project/llm-compressor: `examples/`, `modifiers/`.
- neuralmagic/compressed-tensors: `quant_scheme.py`, `quant_args.py`, `compressors/`.
- ModelCloud/GPTQModel, and the NVIDIA/TensorRT-Model-Optimizer (ModelOpt) docs.
- EleutherAI/lm-evaluation-harness (`lm_eval/api/metrics.py: mean_stderr`).
- ggml-org/llama.cpp `ggml-common.h` and the quantize README, and openai/gpt-oss (`weights.py`, MXFP4).

In this repo:

- [serving-engine PRIMER §8](../serving-engine/PRIMER.md), `minengine.quant` and `servelab.sizing`.
- [vllm-internals §6.3, §8](../vllm-internals/vllm-internals-primer.md) and the
  [FlashAttention deep dive §9](../flash-attention/flash-attention-deep-dive.md).
- [layer 01 PRIMER §1–3, §8](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) and
  [gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md).
- [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) (bytes per parameter).

## Verify list

Dated 2026-09-26. Each item comes from the sources in the Sources section, and it can change.

- vLLM **0.30.0** (PyPI) and `main@a4eb3f25`: the `QuantizationMethods` list. `--quantization fp8` quantizes a BF16
  checkpoint online at 0.30.0, but it raises an error at `main` (use `fp8_per_tensor`). In vLLM, bitsandbytes and GGUF
  are out-of-tree plugins (`vllm-bnb-plugin` 0.0.3, `vllm-gguf-plugin` 0.0.5).
- vLLM minimum capabilities:
  - Marlin SM75,
  - Machete SM90 only,
  - CUTLASS FP8 SM89 (CUDA ≥ 12.4) and SM90+,
  - CUTLASS block FP8 SM90+ (none on SM89),
  - NVFP4 W4A4 SM100–129 with CUDA ≥ 12.8,
  - INT8 W8A8 not on compute capability ≥ 10.0,
  - MXFP4 minimum SM80,
  - FP8 KV unavailable on SM75.
- The vLLM KV-cache dtypes (`CacheDType`), among them `fp8`, `fp8_e5m2`, `*_per_token_head`, `nvfp4` (SM100 family
  only) and `turboquant_*`. `k_scale`/`v_scale` default to 1.0. Per-head scales work only with FlashAttention. FP8
  KV on L4/A100 selects FlashInfer, and on H100 FA3.
- llm-compressor **0.14.0**, compressed-tensors **0.19.0**, lm-eval **0.4.13**, GPTQModel **7.5.0**, ModelOpt
  **0.47.0**, AutoAWQ 0.2.9 (deprecated): the recipe names, the defaults and the calibration sizes in the examples.
  The defaults are GPTQ `dampening_frac` 0.01 and `block_size` 128, AWQ `n_grid` 20, and SmoothQuant 0.5.
- The tensor names, scale shapes and INT4/FP4 pack order of compressed-tensors. The NVFP4 global scale is a
  multiplier (448 × 6 / amax). The MXFP4 scale exponent rounds up at a mantissa ≥ 1.75 (`round_to_power_2`).
  llm-compressor takes its GPTQ group scales from the weight observer before the loop.
- DeepSeek-V3 report §3.3: the FP8 MMA accumulation of Hopper has about 14 bits, with promotion to FP32 every 128
  elements.
- Reported accuracy figures, quoted from sources:
  - the LLaMA perplexities in the GPTQ README,
  - vLLM's FP8 GSM8K example (0.768 ± 0.0268 on 250 items),
  - the SmoothQuant $\alpha$ adjusted per model,
  - KIVI's memory and batch claims,
  - ModelOpt's NVFP4 QAD note (2026-09-16).
- GPU figures (dense TFLOP/s, bandwidth, memory) from `roofline.specs` and `quantcore.cost.GPUS`: T4, A100, L4,
  H100, B200, RTX PRO 6000. The RTX 4090 figures come from `servelab.GPUS`, without verification.
- Model configs: Llama-3.1-8B/70B, Qwen2.5-0.5B/1.5B (parameters, layers, heads, vocab, tied embeddings).
- Prices: L4 ~$0.70/hr and H100 ~$11/GPU-hr on demand in us-central1, and RTX 4090 ~$0.3–0.4/hr on Vast/RunPod.
- llama.cpp k-quant sizes (Q4_K_M 4.89 bpw on Llama-3.1-8B), and the Hub ids of pre-quantized checkpoints.
