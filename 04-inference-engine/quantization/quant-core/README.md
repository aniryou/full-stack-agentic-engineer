# quant-core — quantize a model from first principles and predict what it buys

After this you can pick a quantization scheme for a model and a GPU and defend it with numbers. You will know what
each format's grid is, which outliers each granularity survives, and what GPTQ, AWQ and SmoothQuant do to the
codes. You will know what an FP8 or 4-bit KV cache costs in accuracy and buys in sessions, and what a checkpoint
runs as on each GPU generation. You get there by filling in the code yourself in `quantcore`, a numpy package
small enough to read in a sitting (~640 lines of code, ~1,130 with docstrings).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md): "The one-minute version", then §1 *Why quantize* and §2 *Number formats*.
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 71 tests in about 5 s. They include
   "GPTQ equals the reference implementation", "FP8 rounding is bit-identical to the published grid", and
   "every number the primer computes is still what the code computes".
3. Open [`notebooks/01_number_formats_and_error.ipynb`](notebooks/01_number_formats_and_error.ipynb) and round a
   number to E4M3, E5M2 and E2M1 with one function.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network, in seconds.* Each notebook
opens with "The one-minute version", works examples against the code, and sets exercises; each exercise is
followed by a check cell that prints ✅. Each notebook ends with "In a design review". Finished versions are in
[`solutions/`](solutions/). It takes about 10 hours in all with the primer (module 04.9 in the repo's
curriculum).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_number_formats_and_error`](notebooks/01_number_formats_and_error.ipynb) | enumerate INT, FP8 (E4M3/E5M2) and FP4 grids from their bits and round onto them; compare INT4, FP4, MXFP4 and NVFP4 on Gaussian and heavy-tailed weights; predict SQNR from bits and crest factor (~6 dB per bit, 20 dB per 10× outlier); count bits per weight and whole-model GB; pack INT4 and FP4 codes as a checkpoint does | §2, §3, §9 | ~1.5 h | T0 |
| [`02_granularity_and_outliers`](notebooks/02_granularity_and_outliers.ipynb) | reproduce serving-engine §8's six-scheme table; show why per-channel scales fix an outlier row but not an outlier input column; measure what per-token INT8 does to activations with outlier channels (and what FP8 does instead); choose a static activation scale on calibration data; use 128×128 block scales | §2, §3 | ~1.5 h | T0 |
| [`03_gptq_awq_and_smoothquant_from_scratch`](notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) | implement GPTQ's column loop and AWQ's scale search; fold SmoothQuant and AWQ scales into a model without changing it; say which layers each method helps and why; measure how much calibration data is enough | §4, §5 | ~2 h | T0 |
| [`04_activation_and_kv_cache_quantization`](notebooks/04_activation_and_kv_cache_quantization.ipynb) | implement a W8A8 GEMM epilogue; compare static and dynamic activation scales; show why the LM head stays 16-bit; judge FP8 KV with and without calibrated scales; implement KIVI's per-channel keys; size the cache | §5, §6 | ~1.5 h | T0 |
| [`05_choosing_a_scheme`](notebooks/05_choosing_a_scheme.ipynb) | reproduce serving-engine §8's L4 table and vllm-internals §8.1's GEMM table; find where W4A16 stops paying; say what a checkpoint runs as on T4 to B200; choose schemes for a T4, an L4 and an H100 serving a 70B; price a million tokens (simulated) | §1, §10 | ~1.5 h | T0 |

## Run it

```bash
cd quant-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 71 tests, ~5 s
python3 -m jupyterlab notebooks                # do the exercises
```

The library itself needs only numpy:

```python
from quantcore import TinyModel, quantize_model, eval, cost

m = TinyModel()                                               # a small model with LLM-like activation outliers
X, y = m.sample(4000, "test"); Xc, _ = m.sample(256, "calib")
for method in ("rtn", "gptq", "awq"):
    q = quantize_model(m, method, bits=4, group_size=32, calib=Xc)
    r = eval.compare(m.forward(X), q.forward(X), y)
    print(f"{method:5} acc {r['acc']:.1%}  KL {r['kl']:.3f}")  # rtn 84.8%  gptq 89.8%  awq 85.0%  (fp 90.3%)
print(cost.supported(cost.GPUS["A100-80GB"], "w8a8-fp8"))     # an FP8 checkpoint on an A100 runs weight-only
```

Every notebook's first code cell pins BLAS to one thread (`OPENBLAS_NUM_THREADS=1`), and so does
`tests/conftest.py`. The matrices here are small, and on a busy shared machine a multi-threaded OpenBLAS can make
one 256×256 inverse a hundred times slower.

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`quantcore/formats.py`](quantcore/formats.py) | ~195 | a format is a grid plus a scale: INT conventions (restricted, full, asymmetric), FP8 E4M3/E5M2 and FP4 E2M1 grids from their bit patterns, rounding (bit-identical to torch's FP8 casts), MXFP4 and NVFP4 block formats, bits per weight, the SQNR rule, INT4/FP4 packing as compressed-tensors lays it out |
| [`quantcore/granularity.py`](quantcore/granularity.py) | ~130 | who shares a scale: tensor, channel, group, token, 2-D block, with scales in the checkpoint's shapes; static and dynamic activation scales; error metrics that do and do not hide per-channel damage |
| [`quantcore/gptq.py`](quantcore/gptq.py) | ~85 | the Hessian from calibration inputs, damping, the Optimal Brain Surgeon column update, act-order with static groups, RTN |
| [`quantcore/awq.py`](quantcore/awq.py) | ~55 | activation-aware scaling: the α grid search and why α = 0 is RTN |
| [`quantcore/smoothquant.py`](quantcore/smoothquant.py) | ~45 | migrating activation outliers into weights for W8A8, and the α sweep |
| [`quantcore/w8a8.py`](quantcore/w8a8.py) | ~60 | the W8A8 GEMM with integer accumulation and the scale epilogue; static-scale saturation; DeepSeek-V3-style block FP8 |
| [`quantcore/kvquant.py`](quantcore/kvquant.py) | ~85 | FP8 KV with default, calibrated and per-token scales; KIVI (keys per channel, values per token, residual window); attention-output error |
| [`quantcore/tinymodel.py`](quantcore/tinymodel.py) | ~140 | a residual MLP stack whose norm gains create LLM-like outlier channels, a closed-form head, calibration capture, exact scale folding, and `quantize_model` (RTN, GPTQ, AWQ, AWQ then GPTQ) |
| [`quantcore/eval.py`](quantcore/eval.py) | ~55 | KL, top-1 agreement, perplexity, task accuracy with flips both ways, lm-eval's standard error, the unpaired difference's error bar and McNemar's paired z, a budget check |
| [`quantcore/cost.py`](quantcore/cost.py) | ~210 | what a scheme runs as per GPU generation (vLLM's rules), one GEMM on the roofline and the W4A16 crossover, the step-time model of `minengine.perf`, KV blocks and sessions, the decision table, $/M tokens |

## What the tests prove

`tests/` has one focused test per concept: 71 tests, offline, about 5 s. The ones that carry the correctness
claims:

- **It reproduces the repo's published numbers with its own code** (`test_repo_numbers.py`). These are
  serving-engine PRIMER §8's six-scheme error table, FP8 grid statistics, outlier-channel and SmoothQuant numbers
  (replaying `mini-engine-core` notebook 06's seed), and the SIMULATED Llama-3.1-8B-on-L4 table to the digit. They
  also include vllm-internals §8.1's `down_proj` table (392/102/196 µs …) with the W4A16 crossover at 120 tokens on
  an L4 and 85 on an H100, and servelab's Qwen2.5-0.5B weight bytes. If `mini-engine-core` is on disk, the test
  also checks `fake_quant` against `minengine.quant` directly.
- **The formats are the real formats.** The E4M3 grid enumerated from bit patterns has 253 distinct values and a
  top of 448. Rounding matches the nearest grid value everywhere and vLLM's FP4 tie thresholds exactly. INT4
  packing gives compressed-tensors' `0xfcba9810` and round-trips (`test_formats.py`).
- **GPTQ is GPTQ.** It matches a line-by-line transcription of the reference's blocked "lazy batch" loop to 1e-9
  per channel and whenever groups start on block boundaries (a group that starts mid-block takes its scale from
  weights the reference has not updated yet; the test shows that difference too), reduces to RTN exactly when
  inputs are uncorrelated, and beats RTN out of sample on correlated inputs (`test_gptq.py`).
- **Reparameterisations are exact.** SmoothQuant and AWQ folds leave the model's output unchanged to 1e-9, and
  α = 0 in AWQ's search is RTN (`test_awq_smoothquant.py`).
- **The W8A8 epilogue is exact.** Integer-accumulated INT8 GEMMs and per-block FP8 GEMMs equal the fake-quantized
  products (`test_w8a8_kv.py`).
- **The primer says what the code computes.** Every number `../PRIMER.md` attributes to `quantcore` is recomputed
  and must appear in it verbatim (`test_primer_numbers.py`). The numbers do not depend on the numpy or BLAS build:
  INT4's full convention puts −amax exactly on the tie −7.5, which floating point can land an ulp either side of,
  so `formats.quantize_int` snaps near-ties onto it first. The suite gives the same digits on numpy 1.26, 2.0 and
  2.4.

## Caveats: what is faithful, and what is simplified

Faithful to the sources (read at the commits listed in the primer's Sources; see its Verify list):

- GPTQ's update, damping and act-order (group scales: see `gptq.py`'s docstring).
- AWQ's statistic, normalisation and 20-point grid, and SmoothQuant's scale formula.
- compressed-tensors' INT conventions, NVFP4 global scale and INT4/FP4 packing, and both MXFP4 shared-exponent
  rules (the OCP spec's floor and compressed-tensors' round-up at a mantissa of 1.75).
- KIVI's key/value asymmetry, lm-eval's standard error, and vLLM's capability rules.

Simplified:

- Everything runs in float64 "fake quantization", so errors are visible. No kernel is emulated beyond its
  arithmetic.
- The tiny model is an MLP stack on a synthetic task, not a transformer. Its down-projections are more redundant
  than real ones, which flatters GPTQ.
- AWQ's clipping search is described in the primer but not implemented.
- `cost.supported` is a table of rules, not a probe of your GPU.

Every latency and throughput `quantcore.cost` prints is **simulated** (a roofline model with 80% of bandwidth,
60% of peak FLOP/s and 2 ms per step, as in `minengine.perf`) and labelled so. GPU figures are dense datasheet
values `(verify)`.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs this
package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../quant-lab/`](../quant-lab/) to produce real checkpoints with llm-compressor, serve FP16, INT4 and FP8
side by side in vLLM, measure accuracy with lm-eval, and turn on an FP8 KV cache on a GPU. It has a T0 fake server
for every notebook. For how vLLM picks a quantization method and kernel in its source, read
[`../../vllm-internals/`](../../vllm-internals/README.md) §8. For the roofline and cost-per-token arithmetic
underneath, see layer 01's
[`roofline-and-fabric`](../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). Where each tier runs and what
it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
