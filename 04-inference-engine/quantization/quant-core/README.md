# quant-core — quantize a model from first principles and predict what it buys

After this, you can select a quantization scheme for a model and a GPU, and you can defend it with numbers. You
will know the grid of each format and which outliers each granularity survives. You will also know what GPTQ, AWQ
and SmoothQuant do to the codes. You will know what an FP8 or 4-bit KV cache costs in accuracy and gives in
sessions. You will also know what a checkpoint runs as on each GPU generation.

To get there, you fill in the code yourself in `quantcore`. This numpy package is small, and you can read all of it at
one time (~640 lines of code, ~1,130 with docstrings).

## Start here

1. Read "The one-minute version" in [`../PRIMER.md`](../PRIMER.md). Then read §1 *Why quantize* and §2 *Number
   formats*.
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 86 tests run in about 30 s. They
   include these tests:
   - "GPTQ equals the reference implementation"
   - "FP8 rounding is bit-identical to the published grid"
   - "every number the primer computes is still what the code computes"
3. Open [`notebooks/01_number_formats_and_error.ipynb`](notebooks/01_number_formats_and_error.ipynb). Round a
   number to E4M3, E5M2 and E2M1 with one function.

## What you get

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network, in seconds.*

Each notebook opens with "The one-minute version". It works examples against the code and sets exercises. A check
cell that prints ✅ comes after each exercise. Each notebook ends with "In a design review". The finished versions
are in [`solutions/`](solutions/). With the primer, the work takes about 10 hours in all (module 04.9 in the
curriculum of the repo).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_number_formats_and_error`](notebooks/01_number_formats_and_error.ipynb) | Enumerate INT, FP8 (E4M3/E5M2) and FP4 grids from their bits, and round onto them. Compare INT4, FP4, MXFP4 and NVFP4 on Gaussian and heavy-tailed weights. Predict SQNR from bits and crest factor (~6 dB per bit, 20 dB per 10× outlier). Count bits per weight and whole-model GB. Pack INT4 and FP4 codes as a checkpoint does. | §2, §3, §9 | ~1.5 h | T0 |
| [`02_granularity_and_outliers`](notebooks/02_granularity_and_outliers.ipynb) | Reproduce serving-engine §8's six-scheme table. Show why per-channel scales repair the error of an outlier row, but not of an outlier input column. Measure what per-token INT8 does to activations with outlier channels (and what FP8 does instead). Select a static activation scale on calibration data. Use 128×128 block scales. | §2, §3 | ~1.5 h | T0 |
| [`03_gptq_awq_and_smoothquant_from_scratch`](notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) | Implement GPTQ's column loop and AWQ's scale search. Fold SmoothQuant and AWQ scales into a model with no change to the model. Say which layers each method helps, and why. Measure how much calibration data is sufficient. | §4, §5 | ~2 h | T0 |
| [`04_activation_and_kv_cache_quantization`](notebooks/04_activation_and_kv_cache_quantization.ipynb) | Implement a W8A8 GEMM epilogue. Compare static and dynamic activation scales. Show why the LM head stays 16-bit. Find how good FP8 KV is with and without calibrated scales. Implement KIVI's per-channel keys. Calculate the size of the cache. | §5, §6 | ~1.5 h | T0 |
| [`05_choosing_a_scheme`](notebooks/05_choosing_a_scheme.ipynb) | Reproduce serving-engine §8's L4 table and vllm-internals §8.1's GEMM table. Find where W4A16 no longer pays for itself. Say what a checkpoint runs as on T4 to B200. Select schemes for a T4, for an L4, and for an H100 that serves a 70B. Calculate the price of a million tokens (simulated). | §1, §10 | ~1.5 h | T0 |

## Run it

```bash
cd quant-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 86 tests, ~30 s
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

The first code cell of every notebook pins BLAS to one thread (`OPENBLAS_NUM_THREADS=1`), and
`tests/conftest.py` does the same. The matrices here are small. Also, on a busy shared machine, a multi-threaded OpenBLAS
can make one 256×256 inverse a hundred times slower.

## The whole library

Read the modules in this order. Each module opens with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`quantcore/formats.py`](quantcore/formats.py) | ~195 | A format is a grid plus a scale. The module has INT conventions (restricted, full, asymmetric), and FP8 E4M3/E5M2 and FP4 E2M1 grids from their bit patterns. It has rounding (bit-identical to torch's FP8 casts), MXFP4 and NVFP4 block formats, bits per weight and the SQNR rule. It also packs INT4/FP4 codes as compressed-tensors lays them out. |
| [`quantcore/granularity.py`](quantcore/granularity.py) | ~130 | Who shares a scale: tensor, channel, group, token, 2-D block, with scales in the checkpoint's shapes. Static and dynamic activation scales. Error metrics that hide per-channel damage, and error metrics that do not. |
| [`quantcore/gptq.py`](quantcore/gptq.py) | ~85 | The Hessian from calibration inputs, damping, the Optimal Brain Surgeon column update, act-order with static groups, RTN. |
| [`quantcore/awq.py`](quantcore/awq.py) | ~55 | Activation-aware scaling: the α grid search and why α = 0 is RTN. |
| [`quantcore/smoothquant.py`](quantcore/smoothquant.py) | ~45 | How to move activation outliers into weights for W8A8, and the α sweep. |
| [`quantcore/w8a8.py`](quantcore/w8a8.py) | ~60 | The W8A8 GEMM with integer accumulation and the scale epilogue, static-scale saturation, and DeepSeek-V3-style block FP8. |
| [`quantcore/kvquant.py`](quantcore/kvquant.py) | ~85 | FP8 KV with default, calibrated and per-token scales. KIVI (keys per channel, values per token, residual window). Attention-output error. |
| [`quantcore/tinymodel.py`](quantcore/tinymodel.py) | ~140 | A residual MLP stack whose norm gains make LLM-like outlier channels. A closed-form head, calibration capture and exact folds of the scales. `quantize_model` (RTN, GPTQ, AWQ, AWQ then GPTQ). |
| [`quantcore/eval.py`](quantcore/eval.py) | ~55 | KL, top-1 agreement, perplexity, task accuracy with flips both ways, lm-eval's standard error, the unpaired difference's error bar and McNemar's paired z, a budget check. |
| [`quantcore/cost.py`](quantcore/cost.py) | ~210 | What a scheme runs as per GPU generation (vLLM's rules). One GEMM on the roofline and the W4A16 crossover. The step-time model of `minengine.perf`, KV blocks and sessions, the decision table and $/M tokens. |

## What the tests prove

`tests/` has one focused test per concept (78, plus 8 checks on the notebook tools). That is 86 tests, offline, in
about 30 s. The tests that carry the correctness claims are these:

- **It reproduces the repo's published numbers with its own code** (`test_repo_numbers.py`). The numbers
  include serving-engine PRIMER §8's six-scheme error table and FP8 grid statistics. They also include
  outlier-channel and SmoothQuant numbers (with the same seed as `mini-engine-core` notebook 06), and the SIMULATED
  Llama-3.1-8B-on-L4 table to the digit. They include vllm-internals §8.1's `down_proj` table (392/102/196 µs …),
  with the W4A16 crossover at 120 tokens on an L4 and 85 on an H100. They also include servelab's Qwen2.5-0.5B
  weight bytes. If `mini-engine-core` is on disk, the test also compares `fake_quant` directly with
  `minengine.quant`.
- **The formats are the real formats.** The E4M3 grid, enumerated from bit patterns, has 253 distinct values
  and a top of 448. Rounding gives the nearest grid value everywhere, and it matches vLLM's FP4 tie thresholds
  exactly. When the package packs INT4 codes, the result matches compressed-tensors' `0xfcba9810`, and a round trip
  returns the same values (`test_formats.py`).
- **GPTQ is GPTQ.** It matches a line-by-line transcription of the reference's blocked "lazy batch" loop to
  1e-9 per channel and whenever groups start on block boundaries. A group that starts
  mid-block takes its scale from weights that the reference has not updated yet, and the test shows that difference
  too. When the inputs have no correlation, GPTQ reduces to RTN exactly. On correlated inputs, it beats RTN out of
  sample (`test_gptq.py`).
- **Reparameterisations are exact.** SmoothQuant and AWQ folds keep the model's output the same, to within
  1e-9. Also, α = 0 in AWQ's search is RTN (`test_awq_smoothquant.py`).
- **The W8A8 epilogue is exact.** Integer-accumulated INT8 GEMMs and per-block FP8 GEMMs equal the fake-quantized
  products (`test_w8a8_kv.py`).
- **The primer says what the code computes.** The test calculates again every number that `../PRIMER.md`
  attributes to `quantcore`, and each number must appear in the primer verbatim (`test_primer_numbers.py`). The
  numbers do not depend on the numpy or BLAS build because of a precaution in the code. INT4's full convention
  puts −amax exactly on the tie −7.5, and floating-point arithmetic can land an ulp to either side of it. Thus
  `formats.quantize_int` first moves near-ties onto the tie. The suite gives the same digits on numpy 1.26, 2.0 and 2.4.

## Caveats: what is faithful, and what is simplified

These parts are faithful to the sources, read at the commits that the Sources section of the primer gives (see
also its Verify list):

- GPTQ's update, damping and act-order (group scales: see `gptq.py`'s docstring).
- AWQ's statistic, normalisation and 20-point grid, and SmoothQuant's scale formula.
- compressed-tensors' INT conventions, NVFP4 global scale and INT4/FP4 pack layout. Also both MXFP4
  shared-exponent rules: the OCP spec's floor, and compressed-tensors' round-up at a mantissa of 1.75.
- KIVI's key/value asymmetry, lm-eval's standard error, and vLLM's capability rules.

These parts are simpler than the sources:

- Everything runs in float64 "fake quantization". Thus the errors are visible. The code emulates no kernel beyond
  its arithmetic.
- The small model is an MLP stack on a synthetic task, not a transformer. Its down-projections are more redundant
  than real ones. The extra redundancy makes GPTQ look better than it does on a real model.
- The primer describes AWQ's clipping search, but the code does not implement it.
- `cost.supported` is a table of rules, not a probe of your GPU.

Every latency and throughput that `quantcore.cost` prints is a **simulated** value, and it has that label. The simulation is a
roofline model with 80% of bandwidth, 60% of peak FLOP/s and 2 ms per step, as in `minengine.perf`. The GPU figures
are dense datasheet values `(verify)`.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources, then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three commands and the tests. On Colab, the first cell of each notebook clones the repo and
installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../quant-lab/`](../quant-lab/). There, you produce real checkpoints with llm-compressor, and you serve
FP16, INT4 and FP8 side by side in vLLM. You measure accuracy with lm-eval, and you turn on an FP8 KV cache on a
GPU. The lab has a T0 fake server for every notebook.

To learn how vLLM selects a quantization method and kernel in its source, read
[`../../vllm-internals/`](../../vllm-internals/README.md) §8. For the roofline and the cost-per-token arithmetic that
this package uses, see layer 01's
[`roofline-and-fabric`](../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). For where each tier runs and
what it costs, see [`COMPUTE.md`](../../../COMPUTE.md). This package has the MIT license.
