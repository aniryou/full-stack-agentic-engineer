# quantization — pick a number format for a model and a GPU, and know what it costs you

After this topic you can say what INT4, FP8, NVFP4 or an FP8 KV cache will buy for a given model on a given GPU:
decode speed, prefill speed or concurrency. You can say what each runs as on that GPU generation and how to make
4 bits accurate with GPTQ, AWQ or SmoothQuant. Then you can produce, serve and evaluate a real quantized
checkpoint.

## Start here

1. Read [PRIMER.md](PRIMER.md): "The one-minute version", then §1 *Why quantize, and what it can and cannot speed
   up* (15 min). It deepens [serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization); read that
   first if you have not.
2. `cd quant-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 71 tests in about 5 s;
   then open [`01_number_formats_and_error`](quant-core/notebooks/01_number_formats_and_error.ipynb).
3. The fastest win, under a second on a laptop:
   `cd quant-core && python3 -c "from quantcore import cost; print(cost.supported(cost.GPUS['A100-80GB'], 'w8a8-fp8'))"`
   prints what an FP8 checkpoint really runs as on an A100: weight-only, a memory win but no FP8 math.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain quantization for inference in ten sections — §1 why quantize · §2 number formats · §3 granularity and the bits-per-weight budget · §4 weight-only PTQ (GPTQ, AWQ, kernels) · §5 weight-and-activation quantization · §6 KV-cache quantization · §7 QAT and QLoRA in brief · §8 measuring the accuracy you pay · §9 producing a checkpoint · §10 choosing a scheme — each formula with a worked number and the core function that computes it; then "In a design review", a glossary, sources and a dated Verify list | ~2 h, read alongside the core | — |
| [`quant-core/`](quant-core/) | implement it in `quantcore` (numpy, ~620 lines of code): FP8/FP4/MX/NV grids from their bits, scale granularity, GPTQ, AWQ, SmoothQuant, W8A8 epilogues, FP8 and KIVI KV caches, a tiny model with LLM-like outliers, an eval with error bars, and a per-GPU cost and decision model; five fill-in notebooks | ~10 h with the primer | T0 |
| [`quant-lab/`](quant-lab/) | produce real checkpoints with llm-compressor (FP8 dynamic, W4A16), serve FP16 vs INT4 vs FP8 in vLLM, measure the accuracy cost with lm-eval, turn on an FP8 KV cache, and work through the NVFP4/MXFP4 layouts; every notebook has a T0 path (a bundled tiny model and a fake server, labelled simulated) | ~10 h | T0 → T1 (T3 optional) |

### Work it in this order

Read the primer sections, do the core notebook (T0), then run the lab notebook — at T0 first, then on a GPU if
you have one. The repo's curriculum ([`CURRICULUM.md`](../../CURRICULUM.md)) numbers this topic module 04.9, after
serving-engine (04.1–04.7).

| Step | Primer | Core notebook (T0) | Lab notebook (`quant-lab/notebooks/`) | Tier |
|---|---|---|---|---|
| 1. Formats and their error | §1 Why quantize · §2 Number formats | [`01_number_formats_and_error`](quant-core/notebooks/01_number_formats_and_error.ipynb) | `05_fp4_and_the_blackwell_path` | T0 |
| 2. Granularity and outliers | §3 Granularity and the bits-per-weight budget | [`02_granularity_and_outliers`](quant-core/notebooks/02_granularity_and_outliers.ipynb) | `01_quantize_a_checkpoint` | T0 → T1 |
| 3. Calibration algorithms | §4 Weight-only post-training quantization · §5 Weight-and-activation quantization | [`03_gptq_awq_and_smoothquant_from_scratch`](quant-core/notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) | `01_quantize_a_checkpoint`, `03_measure_the_accuracy_cost` | T0 → T1 |
| 4. Activations and the KV cache | §5 · §6 KV-cache quantization | [`04_activation_and_kv_cache_quantization`](quant-core/notebooks/04_activation_and_kv_cache_quantization.ipynb) | `04_kv_cache_quantization_in_vllm` | T0 → T1 (Ada or newer) |
| 5. Choosing and shipping | §7 QAT and QLoRA · §8 Measuring the accuracy you pay · §9 Producing a checkpoint · §10 Choosing a scheme | [`05_choosing_a_scheme`](quant-core/notebooks/05_choosing_a_scheme.ipynb) | `02_serve_and_compare_schemes`, `03_measure_the_accuracy_cost` | T0 → T1 |

Each notebook ends with "In a design review": the two-minute explanation and its drills. The primer's own
design-review section covers the whole topic.

## Run it

```bash
cd quant-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 71 tests, ~5 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../quant-lab
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 -m jupyterlab notebooks
```

On Colab, every notebook's first cell clones the repo and installs its package; the links are in the
[layer README](../README.md).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | every core notebook; every lab notebook's T0 path (a bundled tiny model written as a compressed-tensors-style checkpoint, a fake server, the offline mini-eval); all latencies are **simulated** | laptop, Colab CPU or CI — $0 |
| **T1** | llm-compressor on a 0.5B model; vLLM serving FP16, INT4 and FP8 checkpoints; lm-eval subsets; FP8 KV cache (Ada or newer) | Colab/Kaggle T4 (free: INT4 with fp16 and INT8 W8A8, no FP8 math, no FP8 KV); an RTX 4090 or L4 for FP8 (~$0.3–0.7/hr, verify) |
| **T3** | the serving lab's Cloud Run GPU or GKE deploy with a quantized model (no new Terraform here) | GCP, pay per use; see [`vllm-serving-lab/deploy/`](../serving-engine/vllm-serving-lab/deploy/) for cleanup |

Prices, free tiers and how to obtain GPUs on GCP and elsewhere: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization) and `mini-engine-core` notebook [`06_quantization`](../serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) | the survey this topic deepens: formats, granularity, weight-only vs W8A8, what each buys on an L4 (`quantcore` reproduces its numbers) |
| before | layer 01 [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §1–3, §8; [gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) | peak FLOP/s by precision, why decode is a weight read, cost per token; tensor cores and the precision ladder |
| before | [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) | bytes per parameter, KV bytes, sessions |
| beside | [`vllm-internals`](../vllm-internals/README.md) §6.3 and §8; the [FlashAttention deep dive](../flash-attention/flash-attention-deep-dive.md) §9 | how vLLM picks a quantization method, kernel and attention backend; FP8 attention's error sources |
| beside | [`vllm-serving-lab`](../serving-engine/vllm-serving-lab/) (`servelab.sizing`, notebook 05, `deploy/`) | the sizing, benchmarking and deploys the lab reuses |
| after | [`00-foundations/mixture-of-experts/`](../../00-foundations/mixture-of-experts/) | quantized experts (MXFP4 in gpt-oss), routers kept 16-bit |
| after | [`05-orchestrator`](../../05-orchestrator/README.md); [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | fleets of quantized replicas; cost per conversation |

## Caveats

- **Simulated vs measured.** The core's latencies, and the lab's fake server, come from a roofline model with
  assumed efficiencies and are labelled SIMULATED. Only the lab's T1 paths measure. Accuracy numbers in the core
  come from a tiny synthetic model: they show the mechanisms and their direction, not a real model's losses.
- **Kernel rules move fast.** What a scheme runs as on each GPU generation — FP8 below Ada, NVFP4 below Blackwell,
  INT8 W8A8 on Blackwell, FP8 KV on a T4 — follows vLLM 0.30.0 and `main` as of September 2026. `--quantization fp8`
  on a BF16 checkpoint works at 0.30.0 and raises at `main` (use `fp8_per_tensor`). Check the `Selected <kernel>`
  log line on your version; the primer's [Verify list](PRIMER.md#verify-list) collects the dated facts.
- **Checked by construction.** The GPU and cloud paths are checked here with tests, `bash -n` and dry runs, not run
  on real hardware. Blackwell (NVFP4) figures are datasheet-derived and marked `(verify)`.
