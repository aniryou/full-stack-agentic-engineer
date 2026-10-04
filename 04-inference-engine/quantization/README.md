# quantization — pick a number format for a model and a GPU, and know what it costs you

After this topic, you can say what INT4, FP8, NVFP4 or an FP8 KV cache will give for a given model on a given
GPU. The gain can be decode speed, prefill speed or concurrency. You can say what each format runs as on that GPU
generation. You can also say how to use GPTQ, AWQ or SmoothQuant to make 4 bits accurate. Then you can produce,
serve and evaluate a real quantized checkpoint.

## Start here

1. Read "The one-minute version" in [PRIMER.md](PRIMER.md). Then read §1 *Why quantize, and what it can and
   cannot speed up*. This takes 15 min. The primer goes deeper than
   [serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization). If you have not read that section, read
   it first.
2. Run `cd quant-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 86 tests run in
   about 30 s. Then open [`01_number_formats_and_error`](quant-core/notebooks/01_number_formats_and_error.ipynb).
3. For the fastest result, run this command. It takes less than a second on a laptop:
   `cd quant-core && python3 -c "from quantcore import cost; print(cost.supported(cost.GPUS['A100-80GB'], 'w8a8-fp8'))"`.
   It prints what an FP8 checkpoint really runs as on an A100. The answer is weight-only. A weight-only checkpoint saves
   memory, but it does no FP8 math.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, and it is optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain quantization for inference in ten sections. The first three are §1 why quantize · §2 number formats · §3 granularity and the bits-per-weight budget. The next three are §4 weight-only PTQ (GPTQ, AWQ, kernels) · §5 weight-and-activation quantization · §6 KV-cache quantization. The last four are §7 QAT and QLoRA in brief · §8 measuring the accuracy you pay · §9 producing a checkpoint · §10 choosing a scheme. Each formula has a worked number and the core function that computes it. After the ten sections come "In a design review", a glossary, sources and a dated Verify list. | ~2 h. Read it together with the core. | — |
| [`quant-core/`](quant-core/) | Implement it in `quantcore` (numpy, ~640 lines of code). It has FP8/FP4/MX/NV grids from their bits, scale granularity, GPTQ, AWQ and SmoothQuant. It also has W8A8 epilogues, FP8 and KIVI KV caches, and a small model with LLM-like outliers. It has an eval with error bars, and a per-GPU cost and decision model. There are five fill-in notebooks. | ~10 h with the primer | T0 |
| [`quant-lab/`](quant-lab/) | Produce real checkpoints with llm-compressor (FP8 dynamic, W4A16). Serve FP16, INT4 and FP8 in vLLM, and compare them. Measure the accuracy cost with lm-eval. Turn on an FP8 KV cache. Examine the NVFP4/MXFP4 layouts step by step. Every notebook has a T0 path: a bundled small model and a fake server, labelled simulated. | ~9 h | T0 to T1 (T3 optional) |

### Work it in this order

Read the primer sections. Then do the core notebook (T0). Then run the lab notebook at T0 first. If you have a GPU,
run the lab notebook on the GPU after that. In the curriculum of the repo ([`CURRICULUM.md`](../../CURRICULUM.md)),
this topic is module 04.9. It comes after serving-engine (04.1–04.7).

| Step | Primer | Core notebook (T0) | Lab notebook (`quant-lab/notebooks/`) | Tier |
|---|---|---|---|---|
| 1. Formats and their error | §1 Why quantize · §2 Number formats | [`01_number_formats_and_error`](quant-core/notebooks/01_number_formats_and_error.ipynb) | No lab notebook. Lab 05's exercises 5.1–5.3 on E2M1, MXFP4 and NVFP4 fit here, if you want them early. | T0 |
| 2. Granularity and outliers | §3 Granularity and the bits-per-weight budget | [`02_granularity_and_outliers`](quant-core/notebooks/02_granularity_and_outliers.ipynb) | [`01_quantize_a_checkpoint`](quant-lab/notebooks/01_quantize_a_checkpoint.ipynb) | T0 to T1 |
| 3. Calibration algorithms | §4 Weight-only post-training quantization · §5 Weight-and-activation quantization | [`03_gptq_awq_and_smoothquant_from_scratch`](quant-core/notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) | [`01_quantize_a_checkpoint`](quant-lab/notebooks/01_quantize_a_checkpoint.ipynb) (RTN, GPTQ and AWQ compared), [`03_measure_the_accuracy_cost`](quant-lab/notebooks/03_measure_the_accuracy_cost.ipynb) | T0 to T1 |
| 4. Activations and the KV cache | §5 · §6 KV-cache quantization | [`04_activation_and_kv_cache_quantization`](quant-core/notebooks/04_activation_and_kv_cache_quantization.ipynb) | [`04_kv_cache_quantization_in_vllm`](quant-lab/notebooks/04_kv_cache_quantization_in_vllm.ipynb) | T0 to T1 (Ada or newer) |
| 5. Selection and release | §7 QAT and QLoRA · §8 Measuring the accuracy you pay · §9 Producing a checkpoint · §10 Choosing a scheme | [`05_choosing_a_scheme`](quant-core/notebooks/05_choosing_a_scheme.ipynb) | [`02_serve_and_compare_schemes`](quant-lab/notebooks/02_serve_and_compare_schemes.ipynb), [`03_measure_the_accuracy_cost`](quant-lab/notebooks/03_measure_the_accuracy_cost.ipynb), then [`05_fp4_and_the_blackwell_path`](quant-lab/notebooks/05_fp4_and_the_blackwell_path.ipynb) (its base is §5's W4A4 and SmoothQuant, and lab 02's GEMM roofline) | T0 to T1 (Blackwell for 05) |

Each notebook ends with "In a design review". That section has the two-minute explanation and its drills. The
primer has its own design-review section, and that section covers the whole topic.

## Run it

```bash
cd quant-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 86 tests, ~30 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../quant-lab
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 -m jupyterlab notebooks
```

On Colab, the first cell of every notebook clones the repo and installs its package. The links are in the
[layer README](../README.md).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | Every core notebook. The T0 path of every lab notebook: a bundled small model written as a compressed-tensors-style checkpoint, a fake server and the offline mini-eval. All latencies are **simulated** values. | laptop, Colab CPU or CI, $0 |
| **T1** | llm-compressor on a 0.5B model. vLLM that serves FP16, INT4 and FP8 checkpoints. lm-eval subsets. An FP8 KV cache (Ada or newer). NVFP4 W4A4 (Blackwell). | Colab/Kaggle T4 (free: INT4 with fp16 and INT8 W8A8, no FP8 math, no FP8 KV). An RTX 4090 or L4 for FP8 (~$0.3–0.7/hr, verify). One rented B200 or RTX PRO 6000 for NVFP4 (verify). |
| **T3** | the serving lab's Cloud Run GPU or GKE deploy with a quantized model (no new Terraform here) | GCP, pay per use. See [`vllm-serving-lab/deploy/`](../serving-engine/vllm-serving-lab/deploy/) for cleanup. |

For prices, free tiers and how to get GPUs on GCP and elsewhere, see [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [serving-engine PRIMER §8](../serving-engine/PRIMER.md#8-quantization) and `mini-engine-core` notebook [`06_quantization`](../serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) | the survey that this topic goes deeper into: formats, granularity, weight-only against W8A8, and what each gives on an L4 (`quantcore` reproduces its numbers) |
| before | layer 01 [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §1–3 and §8, and [gpu-primer §4](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) | peak FLOP/s by precision, why decode is a weight read, cost per token, tensor cores and the precision ladder |
| before | [capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md) | bytes per parameter, KV bytes, sessions |
| beside | [`vllm-internals`](../vllm-internals/README.md) §6.3 and §8, and the [FlashAttention deep dive](../flash-attention/flash-attention-deep-dive.md) §9 | how vLLM selects a quantization method, a kernel and an attention backend, and the error sources of FP8 attention |
| beside | [`vllm-serving-lab`](../serving-engine/vllm-serving-lab/) (`servelab.sizing`, notebook 05, `deploy/`) | the sizing, the benchmarks and the deploys that the lab uses again |
| after | [mixture-of-experts §6.7 *Quantized experts*](../../00-foundations/mixture-of-experts/PRIMER.md#67-quantized-experts) | quantized experts (MXFP4 in gpt-oss), and routers that stay 16-bit |
| after | [`05-orchestrator`](../../05-orchestrator/README.md) and [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | fleets of quantized replicas, and cost per conversation |

## Caveats

- **Simulated against measured.** The latencies of the core and the lab's fake server come from a roofline model with
  assumed efficiencies. They have the label SIMULATED. Only the lab's T1 paths measure. The accuracy numbers in the
  core come from a small synthetic model. They show the mechanisms and their direction, not the losses of a real
  model.
- **Kernel rules move fast.** What a scheme runs as on each GPU generation follows vLLM 0.30.0 and `main` as of
  September 2026. These rules cover FP8 below Ada, NVFP4 below Blackwell, INT8 W8A8 on Blackwell and FP8 KV on a T4.
  `--quantization fp8` on a BF16 checkpoint works at 0.30.0 and raises an exception at `main` (use
  `fp8_per_tensor`). Examine the `Selected <kernel>` log line on your version. The primer's
  [Verify list](PRIMER.md#verify-list) collects the dated facts.
- **Checked by construction.** This topic examines the GPU and cloud paths with tests, `bash -n` and dry runs. It
  does not run them on real hardware. The Blackwell (NVFP4) figures come from datasheets and have the mark
  `(verify)`.
