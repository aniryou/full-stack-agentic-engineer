# serving-engine — what one inference engine does, and which knob trades what

After this topic, you can explain with numbers the work of an engine like vLLM, SGLang or TensorRT-LLM. This work
occurs between "a request arrived" and "tokens are streaming out". It has these parts: continuous batching, chunked prefill, KV
management, prefix caching, sampling, speculation and quantization. You can then size, measure and adjust a real
vLLM server against an SLO.

## Start here

1. Read [PRIMER.md](PRIMER.md). Start with "The one-minute version". Then read §1 Anatomy of an engine and §2 Continuous batching.
2. Run `cd mini-engine-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 75 tests run in
   about 50 s. Then open [`01_the_step_loop_and_continuous_batching`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb).
3. Size a real model before you serve it. This step also runs on a laptop:
   [`vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb`](vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb).

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is a
multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain the engine in twelve sections. The first six are §1 anatomy, §2 continuous batching, §3 chunked prefill, §4 KV cache management, §5 prefix caching and §6 sampling and structured output. The other six are §7 speculative decoding, §8 quantization, §9 parallelism, §10 multi-LoRA, §11 measuring an engine and §12 engines and where to run them. Each formula has a worked number and the core function that computes it. After the sections come "In a design review", a glossary, sources and a dated Verify list. | read alongside the core | — |
| [`mini-engine-core/`](mini-engine-core/) | Build the engine yourself in `minengine`, a numpy "nano-vLLM" (~1,400 lines). It has a small model that reads K/V through block tables, the KV cache manager with prefix caching, the scheduler and the sampler. It also has speculative decoding, quantization and a roofline simulator. The core has six fill-in notebooks. | ~11 h with the primer | T0 |
| [`vllm-serving-lab/`](vllm-serving-lab/) | Size a model from its `config.json`. Drive vLLM with an open- or closed-loop load generator and read its `/metrics`. Do sweeps of its flags against an SLO. Deploy it on any GPU box, Cloud Run GPU or GKE. The package of the lab is `servelab`. A fake vLLM lets every notebook run at T0. The lab has six notebooks. | ~12 h | T0 to T1 (T2, T3 optional) |

### Work it in this order

Read the primer sections. Then do the core notebook (T0). Then run the lab notebook. Run it at T0 against its fake
server first. If you have a GPU, run it on the GPU after that. The module numbers and the times come from the
curriculum of the repository ([`CURRICULUM.md`](../../CURRICULUM.md), modules 04.1–04.7).

| Module | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| 04.1 The step loop | §1 Anatomy of an engine · §2 Continuous batching | [`01_the_step_loop_and_continuous_batching`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) | [`01_size_before_you_serve`](vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb), [`02_serve_and_measure`](vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) | T0 to T1 |
| 04.2 Chunked prefill and the KV budget | §3 Chunked prefill and prefill/decode interference · §4 KV cache management revisited | [`02_chunked_prefill_and_the_token_budget`](mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | [`03_knobs_and_tradeoffs`](vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) | T0 to T1 |
| 04.3 Prefix caching | §5 Prefix caching | [`03_prefix_caching`](mini-engine-core/notebooks/03_prefix_caching.ipynb) | [`04_prefix_caching_for_agents`](vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) | T0 to T1 |
| 04.4 Sampling and structured output | §6 Sampling and structured output | [`04_sampling_and_structured_output`](mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) | — | T0 |
| 04.5 Speculative decoding | §7 Speculative decoding | [`05_speculative_decoding`](mini-engine-core/notebooks/05_speculative_decoding.ipynb) | [`05_speculation_and_quantization_in_vllm`](vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 to T1 |
| 04.6 Quantization | §8 Quantization | [`06_quantization`](mini-engine-core/notebooks/06_quantization.ipynb) | [`05_speculation_and_quantization_in_vllm`](vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 to T1 (FP8 needs sm_89+) |
| 04.7 Parallelism, LoRA, measurement, deployment | §9 Parallelism inside the engine · §10 Multi-LoRA serving · §11 Measuring an engine · §12 Engines and where to run them | [`01`](mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) exercise 1.6 (tensor parallelism). [`02`](mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) worked example 3 (open loop against saturation, goodput). §10 has no notebook exercise. Its numbers come from `perf.lora_params()` and the adapter-salted block hashes. `tests/test_perf.py` and `tests/test_kv.py` pin them. | [`03_knobs_and_tradeoffs`](vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) exercises 3.1–3.5 (measurement) and 3.6 (tensor parallelism on two T4s, §9). [`06_deploy_on_cloud_run_gpu`](vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) (§12) | T0 to T1 (3.6 is T2, deployment is T3) |

Each notebook ends with "In a design review". That section has the two-minute explanation and its drills. The
design-review section of the primer covers the whole topic.

## Run it

```bash
cd mini-engine-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 75 tests, ~50 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../vllm-serving-lab
python3 -m pip install -e ".[dev]"
python3 -m pytest -q                           # 74 tests, ~30 s, offline
python3 -m jupyterlab notebooks
```

On Colab, the first cell of every notebook clones the repo and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | Every core notebook, the sizing notebook of the lab, and the measurement notebooks of the lab against its bundled fake server (with a clear label). Every latency from `minengine.perf` is **simulated**. | laptop, Colab CPU or CI, $0 |
| **T1** | The lab against a real vLLM server with a 0.5–2B model: TTFT/ITL, knob sweeps, prefix caching, n-gram speculation and quantized checkpoints. | Colab/Kaggle T4 (free, fp16 only), any 24 GB GPU (~$0.3–0.7/hr, verify), GCP L4 Spot |
| **T2** | Optional: tensor parallelism across two GPUs (§9, exercise 3.6 of the lab). | Kaggle 2×T4 (PCIe) or a rented NVLink pair |
| **T3** | The Cloud Run GPU deployment of the lab (scale to zero) and its GKE Deployment. | GCP, pay per use. For cleanup, see the `deploy/` READMEs of the lab. |

Prices, free tiers and how to get GPUs on GCP and in other places: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`00-foundations/transformers`](../../00-foundations/transformers/) (primer §7) and [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) | Attention, the KV cache and decoding. Weights, KV bytes, TTFT and TPOT on one page. |
| before | [`01 gpu-primer`](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) §3 and [`roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §2–3 | HBM, bandwidth, why a decode step is a memory read and where the step-time knee comes from |
| before | the kernel topics beside this one: [`kv-cache`](../kv-cache/kv-cache-primer.md), [`paged-attention`](../paged-attention/paged-attention-primer.md), [`flash-attention`](../flash-attention/flash-attention-primer.md) | Block tables and copy-on-write. Block-hash prefix caching never needs copy-on-write (primer §5). Tiling and online softmax. |
| beside | layer 02's [cuda-and-nccl primer](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §4–5 and layer 03's [gpu-scheduling](../../03-kubernetes-gpu/gpu-scheduling/README.md) topic | CUDA Graphs and the all-reduces that tensor parallelism runs on. How the pod of the engine gets its GPUs. |
| after | [`vllm-internals`](../vllm-internals/README.md) | The same mechanisms, read in the source of vLLM, with line numbers |
| after | [`quantization`](../quantization/README.md) | The deep dive behind primer §8. It covers formats to the bit, GPTQ/AWQ/SmoothQuant, what each scheme runs as per GPU, and FP8 KV. It also serves a real quantized checkpoint and runs an eval on it. |
| after | [`00 distillation`](../../00-foundations/distillation/README.md) (primer §7) | That topic trains the draft model of primer §7 as a student of its target. Then it measures the acceptance of the draft under `--speculative-config`. |
| after | [`05-orchestrator`](../../05-orchestrator/README.md) | Many replicas. That layer routes them by prefix-cache affinity and load, and autoscales them on queue depth and KV usage. It also divides them into prefill and decode pools. |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) and [`07-application-agent-framework`](../../07-application-agent-framework/) | Admission, rate limits and cost in front of the fleet. Also the agent workloads that shape all of it: long stable prefixes and append-only histories. |

## Caveats

- **Simulated against measured.** The latencies of the core come from a roofline model that drives its real scheduler.
  The fake server of the lab uses the same kind of model. Both have the label SIMULATED. The lab measures only
  against a real `vllm serve`.
- **Two sets of sizing inputs.** The core sizes KV with round inputs (0.9 of 24 GB minus a flat 1 GB). The lab
  models the defaults of vLLM v0.30.0 (0.92 of the driver-reported total minus profiled overheads). PRIMER §4 sets
  them side by side: 2,164 against 2,363 blocks for Llama-3.1-8B on an L4. The value `Available KV cache memory` in
  the startup log is the measurement.
- **Examined, not run.** The repository examines the GPU, Cloud Run and GKE paths with `bash -n`, `DRY_RUN=1`,
  Terraform `validate` and Kubernetes schema checks. It does not run them on real infrastructure. Expect to adjust
  quotas and regions when you use them for the first time.
- **Dated facts.** The defaults of vLLM v0.30.0, the GPU prices and the Cloud Run details are as of September 2026.
  Each has the tag `(verify)`. The [Verify list](PRIMER.md#verify-list) of the primer collects them.
