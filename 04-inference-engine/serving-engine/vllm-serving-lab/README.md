# vllm-serving-lab — size, measure and tune a real vLLM server against an SLO

After this lab, you can do these things:

- Tell how many sessions of a model fit on a GPU before you pay for the GPU.
- Measure TTFT, ITL and goodput with the definitions of `vllm bench serve`.
- Read the `/metrics` endpoint that vLLM itself supplies.
- Select the flags of vLLM against an SLO.

You do these things first against a bundled fake vLLM on a laptop. Then you do them against a real `vllm serve` on a
GPU, Cloud Run or GKE.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384`.
   It takes less than a second. It shows the KV blocks and the concurrency of an 8B model on a 24 GB L4, line by line.
2. Open [`notebooks/01_size_before_you_serve.ipynb`](notebooks/01_size_before_you_serve.ipynb) (T0). It shows each
   assumption behind that number. It also shows how to calibrate the number against the startup log of vLLM.
3. Start the fake vLLM with `python3 -m servelab fake --port 8000 &`. Then measure it with
   [`notebooks/02_serve_and_measure.ipynb`](notebooks/02_serve_and_measure.ipynb).

## What you get

*Tiers: T0 is a laptop or a Colab CPU, free. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is a
multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, optional.*

Each notebook starts with *the one-minute version*. Then it runs worked examples against the library. Then it gives
3–6 exercises: implement the key function, predict a number, select a setting. After each exercise, a check prints
✅. The notebook ends with *in a design review*.

The answers are in [`solutions/`](solutions/). The times are approximate (about 12 h in all, from the curriculum of
the repository).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`size_before_you_serve`](notebooks/01_size_before_you_serve.ipynb) | T0 (+T1 log) | KV bytes per token from `config.json`. Blocks and "Maximum concurrency". The ~18 concurrent 2K-token sessions of an 8B model on an L4, and each assumption behind that number. Why the model does not start with its 128K context. What FP8 gives. How to calibrate from the startup log. | §4 | ~1.5 h |
| 02 | [`serve_and_measure`](notebooks/02_serve_and_measure.ipynb) | T0 / T1 / T3 | TTFT, ITL, TPOT, E2E, throughput and goodput as `vllm bench serve` defines them (with one stated difference). Histogram quantiles. Open loop against closed loop, and the backlog that an open loop builds. Little's law against the gauges of the engine. Warm-up, burstiness and long-tailed lengths. | §11 | ~1.5 h |
| 03 | [`knobs_and_tradeoffs`](notebooks/03_knobs_and_tradeoffs.ipynb) | T0 / T1 (+T2) | Why batching is almost free. The latency-throughput knee. `max-num-batched-tokens` as a trade between TTFT and the ITL tail. How to select `max-num-seqs` by goodput. Capacity at an SLO. KV blocks and preemption. Tensor parallelism on two T4s. | §2, §3, §9, §11 | ~3 h |
| 04 | [`prefix_caching_for_agents`](notebooks/04_prefix_caching_for_agents.ipynb) | T0 / T1 | Block-hash chains. The rules that tell how vLLM counts hits. The hit rate from `/metrics`, predicted from the prompt layout before you measure it. Prompt layouts that keep the cache useful for agents, or make it useless. The value of a hit in TTFT. | §5 | ~2 h |
| 05 | [`speculation_and_quantization_in_vllm`](notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 (+T1 flags) | Tokens per verify step, `(1-a^(k+1))/(1-a)`. Acceptance from the counters of vLLM. Why the gain of speculation decreases at a high batch and a short context. What INT4 gives against FP8, for prefill and for decode. | §7, §8 | ~2 h |
| 06 | [`deploy_on_cloud_run_gpu`](notebooks/06_deploy_on_cloud_run_gpu.ipynb) | T3 (plannable at T0) | The anatomy of a cold start, and the startup-probe budget. How to set the Cloud Run `concurrency` from a measurement, and what occurs above `max-num-seqs`. Cost per million tokens. Scale-to-zero against a warm instance. | §12 | ~2 h |

Everything runs on a laptop first. `servelab.fakeserver` is a **fake vLLM**. It has the same OpenAI-compatible
streaming API and the same `vllm:*` Prometheus metrics as vLLM. Behind them is an engine emulator with continuous
batching, a block-hash prefix cache, preemption and a roofline step-time model. Its numbers have the label
*simulated*. When you point the same code at a real `vllm serve`, the numbers become measurements.

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | fake vLLM + all analysis | every notebook, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm serve` with a 0.5-1.5B model | `SERVELAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | two GPUs: Kaggle "GPU T4 x2" (free, PCIe) or a rented pair | `vllm serve --tensor-parallel-size 2` | notebook 03 exercise 3.6 (predicted at T0, measured with `SERVELAB_START_VLLM=1`), [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T3** | GCP | vLLM on Cloud Run (L4, scale to zero) or GKE (L4 Spot) | [`deploy/gcp/cloud-run/`](deploy/gcp/cloud-run/), [`deploy/gcp/gke/`](deploy/gcp/gke/) |

vLLM v0.30.0 needs compute capability 7.5 or newer. A T4 works, but the P100 of Kaggle does not. The minimal
from-scratch engine is in the adjacent folder, [`../mini-engine-core/`](../mini-engine-core/). This lab never imports
it. For prices and where to get GPUs, see [`COMPUTE.md`](../../../COMPUTE.md).

## Run it

```bash
cd vllm-serving-lab
python3 -m pip install -e ".[dev]"             # aiohttp + prometheus_client; dev: pytest, jupyter, numpy, pyyaml
python3 -m pytest -q                           # 74 tests, ~30 s, offline
python3 -m servelab fake --port 8000 &         # a fake vLLM (simulated T4 + Qwen2.5-0.5B)
python3 -m servelab bench --url http://127.0.0.1:8000 --rate 5 -n 60 --slo-ttft-ms 300 --slo-tpot-ms 30
python3 -m servelab metrics --url http://127.0.0.1:8000
python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384 --quantization fp8
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

To measure a real engine instead (T1/T3), start one. [`deploy/any-gpu/`](deploy/any-gpu/) has a Colab/Kaggle T4
recipe. Then run `export SERVELAB_URL=http://127.0.0.1:8000`. Also set `SERVELAB_API_KEY`, or set
`SERVELAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run service.

The notebooks detect the engine and measure it. They identify a `servelab fake` server in `SERVELAB_URL` by its
`/version`, and its numbers keep the label simulated. If you have a GPU, vLLM installed and `SERVELAB_START_VLLM=1`,
the sweeps of notebook 03 restart a real `vllm serve` for each configuration.


## The library (`servelab/`, ~3,000 lines)

| Module | Lines | The idea |
|---|---:|---|
| `sizing.py` | ~410 | From `config.json`, a GPU and flags, it calculates the weights, KV bytes/token, blocks, max concurrency, the fit check and the error text of vLLM. It parses the startup log and calibrates against it. Ten bundled configs (Qwen2.5/3, Llama 3.x, Mistral, Mixtral) have parameter counts that agree with the published counts. |
| `bench/` | ~590 | An async OpenAI-compatible load generator. It measures the times of SSE events (`client.py`). It has an open loop with the gamma/Poisson arrivals of vLLM, a closed loop and agent sessions (`runner.py`). It has length distributions and shared-prefix agent workloads (`workload.py`). It has the `vllm bench serve` definitions, goodput included (`summary.py`). It has tables and text curves (`report.py`). |
| `metrics.py` | ~340 | It parses the Prometheus text format. It has a `histogram_quantile` that agrees exactly with PromQL, windows between scrapes, and an engine snapshot (queue, batch, KV, hit rate, preemptions, spec acceptance). It also gives the PromQL for each. |
| `tune.py` | ~250 | It writes the CLI flags for vLLM. It runs sweeps of configs on a pluggable backend (`FakeBackend`, `VLLMBackend`, `URLBackend`). It finds the best config under an SLO and the highest rate under an SLO. |
| `fake_engine.py` | ~540 | The emulator. It has FCFS continuous batching with a token budget and chunked prefill, a block pool with refcounts, chained block hashes and LRU reuse. It also has recompute preemption, roofline step time and Bernoulli speculative acceptance. It has named hardware profiles. |
| `fakeserver.py` | ~360 | The fake vLLM over HTTP (aiohttp). Its endpoints are `/v1/completions`, `/v1/chat/completions` (streaming, usage, cached tokens), `/v1/models`, `/health`, and `/metrics` with the names and bucket edges of vLLM. |
| `env.py`, `textgen.py`, `__main__.py` | ~360 | Tier detection (`SERVELAB_URL`, GPU, vLLM), a toy tokenizer that the client and the server share, and the CLI. |

The metric names and the histogram buckets are the same as in `vllm/v1/metrics/loggers.py`, `buckets.py` and
`vllm/v1/spec_decode/metrics.py` in
vLLM v0.30.0 and main (Sep 2026). The metric names are `vllm:num_requests_running`, `vllm:num_requests_waiting`,
`vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries`/`_hits`, `vllm:num_preemptions`,
`vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`,
`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`,
`vllm:request_decode_time_seconds`, `vllm:request_success`, `vllm:iteration_tokens_total` and
`vllm:spec_decode_num_*`. On the wire, the counters have a `_total` suffix.

## Measurement hygiene (what the code does for you, and why)

* **Definitions match `vllm bench serve`**. TTFT is the time to the first chunk that has a token. ITL is the time
  between chunks that have tokens. TPOT = (E2E − TTFT)/(n − 1), with n from `usage`. Throughput is over the wall time
  of the run. Goodput is the number of requests per second that meet every SLO.

  There is one deliberate difference. Just before the first token, vLLM sends a chat chunk with only the role, and
  this code skips that chunk. `vllm bench serve` counts it as a chunk. In `vllm bench serve`, that chunk sets TTFT and adds one ~0 ms ITL gap
  for each chat request. Here, TTFT is the same, and each chat ITL list has one entry less, the near-zero entry.
  Completions are identical.
* **Open loop for capacity**, closed loop only for "N users" questions. Start the closed-loop users at different
  times (`ramp_s`), so that they do not start as one synchronized burst.
* **Warm-up with different prompts** than the measured ones, and **before** the "before" scrape. If you do not, the
  warm-up adds its prefix-cache hits and its requests to the window.
* **Same seeded workload** for every configuration you compare. Use fixed lengths with `ignore_eos`. Use new prompts for
  each run, because a replayed prompt hits the prefix cache. Also, tell which arrival process (Poisson or
  `burstiness`) and which length distribution (fixed, lognormal) a capacity number assumes.
* **Histogram percentiles are interpolations** inside the bucket edges of vLLM. The means from `_sum/_count` are
  exact. A counter gives information only as the difference between two scrapes.
* **Label the source**: every report says SIMULATED (fake server) or measured-on-URL. The fake server also
  identifies itself as a fake on `/version`, even when you reach it through `SERVELAB_URL`.

The upstream counterpart of the fake server is [`llm-d-inference-sim`](https://github.com/llm-d/llm-d-inference-sim).
It is in Go and OpenAI-compatible, with vLLM metrics and fixed or per-token latency parameters. llm-d uses it for routing experiments
in layer 05. The emulator of this lab *calculates* the latency from a scheduler, a KV cache and a roofline instead.
Thus the flags have their real effects.

## Deploy

[`deploy/`](deploy/) has these targets:

- `any-gpu/`: docker or pip `vllm serve`, the flags with their explanation, a Colab/Kaggle T4 recipe, and RunPod/Vast
  notes.
- `gcp/cloud-run/`: Terraform `google_cloud_run_v2_service` with one L4, scale to zero, weights from Hugging Face or a
  GCS mount, and the HF token from Secret Manager. It also has the `gcloud run deploy` equivalent.
- `gcp/gke/`: a Deployment on an L4 Spot pool, and `PodMonitoring` for Managed Prometheus.

Each target has a README with the cost and the cleanup. The Terraform of this lab is the Cloud Run service
(`deploy/gcp/cloud-run/terraform/`). GKE here is a `gcloud` script and manifests. The cluster as Terraform is in
layer 03's
[`k8s-gpu-lab`](../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) and in layer 05's
[`inference-gateway-lab`](../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/).

## Regenerating notebooks

`tools/build_notebooks.py` generates `notebooks/` (exercises) and `solutions/` from `notebooks_src/*.py` (percent
format with `### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Simulated against measured.** The numbers of the fake server are simulated, and every report says so. The fake server
  also identifies itself as a fake on `/version`, even behind `SERVELAB_URL`. Only a real `vllm serve` gives
  measurements.
- **Sizing is an estimate.** `sizing.size()` uses a model of the defaults of vLLM v0.30.0. It takes 0.92 of the
  driver-reported total, and it estimates the profiled-activation, CUDA-graph and non-torch overheads. The
  `Available KV cache memory` line of the startup log is the measurement, and notebook 01 calibrates against it.
  [`../PRIMER.md`](../PRIMER.md) §4 compares these inputs with the round inputs of the mini engine.
- **Checked by construction.** The checks of the deploy paths are `bash -n`, `DRY_RUN=1`, Terraform `validate` and
  Kubernetes schema checks. This lab does not run the deploy paths on real infrastructure.

## Verify list (facts dated Sep 2026 that move)

A check on 2026-09-26 against the vLLM **v0.30.0** source and Docker Hub found these facts. Do the check again
when you move the pin:

* vLLM **0.30.0** is the latest release (PyPI, 2026-09-22). The image `vllm/vllm-openai:v0.30.0` exists (amd64 and
  arm64). Its base is CUDA 13.0.3, and it has `ENTRYPOINT ["vllm", "serve"]`.
* Defaults: `gpu_memory_utilization` **0.92** (0.9 in older releases), `block_size` 16, prefix caching and chunked
  prefill on. For `vllm serve`, `max_num_batched_tokens` / `max_num_seqs` are 2048 / 256 below 70 GiB or on A100.
  They are 8192 / 1024 on other GPUs from 70 GiB, and 16384 / 1024 from 160 GiB.
* KV budget = `util × total − (weights + profiled activation peak + non-torch) − CUDA-graph
  estimate` (CUDA-graph profiling is on by default since v0.21.0). vLLM keeps one block as the null block.
* `--enable-prompt-tokens-details` adds `usage.prompt_tokens_details.cached_tokens`. `--max-model-len -1` sets the
  maximum length automatically to a value that fits. `--attention-backend` replaces `VLLM_ATTENTION_BACKEND`, and
  `VLLM_USE_V1` no longer exists. `VLLM_BATCH_INVARIANT=1` sets batch-invariant mode on.
* On compute capability 7.5 (T4), the attention backend is `TRITON_ATTN`. In this release, FlashAttention and
  FlashInfer need sm_80+. vLLM refuses `bfloat16` below sm_80. vLLM builds its kernels for sm_75 and newer only (no
  P100/V100).
* The startup-log lines that `sizing.parse_startup_log` parses ("Model loading took", "Available KV cache
  memory", "GPU KV cache size ... Maximum concurrency", "init engine ... took", "The current
  --gpu-memory-utilization=").
* The prefix-cache counters do not include re-admitted preempted requests. Those requests go to separate
  `preempted_*` stats. The `speculative_config` methods are `ngram`, `draft_model` and `eagle3`. The synthetic
  rejection-sampling mode for load tests exists.

These items still need a check that the source here cannot give:

* The minimum driver of CUDA 13 (580 series), and the drivers that Colab and Kaggle supply. The last vLLM release
  built for CUDA 12.
* On each GPU, how far the total memory that CUDA reports is below the total that nvidia-smi reports. On an
  L4, this difference changes the sizing by ~1 session.
* Cloud Run GPU: L4 (24 GB, min 4 vCPU / 16 GiB), GA since June 2025. The flag names `--gpu`, `--gpu-type` and
  `--no-gpu-zonal-redundancy`. The regions, quota names, prices, CPU-always-allocated for GPU services, startup-probe
  limits, and if the probe starts after the image pull.
* GKE: the GPU node taint `nvidia.com/gpu=present:NoSchedule`. `HF_XET_HIGH_PERFORMANCE` in the `huggingface_hub` of
  the image.
* The GPU datasheet numbers in `sizing.GPUS` (memory as the driver reports it, dense TFLOPS, bandwidth).

MIT licensed.
