# vllm-serving-lab — size, measure and tune a real vLLM server against an SLO

After this lab you can say how many sessions a model fits on a GPU before paying for it, measure TTFT, ITL and
goodput the way `vllm bench serve` defines them, read vLLM's own `/metrics`, and choose its flags against an SLO —
first against a bundled fake vLLM on a laptop, then against a real `vllm serve` on a GPU, Cloud Run or GKE.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384`
   — under a second: the KV blocks and concurrency an 8B model gets on a 24 GB L4, line by line.
2. Open [`notebooks/01_size_before_you_serve.ipynb`](notebooks/01_size_before_you_serve.ipynb) (T0) — every
   assumption behind that number, and how to calibrate it against vLLM's startup log.
3. Start the fake vLLM and measure it: `python3 -m servelab fake --port 8000 &`, then
   [`notebooks/02_serve_and_measure.ipynb`](notebooks/02_serve_and_measure.ipynb).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* Each notebook opens with *the one-minute version*,
works examples against the library, then 3–6 exercises (implement the key function, predict a number, pick a setting)
each followed by a check that prints ✅, and closes with *in a design review*. Answers are in
[`solutions/`](solutions/). Times are rough (about 12 h in all, from the repo's curriculum).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`size_before_you_serve`](notebooks/01_size_before_you_serve.ipynb) | T0 (+T1 log) | KV bytes per token from `config.json`; blocks and "Maximum concurrency"; ~18 concurrent 2K-token sessions of an 8B model on an L4, and every assumption behind that number; why its 128K context does not start; what FP8 buys; calibrating from the startup log | §4 | ~1.5 h |
| 02 | [`serve_and_measure`](notebooks/02_serve_and_measure.ipynb) | T0 / T1 / T3 | TTFT, ITL, TPOT, E2E, throughput and goodput as `vllm bench serve` defines them (one stated difference); histogram quantiles; open vs closed loop and the backlog an open loop builds; Little's law against the engine's gauges; warm-up, burstiness and long-tailed lengths | §11 | ~1.5 h |
| 03 | [`knobs_and_tradeoffs`](notebooks/03_knobs_and_tradeoffs.ipynb) | T0 / T1 (+T2) | why batching is nearly free; the latency-throughput knee; `max-num-batched-tokens` as a TTFT-versus-ITL-tail trade; choosing `max-num-seqs` by goodput; capacity at an SLO; KV blocks and preemption; tensor parallelism on two T4s | §2, §3, §9, §11 | ~3 h |
| 04 | [`prefix_caching_for_agents`](notebooks/04_prefix_caching_for_agents.ipynb) | T0 / T1 | block-hash chains; the hit accounting rules; hit rate from `/metrics`, predicted from the prompt layout before it is measured; prompt layouts that keep (or kill) the cache for agents; what a hit is worth in TTFT | §5 | ~2 h |
| 05 | [`speculation_and_quantization_in_vllm`](notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 (+T1 flags) | tokens per verify step `(1-a^(k+1))/(1-a)`; acceptance from vLLM's counters; why speculation fades at high batch and short context; what INT4 vs FP8 buys for prefill vs decode | §7, §8 | ~2 h |
| 06 | [`deploy_on_cloud_run_gpu`](notebooks/06_deploy_on_cloud_run_gpu.ipynb) | T3 (plannable at T0) | cold-start anatomy and the startup-probe budget; setting Cloud Run `concurrency` from a measurement, and what happens above `max-num-seqs`; cost per million tokens; scale-to-zero vs warm | §12 | ~2 h |

Everything runs on a laptop first: `servelab.fakeserver` is a **fake vLLM** — the same OpenAI-compatible streaming
API and the same `vllm:*` Prometheus metrics, over an engine emulator with continuous batching, a block-hash prefix
cache, preemption and a roofline step-time model. Its numbers are labelled *simulated*. Point the same code at a real
`vllm serve` and the numbers become measurements.

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | fake vLLM + all analysis | every notebook, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm serve` with a 0.5-1.5B model | `SERVELAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | two GPUs: Kaggle "GPU T4 x2" (free, PCIe) or a rented pair | `vllm serve --tensor-parallel-size 2` | notebook 03 exercise 3.6 (predicted at T0, measured with `SERVELAB_START_VLLM=1`), [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T3** | GCP | vLLM on Cloud Run (L4, scale to zero) or GKE (L4 Spot) | [`deploy/gcp/cloud-run/`](deploy/gcp/cloud-run/), [`deploy/gcp/gke/`](deploy/gcp/gke/) |

vLLM v0.30.0 needs compute capability 7.5 or newer: a T4 works, Kaggle's P100 does not. The minimal from-scratch
engine is next door in [`../mini-engine-core/`](../mini-engine-core/); this lab never imports it. Prices and where
to get GPUs: `COMPUTE.md` at the repo root.

## Run it

```bash
cd vllm-serving-lab
python3 -m pip install -e ".[dev]"             # aiohttp + prometheus_client; dev: pytest, jupyter, numpy, pyyaml
python3 -m pytest -q                           # 66 tests, a few seconds, offline
python3 -m servelab fake --port 8000 &         # a fake vLLM (simulated T4 + Qwen2.5-0.5B)
python3 -m servelab bench --url http://127.0.0.1:8000 --rate 5 -n 60 --slo-ttft-ms 300 --slo-tpot-ms 30
python3 -m servelab metrics --url http://127.0.0.1:8000
python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384 --quantization fp8
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

Measure a real engine instead (T1/T3): start one ([`deploy/any-gpu/`](deploy/any-gpu/) has a
Colab/Kaggle T4 recipe), then `export SERVELAB_URL=http://127.0.0.1:8000` (plus
`SERVELAB_API_KEY`, or `SERVELAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run
service). The notebooks detect it and measure it — a `servelab fake` server in `SERVELAB_URL` is
recognised by its `/version` and stays labelled simulated; with a GPU, vLLM installed and
`SERVELAB_START_VLLM=1`, notebook 03's sweeps restart a real `vllm serve` per configuration.


## The library (`servelab/`, ~3,000 lines)

| Module | Lines | The idea |
|---|---:|---|
| `sizing.py` | ~410 | `config.json` + GPU + flags -> weights, KV bytes/token, blocks, max concurrency, the fit check and vLLM's error text; parse and calibrate against the startup log. Ten bundled configs (Qwen2.5/3, Llama 3.x, Mistral, Mixtral) whose parameter counts match the published ones |
| `bench/` | ~590 | async OpenAI-compatible load generator: SSE timing (`client.py`), open loop with vLLM's gamma/Poisson arrivals, closed loop, agent sessions (`runner.py`), length distributions and shared-prefix agent workloads (`workload.py`), `vllm bench serve` definitions incl. goodput (`summary.py`), tables and text curves (`report.py`) |
| `metrics.py` | ~340 | parse the Prometheus text format; PromQL-exact `histogram_quantile`; windows between scrapes; an engine snapshot (queue, batch, KV, hit rate, preemptions, spec acceptance); the PromQL for each |
| `tune.py` | ~250 | render vLLM CLI flags; sweep configs on a pluggable backend (`FakeBackend`, `VLLMBackend`, `URLBackend`); best config under an SLO; highest rate under an SLO |
| `fake_engine.py` | ~540 | the emulator: FCFS continuous batching with a token budget and chunked prefill, block pool with refcounts, chained block hashes and LRU reuse, recompute preemption, roofline step time, Bernoulli speculative acceptance; named hardware profiles |
| `fakeserver.py` | ~360 | the fake vLLM over HTTP (aiohttp): `/v1/completions`, `/v1/chat/completions` (streaming, usage, cached tokens), `/v1/models`, `/health`, `/metrics` with vLLM's names and bucket edges |
| `env.py`, `textgen.py`, `__main__.py` | ~360 | tier detection (`SERVELAB_URL`, GPU, vLLM), a toy tokenizer shared by client and server, the CLI |

Metric names and histogram buckets are those of `vllm/v1/metrics/loggers.py`, `buckets.py` and
`vllm/v1/spec_decode/metrics.py` in
vLLM v0.30.0 and main (Sep 2026): `vllm:num_requests_running`, `vllm:num_requests_waiting`,
`vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries`/`_hits`, `vllm:num_preemptions`,
`vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`,
`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`,
`vllm:request_decode_time_seconds`, `vllm:request_success`, `vllm:iteration_tokens_total`,
`vllm:spec_decode_num_*` (counters carry a `_total` suffix on the wire).

## Measurement hygiene (what the code does for you, and why)

* **Definitions match `vllm bench serve`** — TTFT to the first token-carrying chunk, ITL between
  token-carrying chunks, TPOT = (E2E − TTFT)/(n − 1) with n from `usage`, throughput over the run's
  wall time, goodput = requests meeting every SLO per second. The one deliberate difference: the
  chat role-only chunk vLLM sends just before the first token is skipped. `vllm bench serve` counts
  it as a chunk (it sets TTFT and adds one ~0 ms ITL gap per chat request); here TTFT is the same
  and chat ITL lists have one fewer, near-zero entry. Completions are identical.
* **Open loop for capacity**, closed loop only for "N users" questions; stagger closed-loop users
  (`ramp_s`) so they do not start as one synchronized burst.
* **Warm-up with different prompts** than the measured ones, and **before** the "before" scrape —
  otherwise warm-up leaks prefix-cache hits and requests into the window.
* **Same seeded workload** for every configuration you compare; lengths fixed with `ignore_eos`;
  fresh prompts per run (a replayed prompt hits the prefix cache); and say which arrival process
  (Poisson or `burstiness`) and length distribution (fixed, lognormal) a capacity number assumes.
* **Histogram percentiles are interpolations** inside vLLM's bucket edges; `_sum/_count` means are
  exact; counters only mean something as differences between two scrapes.
* **Label the source**: every report says SIMULATED (fake server) or measured-on-URL, and the fake
  server says so on `/version` even when it is reached through `SERVELAB_URL`.

The fake server's upstream counterpart is [`llm-d-inference-sim`](https://github.com/llm-d/llm-d-inference-sim)
(Go, OpenAI-compatible, vLLM metrics, fixed or per-token latency parameters), used by llm-d for
routing experiments in layer 05. This lab's emulator instead *derives* latency from a scheduler, a
KV cache and a roofline, so the flags have their real effects.

## Deploy

[`deploy/`](deploy/) — `any-gpu/` (docker or pip `vllm serve`, the flags explained, Colab/Kaggle
T4 recipe, RunPod/Vast notes), `gcp/cloud-run/` (Terraform `google_cloud_run_v2_service` with one
L4, scale to zero, weights from Hugging Face or a GCS mount, HF token from Secret Manager; plus the
`gcloud run deploy` equivalent), `gcp/gke/` (Deployment on an L4 Spot pool, `PodMonitoring` for
Managed Prometheus). Each has a README with cost and cleanup. This lab's Terraform is the Cloud Run
service (`deploy/gcp/cloud-run/terraform/`); GKE here is a `gcloud` script plus manifests, and the
cluster as Terraform lives in layer 03's
[`k8s-gpu-lab`](../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) and layer 05's
[`inference-gateway-lab`](../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/).

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py` (percent format
with `### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Simulated vs measured.** The fake server's numbers are simulated and every report says so (it also answers
  `/version` as a fake, even behind `SERVELAB_URL`). Only a real `vllm serve` gives measurements.
- **Sizing is an estimate.** `sizing.size()` models vLLM v0.30.0's defaults (0.92 of the driver-reported total,
  profiled activation, CUDA-graph and non-torch overheads estimated); the startup log's `Available KV cache memory`
  is the measurement, and notebook 01 calibrates against it. [`../PRIMER.md`](../PRIMER.md) §4 compares these inputs
  with the mini engine's round ones.
- **Checked by construction.** The deploy paths are checked with `bash -n`, `DRY_RUN=1`, Terraform `validate` and
  Kubernetes schema checks, not run on real infrastructure here.

## Verify list (facts dated Sep 2026 that move)

Checked against the vLLM **v0.30.0** source and Docker Hub on 2026-09-26 (re-check when you move
the pin):

* vLLM **0.30.0** is the latest release (PyPI, 2026-09-22); image `vllm/vllm-openai:v0.30.0` exists
  (amd64 and arm64), built on CUDA 13.0.3 with `ENTRYPOINT ["vllm", "serve"]`.
* Defaults: `gpu_memory_utilization` **0.92** (0.9 in older releases), `block_size` 16, prefix
  caching and chunked prefill on; for `vllm serve`, `max_num_batched_tokens` / `max_num_seqs` are
  2048 / 256 below 70 GiB or on A100, 8192 / 1024 on other GPUs from 70 GiB, 16384 / 1024 from 160 GiB.
* KV budget = `util × total − (weights + profiled activation peak + non-torch) − CUDA-graph
  estimate` (CUDA-graph profiling on by default since v0.21.0); one block is reserved as the null block.
* `--enable-prompt-tokens-details` adds `usage.prompt_tokens_details.cached_tokens`;
  `--max-model-len -1` auto-fits; `--attention-backend` replaces `VLLM_ATTENTION_BACKEND`, and
  `VLLM_USE_V1` is gone; `VLLM_BATCH_INVARIANT=1` turns on batch-invariant mode.
* On compute capability 7.5 (T4) the attention backend is `TRITON_ATTN` (FlashAttention and
  FlashInfer need sm_80+ in this release); `bfloat16` is refused below sm_80; kernels are built for
  sm_75 and newer only (no P100/V100).
* Startup-log lines parsed by `sizing.parse_startup_log` ("Model loading took", "Available KV cache
  memory", "GPU KV cache size ... Maximum concurrency", "init engine ... took", "The current
  --gpu-memory-utilization=").
* Prefix-cache counters exclude re-admitted preempted requests (they go to separate `preempted_*`
  stats); `speculative_config` methods `ngram`, `draft_model`, `eagle3`; the synthetic
  rejection-sampling mode for load tests.

Still to verify (not checkable from source here):

* CUDA 13's minimum driver (580 series) and the drivers Colab and Kaggle ship; the last vLLM release
  built for CUDA 12.
* How far CUDA's reported total memory sits below nvidia-smi's on each GPU (it moves sizing by ~1
  session on an L4).
* Cloud Run GPU: L4 (24 GB, min 4 vCPU / 16 GiB), GA since June 2025; flag names `--gpu`,
  `--gpu-type`, `--no-gpu-zonal-redundancy`; regions, quota names, prices, CPU-always-allocated for
  GPU services, startup-probe limits and whether probing starts after the image pull.
* GKE: GPU node taint `nvidia.com/gpu=present:NoSchedule`; `HF_XET_HIGH_PERFORMANCE` in the image's
  `huggingface_hub`.
* GPU datasheet numbers in `sizing.GPUS` (memory as reported by the driver, dense TFLOPS, bandwidth).

MIT licensed.
