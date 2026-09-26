# vllm-serving-lab

**Serve, measure, size and tune a real inference engine.** The concepts of this topic's
[primer](../PRIMER.md) — continuous batching, the token budget, prefix caching, speculation,
quantization, measuring an engine — applied to **vLLM** through the four things an engineer
actually does with it: size it before paying for a GPU, measure it the way everyone else does,
read its own `/metrics`, and choose its flags against an SLO. The minimal from-scratch engine is
next door in [`../mini-engine-core/`](../mini-engine-core/); this lab never imports it.

Everything runs on a laptop first: `servelab.fakeserver` is a **fake vLLM** — the same
OpenAI-compatible streaming API and the same `vllm:*` Prometheus metrics, over an engine emulator
with continuous batching, a block-hash prefix cache, preemption and a roofline step-time model. Its
numbers are labelled *simulated*. Point the same code at a real `vllm serve` (Colab T4, a rented
GPU) or at Cloud Run / GKE, and the numbers become measurements.

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | fake vLLM + all analysis | every notebook, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm serve` with a 0.5-1.5B model | `SERVELAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T3** | GCP | vLLM on Cloud Run (L4, scale to zero) or GKE (L4 Spot) | [`deploy/gcp/cloud-run/`](deploy/gcp/cloud-run/), [`deploy/gcp/gke/`](deploy/gcp/gke/) |

(T2, multi-GPU, belongs to layers 01-02; prices and where to get GPUs: [`COMPUTE.md`](../../../COMPUTE.md).)

## Quick start (T0, no GPU)

```bash
cd vllm-serving-lab
python3 -m pip install -e ".[dev]"             # aiohttp + prometheus_client; dev: pytest, jupyter, numpy, pyyaml
python3 -m pytest -q                           # ~60 tests, a few seconds, offline
python3 -m servelab fake --port 8000 &         # a fake vLLM (simulated T4 + Qwen2.5-0.5B)
python3 -m servelab bench --url http://127.0.0.1:8000 --rate 5 -n 60 --slo-ttft-ms 300 --slo-tpot-ms 30
python3 -m servelab metrics --url http://127.0.0.1:8000
python3 -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384 --quantization fp8
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

Measure a real engine instead (T1/T3): start one ([`deploy/any-gpu/`](deploy/any-gpu/) has a
Colab/Kaggle T4 recipe), then `export SERVELAB_URL=http://127.0.0.1:8000` (plus
`SERVELAB_API_KEY`, or `SERVELAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run
service). The notebooks detect it and measure it; with a GPU, vLLM installed and
`SERVELAB_START_VLLM=1`, notebook 03's sweeps restart a real `vllm serve` per configuration.

## The notebooks

Each opens with *the one-minute version*, works examples against the library, then 3-6 exercises
(implement the key function, predict a number, pick a setting) each followed by a check that prints ✅,
and closes with *in a design review*: a two-minute walkthrough and drill questions with answers.

| # | Notebook | Tier | You will be able to explain | Primer |
|---|---|---|---|---|
| 01 | `size_before_you_serve` | T0 (+T1 log) | KV bytes per token from `config.json`; blocks and "Maximum concurrency"; why an 8B model's 128K context does not start on an L4; what FP8 buys; calibrating from the startup log | §4 |
| 02 | `serve_and_measure` | T0 / T1 / T3 | TTFT, ITL, TPOT, E2E, throughput and goodput exactly as `vllm bench serve` defines them; histogram quantiles; open vs closed loop; Little's law against the engine's gauges | §11 |
| 03 | `knobs_and_tradeoffs` | T0 / T1 | why batching is nearly free; the latency-throughput knee; `max-num-batched-tokens` as a TTFT-versus-ITL-tail trade; choosing `max-num-seqs` by goodput; capacity at an SLO; KV blocks and preemption | §2, §3, §11 |
| 04 | `prefix_caching_for_agents` | T0 / T1 | block-hash chains; the hit accounting rules; hit rate from `/metrics`; prompt layouts that keep (or kill) the cache for agents; what a hit is worth in TTFT | §5 |
| 05 | `speculation_and_quantization_in_vllm` | T0 (+T1 flags) | tokens per verify step `(1-a^(k+1))/(1-a)`; acceptance from vLLM's counters; why speculation fades at high batch; what INT4 vs FP8 buys for prefill vs decode | §7, §8 |
| 06 | `deploy_on_cloud_run_gpu` | T3 (plannable at T0) | cold-start anatomy; setting Cloud Run `concurrency` from a measurement; cost per million tokens; scale-to-zero vs warm | §12 |

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

Metric names and histogram buckets are those of `vllm/v1/metrics/loggers.py` and `buckets.py` on
vLLM main (Sep 2026): `vllm:num_requests_running`, `vllm:num_requests_waiting`,
`vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries`/`_hits`, `vllm:num_preemptions`,
`vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`,
`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`,
`vllm:request_decode_time_seconds`, `vllm:request_success`, `vllm:iteration_tokens_total`,
`vllm:spec_decode_num_*` (counters carry a `_total` suffix on the wire).

## Measurement hygiene (what the code does for you, and why)

* **Definitions match `vllm bench serve`** — TTFT to the first token chunk, ITL between chunks,
  TPOT = (E2E − TTFT)/(n − 1), throughput over the run's wall time, goodput = requests meeting every
  SLO per second. The one deliberate difference: chat role-only chunks are not counted as tokens.
* **Open loop for capacity**, closed loop only for "N users" questions; stagger closed-loop users
  (`ramp_s`) so they do not start as one synchronized burst.
* **Warm-up with different prompts** than the measured ones, and **before** the "before" scrape —
  otherwise warm-up leaks prefix-cache hits and requests into the window.
* **Same seeded workload** for every configuration you compare; lengths fixed with `ignore_eos`.
* **Histogram percentiles are interpolations** inside vLLM's bucket edges; `_sum/_count` means are
  exact; counters only mean something as differences between two scrapes.
* **Label the source**: every report says SIMULATED (fake server) or measured-on-URL.

The fake server's upstream counterpart is [`llm-d-inference-sim`](https://github.com/llm-d/llm-d-inference-sim)
(Go, OpenAI-compatible, vLLM metrics, fixed or per-token latency parameters), used by llm-d for
routing experiments in layer 05. This lab's emulator instead *derives* latency from a scheduler, a
KV cache and a roofline, so the flags have their real effects.

## Deploy

[`deploy/`](deploy/) — `any-gpu/` (docker or pip `vllm serve`, the flags explained, Colab/Kaggle
T4 recipe, RunPod/Vast notes), `gcp/cloud-run/` (Terraform `google_cloud_run_v2_service` with one
L4, scale to zero, weights from Hugging Face or a GCS mount, HF token from Secret Manager; plus the
`gcloud run deploy` equivalent), `gcp/gke/` (Deployment on an L4 Spot pool, `PodMonitoring` for
Managed Prometheus). Each has a README with cost and cleanup.

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py` (percent format
with `### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Verify list (facts dated Sep 2026 that move)

* vLLM **0.30.0** is the latest release (PyPI, 2026-09-22); image tag `vllm/vllm-openai:v0.30.0` (verify it exists).
* vLLM main defaults: `gpu_memory_utilization` **0.92** (0.9 in older releases), `block_size` 16,
  prefix caching and chunked prefill on; for `vllm serve` on GPUs under 70 GiB
  `max_num_batched_tokens` 2048 and `max_num_seqs` 256 (8192/1024 on H100-class, 16384/1024 at >= 160 GiB).
* `--enable-prompt-tokens-details` adds `usage.prompt_tokens_details.cached_tokens`; `--max-model-len -1` auto-fits.
* Cloud Run GPU: L4 (24 GB, min 4 vCPU / 16 GiB), GA since June 2025; flag names `--gpu`,
  `--gpu-type`, `--no-gpu-zonal-redundancy`; regions, quota names and prices (verify).
* GPU datasheet numbers in `sizing.GPUS` (memory as reported by the driver, dense TFLOPS, bandwidth).

MIT licensed.
