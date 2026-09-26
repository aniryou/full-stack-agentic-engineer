# inference-gateway-lab

**Route LLM requests across engine replicas the way the llm-d Endpoint Picker does — first in one
Python process, then on kind, then behind the GKE Inference Gateway.** Package `igwlab`.

The lab is the detailed companion of the topic primer ([`../PRIMER.md`](../PRIMER.md)); the minimal,
simulation-only companion is [`../orchestrator-core/`](../orchestrator-core/). It is independent of
both: nothing here imports the core.

- `igwlab/router/` — a real async HTTP router in front of N OpenAI-compatible backends: it scrapes
  each backend's `/metrics` (vLLM names), keeps an approximate prefix index, runs
  filters → weighted scorers → picker configured by an `EndpointPickerConfig` YAML, streams
  responses byte for byte, and exports its own metrics. A readable re-implementation of the EPP's
  decision logic (not an ext-proc server).
- `igwlab/fakebackend.py` — an OpenAI-compatible backend with vLLM-shaped *emulated* timing, a
  vLLM-style prefix cache, and vLLM metric names. The T0 stand-in for a GPU.
- `igwlab/autoscale.py` — the Kubernetes HPA recommender ported line by line from
  kube-controller-manager v1.34, plus an HPA manifest generator and a simulated pool with cold starts.
- `igwlab/bench.py` — a shared-prefix, multi-turn agentic workload and a streaming load generator.
- `deploy/` — docker compose, kind + llm-d Router, and GKE (Terraform + manifests).

## Quick start (T0: laptop or Colab CPU, no GPU, no Docker)

```bash
cd inference-gateway-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                     # 64 tests, a few seconds, offline
python3 -m jupyterlab notebooks          # exercises; worked answers in solutions/
```

```python
from igwlab.stack import LocalStack
from igwlab.bench import agentic_sessions, compare

sessions = agentic_sessions(n_sessions=18, turns=5)
results = []
for policy in ("round-robin", "default-weighted"):
    with LocalStack(n_backends=3, config=policy) as s:     # 3 fake backends + router on free ports
        results.append(s.bench(sessions, label=policy))
print(compare(results))                                   # TTFT, hit rate, per-replica split
```

Latencies measured against the fake backend are real wall-clock HTTP on your machine, but the
backend's prefill/decode timing is **emulated** (`EngineProfile`): use them to compare policies,
never as GPU numbers.

## Notebooks

| # | Notebook | Tier | You will be able to explain | Primer |
|---|---|---|---|---|
| 01 | `01_router_in_process` | T0 | prefix hashing and the index, queue/KV scores, the weighted pick; why round-robin loses on agent traffic (measured) | §1 Why a layer above the engine, §2 Routing signals and algorithms |
| 02 | `02_scorer_weights_and_hot_prefixes` | T0 | the locality-vs-load trade-off in numbers, hot prefixes, the scrape-lag herd, choosing an affinity threshold, saturation shedding | §2, §3 Flow control and priorities |
| 03 | `03_autoscaling_recommender` | T0 | the exact HPA algorithm (tolerance, stabilization, policies, unready pods); which vLLM signals to scale on and why queue alone collapses | §4 Autoscaling |
| 04 | `04_local_stack_with_llm_d` | T0 walkthrough / T1-local (Docker, kind; CPU) | InferencePool semantics, the llm-d Router standalone mode, the simulator's latency model; benchmarks a running stack if there is one | §9 The Kubernetes-native stack, September 2026, §10 Where to run it |
| 05 | `05_gke_inference_gateway` | T3 (offline plan/inspect is T0) | the GKE Inference Gateway object graph, CRD validation, GMP → HPA plumbing, what an hour costs | §9, §10 |

Every notebook opens with "The one-minute version", works examples, has 4–5 exercises each followed
by a check (✅), and ends with "In a design review". Blanks are in `notebooks/`, answers in
`solutions/`, sources in `notebooks_src/` (percent format). Colab: the first cell clones the repo,
changes into this directory and `pip install -e .`s it.

## The library

| Module | Lines | The one idea |
|---|---|---|
| `router/tokens.py` | ~60 | a router "tokenizes" by packing request bytes into 4-byte pseudo-tokens (the EPP's `estimate` backend) |
| `router/prefix.py` | ~210 | chained block hashes + a per-endpoint LRU index; the match is the count of leading blocks held |
| `router/datalayer.py` | ~160 | scraped vLLM metrics (lagged) vs router-local in-flight counters (instant, partial) |
| `router/plugins.py` | ~470 | producers, filters, scorers, pickers with upstream types, parameters and formulas |
| `router/config.py` | ~260 | `EndpointPickerConfig`: parse, validate, inject upstream defaults, export for the real EPP |
| `router/scheduler.py` | ~110 | one scheduling cycle and an explainable per-scorer decision table |
| `router/server.py` | ~280 | the proxy: admission (shed priority < 0 at saturation), dispatch, streaming, metrics |
| `fakebackend.py` | ~540 | a replica is a cache with a queue: TTFT = queueing + prefill of *uncached* tokens |
| `autoscale.py` | ~430 | the HPA as a proportional controller with guard rails, exactly |
| `bench.py` | ~310 | agent traffic is prefix-heavy; measure TTFT, hit rate and the per-replica split |
| `k8s.py` | ~210 | the Gateway-mode object graph, validated offline against upstream CRD schemas |
| `stack.py` | ~120 | backends + router in one process on free ports, usable from notebooks and tests |
| `promtext.py` | ~310 | Prometheus text, counters vs gauges, `histogram_quantile` |

### How the router maps to llm-d (Sep 2026, llm-d-router v0.10)

| Upstream | Here | Notes |
|---|---|---|
| Envoy (standalone) or cloud L7 LB (Gateway mode) + ext-proc | `router/server.py` | one process proxies *and* picks; response header `x-gateway-destination-endpoint` |
| `token-producer` (`estimate`) | `router/tokens.py` | same 4-byte packing of tools + role + content |
| `approx-prefix-cache-producer` | `ApproxPrefixCacheProducer` | chained 64-bit hashes seeded by model + cache salt, block size ≥ 64 tokens, LRU 31,250/endpoint, tail-first insert, greedy match |
| `metrics-data-source` + `core-metrics-extractor` | `router/datalayer.py` | `vllm:num_requests_waiting/running`, `kv_cache_usage_perc`, `lora_requests_info`, `cache_config_info`; 50 ms default |
| `prefix-cache-scorer`, `queue-scorer`, `kv-cache-utilization-scorer`, `running-requests-size-scorer`, `active-request-scorer`, `token-load-scorer`, `lora-affinity-scorer` | `router/plugins.py` | formulas pinned in `tests/test_plugins.py` |
| `utilization-filter`, `prefix-cache-affinity-filter` | `router/plugins.py` | affinity filter supports `ttftSource: prefillThroughput` only |
| `max-score-picker`, `random-picker`, `weighted-random-picker` | `router/plugins.py` | ties rotate by a request counter over a name-sorted list, so *no scorers* = round-robin here; upstream's list is map-ordered, so its ties are effectively random |
| legacy admission + `utilization-detector` | `Router.pool_saturation` | priority < 0 → 429 when saturation ≥ 1.0 (the lab also sets `x-llm-d-request-dropped-reason: rejected-saturated`, the reason string flow control uses) |
| `InferenceObjective` priority | `RouterSettings.objectives` | selected by header `x-llm-d-inference-objective` |

Deliberate differences: one scheduling profile only (no P/D disaggregation); flow-control *queueing*
(`featureGates: [flowControl]`) is accepted but not implemented (the legacy shedding applies); the
prefix index may take a lab-only `ttlSeconds` (stripped by `PickerConfig.to_upstream()`); auto-tuned
LRU capacity is converted to index-block units (upstream uses the engine's block count directly).

Presets (`igwlab/configs/`, loadable by name): `round-robin`, `default-weighted` (the Helm chart
default: prefix ×3, queue ×2, KV ×2), `prefix-only`, `load-only`, `queue-only`, `active-requests`,
`sticky-until-saturated` (the optimized-baseline shape, calibrated for the fake backend).

Router metrics (`GET /metrics`) mirror EPP series under an `igw_` prefix: `igw_request_total`
(`llm_d_epp_request_total`), `igw_request_ttft_seconds`, `igw_average_queue_size`,
`igw_average_kv_cache_utilization`, `igw_ready_endpoints`, `igw_scheduler_attempts_total`,
`igw_prefix_indexer_size`, `igw_prefix_indexer_hit_ratio`, plus `igw_pool_saturation`.
`GET /debug/state` returns the endpoints, the prefix index size per endpoint and the last decisions.

## Run the pieces as processes

```bash
python3 -m igwlab.fakebackend --name a --port 8001 &
python3 -m igwlab.fakebackend --name b --port 8002 &
python3 -m igwlab.router --config default-weighted --port 9000 \
    --backend a=http://127.0.0.1:8001 --backend b=http://127.0.0.1:8002 --objective batch=-10
curl -s localhost:9000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"lab/llm","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

## Deploy paths

| Path | Tier | What runs | Cost | README |
|---|---|---|---|---|
| `deploy/local` | T0/T1 (Docker, CPU) | 3 × `llm-d-inference-sim` (or the fake) + the lab router + Prometheus | free | [`deploy/local/README.md`](deploy/local/README.md) |
| `deploy/kind` | T1-local (Docker, kind, CPU) | 3 simulator pods + **llm-d Router** standalone (EPP + Envoy) via Helm | free | [`deploy/kind/README.md`](deploy/kind/README.md) |
| `deploy/gcp/terraform` | T3 | zonal GKE, Gateway API, proxy-only subnet, L4 Spot pool 0→2, Managed Prometheus | ~$0.16/h idle, ~$0.44/h with one L4 Spot node (assumed prices, verify) | [`deploy/gcp/terraform/README.md`](deploy/gcp/terraform/README.md) |
| `deploy/gke` | T3 | vLLM on L4, EPP + InferencePool + objectives (Helm), Gateway + HTTPRoute, HPA on vLLM's queue | as above | [`deploy/gke/README.md`](deploy/gke/README.md) |

Prices and GPU availability for GCP and non-GCP options: [`../../../COMPUTE.md`](../../../COMPUTE.md).
None of the Docker, kind or GKE paths were executed while this lab was written (no Docker daemon or
cloud access); they are correct by construction: `bash -n`, `DRY_RUN=1`, Terraform `validate`,
`kubernetes-validate --strict -k 1.34.0` for core kinds, and CRD-schema checks for the rest
(`tests/test_manifests.py`).

## Pinned versions (verify before relying on them; Sep 2026)

| Component | Pin | Where |
|---|---|---|
| llm-d-router (EPP image, charts, InferenceObjective CRD) | v0.10.0 (charts `--version v0.10.0`; the guides use the floating `v0`) | kind/gke values and scripts |
| Gateway API Inference Extension (InferencePool v1 CRD) | v1.6.2 | kind/gke scripts; GKE ≥ 1.34.0-gke.1626000 manages it |
| Gateway API (Gateway, HTTPRoute schemas) | v1.6.2 standard | `igwlab/crds/` |
| llm-d-inference-sim | v0.11.2 | compose, kind |
| vLLM | `vllm/vllm-openai:v0.29.0`, model `Qwen/Qwen2.5-1.5B-Instruct` | gke/vllm.yaml |
| Prometheus | v3.15.0 | compose |
| kind node image | `kindest/node:v1.34.0` | kind-config.yaml |
| Terraform / google provider | ≥ 1.9 / ≥ 8.0 (validated with 8.4.0) | gcp/terraform |
| GKE Gateway classes | `gke-l7-regional-external-managed` (internal: `gke-l7-rilb`) | gke/gateway.yaml |
| HPA metric name via the Custom Metrics Stackdriver Adapter | `prometheus.googleapis.com\|vllm:num_requests_waiting\|gauge` | gke/hpa.yaml |

## Regenerating notebooks and snapshots

```bash
python3 tools/build_notebooks.py                        # notebooks/ and solutions/ from notebooks_src/
python3 tools/run_notebooks.py solutions                # must run clean (CPU, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
python3 tools/snapshot_crds.py <crd.yaml> <version> "<source>" ...   # refresh igwlab/crds/ from upstream
```

MIT licensed.
