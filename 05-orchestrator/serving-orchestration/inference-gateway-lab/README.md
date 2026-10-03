# inference-gateway-lab — route LLM requests the way the llm-d endpoint picker does, then run it for real

Run a real async router in front of OpenAI-compatible engines. The router makes the same decisions as the llm-d
endpoint picker: filters, weighted scorers and a picker. Run it in three places, in this order:

1. In one Python process, against emulated backends.
2. On kind, with the real llm-d Router.
3. Behind GKE Inference Gateway.

The package is `igwlab`.

## Start here

1. Run `python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q`. The 84
   tests take about 30 s, offline.
2. Run the [comparison under "Run it"](#run-it). It puts three emulated backends behind the router. It compares
   round-robin with the llm-d chart default. In about 4 s, it prints TTFT percentiles, hit rate and the per-replica
   split.
3. Open [`notebooks/01_router_in_process.ipynb`](notebooks/01_router_in_process.ipynb). Keep
   [PRIMER §2](../PRIMER.md#2-routing-signals-and-algorithms) open beside it.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* "T0 + Docker" is
a laptop with Docker, also at no cost.

| Notebook | You will be able to explain… | Time | Tier | Primer |
|---|---|---|---|---|
| [`01_router_in_process`](notebooks/01_router_in_process.ipynb) | Prefix hashing and the index, queue/KV scores, the weighted selection. Why round-robin loses on agent traffic (measured). | ~1.5 h | T0 | §1 Why a layer above the engine, §2 Routing signals and algorithms |
| [`02_scorer_weights_and_hot_prefixes`](notebooks/02_scorer_weights_and_hot_prefixes.ipynb) | The trade-off between locality and load, in numbers. Hot prefixes, the scrape-lag herd, how to select an affinity threshold, the rejection of requests at saturation. | ~1.5 h | T0 | §2, §3 Flow control and priorities |
| [`03_autoscaling_recommender`](notebooks/03_autoscaling_recommender.ipynb) | The HPA controller where the basic rule is not sufficient (absent pods and pods that start, legacy against `behavior`, live scrapes). Which vLLM signals to scale on, why queue alone collapses, and targets that come from batch slots and Little's law. | ~1.5 h | T0 | §4 Autoscaling |
| [`04_local_stack_with_llm_d`](notebooks/04_local_stack_with_llm_d.ipynb) | InferencePool semantics and the standalone mode of the llm-d Router. What the EPP reads and the lab router does not read, and the latency model of the simulator. The notebook measures a stack that runs, or the lab router in front of real vLLM. | ~1.5 h | T0 walkthrough, T0 + Docker (compose or kind), **T1/T2** with a GPU (real vLLM, `deploy/any-gpu`) | §9 The Kubernetes-native stack, September 2026, §10 Where to run it |
| [`05_gke_inference_gateway`](notebooks/05_gke_inference_gateway.ipynb) | The object graph of GKE Inference Gateway, CRD validation, the infrastructure code from GMP to the HPA, the cost of an hour. | ~2 h | T3 (the offline plan and inspection are T0) | §9, §10 |

The times are approximate. The curriculum of the repo ([`CURRICULUM.md`](../../../CURRICULUM.md)) gives about 6 hours to
notebooks 01–04 and 2 to 05. Each notebook starts with "The one-minute version" and then gives worked examples. It has
4–5 exercises, and a check (✅) comes after each exercise. It ends with "In a design review". The blanks are in
`notebooks/`, the answers are in [`solutions/`](solutions/), and the sources are in `notebooks_src/` (percent format).

The code behind the notebooks:

| Path | What it is |
|---|---|
| [`igwlab/router/`](igwlab/router/) | A real async HTTP router in front of N OpenAI-compatible backends. It scrapes the `/metrics` of each backend (vLLM names) and keeps an approximate prefix index. It runs filters, then weighted scorers, then a picker, as an `EndpointPickerConfig` YAML configures them. It sends the responses as a stream, byte for byte, and exports its own metrics. It is a readable re-implementation of the decision logic of the EPP (not an ext-proc server). |
| [`igwlab/fakebackend.py`](igwlab/fakebackend.py) | An OpenAI-compatible backend with an *emulated* latency model in the shape of vLLM, a vLLM-style prefix cache, and vLLM metric names. At T0, it takes the place of a GPU. |
| [`igwlab/autoscale.py`](igwlab/autoscale.py) | The Kubernetes HPA recommender, ported line by line from kube-controller-manager v1.34. Also an HPA manifest generator and a simulated pool with cold starts. |
| [`igwlab/bench.py`](igwlab/bench.py) | A shared-prefix, multi-turn agentic workload and a streaming load generator. |
| [`deploy/`](#deploy-paths) | docker compose (CPU), kind + llm-d Router (CPU), real vLLM on any GPU box, and GKE (Terraform + manifests). |

This lab is the detailed companion of the topic primer ([`../PRIMER.md`](../PRIMER.md)). The minimal companion, with
simulation only, is [`../orchestrator-core/`](../orchestrator-core/). This lab is independent of both: no code here
imports the core.

## Run it

```bash
cd inference-gateway-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                     # 84 tests, ~30 s, offline
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

On Colab, the first cell of each notebook clones the repo and goes into this directory. Then it runs
`pip install -e .` on it. The one-time setup is in [`COLAB.md`](../../../COLAB.md) of the repository.

**The pieces as processes:**

```bash
python3 -m igwlab.fakebackend --name a --port 8001 &
python3 -m igwlab.fakebackend --name b --port 8002 &
python3 -m igwlab.router --config default-weighted --port 9000 \
    --backend a=http://127.0.0.1:8001 --backend b=http://127.0.0.1:8002 --objective batch=-10
curl -s localhost:9000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"lab/llm","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

### Deploy paths

| Path | Tier | What runs | Cost | README |
|---|---|---|---|---|
| `deploy/local` | T0 + Docker (CPU) | 3 × `llm-d-inference-sim` (or the fake) + the lab router + Prometheus | free | [`deploy/local/README.md`](deploy/local/README.md) |
| `deploy/kind` | T0 + Docker (CPU, kind) | 3 simulator pods + **llm-d Router** standalone (EPP + Envoy) through Helm | free | [`deploy/kind/README.md`](deploy/kind/README.md) |
| `deploy/any-gpu` | T1/T2 (real GPU, no cloud account) | 2+ real vLLM replicas (`Qwen2.5-0.5B-Instruct`) from pip or Docker. The lab router is in front. The T1 cells of notebook 04 measure them. | Colab/Kaggle T4 free, rented 24 GB GPU ~$0.3–0.7/h (verify) | [`deploy/any-gpu/README.md`](deploy/any-gpu/README.md) |
| `deploy/gcp/terraform` | T3 | zonal GKE, Gateway API, proxy-only subnet, L4 Spot pool from 0 to 2, Managed Prometheus | ~$0.16/h with the GPU pool at 0, ~$0.44/h with one L4 Spot node (assumed prices, verify) | [`deploy/gcp/terraform/README.md`](deploy/gcp/terraform/README.md) |
| `deploy/gke` | T3 | vLLM on L4, EPP + InferencePool + objectives (Helm), Gateway + HTTPRoute, an HPA on the waiting **and** running requests of vLLM | ~$0.44/h while installed, also when idle (`minReplicas: 1` keeps one L4 node) | [`deploy/gke/README.md`](deploy/gke/README.md) |

Prices and GPU availability for GCP and non-GCP options: [`COMPUTE.md`](../../../COMPUTE.md).

## How the router maps to llm-d (llm-d-router v0.10.0, the pinned release)

| Upstream | Here | Notes |
|---|---|---|
| Envoy (standalone) or cloud L7 LB (Gateway mode) + ext-proc | `router/server.py` | One process is the proxy *and* the picker. The response header is `x-gateway-destination-endpoint`. |
| `token-producer` (`estimate`) | `router/tokens.py` | It packs tools + role + content in 4-byte units, the same as upstream. |
| `approx-prefix-cache-producer` | `ApproxPrefixCacheProducer` | Chained 64-bit hashes with model + cache salt as the seed, block size ≥ 64 tokens, LRU 31,250/endpoint, greedy match. The lab inserts tail-first (see "Tail-first prefix insertion" after this table). |
| `metrics-data-source` + `core-metrics-extractor` | `router/datalayer.py` | `vllm:num_requests_waiting/running`, `kv_cache_usage_perc`, `lora_requests_info`, `cache_config_info`. The default is 50 ms. |
| `prefix-cache-scorer`, `queue-scorer`, `kv-cache-utilization-scorer`, `running-requests-size-scorer`, `active-request-scorer`, `token-load-scorer`, `lora-affinity-scorer` | `router/plugins.py` | `tests/test_plugins.py` pins the formulas. As upstream, an endpoint that the router never scraped has zero metrics, and thus looks idle. `token-load-scorer` adds the uncached tokens of the request itself to each endpoint. |
| `inflight-load-producer` | `InFlightLoadProducer` | Uncached tokens = (total − matched blocks) × block size + tail, as upstream. |
| `utilization-filter`, `prefix-cache-affinity-filter` | `router/plugins.py` | The affinity filter supports only `ttftSource: prefillThroughput`. |
| `max-score-picker`, `random-picker`, `weighted-random-picker` | `router/plugins.py` | See "Ties rotate" after this table. |
| legacy admission + `utilization-detector` | `Router.pool_saturation` | When saturation ≥ 1.0, a request with priority < 0 gets 429. The lab also sets `x-llm-d-request-dropped-reason: rejected-saturated`, the reason string that flow control uses. |
| `InferenceObjective` priority | `RouterSettings.objectives` | The header `x-llm-d-inference-objective` selects it. |

Deliberate differences. The module that implements each difference also states it:

- **One scheduling profile** only. There is no P/D disaggregation.
- **No flow control.** The lab accepts `featureGates: [flowControl]` with a warning. The loader, the
  CLI and `Router` all print the warning, and `describe()` shows it. There are no router-side queues,
  priority bands or TTLs. The legacy shedding applies. With the gate on, the real EPP puts requests in
  queues by priority.
- **Ties rotate.** `max-score-picker` rotates each tie with a per-request counter, over a list sorted
  by name. Thus *no scorers* = exact round-robin. Before a stable sort, v0.10.0 puts the candidates in
  a random order, so its ties are uniformly random. On main, after v0.10.0, the picker rotates over a
  map-ordered list.
- **Tail-first prefix insertion.** The lab inserts the block hashes of a prompt tail-first. Thus the
  index evicts the head of a prompt last, and the greedy match never counts too few blocks. This is
  the behavior of llm-d-router *main* after v0.10.0. v0.10.0 inserts head-first. Under LRU pressure,
  it evicts heads first, and its greedy scan can report 0 for an endpoint that holds most of the
  prompt. `PrefixIndex(head_first=True)` reproduces it, and `tests/test_prefix_and_tokens.py` pins both.
- **Data-parallel ranks are summed.** The lab sums queue/running over the `engine` label of a pod, and
  takes the maximum KV usage. The EPP reads one series per metric: the first, because vLLM sets no
  timestamps. `extract_vllm(fam, aggregate="first")` reproduces the EPP.
- **Lab-only parameters:** the prefix index can take `ttlSeconds`, and `PickerConfig.to_upstream()`
  removes it. The lab changes an auto-tuned LRU capacity into units of index blocks. Upstream uses the
  block count of the engine directly.

The presets are in `igwlab/configs/`, and you can load each one by name:

- `round-robin`.
- `default-weighted`, the Helm chart default: prefix ×3, queue ×2, KV ×2.
- `prefix-only`.
- `load-only`.
- `queue-only`.
- `active-requests`.
- `sticky-until-saturated`, the optimized-baseline shape, calibrated for the fake backend.

The router metrics (`GET /metrics`) copy EPP series under an `igw_` prefix: `igw_request_total`
(`llm_d_epp_request_total`), `igw_request_ttft_seconds`, `igw_average_queue_size`,
`igw_average_kv_cache_utilization`, `igw_ready_endpoints`, `igw_scheduler_attempts_total`,
`igw_prefix_indexer_size`, `igw_prefix_indexer_hit_ratio`, and also `igw_pool_saturation`.
`GET /debug/state` returns the endpoints, the prefix index size per endpoint and the last decisions.

## Caveats

- **Emulated, not GPU, timing at T0.** The latencies that you measure against the fake backend are real wall-clock
  HTTP on your machine. But the prefill/decode times of the backend are **emulated** (`EngineProfile`). Use the
  latencies to compare policies, and never as GPU numbers. The policy that wins can depend on that emulation
  (PRIMER §2.5).
- **The Docker, kind, GPU and GKE paths were not executed** during the work on this lab. There was no Docker daemon,
  no GPU and no cloud access. The paths are correct by construction: `bash -n`, `DRY_RUN=1`, Terraform `validate`,
  `kubernetes-validate --strict -k 1.34.0` for core kinds, and CRD-schema checks for the rest
  (`tests/test_manifests.py`). The T1 cells of notebook 04 use the same router and bench code that the T0 tests use.
  In those cells, the code goes to the servers in `IGW_BACKENDS`.
- **Versions move.** Everything in "Reference" is pinned as of September 2026 and has the mark verify. The GKE costs
  use assumed us-central1 prices.

## Reference

### The library

| Module | Lines | The one idea |
|---|---|---|
| `router/tokens.py` | ~60 | A router "tokenizes": it packs request bytes into 4-byte pseudo-tokens (the `estimate` backend of the EPP). |
| `router/prefix.py` | ~220 | Chained block hashes + a per-endpoint LRU index. The match is the count of the blocks at the start of the prompt that the endpoint holds. |
| `router/datalayer.py` | ~170 | Scraped vLLM metrics (lagged) against router-local in-flight counters (instant, partial). |
| `router/plugins.py` | ~500 | Producers, filters, scorers and pickers, with upstream types, parameters and formulas. |
| `router/config.py` | ~270 | `EndpointPickerConfig`: parse, validate, inject upstream defaults, export for the real EPP. |
| `router/scheduler.py` | ~110 | One scheduling cycle, and a per-scorer decision table that explains the decision. |
| `router/server.py` | ~280 | The proxy: admission (at saturation, reject priority < 0), dispatch, streaming, metrics. |
| `fakebackend.py` | ~560 | A replica is a cache with a queue: TTFT = queue time + prefill of *uncached* tokens. |
| `autoscale.py` | ~470 | The HPA, exactly, as a proportional controller with guard rails. A queue target from Little's law. |
| `bench.py` | ~340 | Agent traffic is prefix-heavy. Measure TTFT, hit rate (or `n/a` when the engine does not report it) and the per-replica split. |
| `k8s.py` | ~210 | The Gateway-mode object graph, validated offline against upstream CRD schemas. |
| `stack.py` | ~140 | Backends + router in one process on free ports. Or the router alone, in front of servers that you already run (T1: real vLLM). |
| `promtext.py` | ~310 | Prometheus text, counters against gauges, `histogram_quantile`. |

### Pinned versions (verify before relying on them; Sep 2026)

| Component | Pin | Where |
|---|---|---|
| llm-d-router (EPP image, charts, InferenceObjective CRD) | v0.10.0 (charts `--version v0.10.0`, the guides use the floating `v0`) | kind/gke values and scripts |
| Gateway API Inference Extension (InferencePool v1 CRD) | v1.6.2 | kind/gke scripts. GKE ≥ 1.34.0-gke.1626000 manages it. |
| Gateway API (Gateway, HTTPRoute schemas) | v1.6.2 standard | `igwlab/crds/` |
| llm-d-inference-sim | v0.11.2 | compose, kind |
| vLLM | `vllm/vllm-openai:v0.30.0` (PyPI 0.30.0, 2026-09-22), models `Qwen/Qwen2.5-1.5B-Instruct` (GKE) and `Qwen/Qwen2.5-0.5B-Instruct` (any-gpu) | gke/vllm.yaml, any-gpu |
| Prometheus | v3.15.0 | compose |
| kind node image | `kindest/node:v1.34.0` | kind-config.yaml |
| Terraform / google provider | ≥ 1.9 / ≥ 8.0 (validated with 8.4.0) | gcp/terraform |
| GKE Gateway classes | `gke-l7-regional-external-managed` (internal: `gke-l7-rilb`) | gke/gateway.yaml |
| HPA metric names via the Custom Metrics Stackdriver Adapter | `prometheus.googleapis.com\|vllm:num_requests_waiting\|gauge`, `…\|vllm:num_requests_running\|gauge` | gke/hpa.yaml |
| Custom Metrics Stackdriver Adapter manifest | k8s-stackdriver commit `9500033f` (master, 2026-09-26, image v0.16.11-gke.0) | gke/install.sh, uninstall.sh |

### Regenerating notebooks and snapshots

```bash
python3 tools/build_notebooks.py                        # notebooks/ and solutions/ from notebooks_src/
python3 tools/run_notebooks.py solutions                # must run clean (CPU, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
python3 tools/snapshot_crds.py <crd.yaml> <version> "<source>" ...   # refresh igwlab/crds/ from upstream
```

MIT licensed.
