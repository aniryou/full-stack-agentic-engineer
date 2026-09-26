# inference-gateway-lab — route LLM requests the way the llm-d endpoint picker does, then run it for real

Run a real async router that makes the llm-d endpoint picker's decisions — filters, weighted scorers, a picker — in
front of OpenAI-compatible engines: first in one Python process against emulated backends, then on kind with the real
llm-d Router, then behind GKE Inference Gateway. Package `igwlab`.

## Start here

1. `python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q` — 76 tests in
   a few seconds, offline.
2. Run the [comparison under "Run it"](#run-it): three emulated backends behind the router, round-robin against the
   llm-d chart default; it prints TTFT percentiles, hit rate and the per-replica split in about 4 s.
3. Open [`notebooks/01_router_in_process.ipynb`](notebooks/01_router_in_process.ipynb) with
   [PRIMER §2](../PRIMER.md#2-routing-signals-and-algorithms) beside it.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* "T0 + Docker" is a laptop with Docker, still free.

| Notebook | You will be able to explain… | Time | Tier | Primer |
|---|---|---|---|---|
| [`01_router_in_process`](notebooks/01_router_in_process.ipynb) | prefix hashing and the index, queue/KV scores, the weighted pick; why round-robin loses on agent traffic (measured) | ~1.5 h | T0 | §1 Why a layer above the engine, §2 Routing signals and algorithms |
| [`02_scorer_weights_and_hot_prefixes`](notebooks/02_scorer_weights_and_hot_prefixes.ipynb) | the locality-vs-load trade-off in numbers, hot prefixes, the scrape-lag herd, choosing an affinity threshold, saturation shedding | ~1.5 h | T0 | §2, §3 Flow control and priorities |
| [`03_autoscaling_recommender`](notebooks/03_autoscaling_recommender.ipynb) | the HPA controller where the basic rule is not enough (missing and starting pods, legacy vs `behavior`, live scrapes); which vLLM signals to scale on, why queue alone collapses, and targets derived from batch slots and Little's law | ~1.5 h | T0 | §4 Autoscaling |
| [`04_local_stack_with_llm_d`](notebooks/04_local_stack_with_llm_d.ipynb) | InferencePool semantics, the llm-d Router standalone mode, what the EPP reads that the lab router does not, the simulator's latency model; benchmarks a running stack, or the lab router in front of real vLLM | ~1.5 h | T0 walkthrough; T0 + Docker (compose or kind); **T1/T2** with a GPU (real vLLM, `deploy/any-gpu`) | §9 The Kubernetes-native stack, September 2026, §10 Where to run it |
| [`05_gke_inference_gateway`](notebooks/05_gke_inference_gateway.ipynb) | the GKE Inference Gateway object graph, CRD validation, GMP → HPA plumbing, what an hour costs | ~2 h | T3 (offline plan/inspect is T0) | §9, §10 |

Times are rough: the repo's curriculum (`CURRICULUM.md` at the repo root) budgets about 6 hours for notebooks 01–04
and 2 for 05. Every notebook opens with "The one-minute version", works examples, has 4–5 exercises each followed by
a check (✅), and ends with "In a design review". Blanks are in `notebooks/`, answers in [`solutions/`](solutions/),
sources in `notebooks_src/` (percent format).

The code behind the notebooks:

| Path | What it is |
|---|---|
| [`igwlab/router/`](igwlab/router/) | a real async HTTP router in front of N OpenAI-compatible backends: it scrapes each backend's `/metrics` (vLLM names), keeps an approximate prefix index, runs filters → weighted scorers → picker configured by an `EndpointPickerConfig` YAML, streams responses byte for byte, and exports its own metrics. A readable re-implementation of the EPP's decision logic (not an ext-proc server) |
| [`igwlab/fakebackend.py`](igwlab/fakebackend.py) | an OpenAI-compatible backend with vLLM-shaped *emulated* timing, a vLLM-style prefix cache, and vLLM metric names — the T0 stand-in for a GPU |
| [`igwlab/autoscale.py`](igwlab/autoscale.py) | the Kubernetes HPA recommender ported line by line from kube-controller-manager v1.34, plus an HPA manifest generator and a simulated pool with cold starts |
| [`igwlab/bench.py`](igwlab/bench.py) | a shared-prefix, multi-turn agentic workload and a streaming load generator |
| [`deploy/`](#deploy-paths) | docker compose (CPU), kind + llm-d Router (CPU), real vLLM on any GPU box, and GKE (Terraform + manifests) |

This lab is the detailed companion of the topic primer ([`../PRIMER.md`](../PRIMER.md)); the minimal, simulation-only
companion is [`../orchestrator-core/`](../orchestrator-core/). It is independent of both: nothing here imports the core.

## Run it

```bash
cd inference-gateway-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                     # 76 tests, a few seconds, offline
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

On Colab, the first cell of each notebook clones the repo, changes into this directory and `pip install -e .`s it
(one-time setup: the repository's [`COLAB.md`](../../../COLAB.md)).

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
| `deploy/kind` | T0 + Docker (CPU, kind) | 3 simulator pods + **llm-d Router** standalone (EPP + Envoy) via Helm | free | [`deploy/kind/README.md`](deploy/kind/README.md) |
| `deploy/any-gpu` | T1/T2 (real GPU, no cloud account) | 2+ real vLLM replicas (`Qwen2.5-0.5B-Instruct`) from pip or Docker; the lab router in front; notebook 04's T1 cells measure them | Colab/Kaggle T4 free; rented 24 GB GPU ~$0.3–0.7/h (verify) | [`deploy/any-gpu/README.md`](deploy/any-gpu/README.md) |
| `deploy/gcp/terraform` | T3 | zonal GKE, Gateway API, proxy-only subnet, L4 Spot pool 0→2, Managed Prometheus | ~$0.16/h with the GPU pool at 0, ~$0.44/h with one L4 Spot node (assumed prices, verify) | [`deploy/gcp/terraform/README.md`](deploy/gcp/terraform/README.md) |
| `deploy/gke` | T3 | vLLM on L4, EPP + InferencePool + objectives (Helm), Gateway + HTTPRoute, HPA on vLLM's waiting **and** running requests | ~$0.44/h even when idle while installed (`minReplicas: 1` keeps one L4 node) | [`deploy/gke/README.md`](deploy/gke/README.md) |

Prices and GPU availability for GCP and non-GCP options: `COMPUTE.md` at the repo root.

## How the router maps to llm-d (llm-d-router v0.10.0, the pinned release)

| Upstream | Here | Notes |
|---|---|---|
| Envoy (standalone) or cloud L7 LB (Gateway mode) + ext-proc | `router/server.py` | one process proxies *and* picks; response header `x-gateway-destination-endpoint` |
| `token-producer` (`estimate`) | `router/tokens.py` | same 4-byte packing of tools + role + content |
| `approx-prefix-cache-producer` | `ApproxPrefixCacheProducer` | chained 64-bit hashes seeded by model + cache salt, block size ≥ 64 tokens, LRU 31,250/endpoint, greedy match; the lab inserts tail-first (see below) |
| `metrics-data-source` + `core-metrics-extractor` | `router/datalayer.py` | `vllm:num_requests_waiting/running`, `kv_cache_usage_perc`, `lora_requests_info`, `cache_config_info`; 50 ms default |
| `prefix-cache-scorer`, `queue-scorer`, `kv-cache-utilization-scorer`, `running-requests-size-scorer`, `active-request-scorer`, `token-load-scorer`, `lora-affinity-scorer` | `router/plugins.py` | formulas pinned in `tests/test_plugins.py`; like upstream, a never-scraped endpoint has zero metrics and so looks idle; `token-load-scorer` adds the request's own uncached tokens per endpoint |
| `inflight-load-producer` | `InFlightLoadProducer` | uncached tokens = (total − matched blocks) × block size + tail, as upstream |
| `utilization-filter`, `prefix-cache-affinity-filter` | `router/plugins.py` | affinity filter supports `ttftSource: prefillThroughput` only |
| `max-score-picker`, `random-picker`, `weighted-random-picker` | `router/plugins.py` | see ties below |
| legacy admission + `utilization-detector` | `Router.pool_saturation` | priority < 0 → 429 when saturation ≥ 1.0 (the lab also sets `x-llm-d-request-dropped-reason: rejected-saturated`, the reason string flow control uses) |
| `InferenceObjective` priority | `RouterSettings.objectives` | selected by header `x-llm-d-inference-objective` |

Deliberate differences (each also stated in the module that implements it):

- **One scheduling profile** only (no P/D disaggregation).
- **No flow control.** `featureGates: [flowControl]` is accepted with a warning (the loader, the
  CLI and `Router` all print it; `describe()` shows it): no router-side queues, priority bands or
  TTLs — the legacy shedding applies. The real EPP queues by priority with the gate on.
- **Ties rotate.** `max-score-picker` rotates each tie by a per-request counter over a name-sorted
  list, so *no scorers* = exact round-robin. v0.10.0 shuffles the candidates at random before a
  stable sort, so its ties are uniformly random (main, after v0.10.0, rotates over a map-ordered list).
- **Tail-first prefix insertion.** The lab inserts a prompt's block hashes tail-first, so a prompt's
  head is evicted last and the greedy match never undercounts; that is llm-d-router *main* after
  v0.10.0. v0.10.0 inserts head-first: under LRU pressure it evicts heads first and its greedy scan
  can report 0 for an endpoint that holds most of the prompt (`PrefixIndex(head_first=True)`
  reproduces it; `tests/test_prefix_and_tokens.py` pins both).
- **Data-parallel ranks are summed.** The lab sums queue/running over a pod's `engine` label and takes
  the maximum KV usage; the EPP reads one series per metric (the first, as vLLM sets no timestamps).
  `extract_vllm(fam, aggregate="first")` reproduces the EPP.
- **Lab-only parameters:** the prefix index may take `ttlSeconds` (stripped by
  `PickerConfig.to_upstream()`); auto-tuned LRU capacity is converted to index-block units (upstream
  uses the engine's block count directly).

Presets (`igwlab/configs/`, loadable by name): `round-robin`, `default-weighted` (the Helm chart
default: prefix ×3, queue ×2, KV ×2), `prefix-only`, `load-only`, `queue-only`, `active-requests`,
`sticky-until-saturated` (the optimized-baseline shape, calibrated for the fake backend).

Router metrics (`GET /metrics`) mirror EPP series under an `igw_` prefix: `igw_request_total`
(`llm_d_epp_request_total`), `igw_request_ttft_seconds`, `igw_average_queue_size`,
`igw_average_kv_cache_utilization`, `igw_ready_endpoints`, `igw_scheduler_attempts_total`,
`igw_prefix_indexer_size`, `igw_prefix_indexer_hit_ratio`, plus `igw_pool_saturation`.
`GET /debug/state` returns the endpoints, the prefix index size per endpoint and the last decisions.

## Caveats

- **Emulated, not GPU, timing at T0.** Latencies measured against the fake backend are real wall-clock HTTP on your
  machine, but the backend's prefill/decode timing is **emulated** (`EngineProfile`): use them to compare policies,
  never as GPU numbers. Which policy wins can depend on that emulation (PRIMER §2.5).
- **The Docker, kind, GPU and GKE paths were not executed** while this lab was written (no Docker daemon, GPU or cloud
  access); they are correct by construction: `bash -n`, `DRY_RUN=1`, Terraform `validate`,
  `kubernetes-validate --strict -k 1.34.0` for core kinds, and CRD-schema checks for the rest
  (`tests/test_manifests.py`). The T1 cells of notebook 04 use the same router and bench code the T0 tests exercise,
  pointed at the servers in `IGW_BACKENDS`.
- **Versions move.** Everything below is pinned as of September 2026 and marked verify; GKE costs are assumed
  us-central1 prices.

## Reference

### The library

| Module | Lines | The one idea |
|---|---|---|
| `router/tokens.py` | ~60 | a router "tokenizes" by packing request bytes into 4-byte pseudo-tokens (the EPP's `estimate` backend) |
| `router/prefix.py` | ~220 | chained block hashes + a per-endpoint LRU index; the match is the count of leading blocks held |
| `router/datalayer.py` | ~170 | scraped vLLM metrics (lagged) vs router-local in-flight counters (instant, partial) |
| `router/plugins.py` | ~500 | producers, filters, scorers, pickers with upstream types, parameters and formulas |
| `router/config.py` | ~270 | `EndpointPickerConfig`: parse, validate, inject upstream defaults, export for the real EPP |
| `router/scheduler.py` | ~110 | one scheduling cycle and an explainable per-scorer decision table |
| `router/server.py` | ~280 | the proxy: admission (shed priority < 0 at saturation), dispatch, streaming, metrics |
| `fakebackend.py` | ~560 | a replica is a cache with a queue: TTFT = queueing + prefill of *uncached* tokens |
| `autoscale.py` | ~470 | the HPA as a proportional controller with guard rails, exactly; a queue target from Little's law |
| `bench.py` | ~340 | agent traffic is prefix-heavy; measure TTFT, hit rate (or `n/a` when the engine does not report it) and the per-replica split |
| `k8s.py` | ~210 | the Gateway-mode object graph, validated offline against upstream CRD schemas |
| `stack.py` | ~140 | backends + router in one process on free ports — or the router alone in front of servers you already run (T1: real vLLM) |
| `promtext.py` | ~310 | Prometheus text, counters vs gauges, `histogram_quantile` |

### Pinned versions (verify before relying on them; Sep 2026)

| Component | Pin | Where |
|---|---|---|
| llm-d-router (EPP image, charts, InferenceObjective CRD) | v0.10.0 (charts `--version v0.10.0`; the guides use the floating `v0`) | kind/gke values and scripts |
| Gateway API Inference Extension (InferencePool v1 CRD) | v1.6.2 | kind/gke scripts; GKE ≥ 1.34.0-gke.1626000 manages it |
| Gateway API (Gateway, HTTPRoute schemas) | v1.6.2 standard | `igwlab/crds/` |
| llm-d-inference-sim | v0.11.2 | compose, kind |
| vLLM | `vllm/vllm-openai:v0.30.0` (PyPI 0.30.0, 2026-09-22), models `Qwen/Qwen2.5-1.5B-Instruct` (GKE) and `Qwen/Qwen2.5-0.5B-Instruct` (any-gpu) | gke/vllm.yaml, any-gpu |
| Prometheus | v3.15.0 | compose |
| kind node image | `kindest/node:v1.34.0` | kind-config.yaml |
| Terraform / google provider | ≥ 1.9 / ≥ 8.0 (validated with 8.4.0) | gcp/terraform |
| GKE Gateway classes | `gke-l7-regional-external-managed` (internal: `gke-l7-rilb`) | gke/gateway.yaml |
| HPA metric names via the Custom Metrics Stackdriver Adapter | `prometheus.googleapis.com\|vllm:num_requests_waiting\|gauge`, `…\|vllm:num_requests_running\|gauge` | gke/hpa.yaml |
| Custom Metrics Stackdriver Adapter manifest | k8s-stackdriver commit `9500033f` (master, 2026-09-26; image v0.16.11-gke.0) | gke/install.sh, uninstall.sh |

### Regenerating notebooks and snapshots

```bash
python3 tools/build_notebooks.py                        # notebooks/ and solutions/ from notebooks_src/
python3 tools/run_notebooks.py solutions                # must run clean (CPU, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
python3 tools/snapshot_crds.py <crd.yaml> <version> "<source>" ...   # refresh igwlab/crds/ from upstream
```

MIT licensed.
