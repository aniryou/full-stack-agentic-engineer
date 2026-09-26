# Orchestrating a fleet of engines: routing, autoscaling, disaggregation and KV-cache tiers

*Layer 05 of the stack. Facts checked 26 September 2026; product details that move are collected in the
[Verify list](#verify-list). Every number here is either computed by [`orchestrator-core`](orchestrator-core/)
(package `fleetsim`; the function is named next to the number) or cited. Numbers from the simulator are
labelled **simulated**: they come from an engine model built from spec-sheet arithmetic, not from a GPU.*

One inference engine (vLLM, SGLang, TensorRT-LLM) turns a GPU into a token server; layer 04 is about what happens
inside it. This layer is about many of them: which replica gets a request, how many replicas exist, and whether
one request's prefill and decode should even run on the same GPU. It covers routing signals and algorithms, flow
control, autoscaling, prefill/decode disaggregation, KV-cache tiers beyond HBM, multi-model and LoRA routing, and
the Kubernetes-native stack that implements all of it in 2026 (Gateway API Inference Extension, the llm-d Router,
GKE Inference Gateway, NVIDIA Dynamo). It assumes the engine basics — prefill vs decode, TTFT and TPOT, the KV cache
and paged blocks — from [`00-foundations/gpu-capacity-planning/PRIMER.md`](../../00-foundations/gpu-capacity-planning/PRIMER.md),
[`04-inference-engine/kv-cache/kv-cache-primer.md`](../../04-inference-engine/kv-cache/kv-cache-primer.md) and
[`04-inference-engine/paged-attention/paged-attention-primer.md`](../../04-inference-engine/paged-attention/paged-attention-primer.md).

---

## The one-minute version

- **A replica is a stateful cache, not a stateless server.** Where a request lands changes what it costs: the replica
  that already holds its prefix skips most of the prefill. And requests differ by one to two orders of magnitude in
  prefill work, KV memory and residency time, so counting requests (round-robin, least-connections) does not balance
  work.
- **Routing is a knob between locality and load.** Prefix affinity raises the hit rate until a hot prefix melts its
  replica; load balancing spreads work but re-prefills everything. Production routers (the llm-d endpoint picker)
  filter, score prefix match, queue depth and KV use with weights, and pick the maximum — or stay sticky until an
  estimated TTFT penalty says spread.
- **Autoscale on in-flight work, not GPU utilisation.** The HPA's rule is `ceil(current × metric / target)` with a 10 %
  band, a 5-minute scale-down memory and rate-limited scale-up. Continuous batching pegs "GPU util" at a fraction of
  capacity; queue depth alone reads zero until the cliff. The cold start (node, image, weights, warm-up) decides the
  tail of a traffic step, so headroom is part of the design.
- **Disaggregate prefill from decode only when the numbers say so.** It removes the stall a prefill chunk puts on
  every decode, and costs a KV transfer (`prompt tokens × KV bytes/token ÷ link bandwidth`) plus two pools that must
  match the traffic's input/output mix. A smaller prefill chunk is the first thing to try.
- **Agent sessions need KV beyond HBM.** Their working set (sessions × context × KV bytes/token) dwarfs HBM; offload
  to DRAM/NVMe/remote tiers wins whenever the tier is faster than prefill produces KV, and sticky routing or a shared
  tier keeps a session near its KV.

---

## 1. Why a layer above the engine

### 1.1 Replicas are stateful caches

An engine keeps every running request's KV cache in HBM and, with prefix caching, keeps freed blocks too: each full
block of 16 tokens is registered under a chain hash `h_i = H(h_{i-1}, tokens of block i)`, and a later request whose
prompt starts with the same tokens reuses those blocks instead of recomputing them (`fleetsim.replica.BlockPool`,
the same scheme as vLLM's automatic prefix caching; freed blocks are evicted LRU, a request's tail blocks first). So
each replica holds a different, constantly changing subset of the fleet's reusable state, and the router decides
which subset a request can use.

The engine model used throughout (`fleetsim.replica.engine_profile()`, an 8B bf16 model on one L4 as the running
example): weights 16 GB; a KV pool of `(0.9 × 24 GB − 16 GB − 1 GB) ÷ (131,072 B/token × 16)` = **2,193 blocks =
35,088 tokens**; every step streams the weights once (**53.3 ms** at 0.3 TB/s); prefill runs at
`121 TFLOP/s × 0.5 MFU ÷ (2 × 8 B)` = **3,781 tokens/s**. One step costs
`overhead + max(tokens ÷ prefill speed, weight read + context tokens × KV read time)` (`fleetsim.replica.step_time()`):

| One engine step (simulated, 8B on an L4) | Time |
|---|---|
| decode, 16 requests at 2,000 tokens of context | 70.3 ms |
| a 2,048-token prefill chunk | 545 ms |
| the same on an H100 (`H100_8B`): decode 16 / chunk 2,048 | 9.0 ms / 69 ms |

Decode is memory-bound, so batching is almost free; prefill is compute-bound, and every decode sharing its step
waits for it. Layer 01 ([roofline](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §3) derives both
regimes; this layer exploits them.

### 1.2 Why round-robin and least-connections fail

Web load balancing assumes small, similar, stateless requests. LLM requests are none of those:

- **Cost spread.** A request occupies a replica in three currencies — prefill compute, KV blocks, residency time.
  A 6,000-token RAG prompt has 5× the prefill of a 1,200-token chat turn and 8× its KV-block-seconds; across a
  realistic chat + RAG mix the p99 request costs about **30×** the p1 request (simulated, notebook 01). The output
  length is unknown until generation stops.
- **Capacity is memory.** A replica is full when its KV blocks are; overload shows up as queueing and preemption,
  not as CPU pressure. How many requests fit is a KV question
  ([capacity planning](../../00-foundations/gpu-capacity-planning/PRIMER.md), formula 2).
- **Connections are not work.** A streaming response holds its connection for the whole decode — 0.3 s or 30 s.

On four L4 replicas near saturation with that chat + RAG mix (simulated, notebook 01), p95 TTFT was 6.3 s for random
choice, 4.1 s for round-robin, 1.9 s for power-of-two, 1.7 s for least-outstanding and 1.3 s for least-waiting
(the scraped engine queue). The closer the signal is to the cause of TTFT — requests waiting in the engine — the
better the tail.

### 1.3 The three decisions

```
                  ┌──────────────────────── the orchestration layer ─────────────────────────┐
 request ──►  gateway (06)  ──►  which replica?        how many replicas?      how to split the work?
               auth, quotas       router / endpoint     autoscaler on a         prefill pool ─KV─► decode pool
               admission          picker (§2, §3)       fleet signal (§4)       KV tiers beyond HBM (§5, §6)
                                        │                       │                          │
                                        ▼                       ▼                          ▼
                               ┌─ engine ─┐ ┌─ engine ─┐ ┌─ engine ─┐   (layer 04: batching, prefix cache)
                               └──────────┘ └──────────┘ └──────────┘   (layer 03: pods, nodes, GPUs)
```

**Which replica** is decided per request in milliseconds; **how many** every 15 s to minutes; **how to split** at design
time and re-tuned when the traffic mix shifts. Admission control and per-tenant rate limits sit one layer up, in the
gateway ([06 scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.3).

---

## 2. Routing signals and algorithms

### 2.1 The signals

| Signal | Source | Fresh? | What it predicts |
|---|---|---|---|
| requests in flight to each replica | the router's own counters | always | concurrency; blind to size |
| in-flight **tokens** (uncached prompt tokens sent, not yet prefilled) | the router's own counters | always | prefill backlog → TTFT |
| `vllm:num_requests_waiting` | engine `/metrics` | scrape age | queueing → TTFT, directly |
| `vllm:num_requests_running` | engine `/metrics` | scrape age | batch size → ITL |
| `vllm:kv_cache_usage_perc` (0–1) | engine `/metrics` | scrape age | memory headroom → preemption risk |
| prefix match, **approximate** | router's record of what it sent | never sees evictions | cache hit, if still resident |
| prefix match, **precise** | engine KV events (blocks stored/removed, over ZMQ) | event lag | cache hit |
| adapter resident, session id | `vllm:lora_requests_info`, headers | scrape age / exact | adapter swaps, session locality |

The llm-d endpoint picker refreshes engine metrics every 50 ms by default; a router fed by a 15-second Prometheus
scrape sees a much older world. Metric names are vLLM V1's (`vllm/v1/metrics/loggers.py`); counters get a `_total`
suffix when exposed.

### 2.2 Load-only algorithms

- **Round-robin** — fair in count, blind to cost and cache (`fleetsim.routers.RoundRobin`).
- **Least outstanding** — argmin of what the router has in flight (`LeastOutstanding`). Exact while one router sees
  all traffic; with several router replicas each sees only its share.
- **Power of two choices** — sample two replicas at random, take the less loaded (`PowerOfTwo`). Throwing n requests
  at n servers at random gives a maximum load of about ln n / ln ln n; with two choices it drops to about
  ln ln n / ln 2 (Azar, Broder, Karlin, Upfal; Mitzenmacher). It needs no global scan, and it degrades gracefully
  on stale data: the idle replica is chosen only when sampled — probability `1 − C(n−1,2)/C(n,2)` = 1/2 for 4
  replicas, 2/n in general — so a burst cannot all land on one stale minimum. Envoy's `LEAST_REQUEST` balancer is
  this algorithm (two choices by default).

**Herding** is the failure P2C avoids. With bursts of 16 req/s and a router picking the argmin of scraped
`num_requests_running` (simulated, notebook 01), p95 TTFT grew from 0.28 s (50 ms-old metrics) to 0.91 s (2 s) and
1.62 s (10 s); P2C on the same stale data stayed at 0.27–0.39 s, and the router's own counters were immune.

### 2.3 Locality: prefix hashing, session affinity, bounded loads

**Prefix hashing** puts the hash of a request's first k blocks on a consistent-hash ring (each replica owns ~64
virtual points; adding one of n replicas moves only ~1/n of the keys — `fleetsim.routers.HashRing`, tested). Every
request with the same system prompt lands on the same replica: maximum locality, no load awareness. Hashing the
**session id** instead gives agent-session affinity.

**Consistent hashing with bounded loads** (Mirrokni, Thorup, Zadimoghaddam, SODA 2018) caps every replica at

```
capacity = ceil((1 + ε) × (m + 1) / n)          m requests in flight, n replicas, +1 for the one being placed
```

and walks clockwise past full replicas (`ConsistentHashBoundedLoad.capacity()`): with ε = 0.25, n = 4 and m = 7 the
cap is ceil(1.25 × 8 ÷ 4) = 3; for 100 requests with one key, no replica ever holds more than 32 (pure hashing puts
all 100 on one). ε is the locality/balance knob with a hard guarantee. Envoy and HAProxy expose the same idea as a
hash balance factor (verify the field names for your version).

### 2.4 The endpoint picker: filters → scorers → picker

The llm-d Router's endpoint picker (EPP) runs a **scheduling profile** per request: a chain of **filters** (drop
endpoints), weighted **scorers** (each returns a score clamped to [0, 1]), and a **picker** (default
`max-score-picker`; ties broken at random). `fleetsim.routers.WeightedScorer` reproduces that shape; the scorers
follow the upstream plugin definitions:

| Plugin | Score or rule | fleetsim |
|---|---|---|
| `prefix-cache-scorer` | matched prefix blocks ÷ total prompt blocks (optionally blended with a squared match-length term) | `PrefixCacheScorer` |
| `queue-scorer` | `(maxQ − q) ÷ (maxQ − minQ)` over waiting-queue length; 1.0 for all if equal | `QueueScorer` |
| `kv-cache-utilization-scorer` | `1 − kv_cache_usage` | `KVCacheUtilizationScorer` |
| `token-load-scorer` | `1 − min(1, in-flight tokens ÷ queueThresholdTokens)` (default 4,194,304) | `TokenLoadScorer` |
| `prefix-cache-affinity-filter` | keep endpoints with prefix score ≥ 0.80; if the best sticky endpoint's estimated TTFT (`in-flight tokens ÷ peakPrefillThroughput`) exceeds the best non-sticky one's by more than `maxTTFTPenaltyMs` (18,000), keep all | `PrefixAffinityFilter` |
| LoRA affinity | keep endpoints with the adapter loaded, else those with a free adapter slot | `LoraAffinityFilter` |

A configuration names plugin instances and composes them into profiles:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
- type: queue-scorer
- type: kv-cache-utilization-scorer
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
    weight: 3                      # illustrative weights: tune them on your traffic (notebook 02)
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
```

Omitted pieces are injected (a `max-score-picker`, a `single-profile-handler`, weight 1.0). In September 2026 the
llm-d *optimized baseline* ships a different composition — `prefix-cache-affinity-filter` plus `token-load-scorer`,
"sticky until saturated" — because a weighted prefix scorer with a max-score picker hot-spots popular prefixes, and a
filter with an explicit load gate states the trade-off in one number (`fleetsim.routers.sticky_until_saturated()`).
Its `peakPrefillThroughput` default, 15,928 tokens/s, was measured for Qwen3-32B on two H100s (TP=2); it is
hardware-specific and has to be recalibrated elsewhere.

NVIDIA Dynamo's KV router expresses the same trade-off as a cost in block units:
`cost = prefill_load_scale × (prefill blocks − overlap credit × cached blocks) + decode blocks`, lowest cost wins
(optionally sampled with a temperature). Counting load in the *same units* as the cache saving — tokens or blocks
still to compute — is the idea both routers converged on.

### 2.5 The locality-vs-load tension, in numbers

Agent sessions from six agents with 2,000-token system prompts (Zipf-popular: the top agent sends ~40 % of sessions),
each turn re-sending its history, on four L4 replicas (simulated, notebook 02; SLO TTFT ≤ 1 s, TPOT ≤ 150 ms):

| Router | Hit rate | TTFT p95 | Imbalance (max/mean tokens) | SLO attainment |
|---|---|---|---|---|
| round-robin | 0.52 | 4.35 s | 1.16 | 0.68 |
| power-of-two | 0.51 | 3.38 s | 1.04 | 0.61 |
| prefix hash on the system prompt | 0.63 | 48.7 s | 2.65 | 0.41 |
| prefix hash on the session | 0.84 | 1.01 s | 1.27 | 0.95 |
| bounded-load hash, ε = 0.25 | 0.73 | 1.52 s | 1.48 | 0.86 |
| EPP 3:2:2 (prefix : queue : KV), approximate index | 0.87 | 0.37 s | 1.18 | 0.99 |
| sticky until saturated (2 s penalty) | 0.83 | 0.91 s | 1.09 | 0.96 |

Sweeping only the prefix weight (queue and KV at 2) traces the tension: weight 0 → hit rate 0.55, p95 1.9 s;
weight 3 → 0.87, 0.37 s; weight 30 → 0.82, 3.4 s with imbalance 2.3. Hit rate saturates early; imbalance keeps
climbing. With a hotter prefix (Zipf 2.0, the top agent sending two thirds of the sessions) pure prefix hashing's p95
reached 81 s while the EPP held 0.47 s. The knob has an optimum that depends on the workload, which is why it
should be tuned on a replay of real traffic while watching hit rate *and* per-replica load.

### 2.6 Approximate vs precise prefix indexes

An **approximate** index records the prompt blocks of each request the router sends to each replica, LRU-bounded
(`ApproxPrefixIndex`); it never learns about evictions. A **precise** index is built from the engines' KV events
(`PreciseIndex`; llm-d's `precise-prefix-cache-producer`, whose block size must match the engine's `--block-size`).
In the simulations here the approximate index stays within noise of the precise one — even sized 15× too large, when
73 % of its entries were phantoms — because agent turns come back within seconds and the entries actually queried are
fresh. The difference matters when reuse is distant (long tool pauses, many tenants, small caches) — which is also
when the KV is gone and only tiers help (§6) — and when the router should see offloaded tiers or share one view
across router replicas.

---

## 3. Flow control and priorities

### 3.1 Why queue in the router

Once a request is in an engine's waiting queue it is committed to that replica, even if another frees up first. The
llm-d EPP's **flow control** (feature gate `flowControl`) holds requests in the router instead and dispatches them
when a **saturation detector** says an endpoint has room — "no-regret scheduling": the decision is made with the
latest information, and a burst queues in one place where it can be ordered, prioritised and shed. The out-of-box
detector is `utilization-detector`; the flow-control guide recommends `concurrency-detector` (a per-endpoint cap on
in-flight requests or tokens, e.g. `maxConcurrency: 8` as the guide's example) to avoid telemetry lag. The queue is
bounded (`maxRequests`, `maxBytes`) and requests expire (`defaultRequestTTL`).

Little's law sizes the cap: in-flight = arrival rate × time in system. Four endpoints capped at 8 hold 32 requests;
if each takes 10 s end to end, the pool completes about 3.2 req/s, and anything above that waits in the router,
where it costs nothing but time.

### 3.2 Priorities: `InferenceObjective`

Each request is classified into a flow by a fairness id and a priority. The `InferenceObjective` resource
(`llm-d.ai/v1alpha2`; `spec.poolRef`, `spec.priority` as an int32) sets the priority for a workload on a pool: higher
values are always served first, negative values are allowed, and an unset priority counts as 0. Fairness is enforced
only *within* a priority band (the default ordering is global FCFS). Priority matters only when requests queue —
exactly when the pool is saturated — so it is the tool for "the interactive agent before the nightly eval".

### 3.3 Saturation, shedding and where admission lives

Router-side queues bound latency only if something eventually says no: requests past their TTL are rejected, and a
full queue rejects new arrivals, which the gateway turns into `429` or `503` with `Retry-After`. Who decides what:

| Question | Layer | Mechanism |
|---|---|---|
| Is this tenant within budget? Should we degrade or shed? | 06 gateway | token buckets, in-flight caps, degrade levels — [`scalelab/admission.py`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/admission.py) |
| Which request goes next, and to which replica, when replicas are busy? | 05 router | flow control, priorities, saturation detection |
| Which requests share the next engine step? | 04 engine | continuous batching, token budget, preemption |

A burst is absorbed in milliseconds by the first two rows; autoscaling (§4) restores capacity minutes later.

---

## 4. Autoscaling

### 4.1 The HPA algorithm, exactly

Every sync period (15 s) the Horizontal Pod Autoscaler does, per metric (`fleetsim.autoscale`, which mirrors
`kube-controller-manager`'s `pkg/controller/podautoscaler`):

```
ratio   = currentMetricValue / desiredMetricValue           (a per-pod average, in milli-units)
desired = ceil(currentReplicas × ratio)   — unless 1 − tol_down ≤ ratio ≤ 1 + tol_up   (tolerance 0.1 each way)
```

Worked (`pods_metric_replicas()`): 3 pods at 200m against a 100m target → 6; 4 pods at 50m → 2; 5 pods at a ratio of
1.08 → 5 (inside the band). Then:

1. **Not-ready and missing pods.** Pods with no metric count as using the target on a scale-down and 0 on a scale-up;
   pods not yet ready count as 0 on a scale-up. Two ready pods with 10 queued each (target 2) want 10 replicas — but
   once 8 more are starting, `(10 + 10 + 0 × 8) ÷ 10 ÷ 2 = 1.0` is inside the band and the HPA holds at 10. Replicas
   still loading weights damp the next decision instead of doubling it.
2. **Multiple metrics:** compute each, take the largest.
3. **Stabilization** (`HPA.step()`): scale-down uses the **maximum** recommendation of the last
   `stabilizationWindowSeconds` (default 300); scale-up the minimum over its window (default 0).
4. **Policies**: per `periodSeconds`, each policy allows `Pods: +value` or `Percent: ×(1 ± value/100)` relative to
   the replica count at the start of the period; `selectPolicy: Max` (default) picks the policy allowing the most
   change, `Min` the least, `Disabled` none. Defaults: scale-up `max(+4 pods, +100 %)` per 15 s — from 3 replicas
   toward 40: 7, 14, 28, 40; scale-down `−100 %` per 15 s after the window. A `Percent` scale-down truncates the
   remaining count: 10 % per 60 s takes 80 → 72, holds while those 8 are inside the period, then 72 → 64.
5. **Clamp** to `[minReplicas, maxReplicas]`.

Per-HPA tolerances live under `behavior.scaleUp.tolerance` / `scaleDown.tolerance` (the `HPAConfigurableTolerance`
feature). Object and External metrics with an `AverageValue` target use `desired = ceil(total ÷ target)`
(`external_metric_replicas()`: 95 queued against 10 per pod → 10), which also works from zero replicas.

### 4.2 Which signal

The rule assumes the metric **grows in proportion to load per replica**, **rises before latency does**, and **is not
capped**. One L4 replica under rising chat load (simulated, notebook 03):

| req/s | "GPU util" | running | KV usage | waiting | TTFT p95 |
|---|---|---|---|---|---|
| 0.2 | 0.70 | 1.8 | 0.04 | 0 | 0.24 s |
| 0.5 | 1.00 | 4.4 | 0.09 | 0 | 0.25 s |
| 2.0 | 1.00 | 26 | 0.35 | 0.3 | 0.31 s |
| 3.5 | 1.00 | 60 | 0.75 | 0.3 | 0.64 s |

"GPU utilisation" — the fraction of time a kernel runs, what `nvidia-smi` reports — is 1.00 at a seventh of the load
the replica can serve within a 1 s TTFT SLO, because continuous batching keeps a step running whenever any request is
in flight (layer 02 §8 has the DCGM fields that measure more). Then a 1.5 → 7.5 req/s step with each HPA on one signal
(min 1, max 8, 30 s cold start; simulated, notebook 03):

| Signal (target) | TTFT p95 | SLO attainment | GPU-hours | Behaviour |
|---|---|---|---|---|
| GPU util (0.7) | 0.26 s | 1.00 | 1.80 | scales to max early and never scales down |
| GPU util (0.95) | 428 s | 0.04 | 0.33 | never scales |
| waiting per pod (2) | 17.5 s | 0.89 | 1.63 | sawtooth: the queue empties, the HPA scales down into the next cliff |
| KV usage per pod (0.6) | 7.3 s | 0.90 | 0.79 | proportional but capped at 1.0: climbs one cold start at a time |
| in-flight total, External (40 per pod) | 3.6 s | 0.94 | 0.91 | tracks demand; the rest of its tail is the cold start |

In-flight work — running plus waiting, including requests the router or gateway is holding — is the signal llm-d's
KEDA path uses (EPP flow-control queue plus running requests). Take the target from a load test: the per-replica
value at the highest load that still meets the SLO, minus a margin (notebook 03's `pick_target`). Latency-driven
variants scale on the ratio of estimated or measured TTFT to the SLO.

### 4.3 Cold start anatomy

From "the HPA said +1" to "the replica serves" (`fleetsim.autoscale.ColdStart`): **node** (0 if a GPU node is free;
minutes if the cluster autoscaler must create one — or longer if the GPU is not obtainable, layer 03 §7), **image**
(serving images are around 10 GB), **weights** (bytes ÷ storage bandwidth, layer 01 §6), **engine warm-up** (KV
allocation, CUDA-graph capture, compilation). `ColdStart.estimate()` with a free node, a 10 GB image at 0.25 GB/s,
16 GB of weights at 0.5 GB/s and 30 s of warm-up gives 40 + 32 + 30 = **102 s** (assumptions to replace with
measurements). The Kubernetes-side fixes — image streaming, secondary boot disks, GCS FUSE, Hyperdisk ML, startup
probes — are layer 03 §8.

The cold start sets how much queues during a step: `backlog = (peak − capacity now) × cold start`. With the
in-flight signal (simulated, notebook 03): cold start 0 s → TTFT p95 0.32 s; 30 s → 3.6 s; 120 s → 75 s; and 120 s
with `minReplicas: 3` → 0.29 s on *fewer* GPU-hours (0.75 vs 1.36), because the reactive fleet over-scaled to drain
its backlog. Warm headroom is a cost-and-latency decision, not a waste line.

### 4.4 Scale to zero, and KEDA

A pod metric cannot be read from zero pods, so `minReplicas: 0` needs an Object or External metric (the API server
rejects it otherwise) and the `HPAScaleToZero` feature. KEDA is the common route: it runs the 0 ↔ 1 activation
itself and creates an HPA with External metrics for 1 ↔ N; its Prometheus scaler can read the EPP's queue metrics
directly. The price is the first requests of every burst eating a full cold start. For six 20-minute bursts a day at
1 req/s with a 102 s cold start, scale-to-zero saves about 21 idle L4-hours a day (≈ $15 at an assumed
$0.70/hour — verify) and delays about 600 requests a day by ~100 s: right for an internal batch tool, wrong for a
customer-facing chat. Sleep/wake mechanisms that keep weights resident (llm-d's fast model actuation) shrink the
price without keeping a replica hot.

### 4.5 SLA planners

A planner replaces "one metric, one target" with a model of the engine: NVIDIA Dynamo's Planner scales prefill and
decode pools separately, with optimisation targets `throughput` (default: queue and KV thresholds), `latency`, `load`
(user thresholds on prefill queue tokens and decode KV use) and `sla` (TTFT/ITL targets against a performance model
tuned online from per-iteration engine metrics). llm-d's Workload Variant Autoscaler, which optimised replicas across
hardware variants by cost, is deprecated in favour of the KEDA + EPP paths.

---

## 5. Prefill/decode disaggregation

### 5.1 What it removes

A prefill chunk and the decodes batched with it share one step. With a 2,048-token budget on the L4 model, a
decode step of ~65 ms becomes a 545 ms step whenever a long prompt is being prefilled: on four aggregated replicas
serving ~6,000-token RAG prompts, ITL p50 was 64 ms and ITL p99 545 ms (simulated, notebook 04). Moving prefill to its
own pool leaves decode steps uninterrupted (ITL p99 70 ms in the same test), and lets each pool be batched and
parallelised for its own bottleneck — or run on different hardware. The mechanism is layer 01's
[deployment primer §8](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md); the systems papers
are DistServe and Splitwise.

### 5.2 What it costs: the KV transfer

```
KV bytes    = prompt tokens × KV bytes per token          (2 × layers × KV heads × head dim × bytes)
transfer_s  = latency + KV bytes × 8 ÷ link Gb/s          (× (1 − overlap) if streamed layer by layer)
```

`fleetsim.disagg.transfer_s()`: a 4,096-token prompt on an 8B model (131,072 B/token) is 512 MiB — 10.7 ms over
400 Gb/s RDMA, 43 ms over 100 GbE, 0.43 s over 10 GbE; on a 70B model (327,680 B/token) 1.34 GB, 26.8 ms at
400 Gb/s. Judge it against the prefill it replaces: 6,000 tokens take 1.59 s to prefill on the L4 model but 0.19 s on
an H100, so a 10 % overhead budget admits 100 GbE for the L4 and only 400 Gb/s RDMA or NVLink-class links for the
H100 (notebook 04). A faster GPU makes the network matter more. In the simulator the transfer adds straight to TTFT:
the ~6,060-token prompts of notebook 04 take 0.64 s to cross 10 GbE and 16 ms to cross 400 Gb/s (`transfer_s()`).

### 5.3 Sizing the P:D ratio

```
prefill replicas = rate × input tokens × (1 − hit rate) ÷ (prefill tokens/s × utilisation cap)
decode replicas  = rate × output tokens ÷ (batch ÷ decode step)     at the largest batch meeting the ITL SLO and KV
```

`fleetsim.disagg.pd_plan()` for 1.2 req/s of 6,060 input / 250 output tokens on the L4 model with an 80 ms ITL SLO:
prefill 7,272 tokens/s ÷ (3,781 × 0.7) = **2.75 replicas**; decode batch 5 (KV-bound: 6,185 tokens of context is 387
blocks each, `max_decode_batch()`), 71.6 output tokens/s per replica → **4.19 replicas**. Simulating every split of
eight replicas (`search_pd()`, notebook 04; SLO TTFT ≤ 3 s, TPOT ≤ 80 ms), **3P5D** won with 0.86 SLO attainment,
aggregated serving reached 0.68 (the chunk stall breaks the tight TPOT), and lopsided splits collapsed — 1P7D to a
106 s TTFT p95, 7P1D to a 2.5 s TPOT p95.

### 5.4 When it hurts

- **Short prompts.** There is no stall to remove; the extra hop and transfer are pure cost. Disaggregate
  conditionally: on a chat + RAG mix, a 2P6D fleet reached 0.80 SLO attainment disaggregating everything and 0.92
  sending only prompts of 2,048+ uncached tokens to the prefill pool (simulated, notebook 04). llm-d's P/D sidecar
  makes that decision per request.
- **Slow links** (the transfer rivals the prefill) and **fast GPUs on ordinary networks**.
- **The wrong ratio or a small fleet.** Two pools fragment capacity; aggregated serving multiplexes both phases on
  every GPU. Of the seven splits of eight L4s in §5.3, only 3P5D beat aggregated serving.
- **Before tuning the chunk budget.** With `max_num_batched_tokens` 256 instead of 2,048, the eight aggregated L4s
  reached 0.94 SLO attainment and an ITL p99 of 71 ms — as good as the best split, with one pool. (The simulator has
  no per-chunk efficiency loss, so this is the optimistic end; bigger models shorten decode steps relative to a
  chunk, which is where splitting wins.) The llm-d guide recommends P/D for medium-large models and long inputs
  ("10k ISL | 1k OSL, not 200 ISL | 200 OSL").

### 5.5 The implementations

- **vLLM KV connectors** (`--kv-transfer-config`): the prefill instance is a `kv_producer`, the decode instance a
  `kv_consumer`. `NixlConnector` (NVIDIA's NIXL transfer library over UCX, RDMA or TCP) is llm-d's default and
  supports different TP on the two sides; `MooncakeConnector` requires equal TP.
- **llm-d**: the EPP runs a prefill and a decode scheduling profile; a sidecar next to each decode server
  orchestrates prefill → KV pull → decode. Its guide deploys `gpt-oss-120b` as 8 TP=1 prefill + 2 TP=4 decode
  instances — heterogeneous parallelism, **xPyD** with x = 8, y = 2 — and tunes the x:y ratio to the ISL/OSL mix.
- **NVIDIA Dynamo** (1.0 GA March 2026): disaggregated serving across vLLM, SGLang and TensorRT-LLM, NIXL transfers,
  the KV-aware router of §2.4 and the Planner of §4.5.

---

## 6. KV cache beyond HBM

### 6.1 The working set of agent sessions

An agent turn re-sends the whole history and then waits — for a tool, a sandbox, a human — before the next turn
([07 long-running agents](../../07-application-agent-framework/long-running-durable/00_primer.md) describes the
workloads). `working_set_gb()`: 200 concurrent sessions at 30,000 tokens on an 8B model need
200 × 30,000 × 131,072 B = **786 GB** of KV — 14.6 H100s' worth of KV pool (54 GB each on `H100_8B`), before any
of them generates a token. HBM holds the running requests; idle sessions get whatever is left, and LRU evicts them
during their tool call. On the four-L4 fleet of §2.5, lengthening tool pauses from 1–4 s to 10–40 s dropped the
EPP's hit rate from 0.87 to 0.57 and raised TTFT p95 from 0.37 s to 1.7 s (simulated, notebook 05): no router can
route to KV that no longer exists.

### 6.2 Fetch or recompute?

```
onload_s    = latency + tokens × KV bytes/token ÷ tier bandwidth
recompute_s = tokens ÷ prefill tokens/s
fetch wins when   tier bandwidth > KV bytes/token × prefill tokens/s          (breakeven_gb_s)
```

The break-even is the rate at which prefill *produces* KV: **0.50 GB/s** for the 8B model on an L4, **4.05 GB/s** on an
H100 (`breakeven_gb_s()`). Ten thousand tokens on the H100: recompute 324 ms, fetch from host DRAM at an assumed
50 GB/s 27 ms, fetch over 25 GbE 439 ms — slower than recomputing. Any tier helps an L4; an H100 wants PCIe DRAM, fast
NVMe or RDMA. (Attention also grows with context, which makes long-context recompute worse than this linear model;
bandwidths are assumptions to measure — layer 01 §5–6.)

### 6.3 The tiers and the software

| Tier | Typical role | Software (2026) |
|---|---|---|
| HBM | running requests + hot prefixes | the engine's prefix cache |
| host DRAM | recently idle sessions; the default offload tier | vLLM `OffloadingConnector` (llm-d's recommended native path), SGLang HiCache, LMCache |
| local NVMe / filesystem | larger, slower second tier | vLLM multi-tier offloading, LMCache, Dynamo KVBM (GPU → CPU → SSD → remote) |
| remote / shared store | cross-replica sharing, survives replica loss | Mooncake Store, LMCache server; llm-d P2P KV sharing (experimental) |

`fleetsim.kvtier.TieredKV` models exclusive LRU tiers with demotion on eviction. llm-d's tiered-prefix-cache guide
defaults to 100 GB of CPU offload per Qwen3-32B replica on H100s (verify for your deployment).

### 6.4 Where the KV lives decides where the request can go

`simulate_sessions()` replays 600 agent sessions on four H100 replicas that keep ~8 GB of HBM for idle sessions
(an assumption), scoring the resumed turns (simulated, notebook 05):

| Tiers, routing | Served from HBM | from DRAM / shared | Recomputed | Prefix cost p95 |
|---|---|---|---|---|
| HBM only, sticky | 23 % | — | 77 % | 292 ms |
| + 16 GB DRAM per replica, sticky | 23 % | 77 % | ~0 % | 52 ms |
| + 64 GB DRAM per replica, random | 6 % | 56 % | 39 % | 203 ms |
| HBM + a shared 1 TB store (20 GB/s), random | 5 % | 95 % | 0 % | 86 ms |

A local tier helps only the sessions that come back to it, so offloading and **session-sticky routing** go together;
a **shared** tier lets any replica resume any session, which gives routing back its freedom to chase load. The
recomputed-token floor (~18 % here) is the new tool output every turn must prefill anyway.

---

## 7. Multi-model, multi-LoRA and model routing

- **One pool per base model.** An `InferencePool` selects the pods of one model server deployment (llm-d assumes one
  base model per pool); several models mean several pools, and the gateway picks the pool — by path, header, or the
  model name in the body (body-based routing, now in `llm-d-inference-payload-processor`).
- **Adapters ride the base model.** LoRA adapters are small next to the base weights, so one replica can hold many,
  but a batch mixes only a few (vLLM `--max-loras`) and loading one takes time. Routing therefore prefers replicas
  with the adapter already resident, then replicas with a free slot (`LoraAffinityFilter`; the engine side is
  `Replica._lora_slot`, which skips — rather than blocks on — a request whose adapter has no slot, like vLLM).
  `vllm:lora_requests_info` reports running and waiting adapters.
- **Model rewrite and canaries.** `InferenceModelRewrite` (llm-d Router) rewrites the requested model name — a stable
  public name mapped to versioned adapters or models — which is how A/B tests and canary rollouts are expressed.
- **Model routing proper** — choosing a cheaper model per request — is a gateway decision
  ([06](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1.4); this layer
  routes among replicas of the model already chosen.

---

## 8. Large MoE topologies (wide-EP) in brief

Mixture-of-experts models such as DeepSeek-R1 are served with **expert parallelism** across many GPUs and nodes and
**data-parallel attention**: every rank runs attention for its own requests with its own KV cache, and each MoE layer
exchanges tokens with the experts' ranks in an all-to-all (layer 02 §5 has the collective, layer 01 §5 the fabric
cost). Spreading experts thin frees HBM for KV and allows very large batches — llm-d's wide-EP guide deploys
DeepSeek-R1-0528 on 32 H200 or B200 GPUs as 16-way data-parallel prefill plus 16-way data-parallel decode,
disaggregated with NIXL over InfiniBand or RoCE. For the router, each DP rank is an endpoint (a multi-port model
server, "DP-aware scheduling"), so prefix and load scoring apply per rank; multi-host groups are scheduled as a unit
with LeaderWorkerSet (layer 03 §4). Load imbalance moves inside the model — hot experts — which the engine balances,
not the router.

---

## 9. The Kubernetes-native stack, September 2026

```
 client ─► Gateway (Envoy / GKE L7 LB)  ── HTTPRoute ──► InferencePool (GIE v1: selector, targetPorts, endpointPickerRef)
                       │  ext-proc                                   │
                       ▼                                             ▼
            llm-d Router's endpoint picker (EPP) ─── picks ───► model-server pods (vLLM, SGLang, ...)
            filters · scorers · picker · flow control              KV events, /metrics, P/D sidecar
            InferenceObjective (priority) · InferenceModelRewrite
```

- **Gateway API Inference Extension (GIE)** is GA and now hosts only the `InferencePool` API
  (`inference.networking.k8s.io/v1`: `selector.matchLabels` and 1–8 `targetPorts` required; `endpointPickerRef` with
  `failureMode` `FailClose` or `FailOpen`; `appProtocol` `http` or `kubernetes.io/h2c`), the endpoint-picker protocol
  (Envoy ext-proc), a lightweight reference EPP and conformance tests. Any Gateway implementation with ext-proc
  support becomes an inference gateway.
- **llm-d Router** (repository `llm-d/llm-d-router`; the "Inference Scheduler" renamed) is the EPP plus the
  request-management APIs moved out of GIE: `InferenceObjective` (`llm-d.ai/v1alpha2`) and `InferenceModelRewrite`;
  configuration is an `EndpointPickerConfig` (`llm-d.ai/v1`). It runs **standalone** (Helm chart, self-managed Envoy as
  a sidecar or a service) or in **Gateway mode** behind an InferencePool and HTTPRoute.
- **llm-d** (CNCF Sandbox) packages "well-lit paths": optimized baseline, precise prefix-cache routing, tiered prefix
  cache, P/D disaggregation, wide-EP, flow control, workload autoscaling, agentic serving, batch serving.
- **GKE Inference Gateway** is GKE's managed Gateway implementation of the same model: InferencePool v1 (GKE manages
  the CRD from 1.34.0-gke.1626000), an EPP, `gke-l7-regional-external-managed` or internal GatewayClasses, a
  proxy-only subnet, Managed Prometheus for autoscaling metrics.
- **NVIDIA Dynamo** (1.x) is a full serving stack with its own frontend and KV router, or an EPP behind a GIE gateway.
- **Ray Serve LLM** serves engines as Ray deployments with Ray's autoscaling and routing — the choice when the
  platform is Ray (GKE has a Ray operator add-on). **KServe** wraps engines in Kubernetes serving resources, with an
  LLM-specific resource built on llm-d components (verify versions).

They compose rather than compete: the Gateway API routes, the EPP picks endpoints, the engine batches, and
autoscaling (HPA/KEDA, a planner) sizes pools.

---

## 10. Where to run it

| Concept | Laptop / CI (T0) | Any GPU box (T1/T2) | GCP (T3) |
|---|---|---|---|
| routing, flow control | `fleetsim` notebooks 01–02; the lab's router with fake backends | the lab's router in front of vLLM | GKE Inference Gateway + EPP |
| autoscaling | notebook 03; the lab's HPA recommender | HPA/KEDA on a kind or k3s cluster | HPA on Managed Prometheus metrics, L4 Spot pool 0→N |
| P/D, KV tiers | notebooks 04–05 | vLLM with NIXL on a 2-GPU box; LMCache or the offloading connector | multi-node GPUs with RDMA (A3 Ultra / A4 families) |
| the whole stack | kind + llm-d Router standalone + `llm-d-inference-sim` (no GPU) | Docker compose with real vLLM | the lab's Terraform + manifests |

Cheap and free GPUs (Colab, Kaggle, RunPod, Vast, Lambda) and GCP obtainability are in
[`COMPUTE.md`](../../COMPUTE.md); the order to work the material is in [`CURRICULUM.md`](../../CURRICULUM.md). The
[`inference-gateway-lab`](inference-gateway-lab/) carries these concepts to real servers.

---

## In a design review

**The two-minute walkthrough.** "Above the engines there are three decisions. *Which replica*: the replicas are
caches, so I route on prefix affinity and load together — the llm-d endpoint picker with a prefix-affinity filter or
prefix scorer and a load scorer, tuned on replayed traffic while watching hit rate and per-replica load; flow control
holds bursts in the router with priorities per workload. *How many*: an HPA through KEDA on in-flight work per
replica — running plus waiting, including the router's queue — with the target from a load test at the SLO, the
5-minute scale-down window kept, and the cold start attacked term by term; utilisation is useless because continuous
batching pegs it. *How to split*: aggregated with a tuned chunk budget until ITL p99 or model size says otherwise;
then xPyD sized from the input/output ratio, conditional on prompt length, over RDMA. For agent sessions, a DRAM
offload tier on every replica and session-sticky routing — or a shared KV store if we must rebalance freely."

**Drills**

1. *Round-robin keeps request counts perfectly even. Why is the tail still bad?* Requests differ by one to two orders
   of magnitude in prefill, KV and residency; equal counts are unequal work, and round-robin ignores both queues
   and caches. Least-waiting or power-of-two on in-flight load halves p95 on the notebook-01 mix.
2. *How do you stop a popular system prompt from melting its replica without losing its cache hits?* Bounded-load
   consistent hashing (no replica above ceil((1 + ε) × average)), or a weighted picker whose load scores can outvote
   the prefix score, or sticky-until-saturated with a TTFT penalty gate. Pure prefix hashing hit 48.7 s p95 in the
   notebook-02 test.
3. *What does the HPA do with 2 ready pods at 10 queued each (target 2) while 8 new pods start?* It counts the starting
   pods as 0 on a scale-up: (20 + 0) ÷ 10 ÷ 2 = 1.0, inside the tolerance band, so it holds at 10 instead of asking
   for 50.
4. *Why not autoscale on `num_requests_waiting` alone?* It is ~0 whenever capacity suffices, so the HPA scales down,
   the queue returns, and the fleet saws; pair it with running requests (the HPA takes the max over metrics) or
   scale on in-flight work.
5. *When is P/D disaggregation a bad idea?* Short prompts, slow links (a 4k prompt of an 8B model is 0.43 s over
   10 GbE), a P:D split that does not match the input/output mix, small fleets, and before trying a smaller prefill
   chunk.
6. *A coding agent's TTFT climbs every turn at peak. What do you check?* Its KV working set (sessions × context ×
   bytes/token) against the HBM left after running requests; then add a DRAM offload tier (it beats recompute when
   its bandwidth exceeds KV bytes/token × prefill tokens/s) and keep routing session-sticky.

---

## Glossary

| Term | Meaning |
|---|---|
| **EPP** | Endpoint picker: the service a gateway consults (Envoy ext-proc) to choose the model-server endpoint for a request. |
| **InferencePool** | GIE resource grouping model-server pods and naming their endpoint picker; the backend of an HTTPRoute. |
| **InferenceObjective** | llm-d Router resource giving requests for a pool a priority (and future objectives). |
| **Prefix cache / hit rate** | Reuse of KV blocks for a shared token prefix; hit rate = cached prompt tokens ÷ prompt tokens. |
| **Approximate / precise index** | The router's guess of each replica's cache from what it sent / the engines' KV events. |
| **Power of two choices** | Sample two endpoints at random, pick the less loaded. |
| **Consistent hashing with bounded loads** | Hash-ring affinity where no endpoint exceeds (1 + ε) × average load. |
| **Flow control** | Router-side queues per flow (fairness id, priority), dispatched when endpoints are not saturated. |
| **Saturation detector** | The flow-control component that decides whether an endpoint can take more work. |
| **HPA** | Horizontal Pod Autoscaler; `ceil(current × metric ÷ target)` with tolerance, stabilization and policies. |
| **KEDA** | Event-driven autoscaler that feeds External metrics to an HPA and handles scaling to and from zero. |
| **Cold start** | Time from scale-up decision to a serving replica: node, image, weights, warm-up. |
| **P/D disaggregation, xPyD** | Prefill and decode on separate pools; x prefill and y decode instances. |
| **NIXL** | NVIDIA Inference Xfer Library: point-to-point KV/tensor transfer over RDMA, NVLink or TCP. |
| **KV connector** | vLLM's plug-in interface for moving KV out of the engine (P/D transfer, offload). |
| **KV offload / tier** | Keeping evicted KV blocks in DRAM, NVMe or a remote store to fetch instead of recompute. |
| **Working set** | The KV all live sessions would need resident: sessions × context × bytes/token. |
| **Goodput** | Requests per second meeting every SLO (TTFT and TPOT here). |
| **Wide-EP** | Expert parallelism across many GPUs/nodes for large MoE models, usually with DP attention. |

---

## Sources

- Gateway API Inference Extension — README and `InferencePool` v1 CRD, `github.com/kubernetes-sigs/gateway-api-inference-extension` (fetched 2026-09-26).
- llm-d Router — README, `docs/architecture.md`, plugin READMEs (`prefix-cache-scorer`, `queue-scorer`, `kv-cache-utilization-scorer`, `token-load-scorer`, `prefix-cache-affinity-filter`, `active-request-scorer`, `load-aware-scorer`), `InferenceObjective` types and CRD, `github.com/llm-d/llm-d-router` (fetched 2026-09-26).
- llm-d guides — optimized baseline, precise prefix-cache routing, tiered prefix cache, P/D disaggregation, flow control, workload autoscaling, agentic serving, wide-EP, `github.com/llm-d/llm-d/tree/main/guides` (fetched 2026-09-26).
- NVIDIA Dynamo — README, router design, planner guide, `github.com/ai-dynamo/dynamo` (fetched 2026-09-26).
- Kubernetes — Horizontal Pod Autoscaling (algorithm details, behavior, tolerance, scale to zero) and feature-gate pages, `github.com/kubernetes/website` (fetched 2026-09-26); controller source `kubernetes/kubernetes` `pkg/controller/podautoscaler`.
- vLLM — V1 metrics (`vllm/v1/metrics/loggers.py`), automatic prefix caching design, KV connectors / disaggregated prefill docs.
- Azar, Broder, Karlin, Upfal, "Balanced Allocations", STOC 1994 / SIAM J. Comput. 1999; Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing", IEEE TPDS 2001; Mitzenmacher, "How Useful Is Old Information?", IEEE TPDS 2000.
- Mirrokni, Thorup, Zadimoghaddam, "Consistent Hashing with Bounded Loads", SODA 2018; Karger et al., "Consistent Hashing and Random Trees", STOC 1997.
- Zhong et al., "DistServe", OSDI 2024; Patel et al., "Splitwise", ISCA 2024; Agrawal et al., "Sarathi-Serve", OSDI 2024; Qin et al., "Mooncake", FAST 2025; Kwon et al., "PagedAttention", SOSP 2023; Zheng et al., "SGLang / RadixAttention", 2024.
- LMCache (`lmcache.ai`), Mooncake (`github.com/kvcache-ai/Mooncake`), KEDA (`keda.sh`), Envoy load balancers (least request, ring hash / Maglev).

## Verify list

Checked 2026-09-26; re-check before relying on any of it.

- GIE is GA; `InferencePool` is `inference.networking.k8s.io/v1`; EPP, `InferenceObjective` (`llm-d.ai/v1alpha2`) and `InferenceModelRewrite` live in `llm-d/llm-d-router`; `EndpointPickerConfig` is `llm-d.ai/v1` (`v1alpha1` deprecated).
- GKE manages the InferencePool v1 CRD from 1.34.0-gke.1626000; GatewayClass names `gke-l7-regional-external-managed` and `gke-l7-rilb`; a proxy-only subnet is required (verify).
- llm-d optimized baseline = `prefix-cache-affinity-filter` (threshold 0.80, `maxTTFTPenaltyMs` 18,000, `peakPrefillThroughput` 15,928 for Qwen3-32B on 2× H100 TP=2 with vLLM 0.19) + `token-load-scorer` (`queueThresholdTokens` 4,194,304). EPP metrics refresh 50 ms.
- llm-d flow control: feature gate `flowControl`; default `global-strict-fairness-policy`, `fcfs-ordering-policy`, `utilization-detector`; production guidance `concurrency-detector`.
- llm-d workload autoscaling: KEDA + EPP metrics recommended; Workload Variant Autoscaler deprecated (final v0.9.0).
- Kubernetes HPA: sync 15 s; tolerance 0.1; scale-down window 300 s; default policies as in §4.1. `HPAConfigurableTolerance`: alpha 1.33, beta 1.35, GA 1.37. `HPAScaleToZero`: alpha (off) through 1.36, beta (on by default) from 1.37 — check your cluster's version and feature gates.
- NVIDIA Dynamo 1.0 GA 2026-03-16; runtime images 1.5.0 at the time of writing; Planner targets `throughput`, `latency`, `load`, `sla`.
- vLLM metric names as listed in §2.1 (`vllm:kv_cache_usage_perc`; older releases `vllm:gpu_cache_usage_perc`); `NixlConnector` is llm-d's default P/D connector; `MooncakeConnector` needs equal TP.
- Spec-sheet inputs to the engine profiles: L4 24 GB, 0.3 TB/s, 121 dense bf16 TFLOP/s; H100 SXM 80 GB, 3.35 TB/s, 989 dense bf16 TFLOP/s (layer 01 has the dated catalogue).
- Tier bandwidths in §6 (DRAM 50 GB/s over PCIe, NVMe 6 GB/s, 20 GB/s RDMA store) are assumptions; the L4 price ($0.70/hour on demand, `g2-standard-4`) is approximate.
- KServe's LLM resource and Ray Serve LLM feature details (P/D, prefix-aware routing) — check current releases.
