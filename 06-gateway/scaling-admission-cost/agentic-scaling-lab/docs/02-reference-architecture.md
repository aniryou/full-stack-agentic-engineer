# Reference architecture — customer-support agent on Cloud Run + Gemini

This is the production shape that the lab's core concepts (`scalelab/`) scale up to: what runs where,
how each part scales, what its limits are and why the alternatives were not chosen. The scaling reasoning
is in `01-scaling-primer.md`; the numbers are in `03-capacity-plan.md`; the mapping from each core concept
to a Google Cloud service is in `04-gcp-mapping.md`. Product facts were checked on 5 September 2026.

## 1. Context and requirements

**Workload.** Meridian Mobile's in-app and web support agent. Customers ask about billing,
connectivity, plans, roaming and complaints; the agent answers from the CRM, the billing system,
network operations and the help centre, and can open tickets or change plans.

**Functional requirements.** Multi-turn conversations with streamed answers; tool use with reads and
a small number of writes; escalation to a human by ticket; a bounded, auditable transcript per
session; one tenant in v1 with the seams for more.

**Non-functional requirements (targets, per turn unless stated).**

| Requirement | Target | Notes |
|---|---|---|
| Volume | 100,000 conversations/day; peak ×3; incident ×10 | see capacity plan |
| Latency | first progress event p95 ≤ 2.5 s; turn p95 ≤ 8 s | the visible answer starts after tools |
| Availability | 99.5 % of admitted turns end with an answer (30-day) | shed is a separate SLI, target ≤ 2 % at peak, unbounded in incidents |
| Cost | ≤ $0.08 per conversation at pay-as-you-go prices | routed + cached: $0.068 |
| Durability | no lost turns; no duplicated side effects | at-least-once + idempotency |
| Residency | none in v1 | global endpoint; regional variant noted |
| Security | user identity at the edge; least-privilege service identities; no secrets in code | IAP, per-service accounts |
| Operability | one dashboard, seven alerts, canary rollouts, a kill switch on cost | section 3.7 |

**Constraints.** Model capacity is the organisation's Standard PayGo Flash tier (10 M TPM) plus an
optional Provisioned Throughput commitment; the billing mainframe accepts 40 QPS and the CRM 200 QPS;
Cloud Run is Meridian's standard compute; Python 3.11.

## 2. System overview

```mermaid
flowchart TB
  subgraph Edge
    LB[Global external ALB<br/>managed cert · Cloud Armor · IAP]
  end
  subgraph Run["Cloud Run (region)"]
    GW[gateway<br/>request-based · conc 250 · 2..100]
    OR[orchestrator<br/>instance-based · conc 80 · 2..50]
    TS[tools<br/>request-based · conc 200 · 1..20]
    MCP[mcp (optional)<br/>streamable HTTP]
  end
  subgraph Async
    PS[(Pub/Sub agent-turns<br/>push · OIDC · ordering key)]
    DLQ[(agent-turns-dlq)]
  end
  subgraph State
    FS[(Firestore<br/>sessions/{id}/turns/{id})]
    RS[(Memorystore Valkey/Redis<br/>streams · locks · buckets · idem · level)]
  end
  subgraph Model["Gemini Enterprise Agent Platform"]
    GM[Gemini 3.5 Flash / 3.5 Flash-Lite / 3.1 Pro<br/>global endpoint · PT + spill-over · caching]
  end
  Client((App / web)) --> LB --> GW
  GW -->|publish| PS -->|push| OR
  PS -.->|after 5 attempts| DLQ
  OR --> GM
  OR -->|ID token| TS
  OR -.->|optional| MCP
  GW <--> RS
  OR <--> RS
  GW --> FS
  OR <--> FS
  OT[Cloud Trace · Monitoring · Logging] -.- GW
  OT -.- OR
```

Three Cloud Run services, one queue, two stores, one model endpoint. Every arrow is authenticated:
the load balancer to the gateway by IAP (or the dev bearer scheme locally), Pub/Sub to the
orchestrator by an OIDC token from a dedicated service account with `run.invoker`, the orchestrator to
the tool services by its own identity token, the services to Google APIs by their service accounts.

## 3. Components

### 3.1 Gateway

Responsibilities: authenticate; create sessions (prefetching the customer's profile so the first
turn does not pay a CRM round-trip); admit or shed turns; enqueue; relay the turn's event stream to
the client as SSE, with resume by `Last-Event-ID`.

| Aspect | Design |
|---|---|
| API | `POST /v1/sessions`, `POST /v1/sessions/{id}/messages` (submit + stream), `POST /v1/sessions/{id}/turns` (202), `GET /v1/sessions/{id}/turns/{tid}/events` (SSE), `GET /v1/sessions/{id}`, `/healthz`, `/readyz`, `/metrics` |
| Admission | per-tenant token bucket (429 + `Retry-After`) → degrade level from shared signals → in-flight cap (503 + `Retry-After`); priority 1–2 bypasses the cap |
| Degrade level | computed from in-flight turns, oldest queued turn age, model 429 ratio and breaker state; written to Redis with a 30 s TTL; hysteresis holds levels 1–2 for 15 s; level 3 follows the instantaneous cap |
| Scaling | request-based billing; concurrency 250; 1 vCPU / 512 MiB; min 2, max 100; timeout 600 s; ingress internal + load balancer |
| Limits it lives under | 1,000 concurrent requests and 800 req/s per instance; 60-minute request ceiling; 32 MiB for non-chunked responses (streams are exempt) |
| State it touches | Firestore (session and turn documents), Redis (buckets, gauge, level, streams), Pub/Sub (publish) |

### 3.2 Orchestrator (the lab's `loop.py`, run behind a queue)

Responsibilities: run one turn to completion under a budget; checkpoint each step; execute tools
through the executor; stream events; persist the transcript; compact history; publish model-health
signals; honour the Pub/Sub contract.

| Aspect | Design |
|---|---|
| Entry points | `POST /pubsub/turns` (push envelope → 200 ack / 503 nack), `POST /internal/turns/run` (direct; tests and the sync path) |
| Loop | plan call → (tool calls in parallel → answer call)* until a final answer; budgets: 6 steps, 6 model calls, 40 k tokens, $0.25, 45 s |
| Checkpoints | `StepRecord` per model call (response incl. provider parts) and per tool step (compacted results) appended to the turn document before the next action |
| Idempotency | writes keyed `{turn}:{step}:{i}:{tool}:{hash(args)}`; results stored in Redis for 24 h; replay returns the stored result |
| Concurrency control | session lock in Redis (`SET NX`, TTL = budget + 15 s); Pub/Sub ordering key = session id |
| Scaling | instance-based billing; concurrency 80; 2 vCPU / 2 GiB; min 2, max 50; startup CPU boost; timeout 600 s (= ack deadline); ingress internal |
| Failure policy | `RetryLater` → 503 (transient infrastructure); every other outcome → 200 with a terminal event (completed or gracefully failed) |

Turn state machine:

```mermaid
stateDiagram-v2
  [*] --> queued: gateway admits, publishes
  queued --> running: push delivered, lock taken
  running --> running: model step / tool step checkpointed
  running --> completed: final answer
  running --> failed: budget exceeded / provider error (graceful message)
  running --> queued: RetryLater (overloaded, store unavailable) → redelivery
  queued --> dead_letter: 5 delivery attempts
  completed --> [*]
  failed --> [*]
```

### 3.3 Model gateway (the lab's `resilience.py`, one per model)

A library in the orchestrator process. Per call: route (task × degrade level → model, thinking
level, output cap) → deadline check → circuit breaker → token bucket sized to this deployment's share
of the model's TPM → attempt with timeout → on 429/503/timeout retry with full-jitter backoff bounded
by the deadline, moving to a sibling model after two consecutive 429s or when the breaker is open →
account usage, cost, latency, TTFT and traffic type. Optional hedging for small non-streaming calls.

| Route | Task | Level 0 | Level 1 | Level 2 |
|---|---|---|---|---|
| route / summarise | classification, compaction | Flash-Lite, minimal thinking | same | same |
| plan / answer | the turn's main calls | 3.5 Flash, low thinking, 700 tokens | Flash-Lite, low, 400 | Flash-Lite, minimal, 250 |
| reason | rare hard cases | 3.1 Pro, medium | 3.5 Flash, medium | 3.5 Flash, medium |

Vertex specifics (see `scalelab/model.py::gemini_generate`): `genai.Client(enterprise=True, location="global")`,
PT request-type and spill-tier headers, explicit context caches for the stable prefix when it clears
4,096 / 6,144 tokens, thought-signature round-tripping, thinking-level clamping per model, error
mapping (429 → `RateLimited`, 5xx → `ServiceUnavailable`, 408/504 → `ProviderTimeout`).

### 3.4 Tool executor and tool services (the lab's `tools.py`)

The executor applies bulkhead → breaker → read cache → idempotency → timeout → retry (reads only) →
truncation, and returns structured errors to the model. Tools run in-process locally, as HTTP tool
services on Cloud Run in production (`tools_mode=http`), or behind an MCP server (`tools_mode=mcp`).
Tool specs carry their scaling policy: idempotent or not, timeout, per-instance concurrency cap,
retry attempts, cache TTL.

| Tool | Idempotent | Timeout | Bulkhead | Cache | Backing system |
|---|---|---|---|---|---|
| get_customer | yes | 2 s | 100 | 300 s | CRM (200 QPS) |
| get_invoice | yes | 4 s | 30 | 120 s | billing mainframe (40 QPS) |
| list_plans | yes | 2 s | 50 | 3600 s | CRM |
| check_network_status | yes | 1.5 s | 200 | 30 s | network ops |
| search_kb | yes | 1 s | 500 | 600 s | help centre |
| create_ticket | **no** | 3 s | 50 | — | CRM |
| change_plan | **no** | 3 s | 50 | — | CRM |

At degrade level 2 the registry withholds non-idempotent tools and tools with timeouts above 2 s.

### 3.5 State

| Store | Holds | Access pattern | Limits designed around |
|---|---|---|---|
| Firestore (Standard, Native, regional, PITR) | `sessions/{id}` (bounded transcript, summary, customer context, usage, version); `sessions/{id}/turns/{id}` (status, steps, usage, `expires_at` with a 30-day TTL policy) | a handful of writes per turn; one read per turn | 1 MiB/document; ~1 sustained write/s/document; 500 → +50 %/5 min collection ramp |
| Memorystore for Valkey (or Redis), STANDARD_HA | event streams per turn (5-minute retention), session locks, per-model and per-tenant token buckets (Lua), in-flight gauge, degrade level, idempotency markers, tool read cache, model-health signals | thousands of sub-ms ops/s | ~120 k ops/s per 2-vCPU node; reached over Direct VPC egress |
| Pub/Sub | `agent-turns` (push subscription, ordering, ack 600 s, retry 10–600 s), `agent-turns-dlq` (pull, 7 days) | tens of messages/s | push quota ≈ 10× below pull; 10 MB messages |

### 3.6 Edge

Global external Application Load Balancer with a serverless NEG on the gateway; managed certificate;
Cloud Armor policy with a throttle rule keyed on the `Authorization` header (600 requests per minute
per key) and a per-IP ban rule, plus the preconfigured SQLi/XSS rule sets; IAP on the backend
service for user identity (the gateway verifies the IAP assertion against the backend-service
audience). Cloud Run ingress is `internal-and-cloud-load-balancing`, so the armour cannot be bypassed.

### 3.7 Observability

OpenTelemetry spans per turn, model call and tool call with `gen_ai.*` attributes, exported to Cloud
Trace through the Telemetry API at a 10 % head-based sample; a process-local metrics registry exposed
at `/metrics` and, in production, exported through the OTLP endpoint; structured JSON logs. The
dashboard and the seven alert policies are provisioned with the rest of the estate (Terraform).

## 4. A turn, end to end

```mermaid
sequenceDiagram
  participant C as Client
  participant G as gateway
  participant R as Redis
  participant F as Firestore
  participant P as Pub/Sub
  participant O as orchestrator
  participant M as Gemini
  participant T as tools
  C->>G: POST /v1/sessions/{id}/messages
  G->>R: tenant bucket · level · inflight+1
  alt shed
    G-->>C: 429/503 + Retry-After
  end
  G->>F: create turn (queued)
  G->>P: publish {session, turn} (ordering key = session)
  G->>R: XADD turn.queued
  G-->>C: SSE stream opens (relay from the turn's stream)
  P->>O: push /pubsub/turns (OIDC)
  O->>R: SET NX lock:session
  O->>F: load session + turn (checkpoints)
  loop until final answer or budget
    O->>M: generate_content (stream)
    M-->>O: deltas · tool calls · usage
    O->>R: XADD step.model.delta …
    O->>F: ArrayUnion(step)
    O->>T: tool calls in parallel (idempotency keys)
    T-->>O: results
    O->>F: ArrayUnion(step)
  end
  O->>F: save session (optimistic version) · save turn
  O->>R: XADD turn.completed · inflight-1 · DEL lock
  O-->>P: 200 (ack)
  R-->>G: stream events
  G-->>C: SSE events … turn.completed
  O->>M: summarise (compaction, off the critical path)
```

## 5. Capacity and scaling design

The full plan is in `03-capacity-plan.md`; the settings it produces:

| Setting | Value | Derivation |
|---|---|---|
| in-flight cap | 85 if the tier baseline is the only capacity; ~250 with Provisioned Throughput for the base | token budget ÷ tokens per turn-second, then tuned from load tests |
| `admission.soft_inflight_ratio` | 0.8 | degrade before shedding |
| `admission.queue_age_degrade_s` / `queue_age_shed_s` | 10 s / 30 s | oldest queued turn |
| `admission.rate_limited_ratio_degrade` | 0.05 | level 1 at 5 %, level 2 at 15 % |
| `models.tpm_limit[gemini-3.5-flash]` | 4 M | this deployment's share of the 10 M baseline |
| `budget.*` | 6 steps, 6 calls, 40 k tokens, $0.25, 45 s | bounds cost of the worst turn to ~4× the average |
| orchestrator instances | 3 at peak, 8 in an incident | in-flight × 1.4 ÷ 80 |
| gateway instances | 2–3 | streams × 1.4 ÷ 250, min 2 |
| PT | 69 GSUs of 3.5 Flash on a 1-year term for the base; spill-over to Standard PayGo | break-even 75 % utilisation |

How each component scales:

| Component | Scales on | Bound by | Knob |
|---|---|---|---|
| gateway | open streams and request rate | 1,000 concurrency / 800 req/s per instance | concurrency, max instances |
| orchestrator | in-flight turns (memory per context) | model token budget, not CPU | concurrency, in-flight cap |
| model | tokens per minute | tier baseline + PT | routing, caching, compaction, PT, tiers |
| tools | downstream QPS | CRM 200, billing 40 | caches, prefetch, bulkheads, degrade |
| Redis | ops/s | node size | one node up to hundreds of turns/s |
| Firestore | ops/s and per-document writes | ramp rules | document layout |
| Pub/Sub | messages/s | regional quotas | none needed |

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| Model 429 / 503 / timeout | backoff with jitter within the deadline; sibling model after two 429s; breaker per model; counts toward the degrade level |
| Tool timeout / error / rate limit | structured error to the model; one retry for reads; breaker per tool; bulkhead protects the instance |
| Orchestrator instance dies mid-turn | Pub/Sub redelivers; checkpointed steps are replayed, not re-executed; writes deduplicated by idempotency key |
| Redis unavailable | gateway fails closed on admission (503 + Retry-After); orchestrator cannot take locks → `RetryLater` |
| Firestore unavailable | `RetryLater` → push backoff and redelivery; turn deadline still applies |
| Pub/Sub backlog grows | queue age raises the degrade level, then sheds; alert at 30 s |
| Poison turn | dead-letter after 5 attempts; DLQ alert; replay tooling |
| Overload | admission control: degrade levels 1–3 with hysteresis; priority classes bypass the cap |
| Model retirement / prompt change | model ids and prompt version in configuration; canary rollout; cached-token share on the dashboard |

## 7. Security

User identity terminates at the edge (IAP); the gateway trusts the IAP assertion and never handles
user credentials. Each service runs as its own service account with the minimum roles (gateway:
Pub/Sub publisher, Firestore user; orchestrator: Vertex AI user, Firestore user, invoker on the tool
services; tools: logging only). Tool services accept only the orchestrator's identity. Secrets (Redis
AUTH) come from Secret Manager. Prompt-injection screening is applied with Model Armor when enabled on
the model configuration; tool results are treated as data, truncated and never executed. Transcripts
carry a TTL. Cloud Armor rate-limits per caller before any compute is spent.

## 8. Deployment

Cloud Build builds the images; Terraform provisions everything else (VPC, Memorystore, Firestore, Pub/Sub,
IAM, the three Cloud Run services, the load balancer with Cloud Armor, monitoring, a budget); Cloud Deploy
rolls new revisions of the gateway and orchestrator as canaries (10 % → 50 % → 100 %) with a verify job
that runs a smoke test. Rollback is traffic back to the previous revision. What Terraform cannot do stays
manual: the Provisioned Throughput purchase, the PayGo tier, the IAP consent screen, DNS.

## 9. Alternatives considered (decisions)

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Runtime | Cloud Run | Agent Runtime (managed), GKE | Meridian's standard; every mechanism visible |
| Turn execution | Pub/Sub push + durable loop | synchronous HTTP; Workflows; Cloud Tasks | backlog observability, at-least-once, drain rate set by the bucket |
| Durable state | Firestore | AlloyDB, Spanner | document shape, TTL, PITR, no schema migration |
| Hot state and relay | Redis Streams | Pub/Sub per session; Firestore listeners | resume by sequence, sub-ms, cheap |
| Model capacity | PT for the base, PayGo spill-over | all PayGo; PT for peak | break-even arithmetic |
| Model tiers | 3.5 Flash + 3.5 Flash-Lite + 3.1 Pro (rare) | one model everywhere | 2× cost, latency |
| Overload | degrade levels + shedding at the edge | queue everything; scale instances | feedback loop |
| Transport | SSE | WebSockets | resume, plain requests, no affinity |

## 10. What this design does not do (v1)

No voice; no multi-region; no per-tenant PT allocation; no long-term memory beyond the session
summary; no autonomous multi-step writes without a confirming turn; no self-hosted models (section 11 sketches
that variant). Each has a documented trigger in the primer's growth path.

## 11. Variant: the model on your own GPUs (Mistral models, Kubernetes)

Section 10 leaves self-hosted models out of v1. When data residency, latency control or customisation rules the
hosted API out, the same design runs with an open-weight model on a vLLM fleet. Everything above the model gateway
keeps its shape — the gateway, the durable orchestrator, idempotent tools, degrade levels — with Kubernetes
equivalents in place of the managed services (RabbitMQ quorum queues for Pub/Sub, Postgres for Firestore, Redis for
Memorystore, KEDA for Cloud Run's autoscaler). What changes is the model layer and the numbers that size it.

The worked case is the Mistral provider's anchor, a Singapore telco at 100,000 conversations a day: the reasoning is in
[mistral/01-scaling-primer.md](mistral/01-scaling-primer.md), the numbers are the Mistral section of
[03-capacity-plan.md](03-capacity-plan.md) (`python -m scalelab.mistral`, fleet throughput estimated from first
principles by `scalelab/serving.py`), and the load test is notebook `05_hosted_or_own_gpus` (simulated). Product
facts were checked on 19 September 2026 (verify).

### 11.1 The model layer

**Hosted path.** `api.mistral.ai` is the global endpoint (EU-hosted by default, no committed inference
location); `api.eu.` and `api.us.` are regional at +10 %, GA since 11 August 2026, with function calling as
the only regional tool and no Agents, Batch or Files. Priority Tier is +75 %, enabled per request with
`service_tier="auto"`, falls back to standard when its limits are exceeded and reports `service_tier` in the
response; the docs state a 99.5 % SLA (the AI Cloud page says 99.9 %, verify).

The 2,700-token prefix is sent
first with a `prompt_cache_key` per prompt version; 64-token cached blocks are billed at 10 % and reported in
`usage.prompt_tokens_details.cached_tokens`; TTL and cache-write price are unpublished (the lab assumes
300 s).

The rate-limit request is 60 RPS, 19 M TPM and 209 B tokens a month for Small 4, filed with support
once billing passes the $2,000 tier; the free tier (1 RPS, 500 k TPM, 1 B tokens a month) is 10 % of the
average TPM.

The `mistralai` SDK (2.10.1) has no retries by default and a 300 s default timeout: retries stay
off so `call_with_retries` is the one place that decides, the timeout is 30 s, a 429 (`errors.SDKError`,
`status_code`) maps to `RateLimited`, and a `Retry-After` is honoured if one appears (none is documented).
Model ids are pinned to dated versions; the 22 May 2026 deprecations of Magistral, Devstral, Pixtral and
Medium 3.1 are the reminder. Zero data retention is available on pay-as-you-go stateless endpoints on request.

**Self-hosted path.** vLLM 0.29.0 (9 September 2026) serves `mistralai/Ministral-3-14B-Instruct-2512` in
FP8 (15.7 GB) on one H100 80 GB per replica with Mistral's documented flags (`--tokenizer_mode mistral
--config_format mistral --load_format mistral --enable-auto-tool-choice --tool-call-parser mistral`); prefix
caching and chunked prefill are on by default, so the shared prefix is prefilled once per replica. The lab's
replica model estimates a 54 GB KV budget at 160 KiB per token, 128 resident sequences, a TPOT of 10 ms at
batch 1 and 20 ms at batch 24, 1,201 tokens per second per replica at batch 24 and a 78 ms TTFT for a call's
2,300 new tokens, all estimates until `vllm bench serve` replaces them.

The alternative primary model is
Small 4 (MoE 119B / 6.5B active, MLA KV of 22.5 KiB per token, 121 GB FP8, `--tensor-parallel-size 2` on
H100 with `--attention-backend FLASH_ATTN_MLA`; Mistral's stated minimum of 4× H100, 2× H200 or 1× B200
needs verifying as GPUs or systems).

Replicas are a Deployment, or a KServe 0.17 `LLMInferenceService` on
llm-d when prefill/decode disaggregation and node-local weight caching are wanted as a package. In front
sits an inference gateway implementing the Gateway API Inference Extension (v1.5.0; Envoy Gateway, kgateway
or GKE Inference Gateway): an `InferencePool` per model, `InferenceObjective` priorities, an endpoint picker
scoring replicas on queue depth, KV utilisation and prefix affinity, and the alpha flow-control layer whose
saturation detector (queue depth 5, KV 0.8) holds low-priority requests and drops them on TTL. NIM is an
option at $1 per GPU-hour ($8 k a month on the peak fleet): `ministral-14b-instruct-2512:1.7.0` exists;
Small 4 NIM profiles cover H200 and Blackwell only (verify H100).

Weights load from a node-local NVMe cache
(or KServe `LocalModelCache`) with pre-pulled images and `--kv-cache-memory-bytes` to skip profiling; the
Run:ai streamer (about 2 GiB/s from object storage) is itself a `--load-format` value and so excludes
`--load_format mistral` (verify the trade-off). Cold start is 3–8 minutes end to end, which is why the fleet
is static for peak.

### 11.2 Capacity and scaling design

| Setting | Value | Derivation |
|---|---|---|
| in-flight cap, hosted / fleet | 175 / 293 | 20 M TPM ÷ 60 ÷ (11,440 tokens per turn ÷ 6 s), 29.1 turns/s sustainable; 11 replicas × batch 24 × (10.0 s turn ÷ 2.2 × 4.1 s call), following the live replica count |
| in-flight cap, hybrid | 293 on the fleet plus 175 on the API for spill-eligible turns only | two caps, because the data policy bounds the spill |
| admission thresholds | soft ratio 0.8; pushback 0.05 (0.15 for level 2); saturation 0.8; queue age 10 s / 30 s; dwell 15 s | the inference gateway's own saturation thresholds; degrade before shedding |
| bucket for `mistral-small-2603` | 20 M TPM, burst 6 s | the negotiated limit; verify in the console |
| `budget.*` | 6 steps, 40 k tokens, $0.25, 45 s | the worst turn costs ~4× the average |
| orchestrator / gateway pods | 1 / 3 / 8 (min 2); 2–3 | in-flight × 1.4 ÷ 80; streams ÷ 250 |
| vLLM replicas | 4 / 11 / 34 needed, never fewer than 2 (HA); 11 static, KEDA to 14 | ×1.3 headroom; 4 / 4 / 3 across three zonal node pools |
| `--max-num-seqs`, `--max-num-queued-reqs` | 48, 12 | TPOT ≈ 30 ms at 48 (estimate); ≈ 2 s of queue wait at batch 24 |
| KEDA on the fleet | `sum(vllm:num_requests_waiting)` > 5 per replica or `avg(vllm:kv_cache_usage_perc)` > 0.8; poll 15 s, cooldown 360 s | headroom only, never the peak |
| rate-limit request | 60 RPS, 19 M TPM, 209 B tokens/month | peak × 1.3 |

The fleet is static for peak because a replica takes 3–8 minutes to appear and a Singapore H100 may not be
available on demand at all; autoscaling adds headroom from warm nodes, and the incident is answered by
degrade levels, spill-over and shedding rather than by 34 GPUs ($171 k a month on AWS on-demand) idling for
an outage that comes twice a year. The 6-second hosted turn would need a TPOT of about 10 ms on the fleet,
so batch about 4 and roughly four times the GPUs, or EAGLE speculative decoding (drafts exist for Small 4,
Medium 3.5 and Large 3, not Ministral), or streaming and accepting 10 s.

| GPU price | $/GPU-h | Peak fleet, $/month | $/conversation | vs the hosted mix ($0.0131) | Floor to amortise two replicas, conversations/day |
|---|---:|---:|---:|---:|---:|
| AWS on-demand | 6.88 | 55.2 k | 0.0182 | 1.39× | ≈ 25 k |
| AWS capacity block (priced in Tokyo, Sydney, Mumbai; verify Singapore) | 4.72 | 37.9 k | 0.0125 | 0.95× | ≈ 17 k |
| GCP 3-year | 4.86 | 39.0 k | 0.0128 | 0.98× | ≈ 18 k |
| Azure 3-year | 5.40 | 43.4 k | 0.0143 | 1.09× | ≈ 20 k |
| neocloud | 3.75 | 30.1 k | 0.0099 | 0.76× | ≈ 14 k |
| GCP on-demand | 11.06 | 88.8 k | 0.0292 | 2.23× | ≈ 41 k |

Both bills grow with volume, the API linearly and the fleet in replica-sized steps, so the break-even is a
GPU price, not a volume: the fleet beats the hosted planning mix whenever H100s cost less than about $4.95
per GPU-hour (0.16 GPU-minutes per conversation at 28 % average utilisation). That is under AWS on-demand,
about level with 3-year commitments and AWS APAC capacity blocks, and clearly won by neoclouds.

Volume matters only at the bottom, where the two-replica HA floor is amortised above about 14 k conversations a day at
$3.75, 17 k at $4.72–4.86 and 25 k at $6.88. Residency, latency control and customisation decide the path;
the arithmetic prices the decision at about $15.5 k a month more on on-demand H100s, about level on committed
ones and about $9.6 k a month less on neocloud GPUs.

The gateway scales on open streams, the orchestrator on
queue depth (bound by the model's token budget, never CPU), the hosted model on tokens per minute, the fleet
on batch per replica and then replica count, the tools on downstream QPS. What breaks first at the incident:
the hosted RPS limit at 2.55×, the hosted TPM limit at 2.38×, the peak-sized fleet at 2.31×; the billing
system reaches 0.43× and the CRM 0.24×. Kubernetes, the orchestrator pods, Redis, Postgres and RabbitMQ are
not on the list.
