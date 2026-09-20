# From the core concepts to Google Cloud

Each module in `scalelab/` is one mechanism. This is where that mechanism lives in production, and the
one setting that matters most. Facts checked 5 September 2026.

| Lab module | Mechanism | On Google Cloud | The setting that matters |
|---|---|---|---|
| `capacity.py` | tokens-per-minute demand, Little's law, PT sizing | Gemini on the Gemini Enterprise Agent Platform: Standard PayGo tiers (Flash 2 / 4 / 10 M TPM by org spend), Provisioned Throughput in GSUs by term, global vs regional endpoints (+10 %) | the org's tier baseline, and the PT term (break-even 75 % utilisation on 1-year) |
| `model.py` (`SharedPool`) | the shared pool that answers 429 under contention | the model family's pay-as-you-go pool; `usage_metadata.traffic_type` says which pool served a call | request headers `X-Vertex-AI-LLM-Request-Type: dedicated \| shared` and `X-Vertex-AI-LLM-Shared-Request-Type: priority \| flex` |
| `model.py` (prefix cache) | cached input at 10 % | implicit prefix caching (≥ 4,096 tokens on Gemini 3.x; 6,144 on 3.7/3.8 Flash and 3.1 Pro); explicit caches with a TTL (`client.caches.create`) | put the stable prefix first; version the prompt in the cache key |
| `resilience.py` (`TokenBucket`) | client-side smoothing | a bucket per model in Memorystore (Lua script) shared by all orchestrator instances | rate = your share of the tier baseline; burst ≈ 6 s of refill |
| `resilience.py` (`backoff`, `call_with_retries`) | jittered retries within the deadline | in your model gateway; the `google-genai` SDK retries are off by default — keep them off so one place decides | never past the turn deadline; honour and jitter `Retry-After` |
| `resilience.py` (`CircuitBreaker`) | fail fast, probe, fall back | one breaker per model; fallback to a sibling Flash model with its own pool; model ids in configuration (3.6/3.7/3.8 Flash retire 45 days after a successor) | cooldown 15 s; sibling order |
| `admission.py` | degrade levels, in-flight cap, shedding | the gateway Cloud Run service; the level and the in-flight gauge in Memorystore so every instance agrees; Cloud Armor rate limits per caller in front | `max_inflight` from the token budget, tuned from load tests; hysteresis 15 s |
| `loop.py` (budget) | steps / tokens / cost / deadline | the orchestrator Cloud Run service (instance-based billing so checkpoints and compaction finish after the response) | deadline 45 s, well under the Pub/Sub ack deadline (600 s) |
| `loop.py` (`Store`, checkpoints) | replayable turn | Firestore: `sessions/{id}/turns/{id}` with steps appended atomically (`ArrayUnion`), TTL policy on `expires_at` | 1 MiB per document; ~1 sustained write/s per document; truncate tool results in the checkpoint |
| `loop.py` (idempotency keys) | writes happen once | Redis `SET NX` markers with a 24 h TTL keyed `{turn}:{step}:{i}:{tool}:{hash(args)}` | key on arguments, not on time |
| the queue (`sim.py` runs turns directly) | at-least-once delivery, backlog as one number | Pub/Sub push subscription with OIDC to the orchestrator; ordering key = session id; retry policy 10–600 s; dead-letter after 5 attempts | alert on `subscription/oldest_unacked_message_age` > 30 s |
| `tools.py` | downstream QPS ceilings, structured errors | tool services on Cloud Run (own service account, `run.invoker` for the orchestrator only), or MCP servers; caches for reads; bulkhead per tool | the slowest system's QPS (billing 40) decides degrade level 2 |
| `sim.py` | load test → settings | a load test against a staging environment with the incident intent mix; Cloud Run settings below | concurrency 80 (orchestrator) / 250 (gateway); min instances 2 |

## Cloud Run settings that matter

| Setting | Gateway | Orchestrator | Why |
|---|---|---|---|
| Billing mode | request-based | instance-based (CPU always on) | background work after the response |
| Concurrency | 250 | 80 | I/O-bound coroutines; memory per in-flight turn |
| Min / max instances | 2 / 100 | 2 / 50 | warm capacity; bounded by regional CPU quota and Direct VPC egress (~100–200 instances/revision) |
| Timeout | 600 s | 600 s | SSE streams are requests (60-minute ceiling); equals the Pub/Sub ack deadline |
| Ingress | internal + load balancer | internal only | Cloud Armor cannot be bypassed; Pub/Sub push counts as internal |
| Autoscaling | 60 % CPU over 1 min or 60 % of max concurrency | same | scale-out waits max(10 s, 3.5× predicted cold start) |
| Shutdown | SIGTERM + 10 s | same | checkpoint early; the queue redelivers |

## The estate, in one line

Global external ALB (Cloud Armor, IAP) → gateway (Cloud Run) → Pub/Sub → orchestrator (Cloud Run) → Gemini (global
endpoint, PT + spill-over) and tool services (Cloud Run / MCP); Firestore for sessions, turns and checkpoints;
Memorystore for streams, locks, buckets and the degrade level; Cloud Trace / Monitoring with `gen_ai.*` attributes.
The full design is in `02-reference-architecture.md`.
