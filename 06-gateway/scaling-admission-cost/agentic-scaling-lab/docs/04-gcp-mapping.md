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

## Self-hosting the model: a vLLM fleet on Kubernetes (any cloud)

When the model runs on your own GPUs (the Mistral provider's scenario: `scalelab/mistral.py`, `scalelab/serving.py`,
notebook `05_hosted_or_own_gpus`, and [section 11 of the reference architecture](02-reference-architecture.md#11-variant-the-model-on-your-own-gpus-mistral-models-kubernetes)),
two mechanisms change place. Values are the Mistral capacity plan's ([03-capacity-plan.md](03-capacity-plan.md));
fleet throughput figures are first-principles estimates until `vllm bench serve` replaces them. Facts checked 19
September 2026 (verify).

| Lab module | Mechanism | On Kubernetes (cloud-neutral) | On Mistral's hosted API / marketplaces | The setting that matters |
|---|---|---|---|---|
| `model.py` (`SharedPool`) | the shared pool that answers 429 under contention | none: your replicas never 429; `ServerPool` queues, and `--max-num-queued-reqs` turns the queue into a 503 | one limit per model (RPS, TPM, tokens/month), org- or workspace-scoped (docs disagree; ask); 429 with no documented `Retry-After` | `service_tier="auto"` for Priority Tier queueing ahead of standard traffic, +75 % |
| `serving.py` (`Replica`) | memory → concurrency, bandwidth → TPOT, batch ↔ throughput | one vLLM pod per replica, TP inside the pod; the latency you want is the batch you can afford: batch 24 at 20 ms, 1,201 tok/s per H100 (estimate) | invisible: the provider picks the batch and you get its latency (6 s turn vs 10.0 s on the fleet) | `--max-num-seqs` plus the in-flight cap; measure before anyone buys GPUs |

### The vLLM and Kubernetes settings that matter

| Flag / setting | Ministral 3 14B on 1× H100 | Small 4 on 2× H100 | Why |
|---|---|---|---|
| weights | `mistralai/Ministral-3-14B-Instruct-2512`, FP8, 15.7 GB | `mistralai/Mistral-Small-4-119B-2603`, FP8, 121 GB | pin the dated repo; Small 4's NVFP4 build (70.8 GB) leaves no KV room on one H100 |
| format and tools | `--tokenizer_mode mistral --config_format mistral --load_format mistral --enable-auto-tool-choice --tool-call-parser mistral` | HF format; `--enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral` | Mistral's documented commands |
| `--tensor-parallel-size` | 1 | 2 (Mistral states a minimum of 4× H100, 2× H200 or 1× B200; verify GPUs vs systems) | 121 GB does not fit 80 GB |
| `--attention-backend` | default | `FLASH_ATTN_MLA` | MLA stores the latent (22.5 KiB/token); confirm "GPU KV cache size" in the startup log |
| `--max-model-len` | 16384 | 16384 | frees KV memory against the 262k default; a turn's context is about 5.2k |
| `--max-num-seqs` | 48 | 128 | TPOT ≈ 30 ms at 48 for the dense model, ≈ 35 ms at 128 for Small 4 (estimates) |
| `--max-num-batched-tokens` | 8192 | 16384 | smaller protects inter-token latency; 2,300 new tokens still prefill in one step; 16384 is the Small 4 card's value |
| `--gpu-memory-utilization` | 0.92 (default): ≈ 54 GB of KV | 0.92: ≈ 18 GB of KV (the card's 0.8 assumes H200 memory; verify) | KV budget = memory × utilisation − weights − activations; `--kv-cache-dtype fp8` doubles the 14B figure (verify quality) |
| `--max-num-queued-reqs` | 12 | 12 | ≈ 2 s of queue wait at batch 24; beyond that a 503 the gateway treats like a 429 |
| `--speculative-config` | none (no EAGLE draft published; n-gram possible, verify) | EAGLE draft `-eagle`, 3 tokens | the route back to a 6 s turn |
| pod resources | `nvidia.com/gpu: 1` | `nvidia.com/gpu: 2`, `/dev/shm` emptyDir in memory (16 Gi) | tensor parallelism needs shared memory for NCCL |
| start, drain, weights | startup probe up to 10 min; readiness `/health` plus a warm-up request; grace 120 s; node-local NVMe weight cache, pre-pull DaemonSet, `--kv-cache-memory-bytes` from the startup log | same | 3–8 min cold start, image pull the largest item; a call takes 4.1 s to drain |
| placement | `topologySpreadConstraints` by zone; PDB `minAvailable` 9 of 11 | same, 7 of 9 | a zonal loss removes at most 4 replicas |
| KEDA ScaledObject | `sum(vllm:num_requests_waiting)` at 5 per replica, `avg(vllm:kv_cache_usage_perc)` at 0.8; min 11, max 14; poll 15 s, cooldown 360 s | min 9, max 12 | headroom only; the peak is static |

### Per-hyperscaler notes for a Singapore-resident fleet

| Platform | GPUs in or near Singapore (19 Sep 2026) | Queue, Postgres, Redis | Hosted path for Mistral models | Notes |
|---|---|---|---|---|
| Azure (AKS) | ND H200 v5 in Southeast Asia at $13.78 per GPU-hour on-demand ($110.24 per 8-GPU node), far above the $4.95 break-even; ND H100 v5 priced in Japan East ($17.82), Australia East and Korea Central but not Southeast Asia in the Retail Prices API (verify); 3-year H100 reservation ≈ $5.40 in East US (1.09× the hosted mix) | Service Bus (sessions for per-session ordering, dead-letter, KEDA scaler); Azure Database for PostgreSQL; Azure Cache for Redis | Foundry sells Large 3 and Medium 3.5 directly, with Global Standard, Data Zone Standard and Provisioned Managed deployment types, but the region rows fetched were Americas only (verify); Small 4 and Ministral 3 not found | Application Gateway WAF or Front Door; Azure Local and Foundry Local for disconnected sites |
| AWS (EKS) | p5 (H100) on-demand listed in ap-southeast-1 at $6.88 per GPU-hour (1.39× the hosted mix); Capacity Blocks at $4.72 (0.95×) in Tokyo, Sydney and Mumbai, not Singapore on the pricing page (verify); p5e (H200) Capacity Blocks in Tokyo, Mumbai and Sydney, p5en in Tokyo and Mumbai; EKS guidance is static NodePool replicas for inference | Amazon MQ for RabbitMQ (the same contract) or SQS FIFO (message group = session, DLQ, KEDA scaler); RDS or Aurora PostgreSQL; ElastiCache | Bedrock: Large 3 (in-region only, 32k max output, implicit caching) and Ministral 3 in Tokyo, Mumbai and Sydney, not Singapore as of the fetched tables; no Small 4 or Medium 3.5; regional prices 15–50 % above US (verify) | AWS WAF on the ALB; Karpenter with `consolidationPolicy: WhenEmpty` on reserved GPU nodes |
| Google Cloud (GKE) | a3-highgpu-8g (H100) in asia-southeast1 at $11.06 per GPU-hour on-demand (2.23× the hosted mix), $4.86 on a 3-year commitment (0.98×); a3-ultragpu-8g (H200) in Singapore at $10.60 | Pub/Sub (ordering keys, dead-letter topic) or in-cluster RabbitMQ; Cloud SQL or AlloyDB; Memorystore | Vertex partner models list only Medium 3, Small 3.1, OCR and Codestral on 19 Sep 2026, no Large 3, Medium 3.5 or Small 4; regions unverified (verify) | GKE Inference Gateway implements the Gateway API Inference Extension natively; Cloud Armor |
| In-country sovereign GPU cloud | GPU-as-a-service from an in-country provider, whose sites can sit across the border (only those in Singapore meet a Singapore-only residency rule); GPU counts and prices not public (verify, 19 Sep 2026). It beats the hosted mix only below the $4.95 per H100-hour break-even, where neocloud rates ($3.75, 0.76× the hosted mix) already sit | your own RabbitMQ, CloudNativePG and Redis Sentinel on the provider's Kubernetes | Mistral AI Studio self-hosted or dedicated (Agent Runtime on Temporal) by contract with Mistral, no public pricing or hardware requirements; IBM watsonx Sydney for Large 3 and Ministral 3 | the only announced APAC-hosted sovereign option (verify) |

The Singapore-resident paths are therefore self-hosting on a hyperscaler's Singapore GPUs, on an in-country
GPU cloud, or on Mistral's own platform under a direct contract; every marketplace route fetched is APAC-resident at best.
