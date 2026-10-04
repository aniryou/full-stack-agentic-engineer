# From the core concepts to Google Cloud

Each module in `scalelab/` is one mechanism. This document shows where that mechanism lives in production. It also
gives the one setting that matters most for each mechanism. Date of the fact check: 5 September 2026.

| Lab module | Mechanism | On Google Cloud | The setting that matters |
|---|---|---|---|
| `capacity.py` | tokens-per-minute demand, Little's law, the calculation of the PT size | Gemini on the Gemini Enterprise Agent Platform. Standard PayGo tiers (Flash 2 / 4 / 10 M TPM by org spend). Provisioned Throughput in GSUs by term. Global endpoints against regional endpoints (+10 %). | the tier baseline of the org, and the PT term (break-even at 75 % utilisation on a 1-year term) |
| `model.py` (`SharedPool`) | the shared pool that answers 429 under contention | The pay-as-you-go pool of the model family. `usage_metadata.traffic_type` tells which pool served a call. | request headers `X-Vertex-AI-LLM-Request-Type: dedicated \| shared` and `X-Vertex-AI-LLM-Shared-Request-Type: priority \| flex` |
| `model.py` (prefix cache) | cached input at 10 % | Implicit prefix caching (≥ 4,096 tokens on Gemini 3.x, 6,144 on 3.7/3.8 Flash and 3.1 Pro). Explicit caches with a TTL (`client.caches.create`). | Put the stable prefix first. Put the prompt version in the cache key. |
| `resilience.py` (`TokenBucket`) | a smooth request rate on the client side | A bucket per model in Memorystore (Lua script). All orchestrator instances share it. | Rate = your share of the tier baseline. Burst ≈ 6 s of refill. |
| `resilience.py` (`backoff`, `call_with_retries`) | retries with jitter within the deadline | In your model gateway. The `google-genai` SDK retries are off by default. Keep them off, so that one place decides. | Never retry after the turn deadline. Obey `Retry-After` and add jitter to it. |
| `resilience.py` (`CircuitBreaker`) | fail fast, send a probe, use the fallback | One breaker per model. The fallback is a different Flash model of the same family, with its own pool. Keep the model ids in configuration (3.6/3.7/3.8 Flash retire 45 days after a successor). | Cooldown 15 s. The order of the fallback models. |
| `admission.py` | degrade levels, an in-flight cap, the rejection of excess requests | The gateway Cloud Run service. The level and the in-flight gauge are in Memorystore, so that every instance agrees. Cloud Armor rate limits per caller are in front. | `max_inflight` comes from the token budget, adjusted from load tests. Hysteresis 15 s. |
| `loop.py` (budget) | steps / tokens / cost / deadline | The orchestrator Cloud Run service. It uses instance-based billing, so that checkpoints and compaction finish after the response. | Deadline 45 s, much less than the Pub/Sub ack deadline (600 s). |
| `loop.py` (`Store`, checkpoints) | a turn that you can replay | Firestore: `sessions/{id}/turns/{id}`, with atomic appends of steps (`ArrayUnion`), and a TTL policy on `expires_at`. | 1 MiB per document. ~1 sustained write/s per document. Truncate tool results in the checkpoint. |
| `loop.py` (idempotency keys) | each write occurs one time | Redis `SET NX` markers with a 24 h TTL and the key `{turn}:{step}:{i}:{tool}:{hash(args)}` | Make the key from the arguments, not from the time. |
| the queue (`sim.py` runs turns directly) | at-least-once delivery, the backlog as one number | A Pub/Sub push subscription with OIDC to the orchestrator. Ordering key = session id. Retry policy 10–600 s. Dead-letter after 5 attempts. | Alert on `subscription/oldest_unacked_message_age` > 30 s. |
| `tools.py` | downstream QPS ceilings, structured errors | Tool services on Cloud Run (each with its own service account, `run.invoker` for the orchestrator only), or MCP servers. Caches for reads. A bulkhead per tool. | The QPS of the slowest system (billing 40) decides degrade level 2. |
| `sim.py` | from a load test to the settings | A load test against a staging environment with the incident intent mix. The Cloud Run settings are in the next section. | Concurrency 80 (orchestrator) / 250 (gateway). Min instances 2. |

## Cloud Run settings that matter

| Setting | Gateway | Orchestrator | Why |
|---|---|---|---|
| Billing mode | request-based | instance-based (CPU always on) | background work after the response |
| Concurrency | 250 | 80 | I/O-bound coroutines, memory per in-flight turn |
| Min / max instances | 2 / 100 | 2 / 50 | Warm capacity. The regional CPU quota and Direct VPC egress (~100–200 instances/revision) set the upper limit. |
| Timeout | 600 s | 600 s | SSE streams are requests (60-minute ceiling). The value is equal to the Pub/Sub ack deadline. |
| Ingress | internal + load balancer | internal only | No request can go around Cloud Armor. Pub/Sub push counts as internal. |
| Autoscaling | 60 % CPU over 1 min or 60 % of max concurrency | same | scale-out waits max(10 s, 3.5× predicted cold start) |
| Shutdown | SIGTERM + 10 s | same | Write the checkpoint early. The queue delivers the message again. |

## The estate, in one line

Traffic goes from a global external ALB (Cloud Armor, IAP) to the gateway (Cloud Run), and then to Pub/Sub. From
Pub/Sub, it goes to the orchestrator (Cloud Run). Then it goes to Gemini (global endpoint, PT + spill-over) and to
the tool services (Cloud Run / MCP).

Firestore holds sessions, turns and checkpoints. Memorystore holds streams, locks, buckets and the degrade level.
Cloud Trace / Monitoring use `gen_ai.*` attributes. The full design is in `02-reference-architecture.md`.

## Self-hosting the model: a vLLM fleet on Kubernetes (any cloud)

When the model runs on your own GPUs, two mechanisms change place. A model on your own GPUs is the scenario of the Mistral provider. You find this scenario in
`scalelab/mistral.py`, `scalelab/serving.py`, notebook `05_hosted_or_own_gpus`, and [section 11 of the reference architecture](02-reference-architecture.md#11-variant-the-model-on-your-own-gpus-mistral-models-kubernetes).
The values come from the Mistral capacity plan ([03-capacity-plan.md](03-capacity-plan.md)). The fleet throughput
figures are estimates from first principles, until `vllm bench serve` replaces them. Date of the fact check: 19
September 2026 (verify).

| Lab module | Mechanism | On Kubernetes (cloud-neutral) | On Mistral's hosted API / marketplaces | The setting that matters |
|---|---|---|---|---|
| `model.py` (`SharedPool`) | the shared pool that answers 429 under contention | None. Your replicas never return 429. `ServerPool` puts requests in a queue, and `--max-num-queued-reqs` turns the queue into a 503. | One limit per model (RPS, TPM, tokens/month), with an org or workspace scope. The docs disagree, so ask. A 429 with no documented `Retry-After`. | Set `service_tier="auto"` for the Priority Tier queue, ahead of standard traffic, +75 %. |
| `serving.py` (`Replica`) | Memory sets concurrency. Bandwidth sets TPOT. Batch size and throughput change together. | One vLLM pod per replica, with TP inside the pod. The latency that you want is the batch that you can afford: batch 24 at 20 ms, 1,201 tok/s per H100 (estimate). | Invisible. The provider selects the batch, and you get its latency (6 s turn against 10.0 s on the fleet). | `--max-num-seqs` plus the in-flight cap. Measure before anyone buys GPUs. |

### The vLLM and Kubernetes settings that matter

| Flag / setting | Ministral 3 14B on 1× H100 | Small 4 on 2× H100 | Why |
|---|---|---|---|
| weights | `mistralai/Ministral-3-14B-Instruct-2512`, FP8, 15.7 GB | `mistralai/Mistral-Small-4-119B-2603`, FP8, 121 GB | Pin the dated repo. The NVFP4 build of Small 4 (70.8 GB) leaves no KV room on one H100. |
| format and tools | `--tokenizer_mode mistral --config_format mistral --load_format mistral --enable-auto-tool-choice --tool-call-parser mistral` | HF format, with `--enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral` | the commands that Mistral documents |
| `--tensor-parallel-size` | 1 | 2 (Mistral states a minimum of 4× H100, 2× H200 or 1× B200. Verify: are these GPUs or systems?) | 121 GB does not fit in 80 GB. |
| `--attention-backend` | default | `FLASH_ATTN_MLA` | MLA stores the latent (22.5 KiB/token). Make sure of the "GPU KV cache size" line in the startup log. |
| `--max-model-len` | 16384 | 16384 | The value 16384 frees KV memory, compared with the 262k default. The context of a turn is about 5.2k. |
| `--max-num-seqs` | 48 | 128 | TPOT ≈ 30 ms at 48 for the dense model, ≈ 35 ms at 128 for Small 4 (estimates) |
| `--max-num-batched-tokens` | 8192 | 16384 | A smaller value protects inter-token latency. 2,300 new tokens still prefill in one step. 16384 is the value on the Small 4 card. |
| `--gpu-memory-utilization` | 0.92 (default): ≈ 54 GB of KV | 0.92: ≈ 18 GB of KV. The value 0.8 on the card assumes H200 memory (verify). | KV budget = memory × utilisation − weights − activations. `--kv-cache-dtype fp8` doubles the 14B figure (verify quality). |
| `--max-num-queued-reqs` | 12 | 12 | ≈ 2 s of queue wait at batch 24. When the queue is full, the server returns a 503. The gateway treats the 503 like a 429. |
| `--speculative-config` | none (no published EAGLE draft, n-gram possible, verify) | EAGLE draft `-eagle`, 3 tokens | the route back to a 6 s turn |
| pod resources | `nvidia.com/gpu: 1` | `nvidia.com/gpu: 2`, `/dev/shm` emptyDir in memory (16 Gi) | Tensor parallelism needs shared memory for NCCL. |
| start, drain, weights | Startup probe up to 10 min. Readiness `/health` plus a warm-up request. Grace 120 s. Node-local NVMe weight cache, pre-pull DaemonSet, `--kv-cache-memory-bytes` from the startup log. | same | 3–8 min cold start. The image pull is the largest item. A call takes 4.1 s to drain. |
| placement | `topologySpreadConstraints` by zone, PDB `minAvailable` 9 of 11 | same, 7 of 9 | The loss of a zone removes at most 4 replicas. |
| KEDA ScaledObject | `sum(vllm:num_requests_waiting)` at 5 per replica, `avg(vllm:kv_cache_usage_perc)` at 0.8. Min 11, max 14. Poll 15 s, cooldown 360 s. | min 9, max 12 | Headroom only. The peak is static. |

### Per-hyperscaler notes for a Singapore-resident fleet

| Platform | GPUs in or near Singapore (19 Sep 2026) | Queue, Postgres, Redis | Hosted path for Mistral models | Notes |
|---|---|---|---|---|
| Azure (AKS) | ND H200 v5 in Southeast Asia at $13.78 per GPU-hour on-demand ($110.24 per 8-GPU node). The price per GPU-hour is far above the $4.95 break-even. The Retail Prices API gives ND H100 v5 prices in Japan East ($17.82), Australia East and Korea Central, but not in Southeast Asia (verify). A 3-year H100 reservation ≈ $5.40 in East US (1.09× the hosted mix). | Service Bus (sessions for the message order in each session, dead-letter, KEDA scaler), Azure Database for PostgreSQL, Azure Cache for Redis | Foundry sells Large 3 and Medium 3.5 directly. Its deployment types are Global Standard, Data Zone Standard and Provisioned Managed. But the fetched region rows showed the Americas only (verify). Small 4 and Ministral 3: not found. | Application Gateway WAF or Front Door. Azure Local and Foundry Local for disconnected sites. |
| AWS (EKS) | The listed p5 (H100) on-demand price in ap-southeast-1 is $6.88 per GPU-hour (1.39× the hosted mix). Capacity Blocks at $4.72 (0.95×) in Tokyo, Sydney and Mumbai. The pricing page does not show them in Singapore (verify). p5e (H200) Capacity Blocks in Tokyo, Mumbai and Sydney, p5en in Tokyo and Mumbai. The EKS guidance for inference is static NodePool replicas. | Amazon MQ for RabbitMQ (the same contract) or SQS FIFO (message group = session, DLQ, KEDA scaler), RDS or Aurora PostgreSQL, ElastiCache | Bedrock: Large 3 (in-region only, 32k max output, implicit caching) and Ministral 3 in Tokyo, Mumbai and Sydney. The fetched tables do not show them in Singapore. No Small 4 or Medium 3.5. Regional prices 15–50 % above US (verify). | AWS WAF on the ALB. Karpenter with `consolidationPolicy: WhenEmpty` on reserved GPU nodes. |
| Google Cloud (GKE) | a3-highgpu-8g (H100) in asia-southeast1 at $11.06 per GPU-hour on-demand (2.23× the hosted mix), $4.86 on a 3-year commitment (0.98×). a3-ultragpu-8g (H200) in Singapore at $10.60. | Pub/Sub (ordering keys, dead-letter topic) or in-cluster RabbitMQ, Cloud SQL or AlloyDB, Memorystore | On 19 Sep 2026, Vertex partner models list only Medium 3, Small 3.1, OCR and Codestral. No Large 3, Medium 3.5 or Small 4. The regions are not confirmed (verify). | GKE Inference Gateway implements the Gateway API Inference Extension natively. Cloud Armor. |
| In-country sovereign GPU cloud | GPU-as-a-service from an in-country provider. Its sites can be across the border. Only the sites in Singapore meet a Singapore-only residency rule. GPU counts and prices are not public (verify, 19 Sep 2026). The in-country GPU cloud beats the hosted mix only below the $4.95 per H100-hour break-even. Neocloud rates ($3.75, 0.76× the hosted mix) are already below that level. | your own RabbitMQ, CloudNativePG and Redis Sentinel on the provider's Kubernetes | Mistral AI Studio, self-hosted or dedicated (Agent Runtime on Temporal), by contract with Mistral. No public pricing or hardware requirements. IBM watsonx Sydney for Large 3 and Ministral 3. | the only announced APAC-hosted sovereign option (verify) |

Thus, on each Singapore-resident path, you host the model yourself, in one of these places:

- the Singapore GPUs of a hyperscaler,
- an in-country GPU cloud,
- Mistral's own platform, under a direct contract.

Every fetched marketplace route is APAC-resident at best.
