# From the core concepts to the platform

Each module in `scalelab/` is one mechanism. This is where that mechanism lives on Kubernetes (any cloud or
on-prem), what it becomes on Mistral's hosted API or a cloud marketplace, and the one setting that matters
most. Values are the capacity plan's (`03-capacity-plan.md`); fleet throughput figures are the lab's
first-principles estimates until `vllm bench serve` replaces them. Facts checked 19 September 2026.

## 1. From each mechanism to the platform

| Lab module | Mechanism | On Kubernetes (cloud-neutral) | On Mistral's hosted API / marketplaces | The setting that matters |
|---|---|---|---|---|
| `capacity.py` | tokens-per-minute demand, Little's law, the fleet, the GPU-price break-even | `plan()` sets the static numbers: orchestrator pods 1 / 3 / 8, vLLM replicas 4 / 11 / 34 (11 static, never fewer than 2), caps 175 / 293; both bills grow with volume, so the fleet beats the hosted mix below $4.95 per GPU-hour (0.16 GPU-minutes per conversation), with a two-replica floor amortised above ≈ 14–25 k conversations/day | the rate-limit request to support: 60 RPS, 19 M TPM, 209 B tokens/month; paid tiers are console-only, unlocked at $20 / $100 / $500 / $2,000 billed | the TPM you are granted, or the GPU-hour price you can get in Singapore |
| `model.py` (`SharedPool`) | the shared pool that answers 429 under contention | none: your replicas never 429; `ServerPool` queues, and `--max-num-queued-reqs` turns the queue into a 503 | one limit per model (RPS, TPM, tokens/month), org- or workspace-scoped (docs disagree; ask); 429 with no documented `Retry-After` | `service_tier="auto"` for Priority Tier queueing ahead of standard traffic, +75 % |
| `serving.py` (`Replica`) | memory → concurrency, bandwidth → TPOT, batch ↔ throughput | one vLLM pod per replica, TP inside the pod; the latency you want is the batch you can afford: batch 24 at 20 ms, 1,201 tok/s per H100 (estimate) | invisible: the provider picks the batch and you get its latency (6 s turn vs 10.0 s on the fleet) | `--max-num-seqs` plus the in-flight cap; measure before anyone buys GPUs |
| `model.py` (prefix cache) | cached input at 10 % | prefix caching on by default (block size 16); the 2,700-token prefix is prefilled once per replica; KV-aware routing keeps a session on the replica that holds it; watch `vllm:prefix_cache_hits_total` | `prompt_cache_key`, 64-token blocks, `usage.prompt_tokens_details.cached_tokens`; TTL and cache-write price unpublished; Bedrock Large 3 has implicit caching, Bedrock Ministral 8B none | stable prefix first; version the prompt in the key |
| `resilience.py` (`TokenBucket`) | client-side smoothing | a Lua bucket per model in Redis shared by every orchestrator pod; one per backing system too (billing 40 QPS) | rate = your share of the granted TPM (20 M → 333 k tokens/s), burst ≈ 6 s | never above the limit you were granted |
| `resilience.py` (`backoff`, `call_with_retries`) | jittered retries within the deadline | in the model gateway: 429, 503 and timeout, never past the 45 s deadline | `mistralai` 2.10.1 retries are off by default; keep them off (a `RetryConfig` would retry 429/500/502/503/504); default timeout 300 s, set 30 s | one place decides; honour and jitter `Retry-After` if it ever appears |
| `resilience.py` (`CircuitBreaker`) | fail fast, probe, fall back, spill | one breaker per model; on the fleet the fallback is the same model with shorter answers; `HybridBackend` spills to the API bucket at saturation 0.9 | sibling = Ministral 3 8B with its own pool and breaker; dated ids (the 22 May 2026 deprecations) | cooldown 15 s; sibling order; `spill_allowed` per turn |
| `admission.py` | degrade levels, in-flight cap, shedding | the gateway Deployment; level and gauge in Redis; saturation from `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc`; the Inference Extension's flow control (priority, TTL) as the second line | the pushback ratio comes from 429s; nothing on the API side sheds for you | `max_inflight` 175 hosted / 293 fleet; dwell 15 s |
| `loop.py` (`Budget`) | steps / tokens / cost / deadline | the orchestrator Deployment; deadline 45 s, far under RabbitMQ's consumer timeout (30 min default) | `max_tokens` 700, `reasoning_effort="none"`, `max_cost_usd` 0.25 at list prices | the deadline, which bounds every retry |
| `loop.py` (`Store`, checkpoints) | replayable turn | Postgres `steps(turn_id, idx, kind, payload)` inserted before the next action; 7-day TTL job | the Agents & Conversations API keeps state server-side but is beta-labelled, absent from regional endpoints and from ZDR: not used | truncate tool results in the checkpoint |
| `loop.py` (idempotency keys) | writes happen once | Redis `SET NX` markers, 24 h, keyed `{turn}:{step}:{i}:{tool}:{hash(args)}` | unchanged | key on arguments, not on time |
| the queue (`sim.py` runs turns directly) | at-least-once delivery, backlog as one number | RabbitMQ quorum queues behind a consistent-hash exchange keyed on the session; `x-delivery-limit` 5; dead-letter exchange; KEDA RabbitMQ scaler at 20 ready messages per pod | unchanged (or the cloud's queue, table 3) | alert when the oldest queued turn passes 30 s (the Redis `queued` set) |
| `tools.py` | downstream QPS ceilings, structured errors | tool Deployments or MCP servers reachable only from the orchestrator over mTLS; per-system bucket in Redis; bulkhead per pod | Mistral Connectors (MCP) run hosted-side, are public preview and not regional: tools stay in the cluster | billing at 40 QPS decides level 2 |
| `sim.py` | load test → settings | a load test against staging with the incident intent mix; the lab's regimes are naive p95 39 s hosted and 15.3 s fleet, degrade plus cap 3.3 s and 8.8 s, hybrid 6.5 s | the same harness against the real API at a small granted TPM, to see the 429 and its headers | the cap comes from measurements, not from the plan |

## 2. The vLLM and Kubernetes settings that matter

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

## 3. Per-hyperscaler notes

| Platform | GPUs in or near Singapore (19 Sep 2026) | Queue, Postgres, Redis | Hosted path for Mistral models | Notes |
|---|---|---|---|---|
| Azure (AKS) | ND H200 v5 in Southeast Asia at $13.78 per GPU-hour on-demand ($110.24 per 8-GPU node), far above the $4.95 break-even; ND H100 v5 priced in Japan East ($17.82), Australia East and Korea Central but not Southeast Asia in the Retail Prices API (verify); 3-year H100 reservation ≈ $5.40 in East US (1.09× the hosted mix) | Service Bus (sessions for per-session ordering, dead-letter, KEDA scaler); Azure Database for PostgreSQL; Azure Cache for Redis | Foundry sells Large 3 and Medium 3.5 directly, with Global Standard, Data Zone Standard and Provisioned Managed deployment types, but the region rows fetched were Americas only (verify); Small 4 and Ministral 3 not found | Application Gateway WAF or Front Door; Azure Local and Foundry Local for disconnected sites |
| AWS (EKS) | p5 (H100) on-demand listed in ap-southeast-1 at $6.88 per GPU-hour (1.39× the hosted mix); Capacity Blocks at $4.72 (0.95×) in Tokyo, Sydney and Mumbai, not Singapore on the pricing page (verify); p5e (H200) Capacity Blocks in Tokyo, Mumbai and Sydney, p5en in Tokyo and Mumbai; EKS guidance is static NodePool replicas for inference | Amazon MQ for RabbitMQ (the same contract) or SQS FIFO (message group = session, DLQ, KEDA scaler); RDS or Aurora PostgreSQL; ElastiCache | Bedrock: Large 3 (in-region only, 32k max output, implicit caching) and Ministral 3 in Tokyo, Mumbai and Sydney, not Singapore as of the fetched tables; no Small 4 or Medium 3.5; regional prices 15–50 % above US (verify) | AWS WAF on the ALB; Karpenter with `consolidationPolicy: WhenEmpty` on reserved GPU nodes |
| Google Cloud (GKE) | a3-highgpu-8g (H100) in asia-southeast1 at $11.06 per GPU-hour on-demand (2.23× the hosted mix), $4.86 on a 3-year commitment (0.98×); a3-ultragpu-8g (H200) in Singapore at $10.60 | Pub/Sub (ordering keys, dead-letter topic) or in-cluster RabbitMQ; Cloud SQL or AlloyDB; Memorystore | Vertex partner models list only Medium 3, Small 3.1, OCR and Codestral on 19 Sep 2026, no Large 3, Medium 3.5 or Small 4; regions unverified (verify) | GKE Inference Gateway implements the Gateway API Inference Extension natively; Cloud Armor |
| Partner sovereign | Singtel RE:AI: GPU-as-a-service in Singapore, Johor and Batam, MOU with Mistral on 27 April 2026, an Applied AI Centre of Excellence in Singapore, customer-care chatbots a named trial use case; GPU counts and prices not public (verify), and the price to ask for is under $4.95 per H100-hour, where neocloud rates ($3.75, 0.76× the hosted mix) already sit | the customer's own RabbitMQ, CloudNativePG and Redis Sentinel on the partner's Kubernetes | Mistral AI Studio self-hosted or dedicated (Agent Runtime on Temporal) via sales, no public pricing or hardware requirements; IBM watsonx Sydney for Large 3 and Ministral 3 | the only announced APAC-hosted sovereign option; NCS (MOU 12 May 2026) and NTT DATA (Singapore and Australia) as delivery partners |

The Singapore-resident paths are therefore self-hosting on a hyperscaler's Singapore GPUs, Singtel RE:AI,
or Mistral's own platform through sales; every marketplace route fetched is APAC-resident at best.

## 4. The estate, in one line

Ingress with WAF and OIDC → gateway (admission, SSE) → RabbitMQ quorum queues → orchestrator (the durable loop,
KEDA on queue depth) → model gateway (buckets, retries, breaker, spill-over) → vLLM replicas of Ministral 3 14B
on Singapore GPU node pools behind the Gateway API Inference Extension, with Mistral's API (Small 4, Medium
3.5, Priority Tier) or an in-region marketplace as policy-gated spill-over, and tool services or MCP servers;
Postgres for sessions, turns and checkpoints; Redis for streams, locks, buckets and the degrade level;
OpenTelemetry with `gen_ai.*` attributes and vLLM's `/metrics` into Prometheus, Grafana and Tempo. The full
design is in `02-reference-architecture.md`.
