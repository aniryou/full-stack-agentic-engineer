# Capacity plan — 100,000 conversations/day

6 turns/conversation, 2.2 calls/turn, 5,000 in (2,700 cached) / 200 out tokens per call, peak ×3, incident ×10, hosted turn 6 s, think 60 s. Hosted: mistral-small-2603 with 10% of calls on mistral-medium-3-5, assumed TPM limit 20 M. Self-hosted: Ministral 3 14B on 1 × H100 SXM 80 GB at aws on-demand ($6.88/GPU-h), target TPOT 20 ms.

| Quantity | Average | Peak | Incident |
|---|---:|---:|---:|
| Conversations / s | 1.16 | 3.47 | 11.57 |
| Turns / s | 6.94 | 20.83 | 69.44 |
| Model calls / s | 15.28 | 45.83 | 153 |
| Input tokens / min | 4.58 M | 13.75 M | 45.83 M |
| … of which uncached | 2.11 M | 6.33 M | 21.08 M |
| Output tokens / min | 183.3 k | 550.0 k | 1.83 M |
| Total TPM vs the hosted limit (×) | 0.24 | 0.72 | 2.38 |
| In-flight turns, hosted (Little's law) | 41.67 | 125 | 417 |
| Concurrent sessions | 458 | 1.4 k | 4.6 k |
| Orchestrator pods | 1.00 | 3.00 | 8.00 |

## Hosted (Mistral API)

- rate-limit request for peak with ×1.3 headroom: **60 RPS, 19 M TPM, 209 B tokens/month** (the free tier covers 10% of the average TPM)
- the assumed 20 M TPM limit sustains 29.1 turns/s, i.e. 175 turns in flight — the starting value for the in-flight cap

| Configuration | $ / conversation | $ / month |
|---|---:|---:|
| mistral-small-2603 no cache | 0.0115 | 34,911 |
| mistral-small-2603 cached | 0.0067 | 20,285 |
| planning mix (90% mistral-small-2603 + 10% mistral-medium-3-5) cached | 0.0131 | 39,745 |
| planning mix, priority tier | 0.0229 | 69,553 |
| planning mix, regional endpoint | 0.0144 | 43,719 |
| mistral-large-2512 cached | 0.0209 | 63,603 |
| mistral-medium-3-5 cached | 0.0707 | 214,885 |
| ministral-14b-2512 cached | 0.0073 | 22,231 |

## Self-hosted (Ministral 3 14B on 1 × H100 SXM 80 GB, vLLM)

- batch 24 per replica keeps TPOT at 20 ms → 1,201 tokens/s per replica; TTFT 78 ms; a call takes 4.1 s and a turn 10.0 s (vs 6 s hosted)
- replicas (×1.3 headroom): average 4, peak 11, incident 34 → GPUs 4 / 11 / 34
- the peak-sized fleet costs $55,246/month at aws on-demand ($0.0182/conversation; average utilisation 28%); it can hold 293 turns in flight — the in-flight cap in self-hosted mode
- versus the hosted planning mix ($39,745/month): 1.39× — both bills grow with volume, so the break-even is a GPU price: **$4.95 per GPU-hour** (0.16 GPU-minutes per conversation at 28% average utilisation)

| GPU price | $ / GPU-h | Peak fleet $ / month | $ / conversation | vs hosted | Floor: conversations / day to amortise 2 replicas |
|---|---:|---:|---:|---:|---:|
| aws on-demand | 6.88 | 55,246 | 0.0182 | 1.39× | 25,273 |
| aws capacity block (Tokyo/Sydney/Mumbai) | 4.72 | 37,902 | 0.0125 | 0.95× | 17,339 |
| gcp on-demand | 11.06 | 88,812 | 0.0292 | 2.23× | 40,628 |
| gcp 3-year | 4.86 | 39,026 | 0.0128 | 0.98× | 17,853 |
| azure 3-year | 5.40 | 43,362 | 0.0143 | 1.09× | 19,837 |
| neocloud | 3.75 | 30,112 | 0.0099 | 0.76× | 13,775 |

## What breaks first

| Level | Resource | Demand | Limit | Ratio |
|---|---|---:|---:|---:|
| incident | hosted RPS limit | 153 | 60.00 | 2.55× |
| incident | hosted TPM limit | 47.67 M | 20.00 M | 2.38× |
| incident | self-hosted fleet (sized for peak) | 30.6 k | 13.2 k | 2.31× |
| peak | hosted RPS limit | 45.83 | 60.00 | 0.76× |
| peak | hosted TPM limit | 14.30 M | 20.00 M | 0.72× |
| peak | self-hosted fleet (sized for peak) | 9.2 k | 13.2 k | 0.69× |
| incident | billing system QPS (40) | 17.36 | 40.00 | 0.43× |
| incident | CRM QPS (200) | 48.61 | 200 | 0.24× |
| peak | billing system QPS (40) | 5.21 | 40.00 | 0.13× |
| peak | CRM QPS (200) | 14.58 | 200 | 0.07× |

