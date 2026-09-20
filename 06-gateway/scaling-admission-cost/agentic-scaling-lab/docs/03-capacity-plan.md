# Capacity plan — 100,000 conversations/day

6 turns/conversation, 2.2 calls/turn, 5,000 in (2,700 cached) / 350 out tokens per call, peak ×3, incident ×10, turn 6 s, think 60 s, gemini-3.5-flash with 35% of calls on gemini-3.5-flash-lite, baseline 10 M TPM.

| Quantity | Average | Peak | Incident |
|---|---:|---:|---:|
| Conversations / s | 1.16 | 3.47 | 11.57 |
| Turns / s | 6.94 | 20.83 | 69.44 |
| Model calls / s | 15.28 | 45.83 | 153 |
| Input tokens / min | 4.58 M | 13.75 M | 45.83 M |
| … of which uncached | 2.11 M | 6.33 M | 21.08 M |
| Output tokens / min | 320.8 k | 962.5 k | 3.21 M |
| Input TPM vs baseline (×) | 0.46 | 1.38 | 4.58 |
| In-flight turns (Little's law) | 41.67 | 125 | 417 |
| Concurrent sessions | 458 | 1.4 k | 4.6 k |
| Orchestrator instances | 1.00 | 3.00 | 8.00 |

The 10 M TPM baseline sustains about 14.2 turns/s, i.e. 85 turns in flight — the starting value for the in-flight cap.

## Cost per conversation

- all gemini-3.5-flash, no caching: $0.1406
- all gemini-3.5-flash, cached prefix: $0.0925
- routed + cached: **$0.0677** → $205,831/month (vs $427,363 unoptimised)

## Provisioned Throughput (standard-model share)

- GSUs: average 69, peak 207, incident 688
- monthly cost of the average GSUs: 1-year $137,999, 1-month $186,298
- $/M burndown tokens: paygo 1.50, 1w 2.94, 1m 1.52, 3m 1.35, 1y 1.13
- break-even utilisation: 1w 196%, 1m 101%, 3m 90%, 1y 75%

## What breaks first

| Level | Resource | Demand | Limit | Ratio |
|---|---|---:|---:|---:|
| incident | model TPM baseline | 45.83 M | 10.00 M | 4.58× |
| peak | model TPM baseline | 13.75 M | 10.00 M | 1.38× |
| incident | billing system QPS (40) | 17.36 | 40.00 | 0.43× |
| incident | CRM QPS (200) | 48.61 | 200 | 0.24× |
| peak | billing system QPS (40) | 5.21 | 40.00 | 0.13× |
| peak | CRM QPS (200) | 14.58 | 200 | 0.07× |

