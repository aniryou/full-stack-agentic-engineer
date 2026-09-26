# GPU Deployment Architecture — Exercise Set

**Companion to the primer.** Every answer is derivable from it.

Work through Parts A–C with the primer closed. Check against Part D.

*A note on units before you start:* GPU vendors quote memory in decimal GB (10⁹ bytes); allocators report binary GiB (2³⁰ bytes). The gap is about 7%. All answers below use decimal GB unless marked otherwise, and every number is approximate. That discrepancy is one of several reasons you leave headroom rather than planning to the last byte.

---

## Part A — Recall

Short answers. If you can't answer in one or two sentences, reread the relevant section.

**A1.** In GPU infrastructure, what specifically distinguishes a scale-up network from a scale-out network? Name the dominant technology for each.

**A2.** Which phase of generation is compute-bound and which is memory-bandwidth-bound? Why?

**A3.** Why does increasing batch size help decode throughput so much, but do relatively little for prefill?

**A4.** Weights are a fixed cost. What is the variable cost, and what does it scale with?

**A5.** State the rule governing which parallelism strategies may cross a node boundary and which may not. Give the reason.

**A6.** Roughly how much memory does a model need per billion parameters at BF16? At FP8?

**A7.** What problem does PagedAttention solve, and what operating system concept is it borrowed from?

**A8.** A user reports that output stutters whenever a colleague submits a long document. Name the mechanism and the standard first fix.

**A9.** Why is speculative decoding able to work at all? What property of decode does it exploit?

**A10.** Why is MFU a poor efficiency metric for inference decode? What should you measure instead?

**A11.** What is goodput, and why is it more honest than throughput?

**A12.** Name three ways to reduce KV cache size.

**A13.** Why has the rack, rather than the server, become the atomic unit for large deployments?

**A14.** What does disaggregated serving separate, and what is the hard engineering problem it creates?

**A15.** Name two facilities constraints that gate current-generation GPU deployment independently of GPU supply.

---

## Part B — Numerical workouts

Show your working. Formulas are all in sections 3 and 5 of the primer.

### B1 — Weight budget

A 32B-parameter model. Compute weight memory at BF16, FP8 and FP4.

You have a single 80 GB H100 and you intend to keep 20% of memory as headroom. For each precision, how much memory is left for KV cache? Which precisions are actually viable for serving?

### B2 — KV cache per token

A model has 60 layers, 8 KV heads, head dimension 128, KV cache in FP16.

(a) Bytes of KV cache per token?
(b) For a single request with an 8,000-token context, how much KV cache?
(c) If you switch the KV cache to FP8, what changes?

### B3 — Concurrency budget

You are serving a 70B model at FP8 on a single H200 (141 GB). KV cache costs 320 KiB per token. Reserve 15% of total memory for activations, buffers and fragmentation.

(a) How much memory remains for KV cache?
(b) How many total tokens of KV cache does that buy?
(c) If the average request holds 2,000 tokens of context, roughly how many concurrent requests can you serve?
(d) What happens to (c) if you enable FP8 KV cache?

### B4 — The decode wall

A 70B model at FP8 needs 70 GB of weights read from HBM for every single decode step.

(a) On an H100 (3.35 TB/s HBM), what is the theoretical floor on time per decode step? Convert to tokens/sec at batch size 1.
(b) Same calculation for Rubin (22 TB/s).
(c) Back on the H100, at batch size 64, what is the aggregate token throughput? What is the per-user token rate?
(d) State in one sentence what (c) proves about batching.

### B5 — The interconnect cliff

(a) A Blackwell GPU has 1.8 TB/s of NVLink bandwidth, the marketed bidirectional total. A node's scale-out NIC runs at 800 Gb/s, quoted per direction. Express both links per direction in GB/s, then compute the ratio.
(b) Rubin raises NVLink to 3.6 TB/s (again a bidirectional total). If the NIC stayed at 800 Gb/s, what would happen to the ratio?
(c) What does the trend in (b) imply for how you should partition models over time?

### B6 — Is disaggregation feasible?

A request produces 1.34 GB of KV cache during prefill. Your TTFT budget is 500 ms and prefill itself consumes 200 ms.

(a) How long may the KV transfer take?
(b) Over a 400 Gb/s RDMA link, how long does the transfer actually take?
(c) Over 800 Gb/s?
(d) Is multi-node disaggregation viable for this workload? What would change your answer?

### B7 — Prefix caching ROI

An agentic application serves 10,000 requests per day. Every request carries the same 2,000-token system prompt, followed by an average of 500 tokens of user-specific content.

(a) Total prefill tokens per day with no prefix caching.
(b) Total prefill tokens per day with perfect prefix caching.
(c) Percentage reduction in prefill work.
(d) Why is this optimisation especially valuable for agent workloads specifically?

### B8 — Rack-scale capacity

An NVL72 rack holds 72 GPUs at 288 GB each, giving roughly 20.7 TB of HBM in one NVLink domain.

(a) Ignoring all overhead, how many parameters could you hold at FP4?
(b) Reserving 30% for KV cache and overhead, what is the realistic figure?
(c) A 671B MoE model at FP4 needs roughly 335 GB of weights. What fraction of one rack is that, and what does the remainder buy you?

---

## Part C — Design scenarios

Two to four sentences each. State your reasoning, not just your conclusion.

**C1.** An enterprise wants to deploy a 70B model for internal assistant use. Peak concurrency is 200 users, typical prompts 4,000 tokens, responses 500 tokens. p95 TTFT must stay under one second. Which reference architecture, and what are the first three optimisations you enable?

**C2.** An engineer proposes tensor parallelism of 16 across two 8-GPU nodes to serve a large model. What is wrong with this, and what would you do instead?

**C3.** Your inference platform autoscales on queue depth, but adding a replica takes twelve minutes. Diagnose the cause and give two fixes.

**C4.** A team benchmarks a deployment at 8,000 tokens/sec and declares success. Users complain it feels slow. Reconcile these, and name the metric that would have caught it.

**C5.** Finance asks whether to buy GPUs or rent them. What is the key ratio you would compute, and which two non-obvious costs would you insist on including for the buy case?

**C6.** A colleague wants to jump straight to multi-node prefill/decode disaggregation for a new deployment. What do you tell them to do first, and why?

**C7.** You must serve a 405B model at FP8. You have 8×H100 (80 GB each) in one node. Does it fit? Work out the numbers and state the parallelism configuration.

---

## Part D — Answer key

### Part A

**A1.** Scale-up connects GPUs *within* a tightly-coupled domain (one node or rack) with memory-like semantics — a GPU can read another's HBM at near-local speed. NVLink/NVSwitch. Scale-out connects domains *across* the data centre with message-passing semantics. InfiniBand or RoCE. The bandwidth difference is roughly an order of magnitude; the latency difference is larger.

**A2.** Prefill is compute-bound: it processes all input tokens in one parallel pass, so each weight loaded from memory is used for a great deal of arithmetic. Decode is memory-bandwidth-bound: it must read the entire weight set from HBM to produce a single token, leaving the math units mostly idle.

**A3.** In decode, the expensive operation is loading the weights, and that cost is identical whether you produce one token or sixty-four. Batching amortises it. Prefill already saturates the compute units with a single long prompt, so there is little idle capacity left to fill.

**A4.** KV cache. It scales with concurrent requests × sequence length. It is what determines how many users you can actually serve, and it causes most production OOM failures.

**A5.** Tensor and expert parallelism must stay inside the scale-up domain; pipeline and data parallelism may cross the scale-out fabric. Reason: TP requires an all-reduce synchronisation at every layer, many times per token, so it is intolerant of the latency and bandwidth of a network link. PP passes only boundary activations, and inference DP passes essentially nothing.

**A6.** About 2 GB per billion parameters at BF16; about 1 GB per billion at FP8.

**A7.** Memory fragmentation. Instead of reserving one contiguous block per request sized to the maximum possible length, it allocates KV cache in fixed-size blocks. Borrowed from operating system virtual memory paging. Typically yields 2–4× more concurrent requests from the same memory.

**A8.** A long prefill occupying the GPU and blocking decode steps for other users. First fix: chunked prefill, which breaks the long prefill into pieces and interleaves them with decode. If that is insufficient at scale, prefill/decode disaggregation.

**A9.** Decode is memory-bound, so the compute units have spare capacity. Speculative decoding spends that idle compute verifying several draft tokens in a single forward pass, converting surplus FLOPs into lower latency.

**A10.** MFU measures achieved FLOPs against theoretical peak, but decode is limited by memory bandwidth rather than arithmetic, so a well-tuned decode deployment will show low MFU by construction. Measure memory bandwidth utilisation instead, alongside tokens/sec/GPU.

**A11.** Goodput is requests per second that actually met their latency SLO. Raw throughput can be inflated arbitrarily by increasing batch size, at the cost of latency nobody will accept — so a throughput figure measured at a ten-second TTFT is meaningless.

**A12.** Any three of: grouped-query or multi-query attention (fewer KV heads); FP8 KV cache; paged allocation; prefix caching; shorter maximum context; capping concurrency.

**A13.** Because the size of the scale-up domain is the property that determines how large a model you can tensor-parallelise efficiently. An 8-GPU server gives an 8-GPU domain; an NVL72 rack gives 72 GPUs in a single NVLink domain, which is what makes serving frontier-scale MoE models with low latency practical.

**A14.** It separates prefill and decode onto distinct GPU pools. The hard problem is transferring the KV cache from a prefill worker to a decode worker fast enough that the decode worker does not sit idle — which requires RDMA and careful engineering, since the transfer must fit inside a TTFT budget of a few hundred milliseconds.

**A15.** Power density (past 40 kW per rack for Blackwell) and liquid cooling (mandatory, with no air-cooled option, for Rubin-class systems). Procurement lead times of 6–12 months are a third.

---

### Part B

**B1.**

| Precision | Weights | Free for KV (64 GB usable) | Viable? |
|---|---|---|---|
| BF16 | 64 GB | 0 GB | No |
| FP8 | 32 GB | 32 GB | Yes |
| FP4 | 16 GB | 48 GB | Yes, most headroom |

80 GB with 20% headroom leaves 64 GB usable. BF16 consumes all of it, leaving nothing for KV cache, so the model "fits" but cannot serve a single request. This is exactly the trap the primer warns about in §3.4.

**B2.**
(a) 60 × 8 × 128 × 2 × 2 = **245,760 bytes = 240 KiB per token**
(b) 8,000 × 245,760 ≈ **1.97 GB (1.83 GiB)** for one request
(c) Halves to 120 KiB/token, so about 0.98 GB — doubling either your context length or your concurrency for the same memory.

**B3.**
(a) 15% of 141 GB = 21.2 GB reserve. 141 − 70 − 21.2 = **≈ 50 GB for KV cache**
(b) 50 GB ÷ 320 KiB ≈ **152,000 tokens**
(c) 152,000 ÷ 2,000 ≈ **76 concurrent requests**
(d) FP8 KV halves per-token cost, so roughly **152 concurrent requests**. Note this is a bigger concurrency win than most hardware upgrades, for the price of a config flag.

**B4.**
(a) 70 GB ÷ 3.35 TB/s = **20.9 ms per token → ≈ 48 tokens/sec** at batch 1.
(b) 70 GB ÷ 22 TB/s = **3.2 ms per token → ≈ 314 tokens/sec**. The 6.6× bandwidth improvement translates almost directly into decode speed, which is the whole point of the HBM4 jump.
(c) The 20.9 ms step is unchanged, but produces 64 tokens: **≈ 3,060 tokens/sec aggregate**, still **≈ 48 tokens/sec per user**.
(d) Batching buys throughput essentially for free in decode, because the dominant cost — reading the weights — is paid once regardless of batch size.

**B5.**
(a) NIC: 800 Gb/s ÷ 8 = 100 GB/s per direction. NVLink: 1.8 TB/s is both directions added, so 900 GB/s per direction. Ratio = 900 ÷ 100 = **9×**. Dividing the 1,800 total by the NIC's one-way 100 gives 18×, which double-counts NVLink: compare like with like. The Hopper pair gives the same answer, 450 GB/s per direction of NVLink 4 over a 400 Gb/s (50 GB/s) NIC: **9×** ([roofline primer §1 and §5.1](../roofline-and-fabric/PRIMER.md#51-the-link-ladder), `roofline.fabric.LINKS`).
(b) 1,800 GB/s per direction ÷ 100 = **18×**: the cliff would double. Whether it does depends on the NIC keeping pace; the Rubin generation's NICs are announced at 1.6 Tb/s, 200 GB/s per direction, which would hold the per-GPU ratio near 9× (verify, 2026-09).
(c) The per-GPU ratio has sat near 9× for two generations because NICs doubled with NVLink, while the scale-up domain grew from 8 GPUs to 72 and beyond. So the penalty for letting a tightly coupled collective cross the domain boundary is not shrinking, and more of a model's parallelism can now stay inside the domain. Partition to keep tensor parallelism inside the scale-up domain, and when choosing a platform weight the size of the NVLink domain at least as heavily as per-GPU FLOPs.

**B6.**
(a) 500 − 200 = **300 ms**
(b) 1.34 GB ÷ 50 GB/s = **≈ 27 ms**
(c) 1.34 GB ÷ 100 GB/s = **≈ 13 ms**
(d) Yes, comfortably — the transfer uses under 10% of the budget. What would change the answer: much longer contexts (KV scales linearly, so a 64K-token prompt is 16× this), a tighter TTFT SLO, contended fabric, or a transfer path that falls back off RDMA and goes through the CPU.

**B7.**
(a) 10,000 × 2,500 = **25 million prefill tokens**
(b) 2,000 once, plus 10,000 × 500 = **≈ 5 million tokens**
(c) **80% reduction**
(d) Agent workloads have long, stable system prompts — tool definitions, instructions, retrieved context — that repeat across enormous numbers of requests, and often across many turns of the same session. The shared prefix is a large fraction of total input, so the cache hit rate is high and the saving is close to the theoretical maximum. Ordinary chat workloads share far less.

**B8.**
(a) 20.7 TB ÷ 0.5 bytes/param = **≈ 41 trillion parameters**
(b) 70% of that ≈ **29 trillion parameters**
(c) 335 GB is about **1.6% of the rack's memory**. The remainder is not spare capacity to be embarrassed about — it is what buys you very large KV cache pools, meaning high concurrency and long context, which is where the economics of frontier-model serving actually come from.

---

### Part C

**C1.** Architecture B (Kubernetes cluster), likely 2–4 nodes of 8×H100 or H200. Run the model at FP8 with TP inside each node and DP across them. First three optimisations: continuous batching and paged attention (non-negotiable baseline), FP8 KV cache (roughly doubles concurrency), and chunked prefill (4,000-token prompts at 200 concurrency will otherwise wreck p95 TTFT). Prefix caching is a strong fourth if there is a shared system prompt.

**C2.** TP=16 would force layer-level all-reduce operations across the scale-out fabric, which gives each GPU about 9× less bandwidth per direction than NVLink (B5) and higher latency, and TP synchronises many times per token. Throughput would collapse. Instead: TP=8 within each node, PP=2 across the two nodes — pipeline parallelism only passes boundary activations, which the fabric handles comfortably.

**C3.** Cold-start dominated by loading a very large checkpoint over the network. Twelve minutes means the autoscaler responds long after the traffic spike has passed, so it is not functioning as autoscaling at all. Fixes: keep warm standby replicas sized to your expected burst; move checkpoints to fast local NVMe or a node-local cache so loads are not pulling from object storage; consider tiered scaling where a small model absorbs overflow while large replicas spin up.

**C4.** Throughput and latency trade against each other. 8,000 tokens/sec was almost certainly achieved with a large batch size, which inflates TTFT and inter-token latency. The users are experiencing p95 TTFT and ITL, which the benchmark never reported. Goodput — requests/sec meeting the SLO — would have caught it, as would reporting p95/p99 rather than aggregate throughput.

**C5.** The key ratio is sustained utilisation. Owning generally requires something above roughly 60–70% sustained utilisation to beat renting. Two costs usually omitted from the buy case: facilities — power delivery and liquid cooling retrofit, since current-generation racks exceed 40 kW and Rubin-class systems have no air-cooled option — and the depreciation implied by an annual architecture cadence, where hardware bought today competes with substantially better hardware within a year.

**C6.** Get the fundamentals first: continuous batching, paged attention, prefix caching, FP8 KV cache. Those are the largest wins for almost every workload and require no topology changes. If more is needed, same-node disaggregation across NVLink is the next step, since it avoids the KV transfer problem entirely. Multi-node disaggregation adds a distributed KV transfer path, a control plane, and version-pinning burden, and only pays off at genuine scale.

**C7.** 405B at FP8 = 405 GB of weights. 8×80 GB = 640 GB total. Reserving ~15% overhead leaves roughly 545 GB usable, so after weights you have about 140 GB for KV cache — workable. Configuration: **TP=8 within the single node**, entirely inside the NVLink domain, no pipeline or data parallelism needed for a single replica. Scale by adding replicas (DP across nodes), not by widening TP.

---

## Self-assessment

| Score | Reading |
|---|---|
| Part A mostly correct | You have the vocabulary. Move to B. |
| Part B correct within ~10% | You can size a deployment. This is the practical bar. |
| B4 and B5 correct with reasoning | You understand *why* the architecture looks the way it does. |
| Part C defensible | You can make deployment decisions rather than follow recipes. |

If B3 or B4 gave you trouble, reread §2 and §3 — everything else in the primer follows from those two sections.
