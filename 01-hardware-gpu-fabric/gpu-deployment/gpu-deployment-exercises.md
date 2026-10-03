# GPU Deployment Architecture — Exercise Set

**Companion to the primer.** You can get every answer from the primer.

Do Parts A–C with the primer closed. Then compare your answers with Part D.

*A note on units before you start:* GPU vendors give memory in decimal GB (10⁹ bytes). Allocators report binary GiB (2³⁰ bytes). The difference is approximately 7%. All answers in this file use decimal GB, unless an answer marks a different unit. Every number is approximate. That difference is one of several reasons to keep headroom and not to plan to the last byte.

---

## Part A — Recall

Give short answers. If you cannot answer in one or two sentences, read the applicable section again.

**A1.** In GPU infrastructure, what is the specific difference between a scale-up network and a scale-out network? Name the most common technology for each.

**A2.** Which phase of generation is compute-bound, and which phase is memory-bandwidth-bound? Why?

**A3.** Why does an increase in batch size help decode throughput so much, but give prefill a relatively small gain?

**A4.** Weights are a fixed cost. What is the variable cost, and which quantities does it increase with?

**A5.** State the rule that tells which parallelism strategies can cross a node boundary and which cannot. Give the reason.

**A6.** Approximately how much memory does a model need per billion parameters at BF16? At FP8?

**A7.** What problem does PagedAttention solve? From which operating system concept does it come?

**A8.** A user reports that the output stops and starts each time that a colleague sends a long document. Name the mechanism and the standard first solution.

**A9.** Why can speculative decoding work at all? Which property of decode does it use?

**A10.** Why is MFU a bad efficiency metric for inference decode? What is the correct thing to measure in its place?

**A11.** What is goodput? Why is it more honest than throughput?

**A12.** Name three methods that decrease the size of the KV cache.

**A13.** Why is the rack, and not the server, now the atomic unit for large deployments?

**A14.** What does disaggregated serving separate? What is the difficult technical problem that it causes?

**A15.** Name two facilities constraints that limit current-generation GPU deployment, independently of the GPU supply.

---

## Part B — Numerical workouts

Show your calculations. All the formulas are in sections 3 and 5 of the primer.

### B1 — Weight budget

You have a 32B-parameter model. Calculate the weight memory at BF16, FP8 and FP4.

You have a single 80 GB H100. You intend to keep 20% of memory as headroom. For each precision, how much memory stays for KV cache? Which precisions can you actually use to serve the model?

### B2 — KV cache per token

A model has 60 layers, 8 KV heads and head dimension 128. The KV cache is in FP16.

(a) What is the number of bytes of KV cache per token?

(b) How much KV cache does a single request with an 8,000-token context use?

(c) If you change the KV cache to FP8, what changes?

### B3 — Concurrency budget

You serve a 70B model at FP8 on a single H200 (141 GB). The KV cache costs 320 KiB per token. Keep 15% of total memory for activations, buffers and fragmentation.

(a) How much memory stays available for KV cache?

(b) How many tokens of KV cache in total does that memory give you?

(c) If the average request holds 2,000 tokens of context, approximately how many concurrent requests can you serve?

(d) How does the answer to (c) change if you turn on FP8 KV cache?

### B4 — The decode wall

A 70B model at FP8 has 70 GB of weights. Every single decode step must read all of them from HBM.

(a) On an H100 (3.35 TB/s HBM), what is the theoretical minimum time per decode step? Give the result in tokens/sec at batch size 1.

(b) Do the same calculation for Rubin (22 TB/s).

(c) Go back to the H100. At batch size 64, what is the aggregate token throughput? What is the per-user token rate?

(d) State in one sentence what (c) proves about batching.

### B5 — The interconnect cliff

(a) A Blackwell GPU has 1.8 TB/s of NVLink bandwidth. This value is the bidirectional total that the vendor advertises. The scale-out NIC of a node operates at 800 Gb/s. The vendor gives this value per direction. Give the speed of both links per direction in GB/s. Then calculate the ratio.

(b) Rubin increases NVLink to 3.6 TB/s (again a bidirectional total). If the NIC stays at 800 Gb/s, how does the ratio change?

(c) What does the trend in (b) tell you about how to partition models over time?

### B6 — Is disaggregation feasible?

A request makes 1.34 GB of KV cache during prefill. Your TTFT budget is 500 ms. The prefill itself uses 200 ms.

(a) What is the maximum time for the KV transfer?

(b) On a 400 Gb/s RDMA link, how much time does the transfer actually use?

(c) On an 800 Gb/s link?

(d) Is multi-node disaggregation viable for this workload? Which conditions change your answer?

### B7 — Prefix caching ROI

An agentic application serves 10,000 requests per day. Each request has the same 2,000-token system prompt. After the system prompt, each request has an average of 500 tokens of user-specific content.

(a) Calculate the total prefill tokens per day with no prefix caching.

(b) Calculate the total prefill tokens per day with perfect prefix caching.

(c) Calculate the percentage decrease in prefill work.

(d) Why is this optimisation especially valuable for agent workloads in particular?

### B8 — Rack-scale capacity

An NVL72 rack holds 72 GPUs at 288 GB each. This gives approximately 20.7 TB of HBM in one NVLink domain.

(a) If you ignore all overhead, how many parameters can you hold at FP4?

(b) If you keep 30% for KV cache and overhead, what is the realistic number?

(c) A 671B MoE model at FP4 needs approximately 335 GB of weights. What fraction of one rack is that? What does the remainder give you?

---

## Part C — Design scenarios

Write two to four sentences for each scenario. Give your reasons, not only your conclusion.

**C1.** An enterprise wants to deploy a 70B model as an internal assistant. Peak concurrency is 200 users. Typical prompts are 4,000 tokens, and responses are 500 tokens. p95 TTFT must stay under one second. Which reference architecture do you select? What are the first three optimisations that you turn on?

**C2.** An engineer proposes tensor parallelism of 16 across two 8-GPU nodes to serve a large model. What is incorrect about this plan? What do you do in its place?

**C3.** Your inference platform autoscales on queue depth. But the addition of a replica takes twelve minutes. Find the cause and give two solutions.

**C4.** A team measures a deployment at 8,000 tokens/sec in a benchmark and says that it is a success. Users complain that it feels slow. Explain how both statements are true. Name the metric that shows this problem.

**C5.** The finance team asks if it is better to buy GPUs or to rent them. What is the key ratio that you calculate? Which two non-obvious costs do you make sure that the buy case includes?

**C6.** A colleague wants to go directly to multi-node prefill/decode disaggregation for a new deployment. What do you tell them to do first? Why?

**C7.** You must serve a 405B model at FP8. You have 8×H100 (80 GB each) in one node. Does the model fit? Calculate the numbers and state the parallelism configuration.

---

## Part D — Answer key

### Part A

**A1.** Scale-up connects GPUs *within* a tightly-coupled domain (one node or rack) with memory-like semantics. A GPU can read the HBM of another GPU at near-local speed. The technology is NVLink/NVSwitch.

Scale-out connects domains *across* the data centre with message-passing semantics. The technology is InfiniBand or RoCE. The bandwidth difference is approximately an order of magnitude. The latency difference is larger.

**A2.** Prefill is compute-bound. It processes all input tokens in one parallel pass. Thus the GPU uses each weight that it loads from memory for a large quantity of arithmetic.

Decode is memory-bandwidth-bound. It must read the full weight set from HBM to make a single token. Thus the math units are idle most of the time.

**A3.** In decode, the high-cost operation is the load of the weights. That cost is the same when you make one token and when you make sixty-four tokens. Batching amortises that cost. Prefill already saturates the compute units with a single long prompt. Thus only a small quantity of idle capacity stays available to fill.

**A4.** The variable cost is the KV cache. It increases with concurrent requests × sequence length. It sets how many users you can actually serve. It also causes most OOM failures in production.

**A5.** Tensor and expert parallelism must stay inside the scale-up domain. Pipeline and data parallelism can cross the scale-out fabric. The reason: TP must do an all-reduce synchronisation at every layer, many times per token. Thus TP cannot accept the latency and bandwidth of a network link. PP sends only boundary activations, and inference DP sends almost nothing.

**A6.** Approximately 2 GB per billion parameters at BF16. Approximately 1 GB per billion at FP8.

**A7.** PagedAttention solves memory fragmentation. It does not set aside one contiguous block for each request, with the size of the maximum possible length. In its place, it allocates KV cache in fixed-size blocks. The concept comes from virtual memory paging in an operating system. Typically, it gives 2–4× more concurrent requests from the same memory.

**A8.** The mechanism is a long prefill that occupies the GPU and blocks decode steps for other users. The first solution is chunked prefill. Chunked prefill divides the long prefill into pieces and interleaves them with decode. If that is not sufficient at scale, use prefill/decode disaggregation.

**A9.** Decode is memory-bound. Thus the compute units have spare capacity. Speculative decoding uses that idle compute to verify several draft tokens in a single forward pass. It changes surplus FLOPs into lower latency.

**A10.** MFU compares the achieved FLOPs with the theoretical peak. But memory bandwidth, not arithmetic, limits decode. Thus a well-adjusted decode deployment shows low MFU by construction. Measure memory bandwidth utilisation in its place, together with tokens/sec/GPU.

**A11.** Goodput is the number of requests per second that actually met their latency SLO. You can increase raw throughput as much as you want with a larger batch size. But the cost is a latency that nobody will accept. Thus a throughput number that you measure at a ten-second TTFT is meaningless.

**A12.** Any three of these:

- grouped-query or multi-query attention (fewer KV heads)
- FP8 KV cache
- paged allocation
- prefix caching
- a shorter maximum context
- a limit on concurrency

**A13.** The size of the scale-up domain is the property that sets how large a model you can divide efficiently with tensor parallelism. An 8-GPU server gives an 8-GPU domain. An NVL72 rack gives 72 GPUs in a single NVLink domain. This domain makes it practical to serve frontier-scale MoE models with low latency.

**A14.** It puts prefill and decode on different GPU pools. The difficult problem is the transfer of the KV cache from a prefill worker to a decode worker. The transfer must be sufficiently fast that the decode worker is not idle. A fast transfer needs RDMA and careful technical work, because the transfer must fit in a TTFT budget of a few hundred milliseconds.

**A15.** The two constraints are power density and liquid cooling. Power density is more than 40 kW per rack for Blackwell. Liquid cooling is mandatory for Rubin-class systems, with no air-cooled option. Procurement lead times of 6–12 months are a third constraint.

---

### Part B

**B1.**

| Precision | Weights | Free for KV (64 GB usable) | Viable? |
|---|---|---|---|
| BF16 | 64 GB | 0 GB | No |
| FP8 | 32 GB | 32 GB | Yes |
| FP4 | 16 GB | 48 GB | Yes, most headroom |

80 GB with 20% headroom leaves 64 GB usable. BF16 uses all of it and leaves nothing for KV cache. Thus the model "fits" but cannot serve a single request. This is exactly the trap that the primer gives a warning about in §3.4.

**B2.**

(a) 60 × 8 × 128 × 2 × 2 = **245,760 bytes = 240 KiB per token**

(b) 8,000 × 245,760 ≈ **1.97 GB (1.83 GiB)** for one request

(c) The cost decreases by half to 120 KiB/token. Thus the request uses approximately 0.98 GB. With the same memory, you can double your context length or your concurrency.

**B3.**

(a) 15% of 141 GB = 21.2 GB reserve. 141 − 70 − 21.2 = **≈ 50 GB for KV cache**

(b) 50 GB ÷ 320 KiB ≈ **152,000 tokens**

(c) 152,000 ÷ 2,000 ≈ **76 concurrent requests**

(d) FP8 KV decreases the per-token cost by half. Thus you get approximately **152 concurrent requests**. Note that this is a larger concurrency gain than most hardware upgrades give, and its price is one config flag.

**B4.**

(a) 70 GB ÷ 3.35 TB/s = **20.9 ms per token**, thus **≈ 48 tokens/sec** at batch 1.

(b) 70 GB ÷ 22 TB/s = **3.2 ms per token**, thus **≈ 314 tokens/sec**. The 6.6× bandwidth improvement becomes decode speed almost directly. This is the full purpose of the HBM4 jump.

(c) The 20.9 ms step does not change, but it makes 64 tokens. The result is **≈ 3,060 tokens/sec aggregate**, and still **≈ 48 tokens/sec per user**.

(d) In decode, batching gives throughput almost for free, because the largest cost (the read of the weights) occurs one time for any batch size.

**B5.**

(a) NIC: 800 Gb/s ÷ 8 = 100 GB/s per direction. NVLink: 1.8 TB/s is the sum of both directions, thus 900 GB/s per direction. Ratio = 900 ÷ 100 = **9×**. If you divide the 1,800 total by the one-way 100 of the NIC, you get 18×. That result counts NVLink two times. Compare the same type of value on each side.

The Hopper pair gives the same answer. NVLink 4 has 450 GB/s per direction, and the NIC has 400 Gb/s (50 GB/s). The ratio is **9×** ([roofline primer §1 and §5.1](../roofline-and-fabric/PRIMER.md#51-the-link-ladder), `roofline.fabric.LINKS`).

(b) 1,800 GB/s per direction ÷ 100 = **18×**. Thus the cliff doubles in that case. This result occurs only if the NIC speed does not increase at the same rate as NVLink. The NICs of the Rubin generation have an announced speed of 1.6 Tb/s, 200 GB/s per direction. If the NICs reach that speed, the per-GPU ratio stays near 9× (verify, 2026-09).

(c) The per-GPU ratio stayed near 9× for two generations, because the NICs doubled together with NVLink. During the same time, the scale-up domain grew from 8 GPUs to 72 and more. Thus the penalty for a tightly coupled collective that crosses the domain boundary does not decrease. Also, more of the parallelism of a model can now stay inside the domain. Partition the model so that tensor parallelism stays inside the scale-up domain. When you select a platform, make the size of the NVLink domain at least as important as per-GPU FLOPs.

**B6.**

(a) 500 − 200 = **300 ms**

(b) 1.34 GB ÷ 50 GB/s = **≈ 27 ms**

(c) 1.34 GB ÷ 100 GB/s = **≈ 13 ms**

(d) Yes, with a large margin. The transfer uses less than 10% of the budget. These conditions can change the answer:

- much longer contexts (KV increases linearly, thus a 64K-token prompt is 16× this)
- a tighter TTFT SLO
- a contended fabric
- a transfer path that goes through the CPU in place of RDMA

**B7.**

(a) 10,000 × 2,500 = **25 million prefill tokens**

(b) 2,000 once, plus 10,000 × 500 = **≈ 5 million tokens**

(c) **80% reduction**

(d) Agent workloads have long, stable system prompts. These prompts contain tool definitions, instructions and retrieved context. They repeat across very large numbers of requests, and often across many turns of the same session. The shared prefix is a large fraction of the total input. Thus the cache hit rate is high, and the gain is near the theoretical maximum. Ordinary chat workloads share much less.

**B8.**

(a) 20.7 TB ÷ 0.5 bytes/param = **≈ 41 trillion parameters**

(b) 70% of that ≈ **29 trillion parameters**

(c) 335 GB is approximately **1.6% of the rack's memory**. Do not think of the remainder as wasted spare capacity. It gives you very large KV cache pools. Thus you get high concurrency and long context. The economics of frontier-model serving actually come from these pools.

---

### Part C

**C1.** Use Architecture B (Kubernetes cluster), probably with 2–4 nodes of 8×H100 or H200. Run the model at FP8 with TP inside each node and DP across the nodes. The first three optimisations are:

- continuous batching and paged attention (the mandatory baseline)
- FP8 KV cache (approximately doubles concurrency)
- chunked prefill (without it, 4,000-token prompts at 200 concurrency will cause a large increase in p95 TTFT)

Prefix caching is a strong fourth optimisation if there is a shared system prompt.

**C2.** With TP=16, layer-level all-reduce operations must cross the scale-out fabric. That fabric gives each GPU approximately 9× less bandwidth per direction than NVLink (B5), and higher latency. Also, TP synchronises many times per token. Thus the throughput will collapse.

In its place, use TP=8 within each node and PP=2 across the two nodes. Pipeline parallelism sends only boundary activations, and the fabric carries them easily.

**C3.** The cause is the cold start, and most of the cold start is the load of a very large checkpoint over the network. With twelve minutes, the autoscaler responds long after the traffic spike is over. Thus it does not operate as autoscaling at all. Solutions:

- Keep warm standby replicas, with a size that matches your expected burst.
- Move checkpoints to fast local NVMe or a node-local cache, so that loads do not come from object storage.
- Think about tiered scaling: a small model takes the overflow while the large replicas start.

**C4.** Throughput and latency trade against each other. The team almost certainly got 8,000 tokens/sec with a large batch size. A large batch size increases TTFT and inter-token latency. The users feel the p95 TTFT and ITL, and the benchmark never reported these values.

Goodput (requests/sec that meet the SLO) shows this problem. A report of p95/p99, in place of aggregate throughput, also shows it.

**C5.** The key ratio is sustained utilisation. To be better than rental, ownership usually needs more than approximately 60–70% sustained utilisation. The buy case usually omits two costs:

- Facilities: power delivery and a liquid cooling retrofit. These costs occur because current-generation racks are more than 40 kW, and Rubin-class systems have no air-cooled option.
- The depreciation that comes from an annual architecture cadence. This depreciation occurs because hardware that you buy today competes with much better hardware within a year.

**C6.** Get the fundamentals first: continuous batching, paged attention, prefix caching, FP8 KV cache. These give the largest gains for almost every workload, and they need no topology changes. If you need more, the next step is same-node disaggregation across NVLink, because it has no KV transfer problem at all. Multi-node disaggregation adds a distributed KV transfer path, a control plane and the burden of pinned versions. It gives a return only at genuine scale.

**C7.** 405B at FP8 = 405 GB of weights. 8×80 GB = 640 GB total. If you keep ~15% for overhead, approximately 545 GB is usable. Thus after the weights, you have approximately 140 GB for KV cache, which is workable.

Configuration: **TP=8 within the single node**, fully inside the NVLink domain. A single replica needs no pipeline or data parallelism. To scale, add replicas (DP across nodes). Do not make TP wider.

---

## Self-assessment

| Score | Reading |
|---|---|
| Part A mostly correct | You have the vocabulary. Move to B. |
| Part B correct within ~10% | You can size a deployment. This is the practical bar. |
| B4 and B5 correct with reasoning | You understand *why* the architecture has its shape. |
| Part C defensible | You can make deployment decisions. You do not only do the steps of a recipe. |

If B3 or B4 gave you problems, read §2 and §3 again. Everything else in the primer comes from those two sections.
