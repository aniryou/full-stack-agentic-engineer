# GPU Deployment Architecture for LLM Scale-Up

**A primer for people who understand models but not the machines they run on.**

*Written August 2026. Hardware specifics move fast; the physics and the mental models don't.*

---

## 0. First, a terminology trap

"Scale-up" has two meanings and they collide constantly in this field.

**Colloquially**, it means "make the deployment bigger" — serve more users, handle bigger models.

**Technically**, in GPU infrastructure, scale-up has a precise meaning that sits opposite scale-out:

- **Scale-up network** — the fabric connecting GPUs *inside* one tightly-coupled domain (one server, or one rack). NVIDIA's version is NVLink/NVSwitch. It behaves like memory: a GPU can read another GPU's memory almost as if it were local.
- **Scale-out network** — the fabric connecting *domains* to each other across the data centre. InfiniBand or RDMA-over-Ethernet. It behaves like a network: you send messages, and it's roughly two orders of magnitude slower.

Almost every architectural decision in this document comes down to one question: **which side of that boundary does this piece of work live on?**

The standard pattern is scale up first, then scale out. Build the biggest tightly-coupled domain you can, then replicate it.

---

## 1. Why GPUs at all

An LLM forward pass is, in bulk, a long sequence of large matrix multiplications. Each output element is independent of the others, which makes the work embarrassingly parallel.

A CPU has tens of powerful, general-purpose cores. A GPU has thousands of simple ones plus dedicated matrix-multiply hardware (NVIDIA calls these *Tensor Cores*). For this specific shape of work, a GPU is 10–100× faster.

But raw arithmetic isn't usually the bottleneck. The thing that actually limits LLM serving is **memory** — both how much you have and how fast you can read it.

Modern accelerators use **HBM** (High Bandwidth Memory), stacks of DRAM sitting on the same package as the compute die, connected by an extremely wide bus. HBM is the reason a data centre GPU costs what it does, and HBM supply is currently the binding constraint on the whole industry.

---

## 2. The single most important mental model: prefill vs decode

Generating a response has two phases with **opposite** hardware characteristics. Internalise this and most deployment decisions become obvious.

### Prefill (processing the prompt)

The model reads all input tokens at once, in parallel, in a single forward pass. Every weight it loads gets used for a lot of arithmetic.

- **Compute-bound.** The GPU's math units are the limiting factor.
- Scales with prompt length.
- Determines **TTFT** (time to first token).
- Large batches don't help much — a single long prompt already saturates the GPU.

### Decode (generating tokens)

The model generates one token, then feeds it back in and generates the next. Each step must load *the entire set of model weights* from HBM to produce *one token*.

- **Memory-bandwidth-bound.** The math units sit mostly idle waiting for data.
- Determines **ITL** (inter-token latency), which users experience as generation speed.
- Batching helps enormously — if you load the weights once and use them for 64 requests simultaneously, you've amortised the expensive part 64×.

### Why this matters

These two phases want different things from hardware and different things from a scheduler. Running them on the same GPU means they interfere: a long prefill blocks decode steps, causing stuttering output. This tension drives most of the interesting architecture in section 8.

There's a useful shorthand here: **arithmetic intensity**, the ratio of FLOPs performed to bytes moved. Prefill has high arithmetic intensity. Decode has terrible arithmetic intensity. Almost every optimisation in LLM serving is an attempt to raise decode's arithmetic intensity.

---

## 3. Sizing: does the model fit?

Three things consume GPU memory.

### 3.1 Weights

```
weight memory = parameters × bytes per parameter
```

| Precision | Bytes/param | 70B model | 405B model |
|---|---|---|---|
| FP32 | 4 | 280 GB | 1.6 TB |
| FP16 / BF16 | 2 | 140 GB | 810 GB |
| FP8 | 1 | 70 GB | 405 GB |
| FP4 / INT4 | 0.5 | 35 GB | 203 GB |

**Rule of thumb: at BF16, a model needs roughly 2 GB per billion parameters just for weights.**

Quantisation (running at lower precision) is the single biggest lever you have. Current-generation hardware has native FP4 support, which is why 4-bit inference went from a research curiosity to the production default.

### 3.2 KV cache

During decode, the model caches the key and value vectors for every token it has already seen, so it doesn't recompute them each step. This cache grows linearly with sequence length **and** with the number of concurrent requests.

```
KV bytes per token = layers × kv_heads × head_dim × 2 (K and V) × bytes_per_element
```

Worked example, roughly a 70B-class model at FP16:

```
80 layers × 8 KV heads × 128 dims × 2 × 2 bytes ≈ 320 KB per token
```

A 4,000-token prompt therefore costs about **1.3 GB of KV cache for a single request**. Multiply by your concurrency target.

This is the number that surprises people. Weights are a fixed cost you pay once. **KV cache is the variable cost that determines how many users you can actually serve**, and it's what causes most out-of-memory failures in production.

Levers on KV cache: grouped-query or multi-query attention (fewer KV heads), FP8 KV cache (halves it), paged allocation, and prefix caching.

### 3.3 Everything else

Activations, CUDA graphs, communication buffers, fragmentation, the framework itself. **Budget 10–20% headroom.** A deployment planned to 98% memory utilisation will fall over.

### 3.4 Putting it together

For a 70B model at FP8 on one 80 GB GPU: 70 GB of weights leaves 10 GB, minus overhead, for KV cache. That's roughly 20–25 concurrent 1K-token requests. Workable for a demo, not for production. This is how you end up needing multiple GPUs even when the model technically "fits".

---

## 4. When one GPU isn't enough: the parallelism menu

Five ways to split work across GPUs. They compose — real deployments use two or three at once.

### Tensor parallelism (TP)
Split each individual layer's matrices across GPUs. Every GPU holds a slice of every layer.

- **Communication: very high.** GPUs must synchronise (all-reduce) at every layer, many times per token.
- **Therefore: only inside the scale-up domain.** TP across a slow network is catastrophic.
- Use when the model won't fit on one GPU, or to cut latency.

### Pipeline parallelism (PP)
Split the model by layer. GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on.

- **Communication: low.** Only the activations at each boundary get passed.
- **Therefore: fine across the scale-out network.**
- Cost: "pipeline bubbles" — GPUs idle waiting for the previous stage. Mitigated by splitting batches into micro-batches.

### Data parallelism (DP) / replication
Full copy of the model on each GPU or group; requests are load-balanced across replicas.

- **Communication: near zero for inference.** (For training, gradients must be all-reduced — that's the dominant traffic in a training cluster.)
- Scales throughput, not model size. This is your horizontal autoscaling axis.

### Expert parallelism (EP)
For Mixture-of-Experts models, place different experts on different GPUs and route tokens to them. Communication is an all-to-all shuffle, which is bandwidth-hungry and load-imbalance-prone. MoE is now standard for frontier models, so this matters more than it used to.

### Sequence / context parallelism
Split a single long sequence across GPUs. Needed for very long context windows where the KV cache for one request exceeds one GPU.

### The rule that follows from all of this

> **Tensor and expert parallelism go inside the NVLink domain. Pipeline and data parallelism go across the network.**

The interconnect hierarchy dictates the partitioning strategy. Not the other way around.

---

## 5. The interconnect hierarchy

This is the actual "architecture" in GPU deployment architecture. Order-of-magnitude numbers, current generation:

| Link | Where | Rough bandwidth |
|---|---|---|
| HBM (on-package memory) | GPU ↔ its own memory | 3.4 TB/s (H100) → 8 TB/s (Blackwell) → 22 TB/s (Rubin) |
| **NVLink / NVSwitch** (scale-up) | GPU ↔ GPU, same node/rack | 900 GB/s (H100) → 1.8 TB/s (Blackwell) → 3.6 TB/s (Rubin) |
| PCIe Gen5 ×16 | GPU ↔ CPU/NIC | ~64 GB/s |
| **InfiniBand / RoCE** (scale-out) | node ↔ node | ~50–100 GB/s per NIC (400–800 Gb/s) |
| Storage / general Ethernet | cluster ↔ storage | ~1–12 GB/s |

Note the cliff. Scale-up bandwidth is roughly **an order of magnitude or more** above scale-out. Latency differs by even more: nanoseconds on NVLink, microseconds over the fabric.

Two practical consequences:

1. **Never let a tightly-coupled collective operation cross the cliff.** A tensor-parallel all-reduce that spans nodes will destroy your throughput.
2. **The size of the scale-up domain is the single most important property of your hardware platform.** An 8-GPU server gives you an 8-GPU domain. A rack-scale system gives you 72.

---

## 6. The rack is the new unit of deployment

The most significant recent shift is that the *rack*, not the server, has become the atomic unit for large deployments.

NVIDIA's NVL72 systems place 72 GPUs into a single NVLink domain within one liquid-cooled cabinet, so all 72 can be treated as one large pool of memory for tensor parallelism. This is what makes serving a 671B-parameter MoE model with low latency practical.

Current landscape as of mid-2026:

- **Hopper (H100 80 GB, H200 141 GB)** — still ubiquitous, still the price/performance workhorse for most enterprise workloads.
- **Blackwell Ultra (B300 / GB300 NVL72)** — 288 GB HBM3e at 8 TB/s, ~1,400 W per GPU. The volume production part.
- **Vera Rubin (R100 / VR200)** — entered production June 2026, partner availability H2 2026. 288 GB of HBM4 at 22 TB/s, NVLink 6 at 3.6 TB/s per GPU. Availability is constrained by TSMC 3nm and HBM4 supply.
- **AMD** — MI300X/MI325X mature in production; MI400/MI450 ramping. Credible, especially on memory capacity per GPU, with ROCm as the software risk.
- **Custom silicon** — Google TPU, AWS Trainium/Inferentia, Microsoft Maia, Meta MTIA. Large volumes, mostly captive to internal hyperscaler workloads.

### The facilities constraint nobody plans for

Power density has escalated from ~10 kW/rack for conventional compute to 40 kW+ for Blackwell, and Rubin-class racks require 100% direct-to-chip liquid cooling with no air-cooled option. If you operate your own data centre, cooling and power delivery — not GPU procurement — will be your gating item. Lead times for allocation run 6–12 months.

For most teams this argues for cloud or neocloud consumption rather than owned infrastructure, unless you have sustained utilisation above roughly 60–70%.

---

## 7. The software stack, layer by layer

```
┌─────────────────────────────────────────┐
│  Application / agent framework          │
├─────────────────────────────────────────┤
│  Gateway: auth, rate limits, routing,   │
│  quotas, observability, cost accounting │
├─────────────────────────────────────────┤
│  Orchestrator: Dynamo, llm-d, Ray Serve │
│  (routing, autoscaling, disaggregation) │
├─────────────────────────────────────────┤
│  Inference engine: vLLM / SGLang /      │
│  TensorRT-LLM                           │
├─────────────────────────────────────────┤
│  Kubernetes + GPU Operator + scheduler  │
├─────────────────────────────────────────┤
│  Container runtime, NCCL, CUDA, driver  │
├─────────────────────────────────────────┤
│  GPUs, NVLink, NICs, storage, cooling   │
└─────────────────────────────────────────┘
```

### The inference engine is where most of the value lives

Do not write your own serving loop. The engines implement a set of techniques that collectively deliver 10–20× over naive `model.generate()`:

- **Continuous batching** — instead of waiting for a whole batch to finish, evict completed sequences and admit new ones every step. Keeps the GPU busy. Biggest single win.
- **PagedAttention** — allocate KV cache in fixed-size blocks like OS virtual memory, rather than one contiguous reservation per request. Eliminates fragmentation; typically 2–4× more concurrent requests from the same memory.
- **Prefix caching** — reuse the KV cache for shared prefixes. If 10,000 requests share a 2,000-token system prompt, you compute it once. Enormous for agentic workloads with long, stable system prompts.
- **Chunked prefill** — break long prefills into chunks and interleave them with decode steps, so one long prompt doesn't stall everyone else's token stream.
- **Speculative decoding** — a small draft model proposes several tokens; the big model verifies them in one pass. Trades spare compute for lower latency. Works because decode is memory-bound and has compute to spare.
- **Quantisation** — FP8 or FP4 weights and KV cache.

Which engine: **vLLM** is the default open-source choice with the broadest model coverage. **SGLang** is strong on structured output and prefix-heavy workloads (its RadixAttention prefix cache is excellent). **TensorRT-LLM** gives the best raw performance on NVIDIA hardware at the cost of a compilation step and less flexibility.

### Kubernetes specifics

GPUs are not like CPUs to a scheduler. They're indivisible, expensive, and topology-sensitive.

- **NVIDIA GPU Operator** handles drivers, device plugin, and monitoring as a managed lifecycle.
- **Topology awareness matters.** A pod requesting 8 GPUs must get 8 GPUs on the same NVLink domain, not 8 scattered across a cluster.
- **Gang scheduling** — a distributed job needs all its workers or none of them. Partial allocation deadlocks the cluster.
- **Node pools by GPU type**, with taints and tolerations, so batch jobs don't land on latency-critical inference nodes.
- **Model loading is a real bottleneck.** A 405B model checkpoint is hundreds of GB. Cold-start times of 10+ minutes will wreck your autoscaling story unless you plan for fast storage, caching, or pre-warmed replicas.

---

## 8. Prefill/decode disaggregation

The current architectural frontier, and a direct consequence of section 2.

Instead of running both phases on the same GPUs, you run **separate pools**: prefill workers and decode workers. When prefill finishes, the KV cache is transferred over the fabric to a decode worker, which starts generating immediately.

**Why it helps:**
- Each pool can be sized, tuned, and parallelised independently.
- Prefill workers are never interrupted by decode, so TTFT drops sharply.
- Decode workers are never stalled by long prefills, so token streams stay smooth.
- You can even use different hardware for each phase.

**Why it's hard:** you have to move the KV cache fast enough that the decode worker isn't idle. At roughly 1.3 GB per 4K-token request, and a TTFT budget under 500 ms, the transfer must complete in a couple hundred milliseconds. This requires RDMA and careful engineering.

**Tooling:** NVIDIA **Dynamo** sits above the engines as an orchestration layer with a smart router, a planner, and NIXL for transfers. **llm-d** (Red Hat/IBM/Google, donated to CNCF in March 2026) is the Kubernetes-native equivalent built on vLLM. Both vLLM and SGLang support disaggregation directly.

**Sequencing advice:** get continuous batching, paged attention, prefix caching, and FP8 KV cache working first. Those are the largest wins for most workloads. Same-node disaggregation across NVLink is the sensible next step. Multi-node disaggregation is worth it at scale, and mostly not before.

---

## 9. Training clusters vs inference clusters

Very different shapes, and conflating them causes expensive mistakes.

| | Training | Inference |
|---|---|---|
| Coupling | One synchronous job across all GPUs | Many independent replicas |
| Failure impact | One dead GPU stalls everything | One dead replica sheds some traffic |
| Network | All-reduce-dominated; needs full-bisection fabric | Mostly north-south; modest east-west |
| Storage | Huge checkpoints, high write bandwidth | Read-heavy model loading |
| Scaling unit | The whole cluster | The replica |
| Key metric | MFU (Model FLOPs Utilisation) | Tokens/sec/GPU, $/M tokens, latency SLOs |
| Elasticity | Poor; fixed allocation for weeks | Good; autoscale on queue depth |

Training needs checkpointing and fault tolerance as first-class concerns, because at 10,000 GPUs something fails constantly. Inference needs the opposite: fast, independent, cheap replicas.

---

## 10. The numbers to hold yourself to

**Latency**
- **TTFT** — time to first token. Dominated by prefill and queueing. Target under ~500 ms for interactive use.
- **ITL / TPOT** — inter-token latency. Dominated by decode. Under ~50 ms feels faster than reading speed.
- Always report **p50, p95 and p99**. Averages hide everything that matters.

**Throughput**
- Output tokens/sec/GPU is the honest efficiency number.
- **Goodput** — requests/sec that actually met their SLO. Better than raw throughput, because throughput measured with a 10-second TTFT is a lie.

**Efficiency**
- **MFU** — what fraction of theoretical FLOPs you're achieving. 40–50% is good for training. Inference decode is memory-bound so MFU is naturally low; measure bandwidth utilisation instead.
- **$/million tokens** — the number your finance team cares about, and the honest basis for build-vs-buy.

**The tradeoff you can't escape:** larger batches raise throughput and lower cost per token, but raise latency. There is no configuration that optimises both. Pick your SLO first, then maximise throughput subject to it.

---

## 11. Three reference architectures

### A. Single node (1–8 GPUs)
One 8×H100/H200 server. Model fits with tensor parallelism inside the node. vLLM, one process, a load balancer in front. Good to roughly tens of requests/sec.
*Right for: most enterprise deployments, 7B–70B models, internal tooling.*

### B. Kubernetes cluster (tens to hundreds of GPUs)
Multiple nodes, GPU Operator, TP within nodes and DP across them. Horizontal autoscaling on queue depth. Prefix caching enabled. Separate node pools per model and per workload class. A gateway handling routing, quotas, and per-tenant cost accounting.
*Right for: multi-tenant platforms, several models, production SLOs.*

### C. Rack-scale disaggregated (hundreds to thousands of GPUs)
NVL72-class systems. Disaggregated prefill and decode pools via Dynamo or llm-d. Distributed KV cache pool shared across workers. FP4 quantisation. Topology-aware scheduling. Multi-region for availability.
*Right for: serving frontier-scale MoE models, or inference at genuine platform volume.*

Most organisations should start at A, and are wrong about needing C.

---

## 12. Failure modes to design against

- **KV cache OOM under load.** The deployment works in testing at concurrency 10 and dies at 100. Cause: capacity planned on weights only. Fix: plan on KV cache, cap max concurrency explicitly, enable paged attention.
- **Tensor parallelism across the scale-out fabric.** Someone sets TP=16 on 2×8-GPU nodes. Throughput collapses. Fix: TP within the NVLink domain, PP or DP across it.
- **Cold-start latency.** Autoscaling that takes 12 minutes to load a checkpoint isn't autoscaling. Fix: pre-warmed replicas, fast local storage, tiered scaling.
- **Long prefills stalling decode.** Users see stuttering output when someone submits a 100K-token document. Fix: chunked prefill, then disaggregation.
- **Fragmentation.** Memory is "free" but allocations fail. Fix: paged KV cache.
- **NCCL hangs.** A distributed job silently stalls. Usually topology, MTU, or driver mismatch. Fix: pin driver/CUDA/NCCL/engine versions together and test the collective path before the model.
- **Version drift.** The disaggregation and scheduling APIs are still moving fast. Pin everything; upgrade deliberately.
- **Planning around GPUs you can't get.** Allocation lead times are 6–12 months for current-generation hardware. Design for the hardware you can actually procure.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **HBM** | High Bandwidth Memory — the GPU's on-package memory. Capacity and bandwidth both matter. |
| **NVLink / NVSwitch** | NVIDIA's scale-up interconnect. GPU-to-GPU within a node or rack. |
| **InfiniBand / RoCE** | Scale-out fabrics. Node-to-node, RDMA-based. |
| **RDMA** | Remote Direct Memory Access. Network transfers that bypass the CPU. |
| **NCCL** | NVIDIA's collective communications library. All-reduce, all-gather, etc. |
| **KV cache** | Cached attention keys/values for prior tokens. The dominant variable memory cost. |
| **Prefill / decode** | Prompt processing (compute-bound) vs token generation (memory-bound). |
| **TP / PP / DP / EP** | Tensor / pipeline / data / expert parallelism. |
| **TTFT / ITL** | Time to first token / inter-token latency. |
| **MFU** | Model FLOPs Utilisation — achieved vs theoretical peak. |
| **Continuous batching** | Admitting and evicting requests every decode step. |
| **PagedAttention** | Block-based KV cache allocation, borrowed from OS virtual memory. |
| **Quantisation** | Running weights/activations at lower precision (FP8, FP4, INT4). |
| **MoE** | Mixture of Experts. Only some parameters activate per token. |
| **Disaggregation** | Separating prefill and decode onto different GPU pools. |
| **MIG** | Multi-Instance GPU — partitioning one GPU into isolated slices. |

---

## 14. Where to go next

**Do these in order:**
1. Run vLLM on a single GPU with a 7B model. Watch `nvidia-smi`. Vary batch size and observe the throughput/latency curve yourself.
2. Do the memory arithmetic by hand for a model you care about, then check it against what the engine actually allocates.
3. Benchmark with a realistic request distribution, not a uniform one. Your prompt-length distribution determines everything.
4. Only then reach for multi-node, and only then for disaggregation.

**Reading:**
- vLLM docs and the PagedAttention paper (Kwon et al., 2023)
- Sarathi-Serve on chunked prefill (Agrawal et al., 2024)
- DistServe and Splitwise on disaggregation (2024)
- NVIDIA Dynamo and llm-d documentation
- NVIDIA's own scale-up networking material for the interconnect view

**Sources consulted for the 2026 hardware and stack picture:**
- https://ai-infrastructure.net/nvidia-gpu-roadmap/
- https://vrlatech.com/nvidia-gpu-roadmap-2026-2030/
- https://developer.nvidia.com/blog/nvidia-nvlink-the-scale-up-network-for-ai-factories/
- https://www.naddod.com/blog/scale-up-vs-scale-out-in-ai-infrastructure
- https://llm-d.ai/docs/architecture/advanced/disaggregation
- https://blog.prompt20.com/posts/disaggregated-inference/
- https://www.arccompute.io/resources/arc-blog/beyond-blackwell-preparing-enterprise-data-centers-for-the-nvidia-rubin-architecture-and-the-hbm-crunch

---

## The five things to remember

1. **Decode is memory-bandwidth-bound, prefill is compute-bound.** Nearly every design decision follows from this.
2. **KV cache, not weights, determines how many users you can serve.**
3. **Tensor parallelism inside the NVLink domain, pipeline and data parallelism across the network.** The interconnect hierarchy dictates the partitioning.
4. **Use an existing inference engine.** Continuous batching, paged attention, and prefix caching are worth an order of magnitude and you will not reimplement them well.
5. **Pick your latency SLO before you tune throughput.** You cannot optimise both.
