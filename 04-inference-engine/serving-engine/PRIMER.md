# Inside an inference engine: the step loop, scheduling, batching, caching, speculation and quantization

*A primer for layer 04. Snapshot: September 2026. Product facts are dated and marked (verify); every formula has a
worked number and names the function in [`mini-engine-core/`](mini-engine-core/) (package `minengine`) that computes
it. Numbers from `minengine.perf` are **simulated** — a roofline model, not a measurement.*

This primer explains what one inference engine instance does between "a request arrived" and "tokens are streaming
out": how it turns many independent requests into one forward pass per step, how it allocates, shares and reclaims
the KV cache, how it picks each token, how it guesses several tokens at once without changing the output, what
quantization buys, how it spans several GPUs, and how to measure all of it. The data structures and kernels
underneath are covered next door — [`kv-cache`](../kv-cache/kv-cache-primer.md),
[`paged-attention`](../paged-attention/paged-attention-primer.md),
[`flash-attention`](../flash-attention/flash-attention-primer.md) — and the sizing arithmetic (weights, KV bytes,
TTFT and TPOT budgets) is in [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md). This
primer builds on them rather than repeating them. Every concept is learnable at tier T0 with
[`mini-engine-core/`](mini-engine-core/); [`vllm-serving-lab/`](vllm-serving-lab/) measures the same things on a
real vLLM server (T1) and deploys it on Cloud Run or GKE (T3).

---

## The one-minute version

An engine is **a loop around one forward pass**. Each step the **scheduler** hands out a token budget
(`max_num_batched_tokens`): running requests first — one token each if decoding, a chunk if still prefilling — then
waiting requests while budget, sequence slots and **KV blocks** last. All scheduled tokens are flattened into one
batch, so the weights are read **once** for everyone; attention reads each request's history through its **block
table**. After the pass the engine samples a token for every request whose prompt is done; finished requests free
their blocks at once and a waiting one joins the next step — **continuous batching**. A step is **memory-bound** up
to a knee of ~300 tokens (H100, bf16) and compute-bound after, so decode tokens ride almost free while a long prompt
in one step stalls everyone: **chunked prefill** caps the step. KV memory caps concurrency; when it runs out the
newest request is **preempted** and recomputed. **Prefix caching** names each full block by a hash chained through
its parent, so shared prefixes (system prompts, agent histories) are computed once. The **sampler** reshapes one
distribution per step; **structured output** masks it with a grammar. **Speculative decoding** verifies k guessed
tokens in one pass and, by rejection sampling, keeps the target's distribution exactly. **Quantization** cuts bytes
(decode) or FLOPs (FP8 prefill) or KV (concurrency). **Tensor parallelism** adds two all-reduces per layer. Measure
with open-loop load, report TTFT/ITL percentiles and **goodput** against an SLO — and pick knobs only after picking
the SLO.

---

## 1. Anatomy of an engine

A modern engine (vLLM's V1 architecture, verify; SGLang and TensorRT-LLM are shaped alike) splits into processes so
that nothing on the CPU stalls the GPU:

```
 HTTP ───────► API server ──────────────► engine core ──────────────────────► GPU worker(s)
   (OpenAI       tokenize, validate,         scheduler ◄── KV cache manager      model runner: build input
   compatible)   apply chat template         (who runs, how many tokens,          tensors, attention metadata,
      ▲                                        which blocks)                      CUDA Graphs; forward; sample
      │                                          │  ▲                                         │
      └────────── detokenize, stop strings, ◄────┘  └──── sampled token ids ◄────────────────┘
                  stream deltas (output processor)
```

**One step** is: `schedule()` → build the flat batch → one forward pass → sample → `update()`. In
`minengine.engine.Engine.step()` that is thirty lines:

1. **Schedule.** The scheduler decides, per request, how many new tokens run this step, makes sure the KV cache
   manager has blocks for them, and publishes the blocks those tokens will fill to the prefix cache
   (`Scheduler.schedule()`, §2 and §5).
2. **Flatten.** Every scheduled token of every request goes into one array — no padding — with its position, its
   **slot** (where its K and V are written: `block_table[pos // B] × B + pos % B`) and, per request, the block table
   and context length (`Engine.build_batch()`, the vLLM "attention metadata").
3. **Forward.** Projections and MLPs run on the whole flat batch (one weight read for all); attention runs per
   request over its own K/V, gathered through its block table; logits are computed only for each request's last
   token (`TinyLM.forward()`). The paged path computes exactly what textbook attention computes — the core's tests
   check it against `forward_dense()` to 1e-10.
4. **Sample.** Only requests whose computed tokens reach the end of their sequence get a token; a prefill chunk that
   stops short samples nothing (§6).
5. **Update.** Advance `num_computed_tokens`, append tokens, finish requests that hit EOS, a stop token or
   `max_tokens`, and free their blocks.

**What a step costs.** Llama-3.1-8B in bf16 on an H100 streams 15.0 GB of weights per step (the untied input
embedding is a gather, not a stream): 4.5 ms at 3.35 TB/s — the floor for *any* step, however few tokens it carries
(why HBM bandwidth, not FLOPs, sets it: the [GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md),
§3). Sixty-four requests decoding at 2,048 tokens of context add 17.2 GB of KV reads (5.1 ms), for a 9.6 ms step.
A 2,048-token prefill does 29.7 TFLOP: 30 ms, compute-bound (`perf.step_cost()` with ideal efficiencies; these
match layer 01's `roofline.llm`, [roofline-and-fabric §3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)).
The rest of this primer is about packing work into those steps well.

**Request states.** `WAITING` → `RUNNING` → `FINISHED_{STOPPED, LENGTH_CAPPED, ABORTED}`, with `RUNNING → PREEMPTED
→ WAITING` when memory runs out (§4) — vLLM's `RequestStatus` names (it has a few more, e.g. for requests waiting
on a grammar or a remote KV transfer); the core's `scheduler.Status` mirrors these six. Stop strings are
the detokenizer's job: they need text, and the engine core only sees token ids. Aborts (client disconnects) must
free blocks — the core's tests check that nothing leaks.

**What sits outside the loop but matters.** CUDA Graphs replay a captured decode step to remove kernel-launch
overhead (why engines capture graphs for a set of batch sizes — [cuda-and-nccl
§4](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md)); asynchronous scheduling overlaps the CPU work of step n+1
with the GPU work of step n (on unless disabled in recent vLLM, verify); memory profiling at start-up decides how
many KV blocks exist (§4).

## 2. Continuous batching

**Static batching** runs a batch until its longest request finishes; everyone else's slot idles. **Continuous
(iteration-level) batching** — Orca, OSDI 2022 — re-forms the batch every step: a finished request leaves
immediately and a waiting one takes its place. With output lengths [2, 9, 3, 4] and two slots, static batching takes
13 steps at 69% slot utilisation, continuous batching 9 steps at 100%; for 64 requests with lengths uniform in
10–400 and 8 slots, 2,842 vs 1,812 steps — 1.57× the throughput from scheduling alone (notebook 01, `static_steps` /
`continuous_steps`).

**Unified scheduling.** vLLM V1 dropped the prefill/decode distinction. A request is two numbers: `num_tokens`
(prompt plus everything generated) and `num_computed_tokens` (tokens whose K/V are cached). Each step:

```
budget = max_num_batched_tokens
for req in running (admission order):                     # decodes and unfinished prefills first
    n = min(req.num_tokens - req.num_computed_tokens, budget)      # 1 for a decode, a chunk for a prefill
    allocate KV blocks for n tokens, preempting the newest running request if none are free (§4)
    budget -= n
if nobody was preempted this step:
    for req in waiting (FCFS):                             # while budget, max_num_seqs and blocks allow
        num_cached = prefix-cache hit (§5); n = min(req.num_tokens - num_cached, budget)
        admit if its blocks fit, publishing the full blocks it will compute (§5); budget -= n
```

That is `minengine.scheduler.Scheduler.schedule()`. Worked: budget 16, `r0` has an 11-token prompt, `r1` a 20-token
one → step 1 runs `r0` 11 + `r1` 5 (a chunk; nothing sampled), step 2 runs `r0` 1 (decode) + `r1` 15 (the rest,
which samples its first token). A request's prompt tokens and its generated tokens are handled by the same rule; a
preempted request simply has `num_computed_tokens` reset to 0 and its outputs count as prompt.

**The knobs.** `max_num_batched_tokens` (tokens per step, §3) and `max_num_seqs` (requests per step: sizes
per-request buffers and CUDA-graph batch sizes). vLLM's API-server defaults as of Sep 2026 (verify): 2,048 tokens
and 256 sequences on GPUs under 70 GB and on A100s, 8,192 and 1,024 on H100/H200-class, 16,384 and 1,024 at 160 GB
and above. Policy `fcfs` (default) or `priority` (lower value first; the victim of a preemption is then the least
important, newest request).

**What really caps concurrency is KV memory.** A request with a P-token prompt that generates O tokens holds at most
`⌈(P + O − 1) / B⌉` blocks — the last sampled token is never fed back, so it never gets a slot
(`KVCacheManager.blocks_needed(P + O − 1)`, notebook 01). Llama-3.1-8B on a 24 GB L4 at 90% utilisation leaves 2,164
blocks of 16 tokens (`perf.kv_cache_blocks()`); chat requests of 1,000 + 200 tokens need 75 blocks each → **28
concurrent requests** (31 at vLLM v0.30.0's defaults, which §4 compares). The same arithmetic, plus Little's law
(concurrency = arrival rate × time in system), sizes a fleet —
[capacity planning, formulas 2 and 5](../../00-foundations/gpu-capacity-planning/PRIMER.md).

## 3. Chunked prefill and prefill/decode interference

**The step-time curve.** A step pays `max(bytes / bandwidth, FLOPs / peak) + overhead` (`perf.step_cost()`; the full
per-kernel model is [roofline-and-fabric §2–3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)). Bytes
are nearly constant (the weights); FLOPs grow as 2 × params per token. The **knee** where they cross, in tokens, is
the ridge point scaled by bytes per parameter:

```
knee ≈ peak × bytes_per_param / (2 × bandwidth)          perf.knee_tokens()
H100 bf16: 989e12 × 2 / (2 × 3.35e12) = 295 tokens       L4 bf16: 121e12 × 2 / (2 × 300e9) = 403 tokens
```

(For Llama-3.1-8B the exact flip is between 310 and 320 tokens: its LM head is streamed every step but multiplied
only for the rows being sampled.) With 80% of datasheet bandwidth, 60% of datasheet FLOP/s and 2 ms of per-step
overhead — assumptions to replace with measurements — an H100 step for Llama-3.1-8B costs (SIMULATED):

| Tokens in the step | 1 | 64 | 256 | 512 | 2,048 | 8,192 |
|---|---|---|---|---|---|---|
| Step time | 7.6 ms | 7.6 ms | 8.1 ms | 14.2 ms | 52.0 ms | 224 ms |
| Per token | 7.6 ms | 0.12 ms | 0.032 ms | 0.028 ms | 0.025 ms | 0.027 ms |

**Interference.** Decode tokens are nearly free up to the knee; a prompt costs in proportion to its length. If an
8,000-token prompt runs in one step, every request decoding alongside it waits for that step: on an L4 serving
Qwen2.5-1.5B to 16 users at 1K context, their 16.7 ms token gap becomes **367 ms** (SIMULATED, notebook 02). That
stutter is prefill/decode interference.

**Chunked prefill** (Sarathi-Serve's "stall-free batching", OSDI 2024) caps every step at the budget and schedules
decodes first; the prompt gets what is left, over as many steps as it takes. With a 512-token budget the same prompt
runs in 17 chunks of 496 and the worst gap is **29.6 ms** — the prompt's own TTFT rises a little, everyone else
keeps streaming. vLLM V1 has chunked prefill on by default; turning it off requires `max_num_batched_tokens ≥
max_model_len`, because a whole prompt must then fit in one step (the core enforces the same rule in
`SchedulerConfig`).

**The budget trades TTFT against ITL — and, under saturation, sets capacity.** 200 requests, prompts 500–6,000
tokens, outputs 100–400, H100 + Llama-3.1-8B (SIMULATED, `perf.simulate()` on `perf.Workload(..., seed=1)`, notebook 02
worked example 3; another seed draws other prompts and moves every figure). The
latency columns and goodput (requests per second meeting a 1 s TTFT and a 50 ms TPOT SLO, §11) are for an
open-loop Poisson stream at 6/s; the last column sends all 200 at once, which saturates the engine and so measures
its capacity:

| Setting | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | Goodput at 6/s | Capacity (saturated) |
|---|---|---|---|---|---|---|
| no chunking | 128 ms | 572 ms | 13.2 ms | 150 ms | 5.18 req/s | 1,948 tok/s |
| budget 8,192 | 137 ms | 655 ms | 13.2 ms | 210 ms | 5.17 req/s | 1,907 tok/s |
| budget 2,048 | 151 ms | 625 ms | 12.9 ms | 56.7 ms | 5.20 req/s | 2,157 tok/s |
| budget 512 | 169 ms | 760 ms | 12.3 ms | 16.3 ms | 5.23 req/s | 2,261 tok/s |
| budget 256 | 331 ms | 1,444 ms | 10.1 ms | 11.5 ms | 4.66 req/s | 1,706 tok/s |

At 6/s every row delivers the same ~1,320 tok/s — not because the knob is free, but because an open-loop load
below capacity finishes exactly what it is offered; such a run cannot show a throughput effect. Saturated, the
budget matters. At 512 each step mixes a prompt chunk (compute-bound) with the running decodes' KV reads
(memory-bound), so tensor cores and HBM are busy at once — Sarathi-Serve's case for hybrid batches; whole prompts
or 8,192-token steps alternate compute-heavy prefill steps with memory-bound decode steps and leave one resource
idle in each. Part of the gain is memory, not overlap: saturated, the three larger settings admit prompts faster
than decodes finish, fill the KV pool (peak 100%) and preempt 3–8 requests whose work is then recomputed, while at
512 the pool peaks at 38% and nothing is preempted (SIMULATED; `SimResult.preemptions`, `peak_kv_usage`). At 256
a step sits just above the knee (~221 tokens with these efficiencies): its time is mostly
the weight read, the decodes' KV reads and the 2 ms overhead, and after the decodes take their share the prompt
gets small, poorly amortised chunks — so TTFT doubles at 6/s, goodput drops ~10%, and capacity drops by a quarter.
**Choosing it:** take the largest budget whose worst step — all running decodes plus a full prefill chunk — meets
the ITL SLO, then check capacity under a saturating load. With 64 requests decoding at 2,000 tokens of context and
a 25 ms p99 ITL target: 512 tokens gives a 14.4 ms step, 1,024 gives 26.7 ms → choose 512 (notebook 02, exercise
2.4). vLLM adds finer knobs — `long_prefill_token_threshold` (cap the chunk of a long prompt so several prompts
progress together; 0 = off by default) with an `_adaptive` variant, and `max_num_active_seqs` (cap RUNNING below
`max_num_seqs`); the last two are on `main` after 0.30.0, not in the 0.30.0 wheel (verify).

**When chunking is not enough.** Every chunk still shares its step with decodes, so a prefill-heavy mix (RAG, agents
re-reading long contexts) makes ITL and TTFT fight for the same GPUs. Running prefill and decode on separate pools
and shipping the KV between them — disaggregation — is layer 05's topic
([`05-orchestrator`](../../05-orchestrator/)); [gpu-deployment
§8](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) has the architecture.

## 4. KV cache management revisited

Paging itself — fixed-size blocks, per-request block tables, refcounts, copy-on-write — is in the [paged-attention
primer](../paged-attention/paged-attention-primer.md) and
[`paged_attention_minimal.py`](../paged-attention/paged_attention_minimal.py); why the cache is the binding
constraint is in the [kv-cache primer](../kv-cache/kv-cache-primer.md). This section is the engine's side: how many
blocks exist, who gets them, and what happens when they run out.

**How many blocks.** At start-up the engine loads the weights, runs a profiling forward pass at the maximum batch to
measure activation memory, and gives everything left under `gpu_memory_utilization` to the KV cache:

```
blocks = (HBM × gpu_memory_utilization − weights − activations/workspace) / (block_size × KV bytes per token)
KV bytes per token = 2 × layers × kv_heads × head_dim × bytes      (ModelConfig.kv_bytes_per_token(), perf.LLM)
L4 (24 GB) × 0.9 − 16.06 GB (Llama-3.1-8B bf16) − 1 GB  =  4.54 GB  /  (16 × 128 KiB = 2 MiB)  =  2,164 blocks
H100 (80 GB), same model                                                                     = 26,197 blocks
```

Those are the core's round inputs (`perf.kv_cache_blocks()`), and every simulated number in this primer uses them.
vLLM v0.30.0 starts from different ones; the lab's `sizing.size()` models them from a real `config.json`:

| Input | The core (`perf.kv_cache_blocks()`) | vLLM v0.30.0 defaults (the lab's `sizing.size()`) |
|---|---|---|
| total memory | 24 GB, the datasheet figure | 22.49 GiB = 24.15 GB, the L4's total as `nvidia-smi` reports it; vLLM multiplies the total CUDA reports (`torch.cuda.mem_get_info()[1]`), which may sit slightly below it, and refuses to start if CUDA's *free* figure, lower by the CUDA context (a few hundred MiB) and any other process, is below that share (verify on your card) |
| `gpu_memory_utilization` | 0.9, vLLM's default for a long time | 0.92 (`vllm/config/cache.py`) |
| overhead | a flat 1 GB | profiled at start-up: the activation peak of a 2,048-token pass, CUDA graphs, non-torch buffers — ~1.2 GB by the lab's estimate |
| KV budget → blocks | 4.54 GB → 2,164 | 4.96 GB → 2,363 |
| 2,000-token sessions | 17.3 | 18.9 |

Same formulas, different inputs: the higher utilisation adds ~229 blocks, the larger reported total ~65, and the
profiled overhead takes back ~95 — 199 blocks (about 8%), one or two 2K-token sessions of an 8B model on an L4.
Neither column is a measurement: the `Available KV cache memory` line vLLM prints at start-up is, and the lab's
notebook 01 calibrates its estimate against that line (exercise 1.5). `block_size` is 16 in both. vLLM refuses to
start if one `max_model_len` sequence cannot fit — otherwise that request could never finish (the core raises the
same error in `Scheduler.__init__`). Model choice moves this number more than any knob: per token, Qwen2.5-1.5B needs 28 KiB of KV
(2 KV heads), Qwen3-0.6B 112 KiB (8 KV heads of 128), Llama-3.1-8B 128 KiB — GQA width, not parameter count.

**Admission.** A new request is admitted only if its blocks are free *now*. Two refinements keep the pool from
thrashing: an optional **watermark** (a fraction of blocks kept free when admitting new or preempted requests; vLLM
default 0, verify) and a **whole-prompt check** (admit only if the full prompt, not just its first chunk, fits:
vLLM's scheduler admits a waiting request through
`allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)` in `vllm/v1/core/sched/scheduler.py`;
default on, verify). Without the latter, chunked prefill admits prompts it cannot finish and preempts them a few
steps later. The core's version is `KVCacheManager.allocate_slots(..., admit_whole_prompt=True)`, switched by
`SchedulerConfig.admit_whole_prompt`: in notebook 02's exercise 2.6, 22 preemptions at 2,000 blocks with the check,
87 without (SIMULATED); the core's tests pin both effects on a tight pool.

**Growth and preemption.** Decodes take a new block every 16 tokens. When a running request needs one and the free
queue is empty, the scheduler **preempts** the lowest-priority running request — under FCFS the most recently
admitted — frees its blocks, resets it to zero computed tokens and puts it at the *front* of the waiting queue; no
new request is admitted in that step. Its generated tokens are kept, so when it is readmitted its whole sequence is
prefilled again ("recompute") and generation continues where it stopped. Preemption is invisible in the outputs (the
core's tests compare against the no-preemption reference) and very visible in latency: in notebook 02, a workload
that needs ~4,000 blocks on an H100 runs with 0 preemptions and a 176 ms p99 TTFT at 4,000 blocks, 6 preemptions and
861 ms at 3,000, 22 preemptions and 6.5 s at 2,000 (SIMULATED). **Short KV memory shows up as latency long before it
shows up as errors** — alert on `vllm:num_preemptions`.

**Recompute or swap?** Swapping copies the victim's blocks to host memory and back: for 2,000 tokens of Llama-3.1-8B
that is 262 MB each way (2,000 × `perf.LLM.kv_bytes_per_token`) — about 5 ms per direction at ~50 GB/s effective
over PCIe Gen5 x16 (verify), twice that on Gen4. Recomputing is a 2,000-token prefill: ~51 ms on an H100 (SIMULATED)
— but it needs no host memory, no transfer bookkeeping, and with prefix caching the victim's own blocks are often
still cached when it returns (notebook 02: a preempted request restarted at token 20, not 0). vLLM V1 preempts by
recompute only; swap was a V0 mode, and V1's CPU offloading (`kv_offloading_size`) is a cache tier rather than a
preemption mode (verify). Either way, preemption is a symptom: the fixes are more KV memory (§8's FP8 KV, a smaller
model, higher utilisation), fewer concurrent sequences, or more replicas (05).

**Invariants worth testing** in any block manager — the core's `KVCacheManager.check()` runs after every step in its
tests: a block's refcount equals the number of block tables that hold it; a block is in the free queue iff its
refcount is zero; the prefix-cache map only points at blocks that carry that name; no table holds a block twice;
after every request finishes or aborts, every block is free.

## 5. Prefix caching

Two requests whose prompts start with the same tokens compute identical K/V for that prefix — because a token's K/V
depend only on the tokens before it. Prefix caching computes it once.

**Naming blocks.** Only **full** blocks are cached. A full block's name is

```
name(block i) = H( name(block i−1), tokens in block i, extra keys )        kv.hash_block(), kv.block_hashes()
```

with `name(block −1)` a fixed root. Because the parent's name is inside, a block's name commits to the *entire*
prefix: the same 16 tokens after a different history get a different name. That is not an optimisation, it is
correctness — K/V depend on everything earlier. Notebook 03 replaces the hash with one that ignores the parent: the
engine silently serves K/V computed under another prefix (the logprobs drift by ~1e-2 instead of 1e-15; with a real
model, wrong text). The **extra keys** carry whatever else changes K/V or must isolate caches: the LoRA adapter
(§10), multimodal input hashes, and a per-tenant `cache_salt` added to the first block only (the chain carries it
onward). vLLM hashes with SHA-256 by default (`prefix_caching_hash_algo`; xxhash is optional and non-cryptographic,
verify): a collision would serve another prefix's K/V, and a shared cache is a timing side channel between tenants —
hence collision resistance and salts.

**Lookup, adopt, publish.** A new request walks its prompt's chain until the first miss and **adopts** every hit:
refcount + 1, no compute, no new memory (`KVCacheManager.lookup()`, `allocate_slots(..., hits)`). The hit is capped
at `(len(prompt) − 1) // B` blocks: the last token must be computed to get logits, so two identical 20-token prompts
with B = 4 share 16 tokens, not 20 (vLLM: `max_cache_hit_length = num_tokens − 1`). A block is **published** as
soon as the scheduler schedules the tokens that fill it — vLLM does it inside `allocate_slots`, the core in
`Scheduler.schedule()` (`cache_blocks()`) — not after the step. That is safe because every layer writes the whole
step's K/V before any request attends at that layer, and it matters for agents: requests admitted in the same step
share a prefix that one of them is computing right then, so a burst of N parallel calls behind one system prompt
prefills it once and holds one copy (notebook 03, worked example 3: three requests, one step, `[0, 176, 176]`
tokens from cache). The core publishes a running request's blocks only once no request can be preempted in that
step any more, so a request taken out of the batch never names blocks it will not compute.

**No copy-on-write.** Only full, immutable blocks are shared, the hit stops short of the last prompt token, and a
request always writes its new tokens into blocks of its own — so block-hash prefix caching never copies a shared
block. Copy-on-write (the [paged-attention primer](../paged-attention/paged-attention-primer.md)) matters when
sequences fork mid-block: parallel sampling and beam search.

**Freed but still hittable.** A finished request's blocks go to the back of an LRU **free queue** with their names
intact. They count as free — `vllm:kv_cache_usage_perc` does not include them — and they keep producing hits until
the allocator pops them from the front and evicts the name (lazy eviction). A request frees its blocks **tail
first**, so the private end of a prompt is evicted before the shared head: in notebook 03, after an unrelated
request evicted 13 blocks, the first 32 tokens of the system prompt were still cached. Prefix caching therefore
costs no memory; it uses memory nobody else needs yet.

**Hit accounting** is in tokens: `vllm:prefix_cache_queries` counts prompt tokens looked up by new requests,
`vllm:prefix_cache_hits` those found; a preempted request's re-lookup is counted apart (`preempted_queries` /
`preempted_hits`, not exported), so a victim re-hitting its own blocks does not inflate the hit rate (`CacheStats`).
Three requests sharing a 183-token system prompt: the second and third hit 176 tokens (its 11 full blocks) of
their 205–210 prompt tokens. The payoff is TTFT and prefill compute: 60 requests with 2,000–2,200-token prompts
that share their first 1,800 tokens, H100 + Llama-3.1-8B, arriving at 6/s — hit rate 84%, **TTFT p50 13 ms vs
74 ms** without caching, and the shared blocks cut peak KV use from 7% to 1% of the pool; all 60 at once — TTFT
p50 0.32 s vs 1.6 s (SIMULATED, notebook 03 worked example 6: `perf.Workload(n_requests=60, rate=6,
prompt_len=(2000, 2200), output_len=(40, 80), shared_prefix=1800)`).

**The radix-tree alternative.** SGLang's RadixAttention keeps cached prefixes in a radix tree at token granularity
with LRU eviction of leaves. It reuses the partial last block too — always less than one block per request more than
block hashing (notebook 03, exercise 3.5) — so the difference is not the hit rate but what the structure makes
cheap: a tree answers "which cached prefix does this request extend?", which SGLang uses to schedule for cache
locality; a flat dictionary of block names is trivial to offload, share across processes and publish as events —
which is what KV-aware routers consume (05).

**Designing agent prompts for hits.** The cache matches prefixes exactly, token for token. So:

- stable content first — system prompt, tool schemas in a fixed order, few-shot examples, long documents;
- the conversation **append-only** — never re-render, summarise or reorder earlier turns if you want them cached;
- anything volatile — timestamps, request ids, per-call instructions — at the end, and kept *in* the history once
  sent rather than rewritten.

Notebook 03's four-turn support session: a clock at the top of the system prompt gives 0% hits on every turn; the
same content append-only gives 81–86% from the second turn on (and more as the history grows). Two traps: the same
text re-tokenized can yield different tokens at a boundary (so keep token ids or render deterministically), and
across replicas the next turn must reach the replica that holds the prefix — prefix-affinity routing, 05. The
agent-side view of context layout and caching is the platform lab's [context-engineering
notebook](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/04_context_engineering_and_caching.ipynb)
in [`07-application-agent-framework`](../../07-application-agent-framework/).

## 6. Sampling and structured output

The forward pass ends in logits, one per vocabulary entry. The sampler turns them into a token, in this order (vLLM
V1's `Sampler`, verify; `sampler.process_logits()`):

```
allowed-token / grammar mask  →  penalties  →  greedy if T < 1e-5  →  ÷ temperature  →  min-p  →  top-k  →  top-p  →  draw
```

| Setting | Rule | Worked (probs 0.5, 0.3, 0.15, 0.05) |
|---|---|---|
| temperature T | softmax(z / T): T < 1 sharpens, T > 1 flattens; the ranking never changes | T = 2 keeps the order, flattens the gaps |
| top-k | keep the k largest logits | k = 2 → {0.5, 0.3}, renormalised to 0.625 / 0.375 |
| top-p (nucleus) | keep the smallest top set whose mass reaches p | p = 0.8 → {0, 1}; p = 0.81 → {0, 1, 2} (`top_p_filter()`) |
| min-p | keep tokens with prob ≥ min_p × max prob | min_p = 0.2 → threshold 0.1 → {0, 1, 2} (`min_p_filter()`) |
| repetition penalty r | tokens seen in prompt *or* output: logit ÷ r if positive, × r if negative | r = 2: logits 2, −2 → 1, −4 |
| presence / frequency | output tokens only (OpenAI definitions): − presence × [seen] − frequency × count | (`apply_penalties()`) |

top-k keeps the same number of tokens whatever the model's confidence; top-p and min-p adapt — after "When memory
runs ou" the tiny model's top-p 0.9 set is 3 tokens, after "The engine " it is 16 (notebook 04). Greedy decoding of
a weak model loops (`tofofof…`); penalties break loops by pushing seen tokens down.

**Seeds and reproducibility.** Each request gets its own random generator (from `seed`, or derived), so a seeded
request draws the same tokens whatever else shares its steps — the property you need to replay an incident. On a
GPU, logits themselves can differ in the last bits with batch size (kernel choice, reduction order); vLLM has a
batch-invariant mode for bitwise reproducibility at some speed cost (verify).

**Logprobs** come from the **raw** logits by default in vLLM V1 (`logprobs_mode="raw_logprobs"`, verify): what the
model believed, before temperature, penalties or masks. They are the cheapest monitoring signal an engine emits.

**Stop conditions:** EOS (unless `ignore_eos`), `stop_token_ids`, `max_tokens`, `max_model_len` in the engine core;
stop *strings* in the detokenizer, which then aborts the request (the stop string is excluded from the output by
default).

**Structured output** is sampling with a grammar-shaped mask. A JSON schema, regex or context-free grammar is compiled
to an automaton; each step, the tokens that would leave the grammar get −∞ before sampling, and the automaton advances
on the token drawn (`ChoiceFSM`: `start()`, `allowed(state)`, `advance(state, token)`). With a byte vocabulary that is
trivial; with a 100k-token BPE vocabulary each token spans several characters, so the hard part is computing, per
grammar state, which of 100k tokens keep the text legal — a token is legal only if every one of its characters is,
starting from that state — fast enough to overlap with the GPU's forward pass. Notebook 04 (worked example 6, exercise
4.6) compiles the schema `{"n": <non-negative integer>}` into a 10-state character automaton and precomputes, for each
state, the next state after each of 36 tokens (single characters and multi-character merges): the token `{"n": ` jumps
six states at once, `07` is legal after `{"n": 1` but not right after `{"n": `, the text `{"n": 0}` comes out as 19
different token sequences in 300 samples (the re-tokenization trap of §5), and a `max_tokens` cap leaves a legal
prefix that does not parse — check `finish_reason`. That precomputation is what the backends optimise (vLLM:
`xgrammar`, `guidance`/llguidance, `outlines`, `lm-format-enforcer`, `auto`, verify); SGLang adds "jump-forward"
decoding through stretches the grammar fully determines. The guarantee is **syntax, not sense**: in notebook 04 the
tiny model produces a valid `{"answer": "yes"}` or `{"answer": "no"}` every time, while its own log-probability of
that text is about −123 nats — the grammar forced every character, and it would force a confident-looking answer from
a model that knows nothing. Validate values downstream; watch the logprobs of constrained fields.

## 7. Speculative decoding

Decode is memory-bound (§3): scoring k + 1 positions costs about as much as scoring one. So let a cheap **proposer**
draft k tokens and have the target check them all in one pass.

**The rule** (Leviathan, Kalman & Matias 2023; Chen et al. 2023). For each draft token x ~ q in order:

```
accept x with probability min(1, p(x) / q(x))                              spec.verify()
on the first rejection: emit y ~ norm(max(0, p − q)) and stop              ("recovered" token)
if all k are accepted: emit one more token from the target's next p        ("bonus" token)
```

Why it is exact: the chance of emitting token y is `q(y) min(1, p(y)/q(y)) = min(p(y), q(y))` via acceptance, plus
`P(reject) × residual(y)`. P(reject) = 1 − Σ min(p, q), and the residual's normaliser Σ max(0, p − q) equals that
same 1 − Σ min(p, q), so the second term is `max(0, p(y) − q(y))`. The sum is `min(p, q) + max(0, p − q) = p(y)`.
**Speculation changes speed, never the distribution** — the core's chi-square test over all 64 three-token outcomes
confirms it, and rejects a plausible bug (resampling from p instead of the residual). With temperature 0 both
distributions are one-hot and the rule reduces to "accept iff the draft's argmax is the target's": greedy
speculation reproduces greedy decoding token for token.

**How much it yields.** The per-token acceptance rate is `α = Σ min(p, q) = 1 − TV(p, q)`
(`spec.acceptance_rate()`): for p = (0.5, 0.3, 0.15, 0.05) and q = (0.2, 0.2, 0.2, 0.4), α = 0.6. If each draft
token survives with probability α, one target pass emits on average

```
E[tokens per pass] = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)        spec.expected_tokens()
α = 0.8, k = 4:  (1 − 0.8⁵) / 0.2 = 3.36
```

If one draft step costs c target steps, a round costs k c + 1, so `speedup = E / (k c + 1)` (`spec.speedup()`). For
α = 0.8, c = 0.1 the best depth is k = 6 at 2.47× (k = 4, 5, 6, 7 give 2.40, 2.46, 2.47, 2.45 — a shallow optimum;
`spec.best_k()`). In notebook 05 the tiny target and a 10× smaller draft agree with α ≈ 0.72 per position on
average; the measured tokens per pass match the formula at k = 1 and 2 but fall short at k = 4 (2.61 vs 2.90).
The formula assumes every position is accepted independently with one α; real acceptance varies by position (p10
0.53, p90 0.87 there) and is correlated, so it over-predicts deep speculation — measure acceptance per position
(vLLM: `vllm:spec_decode_num_accepted_tokens_per_pos`) before choosing k.

**What vLLM measures.** vLLM's draft models draft **greedily** by default (`draft_sample_method="greedy"`, verify):
x is the draft's argmax and the rejection sampler treats q as one-hot. The rule is still exact — accept x with
probability p(x), otherwise resample from p with x removed — but the acceptance rate is then p(x), not Σ min(p, q);
`"probabilistic"` samples x ~ q and uses the full q. Exactness holds for the `standard` rejection method and for
`block` (block verification, Sun et al. 2024: the k drafts are verified jointly — still distribution-preserving,
with at least as many tokens accepted in expectation); vLLM's `synthetic` method accepts with a calibrated
probability to benchmark speed and does not preserve the distribution (verify).

**Proposers** (vLLM's methods include `ngram`, `suffix`, `draft_model`, `eagle`, `eagle3`, `medusa`,
`mlp_speculator` and model-specific MTP, verify):

| Proposer | How it drafts | Cost c | When it wins |
|---|---|---|---|
| draft model | a small model with the **same tokenizer** runs k steps | ~0.1–0.4 (a 1B draft is ~0.19 of an 8B step here, SIMULATED) | a good same-family small model exists |
| n-gram / prompt lookup | copy what followed the last occurrence of the current n-gram in the context | ~0 | outputs quote the input: code edits, tool results, RAG answers |
| EAGLE / EAGLE-3 | a light head on the target's own hidden states drafts features, then tokens | a few % | general chat; the common production choice |
| MTP | extra prediction heads trained with the model (e.g. DeepSeek-V3) | a few % | models that ship them |

**When it stops paying** (SIMULATED, `perf.spec_speedup()`, notebook 05: Llama-3.1-8B target, Llama-3.2-1B draft,
H100, α = 0.7, k = 4). A round is one engine step: it pays the step's fixed overhead (2 ms, as everywhere in this
primer) once, k draft forwards at their roofline time plus an assumed 0.5 ms each (a CUDA-graph replay and a draft
sample), and a verify pass of (k + 1) × batch tokens with logits at every position. Once the verify pass crosses
the knee, the extra positions cost real FLOPs:

| Context | B = 1 | B = 16 | B = 64 | B = 128 | B = 256 |
|---|---|---|---|---|---|
| 200 tokens | 1.58× | 1.58× | 1.38× | 0.97× | 0.65× |
| 2,000 tokens | 1.58× | 1.55× | 1.49× | 1.45× | KV does not fit |

The robust conclusion is the shape: at short contexts speculation turns into a slow-down past ~100–250 concurrent
requests (128 here; 256 if a draft forward cost only its roofline time); at long contexts decode stays bound by KV
reads, which verification amortises, so it keeps paying — and memory caps the batch first. The batch-1 figure
rests on the overhead assumption: the draft costs c ≈ 0.19 of a target step here (its weights are 17% of the
target's, plus the 0.5 ms), giving ~1.6×; with no per-draft overhead c ≈ 0.12 and ~1.9×; if every draft forward
paid a full 2 ms step overhead, c ≈ 0.38 and only ~1.1×. Measure c on your stack before quoting a number. An
EAGLE-like head at c = 0.05 would give 2.31×. Treat speculation as a per-workload setting: enable it for
latency-sensitive, low-concurrency or copy-heavy traffic, and watch acceptance rate and ITL.

**Inside the engine** the verify pass is just a (k + 1)-token chunk: the scheduler reserves "lookahead" slots for
the draft tokens, and rejected positions are rolled back by lowering `num_computed_tokens` — their K/V slots are
simply overwritten next step. The core keeps speculation outside the loop (`spec.speculative_generate()`) to show
the rule without that bookkeeping.

## 8. Quantization

Store numbers in fewer bits with a scale that says what the bits mean: `w ≈ code × scale` (`quant.quantize()`).

**Formats.** INT8 (codes −127…127), INT4 (−7…7 symmetric here; GPTQ/AWQ checkpoints often use an asymmetric zero
point), FP8-E4M3 (1 sign, 4 exponent, 3 mantissa bits; largest finite 448; relative step 1/8, so the rounding error
is at most 6.25%; `quant.fp8_e4m3()`), FP8-E5M2 (more range, 2 mantissa bits), and on Blackwell 4-bit floats with
small shared scales (NVFP4, MXFP4, verify). **Granularity** — how many weights share a scale — is where accuracy is
won: one outlier stretches a shared scale and erases everyone else's resolution.

| On a 256×128 Gaussian weight (notebook 06) | Bits per weight | Relative error | SQNR |
|---|---|---|---|
| INT8 per tensor | 8 | 1.03% | 39.8 dB |
| INT8 per output channel | 8 | 0.71% | 43.0 dB |
| FP8-E4M3 per tensor | 8 | 2.63% | 31.6 dB |
| INT4 per channel | 4 | 12.8% | 17.9 dB |
| INT4 groups of 128 | 4.125 | 11.8% | 18.5 dB |
| INT4 groups of 32 | 4.5 | 9.8% | 20.2 dB |

Each bit is worth ~6 dB. Scales are not free: a 16-bit scale per group of 128 adds 16/128 bits, so INT4 g128 costs
**4.125 bits per weight** (`quant.bits_per_weight()`). Nor is every weight quantized: GPTQ, AWQ and FP8 checkpoints
quantize the transformer blocks' linear layers and keep the embedding table and LM head in 16-bit. For
Llama-3.1-8B that is 6.98 B weights at 4.125 bits (3.60 GB) plus an untied 128,256 × 4,096 embedding and LM head
in bf16 (2.10 GB): **5.70 GB**, not the 4.14 GB that 8.03 B × 4.125 bits suggests (`quant.weight_gb(...,
keep16_params=)`, `perf.LLM.weight_bytes`; published checkpoint sizes, verify). On well-behaved
weights INT8 per channel beats FP8 (43 vs 32 dB); FP8's strength is range, which matters for activations. With one
output channel 100× larger than the rest, per-tensor INT8 reports a 3.4% aggregate error while the 31 normal
channels are at 62% — **aggregate metrics hide per-channel damage** (notebook 06).

**Weight-only vs W8A8.** *Weight-only* formats (W4A16/W8A16: GPTQ — second-order error compensation; AWQ — scales
that protect the channels with large activations; run with Marlin-style kernels) dequantize to bf16 inside the GEMM:
they cut **bytes**, not FLOPs. *W8A8* formats quantize activations too (per token, at run time) and use the tensor
cores' native low-precision math: FP8 on Ada (sm_89), Hopper and Blackwell; INT8 more widely. Activations have
outlier channels that make per-token scales coarse; SmoothQuant divides them by `s_j = max|X_j|^α / max|W_j|^(1−α)`
and folds `s` into the weights, leaving `XW` unchanged (`quant.smoothquant_scales()`: 6.6× less output error in
notebook 06).

**What each buys** — Llama-3.1-8B on a 24 GB L4, decode at 1K context, an 1,800-token prefill, sessions of 1,800 +
200 tokens; embedding and LM head in bf16 throughout (SIMULATED, notebook 06). Sessions are given for both sets of
memory inputs in §4: whole sessions with the core's (`perf.kv_cache_blocks()`, as notebook 06 prints them), and at
vLLM v0.30.0's defaults as the lab's `sizing.size(..., typical_len=2000)` estimates them:

| Scheme | Weights | Decode, batch 1 | Decode, batch 32 | Prefill 1.8K | Sessions (core inputs) | Sessions (vLLM defaults) |
|---|---|---|---|---|---|---|
| bf16 | 16.1 GB | 65 ms | 82 ms | 360 ms | 17 | 18.9 |
| INT8 weight-only | 9.1 GB | 36 ms | 53 ms | 360 ms | 43 | 45.5 |
| INT4 weight-only g128 | 5.7 GB | 22 ms | 39 ms | 360 ms | 56 | 58.3 |
| FP8 W8A8 | 9.1 GB | 36 ms | 53 ms | 181 ms | 43 | 45.5 |
| FP8 W8A8 + FP8 KV | 9.1 GB | 36 ms | 44 ms | 181 ms | 87 | 91.1 |
| INT4 weight-only + FP8 KV | 5.7 GB | 22 ms | 30 ms | 360 ms | 113 | 116.6 |

Decode is the weight read, so INT4 weight-only is ~3× faster at batch 1 — not the 3.9× its linear layers' bytes
suggest, because the 16-bit LM head (1.05 GB, read every step), the KV read and the per-step overhead do not
shrink. Prefill is compute-bound, so only FP8 W8A8 speeds it up — and only on FP8 hardware (the
core refuses FP8 compute on a T4 or A100). The **KV cache** is quantized separately (`--kv-cache-dtype fp8`, e4m3 or
e5m2, per-tensor scales that default to 1.0 unless calibrated, verify): half the bytes, twice the sessions — the
core's FP8 KV emulation costs 2e-5 nats of KL and 1% of top-1 agreement on the tiny model. On a small GPU,
**quantization is a concurrency lever before it is a speed lever** ([capacity planning, formula
2](../../00-foundations/gpu-capacity-planning/PRIMER.md)).

**Checking accuracy.** Compare distributions, not strings: mean KL and top-1 agreement over many positions
(`quant.compare_logits()`), perplexity, and above all task-level evals on your own data. Greedy text is a brittle
metric — in notebook 06 an INT8 model with 99.8% top-1 agreement diverges from the full-precision greedy text after
13 tokens, because one flipped near-tie changes everything after it.

**Deep dive:** [quantization](../quantization/PRIMER.md) (module 04.9) — the formats down to their bit patterns,
GPTQ, AWQ and SmoothQuant, what each scheme runs as per GPU generation, KV-cache quantization, producing a checkpoint
with llm-compressor and measuring the accuracy you pay.

## 9. Parallelism inside the engine

When a model does not fit one GPU, or one GPU is too slow, the engine spans several. The menu and the rule — tensor
and expert parallelism inside the NVLink domain, pipeline and data parallelism across it — are in [gpu-deployment
§4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md); the cost of each collective is in
[cuda-and-nccl §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) and in [roofline-and-fabric
§5](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). What the engine does:

**Tensor parallelism (TP)** splits every layer. The Megatron pattern pairs a *column-parallel* matmul (QKV, or the
MLP's gate/up: each GPU computes a slice of the output features, no communication) with a *row-parallel* one (the
attention output projection, the MLP's down projection: each GPU holds a slice of the input features and produces a
partial sum). One **all-reduce** after each row-parallel matmul restores the full activation — **two all-reduces per
layer per forward pass**, each of `tokens × d_model` values (`perf.tp_allreduces()`; notebook 01's exercise 1.6
splits the tiny model's MLP across two "ranks" in numpy and checks that the sum of their partials is the dense
output). A 70B model (80 layers, d =
8,192): 160 all-reduces per step; at 64 decoding requests each is 1 MiB (latency-bound: the α term dominates, which
is why engines use custom all-reduce kernels and NVLink), at a 4,096-token prefill chunk 64 MiB (bandwidth-bound).
Heads are split across GPUs, so the KV cache is too: with 8 KV heads, TP = 8 puts one KV head on each GPU; beyond
that, KV heads are replicated. TP cuts per-GPU weights (141 GB in bf16 → 35 GB per GPU at TP = 4) and the latency of
every step.

**Pipeline parallelism (PP)** places consecutive layers on different GPUs or nodes and passes activations forward;
it communicates little (one activation per boundary per step) but needs several micro-batches in flight to keep
every stage busy, and does not reduce the latency of a single request. Use it across nodes, when one node cannot
hold the model.

**Expert parallelism (EP)** places a MoE layer's experts on different GPUs; tokens are dispatched to their experts
and the results combined — two **all-to-alls** per MoE layer, sensitive to load imbalance between experts. Large MoE
deployments combine data-parallel attention with expert-parallel MoE layers ("wide EP", layer 05). Dispatch and
combine, the slowest rank and TP vs EP for experts are worked in
[MoE primer §6](../../00-foundations/mixture-of-experts/PRIMER.md#6-running-moe-on-gpus).

**Data parallelism (DP)** is replication: independent engine replicas, each with its own KV cache. Throughput scales
linearly; what matters is routing requests to the replica that holds their prefix and is least loaded — layer 05.

**Choosing.** Use the smallest TP that fits weights plus the KV cache you need; add replicas for throughput. An 8B
model fits one 24 GB GPU (§4); a 70B model needs 141 GB in bf16 — TP = 2 on 80 GB GPUs leaves almost nothing for KV,
TP = 4 leaves ~37 GB per GPU for KV at 90% utilisation, or INT4 (39.5 GB with its 16-bit embedding and LM head,
exercise 6.3) fits one GPU. vLLM's flags:
`--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` (verify).
The split itself is learnable at T0 (exercise 1.6); to measure it you need two GPUs (tier T2): the lab's exercise
3.6 predicts TP = 2 on Kaggle's free 2×T4 and measures it there over PCIe; a rented NVLink pair shows the speed
(prices in [`COMPUTE.md`](../../COMPUTE.md)).

## 10. Multi-LoRA serving

A LoRA adapter replaces a frozen weight W with `W + B A`, rank r ≪ d. It adds `r (d_in + d_out)` parameters per
adapted matrix (`perf.lora_params()`): rank 16 on all seven linear layers of Llama-3.1-8B is 42 M parameters, 84 MB
in bf16 — about 0.5% of the base model. So one engine can hold one base model and many adapters, and serve requests
for different adapters **in the same batch**: the base matmul runs once for everyone, and a batched "shrink/expand"
kernel (Punica's SGMV, S-LoRA; vLLM's LoRA kernels, verify) applies each token's own adapter.

What the engine manages:

- **An adapter cache.** A fixed number of GPU slots (`max_loras`) and a larger CPU cache (`max_cpu_loras`); requests
  for an adapter not in a GPU slot wait for a load, and a step can mix at most `max_loras` adapters (verify flag
  names). `max_lora_rank` sizes the kernels' buffers.
- **Prefix caching per adapter.** The same prompt under two adapters produces different K/V, so the adapter id is an
  extra key in every block name (§5): no sharing across adapters, by construction.
- **Cost.** Each adapted step pays the extra shrink/expand matmuls; cold adapters pay a load; many distinct adapters
  per step dilute batching. `vllm:lora_requests_info` reports running and waiting adapters.
- **Routing.** With several replicas, sending an adapter's traffic to replicas that already have it loaded is an
  affinity problem like prefix caching — layer 05.

## 11. Measuring an engine

**The metrics** (vLLM's Prometheus names from FACTS; counters are exported with a `_total` suffix):

| Metric | Definition | vLLM name |
|---|---|---|
| TTFT | first token time − arrival: queueing + prefill | `vllm:time_to_first_token_seconds` |
| ITL | gap between consecutive tokens of one request | `vllm:inter_token_latency_seconds` |
| TPOT | (last token time − first token time) / (tokens − 1), per request | `vllm:request_time_per_output_token_seconds` |
| E2E | last token time − arrival | `vllm:e2e_request_latency_seconds` |
| queue time | arrival → first scheduled | `vllm:request_queue_time_seconds` |
| running / waiting | requests in each state | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| KV usage | fraction of blocks held by requests (cached-but-free blocks excluded) | `vllm:kv_cache_usage_perc` |
| prefix hits | tokens found / tokens looked up | `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries` |
| preemptions | running requests evicted for memory | `vllm:num_preemptions` |
| throughput | output (and input) tokens per second | `vllm:generation_tokens`, `vllm:prompt_tokens` |
| **goodput** | requests per second that met *every* SLO (DistServe's definition) | computed by the load generator |

Report percentiles (p50, p90, p99), never averages: the tail is what users feel and what SLOs bind. Goodput is the
honest summary — throughput measured while TTFT is ten seconds is not capacity (`SimResult.goodput()`; in notebook
02 a saturated engine streams 1,700–2,300 tok/s while serving at most 0.5 requests/s within a 1 s TTFT and 50 ms
TPOT SLO, SIMULATED).

**How to load an engine.**

- **Open loop** (Poisson arrivals at a fixed rate, whatever the server does) shows queueing and overload: as the
  rate approaches capacity, queue time and TTFT grow without bound. **Closed loop** (N users, each sending the next
  request when the last one finishes) caps concurrency at N and hides overload — the server slows down and the load
  slows down with it. Use closed loop to find the throughput at a concurrency; use open loop to find the rate at
  which SLOs break. Sweep the rate and plot latency against throughput; the knee is your capacity.
- **Warm up** before measuring: the first requests pay CUDA Graph capture, compilation, cold caches and empty prefix
  caches.
- **Realistic lengths.** Input and output length distributions (and their tails) decide everything: TTFT scales with
  prompts, KV pressure with prompt + output, ITL with batch and context. Shared prefixes change the answer entirely
  — benchmark agent traffic with its real system prompts and multi-turn histories (the lab's shared-prefix
  workload). Thinking models are the extreme case: thousands of heavy-tailed output tokens hold their KV for the
  whole generation ([RL and thinking-models §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)).
- **Hold everything else fixed** — model, dtype, engine version, flags, GPU clocks — and change one knob at a time.

**The knobs and what each trades.**

| Knob (vLLM flag) | Improves | Costs |
|---|---|---|
| `max_num_batched_tokens` | TTFT; capacity, up to a few hundred tokens past the knee (§3) | ITL tail (§3) |
| `max_num_seqs` | throughput (bigger batches) | ITL, KV pressure, preemptions |
| `gpu_memory_utilization` | KV blocks → concurrency | headroom for activations and other processes (OOM risk) |
| `max_model_len` | longest request served | the worst-case concurrency vLLM logs (blocks ÷ the blocks of one `max_model_len` request); start-up fails if one such request cannot fit (§4). The block count does not change; with chunking off, the budget must be at least this |
| `enable_prefix_caching` (on) | TTFT, compute on shared prefixes | hashing overhead on no-reuse traffic (small) |
| `enable_chunked_prefill` (on) | ITL under long prompts | slightly later first tokens |
| `kv_cache_dtype fp8` | 2× KV capacity, faster long-context decode | a small accuracy cost; needs support per model and GPU |
| `quantization` (fp8, awq, gptq, …) | memory, decode speed (bytes); prefill (FP8 W8A8) | accuracy; kernel availability per GPU |
| `speculative_config` | ITL at low load | throughput at high load; draft memory (§7) |
| `tensor_parallel_size` | fits bigger models, lowers step latency | all-reduces per layer; more GPUs per replica |
| `block_size` (16) | fewer table entries per request | coarser sharing and more per-request waste |
| scheduling `policy` | per-request priority | fairness; starvation of low priority |

**Simulate before you measure.** `perf.simulate()` runs this package's scheduler and KV manager under Poisson load
with the §3 step-time model; it reproduces the *shape* of every trade-off above in seconds on a laptop, so the lab's
measurements become predictions to check. Its output is labelled SIMULATED; its assumptions (efficiencies, overhead,
spec-sheet numbers) are parameters — calibrate them against one real measurement before trusting absolute values.

## 12. Engines and where to run them

**The engines** (Sep 2026; each claim is a summary — verify against current docs):

| Engine | What it is | Choose it when |
|---|---|---|
| **vLLM** | the default open-source server: V1 architecture, PagedAttention, continuous batching, prefix caching and chunked prefill on by default, broad model and quantization coverage, OpenAI-compatible API, Prometheus metrics; NVIDIA, AMD, TPU and CPU backends | most deployments; the base of llm-d and many managed offerings |
| **SGLang** | RadixAttention prefix cache, a fast structured-output path, strong multi-turn and large-MoE (data-parallel attention + expert parallel) support | prefix-heavy, agentic and structured workloads; large MoE |
| **TensorRT-LLM** | NVIDIA's engine: optimized kernels, in-flight batching, paged KV, FP8/FP4 on Hopper and Blackwell; served via Triton or Dynamo | squeezing the most out of NVIDIA hardware at scale |
| **llama.cpp** | C/C++ inference with GGUF quantization (2–8 bit) on CPUs, Apple silicon and consumer GPUs; `llama-server` batches a few parallel slots | laptops, edge, single-user; running a real small model at T0 |

The concepts in this primer are engine-independent; the flags and metric names differ. vLLM is the reference in the
lab because its scheduler and KV manager are readable Python and its metrics are the ones layer 05's routers
consume.

**Where to run it** — concept by concept, on GCP and elsewhere (prices and obtainability in [`COMPUTE.md`](../../COMPUTE.md)):

| To learn | T0 (laptop / Colab CPU) | Non-GCP GPU (T1/T2) | GCP (T3) |
|---|---|---|---|
| §1–8 mechanics | `mini-engine-core` notebooks; the lab's fake server | — | — |
| real TTFT/ITL, knobs, prefix caching | the lab's T0 fake server (labelled) | Colab / Kaggle T4 (free, fp16 only — no bf16 on Turing), RunPod / Vast.ai 24 GB GPU (containers, ~$0.3–0.4/hr), Lambda (VMs) | a `g2-standard-4` L4 VM (Spot) |
| FP8 (sm_89+) | FP8 emulation in `quant.py` | an RTX 4090 / L4 / H100 rental | L4 (G2) or H100 (A3) |
| tensor parallelism (§9) | notebook 01 exercise 1.6 (a column/row-parallel MLP in numpy); `perf.tp_allreduces()` | Kaggle 2×T4 (PCIe; the lab's exercise 3.6), a rented 2–8× NVLink box | A2/A3 multi-GPU shapes |
| serve an endpoint | — | `docker run vllm/vllm-openai` on any GPU box | **Cloud Run with GPUs** (L4 or RTX PRO 6000; per-second billing; scale to zero) · **GKE** (vLLM Deployment on an L4 node pool; autoscaling and routing in 05) · **Vertex AI** (Model Garden deploys open models on managed endpoints with vLLM-based containers, verify) |
| TPUs | — | — | GKE with TPU v5e/v6e or v7 "Ironwood" (GA 2026-04-22); vLLM's TPU backend (verify name and status) |

A 0.5–2B model (Qwen2.5-0.5B/1.5B, Llama-3.2-1B) is enough to see every effect in this primer on a single T4 or L4;
an 8B model in bf16 needs a 24 GB GPU (and holds 17–19 concurrent 2K-token sessions there: 17 with the core's
memory inputs, 18.9 at vLLM's defaults, §4 and §8). The lab's
[`deploy/`](vllm-serving-lab/deploy/) has the any-GPU recipe, the Cloud Run GPU Terraform and the GKE manifests.

---

## In a design review

**The two-minute walkthrough.** "Our engine is a loop around one forward pass. Each step the scheduler hands out a
token budget — running requests first, one token each for decodes and a chunk for unfinished prompts — then admits
waiting requests while there are sequence slots and KV blocks. Everything scheduled is flattened into one batch, so
the weights are read once for all of it, and attention reads each request's history through its block table. A step
is memory-bound up to ~300 tokens on an H100, so decodes batch almost for free and a long prompt is the expensive
thing; chunked prefill caps the step so an 8K-token prompt cannot stall everyone's stream, and I set the budget as
large as the ITL SLO allows. Concurrency is capped by KV memory — about 30 chat sessions for an 8B model on an L4 —
and when it runs out the newest request is preempted and recomputed, which shows up as TTFT, so we alert on
preemptions. Prefix caching names each full block by a hash chained through its parent, so our agents' shared system
prompt and append-only histories are computed once; that decides our prompt layout. Sampling and structured output
reshape one distribution per step — grammar masks guarantee syntax, not correctness. Speculative decoding and
quantization are per-workload levers: speculation keeps the target distribution exactly and helps at low load; FP8
weights and KV buy prefill speed and concurrency; INT4 buys decode speed and memory. We measure with open-loop load
at realistic lengths and report goodput against the SLO."

**Drill.**

1. *Why does adding requests to a decode batch barely slow it down, and when does that stop?* — Below the knee (~300
   tokens on an H100 in bf16; `peak × bytes_per_param / (2 × bandwidth)`) a step is the weight read, shared by every
   token in it. It stops when the batch's tokens pass the knee, or earlier at long contexts, where each request's KV
   read adds bytes (64 decodes at 2,048 tokens of context add 17.2 GB to the 15 GB of weights).
2. *p99 ITL spikes whenever a long document arrives. Diagnose and fix.* — Prefill/decode interference: the prompt's
   step is long and every co-scheduled decode waits for it (the roofline model, §3: 367 ms vs 17 ms on an L4 for
   an 8K prompt, SIMULATED). Check that
   chunked prefill is on and lower `max_num_batched_tokens` until the worst step (all decodes + one full chunk)
   meets the SLO; if prefill-heavy traffic still hurts decode, disaggregate prefill and decode (05).
3. *`vllm:num_preemptions` rises at peak and TTFT p99 is 5×. What is happening?* — Running requests outgrow the KV
   pool; each preemption frees the newest request's blocks and recomputes it later, and no new requests are admitted
   in that step. Add KV capacity (FP8 KV, utilisation, a smaller or quantized model), cap `max_num_seqs`, or add
   replicas; the whole-prompt admission check prevents over-admission with chunked prefill.
4. *Our agent's prefix-cache hit rate is 3%. The system prompt is 4K tokens and identical for everyone. Why?* —
   Something before or inside it varies per request — a timestamp, a request id, a reordered tool list — and a
   block's name commits to every token before it, so one changed token early invalidates everything after. Move
   volatile content to the end, keep the history append-only, and route a session back to the replica that holds its
   prefix.
5. *Does speculative decoding change model outputs? When would you turn it off?* — No: accept with min(1, p/q) and
   resample rejections from the normalised residual max(0, p − q), and the emitted tokens are distributed exactly as
   the target's. Turn it off (or lower k) at high concurrency with short contexts, where the verify pass's extra
   positions cost real compute — the roofline simulation (§7) turns it into a slow-down past ~100–250 concurrent
   requests at 200-token contexts.
6. *INT4 or FP8 for an 8B model on a 24 GB L4 that must hold 48 concurrent 2K-token sessions and prefill 1.8K tokens
   in 300 ms?* — FP8 W8A8 with an FP8 KV cache: it halves prefill (FP8 tensor cores) and doubles KV capacity —
   a 181 ms prefill in the roofline model and 87 sessions (91 at vLLM's defaults; §8, SIMULATED). INT4
   weight-only wins decode (~3×) and memory but its prefill stays at 360 ms, because weight-only formats cut bytes,
   not FLOPs. Then gate the choice on task evals against bf16.

---

## Glossary

| Term | Meaning |
|---|---|
| Step (iteration) | one scheduling decision plus one forward pass over the tokens it chose |
| Engine core | the process running the scheduler, the KV cache manager and the GPU workers' step loop |
| Continuous batching | re-forming the batch every step, so requests join and leave at token granularity (Orca) |
| `num_computed_tokens` | tokens of a request whose K/V are already cached; the scheduler's only notion of progress |
| Token budget | `max_num_batched_tokens`: the most tokens one step may process |
| Prefill / decode | processing prompt tokens (many per step, compute-heavy) / generating one token per step (memory-bound) |
| Chunked prefill | splitting a prompt across steps under the budget so decodes keep flowing (Sarathi-Serve) |
| Knee | tokens per step where a step turns from memory- to compute-bound: peak × bytes/param ÷ (2 × bandwidth) |
| Block / block table | a fixed number of tokens' K/V (16 by default) / a request's map from logical to physical blocks |
| Slot mapping | where each new token's K/V is written: block id × block size + offset |
| Preemption (recompute) | freeing a running request's blocks when memory runs out and prefilling it again later |
| Watermark | blocks kept free when admitting requests, so running ones can grow |
| Prefix caching | reusing the K/V of full blocks whose chained hash matches a new prompt's prefix |
| Block hash chain | name(block i) = H(name(block i−1), tokens, extra keys): commits to the whole prefix |
| Free queue (LRU) | reusable blocks, least recently freed first; cached ones keep their names until evicted |
| RadixAttention | SGLang's token-granular prefix cache kept as a radix tree |
| Top-p / min-p | keep the smallest set reaching mass p / tokens with prob ≥ min_p × the top token's |
| Structured output | masking logits with a grammar automaton so every output parses |
| Speculative decoding | drafting k tokens cheaply and verifying them in one target pass with exact rejection sampling |
| Acceptance rate α | Σ min(p, q) = 1 − TV(p, q) for a sampled draft (p(x) for a greedy one): the chance a draft token survives |
| EAGLE / MTP | draft heads on the target's hidden states / multi-token prediction heads trained with the model |
| Weight-only quantization | low-bit weights dequantized inside the GEMM (W4A16, W8A16): saves bytes, not FLOPs |
| W8A8 | weights and activations in 8 bits (FP8 or INT8), computed natively by the tensor cores |
| Group-wise scales | one scale per g consecutive inputs of an output channel (g = 32–128 for INT4) |
| TP / PP / EP / DP | tensor (split each layer), pipeline (split the layers), expert (split MoE experts), data (replicas) parallelism |
| LoRA | a low-rank adapter W + BA; many can share one base model in one batch |
| TTFT / ITL / TPOT / E2E | time to first token / inter-token latency / time per output token / end-to-end latency |
| Goodput | requests per second that met every SLO |
| Open / closed loop | load at a fixed arrival rate / load from N users who wait for each response |

## Sources

Papers:

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 —
  iteration-level scheduling.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023
  (arXiv:2309.06180) — vLLM.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024
  (arXiv:2403.02310) — chunked prefill, stall-free batching.
- Zhong et al., *DistServe*, OSDI 2024 (arXiv:2401.09670) — goodput, prefill/decode disaggregation.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024 (arXiv:2312.07104)
  — RadixAttention.
- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023
  (arXiv:2211.17192); Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*
  (arXiv:2302.01318); Sun et al., *Block Verification Accelerates Speculative Decoding* (arXiv:2403.10444).
- Li et al., *EAGLE* (arXiv:2401.15077) and *EAGLE-3* (arXiv:2503.01840); Cai et al., *Medusa* (arXiv:2401.10774);
  Saxena, *Prompt Lookup Decoding* (2023); DeepSeek-AI, *DeepSeek-V3 Technical Report* (arXiv:2412.19437) — MTP.
- Chen et al., *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative
  Decoding* (arXiv:2408.11049).
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 — nucleus sampling; Nguyen et al.,
  *Min-p Sampling* (arXiv:2407.01082).
- Willard & Louf, *Efficient Guided Generation for LLMs* (arXiv:2307.09702) — Outlines; Dong et al., *XGrammar*
  (arXiv:2411.15100).
- Frantar et al., *GPTQ*, ICLR 2023 (arXiv:2210.17323); Lin et al., *AWQ*, MLSys 2024 (arXiv:2306.00978); Xiao et
  al., *SmoothQuant*, ICML 2023 (arXiv:2211.10438); Dettmers et al., *LLM.int8()* (arXiv:2208.07339); Micikevicius
  et al., *FP8 Formats for Deep Learning* (arXiv:2209.05433).
- Shoeybi et al., *Megatron-LM* (arXiv:1909.08053) — tensor parallelism.
- Hu et al., *LoRA* (arXiv:2106.09685); Sheng et al., *S-LoRA* (arXiv:2311.03285); Chen et al., *Punica*
  (arXiv:2310.18547).

Code and documentation:

- vLLM, `github.com/vllm-project/vllm` (main, read 2026-09-26): `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`,
  `vllm/v1/sample/sampler.py`, `vllm/v1/sample/rejection_sampler.py`,
  `vllm/config/{cache,scheduler,speculative,structured_outputs}.py`, `vllm/engine/arg_utils.py`,
  `vllm/v1/metrics/loggers.py`; docs at docs.vllm.ai.
- SGLang `github.com/sgl-project/sglang`; TensorRT-LLM `github.com/NVIDIA/TensorRT-LLM`; llama.cpp
  `github.com/ggml-org/llama.cpp`.
- Google Cloud documentation: Cloud Run GPUs, GKE GPUs and TPUs, Vertex AI Model Garden.
- In this repo: the [kv-cache](../kv-cache/kv-cache-primer.md),
  [paged-attention](../paged-attention/paged-attention-primer.md) and
  [flash-attention](../flash-attention/flash-attention-primer.md) primers;
  [roofline-and-fabric](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md);
  [cuda-and-nccl](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md);
  [gpu-scheduling](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md);
  [gpu-capacity-planning](../../00-foundations/gpu-capacity-planning/PRIMER.md); [transformer
  primer](../../00-foundations/transformers/docs/transformer-primer.md);
  [agentic-scaling-lab](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) (admission, rate limits, cost
  per conversation).

## Verify list

Product facts in this primer, `minengine/perf.py` and the notebooks, as of 2026-09-26. vLLM items were read from the
main branch source on that date — re-check them against the release you pin.

- **vLLM defaults and behaviour:** V1 as the architecture (separate API-server and engine-core processes, async
  scheduling); `block_size` 16; `enable_prefix_caching` true; `prefix_caching_hash_algo` `sha256` (options
  `sha256_cbor`, `xxhash`, `xxhash_cbor`); `cache_salt` in the first block's extra keys; `gpu_memory_utilization`
  0.92 on main and in v0.30.0 (the core keeps 0.9, §4); the L4's 22.49 GiB total as `nvidia-smi` reports it and where CUDA's total and
  free figures sit below it; `watermark` 0.0; `scheduler_reserve_full_isl` true; policies `fcfs` and `priority` (lower first);
  API-server defaults for `max_num_batched_tokens` / `max_num_seqs` (2,048/256 below 70 GB or on A100; 8,192/1,024
  H100/H200-class; 16,384/1,024 at ≥160 GB); chunked prefill on by default and `max_num_batched_tokens ≥
  max_model_len` required without it; `long_prefill_token_threshold` (default 0 = off),
  `long_prefill_token_threshold_adaptive` and `max_num_active_seqs` in `SchedulerConfig` on main after 0.30.0, absent
  at the v0.30.0 tag (no partial-prefill cap on main); preemption by recompute only in V1 and `kv_offloading_size` for CPU offload;
  `max_cache_hit_length = num_tokens − 1`; blocks cached inside `KVCacheManager.allocate_slots` (scheduling time);
  prefix-cache stats recorded with a `preempted` flag, preempted re-lookups kept out of
  `vllm:prefix_cache_queries/_hits`; tail-first freeing into an LRU free queue; the start-up error when one
  `max_model_len` sequence cannot fit the KV cache; `async_scheduling` on unless disabled;
  `include_stop_str_in_output` false by default; `RequestStatus` names (`FINISHED_STOPPED`,
  `FINISHED_LENGTH_CAPPED`, `FINISHED_ABORTED`, …).
- **vLLM sampling and outputs:** the sampler order (allowed tokens/bad words/logit bias → penalties → greedy below
  1e-5 → temperature → min-p → top-k/top-p); `logprobs_mode` default `raw_logprobs`; `top_k` 0 or −1 disables;
  batch-invariant mode; structured-output backends `auto`, `xgrammar`, `guidance`, `outlines`, `lm-format-enforcer`;
  speculative methods (`ngram`, `suffix`, `draft_model`, `eagle`, `eagle3`, `medusa`, `mlp_speculator`, MTP
  variants); `draft_sample_method` default `greedy` (or `probabilistic`); `rejection_sample_method` default
  `standard` (also `synthetic`, `block`); `vllm:spec_decode_num_accepted_tokens_per_pos`; quantization methods (`awq`, `gptq`, `fp8`, `compressed-tensors`, `modelopt_fp4`, `mxfp4`, …);
  `kv_cache_dtype` values (`fp8` = `fp8_e4m3`, `fp8_e5m2`) and default KV scales of 1.0; LoRA flags `max_loras`,
  `max_cpu_loras`, `max_lora_rank`; parallelism flags.
- **vLLM metric names** (FACTS, from `vllm/v1/metrics/loggers.py`); `vllm:e2e_request_latency_seconds` read from the
  same file.
- **GPU figures in `perf.GPUS`** (dense 16-bit tensor FLOP/s, memory bandwidth, capacity, FP8 support): T4 65
  TFLOP/s, 320 GB/s, 16 GB; L4 121 TFLOP/s (242 sparse), 300 GB/s, 24 GB; A100-80GB SXM 312 TFLOP/s, 2.039 TB/s;
  H100 SXM 989 TFLOP/s, 3.35 TB/s, 80 GB. The efficiencies (60% FLOPs, 80% bandwidth), the 2 ms per-step overhead
  and the 0.5 ms per draft forward in `perf.spec_speedup()` are assumptions.
- **Quantized checkpoints** keep the embedding table and LM head in 16-bit (GPTQ/AWQ leave `lm_head` unquantized;
  FP8 compressed-tensors checkpoints list it under `ignore`); published sizes of specific checkpoints.
- **Model configs in `perf.LLMS`:** Qwen2.5-0.5B (24 layers, 14 heads, 2 KV heads, head_dim 64, tied), Qwen3-0.6B
  (28, 16, 8, 128, tied), Qwen2.5-1.5B (28, 12, 2, 128, tied), Llama-3.2-1B (16, 32, 8, 64, tied), Llama-3.1-8B (32,
  32, 8, 128, untied, vocab 128,256); the 70B figures (80 layers, d = 8,192, 70.6 B parameters).
- **Transfer rates:** ~50 GB/s effective per direction for PCIe Gen5 x16 (~25 GB/s Gen4) in the swap arithmetic.
- **Where to run:** Cloud Run GPU types (L4; RTX PRO 6000 Blackwell), per-second billing and scale to zero (FACTS);
  Vertex AI Model Garden's vLLM-based serving; TPU v7 "Ironwood" GA 2026-04-22 (FACTS) and vLLM's TPU backend name
  and status; T4 lacks bf16 and FP8, L4 and H100 have FP8; Colab/Kaggle/RunPod/Vast.ai/Lambda offerings and prices
  (maintained in [`COMPUTE.md`](../../COMPUTE.md)).
- **Engine summaries in §12:** SGLang, TensorRT-LLM and llama.cpp feature claims.
