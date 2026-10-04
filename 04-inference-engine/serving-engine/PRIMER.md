# Inside an inference engine: the step loop, scheduling, batching, caching, speculation and quantization

*This is a primer for layer 04. Snapshot: September 2026. Product facts carry a date and the mark (verify). Each
formula has a worked number and the name of the function in [`mini-engine-core/`](mini-engine-core/) (package
`minengine`) that calculates it. Each number from `minengine.perf` is a **simulated** number from a roofline model.
It is not a measurement.*

This primer explains what one inference engine instance does between "a request arrived" and "tokens are streaming
out". It explains these topics:

- how the engine turns many independent requests into one forward pass per step,
- how it allocates, shares and takes back the KV cache,
- how it selects each token,
- how it guesses several tokens at one time and does not change the output,
- what quantization gives,
- how it operates across several GPUs,
- how to measure all of it.

Other topics of this layer explain the data structures and kernels under the engine:
[`kv-cache`](../kv-cache/kv-cache-primer.md),
[`paged-attention`](../paged-attention/paged-attention-primer.md) and
[`flash-attention`](../flash-attention/flash-attention-primer.md). The arithmetic of sizes and budgets (weights, KV
bytes, TTFT and TPOT budgets) is in [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md).
This primer starts from those topics and does not repeat them.

You can learn every concept at tier T0 with [`mini-engine-core/`](mini-engine-core/).
[`vllm-serving-lab/`](vllm-serving-lab/) measures the same things on a real vLLM server (T1). It also deploys the
server on Cloud Run or GKE (T3).

---

## The one-minute version

An engine is **a loop around one forward pass**.

- In each step, the **scheduler** gives out a token budget (`max_num_batched_tokens`). The requests that already run
  come first: one token each for a decode, and a chunk for a prefill that is not complete. Then the requests that
  wait get tokens, while budget, sequence slots and **KV blocks** are available.
- The engine flattens all scheduled tokens into one batch. Thus it reads the weights **once** for all requests.
  Attention reads the history of each request through its **block table**.
- After the pass, the engine samples a token for each request whose prompt is complete. Finished requests free their
  blocks immediately, and a request that waits joins the next step. This is **continuous batching**.
- A step is **memory-bound** up to a knee of ~300 tokens (H100, bf16), and compute-bound after the knee. Thus decode
  tokens cost almost nothing, but a long prompt in one step makes all other requests wait. **Chunked prefill** sets a
  limit on the step.
- KV memory sets the limit on concurrency. When the KV memory is full, the scheduler **preempts** the newest request,
  and the engine computes that request again.
- **Prefix caching** gives each full block a name: a hash chained through the name of its parent. Thus the engine
  computes shared prefixes (system prompts, agent histories) only once.
- The **sampler** changes the shape of one distribution per step. **Structured output** masks that distribution with
  a grammar.
- **Speculative decoding** verifies $k$ guessed tokens in one pass. Rejection sampling keeps the distribution of the
  target exactly.
- **Quantization** decreases the bytes (decode), the FLOPs (FP8 prefill) or the KV (concurrency).
- **Tensor parallelism** adds two all-reduces per layer.
- Measure with an open-loop load. Report TTFT/ITL percentiles and **goodput** against an SLO. Select the knobs only
  after you select the SLO.

---

## 1. Anatomy of an engine

A modern engine (vLLM's V1 architecture, verify) divides into processes, so that no work on the CPU makes the GPU
wait. SGLang and TensorRT-LLM have a similar shape. The diagram shows the processes:

```
 HTTP ───────► API server ──────────────► engine core ──────────────────────► GPU worker(s)
   (OpenAI       tokenize, validate,         scheduler ◄── KV cache manager      model runner: build input
   compatible)   apply chat template         (who runs, how many tokens,          tensors, attention metadata,
      ▲                                        which blocks)                      CUDA Graphs; forward; sample
      │                                          │  ▲                                         │
      └────────── detokenize, stop strings, ◄────┘  └──── sampled token ids ◄────────────────┘
                  stream deltas (output processor)
```

**One step** is this sequence: `schedule()`, the build of the flat batch, one forward pass, the sampling, and
`update()`. In `minengine.engine.Engine.step()`, the sequence is thirty lines:

1. **Schedule.** For each request, the scheduler decides how many new tokens run in this step. It makes sure that
   the KV cache manager has blocks for them. It publishes to the prefix cache the blocks that those tokens will fill
   (`Scheduler.schedule()`, §2 and §5).
2. **Flatten.** Each scheduled token of each request goes into one array, with no padding. Each token carries its
   position and its **slot**. The slot is the place where the engine writes the K and V of the
   token: `block_table[pos // B] × B + pos % B`. Each request also carries its block table and context length
   (`Engine.build_batch()`, the vLLM "attention metadata").
3. **Forward.** Projections and MLPs run on the full flat batch (one weight read for all). Attention runs per
   request over its own K/V, and it gathers that K/V through the block table of the request. The model computes
   logits only for the last token of each request (`TinyLM.forward()`). The paged path computes exactly what
   textbook attention computes. The core's tests compare it with `forward_dense()` to 1e-10.
4. **Sample.** Only a request whose computed tokens reach the end of its sequence gets a token. A prefill chunk that
   stops before the end samples nothing (§6).
5. **Update.** Advance `num_computed_tokens` and append the tokens. Finish each request that gets to EOS, a stop
   token or `max_tokens`, and free its blocks.

**What a step costs.** Llama-3.1-8B in bf16 on an H100 streams 15.0 GB of weights per step, and this takes
4.5 ms at 3.35 TB/s. The untied input embedding is a gather, not a stream. The 4.5 ms is the floor for *any* step,
and the number of tokens in the step does not change it. The [GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md),
§3, explains why HBM bandwidth, not FLOPs, sets this floor. Sixty-four requests that decode at 2,048 tokens of
context add 17.2 GB of KV reads (5.1 ms), for a 9.6 ms step.

A 2,048-token prefill does 29.7 TFLOP. This takes 30 ms, and the step is compute-bound (`perf.step_cost()` with
ideal efficiencies). These numbers match layer 01's `roofline.llm`
([roofline-and-fabric §3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)). The rest of this primer is
about how to pack work into those steps well.

**Request states.** A request goes from `WAITING` to `RUNNING` to `FINISHED_{STOPPED, LENGTH_CAPPED, ABORTED}`. When
memory runs out, a request takes the path `RUNNING → PREEMPTED → WAITING` (§4). These are the names of vLLM's
`RequestStatus`. The `RequestStatus` of vLLM has a few more states, for example for requests that wait for a grammar
or for a remote KV transfer. The core's `scheduler.Status` has the same six states.

Stop strings are the job of the detokenizer: they need text, and the engine core sees only token ids. Aborts (client
disconnects) must free blocks. The core's tests make sure that no block leaks.

**What sits outside the loop but matters.** CUDA Graphs replay a captured decode step to remove the kernel-launch
overhead. [cuda-and-nccl §4](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) explains why engines capture graphs
for a set of batch sizes. Asynchronous scheduling overlaps the CPU work of step n+1 with the GPU work of step n. It is
on in recent vLLM, unless you turn it off (verify). Memory profiling at start-up decides how many KV blocks exist
(§4).

## 2. Continuous batching

**Static batching** runs a batch until its longest request finishes. The slots of the other requests are idle after
those requests finish. **Continuous (iteration-level) batching** (Orca, OSDI 2022) makes the batch again in each
step. A finished request leaves immediately, and a request that waits takes its place. With output lengths
[2, 9, 3, 4] and two slots, static batching takes 13 steps at 69% slot utilisation. Continuous batching takes 9 steps
at 100%.

For 64 requests with lengths uniform in 10–400 and 8 slots, the counts are 2,842 against 1,812 steps. That is 1.57×
the throughput from scheduling alone (notebook 01, `static_steps` / `continuous_steps`).

**Unified scheduling.** vLLM V1 removed the distinction between prefill and decode. A request is two numbers:
`num_tokens` (the prompt and all generated tokens) and `num_computed_tokens` (tokens whose K/V are in the cache). The
scheduler does this in each step:

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

That is `minengine.scheduler.Scheduler.schedule()`. A worked example: the budget is 16, `r0` has an 11-token prompt,
and `r1` has a 20-token prompt. Step 1 runs `r0` 11 + `r1` 5 (a chunk, with no sampled token). Step 2 runs `r0` 1
(decode) + `r1` 15 (the rest, which samples its first token). The same rule applies to the prompt tokens and the
generated tokens of a request. For a preempted request, the scheduler only sets `num_computed_tokens` back to 0, and
its outputs count as prompt.

**The knobs.** There are two: `max_num_batched_tokens` (tokens per step, §3) and `max_num_seqs` (requests per step).
`max_num_seqs` sets the size of the per-request buffers and the CUDA-graph batch sizes. These are vLLM's API-server
defaults as of Sep 2026 (verify):

- 2,048 tokens and 256 sequences on GPUs under 70 GB and on A100s,
- 8,192 and 1,024 on H100/H200-class GPUs,
- 16,384 and 1,024 at 160 GB and above.

The policy is `fcfs` (default) or `priority`. With `priority`, a lower value goes first. In that case, the victim
of a preemption is the least important, newest request.

**What really caps concurrency is KV memory.** A request with a $P$-token prompt that generates $O$ tokens holds at
most $\lceil (P + O - 1) / B \rceil$ blocks. The engine never feeds the last sampled token back, so that token never
gets a slot (`KVCacheManager.blocks_needed(P + O − 1)`, notebook 01). Llama-3.1-8B on a 24 GB L4 at 90% utilisation
leaves 2,164 blocks of 16 tokens (`perf.kv_cache_blocks()`). Chat requests of 1,000 + 200 tokens need 75 blocks each.
Thus the L4 holds **28 concurrent requests** (31 at vLLM v0.30.0's defaults, which §4 compares).

The same arithmetic and Little's law (concurrency = arrival rate × time in system) give the size of a fleet
([capacity primer, formulas 2 and 5](../../00-foundations/gpu-capacity-planning/PRIMER.md)).

## 3. Chunked prefill and prefill/decode interference

**The step-time curve.** A step costs this time (`perf.step_cost()`, and
[roofline-and-fabric §2–3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) has the full per-kernel
model):

$$
\max\left(\frac{\text{bytes}}{\text{bandwidth}}, \frac{\text{FLOPs}}{\text{peak}}\right) + \text{overhead}
$$

The bytes are nearly constant (the weights). The FLOPs increase by $2 \times \text{params}$ per token. The **knee**
is the number of tokens where the two terms cross. It is the ridge point, scaled by the bytes per parameter
(`perf.knee_tokens()`):

$$
\text{knee} \approx \frac{\text{peak} \times \mathtt{bytes\_per\_param}}{2 \times \text{bandwidth}}
$$

- H100 bf16: 989e12 × 2 / (2 × 3.35e12) = 295 tokens
- L4 bf16: 121e12 × 2 / (2 × 300e9) = 403 tokens

For Llama-3.1-8B, the exact change occurs between 310 and 320 tokens. The engine streams the LM head of the model in
each step, but multiplies the LM head only for the rows that it samples.

Take 80% of datasheet bandwidth, 60% of datasheet FLOP/s and 2 ms of per-step overhead. These are assumptions, and
you must replace them with measurements. With these assumptions, an H100 step for Llama-3.1-8B costs this time
(SIMULATED):

| Tokens in the step | 1 | 64 | 256 | 512 | 2,048 | 8,192 |
|---|---|---|---|---|---|---|
| Step time | 7.6 ms | 7.6 ms | 8.1 ms | 14.2 ms | 52.0 ms | 224 ms |
| Per token | 7.6 ms | 0.12 ms | 0.032 ms | 0.028 ms | 0.025 ms | 0.027 ms |

### Chunking long prompts

**Interference.** Decode tokens cost almost nothing up to the knee. A prompt costs in proportion to its length. If
an 8,000-token prompt runs in one step, each request that decodes in the same step waits for that step. Take an L4
that serves Qwen2.5-1.5B to 16 users at 1K context: their 16.7 ms token gap becomes **367 ms** (SIMULATED, notebook
02). That stutter is prefill/decode interference.

**Chunked prefill** (Sarathi-Serve's "stall-free batching", OSDI 2024) keeps each step at or below the budget, and
it schedules decodes first. The prompt gets the rest of the budget, over as many steps as necessary. With a 512-token
budget, the same prompt runs in 17 chunks of 496, and the worst gap is **29.6 ms**. The TTFT of the prompt itself
increases by a small quantity, and all other requests keep streaming.

vLLM V1 has chunked prefill on by default. To turn it off, you must set `max_num_batched_tokens ≥ max_model_len`,
because a whole prompt must then fit in one step. The core applies the same rule in `SchedulerConfig`.

### Choosing the token budget

**The budget trades TTFT against ITL and, under saturation, sets capacity.** The workload has 200 requests, with
prompts of 500–6,000 tokens and outputs of 100–400. It runs on H100 + Llama-3.1-8B (SIMULATED, `perf.simulate()` on
`perf.Workload(..., seed=1)`, notebook 02 worked example 3). Another seed draws other prompts and moves every figure.

The latency columns and goodput are for an open-loop Poisson stream at 6/s. Goodput is the number of requests per
second that meet a 1 s TTFT and a 50 ms TPOT SLO (§11). The last column sends all 200 at once. This saturates the
engine, and thus that column measures its capacity:

| Setting | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | Goodput at 6/s | Capacity (saturated) |
|---|---|---|---|---|---|---|
| no chunking | 128 ms | 572 ms | 13.2 ms | 150 ms | 5.18 req/s | 1,948 tok/s |
| budget 8,192 | 137 ms | 655 ms | 13.2 ms | 210 ms | 5.17 req/s | 1,907 tok/s |
| budget 2,048 | 151 ms | 625 ms | 12.9 ms | 56.7 ms | 5.20 req/s | 2,157 tok/s |
| budget 512 | 169 ms | 760 ms | 12.3 ms | 16.3 ms | 5.23 req/s | 2,261 tok/s |
| budget 256 | 331 ms | 1,444 ms | 10.1 ms | 11.5 ms | 4.66 req/s | 1,706 tok/s |

At 6/s, each row delivers the same ~1,320 tok/s. The cause is not that the knob costs nothing. The cause is that an
engine under an open-loop load below capacity finishes exactly the work that the load offers. Such a run cannot show
a throughput effect.

At saturation, the budget matters. At 512, each step mixes a prompt chunk (compute-bound) with the KV reads of the
decodes that run (memory-bound). Thus the tensor cores and the HBM are busy at the same time. This is
Sarathi-Serve's case for hybrid batches. Whole prompts or 8,192-token steps alternate compute-heavy prefill steps
with memory-bound decode steps, and they leave one resource idle in each step.

Part of the gain comes from memory, not from overlap (SIMULATED, `SimResult.preemptions`, `peak_kv_usage`). At
saturation, the three larger settings admit prompts faster than decodes finish. They fill the KV pool (peak 100%)
and preempt 3–8 requests, and the engine then computes the work of those requests again. At 512, the pool peaks at
38%, and the scheduler preempts no request.

At 256, a step is just above the knee (~221 tokens with these efficiencies). Most of its time is the weight read,
the KV reads of the decodes and the 2 ms overhead. After the decodes take their share, the prompt gets small, poorly
amortised chunks. Thus TTFT doubles at 6/s, goodput decreases by ~10%, and capacity decreases by a quarter.

**To select the budget:** take the largest budget whose worst step meets the ITL SLO. The worst step holds all decodes that
run and a full prefill chunk. Then examine the capacity under a load that saturates the engine. Take 64 requests
that decode at 2,000 tokens of context, and a 25 ms p99 ITL target. 512 tokens gives a 14.4 ms step, and 1,024 gives
26.7 ms. Thus select 512 (notebook 02, exercise 2.4).

vLLM adds finer knobs. `long_prefill_token_threshold` sets a limit on the chunk of a long prompt, so that several
prompts progress together (0 = off by default). It has an `_adaptive` variant. `max_num_active_seqs` sets a limit on
RUNNING below `max_num_seqs`. The last two are on `main` after 0.30.0, not in the 0.30.0 wheel (verify).

**When chunking is not sufficient.** Each chunk still shares its step with decodes. Thus a prefill-heavy mix (RAG,
agents that read long contexts again) makes ITL and TTFT compete for the same GPUs. Disaggregation runs prefill and
decode on separate pools and sends the KV between them. It is the topic of layer 05
([`05-orchestrator`](../../05-orchestrator/)).
[gpu-deployment §8](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) has the architecture.

## 4. KV cache management revisited

The [paged-attention primer](../paged-attention/paged-attention-primer.md) and
[`paged_attention_minimal.py`](../paged-attention/paged_attention_minimal.py) explain paging itself: fixed-size
blocks, per-request block tables, refcounts and copy-on-write. The [kv-cache primer](../kv-cache/kv-cache-primer.md)
explains why the cache is the binding constraint. This section gives the side of the engine: how many blocks exist,
who gets them, and what occurs when they run out.

### Sizing the pool at start-up

**How many blocks.** At start-up, the engine loads the weights. Then it runs a profiling forward pass at the maximum
batch to measure the activation memory. It gives the rest of the memory under `gpu_memory_utilization` to the KV
cache (`ModelConfig.kv_bytes_per_token()`, `perf.LLM`):

$$
\text{blocks} = \frac{\text{HBM} \times \mathtt{gpu\_memory\_utilization} - \text{weights} - \text{activations/workspace}}{\mathtt{block\_size} \times \text{KV bytes per token}}
$$

$$
\text{KV bytes per token} = 2 \times \text{layers} \times \mathtt{kv\_heads} \times \mathtt{head\_dim} \times \text{bytes}
$$

- L4 (24 GB) × 0.9 − 16.06 GB (Llama-3.1-8B bf16) − 1 GB = 4.54 GB / (16 × 128 KiB = 2 MiB) = 2,164 blocks
- H100 (80 GB), same model = 26,197 blocks

Those are the core's round inputs (`perf.kv_cache_blocks()`), and each simulated number in this primer uses them.
vLLM v0.30.0 starts from different inputs. The lab's `sizing.size()` models them from a real `config.json`:

| Input | The core (`perf.kv_cache_blocks()`) | vLLM v0.30.0 defaults (the lab's `sizing.size()`) |
|---|---|---|
| total memory | 24 GB, the datasheet figure | 22.49 GiB = 24.15 GB, the L4's total as `nvidia-smi` reports it. The total that vLLM multiplies is the total that CUDA reports (`torch.cuda.mem_get_info()[1]`), which can be slightly below it. If CUDA's *free* figure is below that share, vLLM does not start. The free figure is lower by the CUDA context (a few hundred MiB) and by any other process (verify on your card) |
| `gpu_memory_utilization` | 0.9, vLLM's default for a long time | 0.92 (`vllm/config/cache.py`) |
| overhead | a flat 1 GB | profiled at start-up: the activation peak of a 2,048-token pass, CUDA graphs and non-torch buffers. The lab estimates ~1.2 GB |
| KV budget and blocks | 4.54 GB gives 2,164 | 4.96 GB gives 2,363 |
| 2,000-token sessions | 17.3 | 18.9 |

The formulas are the same, and the inputs are different. The higher utilisation adds ~229 blocks, and the larger
reported total adds ~65. The profiled overhead takes back ~95. The net difference is 199 blocks (about 8%), or one or
two 2K-token sessions of an 8B model on an L4.

Neither column is a measurement. The measurement is the `Available KV cache memory` line that vLLM prints at
start-up. The lab's notebook 01 calibrates its estimate against that line (exercise 1.5).

`block_size` is 16 in both. If one `max_model_len` sequence cannot fit, vLLM does not start. Without this check,
such a request never finishes. The core raises the same error in `Scheduler.__init__`.

The choice of model changes this number more than any knob. Per token, Qwen2.5-1.5B needs 28 KiB of KV (2 KV
heads), Qwen3-0.6B needs 112 KiB (8 KV heads of 128), and Llama-3.1-8B needs 128 KiB. The GQA width sets the number,
not the parameter count.

### Admission and preemption

**Admission.** The scheduler admits a new request only if its blocks are free *now*. Two refinements prevent
thrashing in the pool:

- An optional **watermark**: a fraction of blocks that the scheduler keeps free when it admits new or preempted
  requests (vLLM default 0, verify).
- A **whole-prompt check**: admit a request only if the full prompt fits, not only its first chunk. In vLLM, the
  scheduler calls `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)` in
  `vllm/v1/core/sched/scheduler.py` to admit a request that waits (default on, verify).

Without the whole-prompt check, chunked prefill admits prompts that it cannot finish, and it preempts them a few
steps later.

The core's version is `KVCacheManager.allocate_slots(..., admit_whole_prompt=True)`, and
`SchedulerConfig.admit_whole_prompt` turns it on or off. In notebook 02's exercise 2.6, there are 22 preemptions at
2,000 blocks with the check, and 87 without it (SIMULATED). The core's tests pin both effects on a tight pool.

**Growth and preemption.** Decodes take a new block every 16 tokens. When a request that runs needs a block and the
free queue is empty, the scheduler **preempts** the lowest-priority request that runs. Under FCFS, that is the most
recently admitted request. The scheduler frees its blocks, sets it back to zero computed tokens, and puts it at the
*front* of the queue of requests that wait. The scheduler admits no new request in that step.

The engine keeps the generated tokens of the request. When the scheduler admits it again, the engine prefills its
whole sequence again ("recompute"), and generation continues where it stopped.

Preemption does not show in the outputs (the core's tests compare against the reference with no preemption). But it
shows clearly in latency. In notebook 02, a workload that needs ~4,000 blocks on an H100 gives these results
(SIMULATED):

- at 4,000 blocks, 0 preemptions and a 176 ms p99 TTFT,
- at 3,000, 6 preemptions and 861 ms,
- at 2,000, 22 preemptions and 6.5 s.

**Short KV memory shows up as latency long before it shows up as errors.** Alert on `vllm:num_preemptions`.

**Recompute or swap?** A swap copies the blocks of the victim to host memory and back. For 2,000 tokens of
Llama-3.1-8B, that is 262 MB each way (2,000 × `perf.LLM.kv_bytes_per_token`). This takes about 5 ms per direction
at ~50 GB/s effective over PCIe Gen5 x16 (verify), and two times that on Gen4.

A recompute is a 2,000-token prefill: ~51 ms on an H100 (SIMULATED). But a recompute needs no host memory and no
work to track transfers. Also, with prefix caching, the victim's own blocks often stay in the cache until it returns.
In notebook 02, a preempted request started again at token 20, not 0.

vLLM V1 preempts by recompute only. Swap was a V0 mode. V1's CPU offloading (`kv_offloading_size`) is a cache tier,
not a preemption mode (verify). In both cases, preemption is a symptom. The solutions are more KV memory (§8's FP8
KV, a smaller model, higher utilisation), fewer concurrent sequences, or more replicas (05).

**The invariants to examine** in any block manager are in this list. In the core's tests,
`KVCacheManager.check()` examines them after each step:

- The refcount of a block equals the number of block tables that hold it.
- A block is in the free queue if and only if its refcount is zero.
- The prefix-cache map points only at blocks that carry that name.
- No table holds a block two times.
- After each request finishes or aborts, each block is free.

## 5. Prefix caching

Two requests whose prompts start with the same tokens compute identical K/V for that prefix. The cause is that the
K/V of a token depend only on the tokens before it. Prefix caching computes that K/V only once.

**Block names.** The engine caches only **full** blocks. The name of a full block is

$$
\operatorname{name}(\text{block } i) = H\bigl(\operatorname{name}(\text{block } i-1), \text{tokens in block } i, \text{extra keys}\bigr)
$$

Here, $\operatorname{name}(\text{block } {-1})$ is a constant root (`kv.hash_block()`, `kv.block_hashes()`). The name
of the parent is inside, so the name of a block commits to the *entire* prefix. The same 16 tokens after a different
history get a different name. That is not an optimisation. It is a matter of correctness, because K/V depend on all
earlier tokens.

Notebook 03 replaces the hash with a hash that ignores the parent. Then the engine serves K/V that it computed under
another prefix, and it gives no warning. The logprobs move by ~1e-2 instead of 1e-15. With a real model, the text is
incorrect.

The **extra keys** carry all other inputs that change K/V or that must isolate caches. They are the LoRA adapter
(§10), multimodal input hashes and a per-tenant `cache_salt`. The engine adds the salt to the first block only, and
the chain carries it forward.

By default, vLLM hashes with SHA-256 (`prefix_caching_hash_algo`), and xxhash is optional and non-cryptographic
(verify). If a collision occurs, the engine serves the K/V of another prefix. Also, a shared cache is a timing side
channel between tenants. These are the reasons for collision resistance and salts.

**Lookup, adopt, publish.** A new request goes along the chain of its prompt until the first miss, and it
**adopts** each hit. An adopted hit adds 1 to the refcount, with no compute and no new memory
(`KVCacheManager.lookup()`, `allocate_slots(..., hits)`). The hit has a limit of `(len(prompt) − 1) // B` blocks,
because the engine must compute the last token to get logits. Thus two identical 20-token prompts with $B$ = 4 share 16
tokens, not 20 (vLLM: `max_cache_hit_length = num_tokens − 1`).

The scheduler **publishes** a block as soon as it schedules the tokens that fill the block, not after the step. In
vLLM, this occurs inside `allocate_slots`, and in the core, inside `Scheduler.schedule()` (`cache_blocks()`). This
is safe, because each layer writes the K/V of the whole step before any request attends at that layer. It also
matters for agents. Requests that the scheduler admits in the same step share a prefix that one of them computes at that time.

Thus a burst of N parallel calls behind one system prompt prefills that prompt once and holds one copy. Notebook 03,
worked example 3, shows this: three requests, one step, `[0, 176, 176]` tokens from cache.

The core does not publish the blocks of a request in the `RUNNING` state at once. The core publishes them only when
the scheduler can preempt no more requests in that step. Thus a request that the scheduler takes out of the batch
never names blocks that it will not compute.

**No copy-on-write.** Requests share only full, immutable blocks. The hit stops before the last prompt token. A
request always writes its new tokens into its own blocks. Thus block-hash prefix caching never copies a shared block.
Copy-on-write (the [paged-attention primer](../paged-attention/paged-attention-primer.md)) matters when sequences fork
in the middle of a block: in parallel sampling and beam search.

### Eviction, hit accounting and radix trees

**Freed but still hittable.** The blocks of a finished request go to the back of an LRU **free queue**, and they keep
their names. They count as free (`vllm:kv_cache_usage_perc` does not include them). They continue to give hits until
the allocator pops them from the front and evicts the name (lazy eviction).

A request frees its blocks **tail first**. Thus the allocator evicts the private end of a prompt before the shared
head. In notebook 03, an unrelated request evicted 13 blocks, and after that the first 32 tokens of the system prompt
were still in the cache. Thus prefix caching costs no memory. It uses memory that no other request needs yet.

**Hit accounting** is in tokens. `vllm:prefix_cache_queries` counts the prompt tokens that new requests look up.
`vllm:prefix_cache_hits` counts the tokens that they find. The engine counts each new lookup of a preempted request
separately (`preempted_queries` / `preempted_hits`, not exported). Thus a victim that hits its own blocks again does
not make the hit rate too high (`CacheStats`).

Take three requests that share a 183-token system prompt. The second and third requests hit 176 tokens (the 11 full
blocks of the system prompt) of their 205–210 prompt tokens.

The gain is in TTFT and prefill compute. Notebook 03, worked example 6, shows it with
`perf.Workload(n_requests=60, rate=6, prompt_len=(2000, 2200), output_len=(40, 80), shared_prefix=1800)`
(SIMULATED): 60 requests with 2,000–2,200-token prompts that share their first 1,800 tokens, on H100 +
Llama-3.1-8B. They arrive at 6/s. The hit rate is 84%, and **TTFT p50 is 13 ms against 74 ms** without caching.
The shared blocks decrease the peak KV use from 7% to 1% of the pool. With all 60 at once, TTFT p50 is 0.32 s
against 1.6 s.

**The radix-tree alternative.** SGLang's RadixAttention keeps cached prefixes in a radix tree at token granularity,
with LRU eviction of leaves. It also uses the partial last block again. The gain from this is always less than one
block per request over block hashing (notebook 03, exercise 3.5).

Thus the difference is not the hit rate. The difference is which operations the structure makes low-cost. A tree
answers "which cached prefix does this request extend?", and SGLang uses this answer to schedule for cache locality.
A flat dictionary of block names is easy to offload, to share across processes and to publish as events. KV-aware
routers use those events (05).

### Prompt layout for agents

**Agent prompts that get hits.** The cache matches prefixes exactly, token for token. Thus, use this layout:

- Put stable content first: the system prompt, tool schemas in an order that does not change, few-shot examples
  and long documents.
- Keep the conversation **append-only**. If you want the cache to keep earlier turns, never render them again,
  summarise them or change their order.
- Put all volatile content at the end: timestamps, request ids and per-call instructions. After you send it, keep it
  *in* the history, and do not write it again.

Notebook 03 has a support session with four turns. A clock at the top of the system prompt gives 0% hits on each
turn. The same content, append-only, gives 81–86% from the second turn on, and more as the history grows.

There are two traps. First, the same text, tokenized again, can give different tokens at a boundary. Thus keep the
token ids or render deterministically. Second, across replicas, the next turn must get to the replica that holds the
prefix. This is prefix-affinity routing (05).

The platform lab's [context-engineering notebook](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/04_context_engineering_and_caching.ipynb)
in [`07-application-agent-framework`](../../07-application-agent-framework/) gives the agent-side view of context
layout and caching.

## 6. Sampling and structured output

The forward pass ends in logits, one per vocabulary entry. The sampler turns them into a token. vLLM V1's `Sampler`
(verify) and `sampler.process_logits()` use this order:

```
allowed-token / grammar mask  →  penalties  →  greedy if T < 1e-5  →  ÷ temperature
  →  min-p  →  top-k  →  top-p  →  draw
```

| Setting | Rule | Worked (probs 0.5, 0.3, 0.15, 0.05) |
|---|---|---|
| temperature $T$ | $\operatorname{softmax}(z / T)$. $T$ < 1 sharpens and $T$ > 1 flattens. The ranking never changes | $T$ = 2 keeps the order and flattens the gaps |
| top-k | keep the $k$ largest logits | $k$ = 2 keeps {0.5, 0.3}, renormalised to 0.625 / 0.375 |
| top-p (nucleus) | keep the smallest top set whose mass reaches $p$ | $p$ = 0.8 keeps {0, 1}, and $p$ = 0.81 keeps {0, 1, 2} (`top_p_filter()`) |
| min-p | keep tokens with prob ≥ min_p × max prob | min_p = 0.2 gives the threshold 0.1, which keeps {0, 1, 2} (`min_p_filter()`) |
| repetition penalty $r$ | for tokens seen in the prompt *or* the output: logit ÷ $r$ if positive, × $r$ if negative | $r$ = 2: logits 2, −2 become 1, −4 |
| presence / frequency | output tokens only (OpenAI definitions): $-\text{presence} \times [\text{seen}] - \text{frequency} \times \text{count}$ | (`apply_penalties()`) |

top-k keeps the same number of tokens for any confidence of the model. But top-p and min-p adapt. After "When memory
runs ou", the top-p 0.9 set of the core's small model is 3 tokens. After "The engine ", it is 16 (notebook 04). Greedy
decoding of a weak model loops (`tofofof…`). Penalties break loops, because they push seen tokens down.

**Seeds and reproducibility.** Each request gets its own random generator (from `seed`, or derived). Thus a seeded
request draws the same tokens, whatever other requests share its steps. You need this property to replay an
incident. On a GPU, the logits themselves can differ in the last bits with batch size (kernel choice, reduction
order). For bitwise reproducibility, vLLM has a batch-invariant mode at some cost in speed (verify).

**Logprobs** come from the **raw** logits by default in vLLM V1 (`logprobs_mode="raw_logprobs"`, verify). They show
what the model believed, before temperature, penalties or masks. They are the lowest-cost monitoring signal that an
engine emits.

**Stop conditions:** the engine core stops a request at EOS (unless `ignore_eos`), `stop_token_ids`, `max_tokens` and
`max_model_len`. The detokenizer finds stop *strings*, and then it aborts the request. By default, the output does not
include the stop string.

**Structured output** is sampling with a mask in the shape of a grammar. The engine compiles a JSON schema, a regex or
a context-free grammar to an automaton. In each step, each token that takes the text out of the grammar gets
$-\infty$ before sampling. Then the automaton advances on the drawn token (`ChoiceFSM`: `start()`, `allowed(state)`,
`advance(state, token)`). With a byte vocabulary, this is simple. With a 100k-token BPE vocabulary, each token spans
several characters.

Thus the difficult part is to calculate, per grammar state, which of 100k tokens keep the text legal. A token is
legal only if each of its characters is legal, from that state. The calculation must be sufficiently fast to overlap
with the forward pass of the GPU.

Notebook 04 (worked example 6, exercise 4.6) compiles the schema `{"n": <non-negative integer>}` into a 10-state
character automaton. For each state, it precomputes the next state after each of 36 tokens (single characters and
multi-character merges). The results are:

- The token `{"n": ` jumps six states at one time.
- `07` is legal after `{"n": 1` but not immediately after `{"n": `.
- The text `{"n": 0}` comes out as 19 different token sequences in 300 samples (the re-tokenization trap of §5).
- A `max_tokens` cap leaves a legal prefix that does not parse. Examine `finish_reason`.

The backends optimise that precomputation (vLLM: `xgrammar`, `guidance`/llguidance, `outlines`,
`lm-format-enforcer`, `auto`, verify). SGLang adds "jump-forward" decoding through parts of the output that the
grammar fully decides.

The guarantee is **syntax, not sense**. In notebook 04, the core's small model produces a valid `{"answer": "yes"}` or
`{"answer": "no"}` every time. But its own log-probability of that text is about −123 nats. The grammar forced every
character. It can also force an answer that looks confident from a model that knows nothing.

Validate the values downstream. Monitor the logprobs of constrained fields.

## 7. Speculative decoding

Because decode is memory-bound (§3), the score of $k$ + 1 positions costs approximately the same as the score of
one position. Thus let a low-cost **proposer** make $k$ draft tokens, and let the target examine them all in one pass.

### Exact rejection sampling

**The rule** (Leviathan, Kalman and Matias 2023, and Chen et al. 2023). Do these steps for each draft token
$x \sim q$, in order:

- Accept $x$ with probability $\min(1, p(x) / q(x))$ (`spec.verify()`).
- At the first rejection, emit $y \sim \operatorname{norm}(\max(0, p - q))$ and stop (the "recovered" token).
- If you accept all $k$, emit one more token from the next $p$ of the target (the "bonus" token).

This is why the rule is exact. The probability to emit token $y$ has two terms. The first term is
$q(y) \min(1, p(y)/q(y)) =$ $\min(p(y), q(y))$, through acceptance. The second term is
$P(\text{reject}) \times \text{residual}(y)$.

$P(\text{reject}) = 1 - \sum \min(p, q)$, and the normaliser of the residual, $\sum \max(0, p - q)$, equals that same
$1 - \sum \min(p, q)$. Thus the second term is $\max(0, p(y) - q(y))$. The sum is
$\min(p, q) + \max(0, p - q) = p(y)$.

**Speculation changes speed, never the distribution.** The core's chi-square test over all 64 three-token outcomes
confirms it. The test also rejects a plausible bug (a resample from $p$ instead of the residual). With temperature 0,
both distributions are one-hot, and the rule becomes "accept iff the draft's argmax is the target's". Thus greedy
speculation reproduces greedy decoding token for token.

### Acceptance and speedup

**How much it yields.** The per-token acceptance rate is $\alpha = \sum \min(p, q) = 1 - \operatorname{TV}(p, q)$
(`spec.acceptance_rate()`). For $p$ = (0.5, 0.3, 0.15, 0.05) and $q$ = (0.2, 0.2, 0.2, 0.4), $\alpha$ = 0.6. If each
draft token survives with probability $\alpha$, one target pass emits this number of tokens on average
(`spec.expected_tokens()`):

$$
E[\text{tokens per pass}] = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

$\alpha$ = 0.8, $k$ = 4: (1 − 0.8⁵) / 0.2 = 3.36

If one draft step costs $c$ target steps, a round costs $k\,c + 1$. Thus $\text{speedup} = E / (k\,c + 1)$
(`spec.speedup()`). For $\alpha$ = 0.8 and $c$ = 0.1, the best depth is $k$ = 6 at 2.47×. $k$ = 4, 5, 6, 7 give 2.40,
2.46, 2.47, 2.45, which is a shallow optimum (`spec.best_k()`).

In notebook 05, the core's small target and a 10× smaller draft agree with $\alpha$ ≈ 0.72 per position on average.
The measured tokens per pass match the formula at $k$ = 1 and 2, but they are lower at $k$ = 4 (2.61 against 2.90).
The formula assumes that the target accepts each position independently, with one $\alpha$. Real acceptance changes
with position (p10 0.53, p90 0.87 there), and it correlates across positions. Thus the formula predicts too much for
deep speculation. Measure the acceptance per position (vLLM: `vllm:spec_decode_num_accepted_tokens_per_pos`) before you
select $k$.

**What vLLM measures.** By default, vLLM's draft models make their drafts **greedily**
(`draft_sample_method="greedy"`, verify). $x$ is the argmax of the draft, and the rejection sampler treats $q$ as
one-hot. The rule is still exact: accept $x$ with probability ${p(x)}$, or else resample from $p$ with $x$ removed.
But then the acceptance rate is ${p(x)}$, not $\sum \min(p, q)$. `"probabilistic"` samples $x \sim q$ and uses the
full $q$.

Exactness holds for the `standard` rejection method and for `block` (block verification, Sun et al. 2024). `block`
verifies the $k$ drafts jointly. It still preserves the distribution, and in expectation it accepts at least as many
tokens. The `synthetic` method of vLLM accepts with a calibrated probability to benchmark speed, and it does not
preserve the distribution (verify).

### Proposers and when speculation pays

**Proposers** (vLLM's methods include `ngram`, `suffix`, `draft_model`, `eagle`, `eagle3`, `medusa`,
`mlp_speculator` and model-specific MTP, verify):

| Proposer | How it drafts | Cost $c$ | When it wins |
|---|---|---|---|
| draft model | a small model with the **same tokenizer** runs $k$ steps | ~0.1–0.4 (a 1B draft is ~0.19 of an 8B step here, SIMULATED) | a good small model of the same family exists |
| n-gram / prompt lookup | copy what followed the last occurrence of the current n-gram in the context | ~0 | outputs quote the input: code edits, tool results, RAG answers |
| EAGLE / EAGLE-3 | a light head on the target's own hidden states makes draft features, then draft tokens | a few % | general chat, the common production choice |
| MTP | extra prediction heads, trained with the model (for example DeepSeek-V3) | a few % | models that ship them |

Training a draft model is distillation with acceptance as the metric. $\alpha$ is $1 - \operatorname{TV}$ between the
pair on the target's own text. Thus, for a fine-tuned target, a draft trained on the target's outputs or logits is
better than an off-the-shelf small model of the same family. See
[distillation primer §7](../../00-foundations/distillation/PRIMER.md#7-a-distilled-draft-for-speculative-decoding).

**When speculation gives no more gain** (SIMULATED, `perf.spec_speedup()`, notebook 05: Llama-3.1-8B target, Llama-3.2-1B draft,
H100, $\alpha$ = 0.7, $k$ = 4). A round is one engine step. It pays these costs:

- the fixed overhead of the step, one time (2 ms, as everywhere in this primer),
- $k$ draft forwards at their roofline time, plus an assumed 0.5 ms each (a CUDA-graph replay and a draft sample),
- a verify pass of $(k + 1) \times \text{batch}$ tokens, with logits at every position.

When the verify pass crosses the knee, the extra positions cost real FLOPs:

| Context | B = 1 | B = 16 | B = 64 | B = 128 | B = 256 |
|---|---|---|---|---|---|
| 200 tokens | 1.58× | 1.58× | 1.38× | 0.97× | 0.65× |
| 2,000 tokens | 1.58× | 1.55× | 1.49× | 1.45× | KV does not fit |

The robust conclusion is the shape of the table. At short contexts, speculation becomes a slow-down past ~100–250
concurrent requests. Here that is 128, and 256 if a draft forward costs only its roofline time. At long contexts,
decode stays bound by KV reads, which verification amortises. Thus speculation still pays, and memory caps the batch
first.

The batch-1 figure depends on the overhead assumption. Here the draft costs $c$ ≈ 0.19 of a target step (its weights
are 17% of the target's, plus the 0.5 ms), which gives ~1.6×. With no per-draft overhead, $c$ ≈ 0.12 and the speedup
is ~1.9×. If every draft forward pays a full 2 ms step overhead, $c$ ≈ 0.38 and the speedup is only ~1.1×. Measure $c$
on your stack before you quote a number. An EAGLE-like head at $c$ = 0.05 gives 2.31× in the same simulation.

Treat speculation as a per-workload setting. Turn it on for latency-sensitive, low-concurrency or copy-heavy traffic.
Monitor the acceptance rate and ITL.

**Inside the engine**, the verify pass is only a ($k$ + 1)-token chunk. The scheduler reserves "lookahead" slots for
the draft tokens. It rolls back rejected positions: it decreases `num_computed_tokens`. The next step writes over
their K/V slots. The core keeps speculation outside the loop (`spec.speculative_generate()`), to show the rule
without the management of those slots.

## 8. Quantization

Store numbers in fewer bits, with a scale that tells what the bits mean: $w \approx \text{code} \times \text{scale}$
(`quant.quantize()`).

### Formats and scales

**Formats.** These are the formats:

- INT8 (codes −127…127).
- INT4 (−7…7 symmetric here). GPTQ/AWQ checkpoints often use an asymmetric zero point.
- FP8-E4M3 (1 sign, 4 exponent, 3 mantissa bits). Its largest finite value is 448. Its relative step is 1/8, so the
  rounding error is at most 6.25% (`quant.fp8_e4m3()`).
- FP8-E5M2 (more range, 2 mantissa bits).
- On Blackwell, 4-bit floats with small shared scales (NVFP4, MXFP4, verify).

**Granularity** is the number of weights that share a scale. Granularity is where you win accuracy. One outlier
stretches a shared scale and removes the resolution of all other weights.

| On a 256×128 Gaussian weight (notebook 06) | Bits per weight | Relative error | SQNR |
|---|---|---|---|
| INT8 per tensor | 8 | 1.03% | 39.8 dB |
| INT8 per output channel | 8 | 0.71% | 43.0 dB |
| FP8-E4M3 per tensor | 8 | 2.63% | 31.6 dB |
| INT4 per channel | 4 | 12.8% | 17.9 dB |
| INT4 groups of 128 | 4.125 | 11.8% | 18.5 dB |
| INT4 groups of 32 | 4.5 | 9.8% | 20.2 dB |

Each bit gives ~6 dB. Scales also cost bits. A 16-bit scale per group of 128 adds 16/128 bits, so INT4 g128 costs
**4.125 bits per weight** (`quant.bits_per_weight()`). Also, checkpoints do not quantize every weight. GPTQ, AWQ and
FP8 checkpoints quantize the linear layers of the transformer blocks, and they keep the embedding table and LM head
in 16-bit.

For Llama-3.1-8B, that is 6.98 B weights at 4.125 bits (3.60 GB), plus an untied 128,256 × 4,096 embedding and LM
head in bf16 (2.10 GB). The total is **5.70 GB**, not the 4.14 GB that 8.03 B × 4.125 bits suggests
(`quant.weight_gb(..., keep16_params=)`, `perf.LLM.weight_bytes`, and the published checkpoint sizes, verify).

On well-behaved weights, INT8 per channel is better than FP8 (43 against 32 dB). The strength of FP8 is range, which
matters for activations. Take one output channel 100× larger than the rest. Then per-tensor INT8 reports a 3.4%
aggregate error, but the 31 normal channels are at 62%. Thus **aggregate metrics hide per-channel damage** (notebook
06).

### Kernels, speed and accuracy

**Weight-only against W8A8.** *Weight-only* formats (W4A16/W8A16) dequantize to bf16 inside the GEMM. GPTQ uses
second-order error compensation. AWQ uses scales that protect the channels with large activations. Both run with
Marlin-style kernels. Weight-only formats decrease the **bytes**, not the FLOPs.

*W8A8* formats also quantize activations (per token, at run time). They use the native low-precision math of the
tensor cores: FP8 on Ada (sm_89), Hopper and Blackwell, and INT8 more widely. Activations have outlier channels that
make per-token scales coarse. SmoothQuant divides them by
$s_j = \max \lvert X_j \rvert^{\alpha} / \max \lvert W_j \rvert^{1-\alpha}$ and folds $s$ into the weights. Then
${XW}$ does not change (`quant.smoothquant_scales()`: 6.6× less output error in notebook 06).

**What each buys.** The setup is Llama-3.1-8B on a 24 GB L4, decode at 1K context, an 1,800-token prefill, and
sessions of 1,800 + 200 tokens. The embedding and LM head stay in bf16 in all rows (SIMULATED, notebook 06). The table
gives the sessions for both sets of memory inputs in §4:

- whole sessions with the core's inputs (`perf.kv_cache_blocks()`, as notebook 06 prints them),
- sessions at vLLM v0.30.0's defaults, as the lab's `sizing.size(..., typical_len=2000)` estimates them.

| Scheme | Weights | Decode, batch 1 | Decode, batch 32 | Prefill 1.8K | Sessions (core inputs) | Sessions (vLLM defaults) |
|---|---|---|---|---|---|---|
| bf16 | 16.1 GB | 65 ms | 82 ms | 360 ms | 17 | 18.9 |
| INT8 weight-only | 9.1 GB | 36 ms | 53 ms | 360 ms | 43 | 45.5 |
| INT4 weight-only g128 | 5.7 GB | 22 ms | 39 ms | 360 ms | 56 | 58.3 |
| FP8 W8A8 | 9.1 GB | 36 ms | 53 ms | 181 ms | 43 | 45.5 |
| FP8 W8A8 + FP8 KV | 9.1 GB | 36 ms | 44 ms | 181 ms | 87 | 91.1 |
| INT4 weight-only + FP8 KV | 5.7 GB | 22 ms | 30 ms | 360 ms | 113 | 116.6 |

Decode is the weight read, so INT4 weight-only is ~3× faster at batch 1. This is not the 3.9× that the bytes of its
linear layers suggest. The cause is that the 16-bit LM head (1.05 GB, read every step), the KV read and the per-step
overhead do not decrease. Prefill is compute-bound, so only FP8 W8A8 makes it faster, and only on FP8 hardware. The
core does not accept FP8 compute on a T4 or A100.

The engine quantizes the **KV cache** separately (`--kv-cache-dtype fp8`, e4m3 or e5m2, per-tensor scales that
default to 1.0 unless calibrated, verify). This gives half the bytes and two times the sessions. The core's FP8 KV
emulation costs 2e-5 nats of KL and 1% of top-1 agreement on its small model. On a small GPU, **quantization is a
concurrency lever before it is a speed lever**
([capacity primer, formula 2](../../00-foundations/gpu-capacity-planning/PRIMER.md)).

**How to examine accuracy.** Compare distributions, not strings. Use the mean KL and top-1 agreement over many positions
(`quant.compare_logits()`), perplexity, and most of all task-level evals on your own data. Greedy text is a brittle
metric. In notebook 06, an INT8 model with 99.8% top-1 agreement diverges from the full-precision greedy text after
13 tokens. The cause is one flipped near-tie, which changes everything after it.

**Deep dive:** [quantization](../quantization/PRIMER.md) (module 04.9). It covers these topics:

- the formats, down to their bit patterns,
- GPTQ, AWQ and SmoothQuant,
- what each scheme runs as per GPU generation,
- KV-cache quantization,
- how to produce a checkpoint with llm-compressor,
- how to measure the accuracy that you pay.

## 9. Parallelism inside the engine

When a model does not fit one GPU, or when one GPU is too slow, the engine operates across several GPUs. The
menu and the rule are in [gpu-deployment §4](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md):
tensor and expert parallelism inside the NVLink domain, and pipeline and data parallelism across it. The cost of
each collective is in [cuda-and-nccl §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) and in
[roofline-and-fabric §5](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). This section tells what the
engine does.

**Tensor parallelism (TP)** divides every layer. The Megatron pattern pairs a *column-parallel* matmul with a
*row-parallel* one:

- The *column-parallel* matmul is QKV, or the gate/up of the MLP. Each GPU computes a slice of the output features,
  with no communication.
- The *row-parallel* matmul is the attention output projection or the down projection of the MLP. Each GPU holds a
  slice of the input features and produces a partial sum.

One **all-reduce** after each row-parallel matmul restores the full activation. Thus there are **two all-reduces per
layer per forward pass**, each of $\text{tokens} \times d_{\text{model}}$ values (`perf.tp_allreduces()`). Notebook
01's exercise 1.6 divides the MLP of the core's small model across two "ranks" in numpy. It makes sure that the sum
of their partials is the dense output.

A 70B model (80 layers, $d$ = 8,192) does 160 all-reduces per step. At 64 requests that decode, each all-reduce is 1
MiB. It is latency-bound: the $\alpha$ term dominates. This is why engines use custom all-reduce kernels and NVLink.
At a 4,096-token prefill chunk, each all-reduce is 64 MiB (bandwidth-bound).

TP divides the heads across GPUs, so it also divides the KV cache. With 8 KV heads, TP = 8 puts one KV head on each
GPU. At a higher TP, the engine replicates the KV heads. TP decreases the weights per GPU (141 GB in bf16 becomes 35
GB per GPU at TP = 4) and the latency of every step.

**Pipeline parallelism (PP)** puts consecutive layers on different GPUs or nodes and passes activations forward. It
communicates a small quantity of data (one activation per boundary per step). But it needs several micro-batches in
flight to keep every stage busy, and it does not decrease the latency of a single request. Use it across nodes, when
one node cannot hold the model.

**Expert parallelism (EP)** puts the experts of a MoE layer on different GPUs. The engine dispatches tokens to their
experts and combines the results. This takes two **all-to-alls** per MoE layer, and they are sensitive to load
imbalance between experts. Large MoE deployments combine data-parallel attention with expert-parallel MoE layers
("wide EP", layer 05). [MoE primer §6](../../00-foundations/mixture-of-experts/PRIMER.md#6-running-moe-on-gpus) works
through dispatch and combine, the slowest rank, and TP against EP for experts.

**Data parallelism (DP)** is replication: independent engine replicas, each with its own KV cache. Throughput scales
linearly. The important part is to route requests to the replica that holds their prefix and has the least load
(layer 05).

**How to select.** Use the smallest TP that holds the weights and the KV cache that you need. Add replicas for throughput.
An 8B model fits one 24 GB GPU (§4). A 70B model needs 141 GB in bf16. For this model, these are the options:

- TP = 2 on 80 GB GPUs leaves almost nothing for KV.
- TP = 4 leaves ~37 GB per GPU for KV at 90% utilisation.
- INT4 (39.5 GB with its 16-bit embedding and LM head, exercise 6.3) fits one GPU.

vLLM's flags are `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size` and
`--enable-expert-parallel` (verify).

You can learn the split itself at T0 (exercise 1.6). To measure it, you need two GPUs (tier T2). The lab's exercise
3.6 predicts TP = 2 on Kaggle's free 2×T4 and measures it there over PCIe. A rented NVLink pair shows the speed
(prices in [`COMPUTE.md`](../../COMPUTE.md)).

## 10. Multi-LoRA serving

A LoRA adapter replaces a frozen weight $W$ with ${W + B A}$, with rank $r \ll d$. It adds
$r\,(d_{\text{in}} + d_{\text{out}})$ parameters per adapted matrix (`perf.lora_params()`). Rank 16 on all seven
linear layers of Llama-3.1-8B is 42 M parameters, or 84 MB in bf16. That is about 0.5% of the base model. Thus one
engine can hold one base model and many adapters. It can serve requests for different adapters **in the same
batch**.

The base matmul runs once for all requests. A batched "shrink/expand" kernel (Punica's SGMV, S-LoRA, and vLLM's LoRA
kernels, verify) applies to each token its own adapter.

The engine manages these things:

- **An adapter cache.** It has a fixed number of GPU slots (`max_loras`) and a larger CPU cache (`max_cpu_loras`). A
  request for an adapter that is not in a GPU slot waits for a load. A step can mix at most `max_loras` adapters
  (verify flag names). `max_lora_rank` sets the size of the buffers of the kernels.
- **Prefix caching per adapter.** The same prompt under two adapters produces different K/V. Thus the adapter id is
  an extra key in every block name (§5). By construction, adapters share no cached blocks.
- **Cost.** Each adapted step pays for the extra shrink/expand matmuls. Cold adapters pay a load. Many distinct
  adapters per step make batching less effective. `vllm:lora_requests_info` reports the adapters that run and the
  adapters that wait.
- **Routing.** With several replicas, the task to send the traffic of an adapter to the replicas that already
  have it loaded is an affinity problem. Prefix caching is the same type of affinity problem (layer 05).

## 11. Measuring an engine

**The metrics** use vLLM's Prometheus names from FACTS. vLLM exports the counters with a `_total` suffix:

| Metric | Definition | vLLM name |
|---|---|---|
| TTFT | $\text{first token time} - \text{arrival}$: time in the queue + prefill | `vllm:time_to_first_token_seconds` |
| ITL | gap between consecutive tokens of one request | `vllm:inter_token_latency_seconds` |
| TPOT | $(\text{last token time} - \text{first token time}) / (\text{tokens} - 1)$, per request | `vllm:request_time_per_output_token_seconds` |
| E2E | $\text{last token time} - \text{arrival}$ | `vllm:e2e_request_latency_seconds` |
| queue time | from arrival until the scheduler first schedules the request | `vllm:request_queue_time_seconds` |
| running / waiting | requests in each state | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| KV usage | the fraction of blocks that requests hold, without cached-but-free blocks | `vllm:kv_cache_usage_perc` |
| prefix hits | tokens found / tokens looked up | `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries` |
| preemptions | requests that the scheduler evicts during their run, for memory | `vllm:num_preemptions` |
| throughput | output (and input) tokens per second | `vllm:generation_tokens`, `vllm:prompt_tokens` |
| **goodput** | requests per second that met *every* SLO (DistServe's definition) | the load generator computes it |

Report percentiles (p50, p90, p99), never averages. Users feel the tail, and SLOs bind the tail. Goodput is the honest
summary. A throughput that you measure at a TTFT of ten seconds is not capacity (`SimResult.goodput()`). In
notebook 02 (SIMULATED), a saturated engine streams 1,700–2,300 tok/s. But it serves at most 0.5 requests/s within a
1 s TTFT and 50 ms TPOT SLO.

**How to load an engine.**

- **Open loop** (Poisson arrivals at a fixed rate, whatever the server does) shows the growth of the queue and
  overload. As the rate approaches capacity, queue time and TTFT increase without bound. **Closed loop** (N users,
  each of whom sends the next request when the last one finishes) caps concurrency at N and hides overload. The
  server slows down, and the load slows down with it.

    Use closed loop to find the throughput at a concurrency. Use open loop to find the rate at which SLOs break.
    Sweep the rate, and plot latency against throughput. The knee is your capacity.

- **Warm up** the engine before you measure. The first requests pay for CUDA Graph capture, compilation, cold caches
  and empty prefix caches.
- **Realistic lengths.** The distributions of input and output length (and their tails) decide everything. TTFT
  scales with prompts, KV pressure scales with prompt + output, and ITL scales with batch and context. Shared
  prefixes change the answer completely. Benchmark agent traffic with its real system prompts and multi-turn
  histories (the lab's shared-prefix workload). Thinking models are the extreme case: thousands of heavy-tailed
  output tokens hold their KV for the whole generation
  ([RL and thinking-models §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)).
- **Hold everything else fixed**: model, dtype, engine version, flags and GPU clocks. Change one knob at a time.

**The knobs and what each trades.**

| Knob (vLLM flag) | Improves | Costs |
|---|---|---|
| `max_num_batched_tokens` | TTFT, and capacity up to a few hundred tokens past the knee (§3) | ITL tail (§3) |
| `max_num_seqs` | throughput (larger batches) | ITL, KV pressure, preemptions |
| `gpu_memory_utilization` | KV blocks, thus concurrency | headroom for activations and other processes (OOM risk) |
| `max_model_len` | the longest request that the engine serves | the worst-case concurrency that vLLM logs (blocks ÷ the blocks of one `max_model_len` request). Start-up fails if one such request cannot fit (§4). The block count does not change. With chunking off, the budget must be at least this value |
| `enable_prefix_caching` (on) | TTFT, compute on shared prefixes | hashing overhead on traffic with no reuse (small) |
| `enable_chunked_prefill` (on) | ITL under long prompts | slightly later first tokens |
| `kv_cache_dtype fp8` | 2× KV capacity, faster long-context decode | a small accuracy cost. It needs support per model and GPU |
| `quantization` (fp8, awq, gptq, …) | memory and decode speed (bytes), prefill (FP8 W8A8) | accuracy, kernel availability per GPU |
| `speculative_config` | ITL at low load | throughput at high load, draft memory (§7) |
| `tensor_parallel_size` | fits larger models, decreases step latency | all-reduces per layer, more GPUs per replica |
| `block_size` (16) | fewer table entries per request | coarser units to share, and more waste per request |
| scheduling `policy` | per-request priority | fairness, starvation of low priority |

**Simulate before you measure.** `perf.simulate()` runs the scheduler and KV manager of this package under Poisson
load, with the step-time model of §3. It reproduces the *shape* of every trade-off in the knob table of this
section in seconds on a laptop. Thus the measurements of the lab become predictions that you examine.

Its output has the label SIMULATED. Its assumptions (efficiencies, overhead, spec-sheet numbers) are parameters.
Calibrate them against one real measurement before you trust absolute values.

## 12. Engines and where to run them

**The engines** (Sep 2026). Each claim is a summary. Compare each claim with the current docs (verify):

| Engine | What it is | Select it when |
|---|---|---|
| **vLLM** | the default open-source server. It has the V1 architecture, PagedAttention, continuous batching, and prefix caching and chunked prefill on by default. It has broad model and quantization coverage, an OpenAI-compatible API and Prometheus metrics. It has NVIDIA, AMD, TPU and CPU backends | most deployments, the base of llm-d and many managed services |
| **SGLang** | RadixAttention prefix cache, a fast structured-output path, strong multi-turn and large-MoE (data-parallel attention + expert parallel) support | prefix-heavy, agentic and structured workloads, large MoE |
| **TensorRT-LLM** | NVIDIA's engine: optimized kernels, in-flight batching, paged KV, FP8/FP4 on Hopper and Blackwell. Triton or Dynamo serves it | to get the most out of NVIDIA hardware at scale |
| **llama.cpp** | C/C++ inference with GGUF quantization (2–8 bit) on CPUs, Apple silicon and consumer GPUs. `llama-server` batches a few parallel slots | laptops, edge, single-user, a real small model at T0 |

The concepts in this primer do not depend on the engine. The flags and metric names are different. vLLM is the
reference in the lab for two reasons. Its scheduler and KV manager are readable Python. Its metrics are the metrics
that layer 05's routers use.

**Where to run it**, concept by concept, on GCP and elsewhere (prices and obtainability in
[`COMPUTE.md`](../../COMPUTE.md)):

| To learn | T0 (laptop / Colab CPU) | Non-GCP GPU (T1/T2) | GCP (T3) |
|---|---|---|---|
| §1–8 mechanics | `mini-engine-core` notebooks, the lab's fake server | — | — |
| real TTFT/ITL, knobs, prefix caching | the lab's T0 fake server (labelled) | Colab / Kaggle T4 (free, fp16 only, no bf16 on Turing), RunPod / Vast.ai 24 GB GPU (containers, ~$0.3–0.4/hr), Lambda (VMs) | a `g2-standard-4` L4 VM (Spot) |
| FP8 (sm_89+) | FP8 emulation in `quant.py` | an RTX 4090 / L4 / H100 rental | L4 (G2) or H100 (A3) |
| tensor parallelism (§9) | notebook 01 exercise 1.6 (a column/row-parallel MLP in numpy), `perf.tp_allreduces()` | Kaggle 2×T4 (PCIe, the lab's exercise 3.6), a rented 2–8× NVLink box | A2/A3 multi-GPU shapes |
| serve an endpoint | — | `docker run vllm/vllm-openai` on any GPU box | **Cloud Run with GPUs** (L4 or RTX PRO 6000, per-second billing, scale to zero). **GKE** (vLLM Deployment on an L4 node pool, autoscaling and routing in 05). **Vertex AI** (Model Garden deploys open models on managed endpoints with vLLM-based containers, verify) |
| TPUs | — | — | GKE with TPU v5e/v6e or v7 "Ironwood" (GA 2026-04-22). vLLM's TPU backend (verify name and status) |

A 0.5–2B model (Qwen2.5-0.5B/1.5B, Llama-3.2-1B) is sufficient to see every effect in this primer on a single T4 or
L4. An 8B model in bf16 needs a 24 GB GPU. There, it holds 17–19 concurrent 2K-token sessions: 17 with the core's
memory inputs, and 18.9 at vLLM's defaults (§4 and §8). The lab's [`deploy/`](vllm-serving-lab/deploy/) has the
any-GPU recipe, the Cloud Run GPU Terraform and the GKE manifests.

---

## In a design review

**The two-minute walkthrough.** "Our engine is a loop around one forward pass. In each step, the scheduler gives out
a token budget. The requests that already run come first: one token each for decodes, and a chunk for prompts that
are not complete. Then the scheduler admits the requests that wait, while there are sequence slots and KV blocks.

"The engine flattens all scheduled tokens into one batch, so it reads the weights one time for all of them. Attention
reads the history of each request through its block table.

"A step is memory-bound up to ~300 tokens on an H100. Thus decodes batch at almost no cost, and a long prompt is the
high-cost item. Chunked prefill caps the step, so an 8K-token prompt cannot make the streams of all other requests wait. I
set the budget as large as the ITL SLO permits.

"KV memory caps concurrency, at about 30 chat sessions for an 8B model on an L4. When the KV memory runs out, the
scheduler preempts the newest request, and the engine computes that request again. The preemption shows up as TTFT,
so we alert on preemptions.

"Prefix caching gives each full block a name: a hash chained through its parent. Thus the engine computes the shared
system prompt and the append-only histories of our agents only once. That decides our prompt layout. Sampling and
structured output change the shape of one distribution per step. Grammar masks guarantee syntax, not correctness.

"Speculative decoding and quantization are per-workload levers. Speculation keeps the target distribution exactly,
and it helps at low load. FP8 weights and KV give prefill speed and concurrency. INT4 gives decode speed and memory.
We measure with open-loop load at realistic lengths, and we report goodput against the SLO."

**Drill.**

1. *Why do more requests in a decode batch barely slow it down, and when does that stop?* Below the knee, a step is
   the weight read, and every token in the step shares it. The knee is ~300 tokens on an H100 in bf16
   ($\text{peak} \times \mathtt{bytes\_per\_param} /$ $(2 \times \text{bandwidth})$). The effect stops when the
   tokens of the batch pass the knee. At long contexts, it stops earlier, because the KV read of each request adds
   bytes. 64 decodes at 2,048 tokens of context add 17.2 GB to the 15 GB of weights.
2. *p99 ITL has a spike whenever a long document arrives. Diagnose the cause and repair it.* This is prefill/decode
   interference. The step of the prompt is long, and every co-scheduled decode waits for it. The roofline model (§3)
   gives 367 ms against 17 ms on an L4 for an 8K prompt (SIMULATED). Make sure that chunked prefill is on. Decrease
   `max_num_batched_tokens` until the worst step (all decodes + one full chunk) meets the SLO. If prefill-heavy
   traffic still hurts decode, disaggregate prefill and decode (05).
3. *`vllm:num_preemptions` increases at peak and TTFT p99 is 5×. What occurs?* The KV of the requests that run becomes
   larger than the KV pool. Each preemption frees the blocks of the newest request, and the engine computes that
   request again later.
   The scheduler admits no new requests in that step. Add KV capacity (FP8 KV, utilisation, a smaller or quantized
   model), set a lower `max_num_seqs`, or add replicas. The whole-prompt admission check prevents over-admission with
   chunked prefill.
4. *Our agent's prefix-cache hit rate is 3%. The system prompt is 4K tokens and identical for everyone. Why?*
   Something before or inside it changes per request: a timestamp, a request id or a reordered tool list. The name
   of a block commits to every token before it. Thus one changed token early makes everything after it invalid. Move
   volatile content to the end. Keep the history append-only. Route a session back to the replica that holds its
   prefix.
5. *Does speculative decoding change model outputs? When do you turn it off?* No. Accept with $\min(1, p/q)$ and
   resample rejections from the normalised residual $\max(0, p - q)$. Then the emitted tokens have exactly the
   distribution of the target. Turn it off (or decrease $k$) at high concurrency with short contexts. There, the
   extra positions of the verify pass cost real compute. In the roofline simulation (§7), speculation becomes a
   slow-down past ~100–250 concurrent requests at 200-token contexts.
6. *INT4 or FP8 for an 8B model on a 24 GB L4? It must hold 48 concurrent 2K-token sessions and prefill 1.8K tokens
   in 300 ms.* Use FP8 W8A8 with an FP8 KV cache. It halves prefill (FP8 tensor cores) and doubles KV capacity. The
   roofline model gives a 181 ms prefill and 87 sessions (91 at vLLM's defaults, §8, SIMULATED). INT4 weight-only
   wins decode (~3×) and memory, but its prefill stays at 360 ms. The cause is that weight-only formats decrease
   bytes, not FLOPs. Then make the choice depend on task evals against bf16.

---

## Glossary

| Term | Meaning |
|---|---|
| Step (iteration) | one scheduling decision and one forward pass over the tokens that it selected |
| Engine core | the process that runs the scheduler, the KV cache manager and the step loop of the GPU workers |
| Continuous batching | a new batch in each step, so that requests join and leave at token granularity (Orca) |
| `num_computed_tokens` | the tokens of a request whose K/V are already in the cache. It is the only measure of progress for the scheduler |
| Token budget | `max_num_batched_tokens`: the maximum number of tokens that one step can process |
| Prefill / decode | the work on prompt tokens (many per step, compute-heavy) / the generation of one token per step (memory-bound) |
| Chunked prefill | a prompt divided across steps under the budget, so that decodes do not stop (Sarathi-Serve) |
| Knee | the tokens per step where a step changes from memory-bound to compute-bound: $\text{peak} \times \text{bytes/param} \div (2 \times \text{bandwidth})$ |
| Block / block table | the K/V of a fixed number of tokens (16 by default) / the map of a request from logical to physical blocks |
| Slot mapping | the place where the engine writes the K/V of each new token: $\text{block id} \times \text{block size} + \text{offset}$ |
| Preemption (recompute) | when memory runs out, the scheduler frees the blocks of a request that runs, and the engine prefills it again later |
| Watermark | blocks that the scheduler keeps free when it admits requests, so that the requests that run can grow |
| Prefix caching | the reuse of the K/V of full blocks whose chained hash matches the prefix of a new prompt |
| Block hash chain | $\operatorname{name}(\text{block } i) = H(\operatorname{name}(\text{block } i-1), \text{tokens}, \text{extra keys})$: commits to the whole prefix |
| Free queue (LRU) | reusable blocks, least recently freed first. Cached blocks keep their names until eviction |
| RadixAttention | SGLang's token-granular prefix cache kept as a radix tree |
| Top-p / min-p | keep the smallest set that reaches mass $p$ / keep tokens with prob ≥ min_p × the top token's |
| Structured output | a mask on the logits from a grammar automaton, so that every output parses |
| Speculative decoding | a low-cost draft of $k$ tokens, which one target pass verifies with exact rejection sampling |
| Acceptance rate $\alpha$ | $\sum \min(p, q) = 1 - \operatorname{TV}(p, q)$ for a sampled draft (${p(x)}$ for a greedy one): the probability that a draft token survives |
| EAGLE / MTP | draft heads on the target's hidden states / multi-token prediction heads trained with the model |
| Weight-only quantization | low-bit weights, dequantized inside the GEMM (W4A16, W8A16). It saves bytes, not FLOPs |
| W8A8 | weights and activations in 8 bits (FP8 or INT8). The tensor cores compute them natively |
| Group-wise scales | one scale per $g$ consecutive inputs of an output channel ($g$ = 32–128 for INT4) |
| TP / PP / EP / DP | tensor (divide each layer), pipeline (divide the layers), expert (divide MoE experts), data (replicas) parallelism |
| LoRA | a low-rank adapter ${W + BA}$. Many adapters can share one base model in one batch |
| TTFT / ITL / TPOT / E2E | time to first token / inter-token latency / time per output token / end-to-end latency |
| Goodput | requests per second that met every SLO |
| Open / closed loop | load at a fixed arrival rate / load from N users who wait for each response |

## Sources

Papers:

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022. Topic:
  iteration-level scheduling.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023
  (arXiv:2309.06180). Topic: vLLM.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024
  (arXiv:2403.02310). Topic: chunked prefill, stall-free batching.
- Zhong et al., *DistServe*, OSDI 2024 (arXiv:2401.09670). Topic: goodput, prefill/decode disaggregation.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024 (arXiv:2312.07104).
  Topic: RadixAttention.
- Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023
  (arXiv:2211.17192). Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*
  (arXiv:2302.01318). Sun et al., *Block Verification Accelerates Speculative Decoding* (arXiv:2403.10444).
- Li et al., *EAGLE* (arXiv:2401.15077) and *EAGLE-3* (arXiv:2503.01840). Cai et al., *Medusa* (arXiv:2401.10774).
  Saxena, *Prompt Lookup Decoding* (2023). DeepSeek-AI, *DeepSeek-V3 Technical Report* (arXiv:2412.19437). Topic:
  MTP.
- Chen et al., *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative
  Decoding* (arXiv:2408.11049).
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020. Topic: nucleus sampling. Nguyen et al.,
  *Min-p Sampling* (arXiv:2407.01082).
- Willard and Louf, *Efficient Guided Generation for LLMs* (arXiv:2307.09702). Topic: Outlines. Dong et al.,
  *XGrammar* (arXiv:2411.15100).
- Frantar et al., *GPTQ*, ICLR 2023 (arXiv:2210.17323). Lin et al., *AWQ*, MLSys 2024 (arXiv:2306.00978). Xiao et
  al., *SmoothQuant*, ICML 2023 (arXiv:2211.10438). Dettmers et al., *LLM.int8()* (arXiv:2208.07339). Micikevicius
  et al., *FP8 Formats for Deep Learning* (arXiv:2209.05433).
- Shoeybi et al., *Megatron-LM* (arXiv:1909.08053). Topic: tensor parallelism.
- Hu et al., *LoRA* (arXiv:2106.09685). Sheng et al., *S-LoRA* (arXiv:2311.03285). Chen et al., *Punica*
  (arXiv:2310.18547).

Code and documentation:

- vLLM, `github.com/vllm-project/vllm` (main, read 2026-09-26): `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`,
  `vllm/v1/sample/sampler.py`, `vllm/v1/sample/rejection_sampler.py`,
  `vllm/config/{cache,scheduler,speculative,structured_outputs}.py`, `vllm/engine/arg_utils.py`,
  `vllm/v1/metrics/loggers.py`. The docs are at docs.vllm.ai.
- SGLang `github.com/sgl-project/sglang`, TensorRT-LLM `github.com/NVIDIA/TensorRT-LLM`, llama.cpp
  `github.com/ggml-org/llama.cpp`.
- Google Cloud documentation: Cloud Run GPUs, GKE GPUs and TPUs, Vertex AI Model Garden.
- In this repo: the [kv-cache](../kv-cache/kv-cache-primer.md),
  [paged-attention](../paged-attention/paged-attention-primer.md) and
  [flash-attention](../flash-attention/flash-attention-primer.md) primers,
  [roofline-and-fabric](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md),
  [cuda-and-nccl](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md),
  [gpu-scheduling](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md),
  [gpu-capacity-planning](../../00-foundations/gpu-capacity-planning/PRIMER.md), the [transformer
  primer](../../00-foundations/transformers/docs/transformer-primer.md) and
  [agentic-scaling-lab](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) (admission, rate limits, cost
  per conversation).

## Verify list

This list holds the product facts in this primer, `minengine/perf.py` and the notebooks, as of 2026-09-26. The vLLM
items come from the source of the main branch, as read on that date. Examine them again against the release that you
pin.

- **vLLM defaults and behaviour:**
    - V1 as the architecture (separate API-server and engine-core processes, async scheduling),
    - `block_size` 16,
    - `enable_prefix_caching` true,
    - `prefix_caching_hash_algo` `sha256` (options `sha256_cbor`, `xxhash`, `xxhash_cbor`),
    - `cache_salt` in the extra keys of the first block,
    - `gpu_memory_utilization` 0.92 on main and in v0.30.0 (the core keeps 0.9, §4),
    - the L4's 22.49 GiB total as `nvidia-smi` reports it, and where CUDA's total and free figures sit below it,
    - `watermark` 0.0,
    - `scheduler_reserve_full_isl` true,
    - policies `fcfs` and `priority` (lower first),
    - API-server defaults for `max_num_batched_tokens` / `max_num_seqs` (2,048/256 below 70 GB or on A100,
      8,192/1,024 H100/H200-class, 16,384/1,024 at ≥160 GB),
    - chunked prefill on by default, and `max_num_batched_tokens ≥ max_model_len` necessary without it,
    - `long_prefill_token_threshold` (default 0 = off), `long_prefill_token_threshold_adaptive` and
      `max_num_active_seqs` in `SchedulerConfig` on main after 0.30.0, absent at the v0.30.0 tag (no partial-prefill
      cap on main),
    - preemption by recompute only in V1, and `kv_offloading_size` for CPU offload,
    - `max_cache_hit_length = num_tokens − 1`,
    - blocks cached inside `KVCacheManager.allocate_slots` (scheduling time),
    - prefix-cache stats recorded with a `preempted` flag, and preempted re-lookups kept out of
      `vllm:prefix_cache_queries/_hits`,
    - blocks freed tail first into an LRU free queue,
    - the start-up error when one `max_model_len` sequence cannot fit the KV cache,
    - `async_scheduling` on unless disabled,
    - `include_stop_str_in_output` false by default,
    - `RequestStatus` names (`FINISHED_STOPPED`, `FINISHED_LENGTH_CAPPED`, `FINISHED_ABORTED`, …).
- **vLLM sampling and outputs:**
    - the sampler order (allowed tokens/bad words/logit bias, then penalties, then greedy below 1e-5, then temperature,
      then min-p, then top-k/top-p),
    - `logprobs_mode` default `raw_logprobs`,
    - `top_k` 0 or −1 turns top-k off,
    - batch-invariant mode,
    - structured-output backends `auto`, `xgrammar`, `guidance`, `outlines`, `lm-format-enforcer`,
    - speculative methods (`ngram`, `suffix`, `draft_model`, `eagle`, `eagle3`, `medusa`, `mlp_speculator`, MTP
      variants),
    - `draft_sample_method` default `greedy` (or `probabilistic`),
    - `rejection_sample_method` default `standard` (also `synthetic`, `block`),
    - `vllm:spec_decode_num_accepted_tokens_per_pos`,
    - quantization methods (`awq`, `gptq`, `fp8`, `compressed-tensors`, `modelopt_fp4`, `mxfp4`, …),
    - `kv_cache_dtype` values (`fp8` = `fp8_e4m3`, `fp8_e5m2`) and default KV scales of 1.0,
    - LoRA flags `max_loras`, `max_cpu_loras`, `max_lora_rank`,
    - parallelism flags.
- **vLLM metric names** (FACTS, from `vllm/v1/metrics/loggers.py`), and `vllm:e2e_request_latency_seconds`, read
  from the same file.
- **GPU figures in `perf.GPUS`** (dense 16-bit tensor FLOP/s, memory bandwidth, capacity, FP8 support):
    - T4: 65 TFLOP/s, 320 GB/s, 16 GB,
    - L4: 121 TFLOP/s (242 sparse), 300 GB/s, 24 GB,
    - A100-80GB SXM: 312 TFLOP/s, 2.039 TB/s,
    - H100 SXM: 989 TFLOP/s, 3.35 TB/s, 80 GB.

    The efficiencies (60% FLOPs, 80% bandwidth), the 2 ms per-step overhead and the 0.5 ms per draft forward in
    `perf.spec_speedup()` are assumptions.

- **Quantized checkpoints** keep the embedding table and LM head in 16-bit. GPTQ/AWQ leave `lm_head` unquantized,
  and FP8 compressed-tensors checkpoints list it under `ignore`. Also examine the published sizes of specific
  checkpoints.
- **Model configs in `perf.LLMS`:**
    - Qwen2.5-0.5B (24 layers, 14 heads, 2 KV heads, head_dim 64, tied),
    - Qwen3-0.6B (28, 16, 8, 128, tied),
    - Qwen2.5-1.5B (28, 12, 2, 128, tied),
    - Llama-3.2-1B (16, 32, 8, 64, tied),
    - Llama-3.1-8B (32, 32, 8, 128, untied, vocab 128,256),
    - the 70B figures (80 layers, d = 8,192, 70.6 B parameters).
- **Transfer rates:** ~50 GB/s effective per direction for PCIe Gen5 x16 (~25 GB/s Gen4) in the swap arithmetic.
- **Where to run:**
    - Cloud Run GPU types (L4, RTX PRO 6000 Blackwell), per-second billing and scale to zero (FACTS),
    - Vertex AI Model Garden's vLLM-based serving,
    - TPU v7 "Ironwood" GA 2026-04-22 (FACTS), and the name and status of vLLM's TPU backend,
    - T4 has no bf16 and no FP8, and L4 and H100 have FP8,
    - Colab/Kaggle/RunPod/Vast.ai/Lambda offers and prices ([`COMPUTE.md`](../../COMPUTE.md) maintains them).
- **Engine summaries in §12:** SGLang, TensorRT-LLM and llama.cpp feature claims.
