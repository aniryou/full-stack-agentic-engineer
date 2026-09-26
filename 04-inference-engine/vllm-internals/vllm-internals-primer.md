# vLLM internals: a source-level primer

This primer walks through how vLLM actually runs a request, with every mechanism tied to the file
and function that implements it. It is written for an engineer who has to explain an inference
engine in a design review: why a request waited, where its time went, why the KV cache holds as
many tokens as it does, and which flag moves which number. The engine concepts themselves
(continuous batching, chunked prefill, prefix caching, speculation) are introduced in the
serving-engine primer ([`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md)); this document
is the "now read the real code" companion. To run what is described here on a GPU, use
[`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/).

**State of the source.** vLLM `main` at commit `5840d95` (2026-09-25), fetched 2026-09-26 from
`raw.githubusercontent.com/vllm-project/vllm/main`. The package version comes from git tags via
`setuptools_scm` (`vllm/version.py`, `pyproject.toml`), so `main` reports no fixed number; the
latest PyPI release at fetch time was **0.30.0** (2026-09-22), and `pyproject.toml` pins
`torch == 2.13.0` for the build. `main` moves daily: file paths in this document were checked at
that commit, and anything not confirmed in source is marked `(verify)`.

**Citation convention.** `(path: Class.method)` means "read it there". Paths are relative to the
vLLM repository root. [`source-map.md`](source-map.md) lists the same files with line numbers at
`5840d95` and a three-hour reading order.

**Tier.** Reading this and the source is **T0** (laptop, no GPU). Observing the behaviour it
describes (metrics, log lines, preemptions) is **T1** in the serving lab.

---

## The one-minute version

vLLM splits serving into two kinds of processes. An **API server** process (FastAPI on asyncio)
renders the chat template, tokenizes, builds an `EngineCoreRequest`, and later detokenizes and
streams. An **EngineCore** process runs a tight loop: `Scheduler.schedule()` picks how many tokens
each request computes this step, the executor runs one forward pass on the GPU workers,
the sampler picks tokens, and `Scheduler.update_from_output()` appends them and checks stop
conditions. The two sides talk over ZMQ sockets with msgpack, so tokenization and HTTP never steal
time from the GPU loop.

The scheduler has no "prefill phase" and no "decode phase". Each request has `num_computed_tokens`
and a target length; every step hands out a token budget (`max_num_batched_tokens`) first to
running requests, then to waiting ones. A decode is a request that needs 1 token; a prefill is one
that needs many and gets chunked to fit. Chunked prefill and prefix caching are on by default.

The KV cache is a pool of fixed-size blocks (16 tokens by default). Full blocks are identified by
a hash chained over the whole prefix, so any request whose prompt starts with the same tokens
reuses them; freed blocks stay cached until reallocated, evicted in LRU order with a chain's tail
going first. The pool size is whatever is left of `gpu_memory_utilization × total memory` after
weights, the activation peak of a profiling forward pass, and CUDA graphs.

On the GPU side, the model is compiled with `torch.compile` into pieces split at attention, and
replayed as CUDA graphs: full graphs for pure-decode batches, piecewise graphs for mixed ones.
Attention is a pluggable backend (FlashAttention, FlashInfer, Triton, MLA variants) chosen per GPU
generation. After this primer you should be able to trace a request through those classes, compute
a KV block budget by hand, predict when preemption happens, and say which engine argument trades
TTFT against inter-token latency against memory.

---

## 1. Why vLLM and what "V1" changed

### 1.1 The problem

An LLM server is bound by two resources: HBM bandwidth during decode (every step re-reads the
weights and the KV cache) and HBM capacity for the KV cache (it decides how many sequences can be
in flight). The KV-cache arithmetic and the paging idea are in
[`../kv-cache/kv-cache-primer.md`](../kv-cache/kv-cache-primer.md) and
[`../paged-attention/paged-attention-primer.md`](../paged-attention/paged-attention-primer.md);
vLLM began as the reference implementation of PagedAttention (Kwon et al., SOSP 2023). What the
engine adds on top of paging is a scheduler that keeps the batch full, a cache that shares blocks
across requests, and a GPU execution path with almost no per-step host overhead.

### 1.2 What V1 is

"V1" is the engine re-architecture that is now the only engine in the tree: the old V0 code paths
are gone, the `VLLM_USE_V1` environment variable no longer exists in `vllm/envs.py`, and the
remaining mentions of V0 are comparisons in docstrings (for example "This is different from the V0
sampler", `vllm/v1/sample/sampler.py: Sampler.forward`). The release in which V0 was removed is
`(verify)`. The design choices that define V1:

| Choice | What it means | Where |
|---|---|---|
| Process split | API server(s) and EngineCore in separate processes, ZMQ + msgpack between them | `vllm/v1/engine/core_client.py: AsyncMPClient`, `vllm/v1/engine/core.py: EngineCoreProc` |
| asyncio front-end | `AsyncLLM` owns tokenization, detokenization, streaming; a background task pulls outputs | `vllm/v1/engine/async_llm.py: AsyncLLM._run_output_handler` |
| Unified scheduler | no prefill/decode phases; requests catch `num_computed_tokens` up to their length | comment at top of `vllm/v1/core/sched/scheduler.py: Scheduler.schedule` |
| Chunked prefill on | default for decoder-only generative models | `vllm/engine/arg_utils.py: EngineArgs._set_default_chunked_prefill_and_prefix_caching_args` |
| Prefix caching on | same; `CacheConfig.enable_prefix_caching = True` | `vllm/config/cache.py: CacheConfig` |
| Async scheduling on | schedule step N+1 while step N runs on the GPU | `vllm/config/vllm.py` (async-scheduling resolution in `VllmConfig`), `vllm/v1/core/sched/async_scheduler.py` |
| Persistent batch / GPU-resident request state | per-request state kept on device across steps, only deltas sent | `vllm/v1/worker/gpu_input_batch.py: InputBatch`, `vllm/v1/worker/gpu/states.py: RequestState` |
| torch.compile + CUDA graphs | `CompilationMode.VLLM_COMPILE`, `CUDAGraphMode.FULL_AND_PIECEWISE` by default | `vllm/config/compilation.py` |
| Symmetric workers | the scheduler lives in EngineCore, not in rank-0 worker; all workers receive the same `SchedulerOutput` | `vllm/v1/executor/multiproc_executor.py: MultiprocExecutor.collective_rpc` |

### 1.3 The process map

```
  clients ──HTTP──▶ API server process(es)            EngineCore process                     GPU worker process(es)
                    (uvloop + FastAPI)                (one per data-parallel rank)           (one per TP×PP rank; none if TP=PP=1)
                    ┌──────────────────────────┐     ┌──────────────────────────────────┐    ┌──────────────────────────┐
                    │ OpenAIServingChat        │     │ input thread  (ZMQ DEALER recv)  │    │ Worker                   │
                    │ AsyncLLM                 │ ZMQ │   preprocess_add_request         │    │  GPUModelRunner          │
                    │  InputProcessor ─────────┼────▶│ busy loop (main thread)          │shm │   (V1 or V2 runner)      │
                    │  OutputProcessor ◀───────┼─────│   Scheduler + KVCacheManager     │───▶│  model (compiled)        │
                    │  output_handler task     │     │   Executor ──────────────────────┼───▶│  attention backend       │
                    └──────────────────────────┘     │ output thread (ZMQ PUSH send)    │    │  Sampler                 │
                                                     └──────────────────────────────────┘    └──────────────────────────┘
```

- With one GPU the executor is `UniProcExecutor` and the worker runs **inside** the EngineCore
  process; with TP×PP > 1 on one node it is `MultiprocExecutor` with one process per GPU; `ray`
  is chosen for placement groups or when requested (`vllm/config/parallel.py:
  ParallelConfig.__post_init__`, `vllm/v1/executor/abstract.py: Executor.get_class`).
- `vllm serve` runs one API server process by default; with internal data-parallel load balancing
  it defaults `--api-server-count` to the DP size, and `--headless` runs engines with no API server
  (`vllm/entrypoints/cli/serve.py: ServeSubcommand.cmd`).
- Inside EngineCore, socket I/O and msgpack (de)serialization run on two daemon threads so they
  overlap the GPU step (`vllm/v1/engine/core.py: EngineCoreProc.__init__`, `process_input_sockets`,
  `process_output_sockets`).

### 1.4 Which model runner

There are two GPU model runners on `main`. The long-standing one is
`vllm/v1/worker/gpu_model_runner.py: GPUModelRunner` ("MRV1": persistent `InputBatch`, CPU-built
inputs). The newer one is `vllm/v1/worker/gpu/model_runner.py: GPUModelRunner` ("MRV2": request
state in fixed GPU slots, Triton kernels build the inputs). Its README still says
"[Experimental]" (`vllm/v1/worker/gpu/README.md`), but `VllmConfig.use_v2_model_runner`
(`vllm/config/vllm.py`) returns **True by default** when Triton is available and none of the
unsupported features are requested; it falls back to MRV1 for, among others, the `ngram`,
`draft_model`, `suffix`, `medusa` speculative methods, stock `torch.compile` mode, and sequence
parallelism (`VllmConfig._get_v2_model_runner_unsupported_features`). `VLLM_USE_V2_MODEL_RUNNER=0|1`
forces the choice (`vllm/envs.py`). The startup log says which one you got
("Model Runner V2 does not yet support ...; using the V1 model runner instead"). Section 5 covers
both; most concepts (slot mapping, attention metadata, the execute/sample split) are shared.

---

## 2. A request's life, end to end

### 2.1 Sequence diagram

A streaming `POST /v1/chat/completions` on a single-GPU `vllm serve`:

```
client      API server process                                EngineCore process                        GPU (same process when TP=1)
  │ POST /v1/chat/completions (stream=true)
  ├────────▶ create_chat_completion           [chat_completion/api_router.py]
  │          OpenAIServingChat._create_chat_completion [chat_completion/serving.py]
  │           ├ render_chat_request → OnlineRenderer.render_chat   (chat template + tokenize → EngineInput)
  │           ├ request.to_sampling_params(max_tokens, defaults)
  │           └ engine_client.generate(...)  = AsyncLLM.generate  [v1/engine/async_llm.py]
  │               AsyncLLM.add_request
  │                ├ InputProcessor.process_inputs → EngineCoreRequest   [v1/engine/input_processor.py]
  │                ├ OutputProcessor.add_request (RequestState + IncrementalDetokenizer)
  │                └ AsyncMPClient.add_request_async ──ZMQ ROUTER, (ADD, msgpack)──▶ input thread: process_input_sockets
  │                                                                          preprocess_add_request → Request
  │                                                                            (block hashes computed here)
  │                                                                          input_queue.put
  │                                                                        busy loop: run_busy_loop
  │                                                                          _process_input_queue → Scheduler.add_request (waiting)
  │                                                                          step_with_batch_queue (async scheduling):
  │                                                                           Scheduler.schedule() → SchedulerOutput
  │                                                                           Executor.execute_model(so, non_block=True) ──▶ Worker.execute_model
  │                                                                           Scheduler.get_grammar_bitmask(so)               GPUModelRunner.execute_model:
  │                                                                             (CPU, overlaps the forward)                    _update_states, _prepare_inputs,
  │                                                                                                                            attention metadata, forward,
  │                                                                                                                            compute_logits (kept on GPU)
  │                                                                           Executor.sample_tokens(grammar) ─────────────▶ GPUModelRunner.sample_tokens:
  │                                                                                                                            apply bitmask, Sampler or
  │                                                                           ◀──────────── ModelRunnerOutput ──────────────── RejectionSampler, draft proposal
  │                                                                           Scheduler.update_from_output → EngineCoreOutputs
  │                                                                             (append tokens, check_stop, free finished)
  │                                                                          output_queue → output thread: process_output_sockets
  │                ◀────────────────── ZMQ PUSH→PULL, EngineCoreOutputs (msgpack) ─────
  │               output_handler task: AsyncMPClient.get_output_async
  │                OutputProcessor.process_outputs: detokenize (IncrementalDetokenizer.update),
  │                  stop strings, logprobs, RequestOutput → per-request RequestOutputCollector
  │                  (stop string hit → engine_core.abort_requests_async)
  │               AsyncLLM.generate yields RequestOutput
  │              chat_completion_stream_generator → "data: {json}\n\n"
  │◀──────── StreamingResponse(media_type="text/event-stream") ... "data: [DONE]\n\n"
```

### 2.2 The same path as a table

| # | Process | Class.function | What happens | Cost that shows up in |
|---|---|---|---|---|
| 1 | API | `create_chat_completion` (`vllm/entrypoints/openai/chat_completion/api_router.py`) | FastAPI route; returns `StreamingResponse` with SSE keep-alive for streams | HTTP latency |
| 2 | API | `OpenAIServingChat.render_chat_request` → `OnlineRenderer.render_chat` (`vllm/entrypoints/openai/chat_completion/serving.py`) | chat template + tokenization via the Renderer (`vllm/renderers/`) | front-end CPU |
| 3 | API | `AsyncLLM.add_request` (`vllm/v1/engine/async_llm.py`) | raw prompts go through `InputProcessor.process_inputs_async`, which runs on the renderer's thread pool so the event loop is not blocked (`vllm/v1/engine/input_processor.py: InputProcessor.__init__`) | TTFT |
| 4 | API | `InputProcessor.process_inputs` | validates params and LoRA, clones `SamplingParams`, sets `max_tokens = max_model_len − prompt_len` if unset, applies generation-config defaults, sorts multimodal features, returns `EngineCoreRequest` (a `msgspec.Struct`, `array_like=True`) | TTFT |
| 5 | API | `AsyncLLM.add_request` | `n > 1` fans out into `n` child requests with a `ParentRequest` (`vllm/v1/engine/parallel_sampling.py`) | — |
| 6 | API→Core | `AsyncMPClient._send_input` (`vllm/v1/engine/core_client.py`) | ROUTER socket, frames = (request type byte, msgpack payload) | IPC |
| 7 | Core, input thread | `EngineCoreProc.process_input_sockets` → `EngineCore.preprocess_add_request` (`vllm/v1/engine/core.py`) | decode, build `Request` via `Request.from_engine_core_request`, which computes the prompt's block hashes; start async grammar compile for structured output | runs while the GPU steps |
| 8 | Core, main thread | `EngineCoreProc.run_busy_loop` → `_process_input_queue` → `Scheduler.add_request` | request enters `waiting`; `QUEUED` event timestamped | queue time |
| 9 | Core | `Scheduler.schedule` (`vllm/v1/core/sched/scheduler.py`) | token budget, prefix-cache lookup, block allocation, preemption; builds `SchedulerOutput` | queue time, TTFT |
| 10 | Core→GPU | `Executor.execute_model(scheduler_output, non_block=True)` | forward pass up to logits; returns a future | step time |
| 11 | Core | `Scheduler.get_grammar_bitmask` | structured-output bitmask computed on CPU while the GPU runs step 10 (`vllm/v1/engine/core.py: EngineCore.step`) | hidden |
| 12 | GPU | `GPUModelRunner.sample_tokens(grammar_output)` | bitmask applied to logits, sampling or rejection sampling, drafts proposed | step time |
| 13 | Core | `Scheduler.update_from_output` | appends tokens, rolls back rejected drafts, `check_stop` (EOS, stop ids, length), frees finished requests, builds `EngineCoreOutputs` per client | step time |
| 14 | Core, output thread | `EngineCoreProc.process_output_sockets` | msgpack encode into reused buffers, ZMQ PUSH | overlapped |
| 15 | API | `AsyncLLM._run_output_handler` → `OutputProcessor.process_outputs` (`vllm/v1/engine/output_processor.py`) | the only loop over all outputs: stats, incremental detokenize, stop-string check, logprobs, push `RequestOutput` to the request's queue; processed in chunks of `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` with `await asyncio.sleep(0)` between chunks | ITL jitter under load |
| 16 | API | `OpenAIServingChat.chat_completion_stream_generator` | formats SSE deltas, ends with `data: [DONE]` | — |

Two consequences worth stating in a review. First, **stop strings are detected in the API server,
not in the engine**: `IncrementalDetokenizer.update` returns the matched stop string
(`vllm/v1/engine/detokenizer.py`), `OutputProcessor.process_outputs` adds the request to
`reqs_to_abort`, and `AsyncLLM` sends an abort to the engine. The engine may therefore compute a
token or two past a stop string; EOS, `stop_token_ids`, `max_tokens` and `max_model_len` are
checked inside the engine by `check_stop` (`vllm/v1/core/sched/utils.py`). Second, **a client
disconnect is an abort**: when the SSE generator is cancelled, `AsyncLLM.generate` catches
`asyncio.CancelledError`/`GeneratorExit` and calls `self.abort`, which frees the request's blocks.

### 2.3 The four messages that matter

| Message | Direction | Key fields | Defined in |
|---|---|---|---|
| `EngineCoreRequest` | API → Core | `request_id`, `prompt_token_ids`, `mm_features`, `sampling_params`, `lora_request`, `cache_salt`, `priority`, `arrival_time`, `data_parallel_rank`; `kv_transfer_params` rides in `sampling_params.extra_args` and is lifted out in `Request.__init__` (`vllm/v1/request.py`) | `vllm/v1/engine/__init__.py` |
| `SchedulerOutput` | Core → workers | `scheduled_new_reqs` (full data once), `scheduled_cached_reqs` (deltas: new token ids, new block ids), `num_scheduled_tokens` per request, `total_num_scheduled_tokens`, `scheduled_spec_decode_tokens`, `scheduled_encoder_inputs`, `num_common_prefix_blocks`, `finished_req_ids`, `preempted_req_ids`, `kv_connector_metadata` | `vllm/v1/core/sched/output.py` |
| `ModelRunnerOutput` | workers → Core | `req_ids`, `req_id_to_index`, `sampled_token_ids` (list per request, several when drafts are accepted), `logprobs`, `prompt_logprobs_dict`, `kv_connector_output` | `vllm/v1/outputs.py` |
| `EngineCoreOutputs` | Core → API | per request: `new_token_ids`, `finish_reason` (`STOP`/`LENGTH`/`ABORT`/`ERROR`/`REPETITION`), `new_logprobs`, `events` (QUEUED/SCHEDULED/PREEMPTED timestamps), `kv_transfer_params`; per batch: `scheduler_stats`, `timestamp` | `vllm/v1/engine/__init__.py` |

`SchedulerOutput` is deliberately a diff: after a request's first step, workers get only its new
tokens and new block ids, because they keep the rest in their persistent state (Section 5).

### 2.4 Where TTFT goes

The front-end measures TTFT from `arrival_time` (set in `InputProcessor.process_inputs`) to the
iteration timestamp at which the first token is processed (`vllm/v1/metrics/stats.py:
IterationStats.update_from_output`). The engine-side events split it further
(`IterationStats.update_from_finished_request`):

```
arrival ──render/tokenize──▶ QUEUED ──queue──▶ SCHEDULED ──prefill (all chunks)──▶ first token ──decode──▶ last token
          (front-end CPU)            vllm:request_queue_time_seconds    vllm:request_prefill_time_seconds      vllm:request_decode_time_seconds
◀──────────────────────────── vllm:time_to_first_token_seconds ────────────────────▶
```

`vllm:request_time_per_output_token_seconds` is `decode_time / (num_generation_tokens − 1)` per
request; `vllm:inter_token_latency_seconds` is observed per iteration. If TTFT is high and queue time
is high, the engine is saturated (see Section 3); if TTFT is high and prefill time is high, prompts
are long or chunked behind other prefills; if both are low, look at the front-end (tokenization,
multimodal preprocessing, `--api-server-count`).

---

## 3. The scheduler

### 3.1 One idea: tokens to compute

The comment at the top of `Scheduler.schedule` states the whole design
(`vllm/v1/core/sched/scheduler.py`):

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the
> num_computed_tokens and num_tokens_with_spec. [...] At each step, the scheduler tries to assign
> tokens to the requests so that each request's num_computed_tokens can catch up its
> num_tokens_with_spec.

`num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)`
(`vllm/v1/request.py: Request.num_tokens_with_spec`). A fresh request is 3,000 tokens behind; a
decoding request is 1 behind (the token sampled last step); a request with 3 draft tokens is 4
behind. Chunked prefill, prefix caching (which advances `num_computed_tokens` without compute) and
speculative decoding are all the same operation.

### 3.2 The budgets

| Budget | Source | Default for `vllm serve` | Enforced in |
|---|---|---|---|
| tokens per step | `--max-num-batched-tokens` (`SchedulerConfig.max_num_batched_tokens`); `max_num_scheduled_tokens` defaults to it | 2048 on GPUs under 70 GiB and on A100; 8192 on ≥70 GiB non-A100 (H100/H200); 16384 on ≥160 GiB (B200/B300); doubled by `--performance-mode throughput` | `Scheduler.schedule` (`token_budget`) |
| requests in RUNNING | `--max-num-seqs`, optionally lowered for admission only by `--max-num-active-seqs` | 256 below 70 GiB / A100; 1024 above | `Scheduler.schedule` waiting loop |
| per-request chunk cap | `--long-prefill-token-threshold` | 0 (off); ignored when only one request is eligible | `Scheduler.schedule` |
| encoder tokens per step | multimodal budget (`MultiModalBudget.encoder_compute_budget`) | derived | `Scheduler._try_schedule_encoder_inputs` |
| distinct LoRAs per step | `--max-loras` | 1 | waiting loop |

The GPU-dependent defaults come from `EngineArgs.get_batch_defaults` (`vllm/engine/arg_utils.py`),
which keys on device memory and name; the comment there notes that a large
`max_num_batched_tokens` reduced A100 throughput, hence the A100 exception. `SchedulerConfig`
itself requires `max_num_batched_tokens >= max_num_seqs`, and, when chunked prefill is off,
`max_num_batched_tokens >= max_model_len` (`vllm/config/scheduler.py:
SchedulerConfig.verify_max_model_len`).

### 3.3 The algorithm

Condensed from `Scheduler.schedule` (names as in the source; encoder, Mamba-alignment, KV-connector
and DP-balancing details omitted):

```
token_budget = max_num_scheduled_tokens
lpt = long_prefill_token_threshold if len(running)+len(waiting)+len(skipped_waiting) > 1 else 0

# (1) RUNNING requests, in list order
for req in running:
    n = req.num_tokens_with_spec + req.num_output_placeholders - req.num_computed_tokens
    if 0 < lpt < n:  n = lpt
    n = min(n, token_budget, max_model_len - req.num_computed_tokens - 1)
    if n == 0: continue                       # skip, do not stop (not strict FCFS)
    while (new_blocks := kv_cache_manager.allocate_slots(req, n, num_lookahead_tokens)) is None:
        victim = running[-1]                  # FCFS
        #   or  max(running, key=(priority, arrival_time))   with --scheduling-policy priority
        _preempt_request(victim)              # free blocks, num_computed_tokens=0, prepend to waiting
        if victim is req: break
    if new_blocks is None: break
    schedule(req, n); token_budget -= n

# (2) WAITING requests, only if nothing was preempted in (1)
if not preempted_reqs:
    while (waiting or skipped_waiting) and token_budget > 0 and len(running) < max_num_active_seqs:
        req = queue.peek_request()            # deque (fcfs) or heap on (priority, arrival, id)
        if blocked (grammar compiling, remote KV pending, LoRA cap reached): move to skipped; continue
        if req.num_computed_tokens == 0:
            hit_blocks, hit_tokens = kv_cache_manager.get_computed_blocks(req)
            (+ connector.get_num_new_matched_tokens for external hits)
        n = req.num_tokens - computed;  apply lpt;  n = min(n, token_budget)
        new_blocks = kv_cache_manager.allocate_slots(req, n, hit_tokens, hit_blocks,
                                                     full_sequence_must_fit=True, ...)
        if new_blocks is None: break          # head of the queue waits; nothing behind it jumps ahead
        running.append(req); schedule(req, n); token_budget -= n

output = SchedulerOutput(...);  _update_after_schedule(output)   # advance num_computed_tokens now
```

Properties that fall out of this code and matter in reviews:

- **Decodes first.** Running requests (mostly decodes) take budget before any new prefill, so
  admitting work never starves in-flight streams.
- **Chunking is automatic.** A waiting request gets `min(remaining, token_budget)` tokens; a long
  prompt is cut to whatever budget is left after the decodes.
- **Not strictly FCFS.** A running request that cannot take tokens this step is skipped with
  `continue` ("we do not strictly follow the FCFS scheduling policy", comment in `schedule`); in the
  waiting pass, requests blocked on grammar compilation or remote KV are parked in `skipped_waiting`.
  But a waiting request that simply does not fit stops the waiting pass (`break`).
- **No admissions in a preempting step.** `if not preempted_reqs` guards the waiting pass.
- **Priority ordering.** `PriorityRequestQueue` is a heap ordered by `Request.__lt__`:
  lower `priority` value first, then earlier `arrival_time`, then `request_id`
  (`vllm/v1/core/sched/request_queue.py`, `vllm/v1/request.py`).
- **`num_computed_tokens` advances at schedule time**, not after execution
  (`Scheduler._update_after_schedule`), so the next step can schedule the next chunk immediately;
  rejected draft tokens are subtracted later in `update_from_output`.

### 3.4 Chunked prefill and `long_prefill_token_threshold`

Chunked prefill is not a separate code path: it is the `min(n, token_budget)` above. The Sarathi-Serve
argument for it (bounded step time, so decodes co-scheduled with a prefill keep a steady ITL) is in
the serving-engine primer. Two knobs shape the chunks:

- `--max-num-batched-tokens` sets the step size. A step of 2,048 tokens on an 8B model is a few
  tens of milliseconds on an H100 `(verify: measure)`; every decode sharing that step waits for it.
- `--long-prefill-token-threshold` caps any single request's chunk. With the default 0 there is no
  cap, and one long prompt can take the whole remaining budget; with, say, 512, two concurrent long
  prompts each advance 512 per step instead of one blocking the other. The cap is dropped when the
  request is the only eligible one ("there is nobody to starve"), and
  `--long-prefill-token-threshold-adaptive` floors it at `max_num_batched_tokens / num_requests`
  (`SchedulerConfig.long_prefill_token_threshold_adaptive`).

When a prefill chunk ends mid-prompt the model still computes logits for it, but the sampled token is
discarded: the runner marks such rows in `discard_request_mask` (`vllm/v1/worker/gpu_model_runner.py:
GPUModelRunner._prepare_inputs`) and `update_from_output` receives no tokens for them.

### 3.5 Worked example: four requests through six steps

Setup: `vllm serve` on an L4, so `max_num_batched_tokens = 2048`, `max_num_seqs = 256`,
`block_size = 16`, `long_prefill_token_threshold = 0`, empty prefix cache, enough free blocks.
Request A (3,000-token prompt) and B (500) arrive together; C (6,000) arrives before step 3.
Blocks are `ceil(tokens / 16)`: A needs 188, B 32, C 375.

| Step | Running pass (decodes first) | Waiting pass | Tokens | New blocks | Who samples a real token |
|---|---|---|---|---|---|
| 1 | — | A: `min(3000, 2048) = 2048` (admission checked all 188 blocks fit: `full_sequence_must_fit`) → B does not fit (budget 0) | A 2048 | A +128 | nobody (A mid-prefill, token discarded) |
| 2 | A: `3000 − 2048 = 952` | B: 500 fits in 1,096 left | A 952, B 500 = 1,452 | A +60, B +32 | A and B (first tokens: TTFT) |
| 3 | A: 1, B: 1 | C: `min(6000, 2046) = 2046` | 2,048 | C +128 (A's token at position 3000 lands in block 187, already allocated; B's in block 31) | A, B |
| 4 | A: 1, B: 1, C: `min(3954, 2046) = 2046` | — | 2,048 | C +128 | A, B |
| 5 | A: 1, B: 1, C: `6000 − 4092 = 1908` | — | 1,910 | C +119 | A, B, C (C's TTFT: 3 steps) |
| 6 | A: 1, B: 1, C: 1 | — | 3 | 0 | A, B, C (pure decode → full CUDA graph, padded to size 4) |

Read it the way a reviewer would. A and B keep producing one token per step while C prefills, but
steps 3–5 each carry ~2,048 tokens, so A's and B's inter-token latency during C's prefill is the time
of a 2,048-token step rather than a 6,000-token one. Raising `max_num_batched_tokens` to 8,192 would
let C finish in one step (better TTFT for C) and make that one step four times longer for A and B
(worse ITL). That is the TTFT-versus-ITL dial.

### 3.6 Admission control inside the engine

`allocate_slots` refuses a waiting request unless its **whole** sequence fits, not just its first
chunk: the scheduler passes `full_sequence_must_fit=self.scheduler_reserve_full_isl` (default
`True`, `SchedulerConfig.scheduler_reserve_full_isl`), and `KVCacheManager.allocate_slots` checks
`get_num_blocks_to_allocate(num_tokens=min(request.num_tokens, max_model_len))` plus the watermark
against `BlockPool.get_num_free_blocks()`. Prefix-cache hits that sit in the free queue count as
needed capacity, because touching them removes them from the free pool
(`vllm/v1/core/single_type_kv_cache_manager.py: SingleTypeKVCacheManager.get_num_blocks_to_allocate`).
`--watermark` (default 0.0) keeps `int(watermark × num_blocks)` blocks free when admitting waiting or
preempted requests, but only when something else is already running
(`KVCacheManager.__init__`, `allocate_slots`). A request that never fits waits in the queue: the
engine itself has no queue bound. The bounds live in the API server: `--max-num-queued-reqs` and
`--max-num-queued-tokens` reject with HTTP 503 (`vllm/config/scheduler.py`, docstrings of those fields).

### 3.7 Preemption: who, how, and what it costs

When a running request needs a block and none is free, the scheduler preempts until the allocation
succeeds (Section 3.3). Mechanics, from `Scheduler._preempt_request`:

1. Victim: `running[-1]` under FCFS (the most recently admitted or resumed), or the maximum of
   `(priority, arrival_time)` under priority scheduling; if the victim had already been scheduled
   this step its tokens and blocks are returned to the budget.
2. `_free_request_blocks` returns all its blocks; encoder-cache entries are freed.
3. `status = PREEMPTED`, `num_computed_tokens = 0`, draft tokens dropped, `num_preemptions += 1`,
   a `PREEMPTED` event is recorded, and the request is **prepended** to `waiting`.

There is no swap-to-CPU path in the V1 scheduler; `CacheConfig` has no `swap_space` field. The
preempted request keeps its token ids and recomputes. With prefix caching on, "recompute" is usually
cheap: its full blocks went back to the free queue **with their hashes** (Section 4.6), so on
re-admission `get_computed_blocks` finds them again unless they were reallocated in the meantime.

Worked example (`--num-gpu-blocks-override 300`, which `CacheConfig` documents as "used for testing
preemption"; 299 usable blocks because block 0 is the null block):

| Moment | R1 (2,000-token prompt) | R2 (2,000-token prompt) | Free blocks |
|---|---|---|---|
| both prefilled | 125 blocks | 125 blocks | 49 |
| each decodes; a new block every 16 tokens | +24 blocks | +24 blocks | 1 |
| next 16-token boundary | takes the last block | `allocate_slots` → `None`; R2 is `running[-1]`, preempts itself | 0 |
| after preemption | running | ~149 blocks freed (full ones keep hashes), back at the head of `waiting` | ~149 |
| next step | decodes | prefix hit on its own ~148 blocks; admitted if the full sequence fits; recomputes ~1 block | ~0 |
| 16 tokens later | needs a block | needs a block | ping-pong |

The pair thrashes every 16 tokens and `vllm:num_preemptions` climbs. The fixes are capacity
(FP8 KV, fewer concurrent sequences, shorter `max_model_len`, more GPUs) or headroom
(`--watermark`), not a bigger token budget.

### 3.8 Async scheduling and the batch queue

With async scheduling (the default when compatible, resolved in `VllmConfig`; turned off by default
for pooling models, for speculative methods outside the EAGLE/MTP family, `ngram_gpu`,
`draft_model`, DFlash and DSpark, and for executors that do not support it), `SchedulerConfig.get_scheduler_cls` returns `AsyncScheduler` and
`VllmConfig.max_concurrent_batches` is 2 (V1 runner, PP=1) or `pp_size + 1` (V2 runner). EngineCore
then uses `step_with_batch_queue` instead of `step` (`vllm/v1/engine/core.py: EngineCore.__init__`):

```
step N:   schedule(N) ─▶ execute_model(N) (non-blocking) ─▶ sample_tokens(N) (non-blocking) ─▶ return early
step N+1: schedule(N+1) while GPU runs N ─▶ enqueue ─▶ block on N's result ─▶ update_from_output(N)
```

The scheduler must schedule a decode for a token it has not seen yet. `AsyncScheduler._update_after_schedule`
adds `num_output_placeholders` for the token(s) the step will produce and sets placeholder draft ids
of `-1`; the model runner reads the real token ids from the previous step's GPU tensor
(`vllm/v1/worker/gpu_model_runner.py: GPUModelRunner._prepare_input_ids`, `prev_sampled_token_ids`).
When outputs arrive, `AsyncScheduler._update_request_with_output` decrements the placeholders and
calls `kv_cache_manager.cache_blocks`. The cost: a request that stops on EOS may already have
one more step scheduled for it, whose output is dropped; the running loop skips that extra step when
the stop is predictable from `max_tokens` (the placeholder check at the top of the running loop). The benefit is that CPU scheduling time disappears from the step time.

### 3.9 `update_from_output`

For each request in the step (`Scheduler.update_from_output`):

- Speculative decoding: `num_rejected = num_draft − num_accepted`, and `num_computed_tokens` (and
  placeholders) roll back by that amount; the step's KV writes for rejected positions are simply
  overwritten later.
- `_update_request_with_output` appends tokens one by one and calls `check_stop`
  (`vllm/v1/core/sched/utils.py`): EOS → `FINISHED_STOPPED`; a `stop_token_ids` hit →
  `FINISHED_STOPPED` with `stop_reason`; `num_tokens >= max_model_len` or
  `num_output_tokens >= max_tokens` → `FINISHED_LENGTH_CAPPED`; optional repetition detection →
  `FINISHED_REPETITION`. `min_tokens` is enforced earlier by a logits processor that masks stop
  tokens (Section 7.1).
- The structured-output grammar advances with `accept_tokens`; a rejection terminates the request
  with `FINISHED_ERROR`.
- Finished requests are freed (`_free_request`); a KV connector may delay the free
  (Section 10), in which case `kv_transfer_params` ride back on the output.

---

## 4. KV cache management

### 4.1 Object model

```
Scheduler
 └─ KVCacheManager                      vllm/v1/core/kv_cache_manager.py
     └─ KVCacheCoordinator              vllm/v1/core/kv_cache_coordinator.py
         │   Unitary (1 group) | Hybrid (several groups) | NoPrefixCache
         ├─ SingleTypeKVCacheManager ×  one per KV cache group      vllm/v1/core/single_type_kv_cache_manager.py
         │   FullAttentionManager, SlidingWindowManager, MambaManager, CrossAttentionManager, ...
         └─ BlockPool (shared by all groups)                         vllm/v1/core/block_pool.py
             ├─ blocks: list[KVCacheBlock]  (block_id, ref_cnt, _block_hash, prev/next_free_block)
             ├─ free_block_queue: FreeKVCacheBlockQueue (doubly linked, eviction order)
             └─ cached_block_hash_to_block: BlockHashToBlockMap (hash+group_id → block)
```

The scheduler-side cache is pure bookkeeping: integers and Python objects on the CPU. The actual KV
tensors live on the workers; the scheduler hands them block ids, and the runner turns those into a
block table and a slot mapping (Section 5.3).

### 4.2 Blocks and the free queue

`KVCacheBlock` (`vllm/v1/core/kv_cache_utils.py`) is a slotted dataclass: `block_id`, `ref_cnt`,
`_block_hash` (set only when the block is full and cached), and `prev_free_block`/`next_free_block`.
`FreeKVCacheBlockQueue` threads those two pointers into a doubly linked list with a fake head and
tail, rather than using `collections.deque`, so that a block in the middle can be removed in O(1)
when a prefix hit "touches" it, without allocating Python objects (class docstring). Operations:
`popleft`/`popleft_n` (allocate), `remove` (touch), `append`/`append_n` (free cached blocks to the
tail), `prepend_n` (free uncached blocks to the head).

At construction `BlockPool` pops block 0 and marks it `is_null`: a placeholder used, for example,
for positions outside a sliding window. That is why usable capacity is `num_gpu_blocks − 1` and why
`BlockPool.get_usage` computes `1 − free / (num_gpu_blocks − 1)`.

### 4.3 Block hashes: a chain over the prefix

Every full block gets a hash that commits to **all tokens before it**, not just its own 16
(`hash_block_tokens`, `get_request_block_hasher` in `vllm/v1/core/kv_cache_utils.py`):

```
NONE_HASH = H(seed)                                  # init_none_hash
h0 = H( (NONE_HASH, (t0 … t15),  extra_keys_0) )      # hash_block_tokens(parent=None → NONE_HASH)
h1 = H( (h0,        (t16 … t31), extra_keys_1) )
h2 = H( (h1,        (t32 … t47), extra_keys_2) )      # H = sha256(pickle.dumps(x)) by default
```

- **Function.** `--prefix-caching-hash-algo` defaults to `sha256`, which pickles the tuple and
  hashes it (`vllm/utils/hashing.py: sha256`); `sha256_cbor` is reproducible across languages;
  `xxhash` variants are faster and non-cryptographic (`vllm/config/cache.py: CacheConfig`).
- **Seed.** For cryptographic algorithms `NONE_HASH` derives from a fixed seed
  (`"vllm-none-hash"`), so separate vLLM processes compute identical hashes and can share KV across
  nodes; for xxhash it is random per process unless `PYTHONHASHSEED` is set, to stop an attacker
  precomputing collisions (`resolve_none_hash_seed`, `init_none_hash`).
- **Extra keys** (`generate_block_hash_extra_keys`): the LoRA adapter name on every block, so two
  adapters never share KV; for multimodal inputs, `(mm identifier, offset of the item within the
  block)` for each item overlapping the block; `cache_salt` on the **first block only**, which
  isolates tenants because every later hash chains from it; and a digest of prompt embeddings when
  those are used.
- **When.** Hashes are computed as tokens become known: at request construction for the prompt
  (`Request.__init__` → `update_block_hashes`, in the EngineCore input thread) and again as output
  tokens fill blocks. Only full blocks are hashed.
- **Key in the map.** The pool indexes `BlockHashWithGroupId`: the hash bytes plus a 4-byte group
  id (`make_block_hash_with_group_id`), because hybrid models keep one physical block per group per
  logical block.
- **No deduplication.** If two requests compute the same block concurrently, both copies are
  cached; the map can hold several blocks per hash (`BlockHashToBlockMap`, NOTE #1). Block tables
  stay append-only; a block id never changes under a running request.

Because each hash includes its parent, "longest cached prefix" is a scan from the first block until
the first miss: a miss at block *k* implies misses at every later block.

### 4.4 Lookup: `get_computed_blocks`

`KVCacheManager.get_computed_blocks` returns an empty hit if prefix caching is off or the request
skips cache reads; `SamplingParams` sets `skip_reading_prefix_cache` automatically when
`prompt_logprobs` is requested, because a cache hit would leave those prompt positions without
logits (`vllm/sampling_params.py`).
Otherwise it calls `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length =
request.num_tokens − 1)`. The `− 1` exists because the last prompt token must run through the model
to produce logits. `FullAttentionManager.find_longest_cache_hit` walks
`max_length // block_size` hashes and stops at the first miss.

Worked example, `block_size = 16`:

| Prompt | Cached | Hit | Computed this time | Saved |
|---|---|---|---|---|
| 1,000-token system prompt + 200-token user turn, second user | first request's blocks | 62 full blocks = 992 tokens (block 63 mixes system and user tokens, so its hash differs per user) | 208 | 82.7% of prefill tokens |
| the identical 1,200-token prompt again | all 75 blocks | `(1199 // 16) × 16 = 1,184` | 16 (one block, because hits are block-aligned) | 98.7% |

The prompt-design rule that follows: put everything shared (system prompt, tool schemas, few-shot
examples, long documents) first and byte-identical, and put anything per-request (timestamps, user
ids) after it. One changed token early in the prompt changes every later hash.

Accounting: `record_prefix_cache_stats` adds `request.num_tokens` to queries and the hit length to
hits at admission, so `vllm:prefix_cache_hits / vllm:prefix_cache_queries` is a token-weighted
hit rate (`vllm/v1/metrics/stats.py: PrefixCacheStats.record`).

### 4.5 Allocation: `allocate_slots`

The docstring of `KVCacheManager.allocate_slots` draws the layout:

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
 already    prefix-cache   from a KV      tokens to   slots for draft
 held       hits (local)   connector      compute     tokens (EAGLE etc.)
```

Three stages: (1) free blocks the request no longer needs (outside a sliding window) and check
capacity, returning `None` if `required + watermark > free − reserved`; (2) attach the hit blocks
(`BlockPool.touch`: `ref_cnt += 1`, and if `ref_cnt` was 0 the block is `remove`d from the free
queue); (3) allocate new blocks for `new + lookahead` with `BlockPool.get_new_blocks`, which pops
from the head of the free queue and, if a popped block still carries a hash, evicts it from the hash
map first (`_maybe_evict_cached_block`).

Then it **caches immediately**: `coordinator.cache_blocks(request, min(computed + new,
request.num_tokens))` assigns hashes to every block that will be full once this step runs, before
the forward pass. Capping at `num_tokens` keeps unverified draft tokens out of the cache
(NOTE in `allocate_slots`). A consequence `(verify)`: a second request with the same prefix
admitted later in the same `schedule()` call can already hit those blocks, relying on KV being
written layer by layer before attention reads it.

### 4.6 Free and eviction order

`SingleTypeKVCacheManager.free` calls `block_pool.free_blocks(reversed(blocks))`, and
`BlockPool.free_blocks` splits the result:

- blocks with `ref_cnt` reaching 0 and **no hash** (the partial last block) are `prepend_n`-ed to the
  **head**: reused first, "LIFO reuse of non-cached blocks for better GPU locality";
- blocks with a hash are `append_n`-ed to the **tail**: reused last, in LRU order.

Worked example. Request X holds `[b1(h1), b2(h2), b3(h3), b4(partial)]` and finishes while the free
queue holds older blocks `[f1, f2]`:

```
before:   head → f1 → f2 → tail
reversed: b4, b3, b2, b1
after:    head → b4 → f1 → f2 → b3 → b2 → b1 → tail
```

The next allocations take `b4` (no loss), then `f1`, `f2` (older), then `b3` before `b2` before
`b1`. Within one chain the **tail is evicted first** and the root last, because the root is the part
most likely shared by the next request (the `FreeKVCacheBlockQueue` docstring states this ordering).
A later prefix hit on `b1`/`b2` removes them from the middle of the list in O(1).

`vllm:kv_cache_usage_perc` therefore measures blocks **referenced by live requests**: cached blocks
sitting in the free queue count as free (`BlockPool.get_usage`). A replica can show 30% usage while
holding a large, useful prefix cache; a router that wants cache affinity needs hashes (Section 12,
and the orchestration layer in `../../05-orchestrator/`), not this gauge.

### 4.7 Sizing the pool: from `gpu_memory_utilization` to `num_gpu_blocks`

At startup `EngineCore._initialize_kv_caches` (`vllm/v1/engine/core.py`) asks each worker for its
KV specs, runs `Worker.determine_available_memory` (`vllm/v1/worker/gpu_worker.py`), then
`get_kv_cache_configs` (`vllm/v1/core/kv_cache_utils.py`), and finally allocates tensors and
compiles/captures (`initialize_from_config`, `compile_or_warm_up_model`). The memory arithmetic:

```
requested      = ceil(total_gpu_memory × gpu_memory_utilization)                   # request_memory (vllm/v1/worker/utils.py)
non_kv         = weights + transient activation peak + non-torch growth             # memory_profiling (vllm/utils/mem_utils.py)
available_kv   = requested − non_kv − cudagraph_estimate                             # determine_available_memory
page_bytes     = num_kv_heads × block_size × (head_size + head_size_v) × dtype_bytes # AttentionSpec.page_size_bytes, per layer
bytes_per_block= Σ over layers in the group of page_bytes                           # _get_kv_cache_bytes_per_block
num_blocks     = available_kv // bytes_per_block                                     # get_kv_cache_config_from_groups
```

- The activation peak comes from a profiling forward pass of `max_num_batched_tokens` tokens plus a
  dummy sampler run over `max_num_seqs` rows (`GPUModelRunner.profile_run`, `_dummy_run`,
  `_dummy_sampler_run`). Larger budgets therefore shrink the KV pool.
- CUDA-graph memory is estimated by a capture-profiling pass (`profile_cudagraph_memory`) and
  subtracted when `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is on; the log line says this has been
  the default since v0.21.0 and suggests how much to raise `--gpu-memory-utilization` to keep the
  old KV size (`determine_available_memory`).
- `--kv-cache-memory-bytes` skips profiling and sets the KV size directly;
  `--num-gpu-blocks-override` forces the block count.
- `gpu_memory_utilization` is a fraction of **total** memory for this instance; if free memory at
  startup is below `requested`, startup fails ("Free memory on device ... on startup is less than
  desired GPU memory utilization", `request_memory`).

**Worked budget: Llama-3.1-8B-Instruct, BF16.** Shape: 32 layers, 32 query heads, 8 KV heads,
head_dim 128, hidden 4,096, vocab 128,256, 8,030,261,248 parameters (computed from the config shape).

```
weights          = 8,030,261,248 × 2 B                  = 16.06 GB = 14.96 GiB
KV per token     = 2 (K,V) × 32 layers × 8 heads × 128 × 2 B = 131,072 B = 128 KiB
page per layer   = 8 heads × 16 tokens × (128+128) × 2 B = 65,536 B = 64 KiB
bytes_per_block  = 32 layers × 64 KiB                    = 2 MiB   (16 tokens)
```

The activation and graph terms below are assumptions for illustration; your own
`Available KV cache memory: … GiB` log line replaces them.

| | L4 24 GB | H100 80 GB (SXM) |
|---|---|---|
| total memory seen by CUDA | 23,034 MiB = 22.49 GiB `(verify)` | 81,559 MiB = 79.65 GiB `(verify)` |
| `requested` at 0.92 | 20.69 GiB | 73.28 GiB |
| weights | 14.96 GiB | 14.96 GiB |
| activation peak + non-torch (assumed) | 1.0 GiB (2,048-token profile, 256-row sampler) | 2.0 GiB (8,192-token profile, 1,024-row sampler) |
| CUDA graphs (assumed) | 0.5 GiB | 1.0 GiB |
| **available KV** | **4.24 GiB** | **55.32 GiB** |
| `num_blocks` (÷ 2 MiB) | 2,169 | 28,322 |
| token capacity (× 16) | 34,704 | 453,152 |
| max concurrency at `max_model_len` 8,192 (512 blocks each) | 4.24× | 55.3× |
| at 32,768 (2,048 blocks) | 1.06× | 13.8× |
| at 131,072, the model's default (8,192 blocks = 16 GiB) | **does not start** | 3.46× |
| with `--kv-cache-dtype fp8` (1 MiB/block) | 4,338 blocks, 69,408 tokens | 56,645 blocks, 906,320 tokens |

The L4 row is the most common first-deployment failure. With no `--max-model-len`, the model's
131,072-token context needs 16 GiB of KV for a single request, so `_check_enough_kv_cache_memory`
raises: "To serve at least one request with the model's max seq len (131072), (16.00 GiB KV cache is
needed, which is larger than the available KV cache memory (…)". Fix it with `--max-model-len
16384` (2.12× concurrency), with `--max-model-len auto` (`-1`: `_auto_fit_max_model_len` picks the
largest length that fits, about 34.7k tokens with the assumed numbers above), with FP8 KV, or with a bigger GPU. The startup
log reports the result as "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request:
X.XXx" (`update_kv_cache_capacity`).

Per-request concurrency is computed as `num_blocks / ceil(max_model_len / block_size)`
(`get_max_concurrency_for_kv_cache_config`): a worst case. Real concurrency is higher because
requests are shorter than `max_model_len`, and prefix sharing makes it higher still.

### 4.8 Hybrid models: several KV cache groups

When layers need different cache behaviour (full attention, sliding-window attention, Mamba state,
cross-attention), `get_kv_cache_groups` splits them into groups that repeat a pattern; the
`_get_kv_cache_groups_uniform_page_size` docstring's example: 10 full-attention and 20
sliding-window layers form the pattern (1 × full, 2 × sw), giving 3 groups of 10 layers each, all
drawing block ids from one pool, each with its own block table. Page sizes are unified so any block
id can serve any group (`unify_kv_cache_spec_page_size`). Per group:

- `SlidingWindowManager` needs only `ceil((window − 1) / block_size)` contiguous blocks for a hit
  and frees blocks that fall out of the window during decoding, replacing them with the null block
  (`remove_skipped_blocks`); its admission cap is bounded by the window plus in-flight tokens
  (`SlidingWindowSpec.max_admission_blocks_per_request`).
- `MambaManager` stores recurrent state rather than per-token KV; with prefix caching,
  `mamba_cache_mode = "align"` checkpoints state at block boundaries of the scheduled steps
  (`CacheConfig.mamba_cache_mode`).
- `HybridKVCacheCoordinator.find_longest_cache_hit` runs a fixed point: each group accepts or
  shortens the candidate hit length until all agree (its docstring).

`--disable-hybrid-kv-cache-manager` forces every layer to allocate as full attention (simpler, more
memory).

---

## 5. The model runner and the executor

### 5.1 Executors: how a `SchedulerOutput` reaches the GPUs

`Executor.get_class` (`vllm/v1/executor/abstract.py`) maps `--distributed-executor-backend` to a
class; `ParallelConfig.__post_init__` (`vllm/config/parallel.py`) picks the default:

| Backend | Chosen when | Topology | Transport |
|---|---|---|---|
| `uni` → `UniProcExecutor` | world size 1 | the worker lives inside the EngineCore process | direct call |
| `mp` → `MultiprocExecutor` | world size > 1 on CUDA, fits on the node (or `--nnodes` set) | one `WorkerProc` per GPU | shared-memory `MessageQueue` broadcast |
| `ray` → `RayDistributedExecutor` (`RayExecutorV2` with `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`) | inside a Ray placement group, or `--data-parallel-backend ray`, or explicit | Ray actors, multi-node | Ray compiled DAG (`RayDistributedExecutor._compiled_ray_dag`) |
| `external_launcher` | `torchrun`-style launch (RL, SPMD) | caller owns the processes | — |

`MultiprocExecutor.collective_rpc` enqueues `(method, args, kwargs, output_rank)` on one
`MessageQueue` (`vllm/distributed/device_communicators/shm_broadcast.py`): a shared-memory ring
buffer for readers on the same node (default chunk 24 MiB, sized for grammar bitmasks of 1,024
requests) and a ZMQ XPUB socket for remote readers. Every worker executes the same
`SchedulerOutput`; only one replies: `_get_output_rank` returns the first tensor-parallel rank of
the last pipeline stage (`world_size − tp_size × pcp_size`).

### 5.2 A worker's life

`Worker` (`vllm/v1/worker/gpu_worker.py`) goes through a fixed startup sequence, driven by
`EngineCore.__init__` and `_initialize_kv_caches`:

1. `init_device`: set the CUDA device, initialize the distributed environment and model-parallel
   groups, take the baseline memory snapshot, compute `requested_memory`.
2. `load_model` → `GPUModelRunner.load_model` → model loader (Section 8). Log: "Model loading took
   X GiB memory and Y seconds".
3. `get_kv_cache_spec`: one `KVCacheSpec` per attention layer (`FullAttentionSpec`,
   `SlidingWindowSpec`, `MLAAttentionSpec`, `MambaSpec`, ...; `vllm/v1/kv_cache_interface.py`).
4. `determine_available_memory`: profiling forward pass and CUDA-graph estimate (Section 4.7).
5. `initialize_from_config(kv_cache_config)`: allocate the KV tensors, bind them to layers, build
   attention metadata builders.
6. `compile_or_warm_up_model`: compile and warm up extra sizes, `kernel_warmup`, `capture_model`
   (large shapes first "so that the smaller shapes can reuse the memory pool", comment in
   `GPUModelRunner.capture_model`), then warm the sampler at maximum shape. Log: "Graph capturing
   finished in N secs, took X GiB"; EngineCore logs "init engine (profile, create kv cache, warmup
   model) took N s (compilation: M s)".

At runtime each step is two RPCs: `execute_model(scheduler_output)` then
`sample_tokens(grammar_output)`. The split exists so EngineCore can compute the structured-output
bitmask on the CPU while the forward pass runs (`vllm/v1/engine/core.py: EngineCore.step`).

### 5.3 Model runner V1: the persistent batch

`GPUModelRunner` in `vllm/v1/worker/gpu_model_runner.py` keeps an `InputBatch`
(`vllm/v1/worker/gpu_input_batch.py`): fixed-capacity CPU/GPU arrays indexed by a row per request:
`token_ids_cpu_tensor` (`max_num_reqs × max_model_len`), `num_computed_tokens_cpu_tensor`, a
`MultiGroupBlockTable` (one block table per KV cache group), and sampling parameters as tensors
(`temperature`, `top_p`, `top_k`, penalties, generators). The design bet, stated in
`_update_states`: "consecutive batches contain mostly the same requests". Per step:

1. `_update_states(scheduler_output)`: drop finished requests; remove unscheduled ones from the
   batch but keep their cached state; add new and resumed ones; append new token ids and new block
   ids for running ones; `condense()` to close gaps. Only rows that changed are touched.
2. `_prepare_inputs`: build the flattened ragged batch. Worked example with three requests
   scheduled for `[2, 5, 3]` tokens and `num_computed_tokens = [10, 0, 40]` (the example in the
   source comments, extended with positions):

```
num_scheduled_tokens = [2, 5, 3]
req_indices          = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]          np.repeat(arange(3), [2,5,3])
cu_num_tokens        = [2, 7, 10]  → query_start_loc = [0, 2, 7, 10]
query_pos (arange)   = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
positions            = computed[req_indices] + query_pos
                     = [10,11, 0,1,2,3,4, 40,41,42]
input_ids            = token_ids_cpu.flatten()[positions + req_indices × max_model_len]
seq_lens             = computed + scheduled = [12, 5, 43]
```

3. Slot mapping: for every token, `slot = block_table[req, pos // block_size] × block_size +
   pos % block_size`, computed by a Triton kernel on the GPU; padded slots are set to a pad id so the
   CUDA graph shape stays fixed (`vllm/v1/worker/block_table.py: BlockTable.compute_slot_mapping`,
   `ComputeSlotMappingKernel`). With `block_size = 16` and request 2's block table `[7, 3, 12]`,
   position 42 goes to block `12` (42 // 16 = 2), offset 10, slot `12 × 16 + 10 = 202`.
4. `_build_attention_metadata`: one `CommonAttentionMetadata` (`query_start_loc`, `seq_lens`,
   `block_table_tensor`, `slot_mapping`, `max_query_len`, `max_seq_len`, `num_actual_tokens`;
   `vllm/v1/attention/backend.py`) turned into backend-specific metadata by each group's
   `AttentionMetadataBuilder.build`.
5. `_determine_batch_execution_and_padding`: ask the `CudagraphDispatcher`
   (`vllm/v1/cudagraph_dispatcher.py`) for the runtime mode (FULL, PIECEWISE or NONE) and the padded
   size (Section 5.5).
6. Forward under `set_forward_context(attn_metadata, ...)` (`vllm/forward_context.py`); each
   `Attention` layer reads its metadata from the forward context inside the custom ops
   `vllm::unified_kv_cache_update` and `vllm::unified_attention_with_output`
   (`vllm/model_executor/layers/attention/attention.py`). Logits are computed only for the rows that
   will be sampled, and the state is parked in `execute_model_state`; `execute_model` returns `None`.
7. `sample_tokens(grammar_output)`: apply the bitmask, run `Sampler` or `RejectionSampler`, run the
   drafter, and return `ModelRunnerOutput` (asynchronously under async scheduling:
   `AsyncGPUModelRunnerOutput` copies tokens to the host on a side stream).

The batch is reordered so decodes come first: `reorder_batch_to_split_decodes_and_prefills` sorts
rows into decode → short extend → long extend → prefill (`vllm/v1/attention/backends/utils.py`),
which lets backends run separate decode and prefill kernels on contiguous slices.

### 5.4 Model runner V2: request state in GPU slots

`vllm/v1/worker/gpu/model_runner.py: GPUModelRunner` (default when supported, Section 1.4) keeps
per-request state in fixed slots that live across steps: `RequestState`
(`vllm/v1/worker/gpu/states.py`) holds `all_token_ids` (`max_num_reqs × max_model_len`, placed in
UVA host memory because it can be several GB), `prompt_len`, `prefill_len`, `total_len`,
`num_computed_tokens`, with free slot indices recycled. A step's batch is an `idx_mapping` from
batch rows to slots rather than a reordered persistent batch, and the inputs are built by Triton
kernels on the GPU (`_prepare_prefill_inputs_kernel`, `_prepare_pos_seq_lens_kernel`,
`_combine_sampled_and_draft_tokens_kernel` in `vllm/v1/worker/gpu/input_batch.py`). Sampled and
draft tokens stay on the device between steps, which is what makes async scheduling and speculative
decoding compose without host synchronisation. Its sampler uses Gumbel-max sampling with Philox
noise in Triton (`vllm/v1/worker/gpu/sample/gumbel.py`). The file's header states its own rule:
code shared by every model only, "Be paranoid about changing this file".

### 5.5 torch.compile and CUDA graphs

Two orthogonal mechanisms (the `CompilationConfig.cudagraph_mode` docstring says so explicitly):

**Compilation.** `CompilationMode.VLLM_COMPILE` (3) is the default. Models opt in with the
`@support_torch_compile` decorator (`vllm/compilation/decorators.py`). Dynamo traces the forward once
with a symbolic batch size; `VllmBackend` (`vllm/compilation/backends.py`) runs custom Inductor
passes (fusions of norm/activation with quantization, collective fusions) and `split_graph` cuts the
FX graph at the **splitting ops**, by default the attention-like custom ops
(`vllm::unified_attention_with_output`, `vllm::unified_mla_attention_with_output`, Mamba mixers, ...;
`CompilationConfig._attention_ops`). Each piece is compiled by a `PiecewiseBackend`
(`vllm/compilation/piecewise_backend.py`), for a general dynamic shape and optionally for specific
`compile_sizes`. Artifacts are cached on disk (`cache_dir`; `VLLM_COMPILE_CACHE_SAVE_FORMAT`), so the
second start of the same model and config skips most compile time ("Directly load the compiled
graph(s) ...", `CompilerManager`).

**CUDA graphs.** `CUDAGraphMode` (`vllm/config/compilation.py`):

| Mode | Captured | Use |
|---|---|---|
| `NONE` | nothing | debugging, `--enforce-eager` |
| `PIECEWISE` | the compiled pieces between attention ops; attention runs eagerly | any batch shape, any backend |
| `FULL` | the whole forward, attention included | backends with `AttentionCGSupport.ALWAYS` |
| `FULL_DECODE_ONLY` | full graphs for uniform decode batches, eager otherwise | decode instances in P/D |
| `FULL_AND_PIECEWISE` (default) | full graphs for uniform decode batches, piecewise for mixed | the best general choice per the docstring |

Whether a full graph can include attention depends on the backend's `AttentionCGSupport`
(`ALWAYS`, `UNIFORM_BATCH`, `UNIFORM_SINGLE_TOKEN_DECODE`, `NEVER`; `vllm/v1/attention/backend.py`):
FlashAttention 3 reports `ALWAYS`, FlashAttention 2 `UNIFORM_BATCH`, Triton attention `ALWAYS`,
FlashInfer `UNIFORM_BATCH` or `UNIFORM_SINGLE_TOKEN_DECODE` depending on whether its TRT-LLM decode
kernels apply (`get_cudagraph_support` in each backend).

**Capture sizes.** Unless `--cudagraph-capture-sizes` is given, `VllmConfig._set_cudagraph_sizes`
builds:

```
max_capture = min(max_num_seqs × uniform_decode_query_len × 2, 512)   # 1024 on data-center Blackwell (SM100 family)
sizes       = [1, 2, 4] + range(8, 256, 8) + range(256, max_capture + 1, 16)
            (+ max_num_batched_tokens if it is ≤ max_capture)
```

With `max_num_seqs = 256` and no speculation, `max_capture = min(512, 512) = 512`: 3 + 31 + 17 =
**51 sizes** (1, 2, 4, 8, 16, …, 248, 256, 272, …, 512). With speculation, `uniform_decode_query_len
= 1 + num_speculative_tokens`, so a decode batch is measured in tokens, not requests. At runtime a
batch of *n* tokens is padded up to the next captured size (a 3-token decode batch replays the
size-4 graph); a batch larger than the largest size runs without a CUDA graph. `--performance-mode
interactivity` captures every size from 1 to 32 to avoid padding waste at small batch sizes.

**Optimization levels.** `-O0` no compilation, no graphs; `-O1` compilation plus piecewise graphs;
`-O2` (default) adds full graphs; `-O3` currently equals `-O2` (`vllm/config/vllm.py:
OptimizationLevel`). `--enforce-eager` disables both compilation and graphs
(`vllm/config/model.py: ModelConfig.enforce_eager`).

Why this matters in a review: a decode step of a small model is dozens of kernel launches per layer;
at ~5 µs per launch from Python, a 32-layer model spends milliseconds just launching. CUDA-graph
replay collapses that into one launch per graph, which is why `--enforce-eager` often shows a large
ITL regression on small models and a small one on large models `(verify: measure in the lab)`.

### 5.6 Parallelism inside one engine

- **Tensor parallelism** (`--tensor-parallel-size`): weights of each linear layer are split
  column-wise or row-wise; each transformer layer does two all-reduces (after attention output and
  after the MLP). KV heads are split across ranks, so each GPU's KV pool holds `num_kv_heads / tp`
  heads and the block budget of Section 4.7 is computed per GPU. `initialize_model_parallel`
  (`vllm/distributed/parallel_state.py`) builds the groups (its docstring example: 8 GPUs with TP=2,
  PP=4 give TP groups `[g0,g1] [g2,g3] [g4,g5] [g6,g7]` and PP groups `[g0,g2,g4,g6]
  [g1,g3,g5,g7]`). The all-reduce implementation is picked per call in
  `CudaCommunicator.all_reduce` (`vllm/distributed/device_communicators/cuda_communicator.py`):
  FlashInfer all-reduce, NCCL symmetric memory, quick all-reduce, custom (IPC) all-reduce, torch
  symmetric memory, then PyNCCL, depending on availability and message size.
- **Pipeline parallelism** (`--pipeline-parallel-size`): layers split into stages; to keep all stages
  busy EngineCore keeps `pp_size` batches in flight through its batch queue
  (`VllmConfig.max_concurrent_batches`, `EngineCore.step_with_batch_queue`).
- **Data parallelism** (`--data-parallel-size`): independent EngineCore processes, each with its own
  scheduler and KV cache; the front-end client either load-balances across them
  (`DPLBAsyncMPClient`) or is told the rank by an external balancer (`DPAsyncMPClient`)
  (`vllm/v1/engine/core_client.py: EngineCoreClient.make_async_mp_client`). For MoE models DP ranks
  are coupled (`DPEngineCoreProc`) so expert-parallel collectives stay in lockstep.

The parallelism trade-offs (when TP pays off, NVLink versus PCIe) are covered in
[`../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md`](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md).

---

## 6. Attention backends

### 6.1 The interface

Three classes per backend (`vllm/v1/attention/backend.py`):

| Class | Responsibility | Key methods |
|---|---|---|
| `AttentionBackend` | static description and capability checks | `get_name`, `get_impl_cls`, `get_builder_cls`, `get_supported_kernel_block_sizes`, `supports_head_size`, `supports_dtype`, `supports_kv_cache_dtype`, `supports_compute_capability`, `is_mla`, `validate_configuration` |
| `AttentionMetadataBuilder` | turn the step's `CommonAttentionMetadata` into backend metadata (scheduler metadata for FA3, `plan()` for FlashInfer) | `build`, `build_for_cudagraph_capture`, `get_cudagraph_support`, `use_cascade_attention` |
| `AttentionImpl` | the kernel calls | `forward(layer, query, key, value, kv_cache, attn_metadata, output)` |

The model code never names a backend: `Attention` layers call the registered custom ops, and the
runner supplies per-layer metadata through the forward context (Section 5.3).

### 6.2 The paged layout, concretely

For FlashAttention the per-layer KV tensor is documented as `[num_blocks, num_kv_heads, block_size,
2 × head_size]`, K and V packed in the last dimension and split with
`kv_cache.transpose(1, 2).split(head_size, dim=-1)` (`vllm/v1/attention/backends/flash_attn.py:
FlashAttentionImpl.forward`). Other layouts are expressed through `KVCacheLayout` and resolved once
by EngineCore before memory profiling (`resolve_kv_cache_layout`, `CacheConfig.kv_cache_layout`).
Each step does two things per layer:

1. **Write**: the new K and V for every scheduled token go to `slot_mapping[i]`
   (`reshape_and_cache_flash`, or fused into RoPE/norm kernels when the backend supports it).
2. **Read**: one varlen call covers the whole mixed batch: `flash_attn_varlen_func(q, key_cache,
   value_cache, cu_seqlens_q=query_start_loc, seqused_k=seq_lens, block_table=..., causal=True,
   ...)`. Prefill rows (many queries) and decode rows (one query) share the launch; the kernel walks
   each row's block table.

`num_common_prefix_blocks` from the scheduler enables **cascade attention**: when every running
request shares a long prefix, the shared blocks are attended once and merged with per-request suffix
attention (`use_cascade_attention`, `_compute_cascade_attn_prefix_len` in the V1 runner); it is
disabled with async speculative decoding (`VllmConfig`) and by `--disable-cascade-attn`.

### 6.3 How a backend is chosen

`get_attn_backend` (`vllm/v1/attention/selector.py`) collects the layer's properties (head size,
dtype, KV-cache dtype, MLA, sinks, sliding window, block size if the user set one) and asks the
platform. On CUDA, `CudaPlatformBase.get_attn_backend_cls` (`vllm/platforms/cuda.py`) either validates
the backend named by `--attention-backend` (an error if invalid) or walks `_get_backend_priorities`
and takes the highest-priority backend whose `validate_configuration` returns no reasons. The
`VLLM_ATTENTION_BACKEND` environment variable no longer exists in `vllm/envs.py`; the flag is the
interface.

| GPU | Compute capability | Standard attention, in priority order | MLA models |
|---|---|---|---|
| T4 | 7.5 | FlashAttention needs ≥ 8.0; FlashInfer is temporarily floored at 8.0 → **Triton attention** | Triton MLA |
| A100 / L4 | 8.0 / 8.9 | **FlashAttention (FA2)** → FlashInfer → Triton → FlexAttention | FA MLA → FlashMLA (needs SM 9.x/10.x) → FlashInfer MLA → Triton MLA |
| H100 / H200 | 9.0 | **FlashAttention (FA3)** → FlashInfer → Triton → FlexAttention | same list; FA MLA or FlashMLA |
| B200 / GB200 | 10.0 | **FlashInfer** (TRT-LLM kernels) → FlashAttention (FA4 when supported) → Triton → FlexAttention | FlashInfer MLA → TokenSpeed MLA → CUTLASS MLA → FA MLA → FlashMLA → Triton MLA |
| RTX PRO 6000 / 5090 | 12.0 | FlashAttention → FlashInfer → Triton | Triton MLA |

Sources: priority lists in `_get_backend_priorities`; floors in each backend's
`supports_compute_capability` (`flash_attn.py`: ≥ 8.0; `flashinfer.py`: 8.0 to 12.1, with the SM75
bug reference; `triton_attn.py`: any; `mla/flashmla.py`: major 9 or 10); FA version in
`get_flash_attn_version` (`vllm/v1/attention/backends/fa_utils.py`: FA3 on SM90, FA4 on SM100 if
supported, FA2 otherwise, overridable by `attention_config.flash_attn_version`). If `--block-size` is
set and it excludes a higher-priority backend, the selector logs a warning suggesting you drop the
flag.

### 6.4 The main backends in one paragraph each

- **FlashAttention** (vLLM's fork, `vllm.vllm_flash_attn`): one varlen kernel for mixed batches,
  paged KV via `block_table`, block sizes in multiples of 16, FP8 KV with descales, sliding windows,
  soft-capping. FA3 on Hopper uses asynchronous TMA/warp specialisation and supports full CUDA graphs
  for mixed batches; FA2 only for uniform batches. The algorithm itself is in
  [`../flash-attention/`](../flash-attention/).
- **FlashInfer** (`flashinfer.py`): `BatchPrefillWithPagedKVCacheWrapper` and
  `BatchDecodeWithPagedKVCacheWrapper` with a `plan()` step per batch, plus TRT-LLM-generated decode
  and context kernels (`trtllm_batch_decode_with_kv_cache`, `trtllm_batch_context_with_kv_cache`) on
  SM100. Default on Blackwell.
- **Triton attention** (`triton_attn.py`, kernels in `vllm/v1/attention/ops/`): portable, any
  capability, `AttentionCGSupport.ALWAYS`; the fallback on Turing and on non-NVIDIA platforms.
- **MLA backends** (`vllm/v1/attention/backends/mla/`): for DeepSeek-style Multi-head Latent
  Attention; see 6.5.

### 6.5 GQA and MLA change the bytes, not the paging

With grouped-query attention, `num_kv_heads < num_heads`: Llama-3.1-8B has 32 query heads sharing 8
KV heads, so its KV is 4× smaller than the 32-head equivalent; FlashAttention handles the grouping
inside the kernel. With MLA, `MLAAttentionSpec` stores one latent vector per token per layer
(`head_size_v = 0`, a single "head"; `vllm/v1/kv_cache_interface.py`). For DeepSeek-V3
(61 layers, latent 512 + RoPE 64 = 576 values; config values `(verify)`):

```
MLA, BF16:           61 × 576 × 2 B              =  70,272 B/token ≈ 68.6 KiB
MHA equivalent:      61 × 128 heads × 128 × 2 × 2 B ≈ 4.0 MB/token
```

A 57× reduction, which is why MLA models get their own kernels (FlashMLA, CUTLASS MLA, FlashInfer
MLA) and their own KV dtypes (`fp8_ds_mla`, `nvfp4_ds_mla` in `CacheDType`).

---

## 7. Sampling and structured output

### 7.1 The sampler's order of operations

`Sampler.forward` (`vllm/v1/sample/sampler.py`) documents its nine steps; condensed:

1. If logprobs are requested, compute them from the **raw** logits (default `logprobs_mode =
   "raw_logprobs"`: before penalties and temperature, unlike V0).
2. Cast logits to float32.
3. `allowed_token_ids` whitelist; 4. `bad_words` exclusion.
5. Non-argmax-invariant logits processors: `MinTokensLogitsProcessor` (masks stop tokens until
   `min_tokens`), `LogitBiasLogitsProcessor`.
6. Penalties: repetition, frequency, presence.
7. Sample: greedy for greedy rows; otherwise temperature, then argmax-invariant processors
   (`MinPLogitsProcessor`), then top-k/top-p, then draw.
8. Gather top-`logprobs` and the sampled token's logprob.

Logits processors in V1 are **batch-level classes**, not per-request callables: each implements
`is_argmax_invariant`, updates its state from a `BatchUpdate` when rows are added, removed or moved,
and applies to the whole logits tensor (`vllm/v1/sample/logits_processor/interface.py`,
`builtin.py`). Custom ones are loaded with `--logits-processors`. Argmax-invariant processors (min-p)
are skipped entirely for greedy rows.

### 7.2 Drawing a token without a host sync

`random_sample` (`vllm/v1/sample/ops/topk_topp_sampler.py`) avoids `torch.multinomial` because it
forces a CPU-GPU synchronisation. It draws exponential noise `q ~ Exp(1)` and returns
`argmax(probs / q)`, which samples exactly from `probs` (the exponential-race form of the Gumbel-max
trick). Requests with a `seed` get their own `torch.Generator`, applied row by row
("This can be slow because we handle each request one by one", comment in the same function); all
others share one batched draw. For top-k/top-p rows, FlashInfer's rejection-based sampler is used by
default when the GPU supports it (SM 8.0 to 12.1, more than 16 SMs; log line "Using FlashInfer for
top-p & top-k sampling."; opt out with `VLLM_USE_FLASHINFER_SAMPLER=0`; `flashinfer_sampler_supported`,
`flashinfer_sample`); it is statistically equivalent to `random_sample` but not bit-identical. Model Runner V2 does the same with Philox Gumbel noise in Triton (Section 5.4).

### 7.3 Structured output

`StructuredOutputManager` (`vllm/v1/structured_output/__init__.py`) lives in EngineCore:

1. **Compile.** `grammar_init` is called from `preprocess_add_request` in the input thread and
   submits grammar compilation to a thread pool; the request sits in
   `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` until it finishes, and the scheduler skips it meanwhile
   (`vllm/v1/request.py`, `Scheduler._try_promote_blocked_waiting_request`).
2. **Backend.** `--structured-outputs-config` `backend = "auto"` (default), `xgrammar`, `guidance`,
   `outlines` or `lm-format-enforcer` (`vllm/config/structured_outputs.py`); only one backend per
   engine ("We do NOT support different backends on a per-request basis in V1", `grammar_init`).
   Request constraints come from `SamplingParams.structured_outputs` (`json`, `regex`, `choice`,
   `grammar`, `json_object`, `structural_tag`; `vllm/sampling_params.py`).
3. **Mask.** Each step, `Scheduler.get_grammar_bitmask` → `StructuredOutputManager.grammar_bitmask`
   fills a packed int32 bitmask (one bit per vocabulary entry, one row per structured request, plus
   one row per speculative position) while the GPU runs the forward (`EngineCore.step`). The worker
   applies it to the logits before sampling (`apply_grammar_bitmask`,
   `vllm/v1/structured_output/utils.py`).
4. **Advance.** `update_from_output` calls `accept_tokens`; draft tokens are pre-validated with
   `validate_tokens` so invalid drafts become `-1` and are rejected.

Size, worked: a 128,256-token vocabulary needs `ceil(128256 / 32) = 4,008` int32 words = 16,032 B
per row; 256 rows are 4.1 MB, 1,024 rows 16.4 MB, which is why the executor's shared-memory
message chunk defaults to 24 MiB (Section 5.1).

### 7.4 Speculative decoding

**Configuration.** `--speculative-config '{"method": ..., "num_speculative_tokens": k, ...}'`
(`vllm/config/speculative.py: SpeculativeConfig`). Methods on `main`: `ngram` (prompt lookup on the
CPU, a Numba KMP search for the longest suffix match, `vllm/v1/spec_decode/ngram_proposer.py`),
`ngram_gpu`, `draft_model` (a small model of the same family), `eagle` and `eagle3` (a light
head fed the target's hidden states; EAGLE-3 uses auxiliary hidden states from several layers,
`vllm/v1/spec_decode/eagle.py`, `llm_base_proposer.py`), a long list of model-specific **MTP**
types (`deepseek_mtp`, `qwen3_next_mtp`, `glm4_moe_mtp`, …, `MTPModelTypes`) that reuse the
checkpoint's multi-token-prediction layers, plus `medusa`, `mlp_speculator`, `suffix`, `dflash`,
`dspark` and `custom_class`.

**Scheduling.** Drafts are just more "tokens to compute". After step *N* the drafter proposes `k`
tokens (`GPUModelRunner.propose_draft_token_ids`; with async scheduling they stay on the GPU), the
scheduler schedules `1 + k` tokens for that request in step *N+1*, `allocate_slots` reserves
`num_lookahead_tokens` extra slots, and `update_from_output` subtracts the rejected ones from
`num_computed_tokens` (Section 3.9). New decodes are padded to `1 + k` rows so uniform-decode full
CUDA graphs still apply (`pad_spec_decode` in `schedule`).

**Verification.** `RejectionSampler` (`vllm/v1/sample/rejection_sampler.py`) "strictly follows"
Leviathan et al. (arXiv:2211.17192): accept draft token *x* with probability `min(1, p(x)/q(x))`;
on the first rejection sample a recovered token from `normalize(max(p − q, 0))`; if all `k` are
accepted, append the **bonus** token sampled from the target. With the default
`draft_sample_method = "greedy"` the draft distribution is a point mass, so `q(x) = 1` and the test
becomes `accept iff p(x) ≥ u`, with the recovered token drawn from `p` with *x* removed
(`NO_DRAFT_PROBS` path of `rejection_random_sample_kernel` and `sample_recovered_tokens_kernel`).
For greedy target requests, acceptance is exact-match with the target argmax. Either way the output
distribution equals the target's; speculation changes speed, not quality.

**Worked expectation.** With `k = 3` and a per-token acceptance rate α = 0.7 (independence
assumed), expected tokens per target step = `(1 − α^(k+1)) / (1 − α) = (1 − 0.2401) / 0.3 = 2.53`.
At small batch sizes the 4-token verification step costs about the same as a 1-token step
(decode is bandwidth-bound), so ITL drops roughly 2.5× minus drafter cost. At large batch sizes the
step turns compute-bound, verifying 4 tokens per request costs real FLOPs, and the gain shrinks or
inverts. The scheduler exposes `num_speculative_tokens_per_batch_size` to vary `k` with batch size
(`dynamic_sd_lookup` in `Scheduler.__init__`). Acceptance is observable: `vllm:spec_decode_num_drafts`,
`vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens` and
`vllm:spec_decode_num_accepted_tokens_per_pos`, plus a periodic log line with "Mean acceptance length"
and "Per-position acceptance rate" (`vllm/v1/spec_decode/metrics.py`). Acceptance length is the number
to watch: it is the measured version of the 2.53 above.

---

## 8. Quantization and weight loading

### 8.1 How a quantization method is chosen

`--quantization` is optional: `ModelConfig` first reads `quantization_config` from the checkpoint's
`config.json`, and only if that is absent treats the weights as unquantized in `--dtype`
(`vllm/config/model.py`, `quantization` field docstring). The accepted names are the
`QuantizationMethods` literal in `vllm/model_executor/layers/quantization/__init__.py`, and
`get_quantization_config` maps each to a `QuantizationConfig` class:

| Family | Names | Config class | Notes from source |
|---|---|---|---|
| FP8 checkpoints | `fp8` | `Fp8Config` (`fp8.py`) | serialized FP8 weights, static or dynamic activation scales, optional block scales; `fp8` **no longer does online quantization** and raises with a pointer to `fp8_per_tensor` |
| online (quantize a BF16 checkpoint at load) | `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `mxfp4`, `int8_per_channel_weight_only`, `nvfp4_per_token` | `OnlineQuantizationConfig` (`vllm/config/quantization.py: _ONLINE_SHORTHANDS`) | weights quantized as they stream in, then finalized |
| GPTQ | `gptq`, `gptq_marlin`, `auto_gptq` | `AutoGPTQConfig` (`auto_gptq.py`) | "Config class for AutoGPTQ quantization using Marlin kernels" |
| AWQ | `awq`, `awq_marlin`, `auto_awq` | `AutoAWQConfig` (`auto_awq.py`) | Triton, Marlin and XPU backends; min capability 7.5 |
| llm-compressor | `compressed-tensors` | `CompressedTensorsConfig` | schemes W8A8 FP8/INT8, W8A16 FP8, W4A16 (`wNa16`), W4A8, W4A4 NVFP4/MXFP4 (`compressed_tensors/schemes/`) |
| vendor formats | `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `quark`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`, `moe_wna16`, `experts_int8`, … | various | `fbgemm_fp8`, `fp_quant` are listed as deprecated |

Kernels are chosen per layer, not per model. Weight-only int4/int8 layers go through
`choose_mp_linear_kernel`, whose CUDA priority list is `CutlassW4A8`, `Machete` (Hopper only,
`MacheteLinearKernel.can_implement`), `Marlin` (compute capability ≥ 7.5), `Conch`, `Exllama`,
`TritonW4A16`, `Humming` (`vllm/model_executor/kernels/linear/__init__.py: _POSSIBLE_KERNELS`).
FP8 GEMMs try FlashInfer, CUTLASS, torch `_scaled_mm` variants, then Marlin
(`_POSSIBLE_FP8_KERNELS`); `Fp8LinearMethod` falls back to Marlin weight-only FP8 on GPUs without
FP8 tensor cores ("For GPUs that lack FP8 hardware support, we can leverage the Marlin kernel",
`fp8.py`). The chosen kernel is logged once per method ("Using MarlinLinearKernel for
AutoGPTQLinearMethod").

### 8.2 What quantization buys, worked

Decode re-reads every weight each step, so weight bytes set a floor on ITL:

| Llama-3.1-8B weights | Bytes | Floor per decode step on L4 (300 GB/s `(verify)`) | on H100 SXM (3.35 TB/s `(verify)`) | KV blocks freed on L4 (2 MiB each) |
|---|---|---|---|---|
| BF16 | 16.06 GB | 53.5 ms | 4.8 ms | — |
| FP8 (W8A8) | 8.03 GB | 26.8 ms | 2.4 ms | +3,829 blocks (61,264 tokens) |
| INT4 group-128 (≈ 4.25 bits/weight) | ≈ 4.27 GB | 14.2 ms | 1.3 ms | +5,624 blocks (89,984 tokens) |

Two effects compound: faster decode and a bigger KV pool, which on a 24 GB card is often the larger
win. Prefill is compute-bound, so W8A8 FP8 (FP8 tensor cores on Ada/Hopper/Blackwell) speeds it
up, while weight-only INT4 does not reduce FLOPs and can even slow large-batch prefill through
dequantization. KV-cache quantization (`--kv-cache-dtype fp8`) is independent of weight
quantization and halves KV bytes (Section 4.7). Accuracy has to be measured per model and task; the
serving-engine primer covers what to check.

### 8.3 The loading path

```
LoadConfig.load_format ("auto")                                   vllm/config/load.py
 → get_model_loader → DefaultModelLoader                          vllm/model_executor/model_loader/__init__.py, default_loader.py
   → BaseModelLoader.load_model                                   base_loader.py
       with set_default_torch_dtype(dtype), with target_device:
           model = create_model(...)          # modules allocated directly on the GPU, each with a quant_method
       load_weights(model):
           _prepare_weights → download_weights_from_hf (safetensors preferred; falls back to .bin)
           weights iterator: safetensors_weights_iterator (mmap, "lazy")
                           | multi_thread_safetensors_weights_iterator (enable_multithread_load)
                           | fastsafetensors_weights_iterator (load_format="fastsafetensors")
                           | "eager" / prefetch strategies for network filesystems
           model.load_weights(iterator)       # per-model name mapping; each parameter's weight_loader
                                              # stacks q/k/v into qkv_proj and slices its TP shard
       process_weights_after_loading(...)     # repack for the kernel (e.g. Marlin layout), fuse scales
```

Other formats: `runai_streamer` (streams safetensors from object storage), `tensorizer`,
`sharded_state` (pre-sharded per TP rank), `instanttensor`, `ipc_cache` (map already-quantized weights
from a local cache daemon started with `vllm preload`), `modelexpress`, `mistral`, `npcache`, `dummy`
(random weights for profiling), and plugin-registered loaders (`LoadConfig.load_format` docstring,
loader registry in `model_loader/__init__.py`). GGUF and bitsandbytes are not in that registry at this
commit `(verify: where they are handled now)`. `--safetensors-load-strategy` defaults to memory-mapped lazy loading and
turns on prefetching automatically when it detects NFS and the checkpoint fits in 90% of RAM
(`LoadConfig.safetensors_load_strategy`).

Where cold-start time goes, from the log lines: "Loading weights took X seconds" (I/O plus
per-tensor copies; `default_loader.py`), "Model loading took X GiB memory and Y seconds" (the whole
load, V1 runner), compilation ("Compiling a graph for compile range … takes X s", or a cache hit),
"Graph capturing finished in N secs", then "init engine (profile, create kv cache, warmup model)
took N s". For a 16 GB checkpoint the I/O floor alone is 16 GB divided by storage bandwidth
(about 8 s at 2 GB/s); the storage side of cold start is covered in
[`../../01-hardware-gpu-fabric/`](../../01-hardware-gpu-fabric/).

---

## 9. Multi-LoRA, multimodal and hybrid models in brief

### 9.1 Multi-LoRA

- **Config** (`vllm/config/lora.py: LoRAConfig`): `--enable-lora`, `--max-loras` (default **1**:
  adapters in one batch), `--max-lora-rank` (default 16), `--max-cpu-loras` (host cache, ≥
  `max_loras`), `--fully-sharded-loras`, `--lora-target-modules`.
- **Requests** carry a `LoRARequest(lora_name, lora_int_id, lora_path)`; `lora_int_id` "must be
  globally unique for a given adapter. This is currently not enforced" (`vllm/lora/request.py`).
- **Scheduling**: the waiting pass refuses a request whose adapter would push the step past
  `max_loras` distinct adapters and parks it in `skipped_waiting` (Section 3.3). With the default of
  1, two tenants on different adapters alternate steps rather than share them, a frequent surprise.
- **Memory**: `LRUCacheLoRAModelManager` keeps up to `max_loras` adapters in GPU slots and more in
  host memory, evicting LRU (`vllm/lora/model_manager.py`, `worker_manager.py:
  LRUCacheWorkerLoRAManager`).
- **Compute**: the base GEMM runs once for the whole batch; per-token adapter deltas are applied by
  Triton `lora_shrink`/`lora_expand` kernels indexed by each token's adapter slot
  (`vllm/lora/punica_wrapper/punica_gpu.py: PunicaWrapperGPU`). `cudagraph_specialize_lora`
  (default True) captures separate graphs with and without active adapters.
- **Isolation**: the adapter name is an extra key in every block hash, so prefixes are never shared
  across adapters (Section 4.3).
- **Runtime loading**: `POST /v1/load_lora_adapter` and `/v1/unload_lora_adapter`, only when
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set (`vllm/entrypoints/serve/lora/api_router.py`).

### 9.2 Multimodal

The renderer turns images, audio or video into placeholder tokens plus processed tensors and a
content hash per item (`MultiModalHasher`, `vllm/multimodal/hasher.py`); processed items can be
cached in the front-end and mirrored in EngineCore so repeated images are not re-sent
(`mm_receiver_cache` in `EngineCore`, `--mm-processor-cache-gb`). In the engine, each item's encoder
run is scheduled like tokens: `_try_schedule_encoder_inputs` spends an encoder compute budget and
allocates space in the `EncoderCacheManager`, which caches encoder outputs by item hash, shares them
across requests, and evicts unreferenced entries oldest-first (`vllm/v1/core/encoder_cache_manager.py`).
Chunked prefill can split a prompt inside an image's placeholder range unless
`--disable-chunked-mm-input` is set (`SchedulerConfig.disable_chunked_mm_input`). Prefix-cache
hashes include the item identifier and its offset within the block (Section 4.3), so the same text
with a different image never hits. Profiling runs the encoder at its worst case, which is why large
`--limit-mm-per-prompt` values shrink the KV pool.

### 9.3 Hybrid and sliding-window models

Covered in Section 4.8: layers are grouped by cache type, all groups share one block pool, sliding
windows free out-of-window blocks, and Mamba-style layers keep fixed-size state per request with
block-aligned checkpoints for prefix caching. Whether a given hybrid model can use Model Runner V2 or
prefix caching is decided at startup and logged.

---

## 10. Disaggregation and KV transfer

### 10.1 The connector interface

`KVConnectorBase_V1` (`vllm/distributed/kv_transfer/kv_connector/v1/base.py`) is instantiated twice
per engine: once with `KVConnectorRole.SCHEDULER` inside the scheduler, once with
`KVConnectorRole.WORKER` inside each worker. Its docstring lists the split:

| Side | Method | Called from |
|---|---|---|
| scheduler | `get_num_new_matched_tokens(request, num_computed_tokens)` → (external tokens, load async?) | waiting pass, after the local prefix lookup |
| scheduler | `update_state_after_alloc(request, blocks, num_external_tokens)` | after `allocate_slots` |
| scheduler | `build_connector_meta(scheduler_output)` | end of `schedule()`, attached as `kv_connector_metadata` |
| scheduler | `request_finished(request, block_ids)` → (delay free?, `kv_transfer_params`) | `_free_request` |
| worker | `start_load_kv`, `wait_for_layer_load(layer)` | before and inside the forward |
| worker | `save_kv_layer(layer, ...)`, `wait_for_save()` | inside and after the forward |
| worker | `get_finished` / `get_transfer_results` | reported back in `kv_connector_output` |

Configured with `--kv-transfer-config '{"kv_connector": "...", "kv_role": "kv_producer" |
"kv_consumer" | "kv_both", ...}'` (`vllm/config/kv_transfer.py`). Registered connectors
(`kv_connector/factory.py`): `NixlConnector` (alias of `NixlPullConnector`), `NixlPushConnector`,
`LMCacheConnectorV1`, `LMCacheMPConnector`, `MooncakeConnector`, `MooncakeStoreConnector`,
`OffloadingConnector`, `SimpleCPUOffloadConnector`, `FlexKVConnectorV1`, `HF3FSKVConnector`,
`MoRIIOConnector`, `MultiConnector` (chain several), and example/bench connectors.

### 10.2 Prefill/decode disaggregation with NIXL, step by step

From `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` (pull mode, the default
`NixlConnector`):

1. A router sends the request to a **prefill** instance with `kv_transfer_params =
   {"do_remote_decode": true}` and `max_tokens = 1` `(verify: the proxy's exact convention)`.
2. The prefill engine computes the prompt, finishes, and `request_finished` returns
   `delay_free_blocks = True` plus `kv_transfer_params` with `do_remote_prefill`, `remote_block_ids`,
   `remote_engine_id`, `remote_host`, `remote_port`: the prefill instance keeps the blocks pinned
   under a lease (`_kv_lease_duration`) instead of freeing them.
3. The router forwards those params to a **decode** instance. There,
   `get_num_new_matched_tokens` returns `(prompt tokens − local hits, load_async = True)`; the
   scheduler allocates blocks, marks the request `WAITING_FOR_REMOTE_KVS` and moves on
   (Section 3.3, `load_kv_async`).
4. The decode worker's connector issues RDMA **reads** of the remote blocks via NIXL. When
   `finished_recving` arrives, `_update_waiting_for_remote_kv` caches the loaded blocks and the
   request becomes schedulable; its first decode step starts from a full KV cache.
5. The decode side notifies the prefill side, which frees the blocks; the lease frees them anyway if
   the notification never comes. Failed loads follow `kv_load_failure_policy` (default recompute:
   `Scheduler.__init__`, `_handle_invalid_blocks`).

The TP layouts of the two sides may differ; the connector maps them (`nixl/tp_mapping.py`).
Routing, P:D ratios and when disaggregation pays off belong to the orchestration layer
([`../../05-orchestrator/`](../../05-orchestrator/)).

### 10.3 Offloading and shared caches

`--kv-offloading-size <GiB>` with `--kv-offloading-backend native|lmcache` adds a CPU KV tier through
the connector interface (`vllm/config/cache.py: CacheConfig.kv_offloading_size`); `OffloadingConnector`
and `LMCacheConnectorV1` use the same scheduler hooks, reporting external hits through
`get_num_new_matched_tokens`. External hits are counted separately:
`vllm:external_prefix_cache_queries` / `vllm:external_prefix_cache_hits`.

### 10.4 Worked numbers: is the transfer worth it?

Llama-3.1-8B, 4,000-token prompt, BF16 KV:

```
KV to move        = 4,000 × 131,072 B = 524 MB (500 MiB)
prefill compute   ≈ 2 × 8.03e9 params × 4,000 tokens = 64 TFLOP ≈ 107 ms at an assumed 600 TFLOP/s on H100
transfer at 100 GB/s (NVLink-class)       ≈   5.2 ms
transfer at  50 GB/s (400 Gb/s RDMA)      ≈  10.5 ms
transfer at  12.5 GB/s (100 Gb/s)         ≈  41.9 ms
transfer at   1.25 GB/s (10 Gb/s TCP)     ≈ 419   ms   (four times the prefill it saves)
```

Over RDMA the transfer is a tenth of the prefill and overlaps with other work; over commodity TCP it
dominates. FP8 KV halves every transfer figure.

---

## 11. Engine arguments that matter, and what each trades

Defaults are for `vllm serve` on `main` at `5840d95`; "GPU-dependent" follows
`EngineArgs.get_batch_defaults` (Section 3.2).

| Flag | Default | Mechanism | TTFT | ITL | Throughput | Memory |
|---|---|---|---|---|---|---|
| `--gpu-memory-utilization` | 0.92 | `requested = total × util`; KV gets what is left (`request_memory`, `determine_available_memory`) | — | — | ↑ more concurrency | ↑ OOM risk with co-tenants |
| `--kv-cache-memory-bytes` | unset | fixes KV bytes, skips profiling | — | — | — | exact control |
| `--max-model-len` | model's `max_position_embeddings`; `-1`/`auto` fits memory | per-request cap; startup requires one max-length request to fit | — | — | — | ↓ lets small GPUs start |
| `--max-num-batched-tokens` | 2048 / 8192 / 16384 (GPU-dependent) | per-step token budget; profile size | ↓ under load (fewer chunks) | ↑ longer mixed steps | ↑ | ↑ activations → ↓ KV |
| `--max-num-seqs` | 256 / 1024 | running slots, sampler buffers, capture-size ceiling | — | ↑ at high occupancy | ↑ | ↑ runner buffers |
| `--max-num-active-seqs` | unset (= max-num-seqs) | admission-only cap | ↑ queueing | ↓ smaller decode batches | ↓ | — |
| `--long-prefill-token-threshold` | 0 (off) | cap per-request chunk | ↓ for short prompts behind long ones, ↑ for the long one | ↓ spikes | ≈ | — |
| `--enable-chunked-prefill` | on (decoder models) | split prompts across steps | ≈ | ↓ | ↑ | — |
| `--enable-prefix-caching` | on | block-hash reuse (Section 4) | ↓↓ on shared prefixes | — | ↑ | hashing CPU; blocks stay cached |
| `--prefix-caching-hash-algo` | `sha256` | hash function | — | — | xxhash faster | — |
| `--block-size` | 16 (backend may choose) | tokens per block | hit granularity | — | — | waste ≤ 15 tokens/request; may exclude backends |
| `--kv-cache-dtype` | `auto` (model dtype) | `fp8` halves KV bytes | — | ↓ slightly (fewer bytes read) | ↑ (2× blocks) | ↓ ; accuracy to check |
| `--dtype` | `auto` (BF16, FP16 where BF16 is unsupported) | weights/activations dtype | — | — | — | — |
| `--quantization` | from checkpoint | weight format and kernels (Section 8) | ↓ with FP8 | ↓ | ↑ | ↓ weights → ↑ KV |
| `--tensor-parallel-size` | 1 | shard layers; 2 all-reduces/layer | ↓ | ↓ if fast interconnect | per-GPU ↓ | weights and KV split |
| `--pipeline-parallel-size` | 1 | layer stages, `pp_size` batches in flight | ↑ | ↑ | ↑ | weights split |
| `--data-parallel-size` | 1 | independent engines | ↓ queueing | — | ↑ | per-engine copies |
| `--async-scheduling` | on when compatible | overlap CPU scheduling with GPU (Section 3.8) | ↓ | ↓ | ↑ | — |
| `--scheduling-policy` | `fcfs` | `priority` orders and preempts by `priority` | per class | per class | — | — |
| `--watermark` | 0.0 | free-block headroom at admission | ↑ queueing | ↓ preemption churn | ≈ | — |
| `--scheduler-reserve-full-isl` | true | admit only if the whole prompt fits | ↑ slightly | ↓ thrash | ≈ | — |
| `--enforce-eager` / `-O0` | off / `-O2` | no compile, no CUDA graphs | startup ↓↓ | ↑ (launch overhead) | ↓ | ↓ graph memory |
| `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | formula in 5.5 | which batch sizes replay graphs | — | ↓ inside captured range | — | ↑ per graph; startup ↑ |
| `--performance-mode` | `balanced` | `throughput` doubles batch defaults; `interactivity` captures sizes 1–32 | — | `interactivity` ↓ | `throughput` ↑ | — |
| `--speculative-config` | unset | draft + verify (Section 7.4) | — | ↓ at low batch | ↓ at high batch | drafter weights + KV |
| `--stream-interval` | 1 | tokens per streamed chunk | — | smoother at 1 | ↑ with larger values | — |
| `--max-num-queued-reqs`, `--max-num-queued-tokens` | unset | API-server admission, 503 when full | bounded | — | — | — |
| `--api-server-count` | 1 (DP size with internal LB) | more front-end processes | ↓ when tokenization-bound | ↓ | ↑ | CPU |
| `--enable-lora`, `--max-loras`, `--max-lora-rank` | off, 1, 16 | Section 9.1 | — | ↑ with LoRA kernels | adapter mix | slots |
| `--kv-offloading-size` | unset | CPU KV tier (Section 10.3) | ↓ on re-use | — | ↑ hit rate | host RAM |
| `--kv-transfer-config` | unset | P/D connector (Section 10) | — | ↓ interference | — | — |
| `--attention-backend` | auto (Section 6.3) | force a backend | — | — | — | — |

The one-line summary a reviewer expects: **capacity** is `gpu_memory_utilization`, weight and KV
dtypes, `max_model_len`; **latency shape** is `max_num_batched_tokens`,
`long_prefill_token_threshold`, speculation; **concurrency** is `max_num_seqs` bounded by capacity;
**host overhead** is async scheduling, CUDA graphs, `stream_interval`, API-server count.

---

## 12. Observability

### 12.1 Prometheus metrics

All defined in `vllm/v1/metrics/loggers.py: PrometheusStatLogger` (names match `FACTS.md`);
every metric carries `model_name` and `engine` labels; counters are exposed with a `_total` suffix
by `prometheus_client`.

| Metric | Type | Meaning (from the source) | Use it for |
|---|---|---|---|
| `vllm:num_requests_running` | gauge | requests in model execution batches (`len(running)`) | occupancy vs `max_num_seqs` |
| `vllm:num_requests_waiting` | gauge | waiting + skipped-waiting requests | saturation; autoscaling signal |
| `vllm:num_requests_waiting_by_reason` | gauge, `reason` = `capacity` / `deferred` | deferred = LoRA budget, KV transfer, blocked status | tells "full" from "blocked" |
| `vllm:kv_cache_usage_perc` | gauge 0–1 | `BlockPool.get_usage`: referenced blocks only | headroom; not cache fullness (Section 4.6) |
| `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` | counters, tokens | token-weighted lookups and hits at admission | hit rate = rate(hits)/rate(queries) |
| `vllm:external_prefix_cache_queries/_hits` | counters, tokens | hits served by a KV connector | offload / cross-instance value |
| `vllm:num_preemptions` | counter | cumulative preemptions | KV pressure (Section 3.7) |
| `vllm:prompt_tokens`, `vllm:generation_tokens` | counters | prefill and generated tokens | throughput |
| `vllm:prompt_tokens_by_source`, `vllm:prompt_tokens_cached` | counters | prompt tokens split by computed / local cache / external | effective prefill work |
| `vllm:iteration_tokens_total` | histogram | tokens per engine step | how full steps are vs the budget |
| `vllm:time_to_first_token_seconds` | histogram | arrival → first token, measured in the front-end | TTFT SLO |
| `vllm:inter_token_latency_seconds` | histogram | per-iteration gaps between tokens | ITL SLO |
| `vllm:request_time_per_output_token_seconds` | histogram | per request `decode_time / (n − 1)` | TPOT |
| `vllm:e2e_request_latency_seconds` | histogram | arrival → finish | end-to-end |
| `vllm:request_queue_time_seconds`, `_prefill_time_`, `_decode_time_`, `_inference_time_` | histograms | phase durations from engine events (Section 2.4) | where TTFT goes |
| `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:request_params_max_tokens`, `vllm:request_params_n`, `vllm:request_max_num_generation_tokens` | histograms | per-request shape | workload characterization |
| `vllm:request_prefill_kv_computed_tokens` | histogram | new KV tokens computed in prefill, excluding cached | real prefill cost |
| `vllm:request_num_preemptions` | histogram | preemptions per request | tail-latency cause |
| `vllm:request_success` | counter, `finished_reason` | finished requests by `stop`/`length`/`abort`/`error`/`repetition` | error and truncation rates |
| `vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`, `_num_accepted_tokens_per_pos` | counters | speculative proposals and acceptances (`vllm/v1/spec_decode/metrics.py`) | acceptance length = 1 + accepted / drafts |
| `vllm:kv_block_lifetime_seconds`, `_idle_before_evict_`, `_reuse_gap_` | histograms | sampled block residency (`--kv-cache-metrics`) | tuning cache size for reuse |
| `vllm:mm_cache_queries/_hits` | counters, items | multimodal processor cache | image re-use |
| `vllm:lora_requests_info` | gauge | running/waiting adapters | LoRA mix |
| `vllm:cache_config_info` | gauge fixed at 1 (emulated Info) | `CacheConfig` fields as labels (block size, num_gpu_blocks, …) | confirm the KV pool from the scrape |
| `vllm:corrupted_requests` | counter | requests with NaN logits (`VLLM_COMPUTE_NANS_IN_LOGITS`) | numerical faults |
| `vllm:engine_sleep_state` | gauge | sleep level | RL weight-swap flows |

Queries worth keeping: prefix hit rate
`rate(vllm:prefix_cache_hits_total[5m]) / rate(vllm:prefix_cache_queries_total[5m])`; TTFT p99
`histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))`;
preemption rate `rate(vllm:num_preemptions_total[5m])`; step fullness via the
`vllm:iteration_tokens_total` histogram against `max_num_batched_tokens`.

### 12.2 Log lines to read

At startup (in order):

| Line | Source | What it tells you |
|---|---|---|
| `Initializing a V1 LLM engine (v…) with config: …` | `EngineCore.__init__` | the fully resolved config (read it once per deployment) |
| `Model Runner V2 does not yet support …; using the V1 model runner instead` | `VllmConfig.use_v2_model_runner` | which runner |
| `Using … attention backend out of potential backends: …` (or `Using … backend.` when forced) | `CudaPlatformBase.get_attn_backend_cls` | attention backend |
| `Chunked prefill is enabled with max_num_batched_tokens=…` | `SchedulerConfig.__post_init__` | step budget |
| `Loading weights took … seconds`, `Model loading took … GiB memory and … seconds` | loader, V1 runner | cold-start split |
| `Compiling a graph for compile range … takes … s` or `Directly load the compiled graph(s) …` | `vllm/compilation/backends.py` | compile cost, cache hit |
| `Available KV cache memory: … GiB` | `Worker.determine_available_memory` | the number Section 4.7 predicts |
| `CUDA graph memory profiling is enabled … increase --gpu-memory-utilization to …` | same | graph memory accounting |
| `GPU KV cache size: … tokens, Maximum concurrency for … tokens per request: …x` | `update_kv_cache_capacity` | capacity |
| `Graph capturing finished in … secs, took … GiB` | `GPUModelRunner.capture_model` | graph cost |
| `init engine (profile, create kv cache, warmup model) took … s` | `EngineCore._initialize_kv_caches` | total engine startup |

At runtime, every `VLLM_LOG_STATS_INTERVAL` seconds (default 10.0; off with `--disable-log-stats`;
`LoggingStatLogger.log`):

```
Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: R reqs,
Waiting: W reqs, [Deferred: D reqs,] [Preemptions: P,] GPU KV cache usage: U%, Prefix cache hit rate: H%
```

followed by speculative-decoding acceptance lines when speculation is on. Rules of thumb: Waiting > 0
with KV usage near 100% means capacity-bound; Waiting > 0 with KV usage low means budget-bound
(`max_num_seqs`, `max_num_batched_tokens`) or blocked (check Deferred); nonzero Preemptions means
the pool is too small for the admitted mix.

---

## 13. Reading and debugging vLLM

### 13.1 Put everything in one process

The fastest way to understand a mechanism is to stop in it. Three settings collapse vLLM into one
debuggable process (T1: needs a GPU; a 0.6B model fits anywhere):

```python
# debug_one_process.py  (T1)
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # LLMEngine uses InprocClient: EngineCore in this process
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"           # default INFO
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True,   # no torch.compile, no CUDA graphs: steppable model code
          max_model_len=2048, gpu_memory_utilization=0.5)  # TP=1 → UniProcExecutor: worker in this process too
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print(out[0].outputs[0].text)
```

Why each line works: `LLMEngine.from_engine_args` passes `multiprocess_mode =
envs.VLLM_ENABLE_V1_MULTIPROCESSING` to `EngineCoreClient.make_client`, which returns an
`InprocClient` when it is false (`vllm/v1/engine/llm_engine.py`, `core_client.py`); with world size
1 the executor is `uni` (Section 5.1); `enforce_eager` disables compilation and graphs, which would
otherwise hide Python frames. `AsyncLLM` (the server path) always runs EngineCore in a separate
process ("Running EngineCore in asyncio without multiprocessing is not currently supported",
`make_client`), so debug the server with logs and profiles, and the engine with the snippet above.

### 13.2 Where to put breakpoints

| Question | Breakpoint |
|---|---|
| Why is my request still waiting? | `Scheduler.schedule` (waiting loop), `KVCacheManager.allocate_slots` returning `None`, `Scheduler._try_promote_blocked_waiting_request` |
| What did the prefix cache find? | `KVCacheManager.get_computed_blocks`, `FullAttentionManager.find_longest_cache_hit` |
| Which blocks got evicted? | `BlockPool.get_new_blocks` → `_maybe_evict_cached_block` |
| Who got preempted and why? | `Scheduler._preempt_request` |
| What exactly goes to the GPU? | `GPUModelRunner._prepare_inputs` (positions, `query_start_loc`), `_build_attention_metadata`, `BlockTable.compute_slot_mapping` |
| Which attention kernel / GEMM kernel? | `CudaPlatformBase.get_attn_backend_cls`, `choose_mp_linear_kernel` |
| Why did generation stop? | `check_stop` (`vllm/v1/core/sched/utils.py`), `IncrementalDetokenizer.update` (stop strings) |
| Why is a token masked? | `apply_grammar_bitmask`, `StructuredOutputManager.grammar_bitmask` |
| How much memory went where at startup? | `Worker.determine_available_memory`, `get_kv_cache_configs` |

Two cautions. A breakpoint in `Scheduler.schedule` with async scheduling stops the loop while a step
is in flight on the GPU; with `VLLM_ENGINE_ITERATION_TIMEOUT_S` (default 60) long pauses can trigger
timeouts in multi-process setups. And code inside compiled, graph-captured regions runs once at
capture and then replays, so a breakpoint there fires during warm-up, not during serving.

### 13.3 Environment variables that matter (and ones that no longer exist)

| Variable | Effect (`vllm/envs.py`) |
|---|---|
| `VLLM_LOGGING_LEVEL` | default `INFO`; `DEBUG` prints scheduler/executor details, including the per-backend rejection reasons of the attention selector |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | default 1; 0 runs EngineCore in-process for the offline `LLM` class |
| `VLLM_USE_V2_MODEL_RUNNER` | unset = config default; 0/1 forces the runner (Section 1.4) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 stops subtracting estimated graph memory from the KV budget |
| `VLLM_LOG_STATS_INTERVAL` | seconds between the periodic stats line (default 10.0) |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | count NaNs in logits per request (`vllm:corrupted_requests`), at some cost |
| `VLLM_TRACE_FUNCTION` | 1 traces every Python function call (very slow; for hangs) |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 forces the PyTorch/Triton sampling path |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | enables `/v1/load_lora_adapter` |
| `CUDA_LAUNCH_BLOCKING=1` | CUDA, not vLLM: synchronous launches so errors point at the right kernel (use with `--enforce-eager`) |
| `VLLM_USE_V1` | **gone**: V1 is the only engine |
| `VLLM_ATTENTION_BACKEND` | **gone**: use `--attention-backend` |
| `VLLM_TORCH_PROFILER_DIR` | **gone**: use `--profiler-config` |

### 13.4 Profiling

- **torch profiler**: `vllm serve MODEL --profiler-config '{"profiler": "torch", "torch_profiler_dir":
  "/abs/path"}'`, then `curl -X POST localhost:8000/start_profile`, send load, `curl -X POST
  localhost:8000/stop_profile`. Traces are written for the API server (CPU) and each worker (CPU and
  CUDA) (`vllm/config/profiler.py: ProfilerConfig`; routes in
  `vllm/entrypoints/serve/profile/api_router.py`, attached only when a profiler is configured). Open
  them in Perfetto; look for gaps between GPU kernels (host overhead) and for the
  `execute_model`/`sample_tokens` split.
- **Nsight Systems**: `--enable-layerwise-nvtx-tracing` adds NVTX ranges per layer
  (`ObservabilityConfig`, `GPUModelRunner._register_layerwise_nvtx_hooks`); run under
  `nsys profile -t cuda,nvtx`. Other options: `profiler: "cuda"` (CUDA profiler API ranges) and
  `profiler: "proton"` (Triton Proton).
- **Per-step details**: `--enable-logging-iteration-details` attaches per-iteration scheduling
  details to stats (`EngineCore.capture_iteration_details`).

### 13.5 Common errors

| Symptom | Cause | Fix | Source of the message |
|---|---|---|---|
| `To serve at least one request with the model's max seq len (N), (X GiB KV cache is needed, which is larger than the available KV cache memory (Y GiB)` | default `max_model_len` too long for the KV pool (Section 4.7) | `--max-model-len` lower or `auto`, `--kv-cache-dtype fp8`, quantized weights, larger GPU/TP | `_check_enough_kv_cache_memory` |
| `No available memory for the cache blocks` | weights + activations exceed `requested` | raise `--gpu-memory-utilization`, lower `--max-num-batched-tokens`/`--max-num-seqs`, quantize | same |
| `Free memory on device ... on startup is less than desired GPU memory utilization` | another process holds GPU memory | lower utilization or isolate the GPU | `request_memory` |
| `Error in memory profiling. Initial free memory ..., current free memory ...` | another process released memory during profiling | isolate vLLM in its own container | `determine_available_memory` |
| CUDA OOM during graph capture or first requests | estimates too tight | lower `--gpu-memory-utilization` a little, cap `--max-cudagraph-capture-size`, or `--enforce-eager` to confirm | — |
| `no kernel image is available for execution on the device` | binaries lack SASS/PTX for your compute capability: e.g. the official image builds `TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` with CUDA 13.0.3 (`docker/Dockerfile`), so SM 7.0 (V100) is absent; or a wheel older than your GPU | use a build that includes your arch, or build from source with `TORCH_CUDA_ARCH_LIST` set; compare `torch.cuda.get_device_capability()` with `torch.cuda.get_arch_list()` | CUDA runtime |
| On a T4: `Your device ... doesn't support torch.bfloat16. Falling back to torch.float16` | T4 is SM 7.5; `supported_dtypes` excludes BF16 below SM 8.0 | expected with `--dtype auto`; `--dtype bfloat16` instead raises "Bfloat16 is only supported on GPUs with compute capability of at least 8.0 ... --dtype=half" | `_resolve_auto_dtype` (`vllm/config/model.py`), `CudaPlatformBase.check_if_supports_dtype` |
| On a T4: attention is slower than expected | FlashAttention needs SM ≥ 8.0 and FlashInfer is floored at 8.0, so Triton attention is used (Section 6.3) | expected; FP8 KV and FP8 weights also lack hardware support there (Marlin weight-only FP8 fallback) | backend `supports_compute_capability` |
| Waiting grows, `kv_cache_usage_perc` ≈ 1, `num_preemptions` rising | too few KV blocks for the admitted mix (Section 3.7) | capacity (FP8 KV, shorter context, more GPUs), fewer `max_num_seqs`, `--watermark` | metrics |
| Waiting grows, KV usage low | budget-bound or blocked (LoRA cap, grammar compile, remote KV) | check `num_requests_waiting_by_reason`, raise `--max-num-seqs`/`--max-loras` | metrics |
| First request after start is slow | lazy JIT/compile or cold caches | warm-up requests; keep the compile cache (`cache_dir`) on persistent storage | logs |

---

## 14. vLLM vs SGLang vs TensorRT-LLM

All three now share the same core ideas (paged KV, continuous batching with chunked prefill, CUDA
graphs, speculative decoding, prefill/decode disaggregation, FP8/FP4 quantization), so the useful
comparison is architectural. vLLM facts below come from this primer; SGLang and TensorRT-LLM facts
come from their READMEs fetched on 2026-09-26 (`sgl-project/sglang`, `NVIDIA/TensorRT-LLM`), and
anything more specific is `(verify)`.

| Dimension | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Execution | PyTorch model code, `torch.compile` (Inductor) pieces + CUDA graphs, custom CUDA/Triton kernels | PyTorch runtime with CUDA graphs; "zero-overhead CPU scheduler" (README) | "Architected on PyTorch" with a Python LLM API; custom kernels for attention, GEMM, MoE (README); historically ahead-of-time TensorRT engines |
| Prefix reuse | block-hash chain, LRU free queue (Section 4) | RadixAttention: a radix tree over token sequences (README; SGLang paper) | "KV cache reuse" (README news); structure `(verify)` |
| Scheduling | unified token budget, async scheduling | overlapped CPU scheduling (README) | in-flight batching, chunked context `(verify)` |
| Structured output | bitmask from xgrammar/guidance/outlines, overlapped with the forward | "compressed finite state machine" (README news, 2024) and grammar backends `(verify)` | guided decoding combined with speculation (README news) |
| Disaggregation | KV connector API: NIXL, LMCache, Mooncake, … | prefill-decode disaggregation, large-scale EP (README) | disaggregated serving, wide EP (README) |
| Hardware | NVIDIA, AMD, TPU, CPU, XPU and platform plugins | NVIDIA, AMD, Intel Xeon, Google TPU, Ascend (README) | NVIDIA only |
| Serving entry | `vllm serve` (OpenAI-compatible) | OpenAI-compatible server | `trtllm-serve`; integrates with Dynamo and Triton Inference Server (README) |

When each fits, as a design argument rather than a benchmark claim:

- **vLLM** when breadth matters: the widest model and hardware coverage, a stable OpenAI surface,
  first-class integration points for routers and caches (KV events, connectors, metrics), and a
  codebase you can read and patch in Python. Most orchestration layers (llm-d, Dynamo, KServe)
  support it first.
- **SGLang** when requests share structure heavily (agents, multi-turn, tree search, many
  structured-output calls): the radix tree makes partial-prefix reuse natural, and it is widely used
  as an RL rollout engine (README). Also a strong default for large MoE deployments.
- **TensorRT-LLM** when the fleet is NVIDIA-only and the last 10–30% of per-GPU performance is worth
  a narrower, vendor-tied stack; it is the engine NVIDIA optimizes first for new GPUs `(verify)`.

Measure on your own workload: the ranking between these engines changes release to release, and the
serving lab's benchmark harness ([`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/))
is built for exactly that comparison against an OpenAI-compatible endpoint.

---

## In a design review

### The two-minute walkthrough

"A request hits the API server, which templates and tokenizes it and sends an `EngineCoreRequest`
over ZMQ to the EngineCore process. The EngineCore input thread builds a `Request` and hashes its
prompt in 16-token blocks, each hash chained to the previous one, so the hash of block *k* names the
whole prefix up to *k*.

"Every step, the scheduler hands out a token budget, 2,048 tokens by default on a 24 GB card. Running
requests go first; they are mostly decodes needing one token. Whatever budget is left goes to waiting
requests: first we look up the longest cached prefix, then we check that the whole prompt fits in
free blocks, then we schedule as many of its tokens as the budget allows. That is chunked prefill:
no phases, just requests catching up to their length.

"If a running request needs a block and none is free, we preempt the most recently admitted request,
free its blocks and put it back at the head of the queue. Freed blocks keep their hashes and sit in an
LRU free list, tail of each prefix chain first, so a preempted or repeated request usually finds its
blocks again.

"The GPU side gets only a diff: new requests, new tokens, new block ids. It builds positions and a
slot mapping, runs a compiled model that replays CUDA graphs, full graphs for pure decode batches,
piecewise ones around attention otherwise, and one varlen attention call per layer over the paged
cache. Sampling happens in a second call so the CPU can build grammar masks during the forward pass.
The scheduler appends tokens, checks EOS and length, and ships outputs back; the API server
detokenizes, checks stop strings, and streams SSE.

"Capacity is gpu_memory_utilization times memory minus weights, activations and graphs, divided by
bytes per block: about 2,169 blocks, 35k tokens, for an 8B BF16 model on an L4. Latency shape is the
token budget; concurrency is max_num_seqs bounded by that capacity."

### Drill questions

1. **We doubled `--max-num-batched-tokens` and p99 ITL got worse. Why?** Decodes share steps with
   prefill chunks; a bigger budget means longer mixed steps, and every co-scheduled decode waits for
   the whole step. The profiling pass also ran at the larger size, so the KV pool shrank and
   preemptions may have increased. Lower the budget, set `--long-prefill-token-threshold`, or move
   prefill to separate instances.

2. **KV usage reads 35% but the prefix hit rate fell after we added replicas. Is the gauge wrong?**
   No. `kv_cache_usage_perc` counts only blocks referenced by live requests; cached blocks in the
   free queue count as free. The hit rate fell because traffic now spreads across more caches, each
   seeing fewer repeats. The fix is prefix-aware routing, not more memory.

3. **Estimate KV capacity for Llama-3.1-8B BF16 on an L4 at the default utilization.** KV is
   2 × 32 × 8 × 128 × 2 B = 128 KiB per token, 2 MiB per 16-token block. 22.5 GiB × 0.92 = 20.7 GiB;
   minus 15.0 GiB weights and roughly 1.5 GiB of activations and graphs leaves about 4.2 GiB, about
   2,169 blocks or 35k tokens. The model's 131k default context needs 16 GiB for one request, so the
   engine refuses to start until `--max-model-len` is reduced (16k gives 2.1× concurrency).

4. **Why recompute on preemption instead of swapping to CPU?** V1's scheduler has no swap path;
   recompute is one prefill pass, and with prefix caching the victim's full blocks keep their hashes
   in the free queue, so re-admission usually recomputes one block. A CPU KV tier exists, but as a
   connector (`--kv-offloading-size`), not as preemption.

5. **A prompt is fully cached. Why does vLLM still compute 16 tokens of it?** The last prompt token
   must go through the model to produce logits, so the hit is capped at `num_tokens − 1`, and hits are
   whole blocks, so the final block is recomputed.

6. **Where are stop conditions checked?** EOS, `stop_token_ids`, `max_tokens` and `max_model_len` in
   the engine (`check_stop`); stop strings in the API server's incremental detokenizer, which then
   aborts the request in the engine. A token or two may be computed past a stop string.

7. **Two tenants share a system prompt but use different LoRA adapters. Do they share KV?** No: the
   adapter name is an extra key in every block hash. And with the default `--max-loras 1` their
   requests cannot even share a step; the scheduler parks the second adapter's requests until the
   next step.

8. **Does speculative decoding change what the model outputs?** No. The rejection sampler accepts a
   draft token with probability `min(1, p/q)` and resamples from the residual otherwise, which
   preserves the target distribution; with greedy drafts `q` is one-hot, and greedy targets reduce
   to exact match. It changes speed: a good win at small batch where decode is bandwidth-bound,
   shrinking at large batch where verification FLOPs are no longer free.

---

## Glossary

| Term | Meaning here |
|---|---|
| API server | front-end process: FastAPI routes, `AsyncLLM`, tokenization/detokenization, SSE |
| EngineCore | process running the scheduler, KV cache manager and executor loop (`vllm/v1/engine/core.py`) |
| `EngineCoreRequest` / `EngineCoreOutputs` | msgspec messages between API server and EngineCore |
| `SchedulerOutput` | per-step plan sent to workers: new requests, deltas, tokens per request |
| token budget | `max_num_scheduled_tokens` (= `max_num_batched_tokens`): tokens computed per step across all requests |
| `num_computed_tokens` | how far a request's KV is computed; the scheduler moves it toward `num_tokens_with_spec` |
| chunked prefill | computing a prompt over several steps within the token budget |
| preemption | freeing a running request's blocks and requeueing it for recompute |
| KV block | fixed group of `block_size` token slots (default 16) for every layer of a KV cache group |
| null block | block 0, a placeholder never allocated to real tokens |
| block hash | hash of (parent hash, block token ids, extra keys); identifies a prefix |
| extra keys | LoRA name, multimodal item id and offset, cache salt (first block), prompt-embeds digest |
| free block queue | doubly linked list of free blocks in reuse order; cached blocks at the tail |
| KV cache group | set of layers with the same cache type and block table (hybrid models have several) |
| `gpu_memory_utilization` | fraction of total GPU memory this instance may use (default 0.92) |
| slot mapping | per token, the flat index `block_id × block_size + offset` where its K/V are written |
| persistent batch | MRV1's `InputBatch`, updated incrementally across steps |
| Model Runner V2 | `vllm/v1/worker/gpu/`: slot-based GPU request state, Triton input preparation |
| piecewise CUDA graph | graphs over the compiled pieces between attention ops |
| full CUDA graph | one graph for the whole forward, used for uniform decode batches |
| capture size | batch token count for which a graph is captured; batches are padded up to one |
| async scheduling | scheduling step N+1 while step N runs, using output placeholders |
| attention backend | pluggable kernel family (FlashAttention, FlashInfer, Triton, MLA variants) |
| cascade attention | attending a shared prefix once for all requests, then merging |
| logits processor | batch-level class that edits logits (min tokens, logit bias, min-p, custom) |
| bitmask | packed allowed-token mask from a grammar, one row per structured request/position |
| rejection sampler | verifies draft tokens against the target distribution |
| bonus token | extra target-sampled token when all drafts are accepted |
| KV connector | plugin that loads/saves KV outside the local pool (P/D transfer, offload, shared caches) |
| P/D disaggregation | prefill and decode on different instances, KV moved between them |

---

## Sources

**vLLM source read for this primer** (repository `vllm-project/vllm`, `main` at `5840d95`,
2026-09-25, fetched 2026-09-26; 93 files):

- Engine and API: `vllm/v1/engine/__init__.py`, `core.py`, `core_client.py`, `async_llm.py`,
  `llm_engine.py`, `input_processor.py`, `output_processor.py`, `detokenizer.py`,
  `parallel_sampling.py`; `vllm/entrypoints/openai/api_server.py`,
  `vllm/entrypoints/openai/chat_completion/{api_router,serving}.py`,
  `vllm/entrypoints/launchers/api_server/entry.py`, `vllm/entrypoints/cli/serve.py`,
  `vllm/entrypoints/serve/profile/api_router.py`, `vllm/entrypoints/serve/lora/api_router.py`.
- Scheduling and requests: `vllm/v1/core/sched/{scheduler,async_scheduler,output,request_queue,utils,interface}.py`,
  `vllm/v1/request.py`, `vllm/v1/outputs.py`, `vllm/sampling_params.py`.
- KV cache: `vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_utils,single_type_kv_cache_manager,kv_cache_coordinator,encoder_cache_manager}.py`,
  `vllm/v1/kv_cache_interface.py`, `vllm/utils/hashing.py`, `vllm/utils/mem_utils.py`.
- Workers and executors: `vllm/v1/worker/{gpu_worker,gpu_model_runner,gpu_input_batch,block_table,utils}.py`,
  `vllm/v1/worker/gpu/{README.md,model_runner.py,states.py,input_batch.py,sample/gumbel.py}`,
  `vllm/v1/executor/{abstract,multiproc_executor,ray_executor,uniproc_executor}.py`,
  `vllm/v1/cudagraph_dispatcher.py`, `vllm/distributed/device_communicators/{shm_broadcast,cuda_communicator}.py`,
  `vllm/distributed/parallel_state.py`, `vllm/forward_context.py`.
- Compilation: `vllm/compilation/{backends,piecewise_backend,cuda_graph,decorators}.py`,
  `vllm/config/compilation.py`.
- Attention: `vllm/v1/attention/{backend,selector}.py`,
  `vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn,fa_utils,utils}.py`,
  `vllm/v1/attention/backends/mla/{flashmla,triton_mla}.py`, `vllm/platforms/cuda.py`,
  `vllm/model_executor/layers/attention/attention.py`.
- Sampling, structured output, speculation: `vllm/v1/sample/{sampler,rejection_sampler}.py`,
  `vllm/v1/sample/ops/topk_topp_sampler.py`, `vllm/v1/sample/logits_processor/{__init__,builtin,interface}.py`,
  `vllm/v1/structured_output/{__init__,utils}.py`, `vllm/v1/spec_decode/{eagle,ngram_proposer,llm_base_proposer}.py`.
- Quantization and loading: `vllm/model_executor/layers/quantization/{__init__,fp8,auto_gptq,auto_awq}.py`,
  `vllm/model_executor/kernels/linear/__init__.py`, `.../mixed_precision/{marlin,machete}.py`,
  `vllm/model_executor/model_loader/{__init__,base_loader,default_loader}.py`, `vllm/config/{quantization,load}.py`.
- LoRA, multimodal, KV transfer: `vllm/config/lora.py`, `vllm/lora/{request,model_manager,worker_manager}.py`,
  `vllm/lora/punica_wrapper/punica_gpu.py`, `vllm/multimodal/hasher.py`,
  `vllm/distributed/kv_transfer/kv_connector/{factory.py,v1/base.py,v1/nixl/connector.py,v1/nixl/pull_scheduler.py}`,
  `vllm/config/kv_transfer.py`.
- Configuration and metrics: `vllm/config/{vllm,cache,scheduler,parallel,model,speculative,structured_outputs,profiler}.py`,
  `vllm/engine/arg_utils.py`, `vllm/envs.py`, `vllm/v1/metrics/{loggers,stats}.py`,
  `pyproject.toml`, `vllm/version.py`, `CMakeLists.txt`, `docker/Dockerfile`.
- Other engines: `sgl-project/sglang` `README.md`, `NVIDIA/TensorRT-LLM` `README.md` (main, fetched 2026-09-26).

**Papers**

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, arXiv:2309.06180.
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024, arXiv:2403.02310.
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023, arXiv:2211.17192; Chen et al., arXiv:2302.01318.
- Li et al., "EAGLE", arXiv:2401.15077; "EAGLE-3", arXiv:2503.01840; Cai et al., "Medusa", arXiv:2401.10774.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104.
- Zhong et al., "DistServe", arXiv:2401.09670; Patel et al., "Splitwise", arXiv:2311.18677; Qin et al., "Mooncake", arXiv:2407.00079.
- Dao, "FlashAttention-2", arXiv:2307.08691; Shah et al., "FlashAttention-3", arXiv:2407.08608; Ye et al., "FlashInfer", arXiv:2501.01005.
- DeepSeek-AI, "DeepSeek-V2" (MLA), arXiv:2405.04434.
- Dong et al., "XGrammar", arXiv:2411.15100.
- Frantar et al., "GPTQ", arXiv:2210.17323; Lin et al., "AWQ", arXiv:2306.00978; Frantar et al., "MARLIN", arXiv:2408.11743.

**Related material in this repo**: [`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md),
[`../kv-cache/kv-cache-primer.md`](../kv-cache/kv-cache-primer.md),
[`../paged-attention/paged-attention-primer.md`](../paged-attention/paged-attention-primer.md),
[`../flash-attention/flash-attention-primer.md`](../flash-attention/flash-attention-primer.md),
[`../../00-foundations/gpu-capacity-planning/PRIMER.md`](../../00-foundations/gpu-capacity-planning/PRIMER.md).

---

## Verify list (dated 2026-09-26)

| Item | Value used | Why it needs checking |
|---|---|---|
| vLLM state | `main` at `5840d95` (2026-09-25); PyPI latest 0.30.0 (2026-09-22); `torch == 2.13.0` build pin | `main` changes daily; line numbers in `source-map.md` drift |
| Model Runner V2 | default when supported, README still says "[Experimental]" | defaults and the unsupported-feature list change often |
| V0 removal | V1 is the only engine; `VLLM_USE_V1` absent | the release that removed V0 is not recorded here |
| GPU memory seen by CUDA | L4 23,034 MiB; H100 80 GB 81,559 MiB | from typical `nvidia-smi` output, not measured in this session |
| HBM/GDDR bandwidth | L4 300 GB/s; H100 SXM 3.35 TB/s | spec sheets; used only for floors |
| Activation and CUDA-graph memory | 1.0 + 0.5 GiB (L4), 2.0 + 1.0 GiB (H100) | assumptions; read `Available KV cache memory` and `Graph capturing finished ... took X GiB` from logs |
| Step time of a 2,048-token step | "a few tens of ms" on H100 for 8B | measure in the serving lab |
| Same-step prefix hits | blocks are hashed at allocation, so a later request in the same `schedule()` may hit them | behaviour inferred from `allocate_slots`; confirm with a test |
| DeepSeek-V3 MLA shape | 61 layers, latent 512 + RoPE 64 | model config not fetched here |
| NIXL proxy convention | prefill request sent with `max_tokens = 1` and `do_remote_decode` | proxy/router implementations differ |
| GGUF and bitsandbytes loading | not in the loader registry at this commit | locate the current path before relying on it |
| `--enforce-eager` ITL cost | "large on small models, small on large models" | measure |
| SGLang / TensorRT-LLM specifics marked `(verify)` | from READMEs only | check their docs and code before a decision |
