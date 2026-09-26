# vLLM internals: a source-level primer

This primer follows a request through vLLM with every mechanism tied to the file and function that
implements it. It is written for an engineer who has to explain an inference engine in a design review:
why a request waited, where its time went, why the KV cache holds as many tokens as it does, and which flag
moves which number. The engine concepts themselves (continuous batching, chunked prefill, prefix caching,
speculation) are introduced in the serving-engine primer ([`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md));
this is the "now read the real code" companion. To run what is described here on a GPU, use
[`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/).

**State of the source.** vLLM `main` at commit `5840d95` (2026-09-25), fetched 2026-09-26 from
`raw.githubusercontent.com/vllm-project/vllm/main`. The version string comes from git tags via
`setuptools_scm` (`vllm/version.py`, `pyproject.toml`), so `main` carries no fixed number; the latest PyPI
release at fetch time was **0.30.0** (2026-09-22), and `pyproject.toml` pins `torch == 2.13.0` for the
build. Paths were checked at that commit; anything not confirmed in source is marked `(verify)`.

**Conventions.** `(path: Class.method)` means "read it there"; paths are relative to the vLLM repository
root. [`source-map.md`](source-map.md) lists the same files with line numbers at `5840d95` and a reading
plan in three sittings of about two hours. **Tier:** reading this and the source is **T0** (no GPU); observing the behaviour
(metrics, log lines, preemptions) is **T1** in the serving lab.

---

## The one-minute version

vLLM splits serving into two kinds of processes. An **API server** (FastAPI on asyncio) renders the chat
template, tokenizes, builds an `EngineCoreRequest`, and later detokenizes and streams. An **EngineCore**
process runs a tight loop: `Scheduler.schedule()` decides how many tokens each request computes this step,
the executor runs one forward pass on the GPU workers, the sampler picks tokens, and
`Scheduler.update_from_output()` appends them and checks stop conditions. The two sides talk over ZMQ with
msgpack, so tokenization and HTTP never steal time from the GPU loop.

The scheduler has no prefill phase and no decode phase. Each request has `num_computed_tokens` and a target
length; every step hands out a token budget (`max_num_batched_tokens`) first to running requests, then to
waiting ones. A decode is a request one token behind; a prefill is one many tokens behind, chunked to fit.
The KV cache is a pool of 16-token blocks. Full blocks are named by a hash chained over the whole prefix,
so any request starting with the same tokens reuses them; freed blocks stay cached until reallocated, the
tail of each chain evicted first. The pool is whatever remains of `gpu_memory_utilization × memory` after
weights, a profiled activation peak, and CUDA graphs.

On the GPU, the model is compiled with `torch.compile` into pieces split at attention and replayed as CUDA
graphs (full graphs for pure-decode batches, piecewise for mixed ones); attention is a pluggable backend
chosen per GPU generation. After this primer you should be able to trace a request through those classes,
compute a KV block budget by hand, predict when preemption happens, and say which engine argument trades
TTFT against inter-token latency against memory.

---

## 1. Why vLLM and what "V1" changed

### 1.1 The problem

An LLM server is bound by HBM bandwidth during decode (every step re-reads the weights and the KV cache)
and by HBM capacity for the KV cache (it decides how many sequences can be in flight). The arithmetic and
the paging idea are in [`../kv-cache/kv-cache-primer.md`](../kv-cache/kv-cache-primer.md) and
[`../paged-attention/paged-attention-primer.md`](../paged-attention/paged-attention-primer.md); vLLM began
as the reference implementation of PagedAttention (Kwon et al., SOSP 2023). What the engine adds on top of
paging is a scheduler that keeps the batch full, a cache that shares blocks across requests, and a GPU
execution path with almost no per-step host overhead.

### 1.2 What V1 is

"V1" is the re-architecture that is now the only engine in the tree: `VLLM_USE_V1` no longer exists in
`vllm/envs.py`, and V0 survives only in docstring comparisons ("This is different from the V0 sampler",
`vllm/v1/sample/sampler.py: Sampler.forward`). The release that removed V0 is `(verify)`.

| Choice | What it means | Where |
|---|---|---|
| Process split | API server(s) and EngineCore in separate processes, ZMQ + msgpack between them | `vllm/v1/engine/core_client.py: AsyncMPClient`, `vllm/v1/engine/core.py: EngineCoreProc` |
| asyncio front-end | `AsyncLLM` owns tokenization, detokenization, streaming; a background task pulls outputs | `vllm/v1/engine/async_llm.py: AsyncLLM._run_output_handler` |
| Unified scheduler | no phases; requests catch `num_computed_tokens` up to their length | comment atop `vllm/v1/core/sched/scheduler.py: Scheduler.schedule` |
| Chunked prefill, prefix caching on | defaults for decoder-only generative models | `vllm/engine/arg_utils.py: EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`, `vllm/config/cache.py` |
| Async scheduling on | schedule step N+1 while step N runs on the GPU | `vllm/config/vllm.py`, `vllm/v1/core/sched/async_scheduler.py` |
| Persistent GPU-side request state | only per-step deltas cross to the workers | `vllm/v1/worker/gpu_input_batch.py: InputBatch`, `vllm/v1/worker/gpu/states.py: RequestState` |
| torch.compile + CUDA graphs | `CompilationMode.VLLM_COMPILE`, `CUDAGraphMode.FULL_AND_PIECEWISE` by default | `vllm/config/compilation.py` |
| Symmetric workers | the scheduler lives in EngineCore; every worker receives the same `SchedulerOutput` | `vllm/v1/executor/multiproc_executor.py: MultiprocExecutor.collective_rpc` |

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

With one GPU the executor is `UniProcExecutor` and the worker runs **inside** EngineCore; with TP×PP > 1 on
one node it is `MultiprocExecutor` with a process per GPU; Ray is used for placement groups or on request
(`vllm/config/parallel.py: ParallelConfig.__post_init__`, `vllm/v1/executor/abstract.py: Executor.get_class`).
`vllm serve` starts one API server by default, `--api-server-count` defaults to the DP size under internal
load balancing, and `--headless` runs engines with no API server (`vllm/entrypoints/cli/serve.py:
ServeSubcommand.cmd`). Inside EngineCore, socket I/O and msgpack run on two daemon threads so they overlap
the GPU step (`EngineCoreProc.__init__`, `process_input_sockets`, `process_output_sockets`).

### 1.4 Which model runner

`main` has two GPU model runners. `vllm/v1/worker/gpu_model_runner.py: GPUModelRunner` ("MRV1") keeps a
persistent `InputBatch` and builds inputs on the CPU. `vllm/v1/worker/gpu/model_runner.py: GPUModelRunner`
("MRV2") keeps request state in fixed GPU slots and builds inputs with Triton kernels. MRV2's README still
says "[Experimental]", but `VllmConfig.use_v2_model_runner` (`vllm/config/vllm.py`) returns **True by
default** when Triton is available and nothing unsupported is requested; it falls back to MRV1 for, among
others, the `ngram`, `ngram_gpu`, `draft_model`, `suffix` and `medusa` speculative methods, stock
`torch.compile`, and sequence parallelism with TP > 1 (`_get_v2_model_runner_unsupported_features`).
Three exceptions to "default": on ROCm, the architectures in `ROCM_DEFAULT_MRV1_ARCHITECTURES` (DeepSeek-V3.2,
DeepSeek-V4, GLM-MoE-DSA) default to MRV1 ("Defaulting to V1 model runner on ROCm") unless MRV1 cannot serve
the configuration; HiSparse attention requires MRV2 and rejects `VLLM_USE_V2_MODEL_RUNNER=0`; watermarking forces
MRV2 over it. Otherwise `VLLM_USE_V2_MODEL_RUNNER=0|1`
forces the choice; the worker logs "Using V2 Model Runner" (`vllm/v1/worker/gpu_worker.py`) or the config logs
the fallback reason. A source comment in `_get_v1_model_runner_unsupported_features` already calls MRV1
"deprecated". **This primer traces the default, MRV2**, and names the MRV1 equivalent where it differs;
Section 5.3 covers MRV2 and 5.4 keeps MRV1 as the contrast.

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
  │                                                                           Scheduler.get_grammar_bitmask(so)               GPUModelRunner.execute_model [MRV2]:
  │                                                                             (CPU, overlaps the forward)                    add_requests, update_requests,
  │                                                                                                                            prepare_inputs, prepare_attn,
  │                                                                                                                            forward → hidden states (kept)
  │                                                                           Executor.sample_tokens(grammar) ─────────────▶ GPUModelRunner.sample_tokens:
  │                                                                                                                            sample: logits, bitmask, Sampler or
  │                                                                                                                            RejectionSampler; postprocess_sampled;
  │                                                                           ◀──────────── ModelRunnerOutput ──────────────── speculator.propose (drafts)
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

| # | Process | Class.function | What happens | Shows up in |
|---|---|---|---|---|
| 1 | API | `create_chat_completion` (`vllm/entrypoints/openai/chat_completion/api_router.py`) | FastAPI route; `StreamingResponse` with SSE keep-alive | HTTP |
| 2 | API | `OpenAIServingChat.render_chat_request` → `OnlineRenderer.render_chat` (`.../chat_completion/serving.py`, `vllm/renderers/`) | chat template + tokenization | front-end CPU |
| 3 | API | `AsyncLLM.add_request` → `InputProcessor.process_inputs` (`vllm/v1/engine/input_processor.py`) | raw prompts run on the renderer's thread pool (`process_inputs_async`); params and LoRA validated, `SamplingParams` cloned, `max_tokens = max_model_len − prompt_len` if unset; `n > 1` fans out into child requests (`vllm/v1/engine/parallel_sampling.py: ParentRequest`) | TTFT |
| 4 | API→Core | `AsyncMPClient._send_input` (`vllm/v1/engine/core_client.py`) | ROUTER socket; frames = (request-type byte, msgpack payload) | IPC |
| 5 | Core, input thread | `EngineCoreProc.process_input_sockets` → `EngineCore.preprocess_add_request` (`vllm/v1/engine/core.py`) | build `Request` (computes prompt block hashes); start async grammar compile | overlaps the GPU step |
| 6 | Core | `run_busy_loop` → `_process_input_queue` → `Scheduler.add_request` | request enters `waiting`; `QUEUED` event | queue time |
| 7 | Core | `Scheduler.schedule` (`vllm/v1/core/sched/scheduler.py`) | budget, prefix lookup, block allocation, preemption → `SchedulerOutput` | queue time, TTFT |
| 8 | Core→GPU | `Executor.execute_model(so, non_block=True)` → `GPUModelRunner.execute_model` (`vllm/v1/worker/gpu/model_runner.py`, MRV2) | apply the `SchedulerOutput` delta to GPU request slots, build inputs and slot mappings, forward pass to hidden states | step time |
| 9 | Core | `Scheduler.get_grammar_bitmask` | structured-output mask built on the CPU during step 8 (`EngineCore.step`) | hidden |
| 10 | GPU | `GPUModelRunner.sample_tokens(grammar_output)` → `sample` | logits for the sampled rows only, bitmask, sample or rejection-sample, update GPU state (`postprocess_sampled`), propose drafts | step time |
| 11 | Core | `Scheduler.update_from_output` | append tokens, roll back rejected drafts, `check_stop`, free finished requests; outputs encoded and pushed by the output thread | step time |
| 12 | API | `AsyncLLM._run_output_handler` → `OutputProcessor.process_outputs` (`vllm/v1/engine/output_processor.py`) → `chat_completion_stream_generator` | the one loop over all outputs: stats, incremental detokenize, stop strings, logprobs; chunks of `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` with `await asyncio.sleep(0)` between; SSE deltas, then `data: [DONE]` | ITL jitter under load |

Two consequences for a review. **Stop strings are detected in the API server, not in the engine**:
`IncrementalDetokenizer.update` returns the matched string (`vllm/v1/engine/detokenizer.py`),
`process_outputs` adds the request to `reqs_to_abort`, and `AsyncLLM` aborts it in the engine, which may
already have computed a token or two past it. EOS, `stop_token_ids`, `max_tokens` and `max_model_len` are
checked in the engine by `check_stop` (`vllm/v1/core/sched/utils.py`). **A client disconnect is an abort**:
when the SSE generator is cancelled, `AsyncLLM.generate` catches `CancelledError`/`GeneratorExit` and calls
`abort`, which frees the request's blocks.

### 2.3 The four messages that matter

| Message | Direction | Key fields | Defined in |
|---|---|---|---|
| `EngineCoreRequest` | API → Core | `request_id`, `prompt_token_ids`, `mm_features`, `sampling_params`, `lora_request`, `cache_salt`, `priority`, `arrival_time`, `data_parallel_rank`; `kv_transfer_params` rides in `sampling_params.extra_args` and is lifted out in `Request.__init__` (`vllm/v1/request.py`) | `vllm/v1/engine/__init__.py` |
| `SchedulerOutput` | Core → workers | `scheduled_new_reqs` (full data once), `scheduled_cached_reqs` (deltas: new token ids, new block ids), `num_scheduled_tokens` per request, `total_num_scheduled_tokens`, `scheduled_spec_decode_tokens`, `scheduled_encoder_inputs`, `num_common_prefix_blocks`, `finished_req_ids`, `preempted_req_ids`, `kv_connector_metadata` | `vllm/v1/core/sched/output.py` |
| `ModelRunnerOutput` | workers → Core | `req_ids`, `req_id_to_index`, `sampled_token_ids` (several per request when drafts are accepted), `logprobs`, `prompt_logprobs_dict`, `kv_connector_output` | `vllm/v1/outputs.py` |
| `EngineCoreOutputs` | Core → API | per request `new_token_ids`, `finish_reason` (`STOP`/`LENGTH`/`ABORT`/`ERROR`/`REPETITION`), `new_logprobs`, `events` (QUEUED/SCHEDULED/PREEMPTED timestamps), `kv_transfer_params`; per batch `scheduler_stats`, `timestamp` | `vllm/v1/engine/__init__.py` |

`SchedulerOutput` is deliberately a diff: after a request's first step, workers get only its new tokens and
new block ids, because they keep the rest in their persistent state (Section 5).

### 2.4 Where TTFT goes

The front-end measures TTFT from `arrival_time` (set in `InputProcessor.process_inputs`) to the iteration
in which the first token is processed (`vllm/v1/metrics/stats.py: IterationStats.update_from_output`).
Engine events split it further (`IterationStats.update_from_finished_request`):

```
arrival ──render/tokenize──▶ QUEUED ──queue──▶ SCHEDULED ──prefill (all chunks)──▶ first token ──decode──▶ last token
          (front-end CPU)            vllm:request_queue_time_seconds    vllm:request_prefill_time_seconds      vllm:request_decode_time_seconds
◀──────────────────────────── vllm:time_to_first_token_seconds ────────────────────▶
```

`vllm:request_time_per_output_token_seconds` is `decode_time / (num_generation_tokens − 1)` per request;
`vllm:inter_token_latency_seconds` is observed per iteration. High TTFT with high queue time means the
engine is saturated (Section 3); high TTFT with high prefill time means long or heavily chunked prompts; if
both are low, look at the front-end (tokenization, multimodal preprocessing, `--api-server-count`).

---

## 3. The scheduler

### 3.1 One idea: tokens to compute

The comment at the top of `Scheduler.schedule` (`vllm/v1/core/sched/scheduler.py`) states the design:

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the
> num_computed_tokens and num_tokens_with_spec. [...] At each step, the scheduler tries to assign tokens to
> the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.

`num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)` (`vllm/v1/request.py`). A fresh
request is 3,000 tokens behind; a decoding request is 1 behind (the token sampled last step); a request with
3 draft tokens is 4 behind. Chunked prefill, prefix caching (which advances `num_computed_tokens` without
compute) and speculative decoding are the same operation.

### 3.2 The budgets

| Budget | Source | Default for `vllm serve` | Enforced in |
|---|---|---|---|
| tokens per step | `--max-num-batched-tokens`; `max_num_scheduled_tokens` defaults to it | 2048 on GPUs under 70 GiB and on A100; 8192 on ≥ 70 GiB non-A100 (H100/H200); 16384 on ≥ 160 GiB (B200/B300); doubled by `--performance-mode throughput` | `Scheduler.schedule` (`token_budget`) |
| requests in RUNNING | `--max-num-seqs`; `--max-num-active-seqs` (main after 0.30.0, verify) lowers admission only | 256 below 70 GiB and on A100; 1024 above | waiting loop |
| per-request chunk | `--long-prefill-token-threshold` | 0 (off); ignored when one request is eligible | both loops |
| encoder tokens per step | `MultiModalBudget.encoder_compute_budget` | derived | `_try_schedule_encoder_inputs` |
| distinct LoRAs per step | `--max-loras` | 1 | waiting loop |

The GPU-dependent defaults come from `EngineArgs.get_batch_defaults` (`vllm/engine/arg_utils.py`), keyed on
device memory and name (a comment there explains the A100 exception: large budgets cut its throughput).
`SchedulerConfig.verify_max_model_len` requires `max_num_batched_tokens >= max_num_seqs`, and
`>= max_model_len` when chunked prefill is off (`vllm/config/scheduler.py`).

### 3.3 The algorithm

Condensed from `Scheduler.schedule` (encoder, Mamba-alignment, KV-connector and DP-balancing details omitted):

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
        victim = running[-1]                  # FCFS; with --scheduling-policy priority:
                                              #   max(running, key=(priority, arrival_time))
        _preempt_request(victim)              # free blocks, num_computed_tokens=0, prepend to waiting
        if victim is req: break
    if new_blocks is None: break
    schedule(req, n); token_budget -= n

# (2) WAITING requests, only if nothing was preempted in (1)
if not preempted_reqs:
    # max_num_active_seqs: main after 0.30.0 (verify); in the 0.30.0 wheel the cap is max_num_seqs
    while (waiting or skipped_waiting) and token_budget > 0 and len(running) < max_num_active_seqs:
        req = queue.peek_request()            # deque (fcfs) or heap on (priority, arrival, id)
        if blocked (grammar compiling, remote KV pending, LoRA cap): move to skipped; continue
        if req.num_computed_tokens == 0:
            hit_blocks, hit_tokens = kv_cache_manager.get_computed_blocks(req)
            (+ connector.get_num_new_matched_tokens for external hits)
        n = req.num_tokens - computed;  apply lpt;  n = min(n, token_budget)
        new_blocks = kv_cache_manager.allocate_slots(req, n, hit_tokens, hit_blocks,
                                                     full_sequence_must_fit=True, ...)
        if new_blocks is None: break          # head of the queue waits; nothing jumps ahead
        running.append(req); schedule(req, n); token_budget -= n

output = SchedulerOutput(...);  _update_after_schedule(output)   # advance num_computed_tokens now
```

What falls out, and matters in a review:

- **Decodes first.** Running requests take budget before any new prefill; admitting work never starves streams.
- **Chunking is automatic.** A waiting request gets `min(remaining, token_budget)`.
- **Not strictly FCFS.** A running request that cannot take tokens is skipped with `continue` ("we do not
  strictly follow the FCFS scheduling policy"), and waiting requests blocked on grammar compilation or remote
  KV are parked in `skipped_waiting`; but a waiting request that simply does not fit stops the pass (`break`).
- **No admissions in a preempting step** (`if not preempted_reqs`).
- **Priority order** is `Request.__lt__`: lower `priority` first, then earlier `arrival_time`, then
  `request_id` (`vllm/v1/core/sched/request_queue.py: PriorityRequestQueue`).
- **`num_computed_tokens` advances at schedule time** (`_update_after_schedule`), so the next chunk can be
  scheduled immediately; rejected drafts are subtracted later in `update_from_output`.

### 3.4 Chunked prefill and `long_prefill_token_threshold`

Chunked prefill is the `min(n, token_budget)` above, not a separate code path; the Sarathi-Serve argument for
it (bounded step time keeps co-scheduled decodes' ITL steady) is in the serving-engine primer. Two knobs shape
chunks. `--max-num-batched-tokens` sets the step size, and every decode sharing a step waits for all of it.
`--long-prefill-token-threshold` caps one request's chunk: at the default 0 a long prompt can take the whole
remaining budget; at 512, two concurrent long prompts each advance 512 per step instead of one blocking the
other. The cap is dropped when only one request is eligible, and `--long-prefill-token-threshold-adaptive`
(main after 0.30.0, verify) floors it at `max_num_batched_tokens / num_requests` (`SchedulerConfig`). A chunk that ends mid-prompt still
gets a logits row and a draw, which is then thrown away: in MRV2, `get_num_sampled_and_rejected`
(`vllm/v1/worker/gpu/input_batch.py`) sets `num_sampled = 0` for any row with `seq_len < prefill_len`, so
`postprocess_sampled` records nothing for it (MRV1 does the same with `discard_request_mask` in
`GPUModelRunner._prepare_inputs`, `vllm/v1/worker/gpu_model_runner.py`).

### 3.5 Worked example: four requests through six steps

`vllm serve` on an L4: `max_num_batched_tokens = 2048`, `max_num_seqs = 256`, `block_size = 16`,
`long_prefill_token_threshold = 0`, empty prefix cache, enough free blocks. A (3,000-token prompt) and B (500)
arrive together; C (6,000) arrives before step 3. Blocks are `ceil(tokens / 16)`: A 188, B 32, C 375.

| Step | Running pass (decodes first) | Waiting pass | Tokens | New blocks | Real tokens sampled |
|---|---|---|---|---|---|
| 1 | — | A: `min(3000, 2048) = 2048` (admission checked that all 188 blocks fit); B does not fit (budget 0) | 2,048 | A +128 | none (A mid-prefill) |
| 2 | A: `3000 − 2048 = 952` | B: 500 fits in the 1,096 left | 1,452 | A +60, B +32 | A, B (their TTFT) |
| 3 | A: 1, B: 1 | C: `min(6000, 2046) = 2046` | 2,048 | C +128 (A's position 3000 is in block 187, already held; B's 500 in block 31) | A, B |
| 4 | A: 1, B: 1, C: `min(3954, 2046) = 2046` | — | 2,048 | C +128 | A, B |
| 5 | A: 1, B: 1, C: `6000 − 4092 = 1908` | — | 1,910 | C +119 | A, B, C (C's TTFT: 3 steps) |
| 6 | A: 1, B: 1, C: 1 | — | 3 | C +1 (position 6000 opens block 375; A's 3003 and B's 503 fit in held blocks) | A, B, C (pure decode: full CUDA graph, padded to 4) |

The rule behind the "New blocks" column: `allocate_slots` needs `ceil((num_computed_tokens + n) / 16)` blocks,
so a decode takes a new block exactly when `num_computed_tokens` is a multiple of 16. C's prompt fills 375 blocks
exactly (6,000 = 375 × 16, block indices 0–374), so its first decode needs a 376th block (index 375); A and B took their partial last blocks at
admission and next need one at positions 3,008 and 512.

A and B keep producing one token per step while C prefills, but steps 3–5 each carry ~2,048 tokens, so their
ITL during C's prefill is the time of a 2,048-token step, not a 6,000-token one. Raising the budget to 8,192
would finish C in one step (better TTFT for C) and make that step four times longer for A and B (worse ITL):
that is the TTFT-versus-ITL dial.

### 3.6 Admission control inside the engine

`allocate_slots` admits a waiting request only if its **whole** sequence fits, not just its first chunk: the
scheduler passes `full_sequence_must_fit = scheduler_reserve_full_isl` (default `True`), and
`KVCacheManager.allocate_slots` compares the blocks for `min(num_tokens, max_model_len)` plus the watermark
with `BlockPool.get_num_free_blocks()` (the mini engine calls the same gate `admit_whole_prompt`,
`../serving-engine/mini-engine-core/minengine/kv.py`). `num_tokens` is the *current* length, prompt plus any
outputs so far: a resumed request must fit everything it has generated, and nothing reserves room for
`max_tokens`, so admitted requests can still outgrow the pool later (Section 3.7). Prefix hits sitting in the
free queue count as needed capacity, since touching them removes them from the free pool
(`vllm/v1/core/single_type_kv_cache_manager.py: SingleTypeKVCacheManager.get_num_blocks_to_allocate`: "If a
computed block is an eviction candidate ... we must count it in the free-capacity check"):

```
required = ceil(num_tokens / 16) − len(hits)          # new blocks
         + #(hits with ref_cnt == 0)                   # evictable hits: leaving the free queue
         + watermark_blocks                            # waiting/preempted requests only
admit iff required ≤ get_num_free_blocks()             # free count includes cached, unreferenced blocks
```

`--watermark` (default 0.0) holds back
`int(watermark × num_blocks)` blocks when admitting waiting or preempted requests while something else runs.
The engine's queue is unbounded; bounds live in the API server (`--max-num-queued-reqs`,
`--max-num-queued-tokens`, HTTP 503 when full; `vllm/config/scheduler.py`).

### 3.7 Preemption: who, how, and what it costs

When a running request needs a block and none is free, `Scheduler._preempt_request` runs on a victim:
`running[-1]` under FCFS (the most recently admitted or resumed) or the maximum `(priority, arrival_time)`
under priority scheduling (if that victim was already scheduled this step, its tokens go back to the budget).
It frees all the
victim's blocks and encoder-cache entries, sets `status = PREEMPTED` and `num_computed_tokens = 0`, drops
drafts, increments `num_preemptions`, records a `PREEMPTED` event, and **prepends** it to `waiting`.

There is no swap-to-CPU path in the V1 scheduler (`CacheConfig` has no `swap_space`); the request keeps its
token ids and recomputes. Its full blocks go back to the free queue **with their hashes** (Section 4.6), so a
later re-admission re-hits whatever has not been reallocated in the meantime. Whether that happens soon depends
on the admission rule of Section 3.6, and the worked example shows it often does not.

Worked example with `--num-gpu-blocks-override 300` (documented in `CacheConfig` as "used for testing
preemption"; 299 usable blocks because block 0 is the null block). Two requests with 2,000-token prompts and
long `max_tokens`, FCFS, synchronous accounting. Under the default async scheduling the one in-flight token shifts
some counts by one, and until R2's in-flight output drains (a step) the waiting pass parks it in `skipped_waiting`
with `continue` (the `num_stale_output_tokens` check), which can let that step's admissions past it; the outcome is
the same. "Free" is `get_num_free_blocks()`, which counts cached, unreferenced blocks.

| Moment | R1 | R2 | Free |
|---|---|---|---|
| both prefilled | 125 blocks | 125 blocks | 49 |
| lockstep decode, one block each per 16 tokens | +24 → 149 | +24 → 149 | 1 |
| next boundary (both at position 2,384) | takes the last free block (150 held) | `allocate_slots` → `None`; R2 is `running[-1]`, so it preempts itself: 149 full, hashed blocks go to the free-queue tail in reverse (its block 149 nearest the head), `num_computed_tokens = 0`, prepended to `waiting`; no admissions this step | 149 |
| next step | decodes (no new block for 15 tokens) | head of `waiting`: hit = `(2385 − 1) // 16` = 149 blocks, but the full-sequence check needs `ceil(2385 / 16)` = 150 = 1 new + 149 evictable hits > 149 free → `None` → `break` | 149 |
| 16 tokens later | needs block 151: `get_new_blocks` pops the queue head, R2's block 149, and evicts its hash | hit falls to 148 blocks; still needs 150 against 148 free | 148 |
| every further 16 tokens | +1 block | loses its current tail block; still refused | −1 |
| R1 finishes, having taken *j* blocks after the preemption | frees 150 + *j* blocks | re-admitted: hits the 149 − *j* surviving blocks, recomputes 16*j* + 1 tokens | 299, then 149 once R2 holds 150 |

So one preemption, no ping-pong: the victim cannot come back while the request that displaced it runs, because it
needs 150 blocks and at most 149 exist outside that request (one fewer every time R1 grows), and every block the
survivor takes comes from the victim's cached tail. Meanwhile the victim sits at the head of `waiting` and, because the waiting pass stops at
the first request that does not fit (`break`), **no request behind it is admitted either**: the engine serializes
on R1, and TTFT for everything queued grows by R1's remaining decode time. The notebook replays this at small
scale ([`notebooks/01_block_hashes_and_eviction.ipynb`](notebooks/01_block_hashes_and_eviction.ipynb), exercise 4).

`vllm:num_preemptions` climbs under a different pattern: many requests admitted because their *current* length
fits (Section 3.6 reserves nothing for `max_tokens`), then all growing. Each time a running request finds no free
block the newest running request is preempted, and as victims are re-admitted when others finish, the cycle
repeats. The fixes are capacity (FP8 KV, fewer concurrent sequences via `--max-num-seqs`, shorter
`max_model_len`, more GPUs) or admission headroom (`--watermark`), not a bigger token budget.

### 3.8 Async scheduling and the batch queue

With async scheduling (the default when compatible; `VllmConfig` leaves it off for pooling models, for
speculative methods other than the EAGLE/MTP family, `ngram_gpu`, `draft_model`, DFlash and DSpark, and for
executors that do not support it), `SchedulerConfig.get_scheduler_cls` returns `AsyncScheduler` and
`VllmConfig.max_concurrent_batches` is 2 (V1 runner, PP=1) or `pp_size + 1` (V2 runner), so EngineCore uses
`step_with_batch_queue` instead of `step` (`vllm/v1/engine/core.py: EngineCore.__init__`):

```
step N:   schedule(N) ─▶ execute_model(N) (non-blocking) ─▶ sample_tokens(N) (non-blocking) ─▶ return early
step N+1: schedule(N+1) while GPU runs N ─▶ enqueue ─▶ block on N's result ─▶ update_from_output(N)
```

The scheduler must schedule a decode for a token it has not seen. `AsyncScheduler._update_after_schedule` adds
`num_output_placeholders` and `-1` placeholder draft ids; the runner fills in the real ids on the GPU. In MRV2,
`postprocess_sampled` writes each request's sampled token into `RequestState.last_sampled_tokens`, and the next
step's `combine_sampled_and_draft_tokens` kernel (`vllm/v1/worker/gpu/input_batch.py`) copies it, plus any
drafts, into `input_ids`, so the token never visits the host (MRV1: `GPUModelRunner._prepare_input_ids` with
`prev_sampled_token_ids`). When outputs arrive,
`AsyncScheduler._update_request_with_output` retires the placeholders and calls `cache_blocks`. The cost: a
request that stops on EOS may already have one more step scheduled, whose output is dropped (the running loop
avoids it when `max_tokens` makes the stop predictable). The benefit: CPU scheduling time leaves the step time.

### 3.9 `update_from_output`

Per scheduled request (`Scheduler.update_from_output`): with speculation, `num_rejected = num_draft −
num_accepted` is subtracted from `num_computed_tokens` (rejected positions are simply overwritten later);
`_update_request_with_output` appends tokens one at a time and calls `check_stop`
(`vllm/v1/core/sched/utils.py`): EOS or a `stop_token_ids` hit → `FINISHED_STOPPED`; `num_tokens >=
max_model_len` or `num_output_tokens >= max_tokens` → `FINISHED_LENGTH_CAPPED`; optional repetition
detection → `FINISHED_REPETITION` (`min_tokens` is enforced earlier by a logits processor, Section 7.1). The
grammar advances with `accept_tokens` (a rejection ends the request with `FINISHED_ERROR`). Finished
requests are freed by `_free_request`, unless a KV connector delays the free (Section 10).

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

The scheduler-side cache is pure CPU bookkeeping. The KV tensors live on the workers; the scheduler hands them
block ids, and the runner turns those into a block table and a slot mapping (Section 5.3).

The repo's mini engine builds the same structure from scratch
([`../serving-engine/mini-engine-core/minengine/kv.py`](../serving-engine/mini-engine-core/minengine/kv.py), with
its notebook `03_prefix_caching`); read it first if the idea is new. Where the two differ is what this section and
the notebook focus on:

| Concern | mini engine (`minengine/kv.py`) | vLLM (`vllm/v1/core/`) |
|---|---|---|
| free queue | `OrderedDict` of block ids, `popitem(last=False)` | `FreeKVCacheBlockQueue`, intrusive doubly linked list, O(1) `remove` |
| freeing | every ref-0 block to the back, tail first | cached blocks to the back, **uncached (partial) blocks to the front** (4.6) |
| reserved block | none; usage is `1 − free / num_blocks` | block 0 is the null block; usage is `1 − free / (num_blocks − 1)` |
| when a block is named | after the step that computed it (`cache_blocks`) | at scheduling time, inside `allocate_slots` (4.5) |
| duplicate full blocks | the second copy stays unpublished | both kept under the same hash (`BlockHashToBlockMap`, no dedup) |
| extra keys | one `extra` value on every block | LoRA name on every block, `cache_salt` on the first only, MM offsets, group id in the key (4.3) |
| admission gate | `allocate_slots(..., admit_whole_prompt=True)`, revived hits counted | `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)`, evictable hits counted (3.6) |
| hit cap | `(len − 1) // block_size` blocks | the same, via `max_cache_hit_length = num_tokens − 1` (4.4) |

### 4.2 Blocks and the free queue

`KVCacheBlock` (`vllm/v1/core/kv_cache_utils.py`) holds `block_id`, `ref_cnt`, `_block_hash` (set only when
full and cached) and `prev_free_block`/`next_free_block`. `FreeKVCacheBlockQueue` threads those pointers into a
doubly linked list with fake head and tail instead of using `collections.deque`, so a block in the middle can
be removed in O(1) when a prefix hit touches it, without allocating Python objects (class docstring):
`popleft_n` allocates, `remove` touches, `append_n` frees cached blocks to the tail, `prepend_n` frees uncached
blocks to the head. `BlockPool.__init__` pops block 0 as the `is_null` placeholder (used, for example, for
positions outside a sliding window); usable capacity is `num_gpu_blocks − 1`, and `BlockPool.get_usage` is
`1 − free / (num_gpu_blocks − 1)`.

### 4.3 Block hashes: a chain over the prefix

Each full block's hash commits to **all tokens before it** (`hash_block_tokens`, `get_request_block_hasher`):

```
NONE_HASH = H(seed)                                   # init_none_hash
h0 = H( (NONE_HASH, (t0 … t15),  extra_keys_0) )      # hash_block_tokens(parent=None → NONE_HASH)
h1 = H( (h0,        (t16 … t31), extra_keys_1) )
h2 = H( (h1,        (t32 … t47), extra_keys_2) )      # H = sha256(pickle.dumps(x)) by default
```

- **Function**: `--prefix-caching-hash-algo` defaults to `sha256`, which pickles the tuple
  (`vllm/utils/hashing.py: sha256`); `sha256_cbor` is reproducible across languages; `xxhash` variants are
  faster and non-cryptographic (`vllm/config/cache.py`).
- **Seed**: for cryptographic hashes `NONE_HASH` comes from a fixed seed (`"vllm-none-hash"`), so separate
  processes agree and can share KV across nodes; for xxhash it is random per process unless `PYTHONHASHSEED`
  is set, so collisions cannot be precomputed (`resolve_none_hash_seed`, `init_none_hash`).
- **Extra keys** (`generate_block_hash_extra_keys`): the LoRA name on every block (adapters never share KV);
  `(mm identifier, offset within the block)` for each multimodal item overlapping the block; `cache_salt` on
  the **first block only**, which isolates tenants because every later hash chains from it; a digest of
  prompt embeddings when used.
- **When**: hashes are computed as tokens become known, for the prompt in `Request.__init__` (in the
  EngineCore input thread) and later as output tokens fill blocks; only full blocks are hashed.
- **Keys**: the map is indexed by hash bytes plus a 4-byte group id (`make_block_hash_with_group_id`), since
  hybrid models keep one physical block per group per logical block. Identical blocks computed concurrently
  are both kept (no deduplication, `BlockHashToBlockMap` NOTE #1), so block tables stay append-only.

Because each hash includes its parent, the longest cached prefix is a scan that stops at the first miss.

### 4.4 Lookup: `get_computed_blocks`

`KVCacheManager.get_computed_blocks` returns an empty hit if prefix caching is off or the request skips cache
reads (`SamplingParams` sets `skip_reading_prefix_cache` automatically when `prompt_logprobs` is requested,
since a hit would leave those positions without logits; `vllm/sampling_params.py`). Otherwise it calls
`coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length = request.num_tokens − 1)`;
the `− 1` exists because the last prompt token must run through the model to produce logits.
`FullAttentionManager.find_longest_cache_hit` walks `max_length // block_size` hashes until the first miss.

| Prompt (`block_size = 16`) | Cached | Hit | Computed | Saved |
|---|---|---|---|---|
| 1,000-token system prompt + 200-token user turn, second user | first request's blocks | 62 blocks = 992 tokens (block 63 mixes system and user tokens) | 208 | 82.7% |
| the identical 1,200-token prompt again | all 75 blocks | `(1199 // 16) × 16 = 1,184` | 16 (hits are block-aligned) | 98.7% |

The prompt-design rule: put everything shared (system prompt, tool schemas, few-shot examples, documents)
first and byte-identical, and anything per-request (timestamps, user ids) after it; one changed token early on
changes every later hash. `record_prefix_cache_stats` adds `request.num_tokens` to queries and the hit length
to hits at admission, so `vllm:prefix_cache_hits / vllm:prefix_cache_queries` is a **token-weighted** hit rate
(`vllm/v1/metrics/stats.py: PrefixCacheStats.record`). It covers **first admissions only**: `record` is called
with `preempted = request.num_preemptions > 0`, and a re-admission after preemption goes to separate
`preempted_queries`/`preempted_hits` fields that `PrometheusStatLogger` does not export, so the re-hits of
Section 3.7 never show in this ratio.

### 4.5 Allocation: `allocate_slots`

The docstring of `KVCacheManager.allocate_slots` draws the layout:

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
 already    prefix-cache   from a KV      tokens to   slots for draft
 held       hits (local)   connector      compute     tokens (EAGLE etc.)
```

Three stages: (1) free blocks no longer needed (outside a sliding window) and return `None` if `required +
watermark > free − reserved`; (2) attach the hit blocks (`BlockPool.touch`: `ref_cnt += 1`, removing the block
from the free queue if it was 0); (3) allocate `new + lookahead` blocks with `BlockPool.get_new_blocks`, which
pops from the free-queue head and first evicts any hash the popped block still carries
(`_maybe_evict_cached_block`). It then **caches immediately**: `coordinator.cache_blocks(request, min(computed +
new, request.num_tokens))` hashes every block that will be full once this step runs, before the forward pass;
the cap at `num_tokens` keeps unverified drafts out. A consequence `(verify)`: a later request in the same
`schedule()` call can already hit those blocks, relying on KV being written layer by layer before it is read.

### 4.6 Free and eviction order

`SingleTypeKVCacheManager.free` calls `block_pool.free_blocks(reversed(blocks))`, and `BlockPool.free_blocks`
sends blocks whose `ref_cnt` reaches 0 **without a hash** (the partial last block) to the head ("LIFO reuse of
non-cached blocks for better GPU locality") and blocks **with a hash** to the tail ("FIFO reuse of cached blocks
for LRU eviction behavior"). Request X holds `[b1(h1), b2(h2), b3(h3), b4(partial)]` and finishes while the free
queue holds older `[f1, f2]`:

```
before:   head → f1 → f2 → tail
reversed: b4, b3, b2, b1
after:    head → b4 → f1 → f2 → b3 → b2 → b1 → tail
```

Allocation takes `b4` (nothing lost), then `f1`, `f2`, then `b3` before `b2` before `b1`: within a chain the
**tail goes first** and the root last, because the root is what the next request most likely shares (ordering
stated in the `FreeKVCacheBlockQueue` docstring). A later hit on `b1` removes it from the middle in O(1).

So `vllm:kv_cache_usage_perc` measures blocks **referenced by live requests**; cached blocks in the free queue
count as free. A replica at 30% usage can hold a large, useful prefix cache; a router that wants cache affinity
needs hashes (Section 12; orchestration in [`../../05-orchestrator/`](../../05-orchestrator/)), not this gauge.

### 4.7 Sizing the pool: from `gpu_memory_utilization` to `num_gpu_blocks`

`EngineCore._initialize_kv_caches` (`vllm/v1/engine/core.py`) collects each worker's KV specs, calls
`Worker.determine_available_memory` (`vllm/v1/worker/gpu_worker.py`) and `get_kv_cache_configs`
(`vllm/v1/core/kv_cache_utils.py`), then allocates and compiles/captures (`initialize_from_config`,
`compile_or_warm_up_model`):

```
requested       = ceil(total_gpu_memory × gpu_memory_utilization)                   # request_memory (vllm/v1/worker/utils.py)
non_kv          = weights + transient activation peak + non-torch growth             # memory_profiling (vllm/utils/mem_utils.py)
available_kv    = requested − non_kv − cudagraph_estimate                             # determine_available_memory
page_bytes      = num_kv_heads × block_size × (head_size + head_size_v) × dtype_bytes # AttentionSpec.page_size_bytes, per layer
bytes_per_block = Σ over the group's layers of page_bytes                            # _get_kv_cache_bytes_per_block
num_blocks      = available_kv // bytes_per_block                                     # get_kv_cache_config_from_groups
```

The activation peak comes from a profiling forward of `max_num_batched_tokens` tokens plus a dummy sampler run
over `max_num_seqs` rows (`GPUModelRunner.profile_run`), so larger budgets shrink the pool. CUDA-graph memory is
estimated by `profile_cudagraph_memory` and subtracted while `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is on
(the default "since v0.21.0", per the log line, which also suggests how far to raise `--gpu-memory-utilization`
to keep the old KV size). `--kv-cache-memory-bytes` skips profiling; `--num-gpu-blocks-override` forces the count.
If free memory at startup is below `requested`, startup fails in `request_memory`.

**Worked budget: Llama-3.1-8B-Instruct, BF16.** 32 layers, 32 query heads, 8 KV heads, head_dim 128, hidden
4,096, vocab 128,256; 8,030,261,248 parameters (computed from the config shape).

```
weights          = 8,030,261,248 × 2 B                       = 16.06 GB = 14.96 GiB
KV per token     = 2 (K,V) × 32 layers × 8 heads × 128 × 2 B  = 131,072 B = 128 KiB
page per layer   = 8 heads × 16 tokens × (128 + 128) × 2 B     = 65,536 B = 64 KiB
bytes_per_block  = 32 layers × 64 KiB                         = 2 MiB per 16 tokens
```

The non-KV overheads are estimates, and this table uses the serving lab's estimator so that the two documents
give the same numbers: `servelab.sizing.size("llama-3.1-8b-instruct", "L4", max_model_len=8192)` and
`size(..., "H100-80GB", max_num_batched_tokens=8192, max_num_seqs=1024)`
([`../serving-engine/vllm-serving-lab/servelab/sizing.py`](../serving-engine/vllm-serving-lab/servelab/sizing.py);
`overhead_estimate` models the profiling forward's MLP activations plus FP32 sampler logits, 0.5 GiB of graphs and
0.2 GiB non-torch memory). The notebook's last section recomputes every number below and fails if the primer and
the lab drift apart. Your `Available KV cache memory: … GiB` log line replaces all of the estimates.

| | L4 24 GB | H100 80 GB (SXM) |
|---|---|---|
| total memory seen by CUDA (lab GPU table) | 22.49 GiB `(verify)` | 79.65 GiB `(verify)` |
| `requested` at 0.92 | 20.69 GiB | 73.28 GiB |
| weights | 14.96 GiB | 14.96 GiB |
| activations of the profiling pass + sampler logits (estimate) | 0.42 GiB (2,048-token profile, 256-row sampler) | 1.67 GiB (8,192-token profile, 1,024-row sampler) |
| CUDA graphs + non-torch (estimate) | 0.5 + 0.2 GiB | 0.5 + 0.2 GiB |
| **available KV** | **4.62 GiB** | **55.95 GiB** |
| `num_blocks` (÷ 2 MiB) | 2,363 | 28,648 |
| token capacity (× 16) | 37,808 | 458,368 |
| max concurrency at `max_model_len` 8,192 (512 blocks each) | 4.62× | 55.95× |
| at 32,768 (2,048 blocks) | 1.15× | 13.99× |
| at 131,072, the model default (8,192 blocks = 16 GiB) | **does not start** | 3.50× |
| with `--kv-cache-dtype fp8` (1 MiB per block) | 4,727 blocks, 75,632 tokens | 57,297 blocks, 916,752 tokens |

The L4 row is the most common first-deployment failure: with no `--max-model-len`, one 131,072-token request
needs 16 GiB of KV and `_check_enough_kv_cache_memory` raises "To serve at least one request with the model's max
seq len (131072), (16.00 GiB KV cache is needed, which is larger than the available KV cache memory (…)". Fix it
with `--max-model-len 16384` (2.31×), `--max-model-len auto` (`-1`: `_auto_fit_max_model_len` binary-searches the
largest length that fits, about 37.8k tokens with the estimates above), FP8 KV, or a bigger GPU. The result is logged
as "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx" (`update_kv_cache_capacity`).
That concurrency is `num_blocks / ceil(max_model_len / block_size)` (`get_max_concurrency_for_kv_cache_config`),
a worst case: real requests are shorter, and prefix sharing raises it further.

### 4.8 Hybrid models: several KV cache groups

When layers need different cache behaviour (full attention, sliding window, Mamba state, cross-attention),
`get_kv_cache_groups` splits them into groups following the layer pattern; the example in
`_get_kv_cache_groups_uniform_page_size`: 10 full-attention and 20 sliding-window layers form the pattern
(1 × full, 2 × sw), hence 3 groups of 10 layers, each with its own block table, all drawing ids from one pool
with unified page sizes (`unify_kv_cache_spec_page_size`). `SlidingWindowManager` needs only
`ceil((window − 1) / block_size)` contiguous blocks for a hit and swaps out-of-window blocks for the null block
(`remove_skipped_blocks`), with admission bounded by window plus in-flight tokens
(`SlidingWindowSpec.max_admission_blocks_per_request`). `MambaManager` stores recurrent state, checkpointed at
block boundaries for prefix caching with `mamba_cache_mode = "align"` (`CacheConfig`).
`HybridKVCacheCoordinator.find_longest_cache_hit` iterates to a fixed point where every group accepts the hit
length. `--disable-hybrid-kv-cache-manager` allocates every layer as full attention (simpler, more memory).

---

## 5. The model runner and the executor

### 5.1 Executors: how a `SchedulerOutput` reaches the GPUs

`Executor.get_class` (`vllm/v1/executor/abstract.py`) maps `--distributed-executor-backend` to a class;
`ParallelConfig.__post_init__` (`vllm/config/parallel.py`) picks the default:

| Backend | Chosen when | Topology | Transport |
|---|---|---|---|
| `uni` → `UniProcExecutor` | world size 1 | worker inside the EngineCore process | direct call |
| `mp` → `MultiprocExecutor` | world size > 1 on CUDA, fits on the node (or `--nnodes` set) | one `WorkerProc` per GPU | shared-memory `MessageQueue` broadcast |
| `ray` → `RayExecutorV2` (default), `RayDistributedExecutor` with `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` | in a Ray placement group, `--data-parallel-backend ray`, or explicit | one `RayWorkerProc` actor per GPU, placed by a placement group, any number of nodes | V2: the same `MessageQueue` as `mp` (shared memory on the driver's node, TCP to other nodes); legacy: Ray compiled DAG (`_compiled_ray_dag`) |
| `external_launcher` | `torchrun`-style launch (RL, SPMD) | caller owns the processes | — |

The Ray default comes from `vllm/envs.py` (`VLLM_USE_RAY_V2_EXECUTOR_BACKEND` defaults to `"1"`, commented "use
RayExecutorV2 (MQ-based) instead of RayDistributedExecutor (compiled-graph backend)") and `Executor.get_class`.

`MultiprocExecutor.collective_rpc` enqueues `(method, args, kwargs, output_rank)` on one `MessageQueue`
(`vllm/distributed/device_communicators/shm_broadcast.py`): a shared-memory ring buffer for readers on the node
(default chunk 24 MiB, "large enough to accommodate grammar bitmask tensors for large batches (1024 requests)")
plus a ZMQ XPUB socket for remote readers. Every worker executes the same `SchedulerOutput`; only one replies:
`_get_output_rank` returns the first tensor-parallel rank of the last pipeline stage.

**Ray or multiprocessing.** Since `RayExecutorV2` subclasses `MultiprocExecutor`
(`vllm/v1/executor/ray_executor_v2.py`: "Inherits from MultiprocExecutor to reuse the MQ-based control plane and
NCCL data plane. Workers are Ray actors."), the per-step path is the same in both: one `MessageQueue` broadcast of
the `SchedulerOutput`, NCCL for the tensors. What differs is who creates, places and watches the worker
processes:

| | `mp` | `ray` (`RayExecutorV2`) |
|---|---|---|
| default when (`ParallelConfig.__post_init__`) | world size > 1 and it fits on this node, or `--nnodes > 1` on CUDA | already inside a Ray placement group, `--data-parallel-backend ray`, or `--distributed-executor-backend ray`; the only option on TPU per the config docstring |
| multi-node | you start `vllm serve` on every node with `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`; something else (a script, Kubernetes LeaderWorkerSet) owns the processes | `initialize_ray_cluster` creates or reuses a placement group; Ray starts one actor per bundle on whichever nodes hold the GPUs |
| GPU assignment | local rank → device on each node | `RayWorkerProc` discovers the physical GPU Ray bound to its bundle and never rewrites `CUDA_VISIBLE_DEVICES`, so several engines can share a node through externally managed placement groups (class docstring) |
| failure handling | `start_worker_monitor` waits on the local worker processes' sentinels; a death shuts the executor down and fails the engine | the same policy, but the monitor thread `ray.wait`s on every actor's `run()` reference, so it also sees workers on other nodes (`RayExecutorV2.start_worker_monitor`) |
| costs | nothing extra | a Ray installation and cluster, actor start-up before model load, logs and stack traces spread across Ray's per-actor logs |

Pick `mp` for one node, and for multi-node when an orchestrator already places the pods; pick Ray when a Ray
cluster is already the unit of scheduling (Ray Serve, RL frameworks that place engines themselves). Whether
Ray adds per-step latency over `mp` now that both share the `MessageQueue` control plane is `(verify: measure)`.

### 5.2 A worker's life

`Worker` (`vllm/v1/worker/gpu_worker.py`), driven by `EngineCore.__init__` and `_initialize_kv_caches`:

1. `init_device`: set the device, initialize distributed state and model-parallel groups, snapshot memory,
   compute `requested_memory`.
2. `load_model` → `GPUModelRunner.load_model` → model loader (Section 8); logs "Model loading took X GiB
   memory and Y seconds".
3. `get_kv_cache_spec`: one `KVCacheSpec` per attention layer (`FullAttentionSpec`, `SlidingWindowSpec`,
   `MLAAttentionSpec`, `MambaSpec`, ...; `vllm/v1/kv_cache_interface.py`).
4. `determine_available_memory`: profiling forward and CUDA-graph estimate (Section 4.7).
5. `initialize_from_config(kv_cache_config)`: allocate KV tensors, bind them to layers, build attention
   metadata builders.
6. `compile_or_warm_up_model`: compile and warm extra sizes, `kernel_warmup`, `capture_model` (largest shapes
   first "so that the smaller shapes can reuse the memory pool"), then warm the sampler at maximum shape; logs
   "Graph capturing finished in N secs, took X GiB", and EngineCore logs "init engine (profile, create kv
   cache, warmup model) took N s".

At runtime each step is two RPCs, `execute_model(scheduler_output)` then `sample_tokens(grammar_output)`, so
EngineCore can compute the structured-output bitmask while the forward runs (`EngineCore.step`).

### 5.3 Model runner V2, the default: one step in order

`vllm/v1/worker/gpu/model_runner.py: GPUModelRunner` keeps each request in a fixed **slot** of GPU-side state:
`RequestState` (`vllm/v1/worker/gpu/states.py`) holds `all_token_ids` (`max_num_reqs × max_model_len`, placed in
UVA host memory because it "can be extremely large"), `prompt_len`, `prefill_len` (prompt plus any outputs
replayed after a preemption), `total_len`, `num_computed_tokens`, `last_sampled_tokens` and `draft_tokens`,
recycling free slot indices. A step's batch is an `idx_mapping` from batch rows to slots. The file header's rule for
contributors: shared code only, "Be paranoid about changing this file". One step, as `execute_model` and
`sample_tokens` run it:

1. **Apply the diff.** `finish_requests` frees the slots of finished and preempted requests; `add_requests` copies
   each new or resumed request's tokens into its slot (`RequestState.add_request`), appends its block ids
   (`BlockTables.append_block_ids`) and registers its sampling parameters; `update_requests` appends new block ids
   and `num_computed_tokens` for continuing requests. Staged writes are flushed to the GPU in one go
   (`apply_staged_writes`).
2. **Order and pad.** `gather_batch_req_state` sorts the scheduled requests decode/verification first, then short
   extends, then prefills (`sort_batch_req_ids`: "split_decodes_and_prefills relies on decode-like requests
   leading"), and `dispatch_cg_and_sync_dp` picks the CUDA-graph mode and padded size for the batch (FULL for a
   uniform decode batch, PIECEWISE otherwise; Section 5.5), agreeing on it across data-parallel ranks.
3. **Inputs, on the GPU** (`prepare_inputs`). Little crosses from the host beyond `idx_mapping` and
   `query_start_loc` (a CPU cumulative sum of scheduled tokens); three Triton kernels in `vllm/v1/worker/gpu/input_batch.py` do the rest:
   `prepare_prefill_inputs` gathers prompt tokens from `all_token_ids`, `prepare_pos_seq_lens` writes positions
   and sequence lengths, and `combine_sampled_and_draft_tokens` writes each decoding request's last sampled token
   and drafts into `input_ids` and returns `logits_indices`, the rows that will be sampled. With three requests
   scheduled for `[2, 5, 3]` tokens and `num_computed_tokens = [10, 0, 40]` (the example in the MRV1 source
   comments, extended with positions), the kernels produce:

```
row → slot     idx_mapping   = [s0, s1, s2]             (whatever slots the three requests occupy)
query_start_loc              = [0, 2, 7, 10]            cumsum of [2, 5, 3]
positions      = computed[row] + offset in row = [10, 11, 0, 1, 2, 3, 4, 40, 41, 42]
seq_lens       = computed + scheduled = [12, 5, 43]
logits_indices = last row of each request = [1, 6, 9]   (one per request without drafts)
```

4. **Where K/V go** (`prepare_attn`). `BlockTables.gather_block_tables` builds this batch's block tables from the
   slots, and the Triton kernel behind `BlockTables.compute_slot_mappings` (`vllm/v1/worker/gpu/block_table.py`)
   computes `slot = block_table[row, pos // block_size] × block_size + pos % block_size`, padding the tail with
   `PAD_SLOT_ID` so CUDA-graph shapes stay fixed. If request 2's block table is `[7, 3, 12]`, position 42 lands
   in block 12 (42 // 16 = 2) at offset 10: slot `12 × 16 + 10 = 202`. `model_state.prepare_attn` then turns
   `query_start_loc`, `seq_lens`, block tables and slot mappings into each layer group's attention metadata
   (Section 6.1).
5. **Forward.** A FULL batch replays its graph with `cudagraph_manager.run_fullgraph` (inputs are already in the
   graph's static buffers); PIECEWISE runs `run_pw_graph`; NONE (eager, `--enforce-eager`) calls `self.model(...)`.
   The last two run under `set_forward_context(attn_metadata, ..., slot_mapping=...)` (`vllm/forward_context.py`),
   and `Attention` layers read their metadata inside the custom ops `vllm::unified_kv_cache_update` and
   `vllm::unified_attention_with_output` (`vllm/model_executor/layers/attention/attention.py`). The hidden states
   are parked in `execute_model_state`, and `execute_model` returns `None`.
6. **Sample** (`sample_tokens` → `sample`). Logits are computed only for `hidden_states[logits_indices]`;
   `StructuredOutputsWorker.apply_grammar_bitmask` masks them; `Sampler` or `RejectionSampler` (Section 7) draws;
   an `AsyncOutput` starts the device-to-host copy on a side stream; `postprocess_sampled` runs the `post_update`
   kernel, which advances `num_computed_tokens`, stores `last_sampled_tokens` and appends to `all_token_ids` on
   the GPU; then the speculator, if any, proposes the next drafts.

Sampled and draft tokens never leave the device between steps, which is what lets async scheduling and
speculative decoding compose without host syncs.

### 5.4 Model runner V1: the persistent batch (fallback and contrast)

`vllm/v1/worker/gpu_model_runner.py: GPUModelRunner` runs when MRV2 declines a configuration (Section 1.4) or
`VLLM_USE_V2_MODEL_RUNNER=0`. It keeps an `InputBatch` (`vllm/v1/worker/gpu_input_batch.py`): fixed-capacity arrays
with a row per request, including `token_ids_cpu_tensor` (`max_num_reqs × max_model_len`),
`num_computed_tokens_cpu_tensor`, a `MultiGroupBlockTable` (one block table per KV cache group) and sampling
parameters as tensors. The bet, stated in `_update_states`: "consecutive batches contain mostly the same
requests". The step is the same six stages with different machinery:

| Stage | MRV1 | MRV2 |
|---|---|---|
| apply the diff | `_update_states`: drop finished, keep unscheduled rows' state, add new/resumed, append token and block ids, `condense()` the gaps | per-slot writes, no condensing |
| order | `reorder_batch_to_split_decodes_and_prefills` (`vllm/v1/attention/backends/utils.py`) moves rows in the persistent batch | `sort_batch_req_ids` orders `idx_mapping`; slots never move |
| inputs | `_prepare_inputs` on the CPU with NumPy: `np.repeat` for `req_indices`, `input_ids = token_ids_cpu.flatten()[positions + req_indices × max_model_len]`, then copies to the GPU; `_prepare_input_ids` patches in last step's sampled ids under async scheduling | Triton kernels over GPU state |
| slots, metadata | `BlockTable.compute_slot_mapping` (`vllm/v1/worker/block_table.py`); `_build_attention_metadata` builds a `CommonAttentionMetadata` and each group's `AttentionMetadataBuilder.build` | `BlockTables.compute_slot_mappings`; `model_state.prepare_attn` |
| graph choice | `_determine_batch_execution_and_padding` asks the `CudagraphDispatcher` (`vllm/v1/cudagraph_dispatcher.py`) | `dispatch_cg_and_sync_dp` |
| logits | computed in `execute_model` for the rows to sample | computed in `sample_tokens` |
| sampling | `vllm/v1/sample/sampler.py: Sampler`, `vllm/v1/sample/rejection_sampler.py: RejectionSampler`; `AsyncGPUModelRunnerOutput` copies on a side stream | `vllm/v1/worker/gpu/sample/sampler.py`, `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` |

### 5.5 torch.compile and CUDA graphs

The two are orthogonal (the `CompilationConfig.cudagraph_mode` docstring says so).

**Compilation.** `CompilationMode.VLLM_COMPILE` (3) is the default. Models opt in with
`@support_torch_compile` (`vllm/compilation/decorators.py`); Dynamo traces the forward once with a symbolic batch
size; `VllmBackend` (`vllm/compilation/backends.py`) runs custom Inductor passes (norm/activation/quantization
fusions, collective fusions), and `split_graph` cuts the FX graph at the **splitting ops**, by default the
attention-like custom ops (`vllm::unified_attention_with_output`, `vllm::unified_mla_attention_with_output`,
Mamba mixers, ...; `CompilationConfig._attention_ops`). Each piece is compiled by a `PiecewiseBackend`
(`vllm/compilation/piecewise_backend.py`) and cached on disk, so a restart with the same model and config logs
"Directly load the compiled graph(s) ..." instead of recompiling (`CompilerManager`).

**CUDA graphs.** `CUDAGraphMode` (`vllm/config/compilation.py`):

| Mode | Captured | Use |
|---|---|---|
| `NONE` | nothing | debugging, `--enforce-eager` |
| `PIECEWISE` | compiled pieces between attention ops; attention eager | any batch, any backend |
| `FULL` | whole forward including attention | backends with `AttentionCGSupport.ALWAYS` |
| `FULL_DECODE_ONLY` | full graphs for uniform decode batches, eager otherwise | decode instances in P/D |
| `FULL_AND_PIECEWISE` (default) | full for uniform decode batches, piecewise for mixed | "the most performant mode for most models" |

Whether a full graph may contain attention depends on the backend's `AttentionCGSupport` (`ALWAYS`,
`UNIFORM_BATCH`, `UNIFORM_SINGLE_TOKEN_DECODE`, `NEVER`): FA3 and Triton report `ALWAYS`, FA2 `UNIFORM_BATCH`,
FlashInfer `UNIFORM_BATCH` or `UNIFORM_SINGLE_TOKEN_DECODE` depending on its TRT-LLM decode kernels
(`get_cudagraph_support` in each backend). Unless `--cudagraph-capture-sizes` is given,
`VllmConfig._set_cudagraph_sizes` builds:

```
q           = uniform_decode_query_len = 1 + num_speculative_tokens
max_capture = min(max_num_batched_tokens, min(max_num_seqs × q × 2, 512))   # 1024 on data-center Blackwell (SM100 family)
sizes       = [1, 2, 4] + range(8, 256, 8) + range(256, max_capture + 1, 16)   (+ max_num_batched_tokens if ≤ max_capture)
            + uniform-decode sizes (below)
```

With `max_num_seqs = 256` and no speculation: `min(2048, min(512, 512)) = 512`, so 3 + 31 + 17 = **51 sizes** (1, 2,
4, 8, …, 248, 256, 272, …, 512); the uniform-decode size is `max_num_seqs` itself, 256, already in the grid. With
`k` drafts the token grid is **not** rescaled: vLLM appends `n × q` for a request-count grid
`n ∈ {1, 2, 4, 8, 16, …}` wherever `n × q` fits under the ceiling, because a uniform decode batch can only replay a
graph whose size is an exact multiple of `q` (the source's example: at `q = 17` a captured 560 is useless, since the
batch would need 561). With `k = 2` (`q = 3`) the appended sizes are 3, 6, 12, 24, 48, …, 504, of which nine are new
(3, 6, 12, 264, 312, 360, 408, 456, 504), giving 60 sizes; with `k = 3` every `n × 4` already lies on the grid. With
dynamic speculation each tier's `q` gets its own list (`_set_cudagraph_sizes`). A batch of
*n* tokens replays the next captured size up (a 3-token decode replays size 4); larger batches run without
graphs; `--performance-mode interactivity` captures every size 1–32 to cut padding. Optimization levels
(`vllm/config/vllm.py: OptimizationLevel`): `-O0` no compilation or graphs, `-O1` compilation plus piecewise
graphs, `-O2` (default) adds full graphs, `-O3` currently equals `-O2`; `--enforce-eager` disables both.

Why it matters: a small model's decode step is dozens of kernel launches per layer, and at several
microseconds each, a 32-layer model spends milliseconds just launching; one graph replay replaces them. Hence
`--enforce-eager` typically costs much more ITL on small models than on large ones `(verify: measure)`.

### 5.6 Parallelism inside one engine

- **Tensor parallelism** (`--tensor-parallel-size`): linear layers split column- or row-wise, two all-reduces
  per layer (attention output, MLP). KV heads split across ranks, so the block budget of Section 4.7 is per GPU
  with `num_kv_heads / tp` heads. `initialize_model_parallel` (`vllm/distributed/parallel_state.py`) builds the
  groups (its example: 8 GPUs, TP=2, PP=4 give TP groups `[g0,g1] [g2,g3] [g4,g5] [g6,g7]` and PP groups
  `[g0,g2,g4,g6] [g1,g3,g5,g7]`). `CudaCommunicator.all_reduce` (`.../device_communicators/cuda_communicator.py`)
  picks per call among FlashInfer all-reduce, NCCL symmetric memory, quick all-reduce, custom IPC all-reduce,
  torch symmetric memory and PyNCCL, by availability and message size.
- **Pipeline parallelism** (`--pipeline-parallel-size`): layer stages; EngineCore keeps `pp_size` batches in
  flight through its batch queue (`VllmConfig.max_concurrent_batches`, `step_with_batch_queue`).
- **Data parallelism** (`--data-parallel-size`): independent EngineCores, each with its own scheduler and KV
  cache; the client load-balances (`DPLBAsyncMPClient`) or an external balancer picks the rank
  (`DPAsyncMPClient`; `EngineCoreClient.make_async_mp_client`). For MoE, DP ranks run in lockstep
  (`DPEngineCoreProc`) so expert-parallel collectives line up.

When TP pays off, and NVLink versus PCIe, is covered in
[`../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md`](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md).

---

## 6. Attention backends

### 6.1 The interface

Three classes per backend (`vllm/v1/attention/backend.py`):

| Class | Responsibility | Key methods |
|---|---|---|
| `AttentionBackend` | static description, capability checks | `get_impl_cls`, `get_builder_cls`, `get_supported_kernel_block_sizes`, `supports_head_size/dtype/kv_cache_dtype/compute_capability`, `is_mla`, `validate_configuration` |
| `AttentionMetadataBuilder` | step's `CommonAttentionMetadata` → backend metadata (FA3 scheduler metadata, FlashInfer `plan()`) | `build`, `build_for_cudagraph_capture`, `get_cudagraph_support`, `use_cascade_attention` |
| `AttentionImpl` | kernel calls | `forward(layer, query, key, value, kv_cache, attn_metadata, output)` |

Model code never names a backend: `Attention` layers call the registered custom ops, and the runner supplies
per-layer metadata through the forward context (Section 5.3).

### 6.2 The paged layout, concretely

For FlashAttention the per-layer KV tensor is `[num_blocks, num_kv_heads, block_size, 2 × head_size]`, K and V
packed in the last dimension and split by `kv_cache.transpose(1, 2).split(head_size, dim=-1)`
(`vllm/v1/attention/backends/flash_attn.py: FlashAttentionImpl.forward`); other layouts are `KVCacheLayout`
values resolved once by EngineCore before profiling (`resolve_kv_cache_layout`). Per layer per step:
(1) **write** each scheduled token's K and V to `slot_mapping[i]` (`reshape_and_cache_flash`, or fused into
RoPE/norm kernels where supported); (2) **read** with one varlen call for the whole mixed batch,
`flash_attn_varlen_func(q, key_cache, value_cache, cu_seqlens_q=query_start_loc, seqused_k=seq_lens,
block_table=..., causal=True, ...)`: prefill rows and decode rows share the launch, and the kernel walks each
row's block table. `num_common_prefix_blocks` enables **cascade attention**: a prefix shared by every running
request is attended once and merged with per-request suffixes (`use_cascade_attention`,
`_compute_cascade_attn_prefix_len`); it is off with async speculative decoding and with `--disable-cascade-attn`.

### 6.3 How a backend is chosen

`get_attn_backend` (`vllm/v1/attention/selector.py`) gathers head size, dtype, KV dtype, MLA, sinks, sliding
window and any user block size, and asks the platform. `CudaPlatformBase.get_attn_backend_cls`
(`vllm/platforms/cuda.py`) either validates the backend named by `--attention-backend` (error if invalid) or
walks `_get_backend_priorities` and takes the first backend whose `validate_configuration` returns no reasons
(`VLLM_ATTENTION_BACKEND` no longer exists in `vllm/envs.py`).

| GPU | Capability | Standard attention, in priority order | MLA models |
|---|---|---|---|
| T4 | 7.5 | FlashAttention needs ≥ 8.0 and FlashInfer is floored at 8.0, so **Triton attention** | Triton MLA |
| A100 / L4 | 8.0 / 8.9 | **FlashAttention (FA2)** → FlashInfer → Triton → FlexAttention | FA MLA → FlashMLA (SM 9.x/10.x only) → FlashInfer MLA → Triton MLA |
| H100 / H200 | 9.0 | **FlashAttention (FA3)** → FlashInfer → Triton → FlexAttention | same list |
| B200 / GB200 | 10.0 | **FlashInfer** (TRT-LLM kernels) → FlashAttention (FA4 if supported) → Triton → FlexAttention | FlashInfer MLA → TokenSpeed MLA → CUTLASS MLA → FA MLA → FlashMLA → Triton MLA |
| RTX PRO 6000 / 5090 | 12.0 | FlashAttention → FlashInfer → Triton | Triton MLA |

Floors from each backend's `supports_compute_capability` (`flash_attn.py` ≥ 8.0; `flashinfer.py` 8.0–12.1, with
the SM75 bug reference; `triton_attn.py` any; `mla/flashmla.py` major 9 or 10); FA version from
`get_flash_attn_version` (`fa_utils.py`: FA3 on SM90, FA4 on SM100 when supported, else FA2; overridable). If
`--block-size` excludes a higher-priority backend, the selector warns and suggests dropping the flag.

**In one line each.** *FlashAttention* (vLLM's fork, `vllm.vllm_flash_attn`): one varlen kernel for mixed
batches, paged KV via `block_table`, block sizes in multiples of 16, FP8 KV with descales (only with FA3 on SM 9.0 or FA4 on SM 10.x:
`flash_attn_supports_kv_cache_dtype` in `fa_utils.py`; on an L4 or A100, `--kv-cache-dtype fp8` makes the selector
skip FlashAttention for FlashInfer), sliding windows; FA3
graphs mixed batches, FA2 only uniform ones (algorithm in [`../flash-attention/`](../flash-attention/)).
*FlashInfer* (`flashinfer.py`): `BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper`
with a per-batch `plan()`, plus TRT-LLM-generated kernels (`trtllm_batch_decode_with_kv_cache`,
`trtllm_batch_context_with_kv_cache`) on SM100. *Triton* (`triton_attn.py`, `vllm/v1/attention/ops/`): portable,
any capability, `ALWAYS` graph support. *MLA backends* (`vllm/v1/attention/backends/mla/`): below.

### 6.4 GQA and MLA change the bytes, not the paging

With grouped-query attention `num_kv_heads < num_heads`: Llama-3.1-8B's 32 query heads share 8 KV heads, a 4×
smaller KV than the 32-head equivalent, handled inside the kernel. With MLA, `MLAAttentionSpec` stores one latent
vector per token per layer (`head_size_v = 0`, one "head"; `vllm/v1/kv_cache_interface.py`). DeepSeek-V3
(61 layers, latent 512 + RoPE 64 = 576 values; config `(verify)`):

```
MLA, BF16:        61 × 576 × 2 B                  =  70,272 B/token ≈ 68.6 KiB
MHA equivalent:   61 × 128 heads × 128 × 2 × 2 B  ≈  4.0 MB/token          (57× more)
```

The 57× assumes 128-dim keys and values. DeepSeek-V3's MHA-style heads actually use 192-dim keys (128 + 64 RoPE)
and 128-dim values, so the like-for-like figure is 61 × 128 × (192 + 128) × 2 B ≈ 5.0 MB/token, 71× the latent
cache.

Hence dedicated MLA kernels (FlashMLA, CUTLASS MLA, FlashInfer MLA) and KV dtypes (`fp8_ds_mla`, `nvfp4_ds_mla`).

---

## 7. Sampling and structured output

### 7.1 The sampler's order of operations

MRV2's `Sampler` (`vllm/v1/worker/gpu/sample/sampler.py`, `__call__` → `sample` → `apply_sampling_params`) keeps
every sampling parameter as per-slot GPU state and runs, per step:

1. optionally count NaN logits (`VLLM_COMPUTE_NANS_IN_LOGITS`), before anything touches them;
2. if no row in the batch needs processing, keep the logits as they are; otherwise copy them to a new FP32 tensor;
3. the logits processors in pipeline order ("bias adds, penalties scale, so the two do not commute"):
   `LogitBiasState` (`logit_bias`, `allowed_token_ids`, and masking stop tokens until `min_tokens`),
   `PenaltiesState` (repetition, frequency, presence), `BadWordsState`, then any custom processors; then the
   thinking-budget forcing, last so nothing can undo it;
4. temperature, then `min_p`, in place;
5. top-k/top-p and the draw: FlashInfer's fused sampler when the batch is eligible (some row uses top-k or
   top-p, no greedy rows, no per-request seeds, no processed-logprob requests), otherwise `apply_top_k_top_p`
   followed by `gumbel_sample` (Section 7.2); greedy rows (temperature 0) take the plain argmax;
6. logprobs from the **raw** logits by default (`logprobs_mode = "raw_logprobs"`: before penalties and
   temperature), from the processed ones in the `processed_*` modes;
7. `num_sampled` set to 0 for rows still mid-prefill (Section 3.4).

MRV1's `Sampler.forward` (`vllm/v1/sample/sampler.py`) documents the same pipeline as nine steps and implements
parameters as **batch-level logits-processor classes** (`MinTokensLogitsProcessor`, `LogitBiasLogitsProcessor`,
`MinPLogitsProcessor`; `vllm/v1/sample/logits_processor/interface.py`, `builtin.py`), each declaring
`is_argmax_invariant` and updating its state from a `BatchUpdate` as rows move in the persistent batch; that
row-movement bookkeeping is what MRV2's fixed slots remove. Custom processors load via `--logits-processors`.

### 7.2 Drawing a token without a host sync

`torch.multinomial` forces a CPU-GPU sync, so neither runner uses it. MRV2 draws with the **Gumbel-max trick**:
`argmax(logits / T + g)` with `g = −log(−log u)` is an exact sample from `softmax(logits / T)`. The uniform `u` is
not drawn from a random-number generator but computed as a **counter-based hash**: murmur3 of
`(seed, position, token id)` (`vllm/v1/worker/gpu/sample/gumbel.py: gumbel_noised_argmax`, via
`murmur3_uniform32/64`). The docstring gives the reason: "`keys` indexes the noise, so the same token draws the same
noise wherever it appears; `pos` and `seed` place the draw in the request's stream, which is what lets a draft and
its verification agree." It also makes seeded requests free: a seed is just a per-slot number, with no
`torch.Generator` per request. (Philox, via `tl.rand` in `tl_rand32`, remains for the uniform of the rejection
test, Section 7.4, and in the watermarking sampler.)

MRV1's `random_sample` (`vllm/v1/sample/ops/topk_topp_sampler.py`) uses the exponential-race form of the same
trick, `argmax(probs / q)` with `q ~ Exp(1)`; seeded requests get their own `torch.Generator`, applied row by row
("This can be slow"). In both runners top-k/top-p rows go to FlashInfer's rejection-based sampler by default when
the GPU supports it (SM 8.0–12.1, more than 16 SMs; log "Using FlashInfer for top-p & top-k sampling."; opt out with
`VLLM_USE_FLASHINFER_SAMPLER=0`; `flashinfer_sampler_supported`); it is statistically equivalent, not
bit-identical.

### 7.3 Structured output

`StructuredOutputManager` (`vllm/v1/structured_output/__init__.py`) lives in EngineCore:

1. **Compile**: `grammar_init`, called from `preprocess_add_request` in the input thread, submits compilation
   to a thread pool; the request waits in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` and the scheduler skips it
   meanwhile (`Scheduler._try_promote_blocked_waiting_request`).
2. **Backend**: `--structured-outputs-config` `backend = auto` (default), `xgrammar`, `guidance`, `outlines` or
   `lm-format-enforcer` (`vllm/config/structured_outputs.py`), one per engine ("We do NOT support different
   backends on a per-request basis in V1"). Constraints come from `SamplingParams.structured_outputs` (`json`,
   `regex`, `choice`, `grammar`, `json_object`, `structural_tag`).
3. **Mask**: each step `Scheduler.get_grammar_bitmask` → `grammar_bitmask` fills a packed int32 bitmask (one
   bit per vocabulary entry; a row per structured request and per speculative position) while the GPU runs the
   forward (`EngineCore.step`); the worker applies it to the logits before sampling, in MRV2 with a Triton kernel
   (`vllm/v1/worker/gpu/structured_outputs.py: StructuredOutputsWorker.apply_grammar_bitmask`, called from
   `GPUModelRunner.sample`), in MRV1 with `apply_grammar_bitmask` (`vllm/v1/structured_output/utils.py`).
4. **Advance**: `update_from_output` calls `accept_tokens`; drafts are pre-validated (`validate_tokens`) and
   invalid ones become `-1`, which the rejection sampler always rejects.

Size: a 128,256-token vocabulary needs `ceil(128256 / 32) = 4,008` int32 words = 16,032 B per row; 256 rows are
4.1 MB and 1,024 rows 16.4 MB, which is why the executor's shared-memory chunk defaults to 24 MiB (Section 5.1).

### 7.4 Speculative decoding

**Methods** (`--speculative-config '{"method": ..., "num_speculative_tokens": k}'`, `vllm/config/speculative.py`):
`ngram` (prompt lookup on the CPU, a Numba KMP search for the longest matching suffix;
`vllm/v1/spec_decode/ngram_proposer.py`), `ngram_gpu`, `draft_model`, `eagle` and `eagle3` (a light head fed the
target's hidden states; EAGLE-3 uses auxiliary hidden states from several layers; `vllm/v1/spec_decode/eagle.py`,
`llm_base_proposer.py`), many model-specific **MTP** types (`deepseek_mtp`, `qwen3_next_mtp`, `glm4_moe_mtp`, …;
`MTPModelTypes`) that reuse the checkpoint's multi-token-prediction layers, plus `medusa`, `mlp_speculator`,
`suffix`, `dflash`, `dspark`, `custom_class`.

**Scheduling.** Drafts are just more tokens to compute: after step *N* the drafter proposes `k` tokens (MRV2:
`speculator.propose` at the end of `sample_tokens`, stored in `RequestState.draft_tokens` on the GPU and read by the
next step's `combine_sampled_and_draft_tokens`; MRV1: `GPUModelRunner.propose_draft_token_ids`), step *N+1*
schedules `1 + k` tokens
for the request, `allocate_slots` reserves `num_lookahead_tokens` slots, and `update_from_output` rolls back the
rejected ones. New decodes are padded to `1 + k` rows so uniform-decode full graphs still apply
(`pad_spec_decode` in `schedule`).

**Verification** follows Leviathan et al.: accept draft *x* with probability `min(1, p(x)/q(x))`; at the first
rejection sample a recovered token from `normalize(max(p − q, 0))`; if all `k` pass, append a **bonus** token from
the target. In MRV2, `RejectionSampler` (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`) applies the
sampling parameters to the target logits and calls `rejection_sample`
(`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`), whose `_rejection_kernel` walks each request's
drafts in order and stops at the first rejection:

- greedy target (temperature 0): accept iff the draft equals the target argmax; the argmax is stored directly, so
  a rejection needs no resampling;
- otherwise the ratio test in log space, `log p(x) > log u + log q(x)`, with `u` from Philox (`tl_rand32(seed,
  pos)`); with the default `draft_sample_method = "greedy"` the draft is a point mass (`q(x) = 1`), so the test is
  `p(x) > u`, and `_resample_kernel` draws the recovered token from `p` with *x* removed (the one-hot residual),
  using the same position-keyed Gumbel noise as ordinary sampling.

`rejection_sample_method` (`SpeculativeConfig`) defaults to `"standard"`; `"block"` verifies the drafts jointly
(block verification, Sun et al., arXiv:2403.10444) and `"synthetic"` accepts at configured rates, for benchmarking
without a real drafter. MRV1's `vllm/v1/sample/rejection_sampler.py: RejectionSampler` "strictly follows" the same
algorithm with `rejection_random_sample_kernel` and `sample_recovered_tokens_kernel` (their `NO_DRAFT_PROBS` paths
are the greedy-draft case). The output distribution is the target's: speculation changes speed, not quality.

**Worked expectation.** With `k = 3` and per-token acceptance α = 0.7 (independence assumed), tokens per target
step = `(1 − α^(k+1)) / (1 − α) = (1 − 0.2401) / 0.3 = 2.53`. At small batch a 4-token verify step costs about a
1-token step (decode is bandwidth-bound), so ITL drops roughly 2.5× minus drafter cost; at large batch the step
is compute-bound and the gain shrinks or inverts, which `num_speculative_tokens_per_batch_size` addresses by
varying `k` with batch size (`dynamic_sd_lookup` in `Scheduler.__init__`). Measure it with
`vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`, `_num_accepted_tokens_per_pos` and the
"Mean acceptance length" log line (`vllm/v1/spec_decode/metrics.py`): mean acceptance length is the measured
counterpart of the 2.53, and the per-position counts test the independence assumption: under it, each position's
accepted count is α times the previous position's (0.7, 0.49, 0.34 of the drafts here).

---

## 8. Quantization and weight loading

### 8.1 How a quantization method is chosen

`--quantization` is optional: `ModelConfig` first reads `quantization_config` from the checkpoint's
`config.json` and otherwise treats the weights as unquantized in `--dtype` (`vllm/config/model.py`).
`get_quantization_config` (`vllm/model_executor/layers/quantization/__init__.py`) maps each accepted name to a
`QuantizationConfig` class:

| Family | Names | Config class | Notes from source |
|---|---|---|---|
| FP8 checkpoints | `fp8` | `Fp8Config` (`fp8.py`) | serialized FP8 weights, static or dynamic activation scales, optional block scales; **no longer quantizes online** (raises, pointing to `fp8_per_tensor`) |
| online, at load time | `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `mxfp4`, `int8_per_channel_weight_only`, `nvfp4_per_token` | `OnlineQuantizationConfig` (`vllm/config/quantization.py: _ONLINE_SHORTHANDS`) | a BF16 checkpoint quantized as it streams in |
| GPTQ / AWQ | `gptq`, `gptq_marlin`, `auto_gptq` / `awq`, `awq_marlin`, `auto_awq` | `AutoGPTQConfig` ("using Marlin kernels"), `AutoAWQConfig` (Triton, Marlin, XPU; min capability 7.5) | |
| llm-compressor | `compressed-tensors` | `CompressedTensorsConfig` | schemes W8A8 FP8/INT8, W8A16 FP8, W4A16, W4A8, W4A4 NVFP4/MXFP4 (`compressed_tensors/schemes/`) |
| vendor formats | `modelopt*`, `quark`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`, `moe_wna16`, `experts_int8`, … | various | `fbgemm_fp8`, `fp_quant` deprecated |

Kernels are chosen per layer. Weight-only int4/int8 layers go through `choose_mp_linear_kernel`, whose CUDA list
is `CutlassW4A8`, `Machete` (Hopper only), `Marlin` (capability ≥ 7.5), `Conch`, `Exllama`, `TritonW4A16`,
`Humming` (`vllm/model_executor/kernels/linear/__init__.py: _POSSIBLE_KERNELS`). FP8 GEMMs try FlashInfer,
CUTLASS, torch `_scaled_mm` variants, then Marlin (`_POSSIBLE_FP8_KERNELS`); on GPUs without FP8 tensor cores
`Fp8LinearMethod` falls back to Marlin weight-only FP8 (`fp8.py`). The choice is logged once ("Using
MarlinLinearKernel for AutoGPTQLinearMethod").

**What the two kernel families do differently.** A weight-only (W4A16) kernel such as Marlin reads packed 4-bit
weights and their per-group FP16 scales from HBM, dequantizes them to FP16/BF16 **in registers**, and feeds the
ordinary 16-bit tensor-core MMA; activations stay 16-bit and accumulation is FP32. It moves a quarter of the weight
bytes but does exactly the BF16 math. A W8A8 FP8 kernel quantizes the activations to FP8 as well (a per-tensor
or per-token scale, static from the checkpoint or computed on the fly) and runs FP8 MMA at twice the BF16 rate,
applying `scale_a × scale_w` to the FP32 accumulator in the epilogue. `process_weights_after_loading` is where each
method prepares its layout: the Marlin path repacks the checkpoint's int32-packed weights into Marlin's tile order
(`ops.gptq_marlin_repack`) and permutes the scales to match (`marlin_permute_scales`,
`vllm/model_executor/kernels/linear/mixed_precision/marlin.py`); the FP8 path, when a fused layer such as
`qkv_proj` arrives as three shards with three per-tensor scales, requantizes them as one weight with one scale
("torch._scaled_mm needs per tensor", `process_fp8_weight_tensor_strategy` in `fp8.py`).

Roofline time of one GEMM, Llama-3.1-8B's `down_proj` (K = 14,336, N = 4,096, 58.7 M weights), for M tokens in the
step: FLOPs `2·M·K·N`, bytes `K·N·w + M·K·a + M·N·2` with `w`, `a` the weight and activation bytes (W4A16 counts
group scales: 4.16 bits per weight). Peaks from `roofline.specs` in
[`../../01-hardware-gpu-fabric/roofline-and-fabric/`](../../01-hardware-gpu-fabric/roofline-and-fabric/)
(L4: 0.30 TB/s, 121 BF16 and 242.5 FP8 dense TFLOP/s `(verify)`); time is `max(FLOPs / peak, bytes / bandwidth)`:

| M (tokens in the step) | BF16 | W4A16 (Marlin) | W8A8 FP8 | bound |
|---|---|---|---|---|
| 1 (one decode) | 392 µs | 102 µs | 196 µs | memory, all three |
| 256 (a full decode batch) | 423 µs | 248 µs | 215 µs | BF16 and FP8 memory; W4A16 compute (BF16 math) |
| 2,048 (a prefill chunk) | 1,988 µs | 1,988 µs | 992 µs | compute, all three |

W4A16's advantage is the byte ratio (3.85× at M = 1) and disappears once the step carries more than about 120
tokens on an L4 (85 on an H100), where the BF16 math becomes the ceiling; dequantization adds real cost on top,
which is why weight-only INT4 can be *slower* than BF16 in large prefills. FP8 W8A8 halves both ceilings, so it
helps at every M, and it needs FP8 tensor cores (Ada, Hopper, Blackwell). The notebook's last section recomputes
this table.

### 8.2 What quantization buys, worked

Decode re-reads every weight each step, except the input embedding, which is only gathered by row; so the bytes
streamed per step set a floor on ITL. Typical FP8 and GPTQ/AWQ checkpoints quantize only the linear layers and
keep `embed_tokens`, `lm_head` and the norms in BF16 (llm-compressor recipes list `lm_head` under `ignore`
`(verify)`; embeddings are not linear layers; check your checkpoint's `quantization_config`). For Llama-3.1-8B that leaves
1.05 B of the 8.03 B parameters in BF16. Byte counts come from `servelab.sizing.weight_bytes` (INT4 at 4.16 bits
per weight with group-128 scales and zero points), freed blocks from `size(...).num_blocks` against the BF16 row of
Section 4.7:

| Llama-3.1-8B weights | Bytes | Streamed per decode step | Floor, L4 (300 GB/s `(verify)`) | H100 SXM (3.35 TB/s `(verify)`) | KV blocks freed on L4 |
|---|---|---|---|---|---|
| BF16 | 16.06 GB | 15.01 GB | 50.0 ms | 4.48 ms | — |
| FP8 (W8A8) linear layers | 9.08 GB | 8.03 GB | 26.8 ms | 2.40 ms | +3,328 (53,248 tokens) |
| INT4 group-128 linear layers | 5.73 GB | 4.68 GB | 15.6 ms | 1.40 ms | +4,927 (78,832 tokens) |

Faster decode and a bigger KV pool compound; on a 24 GB card the pool is often the larger win (FP8 weights more
than double the L4's 2,363 blocks). Prefill is compute-bound: W8A8 FP8 speeds it up on Ada/Hopper/Blackwell tensor
cores, while weight-only INT4 does not cut FLOPs (the table above). `--kv-cache-dtype fp8` is independent and
halves KV bytes (Section 4.7). Accuracy must be measured per model and task (serving-engine primer).

### 8.3 The loading path

```
LoadConfig.load_format ("auto")                                   vllm/config/load.py
 → get_model_loader → DefaultModelLoader                          vllm/model_executor/model_loader/__init__.py, default_loader.py
   → BaseModelLoader.load_model                                   base_loader.py
       with set_default_torch_dtype(dtype), with target_device:
           model = create_model(...)          # modules allocated on the GPU, each with a quant_method
       load_weights(model):
           _prepare_weights → download_weights_from_hf (safetensors preferred; .bin fallback)
           iterator: safetensors_weights_iterator (mmap, "lazy") | multi_thread_... | fastsafetensors_...
           model.load_weights(iterator)       # per-model name mapping; each parameter's weight_loader
                                              # stacks q/k/v into qkv_proj and slices its TP shard
       process_weights_after_loading(...)     # repack for the kernel (e.g. Marlin layout), fuse scales
```

Other `load_format`s: `runai_streamer` (object storage), `tensorizer`, `sharded_state` (pre-sharded per TP rank),
`instanttensor`, `ipc_cache` (map already-quantized weights from a local daemon started with `vllm preload`),
`modelexpress`, `mistral`, `npcache`, `dummy`, and plugins; GGUF and bitsandbytes are not in the registry at
this commit because both moved to out-of-tree plugins, `vllm-gguf-plugin` and `vllm-bnb-plugin` (the
[quantization primer](../quantization/PRIMER.md#verify-list)'s Verify list; `(verify, 2026-09-26)`). `--safetensors-load-strategy` defaults to lazy memory
mapping and enables prefetch automatically on NFS when the checkpoint fits in 90% of RAM (`LoadConfig`).
Cold start in the logs: "Loading weights took X seconds" (I/O and copies), "Model loading took X GiB memory and
Y seconds", compile ("Compiling a graph for compile range … takes X s" or a cache hit), "Graph capturing
finished in N secs", "init engine (…) took N s". The I/O floor is bytes over read bandwidth: 16 GB at an
*assumed* 2 GB/s (one fast NVMe drive or a good network volume; measure yours) is 8 s. Storage tiers and cold
start are covered in section 6 of
[`../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md).

---

## 9. Multi-LoRA, multimodal and hybrid models in brief

**Multi-LoRA.** `LoRAConfig` (`vllm/config/lora.py`): `--enable-lora`, `--max-loras` (default **1** adapter per
batch), `--max-lora-rank` (16), `--max-cpu-loras` (host cache), `--fully-sharded-loras`, `--lora-target-modules`.
Requests carry `LoRARequest(lora_name, lora_int_id, lora_path)`, where `lora_int_id` "must be globally unique
... This is currently not enforced" (`vllm/lora/request.py`). The cap is enforced **only when admitting**: after
the running pass, `scheduled_loras` collects the adapters of the running requests, and the waiting pass parks a
request whose adapter would exceed `max_loras` in `skipped_waiting` with `continue`, not `break`
(`Scheduler.schedule`); running requests are never parked for it. With the default of 1, once adapter A's requests
are running, a request for adapter B is not admitted until **no** A request is running, while newer A requests
behind it keep being admitted. B waits whole A generations, can starve under a steady A stream, and shows up as
`deferred` in `vllm:num_requests_waiting_by_reason`. Size `--max-loras` to the number of adapters that must be
served concurrently. `LRUCacheLoRAModelManager` keeps `max_loras` adapters in GPU slots
and more on the host (`vllm/lora/model_manager.py`, `worker_manager.py`); the base GEMM runs once per batch and
Triton `lora_shrink`/`lora_expand` kernels add per-token deltas by adapter slot
(`vllm/lora/punica_wrapper/punica_gpu.py: PunicaWrapperGPU`); `cudagraph_specialize_lora` (default True) captures
graphs with and without active adapters. The adapter name is in every block hash, so adapters never share KV.
`POST /v1/load_lora_adapter` / `/v1/unload_lora_adapter` exist only with `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
(`vllm/entrypoints/serve/lora/api_router.py`).

**Multimodal.** The renderer turns images, audio or video into placeholder tokens, processed tensors and a
content hash per item (`vllm/multimodal/hasher.py: MultiModalHasher`); processed items can be cached in the
front-end and mirrored in EngineCore so repeated images are not re-sent (`mm_receiver_cache`,
`--mm-processor-cache-gb`). Encoder runs are scheduled like tokens: `_try_schedule_encoder_inputs` spends an
encoder compute budget and space in the `EncoderCacheManager`, which caches outputs by item hash, shares them
across requests and evicts unreferenced entries oldest-first (`vllm/v1/core/encoder_cache_manager.py`). Chunked
prefill may split inside an image's placeholders unless `--disable-chunked-mm-input`. Block hashes include the
item id and offset, so identical text with a different image never hits. Profiling runs the encoder at its worst
case, so large `--limit-mm-per-prompt` values shrink the KV pool.

**Hybrid and sliding-window models**: Section 4.8. Whether a hybrid model can use Model Runner V2 or prefix
caching is decided at startup and logged.

**MoE models**: fused MoE kernels, `--enable-expert-parallel`, the all-to-all backends and EPLB are worked in
[MoE primer §6](../../00-foundations/mixture-of-experts/PRIMER.md#6-running-moe-on-gpus).

**Thinking models**: `--reasoning-parser` splits the output into `reasoning` and `content`, and
`thinking_token_budget` caps the reasoning; both, and what long outputs do to the KV pool, are in
[RL and thinking-models §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving).

---

## 10. Disaggregation and KV transfer

### 10.1 The connector interface

`KVConnectorBase_V1` (`vllm/distributed/kv_transfer/kv_connector/v1/base.py`) is instantiated with
`KVConnectorRole.SCHEDULER` inside the scheduler and with `KVConnectorRole.WORKER` in each worker:

| Side | Method | Called from |
|---|---|---|
| scheduler | `get_num_new_matched_tokens(request, num_computed_tokens)` → (external tokens, load async?) | waiting pass, after the local prefix lookup |
| scheduler | `update_state_after_alloc(request, blocks, num_external_tokens)` | after `allocate_slots` |
| scheduler | `build_connector_meta(scheduler_output)` | end of `schedule()` → `kv_connector_metadata` |
| scheduler | `request_finished(request, block_ids)` → (delay free?, `kv_transfer_params`) | `_free_request` |
| worker | `start_load_kv`, `wait_for_layer_load(layer)`; `save_kv_layer`, `wait_for_save` | around and inside the forward |
| worker | `get_finished` / `get_transfer_results` | reported back in `kv_connector_output` |

Configure with `--kv-transfer-config '{"kv_connector": "...", "kv_role": "kv_producer" | "kv_consumer" |
"kv_both"}'` (`vllm/config/kv_transfer.py`). Registered (`kv_connector/factory.py`): `NixlConnector` (alias of
`NixlPullConnector`), `NixlPushConnector`, `LMCacheConnectorV1`, `LMCacheMPConnector`, `MooncakeConnector`,
`MooncakeStoreConnector`, `OffloadingConnector`, `SimpleCPUOffloadConnector`, `FlexKVConnectorV1`,
`HF3FSKVConnector`, `MoRIIOConnector`, `MultiConnector` (chains several), plus example and bench connectors.

### 10.2 Prefill/decode disaggregation with NIXL, step by step

From `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` (pull mode, the `NixlConnector`
default):

1. A router sends the request to a **prefill** instance with `kv_transfer_params = {"do_remote_decode": true}`
   and `max_tokens = 1` `(verify: the proxy's exact convention)`.
2. The prefill engine computes the prompt and finishes; `request_finished` returns `delay_free_blocks = True`
   and params with `do_remote_prefill`, `remote_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`.
   The blocks stay pinned under a lease (`_kv_lease_duration`) instead of being freed.
3. The router forwards those params to a **decode** instance, where `get_num_new_matched_tokens` returns
   `(prompt tokens − local hits, load_async = True)`; the scheduler allocates blocks, marks the request
   `WAITING_FOR_REMOTE_KVS` and moves on (Section 3.3).
4. The decode worker's connector RDMA-**reads** the remote blocks via NIXL; on `finished_recving`,
   `_update_waiting_for_remote_kv` caches them and the request starts decoding from a full KV cache.
5. The decode side notifies the prefill side, which frees the blocks (the lease frees them anyway if no
   notification arrives). Failed loads follow `kv_load_failure_policy`, default recompute
   (`Scheduler._handle_invalid_blocks`). Different TP layouts on the two sides are mapped by `nixl/tp_mapping.py`.

Routing, P:D ratios and when disaggregation pays off belong to [`../../05-orchestrator/`](../../05-orchestrator/).

### 10.3 Offloading, shared caches, and whether a transfer is worth it

`--kv-offloading-size <GiB>` with `--kv-offloading-backend native|lmcache` adds a CPU KV tier through the same
connector hooks (`CacheConfig.kv_offloading_size`); `OffloadingConnector` and `LMCacheConnectorV1` report external
hits via `get_num_new_matched_tokens`, counted in `vllm:external_prefix_cache_queries/_hits`. Worked numbers for
Llama-3.1-8B, a 4,000-token prompt, BF16 KV:

```
KV to move       = 4,000 × 131,072 B = 524 MB (500 MiB)
prefill compute  ≈ 2 × 8.03e9 × 4,000 = 64 TFLOP ≈ 107 ms at an assumed 600 TFLOP/s (H100)
transfer at link peak:
  450 GB/s  H100 NVLink 4, one direction (900 GB/s is both directions) (verify)     1.2 ms
   50 GB/s  one 400 Gb/s RDMA NIC                                                   10.5 ms
 12.5 GB/s  100 Gb/s                                                                41.9 ms
 1.25 GB/s  10 Gb/s TCP                                                              419 ms
```

These are peak link rates; achieved copy bandwidth is lower and must be measured (the 01 layer's
[`PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) section 5 covers links and the α-β
model). Inside an NVLink node the transfer is about 1% of the prefill; over RDMA it is a tenth and overlaps other
work; over commodity TCP it is four times the prefill it saves. FP8 KV halves every transfer.

---

## 11. Engine arguments that matter, and what each trades

Defaults for `vllm serve` at `5840d95`; "GPU-dependent" follows `EngineArgs.get_batch_defaults` (Section 3.2).

| Flag | Default | Mechanism | TTFT | ITL | Throughput | Memory |
|---|---|---|---|---|---|---|
| `--gpu-memory-utilization` | 0.92 | `requested = total × util`; KV gets the rest (Section 4.7) | — | — | ↑ concurrency | ↑ OOM risk with co-tenants |
| `--kv-cache-memory-bytes` | unset | fixes KV bytes, skips profiling | — | — | — | exact control |
| `--max-model-len` | model maximum; `-1`/`auto` fits memory | per-request cap; one max-length request must fit | — | — | — | ↓ lets small GPUs start |
| `--max-num-batched-tokens` | 2048 / 8192 / 16384 (GPU-dependent) | per-step budget; profiling size | ↓ under load | ↑ longer mixed steps | ↑ | ↑ activations → ↓ KV |
| `--max-num-seqs` | 256 / 1024 | running slots, sampler buffers, graph-size ceiling | — | ↑ at high occupancy | ↑ | ↑ buffers |
| `--max-num-active-seqs` (main after 0.30.0, verify) | = max-num-seqs | admission-only cap | ↑ queueing | ↓ | ↓ | — |
| `--long-prefill-token-threshold` | 0 (off) | per-request chunk cap | ↓ for short prompts behind long ones | ↓ spikes | ≈ | — |
| `--enable-chunked-prefill` | on (decoders) | split prompts across steps | ≈ | ↓ | ↑ | — |
| `--enable-prefix-caching` | on | block-hash reuse (Section 4) | ↓↓ on shared prefixes | — | ↑ | hashing CPU |
| `--prefix-caching-hash-algo` | `sha256` | hash function | — | — | xxhash faster | — |
| `--block-size` | 16 (backend may choose) | tokens per block | hit granularity | — | — | waste ≤ 15 tokens/request; may exclude backends |
| `--kv-cache-dtype` | `auto` (model dtype) | `fp8` halves KV bytes | — | ↓ slightly | ↑ (2× blocks) | ↓; check accuracy |
| `--dtype` | `auto`: the checkpoint's dtype; FP32 checkpoints drop to the platform's preferred 16-bit type; FP16 where BF16 is unsupported (`_resolve_auto_dtype`) | weights/activations dtype | — | — | — | — |
| `--quantization` | from checkpoint | weight format and kernels (Section 8) | ↓ with FP8 | ↓ | ↑ | ↓ weights → ↑ KV |
| `--tensor-parallel-size` | 1 | shard layers; 2 all-reduces/layer | ↓ | ↓ on fast links | per GPU ↓ | weights and KV split |
| `--pipeline-parallel-size` | 1 | layer stages, `pp_size` batches in flight | ↑ | ↑ | ↑ | weights split |
| `--data-parallel-size` | 1 | independent engines | ↓ queueing | — | ↑ | per-engine copies |
| `--async-scheduling` | on when compatible | overlap CPU scheduling and GPU (Section 3.8) | ↓ | ↓ | ↑ | — |
| `--scheduling-policy` | `fcfs` | `priority` orders and preempts by `priority` | per class | per class | — | — |
| `--watermark` | 0.0 | free-block headroom at admission | ↑ queueing | ↓ preemption churn | ≈ | — |
| `--scheduler-reserve-full-isl` | true | admit only if the whole current sequence fits (prompt, plus outputs for a resumed request) | ↑ slightly | ↓ mid-prefill preemption | ≈ | — |
| `--enforce-eager` / `-O` | off / `-O2` | no compile, no graphs | startup ↓↓ | ↑ launch overhead | ↓ | ↓ graph memory |
| `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | formula (5.5) | batch sizes that replay graphs | — | ↓ inside range | — | ↑ memory, startup |
| `--performance-mode` | `balanced` | `throughput` doubles batch defaults; `interactivity` graphs 1–32 | — | `interactivity` ↓ | `throughput` ↑ | — |
| `--speculative-config` | unset | draft + verify (Section 7.4) | — | ↓ at low batch | ↓ at high batch | drafter weights + KV |
| `--stream-interval` | 1 | tokens per streamed chunk | — | smoother at 1 | ↑ when larger | — |
| `--max-num-queued-reqs`, `--max-num-queued-tokens` | unset | API-server admission, 503 when full | bounded | — | — | — |
| `--api-server-count` | 1 (DP size with internal LB) | more front-end processes | ↓ if tokenization-bound | ↓ | ↑ | CPU |
| `--enable-lora`, `--max-loras`, `--max-lora-rank` | off, 1, 16 | Section 9 | — | ↑ LoRA kernels | adapter mix | slots |
| `--kv-offloading-size` / `--kv-transfer-config` | unset | CPU tier / P/D connector (Section 10) | ↓ on reuse | ↓ interference (P/D) | ↑ hit rate | host RAM |
| `--attention-backend` | auto (Section 6.3) | force a backend | — | — | — | — |

The one-line summary a reviewer expects: **capacity** is `gpu_memory_utilization`, weight and KV dtypes and
`max_model_len`; **latency shape** is `max_num_batched_tokens`, `long_prefill_token_threshold` and speculation;
**concurrency** is `max_num_seqs` bounded by capacity; **host overhead** is async scheduling, CUDA graphs,
`stream_interval` and the API-server count.

---

## 12. Observability

### 12.1 Prometheus metrics

Defined in `vllm/v1/metrics/loggers.py: PrometheusStatLogger` (names match `FACTS.md`), labelled `model_name` and
`engine`; counters are exposed with a `_total` suffix by `prometheus_client`.

| Metric | Type | Meaning (from the source) | Use it for |
|---|---|---|---|
| `vllm:num_requests_running` | gauge | requests in model execution batches | occupancy vs `max_num_seqs` |
| `vllm:num_requests_waiting` (+ `_by_reason`: `capacity`/`deferred`) | gauge | waiting + skipped-waiting; deferred = LoRA budget, KV transfer, blocked status | saturation vs blocked; autoscaling |
| `vllm:kv_cache_usage_perc` | gauge 0–1 | `BlockPool.get_usage`: referenced blocks only | headroom, not cache fullness (4.6) |
| `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` | counters (tokens) | token-weighted lookups and hits at first admission (re-admissions after preemption are not counted, 4.4) | hit rate |
| `vllm:external_prefix_cache_queries/_hits` | counters (tokens) | hits served through a KV connector | offload value |
| `vllm:num_preemptions` | counter | cumulative preemptions | KV pressure (3.7) |
| `vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:prompt_tokens_by_source`, `vllm:prompt_tokens_cached` | counters | prefill and generated tokens; prompt tokens by computed/local-cache/external | throughput, effective prefill work |
| `vllm:iteration_tokens_total` | histogram | tokens per engine step | step fullness vs the budget |
| `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds` | histograms | TTFT (front-end), per-iteration ITL, per-request TPOT, end-to-end | SLOs |
| `vllm:request_queue_time_seconds`, `_prefill_time_`, `_decode_time_`, `_inference_time_` | histograms | phase durations from engine events (2.4) | where TTFT goes |
| `vllm:request_prompt_tokens`, `_generation_tokens`, `_params_max_tokens`, `_params_n`, `_max_num_generation_tokens`, `_prefill_kv_computed_tokens`, `_num_preemptions` | histograms | per-request shape, real prefill cost, preemptions per request | workload characterization, tail causes |
| `vllm:request_success` (`finished_reason`) | counter | finished by `stop`/`length`/`abort`/`error`/`repetition` | error and truncation rates |
| `vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`, `_num_accepted_tokens_per_pos` | counters | proposals and acceptances (`vllm/v1/spec_decode/metrics.py`) | acceptance length |
| `vllm:kv_block_lifetime_seconds`, `_idle_before_evict_`, `_reuse_gap_` | histograms | sampled block residency (`--kv-cache-metrics`) | sizing the cache for reuse |
| `vllm:mm_cache_queries/_hits`, `vllm:lora_requests_info`, `vllm:cache_config_info`, `vllm:corrupted_requests`, `vllm:engine_sleep_state` | various | MM cache; LoRA mix; `CacheConfig` as labels (a gauge fixed at 1 emulating Info); NaN logits; sleep level | configuration and faults |

Queries worth keeping: hit rate `rate(vllm:prefix_cache_hits_total[5m]) / rate(vllm:prefix_cache_queries_total[5m])`;
TTFT p99 `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))`; preemption
rate `rate(vllm:num_preemptions_total[5m])`.

### 12.2 Log lines to read

| Line | Source | Tells you |
|---|---|---|
| `Initializing a V1 LLM engine (v…) with config: …` | `EngineCore.__init__` | the resolved config |
| `Using V2 Model Runner`, or `Model Runner V2 does not yet support …; using the V1 model runner instead` | `Worker`, `VllmConfig.use_v2_model_runner` | which runner |
| `Using … attention backend out of potential backends: …` (`Using … backend.` when forced) | `CudaPlatformBase.get_attn_backend_cls` | attention backend |
| `Chunked prefill is enabled with max_num_batched_tokens=…` | `SchedulerConfig.__post_init__` | step budget |
| `Loading weights took … seconds`, `Model loading took … GiB memory and … seconds` | loader, `GPUModelRunner.load_model` (both runners) | cold-start split |
| `Compiling a graph for compile range … takes … s` / `Directly load the compiled graph(s) …` | `vllm/compilation/backends.py` | compile cost, cache hit |
| `Available KV cache memory: … GiB` | `Worker.determine_available_memory` | the number Section 4.7 predicts |
| `GPU KV cache size: … tokens, Maximum concurrency for … tokens per request: …x` | `update_kv_cache_capacity` | capacity |
| `Graph capturing finished in … secs, took … GiB` | `GPUModelRunner.capture_model` | graph cost |
| `init engine (profile, create kv cache, warmup model) took … s` | `EngineCore._initialize_kv_caches` | total startup |

At runtime, every `VLLM_LOG_STATS_INTERVAL` seconds (default 10.0; off with `--disable-log-stats`;
`LoggingStatLogger.log`): `Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: R
reqs, Waiting: W reqs, [Deferred: D reqs,] [Preemptions: P,] GPU KV cache usage: U%, Prefix cache hit rate: H%`,
plus acceptance lines when speculating. Waiting > 0 with KV usage near 100% means capacity-bound; Waiting > 0 with
low usage means budget-bound (`max_num_seqs`, `max_num_batched_tokens`) or blocked (check Deferred); nonzero
Preemptions means the pool is too small for the admitted mix.

---

## 13. Reading and debugging vLLM

### 13.1 Where to run this

| Tier | What | Where (prices and obtainability: [`COMPUTE.md`](../../COMPUTE.md)) |
|---|---|---|
| T0 | this primer, `source-map.md`, the notebook | any laptop or a Colab/Kaggle CPU runtime |
| T1 | the one-process recipe below, the serving lab, metrics, preemption, spec-decode acceptance | a free Colab or Kaggle T4 (16 GB, SM 7.5: `--dtype auto` falls back to FP16, attention runs on Triton, no FP8 path; Section 13.6); any rented 24 GB card (L4, RTX 4090 on RunPod or Vast.ai); a GCP `g2-standard-4` L4 on Spot |
| T2 | tensor/pipeline parallelism (5.6), P/D disaggregation (10) | Kaggle's free 2×T4 (PCIe, no NVLink) for the mechanics; a RunPod/Vast/Lambda 2–8× A100/H100 NVLink box, or GCP A3, for realistic numbers |

### 13.2 Put everything in one process

The fastest way to understand a mechanism is to stop inside it. Three settings collapse vLLM into one debuggable
process (T1: needs a GPU; a 0.6B model fits anywhere):

```python
# debug_one_process.py  (T1)
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # LLMEngine uses InprocClient: EngineCore in this process
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"           # default INFO
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True,   # no torch.compile, no CUDA graphs: steppable model code
          max_model_len=2048, gpu_memory_utilization=0.5)  # TP=1 → UniProcExecutor: the worker is in-process too
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print(out[0].outputs[0].text)
```

Multiprocessing is on by default (`VLLM_ENABLE_V1_MULTIPROCESSING` defaults to 1); with it set to 0,
`LLMEngine.from_engine_args` passes `multiprocess_mode=False` and `EngineCoreClient.make_client` returns an
`InprocClient` (`vllm/v1/engine/llm_engine.py`, `core_client.py`). The server path cannot do this ("Running
EngineCore in asyncio without multiprocessing is not currently supported", `make_client`), so debug the server with
logs and profiles and the engine with this script. The recipe runs the default runner, MRV2, whose input
preparation is Triton kernels: you inspect the resulting tensors rather than step through the arithmetic. To step
through input building line by line in NumPy, add `os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"` and use the MRV1
breakpoints below.

### 13.3 Where to put breakpoints

| Question | Breakpoint |
|---|---|
| Why is my request still waiting? | `Scheduler.schedule` (waiting loop), `KVCacheManager.allocate_slots` returning `None`, `Scheduler._try_promote_blocked_waiting_request` |
| What did the prefix cache find, and what got evicted? | `KVCacheManager.get_computed_blocks`, `FullAttentionManager.find_longest_cache_hit`, `BlockPool.get_new_blocks` → `_maybe_evict_cached_block` |
| Who got preempted? | `Scheduler._preempt_request` |
| What exactly goes to the GPU? | MRV2 (default): `GPUModelRunner.execute_model` after `prepare_inputs` and `prepare_attn` return (inspect `input_batch.positions`, `input_batch.logits_indices`, `slot_mappings`); `BlockTables.compute_slot_mappings`. MRV1 (`VLLM_USE_V2_MODEL_RUNNER=0`): `GPUModelRunner._prepare_inputs`, `_build_attention_metadata`, `BlockTable.compute_slot_mapping` |
| What did the sampler do? | MRV2: `GPUModelRunner.sample`, `vllm/v1/worker/gpu/sample/sampler.py: Sampler.__call__`, `RejectionSampler.__call__` (`vllm/v1/worker/gpu/spec_decode/`). MRV1: `vllm/v1/sample/sampler.py: Sampler.forward` |
| Which attention or GEMM kernel? | `CudaPlatformBase.get_attn_backend_cls`, `choose_mp_linear_kernel` |
| Why did generation stop, or why is a token masked? | `check_stop`, `IncrementalDetokenizer.update`; `Scheduler.get_grammar_bitmask`, then `StructuredOutputsWorker.apply_grammar_bitmask` (MRV2) or `structured_output/utils.py: apply_grammar_bitmask` (MRV1) |
| Where did startup memory go? | `Worker.determine_available_memory`, `get_kv_cache_configs` |

Two cautions: long pauses in a multi-process setup can trip `VLLM_ENGINE_ITERATION_TIMEOUT_S` (default 60); and
code inside compiled, graph-captured regions runs at capture and then replays, so breakpoints there fire during
warm-up, not serving.

### 13.4 Environment variables that matter (and ones that no longer exist)

| Variable | Effect (`vllm/envs.py`) |
|---|---|
| `VLLM_LOGGING_LEVEL` | default `INFO`; `DEBUG` adds scheduler/executor detail and the attention selector's per-backend rejection reasons |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | default 1; 0 runs EngineCore in-process for the offline `LLM` class |
| `VLLM_USE_V2_MODEL_RUNNER` | unset = config default; 0/1 forces the runner (Section 1.4) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 stops subtracting estimated graph memory from the KV budget |
| `VLLM_LOG_STATS_INTERVAL` | seconds between stats lines (default 10.0) |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | count NaN logits per request (`vllm:corrupted_requests`), at some cost |
| `VLLM_TRACE_FUNCTION` | 1 traces every Python call (very slow; for hangs) |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 forces the PyTorch/Triton sampling path |
| `CUDA_LAUNCH_BLOCKING=1` | CUDA, not vLLM: synchronous launches so errors name the right kernel (pair with `--enforce-eager`) |
| `VLLM_USE_V1`, `VLLM_ATTENTION_BACKEND`, `VLLM_TORCH_PROFILER_DIR` | **gone**: V1 is the only engine; use `--attention-backend`; use `--profiler-config` |

### 13.5 Profiling

`vllm serve MODEL --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path"}'`, then `curl -X POST
localhost:8000/start_profile`, send load, `curl -X POST localhost:8000/stop_profile`: traces are written for the API
server (CPU) and each worker (CPU and CUDA) (`vllm/config/profiler.py: ProfilerConfig`; the routes in
`vllm/entrypoints/serve/profile/api_router.py` exist only when a profiler is configured). In Perfetto, look for
gaps between GPU kernels (host overhead) and the `execute_model`/`sample_tokens` split. For Nsight Systems, add
`--enable-layerwise-nvtx-tracing` (per-layer NVTX ranges, `GPUModelRunner._register_layerwise_nvtx_hooks`) and run
under `nsys profile -t cuda,nvtx`; `profiler: "cuda"` and `"proton"` are the other options.

### 13.6 Common errors

| Symptom | Cause | Fix | Message source |
|---|---|---|---|
| `To serve at least one request with the model's max seq len (N), (X GiB KV cache is needed, which is larger than the available KV cache memory (Y GiB)` | default `max_model_len` too long for the pool (4.7) | lower or `auto` `--max-model-len`, FP8 KV, quantized weights, bigger GPU or TP | `_check_enough_kv_cache_memory` |
| `No available memory for the cache blocks` | weights + activations exceed `requested` | raise `--gpu-memory-utilization`, lower `--max-num-batched-tokens`/`--max-num-seqs`, quantize | same |
| `Free memory on device ... on startup is less than desired GPU memory utilization` | another process holds GPU memory | lower utilization or isolate the GPU | `request_memory` |
| `Error in memory profiling. Initial free memory ..., current free memory ...` | another process released memory during profiling | run vLLM in its own container | `determine_available_memory` |
| CUDA OOM during graph capture or first requests | estimates too tight | lower utilization slightly, cap `--max-cudagraph-capture-size`, confirm with `--enforce-eager` | — |
| `no kernel image is available for execution on the device` | no SASS/PTX for your compute capability: the official image builds `TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` with CUDA 13.0.3 (`docker/Dockerfile`), so SM 7.0 (V100) is absent; or a wheel older than your GPU | a build that includes your arch, or build from source with `TORCH_CUDA_ARCH_LIST`; compare `torch.cuda.get_device_capability()` with `torch.cuda.get_arch_list()` | CUDA runtime |
| T4: `Your device ... doesn't support torch.bfloat16. Falling back to torch.float16` | SM 7.5 lacks BF16 (`supported_dtypes`) | expected with `--dtype auto`; `--dtype bfloat16` raises "Bfloat16 is only supported on GPUs with compute capability of at least 8.0 ... --dtype=half" | `_resolve_auto_dtype`, `CudaPlatformBase.check_if_supports_dtype` |
| T4: attention slower than expected | FlashAttention needs SM ≥ 8.0, FlashInfer floored at 8.0 → Triton attention (6.3) | expected; FP8 has no hardware path either (Marlin weight-only fallback) | backends' `supports_compute_capability` |
| Waiting grows, KV usage ≈ 1, preemptions rising | too few blocks for the admitted mix (3.7) | capacity (FP8 KV, shorter context, more GPUs), fewer `max_num_seqs`, `--watermark` | metrics |
| Waiting grows, KV usage low | budget-bound or blocked (LoRA cap, grammar compile, remote KV) | `num_requests_waiting_by_reason`; raise `--max-num-seqs`/`--max-loras` | metrics |
| First request after start is slow | lazy JIT/compile, cold caches | warm-up requests; keep the compile cache on persistent storage | logs |

---

## 14. vLLM vs SGLang vs TensorRT-LLM

All three now share the core ideas (paged KV, continuous batching with chunked prefill, CUDA graphs, speculative
decoding, P/D disaggregation, FP8/FP4), so the useful comparison is architectural. vLLM facts come from this
primer; SGLang and TensorRT-LLM facts from their READMEs (`sgl-project/sglang`, `NVIDIA/TensorRT-LLM`, fetched
2026-09-26); anything more specific is `(verify)`.

| Dimension | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Execution | PyTorch model code, `torch.compile` pieces + CUDA graphs, custom CUDA/Triton kernels | PyTorch runtime with a "zero-overhead CPU scheduler" (README) | "Architected on PyTorch" with a Python LLM API; custom attention/GEMM/MoE kernels (README); historically ahead-of-time TensorRT engines |
| Prefix reuse | block-hash chain, LRU free queue (Section 4) | RadixAttention, a radix tree over token sequences (README; SGLang paper) | "KV cache reuse" (README news); structure `(verify)` |
| Structured output | xgrammar/guidance/outlines bitmask overlapped with the forward | "compressed finite state machine" (README news, 2024); current backends `(verify)` | guided decoding combined with speculation (README news) |
| Disaggregation | KV connector API: NIXL, LMCache, Mooncake, … | P/D disaggregation, large-scale EP (README) | disaggregated serving, wide EP (README) |
| Hardware | NVIDIA, AMD, TPU, CPU, XPU, platform plugins | NVIDIA, AMD, Intel Xeon, Google TPU, Ascend (README) | NVIDIA only |
| Serving entry | `vllm serve` (OpenAI-compatible) | OpenAI-compatible server | `trtllm-serve`; integrates with Dynamo and Triton Inference Server (README) |

What the rows mean mechanically:

- **Prefix reuse: hash chain versus radix tree.** vLLM hashes each full 16-token block once, as tokens arrive (in
  the EngineCore input thread), and a lookup is one dictionary probe per block until the first miss: cost
  proportional to the prompt's block count, reuse in whole blocks, and the last partial block always recomputed
  (Section 4.4). The radix tree of the SGLang paper (arXiv:2312.07104) matches token by token along its edges, so
  reuse is exact to the token and the sharing structure is explicit: every child shares its parent's prefix. Its
  cost is a tree walk and node splits on insert on the scheduler's CPU path; eviction is LRU over leaves, which
  gives the same "tail before root" order vLLM gets by freeing each chain in reverse onto an LRU queue. Whether
  current SGLang still pages at one token is `(verify)`.
- **Structured output: overlap versus skipping.** vLLM's lever is overlap: the bitmask is computed on the CPU while
  the GPU runs the forward (Section 7.3), so a grammar costs almost nothing per step if the mask is ready in time.
  The compressed-FSM idea from the SGLang paper adds skipping: where the grammar allows only one continuation
  (a fixed key name, a bracket), several tokens are appended without a model step. Current SGLang backends and
  whether they still jump forward are `(verify)`.
- **Execution: JIT versus ahead-of-time.** vLLM compiles with `torch.compile` at startup and caches the result
  (Section 5.5). TensorRT-LLM historically built a TensorRT engine per model, parallel layout and GPU ahead of time:
  fast fused kernels, a long build, a rebuild on any change. Its README now leads with a PyTorch-based
  architecture; how much of the engine-build path remains in a typical deployment is `(verify)`.
- **Disaggregation: where the transfer hooks live.** In vLLM the KV transfer is a plugin with a scheduler half and a
  worker half (Section 10.1), so the same engine binary serves prefill, decode or both by configuration. The
  equivalent seams in SGLang and TensorRT-LLM are `(verify)`.

As a design argument, not a benchmark claim: choose **vLLM** for breadth (models, hardware, a stable OpenAI
surface, integration points for routers and caches such as KV events, connectors and metrics) and a codebase you
can read and patch in Python; llm-d, Dynamo and KServe all integrate it `(verify: each project's current support
matrix)`. Choose
**SGLang** when requests share structure heavily (agents, multi-turn, tree search, many structured calls), where
a radix tree makes partial-prefix reuse natural; it is also widely used for RL rollouts (README). Choose
**TensorRT-LLM** on an NVIDIA-only fleet when the last increment of per-GPU performance is worth a vendor-tied
stack `(verify on your workload)`. Rankings change release to release: measure with the serving lab's harness
([`../serving-engine/vllm-serving-lab/`](../serving-engine/vllm-serving-lab/)), which targets any
OpenAI-compatible endpoint.

---

## In a design review

### The two-minute walkthrough

"A request hits the API server, which templates and tokenizes it and sends an `EngineCoreRequest` over ZMQ to the
EngineCore process. EngineCore's input thread builds a `Request` and hashes its prompt in 16-token blocks, each hash
chained to the previous one, so block *k*'s hash names the whole prefix up to *k*.

"Every step the scheduler hands out a token budget, 2,048 tokens by default on a 24 GB card. Running requests go
first; they are mostly decodes needing one token. The remaining budget goes to waiting requests: look up the longest
cached prefix, check the whole prompt fits in free blocks, then schedule as many tokens as the budget allows. That is
chunked prefill: no phases, just requests catching up to their length.

"If a running request needs a block and none is free, we preempt the most recently admitted request, free its blocks
and put it back at the head of the queue. It comes back only when its whole sequence fits again, and until then
nothing queued behind it is admitted. Freed blocks keep their hashes in an LRU free list, tail of each chain first
in line for reuse, so a repeated prompt finds its blocks again, and a preempted request finds whatever the
survivors did not reuse in the meantime.

"The GPU side gets only a diff: new requests, new tokens, new block ids. It builds positions and a slot mapping,
runs a compiled model that replays CUDA graphs (full graphs for pure-decode batches, piecewise around attention
otherwise), and makes one varlen attention call per layer over the paged cache. Sampling is a second call so the
CPU can build grammar masks during the forward. The scheduler appends tokens and checks EOS and length; the API
server detokenizes, checks stop strings and streams SSE.

"Capacity is gpu_memory_utilization times memory, minus weights, activations and graphs, divided by bytes per
block: about 2,360 blocks, 38k tokens, for an 8B BF16 model on an L4. Latency shape is the token budget;
concurrency is max_num_seqs bounded by that capacity."

### Drill questions

1. **We doubled `--max-num-batched-tokens` and p99 ITL got worse. Why?** Decodes share steps with prefill chunks,
   so a bigger budget means longer mixed steps and every co-scheduled decode waits for the whole step. The profiling
   pass also ran bigger, so the KV pool shrank and preemptions may have risen. Lower the budget, set
   `--long-prefill-token-threshold`, or move prefill to separate instances.
2. **KV usage reads 35% but the prefix hit rate fell after we added replicas. Is the gauge wrong?** No:
   `kv_cache_usage_perc` counts only blocks referenced by live requests; cached blocks in the free queue count as
   free. The hit rate fell because traffic spreads over more caches, each seeing fewer repeats. The fix is
   prefix-aware routing, not memory.
3. **Estimate KV capacity for Llama-3.1-8B BF16 on an L4 at defaults.** 2 × 32 × 8 × 128 × 2 B = 128 KiB per token,
   2 MiB per block. 22.5 GiB × 0.92 = 20.7 GiB, minus 15.0 GiB weights and ~1.1 GiB of activations, graphs and
   non-torch memory (the lab's estimate), leaves ~4.6 GiB: ~2,363 blocks, ~38k tokens. The 131k default context
   needs 16 GiB for one request, so the engine refuses to start until `--max-model-len` drops (16k gives 2.3×
   concurrency). The log line `Available KV cache memory` replaces the estimate.
4. **Why recompute on preemption instead of swapping, and how much does it cost?** V1's scheduler has no swap
   path; a CPU tier exists as a connector (`--kv-offloading-size`), not as preemption. The victim keeps its token
   ids, and its full blocks go to the free queue with their hashes, tail first in line. But re-admission needs its
   whole current sequence to fit, counting its own cached blocks as needed capacity, so under the pressure that
   caused the preemption it usually waits at the head of the queue (blocking every admission behind it) while the
   running requests reuse its blocks from the tail. It then recomputes whatever was taken: little if a request
   finished soon, most of its context if not (Section 3.7).
5. **A prompt is fully cached. Why are 16 tokens still computed?** The last prompt token must run to produce logits,
   so the hit is capped at `num_tokens − 1`, and hits are whole blocks, so the final block is recomputed.
6. **Where are stop conditions checked?** EOS, `stop_token_ids`, `max_tokens` and `max_model_len` in the engine
   (`check_stop`); stop strings in the API server's detokenizer, which then aborts the request. A token or two may
   be computed past a stop string.
7. **Two tenants share a system prompt but use different LoRA adapters. Do they share KV?** No: the adapter name
   is an extra key in every block hash. With the default `--max-loras 1` they cannot even share a step: the second
   adapter's requests are not admitted while any request on the first is running, so their TTFT includes whole
   generations of the other tenant, and they can starve under steady first-tenant traffic.
8. **Does speculative decoding change outputs?** No: acceptance with probability `min(1, p/q)` and residual
   resampling preserve the target distribution (greedy drafts make `q` one-hot; greedy targets reduce to exact
   match). It changes speed: a good win at small batch, shrinking at large batch where verification FLOPs cost.

---

## Glossary

| Term | Meaning here |
|---|---|
| API server | front-end process: FastAPI routes, `AsyncLLM`, tokenization, detokenization, SSE |
| EngineCore | process running scheduler, KV cache manager and executor (`vllm/v1/engine/core.py`) |
| `SchedulerOutput` | per-step plan sent to workers: new requests, deltas, tokens per request |
| token budget | `max_num_batched_tokens`: tokens computed per step across all requests |
| `num_computed_tokens` | how far a request's KV is computed; the scheduler moves it toward `num_tokens_with_spec` |
| chunked prefill | computing a prompt over several steps within the budget |
| preemption | freeing a running request's blocks and requeueing it for recompute |
| KV block / null block | `block_size` token slots (default 16) per layer of a group / block 0, a placeholder |
| block hash, extra keys | hash of (parent hash, block tokens, extra keys: LoRA name, MM item id and offset, cache salt, prompt-embeds digest) |
| free block queue | doubly linked list of free blocks in reuse order; cached blocks at the tail |
| KV cache group | layers sharing a cache type and block table (hybrid models have several) |
| slot mapping | per token, `block_id × block_size + offset`, where its K/V are written |
| persistent batch / Model Runner V2 | MRV1's incrementally updated `InputBatch` / slot-based GPU request state with Triton input prep |
| piecewise / full CUDA graph | graphs over compiled pieces between attention ops / one graph for the whole forward |
| capture size | token count with a captured graph; batches are padded up to one |
| async scheduling | scheduling step N+1 while step N runs, using output placeholders |
| attention backend | pluggable kernel family (FlashAttention, FlashInfer, Triton, MLA variants) |
| cascade attention | attending a shared prefix once, then merging with per-request suffixes |
| logits processor | batch-level class that edits logits (min tokens, logit bias, min-p, custom) |
| bitmask | packed allowed-token mask from a grammar, a row per structured request and position |
| rejection sampler, bonus token | verifies drafts against the target; extra target token when all drafts pass |
| KV connector | plugin that loads/saves KV outside the local pool (P/D transfer, offload, shared caches) |

---

## Sources

**vLLM source read for this primer**: repository `vllm-project/vllm`, `main` at `5840d95` (2026-09-25), fetched
2026-09-26. [`source-map.md`](source-map.md) lists every file read, grouped by concept, with line numbers. The
main ones: `vllm/v1/engine/{core,core_client,async_llm,llm_engine,input_processor,output_processor,detokenizer}.py`;
`vllm/v1/core/sched/{scheduler,async_scheduler,output,request_queue,utils}.py`; `vllm/v1/request.py`;
`vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_utils,single_type_kv_cache_manager,kv_cache_coordinator}.py`;
`vllm/v1/kv_cache_interface.py`; `vllm/v1/worker/{gpu_worker,gpu_model_runner,gpu_input_batch,block_table}.py`;
`vllm/v1/worker/gpu/`; `vllm/v1/executor/`; `vllm/compilation/`; `vllm/v1/attention/`; `vllm/platforms/cuda.py`;
`vllm/v1/sample/`; `vllm/v1/structured_output/`; `vllm/v1/spec_decode/`;
`vllm/model_executor/layers/quantization/`; `vllm/model_executor/model_loader/`; `vllm/lora/`;
`vllm/distributed/kv_transfer/`; `vllm/config/`; `vllm/engine/arg_utils.py`; `vllm/envs.py`;
`vllm/v1/metrics/{loggers,stats}.py`; `vllm/entrypoints/`; `docker/Dockerfile`. Other engines: `sgl-project/sglang`
and `NVIDIA/TensorRT-LLM` `README.md` (main, fetched 2026-09-26).

**Papers**: Kwon et al., PagedAttention, SOSP 2023, arXiv:2309.06180 · Yu et al., Orca, OSDI 2022 · Agrawal et
al., Sarathi-Serve, OSDI 2024, arXiv:2403.02310 · Leviathan, Kalman, Matias, speculative decoding, ICML 2023,
arXiv:2211.17192; Chen et al., arXiv:2302.01318 · Li et al., EAGLE, arXiv:2401.15077, and EAGLE-3,
arXiv:2503.01840; Cai et al., Medusa, arXiv:2401.10774 · Zheng et al., SGLang, arXiv:2312.07104 · Zhong et al.,
DistServe, arXiv:2401.09670; Patel et al., Splitwise, arXiv:2311.18677; Qin et al., Mooncake, arXiv:2407.00079 ·
Dao, FlashAttention-2, arXiv:2307.08691; Shah et al., FlashAttention-3, arXiv:2407.08608; Ye et al., FlashInfer,
arXiv:2501.01005 · DeepSeek-AI, DeepSeek-V2 (MLA), arXiv:2405.04434 · Dong et al., XGrammar, arXiv:2411.15100 ·
Frantar et al., GPTQ, arXiv:2210.17323; Lin et al., AWQ, arXiv:2306.00978; Frantar et al., MARLIN, arXiv:2408.11743 ·
Sun et al., block verification for speculative decoding, arXiv:2403.10444.

**In this repo**: [`../serving-engine/PRIMER.md`](../serving-engine/PRIMER.md),
[`../serving-engine/mini-engine-core/minengine/kv.py`](../serving-engine/mini-engine-core/minengine/kv.py) (the same
cache built from scratch), [`../serving-engine/vllm-serving-lab/servelab/sizing.py`](../serving-engine/vllm-serving-lab/servelab/sizing.py)
(the budgets of 4.7 and 8.2), [`../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)
(rooflines, links, storage),
[`../kv-cache/kv-cache-primer.md`](../kv-cache/kv-cache-primer.md),
[`../paged-attention/paged-attention-primer.md`](../paged-attention/paged-attention-primer.md),
[`../flash-attention/flash-attention-primer.md`](../flash-attention/flash-attention-primer.md),
[`../../00-foundations/gpu-capacity-planning/PRIMER.md`](../../00-foundations/gpu-capacity-planning/PRIMER.md).

---

## Verify list (dated 2026-09-26)

| Item | Value used | Why it needs checking |
|---|---|---|
| vLLM state | `main` at `5840d95` (2026-09-25); PyPI latest 0.30.0 (2026-09-22); `torch == 2.13.0` build pin | `main` moves daily; line numbers in `source-map.md` drift |
| Flags newer than the wheel | `--max-num-active-seqs` and `--long-prefill-token-threshold-adaptive` are in `SchedulerConfig`/`EngineArgs` at `5840d95` but not at the `v0.30.0` tag | `vllm serve` from the 0.30.0 wheel rejects them; check the release notes of the next release |
| Model Runner V2 | default when supported; README still says "[Experimental]" | defaults and the unsupported-feature list change often |
| V0 removal | V1 is the only engine; `VLLM_USE_V1` absent | the release that removed V0 is not recorded here |
| GPU memory seen by CUDA | L4 22.49 GiB; H100 80 GB 79.65 GiB (the serving lab's `sizing.GPUS` table) | typical driver-reported totals, not measured in this session |
| Memory bandwidth and peaks | L4 300 GB/s, 121 BF16 / 242.5 FP8 dense TFLOP/s; H100 SXM 3.35 TB/s, 989 / 1,979 | spec sheets (`roofline.specs` in layer 01); used only for floors and roofline times |
| Non-KV overheads in 4.7 | L4 0.42 + 0.5 + 0.2 GiB, H100 1.67 + 0.5 + 0.2 GiB (`servelab.sizing.overhead_estimate`) | estimates; read `Available KV cache memory` and graph-capture lines from logs |
| Quantized checkpoint layout | FP8 and GPTQ/AWQ keep `embed_tokens`, `lm_head` and norms in BF16; INT4 at 4.16 bits per weight | typical of llm-compressor, GPTQ and AWQ exports; check the checkpoint's `quantization_config` |
| Link and storage rates | NVLink 4 450 GB/s per direction; storage read 2 GB/s (assumed) | datasheet peak and an assumption; measure achieved rates |
| Ray vs `mp` step overhead | same `MessageQueue` control plane since `RayExecutorV2` | no measurement here |
| Same-step prefix hits | blocks are hashed at allocation, so a later request in the same `schedule()` may hit them | inferred from `allocate_slots`; confirm with a test |
| DeepSeek-V3 MLA shape | 61 layers, latent 512 + RoPE 64 | model config not fetched here |
| NIXL proxy convention | prefill request sent with `max_tokens = 1` and `do_remote_decode` | proxy and router implementations differ |
| GGUF and bitsandbytes loading | not in the loader registry at this commit: out-of-tree plugins `vllm-gguf-plugin` 0.0.5 and `vllm-bnb-plugin` 0.0.3 (read by the quantization topic, 2026-09-26) | plugin versions, and whether either returns in-tree |
| `--enforce-eager` ITL cost | larger on small models than large ones | measure in the serving lab |
| SGLang and TensorRT-LLM specifics marked `(verify)` | from READMEs and the SGLang paper only | check their docs and code before a decision |
| Orchestrator support | llm-d, Dynamo and KServe integrate vLLM | from FACTS.md (Dynamo) and project descriptions; check each support matrix |
