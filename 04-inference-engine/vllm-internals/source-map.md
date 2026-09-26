# vLLM source map

Where each concept in [`vllm-internals-primer.md`](vllm-internals-primer.md) lives in the vLLM source, and an
order for reading it.

**State read.** Repository `vllm-project/vllm`, branch `main`, commit `5840d95` ("[Bugfix][ROCm] AMD-Quark
mixed-precision DeepSeek-V4.1 support (#57071)", 2026-09-25), fetched 2026-09-26. Latest PyPI release at that time:
0.30.0 (2026-09-22). Line numbers below are for that commit and drift daily on `main`; search for the symbol name if
a number is off. To browse the same tree:
`git clone --filter=blob:none https://github.com/vllm-project/vllm && git -C vllm checkout 5840d95`.

**Tier:** T0 (reading only).

---

## Concept → file → key symbols

| Concept (primer §) | File | Key classes / functions (line at `5840d95`) |
|---|---|---|
| CLI entry, API server launch (1.3) | `vllm/entrypoints/cli/serve.py` | `ServeSubcommand.cmd` (52): API-server count, headless, DP modes |
| | `vllm/entrypoints/launchers/api_server/entry.py` | `build_async_engine_client_from_engine_args` (67): builds `AsyncLLM` |
| Chat route and SSE (2.1) | `vllm/entrypoints/openai/chat_completion/api_router.py` | `create_chat_completion` (54) |
| | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat._create_chat_completion` (259), `chat_completion_stream_generator` (451) |
| Front-end engine (2.2) | `vllm/v1/engine/async_llm.py` | `AsyncLLM` (80), `add_request` (374), `generate` (666), `_run_output_handler` (793) |
| Request construction (2.2) | `vllm/v1/engine/input_processor.py` | `InputProcessor.process_inputs` (327) |
| Detokenize, stop strings (2.2) | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs` (641) |
| | `vllm/v1/engine/detokenizer.py` | `BaseIncrementalDetokenizer.update` (96), `FastIncrementalDetokenizer` (166) |
| Messages (2.3) | `vllm/v1/engine/__init__.py` | `EngineCoreRequest` (109), `EngineCoreOutput`, `EngineCoreOutputs` (256), `FinishReason` |
| IPC client (1.3) | `vllm/v1/engine/core_client.py` | `EngineCoreClient.make_async_mp_client` (137), `AsyncMPClient` (1086), `InprocClient`, `DPLBAsyncMPClient` |
| Engine loop (1.3, 3.8) | `vllm/v1/engine/core.py` | `EngineCore` (111), `_initialize_kv_caches` (258), `step` (633), `step_with_batch_queue` (673), `preprocess_add_request` (1049), `EngineCoreProc` (1088), `run_busy_loop` (1473), `process_input_sockets` (1757), `process_output_sockets` (1859) |
| Request state (3.1) | `vllm/v1/request.py` | `Request` (60), `num_tokens_with_spec`, `update_block_hashes`, `__lt__`, `RequestStatus` (370) |
| Scheduler (3) | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` (557; running pass 624–857, LoRA set 858–866, waiting pass 868–1344, `SchedulerOutput` built at 1462), `_get_local_prefix_cache_hit` (524), `_preempt_request` (1539), `_update_after_schedule` (1584), `get_grammar_bitmask` (1943), `update_from_output` (1967), `_update_request_with_output` (2413), `_free_request` (2628) |
| | `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler` (14), `_update_after_schedule` (25), `_update_request_with_output` (59) |
| | `vllm/v1/core/sched/output.py` | `NewRequestData`, `CachedRequestData`, `SchedulerOutput` (232), `GrammarOutput` |
| | `vllm/v1/core/sched/request_queue.py` | `FCFSRequestQueue`, `PriorityRequestQueue` (131) |
| | `vllm/v1/core/sched/utils.py` | `check_stop` (98) |
| Scheduler defaults (3.2) | `vllm/config/scheduler.py` | `SchedulerConfig`, `get_scheduler_cls`, `verify_max_model_len` |
| | `vllm/engine/arg_utils.py` | `EngineArgs.get_batch_defaults` (2823) |
| KV manager (4.4–4.5, 3.6) | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.record_prefix_cache_stats` (253), `get_computed_blocks` (264), `allocate_slots` (371; full-sequence check 515–531), `free` (610) |
| Block pool, eviction (4.2, 4.6) | `vllm/v1/core/block_pool.py` | `BlockPool` (135), `get_new_blocks` (668), `touch` (754), `free_blocks` (776), `get_usage` (879), `BlockHashToBlockMap` |
| Blocks, free list, hashing, sizing (4.2–4.3, 4.7) | `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock` (177), `FreeKVCacheBlockQueue` (247), `generate_block_hash_extra_keys` (611), `hash_block_tokens` (650), `get_request_block_hasher` (827), `get_kv_cache_groups` (2286), `get_kv_cache_configs` (2650), `get_kv_cache_config_from_groups`, `update_kv_cache_capacity` |
| | `vllm/utils/hashing.py` | `sha256`, `get_hash_fn_by_name` |
| Per-type managers, hybrid (4.8), admission count (3.6) | `vllm/v1/core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager.get_num_blocks_to_allocate` (180; evictable hits 255–265), `free` (578), `FullAttentionManager` (741; `find_longest_cache_hit` 745), `SlidingWindowManager` (948), `MambaManager` (1445) |
| | `vllm/v1/core/kv_cache_coordinator.py` | `UnitaryKVCacheCoordinator`, `HybridKVCacheCoordinator` (609), `get_kv_cache_coordinator` |
| KV specs, page size (4.7, 6.4) | `vllm/v1/kv_cache_interface.py` | `AttentionSpec` (483, `page_size_bytes`), `FullAttentionSpec`, `SlidingWindowSpec`, `MLAAttentionSpec`, `MambaSpec` |
| Memory profiling (4.7) | `vllm/v1/worker/gpu_worker.py` | `Worker.determine_available_memory` (571), `compile_or_warm_up_model` (820) |
| | `vllm/utils/mem_utils.py`, `vllm/v1/worker/utils.py` | `memory_profiling` (230), `request_memory` |
| Cache defaults (4.3, 4.7) | `vllm/config/cache.py` | `CacheConfig` (block size 16, `gpu_memory_utilization` 0.92, hash algorithm) |
| Executors (5.1) | `vllm/v1/executor/abstract.py` | `Executor.get_class` (52) |
| | `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor` (111), `collective_rpc`, `_get_output_rank`, `WorkerProc` |
| | `vllm/v1/executor/ray_executor_v2.py` (Ray default), `ray_executor.py` (legacy) | `RayWorkerProc` (79), `RayExecutorV2` (239), `_init_executor` (284), `start_worker_monitor` (487); `RayDistributedExecutor` (66), `_compiled_ray_dag` (543) |
| | `vllm/envs.py`, `vllm/config/parallel.py` | `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` (935, default `"1"`); `ParallelConfig.__post_init__` backend default (989–1034) |
| | `vllm/distributed/device_communicators/shm_broadcast.py` | `MessageQueue`, `ShmRingBuffer` |
| Model runner V2, the default (5.3) | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner` (183), `profile_run` (934), `capture_model` (989), `finish_requests` (1082), `add_requests` (1111), `update_requests` (1166), `gather_batch_req_state` (1206), `prepare_inputs` (1259), `prepare_attn` (1460), `sample` (1496), `postprocess_sampled` (1576), `execute_model` (1616), `sample_tokens` (1982), `sort_batch_req_ids` (2317) |
| | `vllm/v1/worker/gpu/states.py` | `RequestState` (9; `last_sampled_tokens` 65) |
| | `vllm/v1/worker/gpu/input_batch.py` | `InputBatch` (39), `prepare_prefill_inputs` (307), `prepare_pos_seq_lens` (371), `combine_sampled_and_draft_tokens` (453), `get_num_sampled_and_rejected` (523), `post_update` (604) |
| | `vllm/v1/worker/gpu/block_table.py` | `BlockTables` (17), `append_block_ids` (112), `gather_block_tables` (148), `compute_slot_mappings` (190), `_compute_slot_mappings_kernel` (276) |
| | `vllm/config/vllm.py` | `VllmConfig.use_v2_model_runner` (701), `ROCM_DEFAULT_MRV1_ARCHITECTURES` (75), `_get_v2_model_runner_unsupported_features` (3023) |
| Model runner V1, fallback (5.4) | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner` (479), `_update_states` (1192), `_prepare_inputs` (1951), `execute_model` (4149), `sample_tokens` (4530), `profile_run` (6406), `capture_model` (6752) |
| | `vllm/v1/worker/gpu_input_batch.py` | `InputBatch` (90), `condense` |
| | `vllm/v1/worker/block_table.py` | `BlockTable.compute_slot_mapping` (201), `MultiGroupBlockTable` |
| Compilation, CUDA graphs (5.5) | `vllm/config/compilation.py` | `CompilationMode` (37), `CUDAGraphMode` (53), `CompilationConfig._attention_ops` |
| | `vllm/compilation/backends.py` | `split_graph` (549), `VllmBackend` (801), `CompilerManager` |
| | `vllm/compilation/piecewise_backend.py`, `cuda_graph.py`, `decorators.py` | `PiecewiseBackend` (86), `CUDAGraphWrapper` (145), `support_torch_compile` |
| | `vllm/v1/cudagraph_dispatcher.py` | `CudagraphDispatcher` |
| | `vllm/config/vllm.py` | `_set_cudagraph_sizes` (2360), `OptimizationLevel` (129), `max_concurrent_batches` |
| Parallelism (5.6) | `vllm/distributed/parallel_state.py` | `initialize_model_parallel` (1967) |
| | `vllm/distributed/device_communicators/cuda_communicator.py` | `CudaCommunicator.all_reduce` (335) |
| | `vllm/config/parallel.py` | `ParallelConfig.__post_init__` (executor default) |
| Attention interface (6.1) | `vllm/v1/attention/backend.py` | `AttentionBackend` (58), `CommonAttentionMetadata` (386), `AttentionCGSupport` (559), `AttentionMetadataBuilder`, `AttentionImpl` |
| Backend selection (6.3) | `vllm/v1/attention/selector.py`, `vllm/platforms/cuda.py` | `get_attn_backend` (105), `_get_backend_priorities` (83), `CudaPlatformBase.get_attn_backend_cls` |
| Backends (6.2–6.4) | `vllm/v1/attention/backends/flash_attn.py`, `fa_utils.py`, `flashinfer.py`, `triton_attn.py`, `mla/` | `FlashAttentionImpl.forward` (1105+), `get_flash_attn_version` (74), `FlashInferBackend` (397) |
| | `vllm/v1/attention/backends/utils.py` | `reorder_batch_to_split_decodes_and_prefills` (878), `split_decodes_and_prefills` |
| | `vllm/model_executor/layers/attention/attention.py`, `vllm/forward_context.py` | `Attention`, custom ops `unified_attention_with_output`, `set_forward_context` |
| Sampling, MRV2 (7.1–7.2) | `vllm/v1/worker/gpu/sample/sampler.py` | `Sampler` (42), `__call__` (134), `apply_sampling_params` (223), `sample` (275) |
| | `vllm/v1/worker/gpu/sample/gumbel.py`, `logit_bias.py` | `tl_rand32` (66), `gumbel_noised_argmax` (165), `gumbel_sample` (326); `LogitBiasState` (25) |
| Sampling, MRV1 | `vllm/v1/sample/sampler.py` | `Sampler` (21) |
| | `vllm/v1/sample/ops/topk_topp_sampler.py` | `flashinfer_sampler_supported` (77, used by both runners), `random_sample` (544), `flashinfer_sample` |
| | `vllm/v1/sample/logits_processor/` | `BUILTIN_LOGITS_PROCESSORS`, `MinPLogitsProcessor` (23), `LogitBiasLogitsProcessor`, `MinTokensLogitsProcessor` |
| Structured output (7.3) | `vllm/v1/structured_output/__init__.py`, `utils.py` | `StructuredOutputManager` (36), `grammar_init`, `grammar_bitmask`; MRV1 `apply_grammar_bitmask` (101) |
| | `vllm/v1/worker/gpu/structured_outputs.py` | MRV2 `StructuredOutputsWorker` (38), `apply_grammar_bitmask` (58) |
| Speculative decoding, MRV2 (7.4) | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`, `rejection_sampler_utils.py` | `RejectionSampler` (77), `__call__` (289); `_rejection_kernel` (485), `_resample_kernel` (730), `rejection_sample` (998) |
| Speculative decoding, MRV1 | `vllm/v1/sample/rejection_sampler.py` | `RejectionSampler` (45), `rejection_random_sample_kernel` (829), `sample_recovered_tokens_kernel` (1004) |
| | `vllm/v1/spec_decode/` | `NgramProposer` (`ngram_proposer.py`), `EagleProposer` (`eagle.py`), `SpecDecodeBaseProposer` (`llm_base_proposer.py`), `metrics.py` |
| | `vllm/config/speculative.py` | `SpeculativeConfig`, `MTPModelTypes`, `EagleModelTypes` |
| Quantization (8.1) | `vllm/model_executor/layers/quantization/__init__.py` | `QuantizationMethods`, `get_quantization_config` (112) |
| | `.../quantization/fp8.py`, `auto_gptq.py`, `auto_awq.py` | `Fp8Config`, `Fp8LinearMethod`, `AutoGPTQConfig`, `AutoAWQConfig` |
| | `vllm/model_executor/kernels/linear/__init__.py` | `_POSSIBLE_KERNELS` (500), `_POSSIBLE_FP8_KERNELS`, `choose_mp_linear_kernel` |
| | `vllm/config/quantization.py` | `_ONLINE_SHORTHANDS` |
| Weight loading (8.3) | `vllm/model_executor/model_loader/` | loader registry (`__init__.py`), `BaseModelLoader.load_model` (56), `DefaultModelLoader` (50), `weight_utils.safetensors_weights_iterator` |
| | `vllm/config/load.py` | `LoadConfig.load_format`, `safetensors_load_strategy` |
| Multi-LoRA (9) | `vllm/config/lora.py`, `vllm/lora/` | `LoRAConfig`, `LoRARequest`, `LoRAModelManager` (77), `LRUCacheLoRAModelManager`, `LRUCacheWorkerLoRAManager`, `PunicaWrapperGPU` |
| Multimodal (9) | `vllm/multimodal/hasher.py`, `vllm/v1/core/encoder_cache_manager.py` | `MultiModalHasher`, `EncoderCacheManager` (19) |
| KV connectors (10) | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | `KVConnectorBase_V1` (178), `KVConnectorRole` |
| | `.../kv_connector/factory.py` | connector registry (`NixlConnector` at 178) |
| | `.../kv_connector/v1/nixl/pull_scheduler.py`, `tp_mapping.py` | `get_num_new_matched_tokens` (34), `request_finished` |
| | `vllm/config/kv_transfer.py` | `KVTransferConfig`, `kv_role` |
| Metrics and logs (12) | `vllm/v1/metrics/loggers.py` | `LoggingStatLogger` (104), `PrometheusStatLogger` (454), `vllm:num_requests_waiting_by_reason` (528) |
| | `vllm/v1/metrics/stats.py` | `IterationStats`, `PrefixCacheStats` (117; `record` 132 splits preempted re-admissions), `update_from_finished_request` (532) |
| Debugging knobs (13) | `vllm/envs.py`, `vllm/config/profiler.py`, `vllm/entrypoints/serve/profile/api_router.py` | env vars, `ProfilerConfig`, `/start_profile`, `/stop_profile` |

---

## Every file read for the primer

132 files of the vLLM repository (read in full or in the sections the primer cites), plus two external READMEs.

- **Engine and API (18):** `vllm/v1/engine/{__init__,core,core_client,async_llm,llm_engine,input_processor,output_processor,detokenizer,parallel_sampling}.py`;
  `vllm/entrypoints/openai/api_server.py`; `vllm/entrypoints/openai/chat_completion/{api_router,serving,protocol}.py`;
  `vllm/entrypoints/launchers/api_server/entry.py`; `vllm/entrypoints/cli/serve.py`;
  `vllm/entrypoints/serve/{profile,lora}/api_router.py`; `vllm/renderers/online_renderer.py`.
- **Scheduling and requests (9):** `vllm/v1/core/sched/{scheduler,async_scheduler,output,request_queue,utils,interface}.py`;
  `vllm/v1/request.py`; `vllm/v1/outputs.py`; `vllm/sampling_params.py`.
- **KV cache (9):** `vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_utils,single_type_kv_cache_manager,kv_cache_coordinator,encoder_cache_manager}.py`;
  `vllm/v1/kv_cache_interface.py`; `vllm/utils/{hashing,mem_utils}.py`.
- **Workers, executors, distributed (22):** `vllm/v1/worker/{gpu_worker,gpu_model_runner,gpu_input_batch,block_table,utils}.py`;
  `vllm/v1/worker/gpu/{README.md,model_runner.py,states.py,input_batch.py,block_table.py,sample/gumbel.py,sample/sampler.py}`;
  `vllm/v1/executor/{abstract,multiproc_executor,ray_executor,ray_executor_v2,uniproc_executor}.py`; `vllm/v1/cudagraph_dispatcher.py`;
  `vllm/distributed/device_communicators/{shm_broadcast,cuda_communicator}.py`; `vllm/distributed/parallel_state.py`;
  `vllm/forward_context.py`.
- **Compilation (5):** `vllm/compilation/{backends,piecewise_backend,cuda_graph,decorators}.py`; `vllm/config/compilation.py`.
- **Attention (11):** `vllm/v1/attention/{backend,selector}.py`; `vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn,fa_utils,utils}.py`;
  `vllm/v1/attention/backends/mla/{flashmla,triton_mla}.py`; `vllm/platforms/cuda.py`;
  `vllm/model_executor/layers/attention/attention.py`.
- **Sampling, structured output, speculation (16):** `vllm/v1/sample/{sampler,rejection_sampler}.py`;
  `vllm/v1/sample/ops/topk_topp_sampler.py`; `vllm/v1/sample/logits_processor/{__init__,builtin,interface}.py`;
  `vllm/v1/structured_output/{__init__,utils}.py`; `vllm/v1/spec_decode/{eagle,ngram_proposer,llm_base_proposer,metrics}.py`;
  MRV2's `vllm/v1/worker/gpu/{structured_outputs.py,sample/logit_bias.py,spec_decode/rejection_sampler.py,spec_decode/rejection_sampler_utils.py}`.
- **Quantization and loading (13):** `vllm/model_executor/layers/quantization/{__init__,fp8,auto_gptq,auto_awq}.py`;
  `vllm/model_executor/kernels/linear/__init__.py`; `vllm/model_executor/kernels/linear/mixed_precision/{marlin,machete}.py`;
  `vllm/model_executor/model_loader/{__init__,base_loader,default_loader,weight_utils}.py`; `vllm/config/{quantization,load}.py`.
- **LoRA, multimodal, KV transfer (12):** `vllm/config/{lora,kv_transfer}.py`; `vllm/lora/{request,model_manager,worker_manager}.py`;
  `vllm/lora/punica_wrapper/punica_gpu.py`; `vllm/multimodal/hasher.py`;
  `vllm/distributed/kv_transfer/kv_connector/factory.py`; `vllm/distributed/kv_transfer/kv_connector/v1/base.py`;
  `vllm/distributed/kv_transfer/kv_connector/v1/nixl/{connector,pull_scheduler,tp_mapping}.py`.
- **Configuration, metrics, build (17):** `vllm/config/{vllm,cache,scheduler,parallel,model,speculative,structured_outputs,profiler,observability}.py`;
  `vllm/engine/arg_utils.py`; `vllm/envs.py`; `vllm/v1/metrics/{loggers,stats}.py`; `pyproject.toml`; `vllm/version.py`;
  `CMakeLists.txt`; `docker/Dockerfile`.
- **Other engines (2):** `sgl-project/sglang/README.md`, `NVIDIA/TensorRT-LLM/README.md` (main, fetched 2026-09-26).

---

## How to read the code: three sittings of about two hours

Keep the primer open beside the code; each block names the primer section it pairs with. Read with a question in
mind rather than top to bottom. Pace: dense engine code that you trace with the primer open goes at roughly 8–12
lines a minute, so each slot below names line ranges (at `5840d95`) and the branches to step over on a first read;
most of `Scheduler.schedule`'s 960 lines are guarded blocks for connectors, encoders, Mamba and data parallelism.
The runner slots read **Model Runner V2**, the default (primer 1.4); MRV1 is optional contrast at the end.

**Sitting 1: the engine loop and the scheduler.**

| Time | Read | Question to answer | Primer |
|---|---|---|---|
| 0:00–0:20 | the message dataclasses only: `vllm/v1/engine/__init__.py` (`EngineCoreRequest` 109, `EngineCoreOutputs` 256), `vllm/v1/request.py` (`Request` 60, `RequestStatus` 370), `vllm/v1/core/sched/output.py` (`SchedulerOutput` 232) | What crosses each process boundary? | 2.3 |
| 0:20–0:45 | `vllm/v1/engine/core.py`: `step` (633), `step_with_batch_queue` (673), `EngineCoreProc.run_busy_loop` (1473), `process_input_sockets` (1757), `process_output_sockets` (1859) | What does one engine iteration do, and what runs on which thread? | 1.3, 2.1, 3.8 |
| 0:45–1:25 | `vllm/v1/core/sched/scheduler.py`: `schedule` lines 557–866 only, the budget setup (557–623) and the running pass (624–857) with its preemption loop around `allocate_slots` (743); step over the `defer_prefills` (653), `ec_connector` (660), Mamba-alignment (691), encoder (696) and `_reserve_prefill_lookahead` (716) branches. Then `_preempt_request` (1539–1583) | When is a running request skipped (`continue`) and when does the pass stop (`break`)? Who is the victim? | 3.3, 3.7 |
| 1:25–2:00 | the waiting pass, 868–1344: head and blocked-status checks (872–909), the LoRA check (911–922), the local prefix lookup (931–940), the token count (1073–1130), the `allocate_slots` call (1214–1227) and the admission bookkeeping (1257–1330); step over the connector block (941–1013) and the Mamba and encoder branches (1132–1170) | Replay the §3.5 table by hand, then the §3.7 table: why is R2 refused with its blocks still cached? | 3.3–3.7, 9 |

**Sitting 2: the KV cache and the step's aftermath.**

| Time | Read | Question to answer | Primer |
|---|---|---|---|
| 0:00–0:40 | `vllm/v1/core/kv_cache_manager.py`: `allocate_slots` (371; the docstring's layout first, then the full-sequence check 515–531 and the per-step check after it), `get_computed_blocks` (264), `record_prefix_cache_stats` (253); `single_type_kv_cache_manager.py`: `get_num_blocks_to_allocate` (180–266), `free` (578), `FullAttentionManager.find_longest_cache_hit` (745; phase 1 only, phase 2 applies only when the hash block is smaller than the cache block) | Where exactly do the 150 blocks of §3.7 come from? | 3.6, 4.4, 4.5 |
| 0:40–1:10 | `vllm/v1/core/block_pool.py`: `BlockHashToBlockMap` (34, its NOTE #1), `cache_full_blocks` (225), `get_new_blocks` (668), `_maybe_evict_cached_block` (731), `touch` (754), `free_blocks` (776), `get_usage` (879); `kv_cache_utils.py`: `KVCacheBlock` (177), `FreeKVCacheBlockQueue` (247; `popleft_n` 338, `remove` 372, `prepend_n` 417, `append_n` 438), `generate_block_hash_extra_keys` (611), `hash_block_tokens` (650), `get_request_block_hasher` (827). Then do the notebook | Trace the eviction order of §4.6 | 4.2–4.6 |
| 1:10–1:55 | `scheduler.py: update_from_output` (1967–2412, some 445 lines; read the per-request loop and step over the connector, pooling and stats branches), `_update_request_with_output` (2413), `_free_request` (2628); `vllm/v1/core/sched/utils.py: check_stop` (98); `async_scheduler.py` (all 78 lines) | What changes when a draft is rejected, and when are a request's blocks freed? | 3.8, 3.9 |
| 1:55–2:15 | `vllm/v1/worker/gpu_worker.py: determine_available_memory` (571); `kv_cache_utils.py: _check_enough_kv_cache_memory` (889), `get_kv_cache_config_from_groups` (1658), a skim of `get_kv_cache_configs` (2650); `core.py: _initialize_kv_caches` (258) | Reproduce the §4.7 budget for your GPU | 4.7 |

**Sitting 3: the GPU side (MRV2) and the front end.**

| Time | Read | Question to answer | Primer |
|---|---|---|---|
| 0:00–0:50 | `vllm/v1/worker/gpu/model_runner.py`: `execute_model` (1616–1980, the `not dummy_run` path only; step over the DP, micro-batch, PCP, EPLB and connector branches), `prepare_inputs` (1259–1458), `prepare_attn` (1460); `gpu/block_table.py: compute_slot_mappings` (190) and its kernel (276) — about 500 lines. Leave the request-state updates (`finish_requests`/`add_requests`/`update_requests`, 1082–1205) and the `gpu/input_batch.py` kernels (307–545) for a later session | Compute one token's slot by hand; what crosses from the host each step? | 5.3 |
| 0:50–1:20 | `sample_tokens` (1982–2165), `sample` (1496), `postprocess_sampled` (1576); `gpu/sample/sampler.py`: `__call__` (134), `apply_sampling_params` (223), `sample` (275); `gpu/sample/gumbel.py: gumbel_noised_argmax` (165); skim `gpu/spec_decode/rejection_sampler_utils.py: _rejection_kernel` (485) | How does a sampled token become the next step's input without a host sync, and where does the randomness come from? | 7.1, 7.2, 7.4 |
| 1:20–1:40 | `vllm/v1/attention/backends/flash_attn.py: FlashAttentionImpl.forward` (1105+); `vllm/platforms/cuda.py: CudaPlatformBase.get_attn_backend_cls` and `_get_backend_priorities` | Which backend does your GPU get, and why? | 6 |
| 1:40–2:00 | `async_llm.py`: `generate` (666), `_run_output_handler` (793); `output_processor.py: process_outputs` (641); `detokenizer.py: update` (96); `arg_utils.py: get_batch_defaults` (2823); `metrics/loggers.py` (skim the metric definitions) | Where is TTFT measured, where are stop strings caught, which defaults did your GPU get? | 2.4, 11, 12 |

Left over from sitting 3, about 40 minutes: MRV2's request-state updates (`finish_requests`/`add_requests`/`update_requests`,
1082–1205) and the `gpu/input_batch.py` kernels (307–545) that apply them on the GPU.

Optional contrast, about an hour: MRV1's `vllm/v1/worker/gpu_model_runner.py`, `_update_states` (1192) and
`_prepare_inputs` (1951), some 700 lines together, against the MRV1-versus-MRV2 table in primer 5.4.

Next sessions, by interest: compilation and graphs (`vllm/compilation/`, `vllm/config/compilation.py`,
`vllm/config/vllm.py: _set_cudagraph_sizes` 2360, §5.5); executors and Ray (`vllm/v1/executor/`, §5.1); KV connectors
(`vllm/distributed/kv_transfer/`, §10); speculative decoding proposers (`vllm/v1/spec_decode/`, §7.4).

Tips: `git log -L :schedule:vllm/v1/core/sched/scheduler.py` shows how the scheduler evolved; the tests under
`tests/v1/core/` (for example the prefix-caching and scheduler tests) are the fastest executable specification of the
cache and scheduler behaviour; to watch it live, use the one-process recipe in primer §13.2.
