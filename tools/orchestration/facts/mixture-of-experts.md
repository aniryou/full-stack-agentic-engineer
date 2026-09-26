# FACTS — mixture-of-experts (00-foundations/mixture-of-experts), verified 2026-09-26

Source of truth for the `moe-core` (`moecore`) and `moe-lab` (`moelab`) builders and reviewers. Extends `$SP/FACTS.md` (do not
repeat it). `$SP` = `/tmp/claude-0/-home-user-full-stack-agentic-engineer/ba870d1c-654f-5a6d-8ced-c7c74aaf3b6e/scratchpad`.
Paths are relative to `$SP/ref/` unless they start with `repo:` (= `/home/user/full-stack-agentic-engineer/`) or `$SP/`.
Snapshots: transformers `27166ea` (2026-09-25, `__version__ = "5.18.0.dev0"`; PyPI latest 5.17.0); vllm **main `a4eb3f2`**
(2026-09-26) — the repo pins **v0.30.0** (PyPI latest), so every vLLM flag below was re-checked against the v0.30.0 tag via
raw.githubusercontent.com (copies in `$SP/moe-research/vllm-v0.30.0/`); deepep `a56d615` (2026-09-16); megablocks `952db33`;
sglang `3ed56a3` (2026-09-25); Megatron-LM `main` fetched 2026-09-26 (copies in `$SP/moe-research/{moe_utils,router,transformer_config}.py`);
llama.cpp `master` `common/arg.cpp` (copy in `$SP/moe-research/llamacpp-arg.cpp`). Parameter arithmetic: `$SP/moe-research/params.py`.
"(unverified)" = not found in any source here; the reason follows.

## 1. What the repo already states (reproduce, never contradict)

- `repo:01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/roofline/llm.py`:
  - `experts_touched(n_experts, top_k, tokens)` returns `n_experts * (1 - (1 - top_k / n_experts) ** tokens)`; returns `1.0` when
    `n_experts == 0` (dense). Uniform routing, **with replacement across tokens, without the "k distinct per token" correction**.
  - `streamed_weight_bytes(model, tokens, weight_bytes=2)` = `(n_layers * (attn_params() + experts * expert_params() + router_params())
    + lm_head_params() + tokens * d_model) * weight_bytes` (input embedding = a gather of `tokens` rows, not a stream).
  - `decode_crossover_batch(model, device, context, max_batch=1<<20, **kw)`: smallest batch whose **one-kernel** `decode()` step is
    compute-bound (bisection on `decode(...).bound == "compute"`), `None` if > max_batch.
  - `ModelConfig.expert_params() = (3 if gated_mlp else 2) * d_model * d_ff`; `router_params() = d_model * n_experts`;
    `params()` counts both embeddings unless `tied_embeddings`; `active_params()` = layers × (attn + top_k experts + router) + both embeddings.
  - `PRESETS["mixtral-8x7b"] = ModelConfig("Mixtral-8x7B", 32, 4096, 32, 8, 128, 14_336, 32_000, n_experts=8, top_k=2)`;
    `PRESETS["qwen3-30b-a3b"] = ModelConfig("Qwen3-30B-A3B", 48, 2048, 32, 4, 128, 768, 151_936, n_experts=128, top_k=8)`.
    Recomputed here: Mixtral `params()` 46,702,526,464 / `active_params()` 12,879,659,008; Qwen3 30,531,911,680 / 3,352,821,760;
    Llama-3.1-8B 8,029,995,008.
  - `specs.DEVICES["h200"]`: 989.4 TFLOPS bf16 dense, 4.8 TB/s, 141 GB → `ridge()` = 989.4e12 / 4.8e12 = **206.125** FLOP/B.
- PRIMER §3.6 (`repo:01-.../roofline-and-fabric/PRIMER.md` lines 267–291), pinned by `roofline-core/tests/test_llm.py::test_moe_streams_more_experts_as_the_batch_grows`
  and `tests/test_primer_numbers.py`: Mixtral on H200, 1K context: batch 1 → 2.00 experts/layer, `decode().bytes` = 25,631,531,008 B
  (**25.6 GB** = 25,497,182,208 weight bytes + 134,348,800 KV bytes), time = bytes / 4.8e12 = **5.34 ms** (memory-bound); batch 4 → 5.47,
  65.1 GB, 13.57 ms; 16 → 7.92, 94.4 GB, 19.66 ms; 64 → 8.00, 101.7 GB, 21.20 ms. Qwen3 column: 8.0/128, 29.1, 82.4, 125.9.
  Crossover at c = 0 on H200: **207** (Llama-3.1-8B), **754** (Mixtral), **2,055** (Qwen3-30B-A3B); the test asserts
  crossover(MoE)/crossover(Llama) ≈ (params − vocab·d)/(active − vocab·d) within 1% (Mixtral 3.6, Qwen3 9.9; total/active 3.6 and 9.1).
- `repo:00-foundations/gpu-capacity-planning/capacity.py`: `ModelSpec(name, params_b, layers, kv_heads, head_dim, active_b=None)`;
  `MISTRAL_LARGE = ModelSpec("Mistral Large 3 (675B MoE)", params_b=675, layers=88, kv_heads=8, head_dim=128, active_b=41)`;
  `prefill_flops(active_b, prompt_tokens) = 2 * active_b * 1e9 * prompt_tokens`; `ttft_s(active_b, ...)`, `prefill_tok_s(active_b, ...)`.
  PRIMER "When one GPU (or one node) won't do": memory by TOTAL (~675 GB fp8, not 8×H100, floor 8×H200 or Blackwell, NVFP4 ~340 GB),
  compute by ACTIVE, decode ~675 GB/step → "~18 ms floor across 8×H200" (675e9 / (8 × 4.8e12) = 17.6 ms).
- `repo:00-foundations/transformers/docs/transformer-primer.md` §9 table row "Mixture of Experts (MoE) | One MLP per block | E MLPs per block
  and a router that sends each token to the top-k; parameters grow ~E× while compute per token barely moves…"; rows for GQA and latent K/V (MLA).
- `repo:02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md` §5.1 table: all-to-all = "rank r's chunk j goes to rank j | MoE expert parallelism:
  dispatch and combine"; §5.6: dispatch and combine "each `tokens × top_k × hidden × bytes` per GPU"; Mixtral-like, hidden 4,096, top-2,
  256 tokens/GPU → 256·2·4096·2 = 4,194,304 B = **4 MiB** per direction: **22 µs pairwise or 10 µs direct** (`model_time("all_to_all", ...)`);
  "TP and EP inside the NVLink domain, PP and DP across it".
- `repo:04-inference-engine/serving-engine/PRIMER.md` §9: EP = "two all-to-alls per MoE layer, sensitive to load imbalance"; lists
  `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` "(verify)" — **now verified for v0.30.0** (§7).
- `repo:05-orchestrator/serving-orchestration/PRIMER.md` §8 "Large MoE topologies (wide-EP) in brief": EP + data-parallel attention;
  llm-d wide-EP guide: DeepSeek-R1-0528 on 32 H200/B200 as 16-way DP prefill + 16-way DP decode, NIXL over IB/RoCE; each DP rank is a router endpoint;
  hot experts are balanced by the engine, not the router.
- `repo:00-foundations/model-landscape/open-weight-llms-primer.md` §4 (dated 2026, keep consistent): gpt-oss-120b 116.8B/5.1B, gpt-oss-20b 21B/3.6B,
  MXFP4, 120b fits one 80 GB GPU, 20b ~16 GB; Mistral Large 3 675B/41B; Kimi K3 2.8T/104B (896 routed experts, 16 active);
  DeepSeek V4-Pro 1.6T/~49B, V4-Flash 284B/~13B; Qwen3.8-2.4T-A95B (512 experts, "10 routed + 1 shared"); MiniMax M3 428B/~23B;
  Nemotron 3 Nano ~31.6B/3.2B, Super ~120B/12B, Ultra ~550B/~55B; Gemma 4 26B MoE ~4B active; Inkling 975B/~41B.
  Those 2026 models have no configs under ref: cite the primer, mark (verify), do not recompute.
- `repo:04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md`: T4 = sm_75 minimum for vLLM v0.30.0 (CUDA 13 builds),
  `--dtype half` on T4 (no bf16), attention backend `TRITON_ATTN` on T4, P100 (6.0) not supported, Kaggle "GPU T4 x2", driver ≥ 580 (verify).
- `repo:02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/node_pools.tf`: pool **`l4x2`** = `g2-standard-24`, 2 × L4,
  PCIe (no NVLink), `enable_multi_gpu_pool` default `true`, `multi_gpu_count` default 2; `deploy/gke/03-nccl-tests-2gpu.yaml` selects it with
  `cloud.google.com/gke-accelerator: nvidia-l4`, `node.kubernetes.io/instance-type: g2-standard-24`, toleration `nvidia.com/gpu` `Exists`.
  Node-pool name (for `cloud.google.com/gke-nodepool`) = `l4x2` (`name = each.key`).

## 2. Model architectures from configs (params: matmul weights + embeddings + biases; norms ignored — < 0.01%)

Conventions matter: "active" is published with different embedding accounting. Column A = our `active_params()` convention (both embeddings,
like `roofline.llm`); B = LM head only (OpenAI's convention for gpt-oss); matmul-only = neither.

| Model | E routed | k | shared | d | expert I | layers (dense) | attention | total | active A / B | source |
|---|---|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B | 8 | 2 | 0 | 4096 | 14336 | 32 (0) | GQA 32/8, hd 128 | 46.70B | 12.88 / 12.75B | transformers `configuration_mixtral.py` defaults |
| Qwen3-30B-A3B | 128 | 8 | 0 | 2048 | 768 | 48 (0) | GQA 32/4, hd 128, q/k-norm | 30.53B | 3.35 / 3.04B | Qwen3 tech report Table 2 + roofline preset |
| Qwen3-235B-A22B | 128 | 8 | 0 | 4096* | 1536* | 94 (0) | GQA 64/4, hd 128 | 235.09B | 22.19 / 21.57B | Table 2; *d and I unverified, consistent with the name |
| Qwen1.5-MoE-A2.7B | 60 | 4 | 1 (I=5632, sigmoid gate) | 2048 | 1408 | 24 (0) | MHA 16/16, qkv bias | 14.32B | 2.69 / 2.38B | `configuration_qwen2_moe.py` defaults (checkpoint `Qwen/Qwen1.5-MoE-A2.7B`) |
| DeepSeek-V3 (671B) | 256 | 8 | 1 (I=2048) | 7168 | 2048 | 61 (3, I=18432) | MLA 128 heads | 671.03B | 37.55 / 36.62B | `deepseek-v3/inference/configs/config_671B.json` |
| Kimi K2 | 384 | 8 | 1 | 7168 | 2048 | 61 (1) | MLA 64 heads | ≈1,026B** | ≈32.9 / 31.7B | `kimi-k2/README.md` table; report says 1.04T / 32.6B |
| gpt-oss-120b | 128 | 4 | 0 | 2880 | 2880 | 36 (0) | GQA 64/8, hd 64, sinks, SWA 128 on even layers | 116.83B | 5.71 / **5.13B** | `gpt-oss/gpt_oss/torch/model.py` `ModelConfig` |
| gpt-oss-20b | 32 | 4 | 0 | 2880 | 2880 | 24 (0) | same | 20.91B | 4.19 / **3.61B** | E=32 from metal `f32_topk_softmax_e32_k4_fn`; 24 layers derived*** |
| Llama 4 Scout (text) | 16 | 1 | 1 (I=8192) | 5120 | 8192 | 48 (0), MoE every layer | GQA 40/8, hd 128 | 107.77B | 17.17 / 16.14B | transformers `Llama4TextConfig` defaults; card: 17B/109B incl. vision |
| Llama 4 Maverick (text) | 128 | 1 | 1 (I=8192) | 5120* | 8192* | 48, MoE every 2nd (24 MoE + 24 dense I=16384)* | GQA 40/8 | 400.71B | 17.18 / 16.15B | card: 17B/400B; *interleave step 2 unverified, reproduces 400B |
| OLMoE-1B-7B-0924 | 64 | 8 | 0 | 2048 | 1024* | 16 (0) | MHA 16/16, q/k-norm | 6.92B | 1.28 / 1.18B | `olmoe/configs/OLMoE-1B-7B-0924.yml`; README 6.9B/1.3B; *I derived |
| granite-3.0-1b-a400m | 32* | 8* | 0 | 1024* | 512* | 24* | GQA 16/8*, tied emb | 1.33B | 0.43B | *unverified config, reproduces "1b-a400m" |
| granite-3.0-3b-a800m | 40* | 8* | 0 | 1536* | 512* | 32* | GQA 24/8*, tied emb | 3.30B | 0.88B | *unverified config, reproduces "3b-a800m" |

\** Kimi K2 attention assumes DeepSeek-V3's MLA ranks (q_lora_rank 1536, kv_lora_rank 512) — not in the README (unverified); vocab "160K" = 163,840 assumed.
\*** 24 layers for gpt-oss-20b is not in ref; it is the only layer count that reproduces 21B/3.6B (unverified).

Worked arithmetic (show this in the primer; `$SP/moe-research/params.py` prints every line):
- **Mixtral-8x7B**: expert = 3·4096·14336 = 176,160,768; attention = 2·4096·4096 + 2·4096·1024 = 41,943,040; router = 4096·8 = 32,768.
  Layer = 41.94M + 8·176.16M + 0.03M = 1,451.3M; × 32 = 46.44B; + embeddings 2·32000·4096 = 262.1M → **46.70B**. Active: 41.94M + 2·176.16M
  + 0.03M = 394.3M × 32 = 12.62B + 0.26B = **12.88B**. bf16 weights 93.4 GB; one expert (one layer) = 352 MB bf16.
- **Qwen3-30B-A3B**: expert = 3·2048·768 = 4,718,592; attention = 2·2048·4096 + 2·2048·512 = 18,874,368; router 262,144.
  Layer = 18.87 + 128·4.72 + 0.26 = 623.1M; × 48 = 29.91B + 2·151936·2048 = 622.3M → **30.53B**. Active: (18.87 + 8·4.72 + 0.26)M × 48
  = 2.73B + 0.62B = **3.35B**. total/active = 9.1 (matches PRIMER §3.6).
- **DeepSeek-V3**: MLA per layer (`inference/model.py` `MLA.__init__`): `wq_a` 7168·1536 + `q_norm` 1536 + `wq_b` 1536·(128·192)
  + `wkv_a` 7168·(512+64) + `kv_norm` 512 + `wkv_b` 512·(128·(128+128)) + `wo` (128·128)·7168 = **187,107,328**. Expert = 3·7168·2048 = 44,040,192;
  shared expert = `MLP(dim, n_shared_experts * moe_inter_dim)` = 44.04M; gate = 256·7168 + 256 bias. MoE layer = 187.1 + 256·44.04 + 44.04 + 1.84
  = 11,507M; × 58 = 667.4B. Dense layers: (187.1M + 3·7168·18432) × 3 = 1.75B. Embeddings 2·129280·7168 = 1.853B → **671.0B**.
  Active: (187.1 + 9·44.04 + 1.84)M × 58 = 33.95B + 1.75B + 1.85B = **37.55B** (README: 37B). README: HF checkpoint is **685B = 671B main + 14B MTP**.
- **gpt-oss** (`MLPBlock`): per expert `mlp1_weight` [2I, d] + `mlp1_bias` [2I] + `mlp2_weight` [d, I] + `mlp2_bias` [d] = 2·2880·2880 + 5760 + 2880·2880
  + 2880 = 24,891,840; attention `qkv` 2880·(64+2·8)·64 + bias + `out` 4096·2880 + bias + `sinks` 64 = 26.55M; gate 2880·E + E.
  120b: 36·(26.55M + 128·24.89M + 0.37M) + 2·201088·2880 = **116.83B**; active 36·(26.55 + 4·24.89 + 0.37)M + 201088·2880 (head only) = **5.13B**.
  20b: 24 layers, E=32 → **20.91B**, active **3.61B**. MXFP4 estimate at 4.25 bits/weight for expert matrices + bf16 rest: 120b ≈ 60.9 + 4.3 = **65.2 GB**;
  20b ≈ 10.2 + 3.6 = **13.8 GB** (consistent with "single 80GB GPU" / "within 16GB of memory", `gpt-oss/README.md` lines 18–19, 45).
- **Llama 4**: expert = shared = 3·5120·8192 = 125,829,120; attention 62.91M. Scout: 48·(62.91 + 16·125.83 + 125.83 + 0.08)M + 2·202048·5120
  = **107.8B** text (card 109B total incl. vision encoder, unverified split); active 48·(62.91 + 2·125.83)M + 2.07B = **17.2B**.
  Maverick: 24 MoE layers (62.91 + 129·125.83 + 0.66)M + 24 dense (62.91M + 3·5120·16384) + 2.07B = **400.7B**; active **17.2B**.
- **Qwen1.5-MoE-A2.7B**: shared I = 5632 = 4 × 1408 → the shared expert is the size of 4 routed experts; each token uses 4 routed + "4" shared
  = 8 of 64 fine-grained units. Total **14.32B**, active **2.69B**.

KV bytes per token (bf16; `2 · layers · kv_heads · head_dim · 2`): Mixtral 131,072; Qwen3-30B-A3B 98,304; Qwen1.5-MoE 196,608; OLMoE 131,072;
granite-1b 49,152; Llama 4 Scout 196,608; gpt-oss-120b 73,728 (half its layers are 128-token sliding window); DeepSeek-V3 MLA latent
(512 + 64)·61·2 = **70,272** (vLLM caches the compressed latent — verify per backend). MoE does not change these: KV follows attention, not experts.

## 3. Router variants as implemented (quote identifiers exactly)

- **Mixtral** `MixtralTopKRouter.forward` (`transformers/.../mixtral/modeling_mixtral.py` 104–111): `router_logits = F.linear(hidden_states, self.weight)`;
  `router_probs = softmax(router_logits.float())`; `torch.topk(router_probs, self.top_k)`; `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)`
  (**always renormalised**); returns `(router_logits, router_scores, router_indices)`. Config: `num_local_experts=8`, `num_experts_per_tok=2`,
  `router_aux_loss_coef=0.001`, `router_jitter_noise=0.0` (training-only multiplicative noise `uniform(1-j, 1+j)`), `output_router_logits=False`.
  Experts are stored as 3D tensors: `gate_up_proj` [E, 2·I, d], `down_proj` [E, d, I] (`MixtralExperts`); block is `self.mlp` (`MixtralSparseMoeBlock`).
- **Qwen3-MoE** `Qwen3MoeTopKRouter` (`qwen3_moe/modeling_qwen3_moe.py` 249–267): softmax → topk → renormalise **only if `norm_topk_prob`**
  (config default `False`; the released Qwen3 MoE configs set `true` — unverified, huggingface.co blocked). `decoder_sparse_step=1`, `mlp_only_layers`
  = dense layers. No shared expert (Qwen3 tech report §2: "Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts"; "global-batch load balancing loss").
- **Qwen2-MoE** `Qwen2MoeSparseMoeBlock` (lines 339–358): routed output + `F.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)`;
  `shared_expert_gate = nn.Linear(hidden, 1, bias=False)`; `norm_topk_prob=False` default; `router_aux_loss_coef=0.001`.
- **OLMoE** `OlmoeTopKRouter`: softmax → topk, `norm_topk_prob=False`; `router_aux_loss_coef=0.01` (HF default). Training yml: `moe_top_k: 8`,
  `moe_num_experts: 64`, `moe_dropless: true`, `moe_mlp_impl: sparse`, `moe_zloss_weight: 0.001`, `moe_loss_weight: 0.01`.
- **GraniteMoE** `GraniteMoeTopKRouter` (granitemoe 124–143): `topk` on **logits**, then `softmax(top_k_logits)` over the k; returns
  `(top_k_index, top_k_weights, router_logits)` (different order!); module path `block_sparse_moe.router`; **no `router_logits` OutputRecorder**.
- **gpt-oss** (`gpt_oss/torch/model.py` `MLPBlock.forward`): `g = self.gate(t)` (Linear **with bias**); `torch.topk(g, k=4, sorted=True)`;
  `expert_weights = softmax(experts.values, dim=1)` (softmax over the top-4 logits); `swiglu(x, alpha=1.702, limit=7.0)` clamps and uses `(x_linear + 1)`.
  HF `GptOssTopKRouter`: same (topk on logits+bias, softmax over top-k). All non-MoE tensors BF16; MoE projection weights MXFP4 (`tensor.blocks` two FP4
  values per uint8, `tensor.scales` block scale along the last dim) — `gpt-oss/README.md` "Precision format".
- **DeepSeek-V3** `Gate.forward` (`deepseek-v3/inference/model.py` 566–598) — quote:
  ```python
  scores = linear(x, self.weight)
  if self.score_func == "softmax": scores = scores.softmax(dim=-1, dtype=torch.float32)
  else: scores = scores.sigmoid()
  original_scores = scores
  if self.bias is not None: scores = scores + self.bias
  if self.n_groups > 1:
      scores = scores.view(x.size(0), self.n_groups, -1)
      if self.bias is None: group_scores = scores.amax(dim=-1)
      else: group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
      indices = group_scores.topk(self.topk_groups, dim=-1)[1]
      mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
      scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
  indices = torch.topk(scores, self.topk, dim=-1)[1]
  weights = original_scores.gather(1, indices)
  if self.score_func == "sigmoid": weights /= weights.sum(dim=-1, keepdim=True)
  weights *= self.route_scale
  ```
  The **bias only picks experts; the weights come from the unbiased scores**. Config: `score_func "sigmoid"`, `route_scale 2.5`, `n_expert_groups 8`,
  `n_limited_groups 4` (8 groups of 32 experts, keep the 4 groups with the largest sum of their top-2 biased scores, then top-8 inside them) =
  node-limited routing. `self.bias` exists only when `self.dim == 7168` (float32). `MoE.forward` adds `shared_experts(x)` for every token; the reference
  shards experts `n_routed_experts // world_size` per rank and combines with **`dist.all_reduce(y)`**, not an all-to-all.
  16B config (`config_16B.json`, DeepSeek-V2-Lite-like): 64 routed, 2 shared, top-6, softmax (default), `route_scale 1.0`, 1 dense layer.
- HF `DeepseekV3TopkRouter` (`deepseek_v3/modeling_deepseek_v3.py` 131–169): same algorithm; bias is buffer `e_score_correction_bias`
  (`_keep_in_fp32_modules_strict`); config names `n_routed_experts=256`, `num_experts_per_tok=8`, `n_group=8`, `topk_group=4`, `routed_scaling_factor=2.5`,
  `norm_topk_prob=True`, `first_k_dense_replace=3`, `n_shared_experts=1`, `moe_intermediate_size=2048`; `num_local_experts` is an alias of `n_routed_experts`;
  `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]` (the MTP layer).
- **Llama 4** (`llama-models/models/llama4/moe.py` `MoE.forward`): `router_scores = x @ router_DE`; `topk(..., top_k)` (`MoEArgs.top_k = 1`);
  scores → `torch.sigmoid`; the **input** to each expert is scaled: `routed_in_EG_D = routed_in_EG_D * router_scores` (not the output); a
  `shared_expert = FeedForward(dim, hidden_dim)` runs on every token. The reference computes every expert on every token with zero-scaled inputs (dense
  compute) — educational, not sparse. `MoEArgs`: `capacity_factor: float = 1.0`, `auto_scale_F: bool = True` ("rescales hidden_dim such that number of
  activated params is same as equivalent dense layer": `hidden_dim / (capacity_factor + 1)`), `interleave_moe_layer_step: int = 1`
  (`model.py` 265: MoE if `(layer_id + 1) % interleave_moe_layer_step == 0`). Card: Scout 17B active/109B total, 16 experts, 10M context;
  Maverick 17B/400B, 128 experts, 1M; Scout fits one H100 with on-the-fly int4; Maverick FP8 fits one H100 DGX host (`MODEL_CARD.md` line 312).
- **Kimi K2** (`kimi-k2/README.md` table; `tech_report.pdf` §2.3 Table 2): 1T total / 32B activated; 61 layers incl. 1 dense; attention hidden 7168;
  MoE hidden per expert 2048; 64 heads; 384 experts; 8 selected; 1 shared; vocab 160K; 128K context; MLA; SwiGLU; vs DeepSeek-V3: 256→384 experts,
  128→64 heads, 3→1 dense layers, "Expert Grouping: Yes / No" (K2 drops group-limited routing).

## 4. Load balance, capacity and dropping — as code

- **HF Switch aux loss** `load_balancing_loss_func(gate_logits, num_experts, top_k=2, attention_mask=None)` (`modeling_mixtral.py` 485–552), core:
  ```python
  routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)
  _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
  tokens_per_expert_sum += torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
  router_prob_sum += routing_weights.float().sum(dim=0); total_rows += routing_weights.shape[0]
  tokens_per_expert = tokens_per_expert_sum / total_rows; router_prob_per_expert = router_prob_sum / total_rows
  overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
  return overall_loss * num_experts
  ```
  Summed over all layers' rows (a layer average). f_e counts **top-k assignments per row**, so Σ f_e = k: perfectly uniform routing gives
  E · Σ (k/E)(1/E) = **k** (2 for Mixtral), not 1. Only P carries gradient (f comes from `topk`). Added as `loss += self.router_aux_loss_coef * aux_loss`
  only when `output_router_logits` and `labels` are set. (The function carries a stray `@use_kernel_forward_from_hub("RMSNorm")` decorator in this snapshot.)
- **Megatron-LM** `switch_load_balancing_loss_func` (`$SP/moe-research/moe_utils.py`): docstring `loss = E * Σ_i (f_i * P_i)`, `f_i = 1/(T*topk) Σ routing_map(x,i)`,
  `P_i = 1/T Σ probs(x,i)`; code `aux_loss = sum(probs.sum(0) * tokens_per_expert) * (num_experts * moe_aux_loss_coeff / (topk * T * T))` → balanced value = coeff × 1.
  `moe_router_load_balancing_type`: `"aux_loss"` (micro-batch, GShard/Switch), `"seq_aux_loss"` (DeepSeek-V2/V3, per sample: reshapes so each sequence
  is scored separately, then divides by `bsz`), `"global_aux_loss"`, `"none"`. `moe_aux_loss_coeff` default 0.0 ("A starting value of 1e-2 is recommended").
- **MegaBlocks** `batched_load_balancing_loss` (`megablocks/layers/moe.py`): `scale = moe_num_experts * moe_loss_weight / (num_layers * tokens * moe_top_k)`;
  `scale * dot(tokens_per_expert, mean expert_scores)` → balanced value = `moe_loss_weight` (normalised by k and layers). `moe_loss_weight: float = 0.1` default.
- **Router z-loss**: MegaBlocks `batched_router_zloss` = `moe_zloss_weight * stack([logsumexp(logits, dim=1).square().mean() for each router])`
  (`megablocks/layers/router.py` 25–42); `moe_zloss_weight: float = 0  # 1e-3 is a reasonable value`. Megatron `z_loss_func`: `mean(logsumexp(logits, -1)**2) * z_loss_coeff`,
  `moe_z_loss_coeff` "1e-3 would be a good start". OLMoE trained with 0.001. Purpose: keeps router logits small (numerics in bf16) — ST-MoE (unverified: paper not in ref).
- **Aux-loss-free bias** (DeepSeek-V3): Megatron `get_updated_expert_bias` (`moe_utils.py` 1217–1253): all-reduce `tokens_per_expert` over TP×DP×CP, then
  `update_direction = torch.sign(total_tokens - tokens_per_expert * num_experts)`; `expert_bias + update_direction * expert_bias_update_rate` — underloaded
  experts' bias rises, overloaded falls, by a fixed step (sign, not magnitude). `moe_router_bias_update_rate: float = 1e-3` ("same as that used in DeepSeekV3");
  `moe_router_enable_expert_bias`; reference arXiv 2408.15664. The bias enters only the top-k choice (§3 code). DeepSeek-V3 also keeps a small
  sequence-wise balance loss (α = 0.0001) and sets the update rate to 0 for the last 500B tokens (unverified: DeepSeek-V3 tech report arXiv 2412.19437 is not in ref;
  the README only says "auxiliary-loss-free strategy for load balancing").
- **Group-limited routing** (Megatron): `moe_router_num_groups`, `moe_router_group_topk`; groups scored by the sum of the top-(`moe_router_topk`/`moe_router_group_topk`)
  scores (8/4 = 2 for DeepSeek-V3 — matches `topk(2)` in `Gate`). `moe_router_score_function`: `'softmax'`, `'sigmoid'`, `'sqrtsoftplus'`;
  `moe_router_pre_softmax` (default False = softmax after top-k); `moe_router_topk_scaling_factor` (= DeepSeek's `route_scale`).
- **Capacity factor**: MegaBlocks `expert_capacity(tokens) = int(moe_capacity_factor * top_k * tokens * world_size / num_experts)`, `moe_capacity_factor: int = 1`;
  capacity 0 → `torch.max(tokens_per_expert)` (no dropping); tokens beyond capacity are dropped (pass through the residual only). Megatron:
  `moe_expert_capacity_factor=None` (no dropping), `moe_token_drop_policy` `'probs'` (drop lowest-probability) or `'position'` (drop the tail),
  `moe_pad_expert_input_to_capacity`. **Dropless** = MegaBlocks dMoE (`megablocks/layers/dmoe.py` `dMoE`, `ParallelDroplessMLP`): block-sparse
  reformulation, "removing the `capacity_factor` hyperparameter altogether"; README: up to 40% faster than Tutel's best `capacity_factor`, up to 2.4× vs dense Megatron.
  Inference engines are dropless (vLLM `fused_moe` pads each expert's rows to `BLOCK_SIZE_M`, §7; unverified that no engine drops).
- **Expert-choice routing** (each expert picks its top-C tokens; perfectly balanced, a token may get 0 or many experts; not causal-safe for decode):
  no implementation in ref; `olmoe/README.md` line 107 says their expert-choice runs used `Muennighoff/megablocks` and "neither was better than dropless
  token choice in our experiments". Formulation from Zhou et al. 2022 (unverified: paper not in ref).
- **Sparsity scaling** (Kimi K2 report §2.3): sparsity = total experts / activated; at fixed activated params (8 active + 1 shared), more experts lowers loss;
  "sparsity 48 reduces FLOPs by 1.69, 1.39, and 1.15 compared to sparsity levels 8, 16, and 32" at equal validation loss 1.5; K2 uses 48 (384/8).
- **Upcycling**: OLMoE README "Sparse upcycling": train dense, convert with `scripts/sparsify_ckpt_unsharded.py`, e.g. OLMo-1B → 8-expert MoE.

## 5. Which experts a batch touches (numbers for `moecore.touched`)

- Closed form (uniform, the repo's): E(1 − (1 − k/E)^T). DeepSeek-V3 (256, 8) per layer: T=1 → 8.0, 8 → 57.4, 32 → 163.3, 128 → 251.6, 256 → 255.9.
- "Exact" variant: a token picks k *distinct* experts, so P(expert e untouched by one token) = 1 − k/E exactly — the closed form is already exact for
  independent tokens with uniform-without-replacement per token (derivation, not a source). Zipf/skewed routing has no source here: it is the builder's
  model (label "simulated"); skew lowers experts touched at a given T and raises the hottest expert's load.
- Consequences already in the repo: weight bytes approach total as T grows; crossover ≈ ridge × (streamed/multiplied) ratio (§1). Mixtral at batch 64, 1K context
  on H200: KV 64·1025·131,072 = 8.6 GB vs 93.1 GB of weights streamed — the KV/weights ratio is lower than a dense model with the same active size.

## 6. Expert-parallel communication and DeepEP

- Bytes per GPU per direction (upper bound, every assignment remote): tokens × k × hidden × bytes. With uniform routing over EP ranks a fraction
  (EP−1)/EP leaves the GPU; DeepEP V1 "normal" kernels forward NVLink→RDMA per node ("asymmetric-domain bandwidth forwarding", `deepep/docs/legacy.md` line 9).
- DeepEP V1 (`deepep/docs/legacy.md`, H800, CX7 400 Gb/s ≈ 50 GB/s, NVLink ≈ 160 GB/s):
  - normal kernels, 4096 tokens, hidden 7168, top-4 groups, top-8, FP8 dispatch, BF16 combine: intranode EP8 dispatch **153 GB/s**, combine **158 GB/s** (NVLink);
    internode EP16 43/43, EP32 58/57, EP64 51/50 GB/s (RDMA). Support SM-count control; not CUDA-graph compatible (CPU waits for GPU counts).
  - low-latency kernels, pure RDMA, 128 tokens, hidden 7168, top-8, FP8 dispatch, BF16 combine — dispatch / combine latency: EP8 **77 / 114 µs**
    (98 / 127 GB/s), EP16 118 / 195, EP32 155 / 273, EP64 173 / 314, EP128 192 / 369, EP256 194 / 360 µs. CUDA-graph compatible; `return_recv_hook=True`
    gives hook-based overlap with no SMs.
  - **Check the formula on it**: dispatch = 128 · 8 · (7168 B FP8 + 7168/128 · 4 B scales) = 7,569,408 B; / 77 µs = **98.3 GB/s** (reported 98).
    Combine = 128 · 8 · 7168 · 2 = 14,680,064 B; / 114 µs = 128.8 GB/s (reported 127).
- DeepEP V2 (`deepep/README.md`, current): single `ElasticBuffer` API for HT and LL; NCCL Gin backend (NCCL ≥ 2.30.4), JIT kernels, up to EP2048;
  8K tokens, hidden 7168, top-8, FP8 dispatch/BF16 combine: SM90 CX7 EP 8×2 **90/81 GB/s** (RDMA, 12 SMs), EP 8×4 61/61 (6 SMs); SM100 EP8 NVLink
  726/740 GB/s (64 SMs), 643/675 (24 SMs). "up to 1.3x peak performance, while saving up to 4x SM count" vs V1; "0 SM RDMA low-latency EP is no longer
  supported". Requirements: SM90 (or SM90 PTX), CUDA ≥ 12.3, PyTorch ≥ 2.10, NVLink intranode, RDMA internode. V1 also ran on SM80. **Not usable on T4/L4.**
- EPLB redundancy memory (`vllm/docs/serving/expert_parallel_deployment.md`): `NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`;
  "approximately 2.4 GB for one redundant expert per EP rank" for DeepSeek-V3 = 58 × 44,040,192 B (FP8) = 2.55e9 B = **2.38 GiB**.

## 7. vLLM MoE serving (verified on v0.30.0 unless marked main-only)

- Flags (`vllm/engine/arg_utils.py` v0.30.0): `--enable-expert-parallel` / `-ep`; `--data-parallel-size` / `-dp`; `--data-parallel-size-local`;
  `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-start-rank`, `--headless`, `--api-server-count` (EP doc examples);
  `--all2all-backend`; `--enable-dbo` (dual batch overlap); `--enable-eplb`; `--eplb-config` (JSON or `--eplb-config.window_size 1000` style);
  `--expert-placement-strategy` (`"linear"` default | `"round_robin"`); `--enable-elastic-ep`; `--cpu-offload-gb`; `--cpu-offload-params`;
  `--offload-backend` (`"auto"` | `"uva"` | `"prefetch"`), `--offload-group-size`, `--offload-num-in-group`, `--offload-prefetch-step`, `--offload-params`;
  `--enable-return-routed-experts`.
- `ParallelConfig` (`vllm/config/parallel.py`): `enable_expert_parallel: bool = False` ("Use expert parallelism instead of tensor parallelism for MoE layers");
  `all2all_backend: All2AllBackend = "allgather_reducescatter"`; v0.30.0 `All2AllBackend` = `"naive"`, `"pplx"`, `"deepep_high_throughput"`,
  `"deepep_low_latency"`, `"deepep_v2"`, `"mori_high_throughput"`, `"mori_low_latency"`, `"nixl_ep"`, `"allgather_reducescatter"`,
  `"flashinfer_all2allv"` (alias of `"flashinfer_nvlink_two_sided"`), `"flashinfer_nvlink_two_sided"`, `"flashinfer_nvlink_one_sided"` (main adds `"moonep"`).
  **`"pplx"` and `"naive"` are removed**: they log "has been removed" and fall back to `allgather_reducescatter`. **There is no `VLLM_ALL2ALL_BACKEND`
  environment variable** in v0.30.0 or main — older guides that use it are stale.
- EP semantics (`docs/serving/expert_parallel_deployment.md`): `EP_SIZE = TP_SIZE × DP_SIZE`; expert layers sharded across all EP ranks; attention
  replicated across DP ranks when `TP = 1`, TP-sharded within each DP group when `TP > 1`; **without** `--enable-expert-parallel` MoE layers use
  tensor parallelism over a group of size `TP × DP`. Example: `vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --data-parallel-size 8
  --enable-expert-parallel` (one 8×H200/H20 node). Multi-node: `--all2all-backend deepep_low_latency` (decode) / `deepep_high_throughput` (prefill),
  `--data-parallel-size 16 --data-parallel-size-local 8`, second node `--data-parallel-start-rank 8 --headless`; IB clusters `export GLOO_SOCKET_IFNAME=eth0`.
  DeepEP kernels "may show poor performance for mixed workloads" (they target P/D-disaggregated serving). Prereqs: DeepEP, DeepGEMM, gdrcopy for disagg;
  `deepep_v2` needs NCCL ≥ 2.30.4.
- DP + MoE (`docs/serving/data_parallel_deployment.md`): DP ranks are not independent for MoE — forward passes are aligned; idle ranks run **dummy forward
  passes** while any rank has work (a DP Coordinator process decides). External DP CLI options are only supported for MoE deployments.
- EPLB (`EPLBConfig`, v0.30.0 defaults): `window_size=1000`, `step_interval=3000`, `num_redundant_experts=0`, `log_balancedness=False`
  (balancedness = avg tokens per expert ÷ max tokens per expert), `use_async=True`, `policy="default"`, `communicator=None`
  (`"torch_nccl"`, `"torch_gloo"`, `"torch_xccl"`, `"nixl"`, `"pynccl"`). Experts per rank: `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS`, or
  `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`. Doc recommends `num_redundant_experts` 32 at large scale.
  Benchmark balanced routing: `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` (also `normal_routing`; `router/routing_simulator_router.py`
  `RoutingSimulator`) and `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`.
- Fused MoE kernel (`vllm/model_executor/layers/fused_moe/`): `moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, ...)` flattens the
  T·k assignments, **sorts them by expert and pads each expert's segment to a multiple of `block_size`** (docstring example: 12 assignments, 4 experts,
  block 4 → 16 slots with pad id 12); with EP, non-local experts get id −1 and their blocks are skipped. `fused_moe_kernel` (Triton, `fused_moe.py` 298)
  takes `sorted_token_ids_ptr`, `expert_ids_ptr`, `num_tokens_post_padded_ptr`, `topk_weights_ptr`; `fused_experts_impl` runs GEMM 1 on `w1`
  (gate+up), `apply_moe_activation`, GEMM 2 on `w2`, then `ops.moe_sum` over the k copies. Unquantized CUDA backends in priority order:
  `FLASHINFER_TRTLLM`, `FLASHINFER_CUTLASS`, `TRITON`, `BATCHED_TRITON` (on SM90 FlashInfer is moved behind Triton) (`oracle/unquantized.py`).
- Tuned configs: `configs/E={E},N={N},device_name={torch.cuda.get_device_name() with _},dtype={...},block_shape=[..].json` (`get_config_file_name`;
  any H200-family name maps to `NVIDIA_H200`); N = `w2_shape[2]` = expert intermediate size **per shard** (×2 for `int4_w4a16`); JSON maps batch size M →
  `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`; the nearest M is used. `VLLM_TUNED_CONFIG_FOLDER` is searched first. No file →
  warning "Using default MoE config. Performance might be sub-optimal!" and `get_default_config(M, E, N, K, topk, dtype, block_shape)`. At main `a4eb3f2`:
  335 files; **none for T4, L4 or A10**; e.g. `E=128,N=768,device_name=NVIDIA_H200.json` (Qwen3-30B-A3B at TP1), `E=8,N=14336,...H200.json` (Mixtral TP1),
  `E=8,N=7168` (Mixtral TP2), FP8 block `...,dtype=fp8_w8a8,block_shape=[128,128].json`. README: generate with `benchmarks/kernels/benchmark_moe.py`.
- Weight offload (`vllm/config/offload.py`, v0.30.0): `cpu_offload_gb: float = 0` — "The space in GiB to offload to CPU, per GPU … one 24 GB GPU and set this to 10,
  virtually … a 34 GB GPU … requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model
  forward pass. This uses UVA". `cpu_offload_params: set[str]` — exact name segments, e.g. `"experts"` matches `mlp.experts.w2_weight` ("expert" or "w2"
  do not) → **offload only expert weights**. Prefetch backend: `offload_group_size` (every N layers), `offload_num_in_group`, `offload_prefetch_step`.
  Every offloaded byte crosses PCIe **every step** (not just touched experts) — at ~25 GB/s (PCIe Gen4 x16 effective, unverified) 4 GiB adds ~170 ms per step.
- Router traces from vLLM: `--enable-return-routed-experts` (v0.30.0) → `CompletionOutput.routed_experts: np.ndarray` shape `[seq_len, layer_num, topk]`;
  OpenAI chat response choice field `routed_experts` = base64 `.npy`, shape `(num_tokens - 1, num_layers, num_experts_per_tok)`, decode with
  `np.load(io.BytesIO(base64.b64decode(s)))`; request field `routed_experts_prompt_start` skips prompt tokens. `None` if the flag is off.
- Quantized MoE: `Mxfp4Config.get_min_capability()` returns **80** and `get_supported_act_dtypes()` = `[torch.bfloat16]` → gpt-oss MXFP4 **cannot run on a
  T4** (sm75, no bf16); fine on L4/4090 (sm89). `Mxfp4MoeBackend` includes `MARLIN`, `BATCHED_MARLIN`, `TRITON`, `TRITON_UNFUSED`, `AITER_MXFP4_BF16`,
  `CPU`, `EMULATION`. `docs/features/quantization/README.md`: "Turing does not support Marlin MXFP4"; GPTQ/AWQ on Turing ✅; llm-compressor FP8 W8A8 needs
  Ada/Hopper. Min capability: `fp8` 75, `moe_wna16` 70, `experts_int8` 80, `awq` 75, `gptq` 60. DeepSeek-V3 FP8 = `"quant_method": "fp8"`, `"fmt": "e4m3"`,
  `"weight_block_size": [128, 128]` (`README_WEIGHTS.md`); activations quantised per 128-channel block (`inference/kernel.py` `act_quant(x, block_size=128)`).
- MoE architectures in `docs/models/supported_models.md`: `MixtralForCausalLM`, `Qwen2MoeForCausalLM` (`Qwen/Qwen1.5-MoE-A2.7B`, `-Chat`),
  `Qwen3MoeForCausalLM` (`Qwen/Qwen3-30B-A3B`), `OlmoeForCausalLM` (`allenai/OLMoE-1B-7B-0924`, `-Instruct`), `GraniteMoeForCausalLM`
  (`ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`), `GptOssForCausalLM`
  (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), `PhiMoEForCausalLM`, `Qwen3NextForCausalLM`, DeepSeek, Llama 4 and more.
- Kimi K2 (`kimi-k2/docs/deploy_guidance.md`): smallest FP8 unit at 128k = **16 H200/H20** (TP16, or DP16+EP), vLLM `--data-parallel-size 16
  --data-parallel-size-local 8 --enable-expert-parallel`; SGLang 4P12D example uses `--enable-deepep-moe`, `--deepep-mode low_latency`, `--ep-num-redundant-experts 96`.

## 8. SGLang for comparison (`sglang/docs/docs/advanced_features/`)

- `expert_parallelism.mdx`: `--moe-a2a-backend` `none` (default: all-reduce/all-gather dispatch; the only one for `ep_size < tp_size`), `deepep`, `mooncake`,
  `flashinfer`, `ascend_fuseep`, `pplx`; `--moe-runner-backend` `auto` (default), `triton`, `deep_gemm`, `cutlass`, `flashinfer_trtllm`, `flashinfer_cutlass`,
  `flashinfer_cutedsl`; `--deepep-mode` `auto` | `normal` (prefill) | `low_latency` (decode, CUDA graphs); DeepEP etc. require `ep_size = tp_size`;
  `--enable-eplb`, `--ep-num-redundant-experts`, `--ep-dispatch-algorithm`; example `--moe-a2a-backend deepep --moe-runner-backend deep_gemm --tp 8 --ep 8`.
  EP size flag: `--expert-parallel-size` / `--ep-size` / `--ep` (`server_arguments.mdx`). Two-batch overlap (TBO) and single-batch overlap (SBO).
- `dp_dpa_smg_guide.mdx`: DP attention `--enable-dp-attention` + `--dp-size` (must be > 1 or DPA is disabled); with MLA, TP duplicates the single
  latent KV head on every GPU — DPA gives each replica its own KV; recommended DeepSeek: `--tp 8 --dp-size 8 --ep 8 --enable-dp-attention --moe-a2a-backend deepep
  --moe-runner-backend deep_gemm`. SGLang's `--tp` is the total GPU count with DPA (vLLM's `--tensor-parallel-size` is per DP rank) — different semantics.
- Older flag `--enable-deepep-moe` (in Kimi's guide) is superseded by `--moe-a2a-backend deepep`.

## 9. Small GPUs: what fits (weights only; T4 16 GB ≈ 15 GiB usable; L4/4090 24 GB)

| Model id | params | fp16/bf16 | 4-bit est. | T4 (fp16) | 24 GB | 2×T4 EP/TP=2 |
|---|---|---|---|---|---|---|
| `ibm-granite/granite-3.0-1b-a400m-instruct` (verify id; `-base` in vLLM docs) | 1.33B | 2.7 GB | – | yes | yes | trivial |
| `ibm-granite/granite-3.0-3b-a800m-instruct` | 3.30B | 6.6 GB | – | yes | yes | yes (3.3 GB/GPU) |
| `allenai/OLMoE-1B-7B-0924(-Instruct)` | 6.92B | 13.8 GB | ~4 GB GGUF q4_0 (verify) | **no KV room** → `--cpu-offload-gb` | yes | **yes (6.9 GB/GPU)** |
| `Qwen/Qwen1.5-MoE-A2.7B(-Chat)` | 14.32B | 28.6 GB | ~8.5 GB (GPTQ-Int4, verify id) | INT4 only | INT4, or bf16 + offload | INT4 |
| `Qwen/Qwen3-30B-A3B` | 30.53B | 61.1 GB | ~17.1 GB (`Qwen/Qwen3-30B-A3B-GPTQ-Int4`) | no | INT4 (≈3 GB KV left) | no |
| `Qwen/Qwen3-30B-A3B-FP8` (xpu.md lists FP8 2507 variants) | 30.53B | – | FP8 ≈ 31.2 GB | no | no (2×L4 EP=2: ~15.6 GB/GPU) | no (no FP8 on sm75) |
| `openai/gpt-oss-20b` | 20.91B | – | MXFP4 ≈ 13.8 GB | **no (sm80+, bf16)** | yes | no |

4-bit estimate = (total − embeddings) × 4.25/8 + embeddings in fp16 (`$SP/moe-research/params.py` arithmetic; GPTQ g128 ≈ 4.16–4.25 bpw).
llm-compressor MoE recipe (`llm-compressor/examples/quantizing_moe/qwen_example.py`): `MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"`,
`ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]` ("the MoE gate layers are sensitive to quantization"); `load_context()` loads a
"calibration-friendly MoE definition to ensure all experts are properly calibrated" (README). Granite: `ignore=["lm_head", "re:.*.block_sparse_moe.router.layer"]`.
llama.cpp (`common/arg.cpp`): `-cmoe` / `--cpu-moe` "keep all Mixture of Experts (MoE) weights in the CPU"; `-ncmoe N` / `--n-cpu-moe N` "…of the first N layers";
`-ot` / `--override-tensor "<pattern>=<buffer type>"`; experts matched by `LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"`.
Unlike vLLM UVA offload, llama.cpp computes offloaded experts **on the CPU** (only activations cross PCIe) (unverified: inferred from buffer-type override semantics).

## 10. Model and tool ids to use

- T0 (no download): numpy toy MoE; configs above as literal dicts in `moecore.sizing.MODELS` (mark `verify` + date 2026-09-26); roofline numbers from §1.
- T0 with torch (CPU, 2.14 here; `torch.nn.functional.grouped_mm` and `torch._grouped_mm` both exist): tiny MoE in `moelab.tinymoe`; do not import transformers
  (not installed; never install models).
- T1 free T4: `ibm-granite/granite-3.0-3b-a800m-instruct` or `...-1b-a400m-...` (fp16, fits); `allenai/OLMoE-1B-7B-0924-Instruct` with
  `--dtype half --cpu-offload-gb 6 --cpu-offload-params experts` (verify it starts; Colab RAM ~12.7 GB, verify); router traces via HF `output_router_logits=True`
  (OLMoE/Qwen2-MoE/Qwen3-MoE/Mixtral/gpt-oss have `OutputRecorder(..TopKRouter, index=0)`) or forward hooks on the `*TopKRouter` modules (Granite: `block_sparse_moe.router`,
  returns indices first), or vLLM `--enable-return-routed-experts`.
- T1 24 GB (L4 / 4090): `Qwen/Qwen3-30B-A3B-GPTQ-Int4`; `openai/gpt-oss-20b` (MXFP4, bf16); `Qwen/Qwen1.5-MoE-A2.7B-Chat` bf16 + offload or its INT4.
- T2 Kaggle 2×T4 (PCIe): `vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --tensor-parallel-size 2 --enable-expert-parallel` (EP=2, attention TP=2)
  vs without `--enable-expert-parallel` (MoE as TP=2); or `--data-parallel-size 2 --enable-expert-parallel` (DP attention). Backend = default `allgather_reducescatter`.
- T3 GCP: 02 lab `l4x2` pool (2×L4, PCIe): Qwen1.5-MoE-A2.7B bf16 (14.3 GB/GPU) or `Qwen/Qwen3-30B-A3B-FP8` (verify id) with EP=2. No new Terraform.
- Tools: vLLM v0.30.0 (`pip install "vllm==0.30.0"`), DeepEP (SM90+, not for T4/L4), llama.cpp `--cpu-moe`/`--n-cpu-moe`, MegaBlocks (training reference), Megatron-LM (router math reference).

## 11. Pitfalls (a builder would otherwise get these wrong)

1. Two Switch-loss normalisations: HF's `load_balancing_loss_func` is minimised at **k** (Σ f = k), MegaBlocks/Megatron at **1** (divide by k). State which.
2. `norm_topk_prob` defaults to `False` in HF Qwen2/Qwen3/OLMoE configs, Mixtral always renormalises, DeepSeek renormalises sigmoid weights then × 2.5,
   gpt-oss/Granite take softmax over the top-k **logits** (renormalised by construction), Llama 4 uses sigmoid of the top-1 score and scales the expert **input**.
3. DeepSeek's bias chooses experts but never weights them; the group score is the **sum of the top-2** biased scores per group (amax only when no bias).
4. "Active parameters" differ by embedding accounting: gpt-oss's 5.1B/3.6B include the LM head only; DeepSeek's 37B and `roofline.llm.active_params()` include both.
   Compute all three and name the convention; the roofline crossover test uses total/active **without** the input embedding.
5. `experts_touched` is per layer; bytes streamed need × layers + attention + LM head, and KV. Do not re-derive with a different formula: reproduce 25.6 GB / 5.34 ms / 754 / 2,055.
6. EP all-to-all bytes = tokens × k × hidden × bytes is per direction and an upper bound; FP8 dispatch adds 4 B per 128 channels of scales; combine is usually BF16 (2×).
7. vLLM: `--all2all-backend` is a CLI flag (no `VLLM_ALL2ALL_BACKEND` env); `pplx`/`naive` are removed; default is `allgather_reducescatter`; EP size is not a flag
   (= TP × DP); without `--enable-expert-parallel` experts are tensor-parallel; with DP, idle ranks still run dummy steps.
8. DeepEP needs SM90 (V2) / SM80+ (V1), NVLink + RDMA — never on T4/L4/Kaggle; there, EP=2 runs over NCCL on PCIe with the default backend.
9. No tuned `fused_moe` JSON for T4/L4/A10: expect the "Using default MoE config" warning; tuning files are per (E, N per shard, device, dtype, block).
10. gpt-oss in vLLM needs sm80+ and bf16 activations (`Mxfp4Config.get_min_capability() == 80`): not on a T4. T4 has no bf16 (`--dtype half`) and no FP8.
11. `--cpu-offload-gb` streams offloaded weights over PCIe every forward pass, independent of routing; it trades memory for step time, per GPU.
    Use `--cpu-offload-params experts` to offload experts only (v0.30.0).
12. transformers v5 stores experts as fused 3D tensors (`gate_up_proj` [E, 2I, d]); there are no per-expert `nn.Linear` modules to hook — hook the router.
    Module names changed (`block_sparse_moe` → `mlp` for Mixtral); Granite has no router-logit recorder. `output_router_logits` returns pre-softmax logits.
13. The DeepSeek and Llama 4 reference implementations are not how engines run MoE (all-reduce instead of all-to-all; dense masked compute).
14. Kimi K2 / 2026 models: only README-level numbers are sourced; do not invent configs. Kimi K2 has no expert grouping (unlike DeepSeek-V3).
15. EPLB redundant experts cost HBM: ~2.4 GiB per redundant expert per rank for DeepSeek-V3 FP8.
16. Qwen3 configs: vocab 151,936 in config vs 151,669 tokenizer entries (Qwen3 report) — use the config value for parameter counts.
17. DeepSeek-V3's HF checkpoint is 685B (671B + 14B MTP); size memory from what is loaded (vLLM loads MTP only with speculative decoding, unverified).
18. Kaggle P100 is not supported by vLLM; pick "GPU T4 x2" (repo any-gpu README).

## 12. Unverified (best knowledge; say "(verify)" in material)

- Qwen3-235B-A22B `hidden_size` 4096, `moe_intermediate_size` 1536; real Qwen3 MoE configs set `norm_topk_prob: true` — huggingface.co blocked; numbers reproduce 235B/22B.
- Llama 4 Maverick `interleave_moe_layer_step: 2`, `intermediate_size_mlp: 16384` for its dense layers; vision encoder share of Scout's 109B — config not in ref.
- gpt-oss-20b 24 layers (derived); OLMoE expert `intermediate_size` 1024 (derived from 6.9B/1.3B); granite-3.0 MoE configs (derived; ids from vLLM docs).
- Kimi K2 MLA ranks (assumed = DeepSeek-V3), vocab 163,840; its 1.04T/32.6B come from the report.
- DeepSeek-V3 sequence-wise loss α = 0.0001, bias rate schedule (0.001 → 0 for the last 500B tokens), node limit M = 4 nodes — tech report not in ref
  (the code's `n_limited_groups = 4` with 8 groups is verified).
- Switch Transformer α = 0.01 and capacity factor 1.0–1.25 in the paper; ST-MoE z-loss 1e-3; expert-choice (Zhou et al.) — papers not in ref;
  the MegaBlocks/Megatron/OLMoE values above are verified.
- Model ids: `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`, `ibm-granite/granite-3.0-3b-a800m-instruct` (in vLLM docs), `...-1b-a400m-instruct`
  (`granite-3.1-1b-a400m-instruct` appears in vLLM docs), `Qwen/Qwen3-30B-A3B-FP8` — existence on the Hub not checkable here.
- Whether vLLM's Triton `fused_moe` and `--cpu-offload-gb` work on T4 for these models — no GPU here; the T0 fallback must be labelled simulated.
- PCIe effective bandwidth ~25 GB/s (Gen4 x16) for offload estimates; T4 is PCIe Gen3 x16 (~12 GB/s effective, verify).
- The fused-MoE tuned-config list for v0.30.0 (checked on main `a4eb3f2` only).
