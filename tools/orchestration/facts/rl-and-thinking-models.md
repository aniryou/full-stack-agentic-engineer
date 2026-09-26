# facts-rl-and-thinking-models — verified 2026-09-26 (research agent). Source of truth for `00-foundations/rl-and-thinking-models`.
Paths are relative to `$SP/ref/` unless they start with a repo dir (`00-…`, `04-…`, `06-…`, `07-…`). Anything marked **(unverified)** or
**(verify)** was not found in a source; builders keep the marker in docs. Extends `$SP/FACTS.md` (vLLM metric names, T4/Colab facts are there).

## 0. Versions pinned (what "current" means here)
- vLLM: repo pin **0.30.0** (PyPI latest = 0.30.0, checked today). Clone `vllm/` is main@a4eb3f25 (2026-09-26); the v0.30.0 files used
  below were fetched into `vllm030/` (`reasoning_outputs.md`, `reasoning_init.py`, `chat_protocol.py`, `chat_serving.py`, `sampling_params.py`,
  `parser_qwen3.py`, `config_reasoning.py`, `thinking_budget.py`, `arg_utils.py`, `loggers.py`). v0.30.0 vs main differ only by the
  `granite_thinking_parser` entry (added after 0.30.0) in the reasoning docs/registry (diff checked).
- TRL: PyPI latest **1.14.0**; clone `trl/` is 1.15.0.dev0 (main 2026-09-25). v1.14.0 files fetched into `trl114/` (grpo_config.py,
  grpo_trainer.py, dpo_trainer.py, dpo_config.py, async_grpo_config.py, rewards, utils.py). Defaults below are **v1.14.0**; the GRPO doc is
  identical to main in the lines used. TRL 1.14.0 requires `transformers>=4.56.2`; extra `trl[vllm]` pins **`vllm>=0.20.0,<=0.30.0`** (PyPI
  `requires_dist`) — compatible with the repo's vLLM 0.30.0.
- SGLang docs: `sglang/` main 2026-09-25. verl docs: `verl/` main 2026-09-24. Qwen3 repo `qwen3/` 2026-01-09. DeepSeek-R1 repo 2025-04-09
  (paper `deepseek-r1/DeepSeek_R1.pdf`, text extracted to `$SP/txt/r1.txt`). DAPO repo 2025-05-11 (paper text `$SP/txt/dapo.txt`).
  Qwen3 tech report text: `$SP/txt/qwen3.txt`. gpt-oss 2026-07-24.

## 1. vLLM reasoning outputs (v0.30.0)
- CLI: **`--reasoning-parser <name>`** (arg_utils.py: `"--reasoning-parser"`; stored in `StructuredOutputsConfig.reasoning_parser`,
  default `""` = off, `vllm/config/structured_outputs.py`). Also `--reasoning-parser-plugin <path>`. **`--enable-reasoning` no longer exists**
  (not in arg_utils.py at v0.30.0 or main) — the Qwen docs (`qwen3/docs/source/deployment/vllm.md`, README "vLLM") still show it: stale.
- Registered parser names at v0.30.0 (`vllm030/reasoning_init.py`, `_REASONING_PARSERS_TO_REGISTER`): `deepseek_r1`, `deepseek_v3`, `deepseek_v4`,
  `deepseek_v41`, `poolside_v1`, `cohere_command3`, `cohere_command4`, `ernie45`, `gemma4`, `glm45`, `glm47`, `ling3`, `openai_gptoss`,
  `granite`, `holo2`, `hunyuan_a13b`, `hy_v3`, `hy_v4`, `kimi_k2`, `kimi_k3`, `k2_horizon`, `mimo`, `minimax_m2`, `minimax_m2_append_think`,
  `minimax_m3`, `mistral`, `nemotron_v3`, `olmo3`, `muse_glimmer`, `qwen3`, `seed_oss`, `step3`, `step3p5`, `inkling`
  (+ `granite_thinking_parser` on main only). Names use **underscores** (`deepseek_r1`), unlike SGLang (`deepseek-r1`).
- Docs table (`vllm/docs/features/reasoning_outputs.md`): Qwen3 series → `qwen3`; DeepSeek R1 series (incl. distills) → `deepseek_r1`;
  QwQ-32B → `deepseek_r1`; gpt-oss uses `openai_gptoss` (registry). Structured output with reasoning: `json`, `regex` for these. Quickstart:
  `vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1`.
- **Response field is `reasoning`, not `reasoning_content`.** Docs: "`reasoning` used to be called `reasoning_content`. To migrate, directly
  replace `reasoning_content` with `reasoning` … your client code could silently read an empty `reasoning_content`." `ChatMessage.reasoning:
  str | None` ("vLLM-specific fields that are not in OpenAI spec", `vllm/entrypoints/openai/chat_completion/protocol.py`). Streaming: the
  `reasoning` key appears in `choices[0].delta` chunks; the OpenAI Python client needs `getattr(delta, "reasoning", None)`.
- Requests: incoming assistant messages with the deprecated `reasoning_content` are renamed to `reasoning` (`_normalize_messages_before`,
  chat protocol). When rendering the chat template vLLM puts **both** `reasoning` and `reasoning_content` on the assistant message dict
  ("keep compatibility", `vllm/entrypoints/chat_utils.py` `_parse_chat_message_content`) so templates that read `message.reasoning_content`
  (Qwen3's) see it.
- Usage: with a reasoning parser, `usage.completion_tokens_details.reasoning_tokens` is filled (`_include_reasoning_tokens_details =
  bool(reasoning_parser)`, `vllm030/chat_serving.py` l.159; `CompletionTokenUsageInfo(reasoning_tokens=…)`). `completion_tokens` counts
  reasoning + answer (reasoning tokens are output tokens).
- No reasoning-specific Prometheus metric in v0.30.0 (`vllm030/loggers.py` has no "reason/think" metric except `vllm:num_requests_waiting_by_reason`
  labels `capacity`/`deferred`). Thinking tokens show up in `vllm:generation_tokens`, `vllm:request_generation_tokens`, ITL/TPOT, KV usage.
- `include_reasoning: bool = True` (request field): `false` still generates the reasoning (quality unchanged) but omits it and its
  logprobs/token ids from the response. Endpoints with reasoning: `/v1/chat/completions`, `/v1/messages`, `/v1/responses` only (not
  `/v1/completions`).
- **`chat_template_kwargs` passes through**: `{"enable_thinking": false}` for Qwen3 (docs); server default via
  `--default-chat-template-kwargs '{"enable_thinking": false}'`, request-level kwargs override (docs; `default_chat_template_kwargs` in
  `chat_completion/serving.py` constructor).
- **`reasoning_effort`** (chat protocol): `Literal["none","minimal","low","medium","high","xhigh","max"] | None = None` ("'max' is specific
  to the DeepSeek V4 series"). Code (quote, `build_chat_params`):
  `if self.reasoning_effort is not None and "enable_thinking" not in user_kwargs: extra_kwargs["enable_thinking"] = self.reasoning_effort != "none"`
  and `reasoning_effort=self.reasoning_effort` is also passed to the template. So for **Qwen3, any effort other than `"none"` only means
  enable_thinking=True — it does not shorten thinking** (Qwen3's template has no effort variable; unknown kwargs are filtered by
  `resolve_chat_template_kwargs`). Explicit `enable_thinking` in `chat_template_kwargs` wins. Responses API: `reasoning.effort`, same rule.
- gpt-oss (Harmony): `REASONING_EFFORT = {"high": …HIGH, "medium": …MEDIUM, "low": …LOW}`; any other value raises `VLLMValidationError`
  "not supported by Harmony" (`vllm/entrypoints/openai/parser/harmony_utils.py` l.68, 123–132).
- **Thinking budget (vLLM-native)**: request field `thinking_token_budget: int | None` (chat and completions protocols; `SamplingParams.
  thinking_token_budget`; validator: non-negative int, **`-1` = unlimited**, `vllm030/sampling_params.py` l.51–76). Server needs
  `--reasoning-parser` and optionally `--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "…</think>"}'`
  (`ReasoningConfig` fields `reasoning_parser`, `reasoning_start_str`, `reasoning_end_str`; token ids derived from the parser if unset,
  `vllm030/config_reasoning.py`). "Once the reasoning token count reaches the configured `thinking_token_budget`, vLLM forces the model to
  produce `reasoning_end_str`." Implemented in the sampler for both runners (`vllm/v1/worker/gpu/sample/thinking_budget.py:
  ThinkingBudgetState` for MRV2; `vllm/v1/sample/thinking_budget_state.py` used by MRV1 `gpu_input_batch.py`). Docs example model:
  `Qwen/Qwen3-0.6B`, `"thinking_token_budget": 10`.
- Structured output + reasoning: grammar applies after the reasoning end by default; `--structured-outputs-config.enable_in_reasoning=True`
  applies it inside reasoning (`StructuredOutputsConfig.enable_in_reasoning: bool = False`; docs/features/structured_outputs.md §Reasoning Outputs).
- Tool calls are parsed only from `content`, never from `reasoning` (docs §Tool Calling).
- Qwen3 parser (v0.30.0, `vllm030/parser_qwen3.py`: `Qwen3Parser`, adapter `Qwen3ParserReasoningAdapter`): reads
  `chat_template_kwargs.get("enable_thinking", True)`; if False, `extract_reasoning` returns `(None, model_output)`; starts in REASONING state
  when thinking (so output without an opening `<think>` is still parsed); `<tool_call>` is an implicit reasoning end; absorbs a duplicate `</think>`.
- DeepSeek-R1 parser (`vllm/reasoning/deepseek_r1_reasoning_parser.py: DeepSeekR1ReasoningParser`, start `<think>`, end `</think>`):
  when the start token never appears, everything before `</think>` is reasoning (R1 templates put `<think>\n` in the prompt).
- vLLM's example R1 template strips earlier thinking: `{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}`
  (`vllm/examples/tool_chat_template_deepseekr1.jinja` l.41–42).

## 2. Qwen3 (source: `qwen3/README.md`, `qwen3/docs/source/…`, tech report `qwen3/Qwen3_Technical_Report.pdf`)
- Dense ids: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` (also 14B, 32B; MoE 30B-A3B, 235B-A22B). Hybrid
  (thinking + non-thinking) = the original "Qwen3-2504" release. 2507 split: `…-Instruct-2507` = non-thinking only, `…-Thinking-2507` =
  thinking only (e.g. `Qwen/Qwen3-4B-Thinking-2507`, `Qwen/Qwen3-4B-Instruct-2507`; README News 2025-08-06). Thinking-2507: template
  auto-inserts `<think>`, so output contains only `</think>`; README serves it with the `deepseek_r1` parser (vLLM) / `deepseek-r1` (SGLang).
- Architecture (tech report Table 1): layers / heads Q/KV / tie / context: 0.6B 28, 16/8, yes, 32K; 1.7B 28, 16/8, yes, 32K; 4B 36, 32/8, yes,
  128K; 8B 36, 32/8, no, 128K. head_dim 128 (bundled config). Tokenizer vocab 151,669 (report); config `vocab_size` 151936.
- Param counts via the repo's `servelab.sizing.param_count` (exact llama/Qwen3 formula, incl. q/k-norm; `04-…/vllm-serving-lab/servelab/sizing.py`):
  | model | total params | non-embedding | fp16 weights | KV/token fp16 = 2·L·kv·128·2 |
  |---|---|---|---|---|
  | Qwen3-0.6B (bundled `qwen3-0.6b.json`: h 1024, ffn 3072) | 596,049,920 | 0.440 B | 1.19 GB | 2·28·8·128·2 = 114,688 B (112 KiB) |
  | Qwen3-1.7B (h 2048, ffn 6144 **(verify config)**) | 1,720,574,976 | 1.409 B | 3.44 GB | 114,688 B (112 KiB) |
  | Qwen3-4B (h 2560, ffn 9728 **(verify config)**) | 4,022,468,096 | 3.634 B | 8.04 GB | 2·36·8·128·2 = 147,456 B (144 KiB) |
  | Qwen3-8B (bundled `qwen3-8b.json`: h 4096, ffn 12288, untied) | 8,190,735,360 | 6.946 B | 16.38 GB | 147,456 B (144 KiB) |
  | DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch: h 1536, 28 L, 12/2 heads, ffn 8960, untied, qkv bias **(verify config)**) | 1,777,088,000 | 1.310 B | 3.55 GB | 2·28·2·128·2 = 28,672 B (28 KiB) |
  Teaching number: an 8K-token thinking trace on Qwen3-0.6B holds 8192 × 114,688 B = 0.94 GB of KV — about the size of its weights (1.19 GB).
- T4 (15.0 GiB, fp16) predictions from `servelab.sizing.size(…, "T4", dtype="half", max_model_len=8192)` (overheads are estimates — label
  "predicted"): Qwen3-0.6B 6,969 blocks (×16 = 111,504 tokens) → 13.6 concurrent 8K requests; Qwen3-1.7B 5,721 blocks → 11.2; Qwen3-4B
  2,368 blocks → 4.6 (fits); R1-Distill-1.5B 22,605 blocks → 44.2. Qwen3-8B fp16 (16.4 GB) does **not** fit a T4. Qwen3-4B on L4 at
  `max_model_len=16384`: 5,504 blocks → 5.4.
- Switches (quickstart.md; README "Switching Thinking/Non-thinking Modes"): hard switch `enable_thinking=False` in `apply_chat_template`
  ("strictly prevent the model from generating thinking content"; default True); soft switch `/think` and `/no_think` in user or system
  message, "In multi-turn conversations, the latest instruction is followed." Report Table 9: non-thinking responses keep an **empty
  `<think>\n\n</think>\n\n` block**; `enable_thinking=False` makes the template append that empty block to the generation prompt (same in
  `qwen3/docs/source/assets/qwen3_nonthinking.jinja`: `'<|im_start|>assistant\n<think>\n\n</think>\n\n'`).
- **History handling (drives primer §7)**, `qwen3_nonthinking.jinja` (the Qwen3 template with a forced non-thinking prompt): computes
  `last_query_index` = the last real user message (not a `<tool_response>`); for assistant turns **before** it, renders
  `'<|im_start|>assistant\n' + content` — reasoning (from `message.reasoning_content`, or split off `</think>` in content) is **dropped**;
  for assistant turns after it (the current tool-calling loop) renders `<think>\n…\n</think>\n\n` + content (kept). The stock Qwen3
  template has the same history logic and differs only in the generation prompt (**(verify)** against the HF tokenizer_config of Qwen/Qwen3-0.6B).
  Consequence: turn N generated `<think>…</think>answer` into the KV cache; turn N+1's prompt renders only `answer` for that turn → the
  prefix-cache match ends right after `<|im_start|>assistant\n` of turn N (rounded down to a full 16-token block, 04 PRIMER §5); the
  thinking KV is never reused and the answer is re-prefilled. Within one multi-step tool loop thinking is kept, so the prefix keeps matching.
- gpt-oss template does the same: "CoT is dropped during all previous turns, so we never render it for inference"
  (`transformers/src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py` l.748–749).
- Sampling (quickstart.md l.222–225): thinking **Temperature=0.6, TopP=0.95, TopK=20, MinP=0** ("the default setting in
  `generation_config.json`"; "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions");
  non-thinking **Temperature=0.7, TopP=0.8, TopK=20, MinP=0**. `presence_penalty` 0–2 against repetition (1.5 in the vLLM doc example;
  higher "may occasionally result in language mixing"). Examples use `max_new_tokens=32768`; Instruct-2507 guidance: output length 16,384.
  vLLM applies `generation_config.json` sampling defaults unless the request overrides them (qwen3 docs deployment/vllm.md tip).
- `</think>` token id **151668** (README code comment "rindex finding 151668 (</think>)"); `<think>` = 151667 **(unverified)**.
- Thinking budget (Qwen): report §4.3 — when thinking reaches a user threshold, insert "Considering the limited time by the user, I have to
  give the solution based on the thinking directly now.\n</think>.\n\n"; "this ability is not explicitly trained but emerges naturally" from
  Thinking Mode Fusion. Client recipe (`docs/source/getting_started/thinking_budget.md: ThinkingBudgetClient`): call 1 with
  `max_tokens=thinking_budget`; if `content is None` append that phrase; call 2 via `/v1/completions` with the prompt + `<think>\n…\n</think>\n\n`
  and `continue_final_message=True`, `max_tokens = max_tokens − reasoning_tokens` (it reads `message.reasoning_content` — **use `reasoning`
  on vLLM 0.30**).
- Qwen3 post-training (report §4): cold-start long-CoT SFT → Reasoning RL with **GRPO on 3,995 query-verifier pairs**; "large batch size
  and a high number of rollouts per query, along with off-policy training"; Qwen3-235B-A22B AIME'24 **70.1 → 85.1 over 170 RL steps** →
  Thinking Mode Fusion (SFT) → general RL; small models (0.6B–14B, 30B-A3B) by **strong-to-weak distillation** (off-policy then on-policy
  logit distillation, report l.1558–1569).

## 3. DeepSeek-R1 (README + paper text `$SP/txt/r1.txt`)
- R1-Zero: RL (GRPO) directly on DeepSeek-V3-Base, no SFT. Rewards are **rule-based**: accuracy (boxed answer / compiler + tests) and
  format (`<think>`…`</think>`); "We do not apply the outcome or process neural reward model … may suffer from reward hacking". Template
  (Table 1): "…The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags…".
- R1-Zero AIME 2024 pass@1 **15.6% → 71.0%**, majority vote **86.7%** ("after thousands of RL steps"). Response length grows —
  "hundreds to thousands of reasoning tokens"; reflection emerges; the "aha moment" (Table 3, "Wait, wait. Wait. That's an aha moment").
  Problems: "endless repetition, poor readability, and language mixing".
- R1 pipeline (README "two RL stages … two SFT stages"; paper §2.3): (1) cold start — "thousands of cold-start data" (long CoT) SFT on
  V3-Base; (2) reasoning RL (as R1-Zero) + **language consistency reward** summed with accuracy; (3) rejection sampling from the RL
  checkpoint → **~600k reasoning** samples (keep only correct; some judged by DeepSeek-V3 as generative reward; filter mixed language,
  long paragraphs, code blocks) + **~200k non-reasoning** = **~800k**, SFT V3-Base 2 epochs; (4) RL for all scenarios (rule rewards for
  reasoning, reward models for helpfulness — on the final summary only — and harmlessness — on reasoning + summary).
- GRPO as in the paper eq.(1)–(3): sample G outputs from π_old; objective (1/G) Σ_i [min(ρ_i A_i, clip(ρ_i,1−ε,1+ε) A_i) − β D_KL(π_θ‖π_ref)]
  with D_KL = π_ref/π_θ − log(π_ref/π_θ) − 1 (k3) and A_i = (r_i − mean(r))/std(r). "foregoes the critic model that is typically the same
  size as the policy model". The v1 paper text gives no G or β values; TRL's docstring says R1 uses **β = 0.001** (see §4).
- Models (README §3): R1-Zero and R1 are **671B total / 37B activated**, 128K context, trained from DeepSeek-V3-Base. Distills (SFT only on
  the 800k samples, **no RL stage**): `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (base Qwen2.5-Math-1.5B), `…-Qwen-7B` (Qwen2.5-Math-7B),
  `…-Llama-8B` (Llama-3.1-8B), `…-Qwen-14B` (Qwen2.5-14B), `…-Qwen-32B` (Qwen2.5-32B), `…-Llama-70B` (Llama-3.3-70B-Instruct).
- Distill results (README table): Qwen-1.5B AIME24 pass@1 **28.9**, cons@64 52.7, MATH-500 83.9; Qwen-7B 55.5 / 83.3 / 92.8; Qwen-32B 72.6 / 83.3 / 94.3.
- Distillation vs RL (paper Table 6): RL on Qwen-32B-Base >10K steps (DeepSeek-R1-Zero-Qwen-32B) AIME24 **47.0** vs distilled
  DeepSeek-R1-Distill-Qwen-32B **72.6** — "distilling more powerful models into smaller ones yields excellent results".
- Unsuccessful attempts (paper §4.2): PRM (hard to define steps, annotation, reward hacking) — good for reranking top-N / guided search
  (cites Snell et al., 2024) but not worth it in large-scale RL; MCTS also listed.
- Evaluation setup: max generation **32,768** tokens; temperature **0.6**, top-p **0.95**, **64 responses per query** to estimate pass@1.
- Usage recommendations (README "Usage Recommendations"): temperature 0.5–0.7 (**0.6**); "**Avoid adding a system prompt; all instructions
  should be contained within the user prompt**"; for math add "Please reason step by step, and put your final answer within \boxed{}.";
  the models may bypass thinking (`<think>\n\n</think>`) → "enforce the model to initiate its response with "<think>\n"".

## 4. TRL GRPO (v1.14.0; `trl114/grpo_config.py`, `trl114/grpo_trainer.py`, `trl/docs/source/grpo_trainer.md`)
- `from trl import GRPOConfig, GRPOTrainer`; `GRPOTrainer(model=…, reward_funcs=…, args=GRPOConfig(...), train_dataset=…)`.
- Docs: "At each training step, we sample a batch of prompts and generate a set of G completions for each prompt"; advantage
  `Â_{i,t} = (r_i − mean(r)) / std(r)`; KL = k3 `π_ref/π_θ − log(π_ref/π_θ) − 1` (Schulman 2020); default loss
  `L = −1/Σ|o_i| Σ_i Σ_t [ π_θ/[π_θ]_no-grad · Â_{i,t} − β D_KL ]` (no 1/|o_i|); with `num_iterations` μ>1 the clipped surrogate
  `min(ρÂ, clip(ρ,1−ε,1+ε)Â)` with ρ = π_θ/π_θold. "When μ = 1 (default in TRL), the clipped surrogate objective simplifies to the original".
- Defaults (field defaults, v1.14.0): `num_generations=8` (effective batch must be divisible by it; generation batch must hold whole groups),
  `max_completion_length=512`, `beta=0.0` ("the reference model is not loaded"; R1 used 0.001), `epsilon=0.2`, `epsilon_high=None`
  (→ equals `epsilon`; DAPO recommends **0.28**), `delta=None` (two-sided clip), `num_iterations=1`, `loss_type="dapo"`,
  `scale_rewards="group"`, `importance_sampling_level="token"`, `mask_truncated_completions=False`, `temperature=1.0`, `top_p=1.0`,
  `top_k=0`, `learning_rate=1e-6`, `use_vllm=False`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.3`,
  `vllm_server_host="0.0.0.0"`, `vllm_server_port=8000`, `vllm_importance_sampling_correction=True`,
  `vllm_importance_sampling_mode="sequence_mask"`, `vllm_importance_sampling_clip_max=3.0`, `use_bias_correction_kl=True`,
  `multi_objective_aggregation="sum_then_normalize"`, `entropy_coef=0.0`, `top_entropy_quantile=1.0`. Overrides vs TrainingArguments:
  `logging_steps` 10, `gradient_checkpointing` True, **`bf16` True if `fp16` not set** (on a T4 set `bf16=False, fp16=True`), lr 1e-6.
  There is **no `max_prompt_length`** field in v1.14.0.
- `loss_type` values (docstring): `"grpo"` (per-sequence mean then batch mean — "Not recommended due to length bias"), `"dr_grpo"`
  (global constant = `max_completion_length`), `"dapo"` (default; ÷ active tokens in the global accumulated batch), `"bnpo"` (÷ active tokens
  in the local batch), `"cispo"` (clip IS weights, MiniMax-M1), `"sapo"` (soft gate, τ_pos 1.0 / τ_neg 1.05), `"luspo"` (needs
  `importance_sampling_level="sequence"`), `"vespo"`.
- `scale_rewards`: `True`/`"group"` (std within group), `"batch"` (std over batch), `False`/`"none"` (Dr. GRPO: std scaling causes
  question-level difficulty bias). `importance_sampling_level`: `"token"` or `"sequence"` (GSPO).
- Code (quote, `_compute_advantages`): `advantages = rewards - mean_grouped_rewards` then `if self.scale_rewards != "none": advantages =
  advantages / (std_rewards + 1e-4)`; std is `nanstd` **with Bessel's correction** (`correction = count / (count - 1)`, `trl114/utils.py`).
  `frac_reward_zero_std` is logged (groups all-correct or all-wrong give zero advantage).
- Code (quote, `_compute_loss`): `log_ratio = per_token_logps - old_per_token_logps`; `coef_1 = torch.exp(log_importance_weights)`;
  `per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1`;
  `coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)`; `per_token_loss = -torch.min(coef_1 * advantages,
  coef_2 * advantages)`; `if self.beta != 0.0: per_token_loss = per_token_loss + self.beta * per_token_kl`; aggregation `"grpo"`:
  `((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()`; `"dr_grpo"`: `(per_token_loss * mask).sum() /
  (per_token_loss.size(0) * self.max_completion_length)`; `"dapo"`: `(per_token_loss * mask).sum() / normalizer` (active tokens).
- Reward function contract (docs "Using a custom reward function"): sync or `async def`; keyword args `prompts`, `completions`,
  `completion_ids`, `trainer_state`, `log_extra`, `log_metric`, every dataset column except `prompt` → use `**kwargs`; returns `list[float]`
  (None = skip). Conversational datasets pass message lists. Built-ins (`trl.rewards`): `accuracy_reward(completions, solution, …)` (needs
  `math-verify`), `reasoning_accuracy_reward`, `think_format_reward` (regex `^<think>(?!.*<think>)(.*?)</think>.*$`), `get_cosine_scaled_reward`,
  `get_repetition_penalty_reward`, `get_soft_overlong_punishment(max_completion_len, soft_punish_cache)` (DAPO eq.13; doc example 90 tokens
  with (100, 20) → −0.5).
- vLLM modes (docs): **colocate** (default: vLLM in the trainer process sharing the GPU; `vllm_enable_sleep_mode` offloads during the
  optimizer step) and **server** (separate GPUs): `VLLM_SERVER_DEV_MODE=1 vllm serve <model> --weight-transfer-config '{"backend": "nccl"}'
  --logprobs-mode processed_logprobs --max-logprobs -1` + `GRPOConfig(use_vllm=True, vllm_mode="server")`; "Make sure that the server is
  using different GPUs than the trainer, otherwise you may run into NCCL errors". Train–inference mismatch corrected by truncated/masked
  importance sampling (logs `sampling/sampling_logp_difference/mean`). Alternative: `use_transformers_continuous_batching=True`.
- Quick start: `Qwen/Qwen2.5-0.5B-Instruct` on `trl-lib/DeepMath-103K`, "Distributed across 8 GPUs, the training takes approximately 1 day".
- Async GRPO (`trl/docs/source/async_grpo_trainer.md`, `trl.experimental.async_grpo.AsyncGRPOTrainer`): needs `vllm>=0.22.0`,
  `transformers>=5.2.0`, FSDP2 only; a rollout worker process streams completions from a vLLM server while training consumes them;
  weights pushed via NCCL every `weight_sync_steps` (default **1**); samples older than `max_staleness` (default **4**) weight versions are
  dropped; `max_inflight_tasks` default −1 = auto `max_staleness × per_device_train_batch_size × gradient_accumulation_steps × num_processes`;
  reward fns must be picklable, rollout process has `CUDA_VISIBLE_DEVICES=""`. With LoRA: `--max-loras ≥ max_staleness + 2`
  (ties to 04 vllm-internals §9 `--max-loras`).

## 5. DAPO (paper `$SP/txt/dapo.txt`; README `dapo/README.md`)
- Name: **D**ecoupled Clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization; built on verl; data `DAPO-Math-17k`; model
  `BytedTsinghua-SIA/DAPO-Qwen-32B` (from Qwen2.5-32B base): **50** on AIME 2024 with "50% training steps" of DeepSeek-R1-Zero-Qwen-32B (47).
- Four techniques: (1) **Clip-Higher** — decouple ε_low / ε_high, **ε_low = 0.2, ε_high = 0.28**, against entropy collapse; (2) **Dynamic
  Sampling** — over-sample and drop prompts whose group accuracy is 0 or 1 (zero advantage ⇒ no gradient), keep sampling until the batch
  is full; (3) **Token-level policy-gradient loss** — normalise by Σ|o_i| (long samples no longer under-weighted; stabilises length/entropy);
  (4) **Overlong reward shaping** — first *overlong filtering* (mask truncated samples' loss), then **soft overlong punishment**
  R_length(y) = 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache if L_max − L_cache < |y| ≤ L_max; −1 if |y| > L_max.
  Also: **KL term removed** (§2.3) and rule reward R = 1 if `is_equivalent(ŷ, y)` else −1 (eq. 7).
- Hyper-parameters (§4.1): AdamW lr 1e-6, 20-step warm-up; prompt batch **512**, **16 responses per prompt**, mini-batch 512 → 16 gradient
  updates per rollout step; expected max length 16,384 + 4,096 cache → max generation **20,480**; eval avg@32, temperature 1.0, top-p 0.7.
- Ablation (Table 1, AIME24 avg@32): naive GRPO **30** → +overlong filtering 36 → +clip-higher 38 → +soft overlong punishment 41 →
  +token-level loss 42 → +dynamic sampling (DAPO) **50**.
- TRL mapping: `epsilon_high=0.28`, `loss_type="dapo"`, `mask_truncated_completions=True`, `get_soft_overlong_punishment(...)`, `beta=0.0`;
  dynamic sampling has no GRPOConfig flag (grep of v1.14.0 grpo_config.py finds none; TRL only logs `frac_reward_zero_std`).
- Dr. GRPO ("Understanding R1-Zero-Like Training: A Critical Perspective", arXiv 2503.20783, per TRL docs): GRPO's 1/|o_i| creates a
  response-length bias (short correct / long wrong answers favoured) and std scaling a difficulty bias → constant normaliser + `scale_rewards="none"`.

## 6. verl (docs `verl/docs/…`)
- Rollout engines: `actor_rollout_ref.rollout.name`: `hf`/`vllm`/`sglang` (`docs/examples/config.rst` l.377); `rollout.n` samples per prompt
  ("set it to values > 1 for grpo, rloo"); `rollout.gpu_memory_utilization` (vLLM: fraction of total memory; SGLang: `mem_fraction_static`);
  `rollout.free_cache_engine` (offload KV after rollout, default True). "vllm 0.18.0 and later versions are supported" (`start/install.rst`).
- GRPO in verl (`docs/algo/grpo.md`): `algorithm.adv_estimator=grpo` (default `gae`), `actor.use_kl_loss` (set True for GRPO),
  `actor.kl_loss_coef` default **0.001**, `actor.kl_loss_type`: `kl`(k1), `abs`, `mse`(k2), `low_var_kl`(k3), `full`; `actor.clip_ratio`
  0.2; `actor.loss_agg_mode` default `"token-mean"` (original GRPO = `"seq-mean-token-mean"`); Dr.GRPO = `"seq-mean-token-sum-norm"` +
  `algorithm.norm_adv_by_std_in_grpo=False`. `data.train_batch_size × rollout.n` trajectories per step. PPO: critic + GAE, `algorithm.lam`.
- HybridFlow (`docs/hybrid_flow.rst`): `ActorRolloutRefWorker` colocates actor + rollout "for fast weight transfer using nccl" and actor +
  reference for LoRA. Colocated = rollout and training time-share the same GPUs.
- Rollout dominates: "in DAPO 32B training, the Rollout phase accounts for approximately **70%** of the total time, and increasing resources
  does not reduce the Rollout duration" — long-tail generations leave GPUs idle (`docs/advance/one_step_off.md`). One-step-off-policy trainer
  overlaps generation of step k+1 with training on step k (AReaL cited); fully async trainer (`docs/advance/fully_async.md`): separate
  Rollouter/Trainer resources, NCCL parameter sync, "2.35x–2.67x" on Qwen2.5-7B with 128 GPUs, staleness from 0.x to multiple steps.
- Weight sync (`docs/advance/delta_weight_sync.md`): disaggregated setups broadcast weights every step; "over 99% of BF16 weight bytes are
  unchanged step-over-step" (~1–3% change) → delta sync 1.3–21× faster (0.5B–235B).
- OpenRLHF (Ray + vLLM, PPO/GRPO/REINFORCE++) — not cloned **(unverified)**; name it as a third framework only.

## 7. Preference optimisation (TRL v1.14.0)
- Bradley–Terry (`trl/docs/source/reward_trainer.md`): p(y⁺ ≻ y⁻ | x) = σ(r(x,y⁺) − r(x,y⁻)); loss −E[log σ(r⁺ − r⁻)]; BT is
  shift-invariant → `center_rewards_coefficient` (recommended 1e-2).
- DPO loss (docs/source/dpo_trainer.md): −E log σ(β[log π_θ(y⁺|x)/π_ref(y⁺|x) − log π_θ(y⁻|x)/π_ref(y⁻|x)]). Code (quote,
  `trl114/dpo_trainer.py` l.1425–1463): `chosen_logratios = chosen_logps - ref_chosen_logps`; `rejected_logratios = rejected_logps -
  ref_rejected_logps`; (`f_divergence_type == "reverse_kl"` = standard DPO) `delta_score = chosen_scores - rejected_scores`;
  `if loss_type == "sigmoid": per_sequence_loss = -F.logsigmoid(self.beta * delta_score)`. Implicit reward (l.1672):
  `chosen_rewards = self.beta * chosen_logratios.detach()` → metrics `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`
  (`chosen_rewards > rejected_rewards`), `rewards/margins`. `logps` are **summed** over completion tokens.
- `DPOConfig.beta = 0.1` (default); `loss_type` default `["sigmoid"]` (list; others `hinge`, `ipo`, `exo_pair`, `robust`, …); `label_smoothing` 0.0.
- IPO (`loss_type="ipo"`, code l.1468–1479): per-token-averaged log-ratio difference, `(ipo_delta - 1 / (2 * self.beta)) ** 2` — a squared
  loss toward a fixed margin 1/(2β) instead of the logistic, so it cannot push the margin to infinity.
- KTO (`from trl import KTOConfig, KTOTrainer`; kto_trainer.md): unpaired binary desirable/undesirable labels, prospect-theory utility
  (loss aversion); KL term estimated from mismatched pairs in the batch (batch size > 1).
- ORPO (`from trl.experimental.orpo import ORPOConfig, ORPOTrainer` — **experimental** in TRL): reference-model-free; NLL (SFT) loss plus a
  log-odds-ratio penalty on the rejected response, SFT and preference in one stage.

## 8. Test-time compute estimators
- Unbiased pass@k (Chen et al. 2021, HumanEval) — `passk/human_eval_evaluation.py: estimate_pass_at_k` and `passk/hf_evaluate_code_eval.py`
  (identical): `"""Calculates 1 - comb(n - c, k) / comb(n, k)."""` `if n - c < k: return 1.0` `return 1.0 - np.prod(1.0 - k /
  np.arange(n - c + 1, n + 1))` (numerically stable product form). lm-eval's `humaneval/utils.py: pass_at_k` wraps HF `code_eval`.
- pass^k (τ-bench, all k trials succeed) — `passk/tau_bench_run.py: display_metrics`: `sum_task_pass_hat_k += comb(c, k) / comb(num_trials,
  k)` averaged over tasks.
- Hand values (computed, both forms agree): n=10,c=3: pass@1 0.3, pass@5 0.916667, pass@8 1.0, pass^5 0.0; n=16,c=4,k=4: pass@4 0.728022,
  pass^4 0.000549; n=64,c=16,k=8: pass@8 0.914746.
- Majority vote in lm-eval (`lm-eval/lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml`): `repeats: 64`, `temperature: 0.2`, filters
  `majority_vote` → `maj@64`; `maj@8` via `take_first_k` ("Using a better estimator would be optimal").
- Other hand values for tests: GRPO advantages for rewards [1,0,0,1] (Bessel std 0.57735) = ±0.865875; [1,0,0,0] (std 0.5) = [1.4997, −0.4999×3];
  k3 at log(π_ref/π_θ)=0.1 → 0.0051709, −0.1 → 0.0048374, 0.5 → 0.1487213; DPO sigmoid loss with β=0.1, logratios +1/−1 → 0.598139
  (ln 2 = 0.693147 at zero margin); soft overlong (L_max 100, cache 20): |y| 80 → 0, 90 → −0.5, 100 → −1.0, 101 → −1.
- Snell et al. 2024 (compute-optimal test-time scaling; PRM reranking/search) is cited by the R1 paper; its specific numbers **(unverified)**.

## 9. Industry pattern for effort control
- OpenAI-compatible `reasoning_effort` on Chat Completions (`reasoning.effort` on Responses): vLLM accepts `none|minimal|low|medium|high|xhigh|max`
  (§1). OpenAI's own current value set **(verify)** — do not claim beyond "low/medium/high plus newer none/minimal/xhigh".
- gpt-oss: `gpt-oss-120b` 117B params / 5.1B active (single 80 GB GPU), `gpt-oss-20b` 21B / 3.6B active (runs within 16 GB), MXFP4 MoE
  weights; "Configurable reasoning effort: … (low, medium, high)"; must use the Harmony format (`gpt-oss/README.md` l.18–45). The chat template
  writes `"Reasoning: " + reasoning_effort` into the system message, default **`"medium"`** (convert_gpt_oss_weights_to_hf.py l.638–642);
  the reference chat CLI default is `low` (`-r/--reasoning-effort`, README l.357). Channels `analysis` (CoT), `commentary`, `final`.
- Repo already routes by effort: `06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/02-reference-architecture.md` l.128
  ("route (task × level × mode → model, `reasoning_effort`, output cap, backend)"); `…/docs/01-scaling-primer.md` l.319 ("reasoning tokens
  are billed as output at four to five times the input price … decode steps the whole batch pays for").

## 10. SGLang comparison (`sglang/docs/docs/advanced_features/separate_reasoning.mdx`, `server_arguments.mdx`)
- `--reasoning-parser` (default None) values: `auto`, `apertus2509`, `deepseek-r1`, `deepseek-v3`, `deepseek-v4`, `dots`, `glm45`, `ling3`,
  `hunyuan`, `gpt-oss`, `k2_horizon`, `kimi`, `kimi_k2`, `kimi_k3`, `mimo`, `muse`, `poolside_v1`, `qwen3`, `qwen3-thinking`, `minimax`,
  `minimax-append-think`, `minimax-m3`, `step3`, `step3p5`, `mistral`, `nemotron_3`, `interns1`, `gemma4`, `gigachat35`, `inkling`, `cohere_command4`.
- Response field **`reasoning_content`** ("follows the DeepSeek API design"); request extras `separate_reasoning` (on by default; set False to
  disable) and `stream_reasoning`. So: vLLM 0.30 → `reasoning`; SGLang, DeepSeek API, Qwen docs, llama.cpp `--reasoning-format deepseek` →
  `reasoning_content`. A portable client reads both.

## 11. Repo material to reuse (exact names and numbers)
- Capacity primer `00-foundations/gpu-capacity-planning/PRIMER.md` + `capacity.py` (stdlib): `BYTES`, `GPU`, `GPUS` (H100 80 GB, 3.35 TB/s,
  990/1979 TFLOPS), `ModelSpec(name, params_b, layers, kv_heads, head_dim, active_b=None)`, `MISTRAL_SMALL` (24B, 40 L, 8 kv, 128),
  `weight_memory_gb(params_b, dtype)`, `usable_hbm_gb(gpu, overhead=0.10)`, `kv_per_token_kb(model, dtype)`, `kv_per_session_gb(model,
  context_tokens, dtype)`, `max_concurrent_sessions(spare_gb, model, context_tokens, dtype)`, `decode_step_ms(bytes_gb, gpu)`,
  `decode_tok_s_single(weight_gb, gpu)`, `decode_aggregate(model, gpu, batch, context_tokens, dtype)` → (agg, per_user), `roofline_batch(gpu)`
  (H100 ≈ 296), `prefill_flops`, `ttft_s(active_b, prompt_tokens, gpu, dtype="fp8", mfu=0.5)`, `prefill_tok_s`, `request_duration_s(active_b,
  in_tokens, out_tokens, gpu, tpot_ms=40, dtype="fp8", mfu=0.5)`, `concurrency(rps, duration_s)`, `provision(raw_gpus, util=0.7, redundancy=1)`.
  Units: GB = 1e9 in weights but `kv_per_session_gb` divides KB by 1024² (mixed; keep as is to reproduce).
- Its bank example (`worked_example.py` B; the no-thinking baseline `rlcore.workload` must reproduce): 10,000 staff × 0.10 × 0.5/min →
  **8.33 RPS**; 1,500 in / 300 out; `AVG_CTX = IN + OUT//2 = 1650`; TPOT 40 ms; fp8; TTFT(24B, 1500) = 0.0728 s; duration **12.07 s**;
  concurrency **100.6**; sessions/GPU bf16 **95.3** (1.06 GPU), fp8 **381.3** (0.26 GPU); decode need 2,500 tok/s vs **9,156**/GPU;
  prefill need 12,500 vs **20,615**/GPU.
- Same formulas with 10× output (3,000 tokens of thinking+answer, computed today): duration **120.07 s**, concurrency **1,000.6**,
  AVG_CTX 3,000, KV/session fp8 0.2289 GB, sessions/GPU fp8 **209.7** → **4.77 GPUs** for memory (18× the baseline's 0.264); decode need
  25,000 tok/s. Caveat: `decode_aggregate` does not cap batch by HBM or the roofline (batch 1,000 × 0.229 GB = 229 GB KV > 80 GB HBM);
  the new `workload.py` must enforce memory and the compute roofline itself.
- KV×time grows faster than tokens: a request's KV-token-seconds ∝ ∫(P + t)dt over L output tokens = P·L + L²/2 (derivation; builders compute
  their own 5–20× working-set figure from the core rather than quoting one).
- `servelab` (04 lab, `04-inference-engine/serving-engine/vllm-serving-lab/servelab/`): `bench` exports `Request(prompt, messages, max_tokens=128,
  prompt_tokens, tag)`, `Lengths.fixed|uniform|lognormal(median, sigma=0.8, lo, hi)|choice`, `random_requests(n, input_len, output_len,
  prefix_tokens=0, seed=0, tag="")`, `mixed_requests`, `arrival_times(n, rate, burstiness=1.0, seed=0)`, `agent_sessions`, `run_open_loop(url,
  requests, rate, **kw)`, `run_closed_loop(url, requests, concurrency, **kw)`, `run_sessions`, `open_loop(url, requests, rate=inf,
  burstiness=1.0, seed=0, *, model, headers, max_concurrency, warmup=0, ignore_eos=True)`, `BenchRun(results, duration_s, mode, params,
  target, model, simulated)` with `.summary(slo, label, tag)`, `.report(slo)`; `SLO(ttft_ms, tpot_ms, e2el_ms)`; `summarize(...) ->
  Summary(completed, failed, …, goodput, slo_attainment, ttft, tpot, itl, e2el: Stat, cached_fraction)`; `littles_law(rate, latency)`;
  `RequestResult(ok, start, ttft, itl, latency, prompt_tokens, output_tokens, cached_tokens, chunk_times, text, …)`, `.tpot`.
  **Pitfalls:** `build_payload` hard-codes `"temperature": 0.0` and `ignore_eos=True` by default; `stream_request` reads only
  `delta.content`/`text` — a reasoning delta counts as a token chunk with empty text (TTFT = first reasoning token; `text` excludes the
  reasoning); `ttft_on_content=True` makes TTFT = first chunk with content text. thinklab needs its own reasoning-aware client.
- `servelab.metrics` constants (`RUNNING="vllm:num_requests_running"`, `KV_USAGE="vllm:kv_cache_usage_perc"`, `PREEMPTIONS`, `ITL`, `TPOT`,
  `GENERATION_TOKENS`, `REQUEST_GENERATION_TOKENS="vllm:request_generation_tokens"`, …) plus `parse`, `scrape(url)`, `delta`,
  `histogram_quantile`, `snapshot(later, earlier) -> EngineSnapshot`. `servelab.fakeserver.FakeServer(profile="t4-qwen2.5-0.5b", …)` (context
  manager returns url; header `x-servelab-simulated: true`; `/version` says simulated; histogram buckets copied from vLLM v0.30.0);
  `fake_engine.PROFILES`: `t4-qwen2.5-0.5b` (T4, `dtype="half"`, max_model_len 4096), `l4-qwen2.5-1.5b`, `l4-llama3.1-8b-fp8`,
  `h100-llama3.1-8b`; `build_profile(model, gpu, dtype, …, gpu_memory_utilization=0.92, max_model_len=4096, block_size=16, mfu=0.5,
  bw_efficiency=0.8, overhead_ms=4.0)`. `servelab.sizing`: `GPUS["T4"] = GPU("T4", 15.0 GiB, 320 GB/s, 65 TFLOPS, 0, "7.5", bf16=False)`,
  `L4` 22.49 GiB / 300 GB/s / 121 / 242; `load_config`, `param_count`, `kv_bytes_per_token`, `size(model, gpu, *, gpu_memory_utilization=0.92,
  max_model_len, dtype, …) -> SizingReport` (`num_blocks`, `max_concurrency`, `notes`). Bundled configs include `qwen3-0.6b`, `qwen3-8b`.
- 04 `deploy/any-gpu/README.md` (reuse verbatim facts): vLLM 0.30.0 is a CUDA 13 build, **min compute capability 7.5** (T4 ok; P100/V100 no),
  driver ≥ 580 (verify); T4 attention backend auto = **`TRITON_ATTN`**; `--dtype half` on T4; `--gpu-memory-utilization 0.92` default (0.85 in
  the Colab recipe); `--enable-prompt-tokens-details` for `cached_tokens`; image `vllm/vllm-openai:v0.30.0`; Colab `pip install -q "vllm==0.30.0"`.
- 04 PRIMER: §5 prefix caching (full 16-token blocks only; chained block hash; LoRA/multimodal/`cache_salt` extra keys), §7 speculative decoding
  (acceptance α = Σ min(p,q)), §11 measuring (metric table; percentiles; goodput). vllm-internals §9: `--max-loras` default 1, `--max-lora-rank` 16,
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING` for `/v1/load_lora_adapter`.
- 06: `agentic-scaling-lab/scalelab/capacity.py: cost_per_call(model, input_tokens, output_tokens, cached_tokens=0)`, `Scenario(output_tokens=350
  # including thinking, …)`. Worked: gemini-3.5-flash row ($1.50 in / $9.00 out / $0.15 cached per 1M): 5,000 in (2,700 cached) + 350 out =
  $0.007005/call; with 3,500 out = $0.035355 (5.05×).
- 07 `agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py`: `agentlab.evals.wilson_interval(passes, n,
  z=1.96)`, `cohen_kappa`, `run_eval`, golden sets, run-to-run noise, release gates — cite for "accuracy needs intervals".
- Tiny GPT to mirror for `tinyrl`: `00-foundations/transformers/lessons/03_tiny_gpt.py` (torch: `Attention`, `Block`, `GPT`, `generate`).

## Model and tool ids to use
- T0 (no download): numpy toy policies; `tinyrl` tiny decoder trained from scratch (torch CPU 2.14 is installed here; lazy import);
  servelab-style fake server emitting `reasoning` deltas; bundled recorded outputs labelled "sample output in the documented format (illustrative)".
- T1 free T4 (fp16, `--dtype half`, TRITON_ATTN): **`Qwen/Qwen3-0.6B`** (0.596 B, 1.19 GB) `--reasoning-parser qwen3`, thinking toggled with
  `chat_template_kwargs.enable_thinking`; **`Qwen/Qwen3-1.7B`** (1.72 B, 3.44 GB) `qwen3`; **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`**
  (1.78 B, 3.55 GB) `--reasoning-parser deepseek_r1` (no system prompt; always thinks); `Qwen/Qwen3-4B` (4.02 B, 8.04 GB; ~4.6 × 8K
  requests predicted) `qwen3`. Budget demo: `--reasoning-config` + `thinking_token_budget`. fp16 numerics of Qwen3 on Turing **(verify: check
  outputs are not garbage/NaN)**.
- T1 rented 24 GB (L4/4090, bf16): Qwen3-4B, `Qwen/Qwen3-4B-Thinking-2507` (thinking-only; parser `deepseek_r1` per Qwen README; `qwen3`
  also parses it since the parser starts in reasoning state **(verify)**); Qwen3-8B in bf16 (16.4 GB) fits a 24 GB card with little KV.
- RL step (lab 05): policy `Qwen/Qwen2.5-0.5B-Instruct` (TRL quick-start model; 0.494 B, 0.99 GB fp16, KV 12 KiB/token) or `Qwen/Qwen3-0.6B`;
  `trl==1.14.0` + `vllm==0.30.0`; on T4 `GRPOConfig(bf16=False, fp16=True, use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization≈0.3,
  num_generations=4–8, max_completion_length≤256)` **(verify it fits 15 GB with gradient checkpointing)**.
- Not on T4: gpt-oss-20b (MXFP4, "within 16GB" — T4 support unverified), Qwen3-8B fp16, anything ≥ 8B fp16.

## Pitfalls
1. **`reasoning` vs `reasoning_content`.** vLLM 0.30 returns `message.reasoning` / `delta.reasoning`; SGLang, DeepSeek API, Qwen docs use
   `reasoning_content`. Parse both; the fake server should emit vLLM's `reasoning` (it mimics vLLM). The SPEC text says `reasoning_content` — follow vLLM.
2. `--enable-reasoning` is gone; parser names use underscores in vLLM (`deepseek_r1`) and hyphens in SGLang (`deepseek-r1`).
3. `reasoning_effort` on Qwen3 is binary (thinking on/off), not a length control; only gpt-oss maps low/medium/high into the prompt, and Harmony
   rejects `none`/`minimal`/`xhigh`.
4. `max_tokens` counts reasoning tokens: a small `max_tokens` ends inside `<think>` with `content = None`, `finish_reason="length"` — not a
   budget. Budget forcing = `thinking_token_budget` (vLLM) or Qwen's two-call recipe; `-1` = unlimited.
5. Greedy decoding on thinking models: Qwen/DeepSeek warn of endless repetition. servelab's bench sends `temperature 0.0` and `ignore_eos` —
   do not reuse `build_payload` for accuracy runs; `ignore_eos=True` also destroys the natural thinking-length distribution.
6. DeepSeek-R1(-Distill): no system prompt; temperature 0.6; the R1 template pre-fills `<think>\n` so outputs may lack the opening tag.
7. Chat templates drop earlier-turn thinking → turn N's generated KV is not a prefix of turn N+1 (prefix-cache hit ends at the assistant
   header); inside one tool loop thinking is kept. Clients must send `reasoning` back only if they want interleaved thinking.
8. TRL GRPO defaults are not the paper's: `beta=0.0` (no reference model), `loss_type="dapo"`, `num_iterations=1` (clip inactive, ratio ≡ 1),
   `scale_rewards="group"`; set them explicitly when reproducing "GRPO". std uses Bessel's correction and +1e-4.
9. GRPOConfig sets `bf16=True` unless `fp16` is set — a T4 has no bf16. vLLM server mode must use different GPUs from the trainer.
10. Groups with all-equal rewards give zero advantage and zero gradient (`frac_reward_zero_std`); DAPO's dynamic sampling resamples them.
11. `decode_aggregate` in `capacity.py` has no memory or roofline cap; `kv_per_session_gb` uses 1024² while weights use 1e9 — reproduce, don't "fix".
12. Qwen3-0.6B's head_dim (128) ≠ hidden/heads (64): always read `head_dim` from config (sizing.py comment).
13. Structured output with a reasoning parser applies after `</think>` unless `enable_in_reasoning=True`; tool calls parse only from `content`.
14. pass@k needs n ≥ k samples and the unbiased estimator (not 1−(1−p̂)^k); pass^k is the all-k-succeed reliability metric (τ-bench).
15. No vLLM metric separates reasoning from answer tokens; use `usage.completion_tokens_details.reasoning_tokens` per request.
16. Prices, OpenAI effort value sets, Colab quotas, and T4 fp16 behaviour of Qwen3 are `(verify)`; simulated numbers are labelled "simulated".
