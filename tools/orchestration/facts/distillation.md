# facts-distillation — verified 2026-09-27 (research agent). Source of truth for `00-foundations/distillation` (module 00.6).
Paths are relative to `$SP/ref/` (`$SP` = this scratch `orch/` dir) unless they start with a repo dir (`00-…`, `01-…`, `04-…`, `06-…`, `07-…`,
`COMPUTE.md`). PDFs were extracted to `$SP/txt/qwen3.txt` and `$SP/txt/r1.txt` (pypdf needs `sys.modules['cryptography']=None` first here —
the system cryptography wheel is broken). **(verify)** = found but version/product-sensitive; **(unverified)** = not in any source here;
**textbook** = standard result, derivation given. Extends `$SP/FACTS.md` and `$SP/facts-rl-and-thinking-models.md` (cited as "rl-sheet §n");
do not repeat what they verify.

## 0. Versions and fetched files
- PyPI today: `trl` **1.14.0** (latest), `vllm` **0.30.0** (latest), `transformers` 5.17.0, `peft` 0.21.0, `lm-eval` 0.4.13. torch here: 2.14.0+cu130 (CPU).
- TRL clone `trl/` = main 1.15.0.dev0 @a7c34f3 (2026-09-26). v1.14.0 files fetched to `trl114d/` (path `/`→`_`, e.g. `trl_experimental_gkd_gkd_trainer.py`,
  `trl_trainer_distillation_config.py`, `trl_trainer_sft_config.py`, `docs_source_gkd_trainer.md`, `trl_chat_templates_qwen3_training.jinja`).
  GKD trainer and config are **byte-identical** between v1.14.0 and main (diff checked).
- TRL 1.14.0 `pyproject.toml`: `transformers>=4.56.2`, `accelerate>=1.4.0`, `datasets>=4.7.0`; extras `peft = ["peft>=0.13.0"]`,
  `vllm = ["vllm>=0.20.0,<=0.30.0", "aiohttp>=3.13.3", "requests"]` → compatible with the repo's vLLM 0.30.0.
- vLLM clone `vllm/` = main @3137ff0 (2026-09-27). v0.30.0 files fetched to `vllm030sd/` (`vllm_config_speculative.py`, `vllm_v1_spec_decode_metrics.py`,
  `docs_features_speculative_decoding_*.md`, `vllm_config_model.py`, `vllm_sampling_params.py`, `vllm_engine_arg_utils.py`, `vllm_entrypoints_openai_*_protocol.py`).
  `metrics.py` = main; `speculative.py` differs only by type renames/a dspark field. `docs/features/spec_decode.md` 404s at v0.30.0 → `docs/features/speculative_decoding/*.md`.
- Other clones: `tinker-cookbook` @5d6e1eb, `eagle` @cb7e084, `specforge` (version.txt 0.2.0; README news v0.3.0), `modelopt` @23355ed, `lm-eval` 0.4.13;
  `peft021/` = peft v0.21.0 `lora_config.py`, `constants.py` (raw).

## 1. TRL: where distillation lives in 1.14.0
- **`GKDTrainer`/`GKDConfig` are experimental**: `from trl.experimental.gkd import GKDConfig, GKDTrainer` (`trl114d/trl_experimental_gkd___init__.py`;
  doc `trl114d/docs_source_gkd_trainer.md`). History (raw.githubusercontent tags): `trl/trainer/gkd_trainer.py` exists at v0.20–v0.28.0, both paths
  at v0.26–v0.28, gone from `trl/trainer/` at v0.29.1 and v1.x. `from trl import GKDTrainer` fails on 1.14.0.
- **`DistillationTrainer`/`DistillationConfig` are stable API in 1.14.0**: `from trl import DistillationConfig, DistillationTrainer` (`trl114d/trl___init__.py`
  l.53–54, 102–103). The old path `trl.experimental.distillation` is a shim that warns "deprecated and will be removed in v2.0.0 … promoted to the
  stable API". It was experimental at v1.5.0/v1.8.0 and in `trl/trainer/` from v1.10.0.
- Also experimental (main and 1.14): `minillm`, `gold`, `async_distillation`, `server_distillation`, `sdft`, `iw_opd`.
- **`GKDConfig(SFTConfig)` fields and defaults** (`trl114d/trl_experimental_gkd_gkd_config.py`): `temperature: float = 0.9` ("Temperature for sampling"),
  `lmbda: float = 0.5` ("student data fraction (i.e., the proportion of on-policy student-generated outputs)"), `beta: float = 0.5`,
  `max_new_tokens: int = 128`, `teacher_model_name_or_path: str | None = None` ("If `None`, the teacher model will be the same as the model being
  trained"), `teacher_model_init_kwargs = None`, `disable_dropout: bool = True`, `seq_kd: bool = False`. `__post_init__` raises unless 0 ≤ `lmbda`, `beta` ≤ 1.
- **`GKDTrainer(model, teacher_model, args, data_collator, train_dataset, eval_dataset, processing_class, …, peft_config, formatting_func)`**:
  subclasses `SFTTrainer`; sets `args.remove_unused_columns = False`, default collator `DataCollatorForChatML(tokenizer=processing_class,
  max_length=args.max_length)`, `dataset_kwargs["skip_prepare_dataset"] = True`. Dataset: conversational `"messages"` lists (doc "Expected dataset type").
  **Vocab check** (quote): `if student_vocab_size != teacher_vocab_size: raise ValueError(… "GKD compares the teacher's full next-token distribution,
  which requires a shared vocabulary. Use a teacher with the same vocab_size, or GOLD for cross-tokenizer distillation.")` — compares
  `config.get_text_config().vocab_size`, not the tokenizer.
- `__init__` warns at `lmbda=1.0, temperature=1.0`: "fully covered by `DistillationTrainer`" (rename `max_new_tokens`→`max_completion_length`; set `beta`).
- **Sampling** (quote): `generation_kwargs = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "do_sample": True, "top_k": 0,
  "top_p": 1.0, "use_cache": False if args.gradient_checkpointing else True, "pad_token_id": …}`; EOS from the model's generation_config.
- **`generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean", num_items_in_batch=None)`** (quote):
  ```
  student_logits = student_logits / temperature; teacher_logits = teacher_logits / temperature
  student_log_probs = F.log_softmax(student_logits, dim=-1); teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
  if beta == 0:   jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
  elif beta == 1: jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
  else: mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, …); kl_student = F.kl_div(mixture_log_probs, student_log_probs, …)
        jsd = beta * kl_teacher + (1 - beta) * kl_student
  ```
  `F.kl_div(input, target, log_target=True)` = Σ exp(target)(target − input) = KL(target ‖ input). So **β = 0 → KL(teacher ‖ student) = forward KL;
  β = 1 → KL(student ‖ teacher) = reverse KL**; else mixture m = (1−β)·p_student + β·p_teacher formed in log space with logsumexp, loss =
  β·KL(teacher‖m) + (1−β)·KL(student‖m) (GKD paper eq. 1). Reduction: mask `labels != -100` selects token rows, the loss **sums over the vocab and
  averages over valid tokens** (`jsd.sum() / mask.sum().clamp_min(1)`), or `sum / num_items_in_batch` when given (grad-accumulation-correct).
- **`compute_loss` calls `generalized_jsd_loss(…, beta=self.beta, num_items_in_batch=…)` without `temperature`** → the loss always runs at T = 1;
  `GKDConfig.temperature` only sets sampling. No T² factor anywhere in TRL. Teacher runs `eval()` under `torch.no_grad()`; logits shifted
  `[:, :-1, :]`, labels `[:, 1:]`.
- **`training_step`** (quote): `if random.random() <= self.lmbda:` student generates (`generate_on_policy_outputs(unwrapped_model, inputs,
  self.generation_config)`) → **on-policy**; `elif self.seq_kd:` the **teacher** generates the completion → sequence-level KD targets; else the dataset's
  own completion (supervised KD). Per batch, not per sample. `generate_on_policy_outputs` generates from `inputs["prompts"]`, masks after the first EOS
  (`torch.isin` over a list of eos ids), sets prompt labels to −100.
- Doc (`docs_source_gkd_trainer.md`): "on-policy data (high `lmbda`) performs better and the optimal `beta` varied depending on the task"; example pair `Qwen/Qwen2-0.5B-Instruct` ← `Qwen/Qwen2-1.5B-Instruct`.
- **`DistillationConfig` (1.14.0) defaults** (`trl114d/trl_trainer_distillation_config.py`): `learning_rate=1e-6`, `max_completion_length=512`,
  `temperature=1.0`, `top_p=1.0`, `top_k=0`, `beta=1.0` ("0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL … Unlike GRPO's `beta` … it selects the divergence
  itself; there is no reference-model KL penalty"), `disable_dropout=False`, `use_vllm=False`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.3`,
  `vllm_server_host="0.0.0.0"`, `vllm_server_port=8000`, `teacher_model_name_or_path=None`, `chat_template_kwargs=None`, `max_tool_calling_iterations=None`.
  Overrides: `logging_steps` 10, **`gradient_checkpointing` True, `bf16` True if `fp16` not set**. No `lmbda`: always fully on-policy.
  The chunked loss applies `temperature` to both logits before the divergence (`_chunk`, l.105–160) in chunks of `_CHUNKED_LM_HEAD_CHUNK_SIZE = 256`
  positions so the `[B, T, V]` logits are never materialised. Dataset: conversational **prompt-only** `{"prompt": [{"role": "user", "content": …}]}`.
  **Quick start uses exactly our pair**: `DistillationTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
  train_dataset=load_dataset("trl-lib/ultrafeedback-prompt", split="train"))`. Logged: `completions/mean_length`, `completions/clipped_ratio`, `entropy`, ….
- **MiniLLM in TRL** (`trl/docs/source/minillm_trainer.md`, main): L = α₁ E_{x∼π_θ} Σ_{t'≥t} γ^{t'−t}/Σγ^{t'−t} [log π_θ(x_{t'+1}|x_{≤t'}) − log π_teacher(…)]
  + α₂ E KL[π_θ(·|x_{≤t}) ‖ π_teacher(·|x_{≤t})]; `MiniLLMConfig(rkl_advantage=True, single_step_decomposition=False, gamma=0.0)` = "the on-policy KD
  implemented in Tinker"; `rkl_advantage=False, single_step_decomposition=True` = "the reverse KLD version of the GKD loss".
- **GOLD** (`trl/docs/source/gold_trainer.md`; `GOLDConfig(SFTConfig)` at 1.14.0): "extension of Universal Logit Distillation (ULD) that supports
  student/teacher pairs with different tokenizers. It aligns the textual spans … and merges the associated logits"; flags `use_uld_loss` (default False),
  `teacher_tokenizer_name_or_path`, `uld_use_hybrid_loss`, inherits `beta`, `lmbda`, `seq_kd`. Experimental ("APIs may change without notice").
- **`SFTConfig` (1.14.0) defaults** (`trl114d/trl_trainer_sft_config.py`): `learning_rate=2e-5`, `max_length=1024`, `dataset_text_field="text"`,
  `packing=False`, `completion_only_loss=None` (→ loss on the completion for prompt-completion datasets, full sequence for language modelling),
  `assistant_only_loss=False` ("only on the assistant responses, which is supported only for conversational datasets"), `loss_type` unset →
  `"chunked_nll"` (lm_head only on non-ignored tokens, chunked CE); overrides `gradient_checkpointing` True, **`bf16` True if `fp16` not set**.
  With `assistant_only_loss=True` and no `{% generation %}` markers, the trainer swaps in a training template (`get_training_chat_template`) and warns if
  the end-of-turn token is outside the mask ("the model may not learn to stop").
- Dataset formats (`docs_source_dataset_formats.md`): language modelling `{"text": …}`; conversational `{"messages": [{"role", "content"}…]}`;
  prompt-only `{"prompt": …}`; prompt-completion `{"prompt": [{"role": "user", …}], "completion": [{"role": "assistant", "content": …}]}`.
- TRL's Qwen3 training template (`trl114d/trl_chat_templates_qwen3_training.jinja`, = main): "Removed the loop.index0 > ns.last_query_index conditional;
  **always include thinking block**" + `{% generation %}` markers; reasoning comes from `message.reasoning_content` or is split off `</think>` in content;
  renders `'<think>\n' + reasoning + '\n</think>\n\n' + content`. So SFT on Qwen3 always trains a (possibly empty) think block (contrast rl-sheet §2 history drop).

## 2. tinker-cookbook: on-policy distillation as policy gradient
- Reference (`tinker-cookbook/skills/research/references/distillation.md`): "**On-policy** (recommended): Student generates, teacher scores via KL
  divergence"; "**Off-policy reasoning**: SFT on teacher-generated traces (e.g., OpenThoughts3)"; "**Multi-teacher**: Different teachers for different
  datasets". `kl_penalty_coef` (default 1.0) "The only supervision signal"; `kl_discount_factor` (0.0 = no discount); `group_size` 4; `groups_per_batch` 1024.
- **The loss, as code** (`tinker_cookbook/distillation/train_on_policy.py: incorporate_kl_penalty`, quote): docstring "Compute reverse KL between the student
  (log p) and the teacher model (log q), computed as log p - log q. We then adjust the advantages in-place as the negative reverse KL."
  `teacher_logprobs_D = await asyncio.gather(*[teacher_client.compute_logprobs_async(sequence_input) …])`;
  `reverse_kl = [(sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask …]`;
  `kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]`; `if kl_discount_factor > 0: kl_advantages = discounted_future_sum_vectorized(…)`;
  added to `loss_fn_inputs["advantages"]`; logged metric `teacher_kl` = mean per-token log p − log q.
  Rewards are zero (README "an `Environment` that has no rewards … The only supervision comes from minimizing the KL against a teacher model").
  So the per-token advantage is **A_t = −(log π_student(y_t|y_<t) − log π_teacher(y_t|y_<t))** on the student's own samples, with the sampler's logprobs.
- Loss function: `Config.loss_fn: LossFnType = "importance_sampling"` (on-policy recipe `on_policy_distillation.py` l.81); tutorial `tutorials/202_loss_functions.py`:
  `importance_sampling` needs `target_tokens`, `logprobs`, `advantages` — "RL with on-policy or near-on-policy data | Corrects for sampler/learner
  mismatch; unbounded ratio"; `cross_entropy` "Supervised fine-tuning (SFT), distillation"; `ppo` clip defaults 0.8/1.2.
- Recipe defaults (`recipes/distillation/on_policy_distillation.py`): student `Qwen/Qwen3.5-9B-Base`, teacher `Qwen/Qwen3.5-9B`, `lora_rank=128`,
  `group_size=4`, `groups_per_batch=1024`, `learning_rate=1e-4`, `max_tokens=4096`, `temperature=1.0`, `kl_penalty_coef=1.0`, `kl_discount_factor=0.0`.
  Off-policy (`off_policy_reasoning.py`): OpenThoughts3, `batch_size=128`, `learning_rate=1e-3` (LoRA), `lora_rank=128`, `max_length=16384`, 1 epoch.
- Results quoted (`recipes/distillation/README.md`): SFT on OpenThoughts3 "AIME'24 score of ~65% using a rank-128 LoRA after 3000 steps"; then on-policy
  distillation on DeepMath "~76.7% using a rank-128 LoRA after 200 steps with 16k-token rollouts" (evaluated at temperature 1.0, top_p 1.0, max_tokens
  64000); personalization: "IF-eval … recover within approximately 100 steps". LR: SFT 1e-3 LoRA / 1e-4 full; on-policy 1e-4 LoRA / 5e-5 full.
  **No compute comparison against RL is in the clone** — the blog's multipliers are (unverified) (thinkingmachines.ai blocked).
- Multi-teacher (README): per dataset a `teacher_model` {`base_model`, `load_checkpoint_path`} + `groups_per_batch`; batches concatenated. Tests
  `tests/recipes/test_recipe_on_policy_multi_teacher.py`, `test_recipe_off_policy_reasoning.py` are `@pytest.mark.integration` (need the Tinker service).
- **Prompt (context) distillation** (`tutorials/406_prompt_distillation.py`, quote): "transfers knowledge embedded in a system prompt into the model's
  weights … Teacher: Generate labels using a detailed system prompt … Student: Train on those labels but *without* the system prompt"; a 70-line
  language-classification prompt; limitations: "Works best when the system prompt encodes *rules* rather than *world knowledge*"; "**The student learns
  similar mistakes as the teacher's.** … *the student can only learn behaviors the teacher demonstrates*"; production recipe "2100 multilingual sentences
  across 4 epochs".
- `lmops/minillm/README.md` news: "MiniLLM's 'minimizing reverse KLD by on-policy distillation' is introduced by Thinking Machine Lab".

## 3. vLLM 0.30.0 speculative decoding (for §7 and lab 04)
- `SpeculativeMethod = Literal["ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class", EagleModelTypes, NgramGPUTypes, DSparkModelTypes]`
  with `EagleModelTypes = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes]` (`"dflash"`), `"ngram_gpu"`, `"dspark"`;
  MTP names (e.g. `"mtp"`, `"gemma4_mtp"`, `"inkling_mtp"`) other than `"mtp"` are "deprecated and replaced with mtp". `RejectionSampleMethod =
  Literal["standard", "synthetic", "block"]`; `DraftSampleMethod = Literal["greedy", "probabilistic"]` (`vllm030sd/vllm_config_speculative.py` l.37–84).
- `SpeculativeConfig` fields: `num_speculative_tokens: int` (gt 0; "default to the number in the draft model config if present, otherwise, it is required"),
  `model: str | None` ("draft model, eagle head, or additional weights"), `method: SpeculativeMethod | None` (auto-detect: `"ngram"`/`"[ngram]"` → ngram,
  a dotted import path → `custom_class`, **else `"draft_model"`**), `draft_tensor_parallel_size` ("Can only be 1 or the same as the target"),
  `tensor_parallel_size` (only to raise "'tensor_parallel_size' is not a valid argument … Please pass 'draft_tensor_parallel_size'"), `quantization`,
  `kv_cache_dtype`, `max_model_len`, `prompt_lookup_max`/`prompt_lookup_min` (ngram; docs: both default 5 when omitted), `parallel_drafting=False`,
  `use_heterogeneous_vocab=False` (TLI, only `draft_model`, greedy only), `rejection_sample_method="standard"`, **`draft_sample_method="greedy"`**
  ("'greedy' always picks the argmax token, and the draft probabilities are treated as one-hot during rejection sampling. 'probabilistic' samples
  stochastically from the draft distribution and uses the full draft logits"), `synthetic_acceptance_rates`/`synthetic_acceptance_length`,
  `num_speculative_tokens_per_batch_size`, suffix-decoding knobs (24 / 10000 / 1.0 / 0.1).
- **`speculative_token_tree` does not exist** at v0.30.0 or main (grep empty) — do not document it.
- **`draft_model` needs equal vocab sizes** (`verify_equal_vocab_size_if_draft_model`, quote): "Target and draft model should have the same vocabulary size.
  … Using models with different tokenizers can cause out-of-bounds errors during speculative decoding." (skipped only with `use_heterogeneous_vocab`).
  Uses `get_vocab_size()` (config `vocab_size`): Qwen3-0.6B → Qwen3-4B/8B pass (151,936 each); **Qwen2.5-0.5B (151,936) → Qwen2.5-7B (152,064) fails** (§7).
- CLI: `--speculative-config` / `-sc` JSON (`arg_utils.py` l.1696–1698); shortcuts `--spec-method`, `--spec-model`, `--spec-tokens` exist at v0.30.0
  ("… and --speculative-config['…'] are mutually exclusive"). Docs warn the old `--speculative-model` / `--num-speculative-tokens` are deprecated.
  Doc example (`draft_model.md`): `vllm serve Qwen/Qwen3-4B-Thinking-2507 --seed 42 -tp 1 --max-model-len 2048 --gpu-memory-utilization 0.8
  --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'`; offline `LLM(model="Qwen/Qwen3-8B",
  speculative_config={"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"})`. EAGLE (`eagle.md`): `"method": "eagle"`,
  model `yuhuili/EAGLE-LLaMA3-Instruct-8B`; `"method": "eagle3"`, model `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3`. Medusa: `"medusa"` is a
  valid method (config code aligns `vocab_size` to the target's because MedusaConfig "falls back to its default (32001)"); no Medusa doc page at v0.30.0.
  The README points to `vllm-project/speculators` for training drafts. Draft/target "lossless up to the precision limits of hardware numerics".
- **Prometheus metrics** (`vllm030sd/vllm_v1_spec_decode_metrics.py: SpecDecodingProm`; counters, exposed with `_total`):
  `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`,
  `vllm:spec_decode_num_accepted_tokens_per_pos` (label `position` = 0..k−1). Docstring PromQL: acceptance rate = `rate(…num_accepted_tokens_total) /
  rate(…num_draft_tokens_total)`; **mean acceptance length = 1 + rate(accepted)/rate(drafts)** ("conventionally including bonus tokens"); per-position
  = `…accepted_tokens_per_pos_total / …num_drafts_total`.
- Log line (`SpecDecodingLogging.log`): "SpecDecoding metrics: Mean acceptance length: %.2f, Accepted throughput …, Drafted throughput …, Accepted: %d tokens,
  Drafted: %d tokens, Per-position acceptance rate: %s, Avg Draft acceptance rate: %.1f%%" with `mean_acceptance_length = 1 + num_accepted/num_drafts`,
  `acceptance_rates = sum(pos_matrix, axis=0) / num_drafts` (per position, **unconditional**: position i counts only if positions 0..i were all accepted).
- **Mapping to `minengine.spec`** (iid α, k drafts): mean acceptance length = `expected_tokens(α, k)` = (1−α^{k+1})/(1−α); per-position[i] = α^{i+1}
  (so α = per-position[0]); "Avg Draft acceptance rate" = (E−1)/k ≠ α. With the serving primer's α = 0.6: k=1 E 1.6000 (rate 0.600); k=3 E 2.1760
  (0.392); k=4 E 2.3056 (0.3264); k=5 E 2.3834 (0.2767; per-pos 0.6, 0.36, 0.216, 0.1296, 0.0778). `speedup(0.6, k, 0.1)`: k=3 1.6738 (= `best_k(0.6, 0.1)`),
  k=4 1.6469, k=5 1.5889 (computed here with `minengine.spec`).
- **Greedy drafting changes the per-token acceptance**: with `draft_sample_method="greedy"`, q is one-hot at x̂ = argmax q, so acceptance = p(x̂), not
  Σ min(p, q). On the primer's own p, q: argmax q = token 3, p(3) = **0.05** vs Σ min = 0.6 (serving-engine PRIMER §7 already states this rule; now
  verified: default `"greedy"`). With a greedy (T = 0) target, acceptance = 1[argmax q = argmax p].
- Per-request (`acceptance_metrics.md`, experimental): `--per-request-spec-decode-metrics none|summary|detailed` (default `none`) adds
  `metrics.speculative_decoding` {`mean_acceptance_length`, `draft_acceptance_rate`, `acceptance_histogram`, `num_draft_tokens`, …} for `n == 1` requests.

## 4. vLLM 0.30.0: scoring a completion under a served teacher
- `/v1/completions` request fields (`…completion_protocol.py`): `echo: bool = False`, `logprobs: int | None = None`, `prompt_logprobs: int | None = None`,
  `logprob_token_ids: list[int] | None` (≤ `MAX_LOGPROB_TOKEN_IDS = 128`; requires `logprobs`). Code (quote): `prompt_logprobs = self.prompt_logprobs;
  if prompt_logprobs is None and self.echo: prompt_logprobs = self.logprobs`; `echo_without_generation = self.echo and self.max_tokens == 0` → sampled
  with `max_tokens=1`. Response: `choices[].prompt_logprobs: list[dict[int, Logprob] | None]` (first prompt token has no logprob → None) and
  `logprobs.token_logprobs` / `top_logprobs` for echoed text. `prompt_logprobs` with `stream=True` → error.
- `/v1/chat/completions`: `logprobs: bool = False`, `top_logprobs: int = 0`, `prompt_logprobs: int | None` (vLLM extension), `echo`.
- **`--max-logprobs`** = `ModelConfig.max_logprobs: int = 20` ("the default for the OpenAI Chat Completions API. -1 means no cap … may cause OOM").
  `_validate_logprobs` raises "Requested sample logprobs of N, which is greater than max allowed: 20" and the same for `prompt_logprobs`; `-1` in a request
  means vocab size. **`--logprobs-mode`** = `"raw_logprobs"` default (`raw_logprobs | processed_logprobs | raw_logits | processed_logits`; "Raw means the
  values before applying any logit processors … Processed means … including temperature and top_k/top_p"; identical for prompt logprobs).
- Consequences for the lab: the reverse-KL reward needs only log π_T(y_t) of the *sampled* token — `prompt_logprobs=0` (or 1) on prompt+completion
  gives it for every position; full-distribution logit KD through an API is impractical (151,936 entries per token even with `--max-logprobs -1`), so
  API teachers give top-k (≤ 20 by default) → a truncated/renormalised KD, marked approximate. TRL GRPO server mode uses `--logprobs-mode
  processed_logprobs --max-logprobs -1` (rl-sheet §4).

## 5. DeepSeek-R1 (reuse rl-sheet §3; new quotes)
- Paper §2.4 (`txt/r1.txt` l.465–476): "we directly fine-tuned open-source models like Qwen … and Llama … using the 800k samples curated with DeepSeek-R1";
  bases Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct; "For distilled models, we apply only SFT and
  do not include an RL stage, even though incorporating RL could substantially boost model performance."
- README distill table (`deepseek-r1/README.md` l.147–152; AIME24 pass@1 / cons@64 / MATH-500 / GPQA-D / LiveCodeBench / CodeForces):
  Qwen-1.5B 28.9 / 52.7 / 83.9 / 33.8 / 16.9 / 954; Qwen-7B 55.5 / 83.3 / 92.8 / 49.1 / 37.6 / 1189; Qwen-14B 69.7 / 80.0 / 93.9 / 59.1 / 53.1 / 1481;
  Qwen-32B 72.6 / 83.3 / 94.3 / 62.1 / 57.2 / 1691; Llama-8B 50.4 / 80.0 / 89.1 / 49.0 / 39.6 / 1205; Llama-70B 70.0 / 86.7 / 94.5 / 65.2 / 57.5 / 1633.
- §4.1 Table 6 (AIME pass@1 / cons@64 / MATH-500 / GPQA / LCB): QwQ-32B-Preview 50.0/60.0/90.6/54.5/41.9; **R1-Zero-Qwen-32B (RL > 10K steps)
  47.0/60.0/91.6/55.0/40.2**; **R1-Distill-Qwen-32B 72.6/83.3/94.3/62.1/57.2**. Quote: "distilling more powerful models into smaller ones yields excellent
  results, whereas smaller models relying on the large-scale RL … may not even achieve the performance of distillation … advancing beyond the boundaries
  of intelligence may still require more powerful base models and larger-scale reinforcement learning."
- Licence (README §7, quote): "This code repository and the model weights are licensed under the MIT License. DeepSeek-R1 series support commercial use,
  allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs." Distills inherit: Qwen ones
  "derived from Qwen-2.5 series, which are originally licensed under Apache 2.0 License, and now finetuned with 800k samples curated with DeepSeek-R1";
  Llama-8B under the Llama3.1 license; Llama-70B under the Llama3.3 license.

## 6. Qwen3 strong-to-weak distillation (`txt/qwen3.txt`)
- l.81–83: "For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models …
  Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency."
- l.475–481: "directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while
  maintaining fine-grained control over their reasoning processes … better immediate performance, as indicated by higher Pass@1 scores, and also improves
  the model's ability of exploration, as reflected in improved Pass@64 results … requiring only 1/10 of the GPU hours compared to the four-stage training method."
- §4.5 (l.616–628, quote): covers "5 dense models (Qwen3-0.6B, 1.7B, 4B, 8B, and 14B) and one MoE model (Qwen3-30B-A3B)". "(1) Off-policy Distillation:
  … we combine the outputs of teacher models generated with both /think and /no think modes for response distillation … (2) On-policy Distillation: …
  prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its
  logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence." (Direction of the KL not stated → do not claim.)
- **Table 21** (Qwen3-8B, from the same off-policy-distilled checkpoint; pass@64 in parentheses; AIME'24 / AIME'25 / MATH500 / LiveCodeBench v5 /
  MMLU-Redux / GPQA-Diamond / GPU hours): Off-policy Distillation 55.0 (90.0) / 42.8 (83.3) / 92.4 / 42.0 / 86.4 / 55.6 / –; **+ Reinforcement Learning
  67.6 (90.0) / 55.5 (83.3) / 94.8 / 52.9 / 86.9 / 61.3 / 17,920**; **+ On-policy Distillation 74.4 (93.3) / 65.5 (86.7) / 97.0 / 60.3 / 88.3 / 63.3 /
  1,800**. 17,920 / 1,800 = 9.96× ("approximately only 1/10 of the GPU hours"); RL leaves pass@64 unchanged, distillation raises it.
- Table 1 (dense): layers, heads Q/KV, tie, context — 0.6B 28, 16/8, Yes, 32K; 1.7B 28, 16/8, Yes, 32K; 4B 36, 32/8, Yes, 128K; 8B 36, 32/8, No, 128K;
  14B 40, 40/8, No, 128K; 32B 64, 64/8, No, 128K. Tokenizer "vocabulary size of 151,669" (config `vocab_size` 151936). Licence (`qwen3/README.md` l.395–397):
  "All our open-weight models are licensed under Apache 2.0."

## 7. Model shapes and parameter counts
Counted with the repo's `servelab.sizing.param_count` (exact: q/k/v/o, Qwen2 qkv bias, Qwen3 q/k-norm, norms, embeddings; `04-…/vllm-serving-lab/servelab/sizing.py`).
Sources: bundled configs `…/servelab/data/configs/{qwen2.5-0.5b-instruct,qwen2.5-1.5b-instruct,qwen2.5-7b-instruct,qwen3-0.6b,qwen3-8b,llama-3.2-1b-instruct}.json`;
SpecForge draft configs mirror the target's width (`specforge/configs/*-eagle3.json`, `max_window_layers`/`num_target_layers` = target depth).

| model | L | h | heads Q/KV | head_dim | ffn | vocab | tied | total params | non-embedding | fp16 GB | KV B/token fp16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B(-Instruct) | 24 | 896 | 14/2 | 64 | 4864 | 151,936 | yes | 494,032,768 | 357,898,112 | 0.988 | 12,288 |
| Qwen2.5-1.5B(-Instruct) | 28 | 1536 | 12/2 | 128 | 8960 | 151,936 | yes | 1,543,714,304 | 1,310,340,608 | 3.087 | 28,672 |
| Qwen2.5-3B **(verify shape)** | 36 | 2048 | 16/2 | 128 | 11008 | 151,936 | yes | 3,085,938,688 | 2,774,773,760 | 6.172 | 36,864 |
| Qwen2.5-7B(-Instruct) | 28 | 3584 | 28/4 | 128 | 18944 | **152,064** | no | 7,615,616,512 | 6,525,621,760 | 15.231 | 57,344 |
| Qwen3-0.6B | 28 | 1024 | 16/8 | 128 | 3072 | 151,936 | yes | 596,049,920 | 440,467,456 | 1.192 | 114,688 |
| Qwen3-1.7B **(verify h, ffn)** | 28 | 2048 | 16/8 | 128 | 6144 | 151,936 | yes | 1,720,574,976 | 1,409,410,048 | 3.441 | 114,688 |
| Qwen3-4B (h/ffn from `specforge/configs/qwen3-4b-eagle3.json`) | 36 | 2560 | 32/8 | 128 | 9728 | 151,936 | yes | 4,022,468,096 | 3,633,511,936 | 8.045 | 147,456 |
| Qwen3-8B | 36 | 4096 | 32/8 | 128 | 12288 | 151,936 | no | 8,190,735,360 | 6,946,075,648 | 16.381 | 147,456 |
| DeepSeek-R1-Distill-Qwen-1.5B **(verify untied)** | 28 | 1536 | 12/2 | 128 | 8960 | 151,936 | no | 1,777,088,000 | 1,310,340,608 | 3.554 | 28,672 |
| Llama-3.2-1B-Instruct | 16 | 2048 | 32/8 | 64 | 8192 | 128,256 | yes | 1,235,814,400 | 973,146,112 | 2.472 | 32,768 |

- Qwen2.5-32B shape = QwQ-32B (`specforge/configs/qwq-32B-eagle3.json`: h 5120, ffn 27648, 40/8 heads, 64 layers, vocab 152,064; untied **(verify)**);
  Qwen3-32B = `qwen3-32b-eagle3.json` (h 5120, ffn 25600, 64/8, head_dim 128, 64 layers). `roofline.llm.ModelConfig.params()` (no norms/biases):
  Qwen2.5-32B 32,762,757,120; Qwen3-32B 32,761,446,400; Qwen2.5-0.5B 493,961,216; Qwen2.5-1.5B 1,543,569,408 (the preset `"qwen2.5-1.5b"`) — the two
  counters differ by norms/biases (≈0.01 %); the core should say which it uses.
- Ratios (total params): Qwen2.5-1.5B / 0.5B = 3.12×; Qwen3-1.7B / 0.6B = 2.89×; Qwen3-4B / 0.6B = 6.75×; Qwen2.5-32B / 1.5B = 21.2×. Non-embedding
  ratios are larger (1.5B/0.5B: 3.66×) because the 151,936 × h embedding is 28 % of the 0.5B and 26 % of the 0.6B.
- Vocab/tokenizer: Qwen2.5 0.5B/1.5B/3B and all Qwen3 have config `vocab_size` 151,936; **Qwen2.5-7B/14B/32B/72B have 152,064** (7B bundled config;
  32B via QwQ) — GKD's `vocab_size` check and vLLM's draft check both fail across that boundary although the tokenizers share ids. Qwen3 token ids
  (from `specforge/tests/test_data/test_references/qwen3-instruct_tool-use_ref.json`): `<tool_call>` 151657, `</tool_call>` 151658, `<tool_response>` 151665,
  `</tool_response>` 151666, `<think>` 151667, `</think>` 151668 (empty think block = `151667, 271, 151668, 271`). That Qwen2.5 and Qwen3 share the BPE
  merges below 151,643 is **(unverified)** — likely (same 151,643 regular tokens), but GKD across families needs the same *template* too (§Pitfalls).
- Chat templates (TRL `trl/chat_templates/`): Qwen2.5 injects a default system prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  when none is given; Qwen3's does not; R1-Distill uses `<｜User｜>` / `<｜Assistant｜>` (`deepseek_r1_distill.jinja`), not ChatML.

## 8. Training memory (T4 = 15.0 GiB usable, fp16 only — FACTS/COMPUTE.md; `servelab.sizing.GPUS["T4"]`, `bf16=False`)
- Bytes per parameter (textbook, ZeRO accounting): mixed-precision AdamW with fp32 master = 2 (fp16 weights) + 2 (grads) + 4 (fp32 master) + 4 + 4 (m, v) =
  **16 B/param**; the HF Trainer path on a T4 (load fp32, `fp16=True` autocast, fp32 grads, AdamW) is also 4 + 4 + 8 = 16 B. Pure bf16 (load bf16,
  `bf16=True`, `torch.optim.AdamW` states in param dtype) = 2 + 2 + 4 = **8 B/param (verify)**. Frozen inference weights 2 B/param. LoRA: frozen base
  2 B/param + adapters at 16 B/param.
- **Loading fp16 weights and training with `fp16=True` fails** — torch's GradScaler: `raise ValueError("Attempting to unscale FP16 gradients.")`
  (`torch/amp/grad_scaler.py`, torch 2.14). On a T4 load the trainable model in fp32 (or keep LoRA adapters in fp32) and set `fp16=True, bf16=False`
  (TRL configs default `bf16=True` when `fp16` is unset — must be overridden). Default `optim` = `"adamw_torch"` (`"adamw_torch_fused"` for torch ≥ 2.8)
  (`transformers/src/transformers/training_args.py` l.227). Flag names (`TrainingArguments`): `per_device_train_batch_size` (default 8),
  `gradient_accumulation_steps`, `learning_rate` (5e-5; TRL SFT 2e-5), `fp16`, `bf16`, `gradient_checkpointing`, `optim`; TRL trainers take
  `peft_config=LoraConfig(...)`.
- `peft` 0.21.0 `LoraConfig` defaults (`peft021/lora_config.py`): `r=8`, `lora_alpha=8`, `lora_dropout=0.0`, `bias="none"`, `target_modules=None` → model
  default `["q_proj", "v_proj"]` for `qwen2`/`qwen3`/`llama` (`peft021/constants.py` l.85, 103–104); `target_modules="all-linear"` picks every linear.
  `use_rslora=False`. Adapter params = r·(d_in + d_out) per matrix × layers.
- Worked (computed here; activations/logits extra):

| student | params | full FT 16 B/p | pure bf16 8 B/p | frozen 2 B/p | LoRA r=16 all-linear params (×16 B) | LoRA r=16 q,v |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 494.0 M | 7.90 GB | 3.95 GB | 0.99 GB | 8,798,208 = 1.78 % (0.141 GB) | 1,081,344 |
| Qwen3-0.6B | 596.0 M | 9.54 GB | 4.77 GB | 1.19 GB | 10,092,544 = 1.69 % (0.161 GB) | 2,293,760 |
| Qwen2.5-1.5B | 1,543.7 M | 24.70 GB | 12.35 GB | 3.09 GB | 18,464,768 = 1.20 % (0.295 GB) | 2,179,072 |
| Qwen3-1.7B | 1,720.6 M | 27.53 GB | 13.76 GB | 3.44 GB | 17,432,576 = 1.01 % (0.279 GB) | 3,211,264 |

- **Logits dominate activations at vocab 151,936**: one fp32 `[B·S, V]` tensor at B·S = 4,096 tokens = 4,096 × 151,936 × 4 B = **2.49 GB** (fp16 1.24 GB);
  checkpointed layer inputs are only L·B·S·h·2 B (0.5B: 0.176 GB). Logit KD holds teacher + student logits (+ softmax copies and grads) → several such
  tensors; hence TRL's `chunked_nll` and DistillationTrainer's 256-position chunks. A home-made `kd.py` must chunk or keep B·S small.
- Fits (predictions — label "predicted", **(verify)** on hardware): 0.5B full FT on T4: 7.9 GB + ~2–5 GB logits/activations at B·S ≤ 2,048 → fits with
  checkpointing; 0.6B full FT: 9.5 GB + extras → tight, B·S ≤ 1,024; logit KD/GKD on T4 adds the teacher's frozen fp16 weights (1.5B: 3.09 GB; 1.7B:
  3.44 GB) + its logits + generation KV → use LoRA (0.5B LoRA: 0.99 + 0.14 + teacher 3.09 GB ≈ 4.2 GB before activations) or B = 1–2. 1.5–1.7B full FT
  does not fit 24 GB at 16 B/p (24.7/27.5 GB) → pure bf16 (12.4/13.8 GB) + activations fits a 24 GB card, or LoRA (fits a T4: 3.1–3.4 GB frozen fp16).
- T4 (reuse): no bf16/FP8 (COMPUTE.md l.80–86); vLLM `--dtype half`, `TRITON_ATTN`; EAGLE README: "When Qwen2 is the target model, please use bf16
  precision instead of fp16 to avoid numerical overflow" → check fp16 outputs/losses for NaN/overflow on a T4 (verify per model).

## 9. Pruning, pretraining-scale KD, licences
- Minitron (`minitron/README.md`, quote): "obtained via pruning and knowledge distillation. We prune model embedding size, attention heads, and MLP
  intermediate dimension, following which, we perform continued training with distillation"; "Deriving the Minitron 8B and 4B models from the base
  Nemotron-4 15B model using our approach requires up to **40x fewer training tokens** per model compared to training from scratch; this results in
  **compute cost savings of 1.8x** for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a **16% improvement in MMLU**
  scores compared to training from scratch"; "SOTA 8B model via pruning and distillation with only **400B tokens**" (Mistral-NeMo-Minitron-8B);
  checkpoints `Llama-3.1-Minitron-4B-Width-Base`, `Llama-3.1-Minitron-4B-Depth-Base`; licence "NVIDIA Open Model License Agreement".
- ModelOpt pruning (`modelopt/examples/pruning/README.md`): Minitron "uses the activation magnitudes to prune the embedding hidden size; mlp ffn hidden size;
  transformer attention heads; … and number of layers"; `mtp.prune(…, mode="mcore_minitron", constraints={"export_config": {"num_layers": 32,
  "hidden_size": 3584, "ffn_hidden_size": 10240}})` (Qwen3-8B example); importance scoring on "512-1024 samples" (~5 min for 8B); NAS mode caps
  `max_width_pruning` 0.4, `max_depth_pruning` 0.2; "ideally we need to perform a short Knowledge Distillation on ~2B tokens for all top-K candidate
  architectures".
- Llama 3.2 (`llama-models/models/llama3_2/MODEL_CARD.md` l.58, quote): "For the 1B and 3B Llama 3.2 models, we incorporated logits from the Llama 3.1 8B
  and 70B models into the pretraining stage of the model development, where outputs (logits) from these larger models were used as token-level targets.
  Knowledge distillation was used after pruning to recover performance." Params "1B (1.23B)", "3B (3.21B)", "Up to 9T tokens", shared embeddings.
- Llama licence (`llama3_1/LICENSE` l.24 and `llama3_2/LICENSE` l.35, identical sentence): "If you use the Llama Materials or any outputs or results of the
  Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama”
  at the beginning of any such AI model name." Plus "prominently display “Built with Llama”"; 3.2 §2: > 700 million MAU needs a licence from Meta.
- Gemma (`transformers/docs/source/en/model_doc/gemma2.md` l.30): "The 2B and 9B models are trained with knowledge distillation"; `gemma3.md` l.29:
  "The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning." Gemma's teacher sizes/tokens (unverified).
- DistilBERT (`model_doc/distilbert.md`): "pretrained by knowledge distillation" with a "triple loss" (LM, distillation, cosine). TinyBERT/MiniLM: (unverified), no numbers.
- Licences summary for the lab: Qwen2.5 0.5B/1.5B/7B and Qwen3 Apache-2.0 (Qwen3 README; Qwen2.5-1.5B per the R1 README link); **Qwen2.5-3B and -72B use
  Qwen's own research/qwen licences (unverified here — do not use 3B as a default)**; DeepSeek-R1 MIT with distillation explicitly allowed; Llama
  requires the "Llama" name prefix on distilled models; hosted-API terms that forbid training competing models on outputs exist (unverified — quote none).
  One dated paragraph, "not legal advice".

## 10. Draft models as students: EAGLE, Medusa, SpecForge
- EAGLE (`eagle/README.md` l.34–57): "extrapolating the second-top-layer contextual feature vectors"; EAGLE: "3x faster than vanilla decoding (13B)",
  "1.6x faster than Medusa (13B)", "trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs"; EAGLE-2 "4x faster than vanilla decoding (13B)", "1.4x
  faster than EAGLE-1"; EAGLE-3 "removes the feature prediction constraint in EAGLE and simulates this process during training using training-time testing
  … replaces them with a fusion of low-, mid-, and high-level semantic features"; "5.6 faster than vanilla decoding (13B)", "1.8x faster than EAGLE-1
  (13B)"; benchmark note "Inference is conducted on 2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model". Qwen3 EAGLE-3 heads listed as
  unofficial: `AngelSlim/Qwen3-1.7B_eagle3`, `AngelSlim/Qwen3-4B_eagle3`, `Tengyunw/qwen3_8b_eagle3`.
- **EAGLE-3 training is distillation from the target** (`eagle/eagle/traineagle3/cnets.py`): the frozen target (`LlamaForCausalLM … torch_dtype=torch.float16`)
  supplies three hidden states (`torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)` → `self.fc = nn.Linear(hidden_size*3, hidden_size)`)
  and its logits `target = outs.logits`; loss per unrolled step (quote): `target_p = nn.Softmax(dim=2)(target_head)`; `out_logp = nn.LogSoftmax(dim=2)(logits)`;
  `loss = -torch.sum(position_mask * target_p * out_logp, 2).mean()` — **soft-target cross-entropy to the target's distribution** (over a reduced draft
  vocab via `t2d`), for `self.length = 7` training-time-test steps; accuracy = argmax agreement. DeepSpeed config: fp16, AdamW β (0.9, 0.95), warmup to
  lr 5e-5, `num_epochs` 40.
- Medusa (`medusa/README.md`): "the original model stays untouched, and only the new heads are fine-tuned during training"; "Medusa-2 (compared to
  Medusa-1, which only trains the new heads) … full-model training … keeping the original model's performance"; "self-distillation, which allows us to add
  Medusa to any fine-tuned LLM without requiring the availability of the original training data"; "2.2-3.6x speedup"; training example
  `--medusa_num_heads 3`.
- SpecForge (`specforge/README.md`, `docs/sections/…`): one entry point `specforge train --config <yaml>`; methods EAGLE3, P-EAGLE, EAGLE3.1, DFlash,
  DFlash2, Domino, DSpark. **Online** "captures target features while the run is active. It uses little disk space but keeps target inference available
  during training"; **Offline** "reads feature checkpoints generated ahead of time, so only the draft model must fit … at the cost of substantially more
  storage" (`training.md` §Online and offline data); the flag is `--target-model-path` (`scripts/prepare_hidden_states.py`) / YAML `target_model_path` —
  there is no `--target-model`. Data regeneration (`data_preparation.md`, quote): "we can regenerate the assistant responses using the target model to
  better align the draft model with the target model's output distribution. This will improve the acceptance rate … According to the EAGLE1 paper, the
  EAGLE method is not very sensitive to the dataset quality"; `scripts/regenerate_train_data.py --temperature 0.8`; `--reasoning save` keeps
  `reasoning_content`.

## 11. MiniLLM, distilling step-by-step, ModelOpt
- MiniLLM abstract (TRL doc, quote): "replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is
  more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher
  distribution. Then, we derive an effective optimization approach … more precise responses with the higher overall quality, lower exposure bias, better
  calibration, and higher long-text generation performance … 120M to 13B parameters". Code (`lmops/minillm/minillm/losses.py`): PPO-style `_pg_loss`
  with `log_ratio = (logprobs - old_logprobs) * mask`, `rev_kl`, discounted `_get_cumsum_rewards` (`args.gamma`). README has KD/SeqKD baselines
  (`scripts/gpt2/kd/*`, `scripts/gpt2/seqkd/*`); its result numbers are only in a figure → (unverified), no numbers.
- Distilling step-by-step (`distilling-step-by-step/README.md`): paper title "Outperforming Larger Language Models with Less Training Data and Smaller Model
  Sizes"; `--alpha`: "Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss", "`--alpha 0.5`: recommended"; `--model_type
  task_prefix` = distilling step-by-step; inputs prefixed `'predict: '` and `'explain: '` (`run.py` l.140–141); T5 v1.1 students; `--label_type gt|llm`,
  `--llm palm`. Headline percentages (unverified — not in README).
- ModelOpt (`modelopt/torch/distill/`): `mtd.convert(model, mode=[("kd_loss", distillation_config)])` (`docs/source/guides/4_distillation.rst` l.54);
  `KDLossConfig` fields `teacher_model`, `criterion`, `loss_balancer`; `DistillationModel.compute_kd_loss()`; losses `LogitsDistillationLoss(temperature=1.0,
  reduction="mean")`, `MFTLoss`, `MGDLoss`; **`LogitsDistillationLoss` multiplies by T²** (quote): "Since the magnitudes of the gradients produced by the
  soft logits scale as 1/(T^2), multiplying them by T^2 ensures that the relative contributions of the logits remain roughly unchanged" — `kd_loss *=
  self._temperature**2` on `F.kl_div(log_softmax(s/T), softmax(t/T))` = KL(teacher_T ‖ student_T). `DistillationConfig` in ModelOpt is the **Megatron plugin**
  dataclass (`plugins/megatron.py` l.53) with `LogitsKLLoss`, `TopKLogitsKLLoss`, `HiddenStateCosineLoss` — not a general API. HF path: `KDTrainer`
  (`plugins/huggingface.py`).

## 12. lm-eval (reuse `$SP/facts-quantization.md` l.334–344)
- `lm_eval --model hf|vllm --model_args pretrained=<id>,dtype=half --tasks gsm8k --num_fewshot 5 --limit 100` (or `lm-eval run …`). gsm8k: 5-shot,
  `generate_until`, **greedy** (`temperature: 0.0`), metric `exact_match` under `strict-match` / `flexible-extract`; `DEFAULT_MAX_GEN_TOKS = 256`
  (`lm_eval/defaults.py`). Thinking models (`docs/interface.md` l.146–176): `--model_args …,enable_thinking=True,think_end_token="</think>"
  --apply_chat_template`. `--limit` is "for testing only" (250 gsm8k ≈ ±0.027).

## 13. Textbook formulas (derivations; numbers computed here with numpy — the core must reproduce them)
- **Soft targets** (Hinton et al. 2015): p_i(T) = exp(z_i/T)/Σ_j exp(z_j/T). Loss = α·T²·KL(p_T^teacher ‖ q_T^student) + (1−α)·CE(y, q_1). Gradient: with
  q = softmax(v/T), p = softmax(z/T), ∂KL(p‖q)/∂v_i = ∂[−Σ p_j log q_j]/∂v_i = (q_i − p_i)/T. So T²·KL has gradient T(q_i − p_i). High-T: q_i ≈ 1/N +
  (v_i − v̄)/(NT) and likewise p_i, so T(q_i − p_i) → ((v_i − v̄) − (z_i − z̄))/N = ∂/∂v_i of (1/2N)Σ((v_i − v̄) − (z_i − z̄))²: **logit matching on centred
  logits**; T² keeps the soft term's gradient O(1) as T grows (at fixed T, KL itself shrinks ~1/T²).
- Five-token example (use it in the primer): teacher z = (4, 3, 1, 0, −1), student v = (3, 3.5, 0, 0.5, −1).
  T=1: p = (0.6931, 0.2550, 0.0345, 0.0127, 0.0047), q = (0.3573, 0.5891, 0.0178, 0.0293, 0.0065), KL(p‖q) = 0.25650.
  T=2: p = (0.4885, 0.2963, 0.1090, 0.0661, 0.0401), KL = 0.06639, T²·KL = 0.26558, ∇_v KL = (q−p)/T = (−0.073543, 0.071047, −0.016410, 0.015853, 0.003053)
  (finite differences agree to 1e-6). T=4: KL 0.01596, T²KL 0.25542; T=10: KL 0.00242, T²KL 0.24196; T=100: 0.23129; T=1000: 0.23013 → limit
  (1/2N)‖c(v) − c(z)‖² = **0.23000** exactly.
- **Divergences**: KL(p‖q) = Σ p log(p/q) (forward, from the teacher's view: mean-seeking/mode-covering — the student pays wherever p > 0 and q ≈ 0);
  KL(q‖p) reverse (mode-seeking — the student pays wherever q > 0 and p ≈ 0). Generalised JSD(β) = β·KL(p‖m) + (1−β)·KL(q‖m), m = βp + (1−β)q (TRL's
  convention, p = teacher). For small β, JSD(β) ≈ β·KL(p‖q); near 1, ≈ (1−β)·KL(q‖p). Example (T=1 above): KL(p‖q) 0.256501, KL(q‖p) 0.271424; JSD at
  β = 0.01 → 0.002539 (÷β = 0.254), 0.1 → 0.023026, 0.5 → 0.064431 (≤ ln 2), 0.9 → 0.024067, 0.99 → 0.002683 (÷(1−β) = 0.268). TRL's endpoints are the
  exact KLs, so the loss scale jumps ~1/β at β = 0 and 1.
- Bimodal teacher p = (0.46, 0.03, 0.02, 0.03, 0.46) vs a discretised-Gaussian student q ∝ exp(−(x−μ)²/2s²) on x = 0..4 (grid μ ∈ [0,4], s ∈ [0.2,4]):
  forward-KL fit μ = 2.0, s = 4.0 (grid edge: as flat as allowed) q = (0.188, 0.206, 0.213, 0.206, 0.188), KL 0.6621 — mass between the modes;
  reverse-KL fit μ = 0.02, s = 0.42, q = (0.938, 0.062, 0, 0, 0), KL 0.7132 — one mode (symmetric at μ ≈ 3.98). The core should pin its own family.
- **TV and acceptance**: TV(p, q) = ½Σ|p−q|; α = Σ min(p, q) = 1 − TV (primer p, q: α 0.6, TV 0.4).
- **On-policy (reverse-KL) gradient as REINFORCE**: KL(π_θ‖π_T) over sequences = E_{y∼π_θ}[log π_θ(y) − log π_T(y)]; ∇ = E[(log π_θ(y) − log π_T(y))·∇log π_θ(y)]
  (the ∇log π term has zero mean). So −∇KL is REINFORCE with sequence reward R(y) = −(log π_θ(y) − log π_T(y)) = Σ_t r_t, **r_t = log π_T(y_t|y_<t) −
  log π_θ(y_t|y_<t)**. In `rlcore.pg.reinforce_grad(policy, trajs, baseline, ref, beta)` this is exactly `ref=teacher, beta=1.0` with all task rewards 0:
  its shaping `r = r - beta * (policy.token_logprobs(t).sum() - ref.token_logprobs(t).sum())` → an unbiased estimate of −∇KL_seq (ascent direction,
  `Policy.step` does θ += lr·grad); verify against `rlcore.pg.kl_seq(p, q, task)` by enumeration on a `SeqTask`. Per-token credit with causality:
  ∇ = E[Σ_t ∇log π(y_t|·)·Σ_{t'≥t} (−r_{t'})]; Tinker's `kl_discount_factor=0` keeps only t' = t — lower variance, biased w.r.t. the sequence KL
  (its README: discounting "we generally do not observe … to improve performance").
- Compute: training ≈ 6·N·D FLOPs (transformer primer `00-foundations/transformers/docs/transformer-primer.md` §6.2 l.167: "C ≈ 6·N·D … 2 per parameter per
  token for the forward pass, 4 for the backward"); a frozen teacher forward = 2·N_T·D; LoRA ≈ 4·N·D (no weight grads — textbook approx., verify).
  MFU = achieved FLOP/s ÷ peak. Wilson interval (repo code, `memcore/harness.py: wilson_interval`, = `agentlab.evals.gate.wilson_interval`,
  `thinklab.thinking.ttc.wilson_interval`): centre (p + z²/2n)/(1 + z²/n), half z·√(p(1−p)/n + z²/4n²)/(1 + z²/n): 30/60 → (0.3773, 0.6227); 170/200 →
  (0.7939, 0.8929); 0/20 → (0, 0.1611). Unbiased pass@k: rl-sheet §8.

## 14. Serving economics (computed here with the repo's roofline — the core's tests should reproduce the method, not hard-code these)
- Reproduction check: `roofline.llm.decode(PRESETS["llama-3.1-8b"], get("h100-sxm"), best_batch_under_itl(…, 2048, 0.010), 2048)` → batch 68, 9.93 ms,
  6,847 tok/s, `cost_per_million_tokens(11, 6847)` = $0.446, at 60 % $0.744 — matches 01 PRIMER §8.1's table (l.636). §3.3: 4.52 ms/221 tok/s (H100),
  50.5 ms/19.8 tok/s (L4) for Llama-3.1-8B batch 1 at 1K.
- **T4 needs `precision="fp16"`** in `roofline.llm.decode/prefill` (`KeyError: 'NVIDIA T4 has no bf16 path'`). Batch-1 decode at 1K context
  (bound, ideal bandwidth): T4 (fp16) Qwen2.5-0.5B 3.127 ms (320 tok/s), Qwen2.5-1.5B 9.739 ms (103), Qwen3-0.6B 4.092 ms (244), Qwen3-1.7B 11.120 ms (90),
  Qwen3-4B 25.612 ms (39); L4: 3.335 / 10.388 / 4.365 / 11.862 / 27.319 ms; H100: 0.299 / 0.930 / 0.391 / 1.062 / 2.446 ms, Qwen2.5-32B 19.175 ms (52 tok/s).
- H100, 2K context, one GPU, bf16: Qwen2.5-32B cannot meet a 10 ms ITL at any batch (batch 1 = 19.2 ms → `best_batch_under_itl` returns 0 and
  `cost_per_million_tokens` divides by zero — guard it); HBM caps it at **12** sequences (21.02 ms, 571 tok/s → **$5.352/M** at $11/GPU-h, $1.800 at
  $3.7 Spot). At the same 30 ms ITL: Qwen2.5-1.5B batch 1,173 (HBM cap) 54,575 tok/s → $0.0560/M; Qwen2.5-0.5B 2,821 → 131,218 tok/s → $0.0233/M.
  32B → 1.5B student ≈ **96×** cheaper per output token on this bound (weights 21× smaller plus 9× smaller KV/token → far bigger batches).
- Fixed cost of distillation, worked: 100k prompts × 2,000 teacher tokens = 2e8 tokens; at gemini-3.5-flash output $9.00/M (`scalelab` PRICES, checked
  5 Sep 2026) = **$1,800**; self-hosted 32B at the roofline bound $5.352/M = $1,070 (Spot $360). Student SFT: 6 × 1.5e9 × 2e8 = 1.8e18 FLOPs; H100 bf16
  989.4 TFLOP/s (`roofline.specs` `h100-sxm`) at MFU 0.4 → 4,548 s = 1.263 GPU-h = **$13.90** at $11 ($4.67 Spot); MFU 0.3 → 1.685 GPU-h, $18.53.
  → the teacher's tokens, not the student's training, dominate the fixed cost.
- Qwen3 Table 21: on-policy distillation 1,800 GPU-h vs RL 17,920 (§6).

## 15. Repo material to reuse (exact names)
- `minengine.spec` (`04-inference-engine/serving-engine/mini-engine-core/minengine/spec.py`): `acceptance_rate(p, q)` = `np.minimum(p, q).sum()`;
  `expected_tokens(alpha, k)` = `k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)`; `speedup(alpha, k, c)` = `expected_tokens / (k * c + 1)`;
  `best_k(alpha, c, k_max=16)`; `verify(p, q, draft, rng)`; `speculative_generate(target_probs, prompt, max_new_tokens, k=4, draft_probs=None, …)`.
  serving-engine PRIMER §7 (l.408+): p = (0.5, 0.3, 0.15, 0.05), q = (0.2, 0.2, 0.2, 0.4) → α 0.6; α 0.8, k 4 → 3.36; α 0.8, c 0.1 → best k 6 at 2.47×;
  notebook 05 tiny target/draft α ≈ 0.72; greedy-draft paragraph l.446–452; proposer table ("draft model … a small model with the **same tokenizer**").
- `roofline` (`01-…/roofline-core/roofline/`): `cost.cost_per_million_tokens(price_per_gpu_hour, tokens_per_s, utilisation=1.0, n_gpus=1)`,
  `cost.utilisation(load_profile)`; `llm.ModelConfig(name, n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab, gated_mlp=True,
  tied_embeddings=False, n_experts=0, top_k=0)` with `.params()`, `.kv_bytes_per_token(kv_bytes=2)`; `llm.PRESETS` (`"qwen2.5-1.5b"`, `"llama-3.1-8b"`,
  `"llama-3.1-70b"`, `"mixtral-8x7b"`, `"qwen3-30b-a3b"`); `llm.decode(model, device, batch, context, *, weight_bytes=2, kv_bytes=2, precision="bf16", …)`
  → `Step` (`.time`, `.tokens_per_s`); `llm.prefill(model, device, prompt_len, batch=1, …)`; `llm.max_batch_by_memory(model, device, context, …, reserve=0.10)`;
  `llm.best_batch_under_itl(model, device, context, itl_s, **kw)`; `specs.get("t4" | "l4" | "h100-sxm" | …)`. 01 PRIMER §3 (l.146), §3.3 (l.185), §8.1 (l.624).
- `capacity.py` (`00-foundations/gpu-capacity-planning/`): see rl-sheet §11 (`ModelSpec`, `weight_memory_gb`, `kv_per_token_kb`, `decode_step_ms`,
  `decode_tok_s_single`, `prefill_flops(active_b, prompt_tokens, model=None)` = 2·P·S, `ttft_s`, `GPUS["H100"]` 80 GB 3.35 TB/s 990/1979) — GB = 1e9.
- `quantcore.eval` (`04-…/quantization/quant-core/quantcore/eval.py`): `kl(ref_logits, test_logits)` = mean over positions of KL(p_ref ‖ p_test) nats;
  `compare(ref_logits, test_logits, labels=None)` → `{kl, top1, n, acc_ref, acc, lost, gained, stderr, diff_stderr, paired_z, ppl}`;
  `accuracy_stderr(p, n)`; `diff_stderr`; `paired_z(lost, gained)`; `within_budget(report, max_kl=0.01, max_drop=None)`. **`argmax_agreement` is defined in
  `quantcore.granularity`** (imported by `eval`). Quantization PRIMER §7 (l.487; QAD l.492–495: "trains the quantized model to match the full-precision
  model's outputs rather than labels", 500 QAD iterations, 67 → 22 GiB, verify), §8 (l.507), §10 (l.658).
- `rlcore` (`00-foundations/rl-and-thinking-models/rl-core/rlcore/`): `pg.advantages(rewards, baseline="mean"|"none"|"loo")`, `pg.reinforce_grad(policy,
  trajs, baseline="mean", ref=None, beta=0.0, entropy_coef=0.0)`, `pg.sft_step(policy, trajs, lr=1.0)` (returns mean NLL), `pg.train_reinforce(policy,
  task, rng, steps=200, batch=16, lr=1.0, …)`, `pg.kl_seq(p, q, task)`, `pg.expected(policy, task, fn)`; `policy.Policy(n_states, n_actions, theta)` with
  `.sample(task, rng, n, prompt)`, `.token_logprobs(traj)`, `.grad_logprob(traj, weights=None)` (per-token weights w_t: row s_t += w_t(onehot − π)),
  `.step(grad, lr)` (ascent), `.stop_probs(task)`; `tasks.SeqTask(kind="brackets", length=8, verifier="true")`, `tasks.ThinkTask(e0=0.8, q=0.1,
  max_think=32, cost=0.0, prompts=None, force_answer=False)` with `.accuracy(L)` = 1 − e0(1−q)^L, `.expected(stop_probs)`, `.length_distribution`,
  `.optimal_length`. rl PRIMER §5 l.409–414 (distill numbers + the toy: SFT on 1,000 traces of the RL-trained teacher → 0.848 vs teacher 0.855, mean
  thinking 11.1 tokens), §7 l.637–641 (cost per correct: thinking everywhere 0.934 at $0.037853, only on hard 0.920 at $0.016859, never 0.755 at
  $0.009278; `workload.cost_per_correct()`), glossary l.789 "Distillation (of traces) | SFT of a smaller model on a stronger model's outputs".
- `scalelab` (`06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/`): `capacity.PRICES` ("checked 5 Sep 2026"; USD/1M in/out/cached):
  `gemini-3.5-flash` 1.50/9.00/0.15, `gemini-3.5-flash-lite` 0.30/2.50/0.03, `gemini-3.1-flash-lite` 0.25/1.50/0.025, `gemini-3.8-flash` 0.75/3.75/0.075
  (introductory to 31 Dec 2026), `gemini-3.1-pro-preview` 2.00/12.00/0.20; `capacity.cost_per_call(model, input_tokens, output_tokens, cached_tokens=0)`;
  `model.cost_per_call(...)` dispatches to `capacity` or `mistral.cost_per_call(..., tier="global")`.
- Wilson: `07-…/agent-memory/memory-core/memcore/harness.py: wilson_interval(passes, n, z=1.96)`; 07 platform lab
  `07-…/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py` (`agentlab.evals.gate.wilson_interval`).
- thinking-lab: `thinklab/thinking/evalset.py` — `Problem(id, kind, difficulty, question, answer)` with `.prompt` (= question + SUFFIX "Please reason step
  by step, and put your final answer within \\boxed{}."), `.messages()` = `[{"role": "user", "content": prompt}]`; `KINDS = ("arith", "digitsum", "days",
  "order", "count")`, difficulty 1–4, scored on the final `\boxed{}` in content; "60 problems near 50% carries a ±12-point 95% interval".
  `deploy/any-gpu/README.md`: model table (T4: Qwen3-0.6B 1.19 GB, Qwen3-1.7B 3.44 GB, R1-Distill-Qwen-1.5B 3.55 GB, Qwen3-4B 8.04 GB; Qwen3-8B does not
  fit a T4 fp16), Colab `pip install -q "vllm==0.30.0"`, `--dtype half --gpu-memory-utilization 0.85`, `thinklab.rollout.trl_grpo_config("T4")`.
- Model-landscape primer (`00-foundations/model-landscape/open-weight-llms-primer.md`): l.38 "Distilled small models are a large share of the ecosystem"; l.193
  Inkling-Small (276B/12B active) "distilled from an Inkling checkpoint and then given two further weeks of agentic-coding RL", beats its parent on
  reasoning/agentic rows, loses on factuality (SimpleQA Verified 20.6 vs 43.9) — "RL on a smaller student buys agentic capability and costs you world knowledge".
- COMPUTE.md §4/§5.2 GPU prices (2026-09-26, verify) = FACTS.md "GCP compute"/"Non-GCP compute" (T4 ~$0.35–0.55, L4 ~$0.70, H100 ~$11/GPU-h, Spot ~$3.7).
- MoE primer: §5 l.341 "MoE at inference: which experts a step touches", §7 l.593 "Sizing and cost". Transformer primer §6.2/§6.3 l.161–175.

## Model and tool ids to use
- **T0** (no download): numpy teacher/student families in `distillcore`; the lab's tiny torch decoders (torch 2.14 CPU is installed here; lazy import);
  a fake OpenAI-compatible teacher emitting `prompt_logprobs`/`logprobs` and vLLM-named `usage`; bundled curves/traces labelled "illustrative".
- **T1 free T4 (fp16, `--dtype half`)**: teacher `Qwen/Qwen2.5-1.5B-Instruct` (3.09 GB) → student `Qwen/Qwen2.5-0.5B-Instruct` (0.99 GB) — same
  vocab_size 151,936, same template, TRL's own DistillationTrainer quick-start pair, Apache-2.0: **the default pair**. Qwen3 pair: `Qwen/Qwen3-1.7B`
  (3.44 GB) → `Qwen/Qwen3-0.6B` (1.19 GB), 151,936 both; thinking on for traces (`--reasoning-parser qwen3`, T 0.6/top-p 0.95/top-k 20). Thinking teacher
  alternative `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (3.55 GB, MIT; R1 template, no system prompt, T 0.6) — SeqKD only into a Qwen student (template
  differs; do not logit-distil across). SFT full FT of 0.5B fits (7.9 GB + activations, predicted); logit KD/GKD with a 1.5B teacher on T4 → LoRA r=16 or
  B = 1–2 (predicted). Tools: `trl==1.14.0`, `transformers>=4.56.2`, `peft` (0.21.0), `vllm==0.30.0` (separate process/env), `lm-eval[vllm]==0.4.13`.
- **T1 rented 24 GB (L4/4090, bf16)**: target `Qwen/Qwen3-4B` (8.05 GB) with draft `Qwen/Qwen3-0.6B` via `--speculative-config '{"method": "draft_model",
  "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4}'` (equal vocab; optional unofficial `AngelSlim/Qwen3-4B_eagle3` with `"method": "eagle3"`,
  verify it loads in 0.30.0); a distilled draft = the 0.6B SFT'd on Qwen3-4B's own outputs; compare `vllm:spec_decode_*` counters. 1.5–1.7B student full FT
  in pure bf16 (12.4/13.8 GB + activations). Not: Qwen2.5-0.5B as a draft for Qwen2.5-7B (vocab 151,936 ≠ 152,064).
- **GCP**: teacher inference through the 04 serving lab's Cloud Run/GKE deploys (`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/`); no new Terraform.

## Pitfalls
1. `from trl import GKDTrainer` fails on 1.x: use `trl.experimental.gkd`; `DistillationTrainer`/`DistillationConfig` are `from trl import …` (1.14.0),
   the `trl.experimental.distillation` path warns. `GKDConfig.beta` default 0.5 vs `DistillationConfig.beta` 1.0; `max_new_tokens` vs `max_completion_length`.
2. GKD's `temperature` (0.9) only affects sampling — `compute_loss` runs the JSD at T = 1; no T² anywhere in TRL (ModelOpt does multiply by T²).
3. GKD `lmbda` is a per-batch coin flip (`random.random() <= lmbda`), and `seq_kd` only acts on the non-on-policy branch; sampling is top_k 0, top_p 1.
4. β endpoints: β = 0 forward KL(teacher‖student), β = 1 reverse KL(student‖teacher) — TRL docs say "approximates", code uses exact KLs, so the loss scale
   jumps ~1/β at the endpoints; interior values are ≤ ln 2.
5. Vocab checks compare config `vocab_size`: Qwen2.5 ≥7B (152,064) vs small Qwen (151,936) fails in GKD and in vLLM `draft_model`, though token ids match.
   Cross-tokenizer → TRL GOLD (`use_uld_loss=True`) or vLLM `use_heterogeneous_vocab` (greedy drafts only).
6. Templates: logit KD must feed identical token ids to both models (TRL renders with the student's `processing_class`); Qwen2.5's template injects a
   default system prompt, Qwen3's does not; R1-Distill uses `<｜User｜>`/`<｜Assistant｜>`. TRL's Qwen3 *training* template always renders a think block
   (empty when absent); the inference template drops earlier-turn thinking (rl-sheet §2). Qwen3 thinking traces in SFT data must go in
   `reasoning_content` or inside `<think>…</think>` in content, and non-thinking examples need the empty block.
7. T4: no bf16 — TRL `SFTConfig`/`GKDConfig`/`DistillationConfig`/`GRPOConfig` set `bf16=True` unless `fp16` is set; loading fp16 weights and training
   with `fp16=True` raises "Attempting to unscale FP16 gradients." (keep trainable params fp32). Qwen was trained in bf16 — watch for fp16 overflow/NaN.
8. Memory is dominated by `[B·S, 151,936]` logits (2.49 GB fp32 per 4,096 tokens); a naive KD loss holding teacher + student logits OOMs a T4 — chunk.
9. vLLM `--max-logprobs` default 20 (requests above it are rejected; `-1` = vocab, may OOM); `--logprobs-mode` default `raw_logprobs` (before
   temperature); `echo` + `max_tokens=0` generates 1 token; first `prompt_logprobs` entry is None. API teachers give top-k only → approximate KD.
10. vLLM spec decode: use `--speculative-config` JSON (old `--speculative-model` deprecated; `-sc`, `--spec-method/--spec-model/--spec-tokens` exist);
    `tensor_parallel_size` is rejected inside it (use `draft_tensor_parallel_size`); no `speculative_token_tree`; `method` defaults to `draft_model` when a
    model is given; default `draft_sample_method="greedy"` → acceptance = p(argmax q), not Σ min(p,q) (0.05 vs 0.6 on the primer's p, q).
11. vLLM's "Avg Draft acceptance rate" = accepted/drafted = (E−1)/k, not per-token α; α ≈ per-position rate at position 0; mean acceptance length includes the bonus token.
12. `roofline.llm.decode` on a T4 needs `precision="fp16"`; `best_batch_under_itl` returns 0 when batch 1 misses the ITL (32B on one H100 at 10 ms)
    and `cost_per_million_tokens(…, 0)` divides by zero.
13. lm-eval gsm8k is greedy with `max_gen_toks` 256 by default — wrong for thinking models (endless repetition, truncated traces): pass `--gen_kwargs`,
    `enable_thinking=True,think_end_token="</think>"`, `--apply_chat_template`; `--limit` → report stderr.
14. Licences: Llama outputs used to train a distributed model → name must start with "Llama"; DeepSeek-R1 MIT explicitly allows distillation; Qwen2.5-3B
    is not Apache-2.0 (unverified — avoid); hosted-API terms (unverified) — one dated, verify-marked paragraph, not legal advice.

## Unverified (collected)
- Shapes without a config here (huggingface.co blocked): Qwen3-1.7B h/ffn, R1-Distill-1.5B untied, Qwen2.5-3B (and its licence), Qwen2.5-32B untied; shared
  Qwen2.5/Qwen3 BPE merges. Pure-bf16 AdamW 8 B/p and every T4/24 GB fit (predictions); LoRA ≈ 4·N·D; Qwen3 fp16 training on Turing. Thinking Machines'
  compute multipliers; MiniLLM/DSS numbers; Gemma teachers; TinyBERT/MiniLM; hosted-API terms; the Qwen3 on-policy KL direction; EAGLE speedups off
  their 13B/2×3090 setting; `AngelSlim/Qwen3-4B_eagle3` under vLLM 0.30.0.
