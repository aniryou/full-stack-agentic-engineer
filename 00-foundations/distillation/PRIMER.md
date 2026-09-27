# Distillation: teaching a small model what a large one knows, and what the student saves in serving

*A primer for 00-foundations (module 00.6). Snapshot: September 2026. Product facts are dated and marked (verify);
every formula has a worked number and names the function in [`distill-core/`](distill-core/) (package
`distillcore`) that computes it. Toy numbers are exact (the output space is enumerated) or seeded runs of
tiny numpy models; serving numbers are a roofline bound — the arithmetic of layer 01's `roofline.llm` and
`roofline.cost`, reproduced in the core's tests — not measurements.*

This primer explains how a small model is trained to behave like a large one, and what that buys in serving:
why a student learns more per example from a teacher's distribution than from labels; which divergence to
minimise and what each one makes a too-small student do; distillation from the teacher's text (sequence-level)
and from the teacher's scores of the student's own samples (on-policy); distilling a thinking model's traces;
feature distillation, pruning and tokenizer mismatch; a draft model for speculative decoding as a student
whose metric is acceptance; how to measure a student; and the economics — the one-off bill against the
per-token saving. It builds on the [transformer primer](../transformers/docs/transformer-primer.md) (§6
training), the [capacity primer](../gpu-capacity-planning/PRIMER.md), the
[roofline primer](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (§3, §8), the
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) (§7) and the
[RL and thinking-models primer](../rl-and-thinking-models/PRIMER.md) (§1–§7), and does not repeat them. Every
concept is learnable at tier T0 with [`distill-core/`](distill-core/); [`distill-lab/`](distill-lab/) distils a
tiny transformer four ways in torch and a real 0.5–0.6B student with TRL and vLLM (T1).

---

## The one-minute version

A teacher's output is a distribution, and its **soft targets** p = softmax(z/T) say how wrong each wrong answer
is — information a hard label does not carry, so a student learns more per example from them than a same-size
model trained from scratch on the same tokens. The classic loss is α·T²·KL(p_T ‖ q_T) + (1 − α)·CE, whose
gradient on the student's logits is T·(q_T − p_T). When the student cannot copy the teacher, the **divergence**
decides what it gets wrong: forward KL makes it cover every mode (and fill the gaps between them), reverse KL
makes it commit to the modes it can fit. **Sequence-level distillation** is SFT on text the teacher wrote — it
needs only samples, and it pays for every token the verifier throws away; its flaw is **exposure bias**, since
the student trains on the teacher's prefixes and decodes on its own. **On-policy distillation** samples from the
student and asks the teacher to score every token: RL with a dense reward r_t = log π_T − log π_S, one teacher
forward pass per token. Distilling a **thinking model** copies its procedure and its thinking-length
distribution, not its knowledge. A **draft model** is a student whose metric is acceptance, α = 1 − TV. Measure
students by agreement *and* by task accuracy with intervals, per slice. The payoff is serving: a 1.5B student of
a 32B teacher is ~96× cheaper per token on the roofline, against a one-off bill dominated by the teacher's
tokens — so break-even is days or months depending on volume, and a cascade sits in between.

---

## 1. Why distil

**The cost argument, from the roofline.** Decode streams the weights and each sequence's KV cache every step
([roofline primer §3](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#3-llm-inference-on-the-roofline));
prefill costs about 2·N FLOPs per token. A student with a twentieth of the parameters streams a twentieth of the
weight bytes per decode step and needs a twentieth of the prefill FLOPs:

| Qwen2.5 (bf16) | Parameters | Weights | KV per token | Prefill per token | Batch-1 decode, 1K context, H100 | Sessions of 2K on one H100 |
|---|---|---|---|---|---|---|
| 32B (teacher) | 32.76 B | 65.53 GB | 262,144 B | 65.53 GFLOP | 19.175 ms | 12.06 |
| 1.5B (student) | 1.54 B | 3.09 GB | 28,672 B | 3.09 GFLOP | 0.930 ms | 1173.58 |

`distillcore.cost.Shape.params()` counts parameters as `roofline.llm.ModelConfig.params()` does (no norms or
biases); `cost.decode_step()` is `roofline.llm.decode()` for dense models; the last column is
`cost.sessions_per_gpu()`, the capacity primer's `max_concurrent_sessions` (usable HBM minus weights, over KV per
session, GB = 10⁹ bytes — [its formulas](../gpu-capacity-planning/PRIMER.md#the-formulas-all-of-capacitypy-in-one-page)).
The weights are 21.2× smaller and the KV per token 9.1× smaller, so the student fits ~100× the concurrent
sequences; §9 turns that into dollars with
[`roofline.cost.cost_per_million_tokens`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#8-the-cost-of-a-token).

**Three routes to a small model.** Train it small from scratch; prune a large one and repair it; or distil —
train it on a teacher's outputs. The routes combine (prune, then distil, §6). Why does a distilled student beat
a same-size model trained from scratch on the *same* tokens? A hard label is one sample from the teacher's
distribution; the soft target is the distribution. In the core's toy language (after tokens a, b the next is
(a + b) mod 11 with probability 0.8 and each neighbour with 0.1), a 16-unit student given the same contexts
reaches (notebook 01, `tinylm.train()` with `losses.hard_ce()` or `losses.kd()`):

| Examples (per context) | Hard labels: rule accuracy / KL to the truth | Soft targets: rule accuracy / KL |
|---|---|---|
| 242 (2) | 0.678 / 3.472 nats | 0.876 / 0.617 nats |
| 605 (5) | 0.917 / 1.303 nats | 0.992 / 0.108 nats |

**Three families for language models.** What each needs from the teacher decides where it can run:

| Family | Trains on | Needs from the teacher | Where it runs |
|---|---|---|---|
| Logit (soft-label) KD | teacher's full distribution at every position of some text | logits over the whole vocabulary: its weights, same tokenizer | your own training job (§2) |
| Sequence-level (SeqKD) | text the teacher generated, as SFT | samples only | any API or server (§3, §5) |
| On-policy (GKD) | the student's own samples, scored per token | log-probabilities of given text (a forward pass) | a served teacher that returns prompt logprobs (§4) |

**What distillation cannot do.** It cannot exceed the student's capacity: an 8-unit student of the toy language
tops out at 0.934 rule accuracy with the *exact* distribution as its target, where 16 units reach 1.000
(`tinylm.fit_language()`, notebook 01). It cannot transfer knowledge the teacher never verbalises: a student
learns only what the teacher's outputs (or logits) on the training prompts reveal. And it cannot elicit behaviours
the data never exercises: a student trained only on the teacher's text for twenty prompts is right on 0 of 101
other prompts (§8). The teacher–student gap is a capacity gap plus a coverage gap.

**Licences and terms, briefly (2026-09-27, verify; not legal advice).** Training on a model's outputs is governed
by the licence of the weights or the terms of the API. Qwen2.5 (0.5B/1.5B/7B) and all Qwen3 open weights are
Apache 2.0; DeepSeek-R1 is MIT and its README says the series allows "distillation for training other LLMs"; the
Llama 3.1 and 3.2 licences require that a model trained on Llama outputs and distributed include "Llama" at the
beginning of its name; some hosted-API terms restrict using outputs to build competing models — read the ones
you use. The student inherits the base model's licence too (the R1 Qwen distills keep Qwen's Apache 2.0).

**Where this starts.** The
[RL primer §1](../rl-and-thinking-models/PRIMER.md#1-from-pretraining-to-post-training) places distillation in
post-training (SFT on a stronger model's outputs) and
[§5](../rl-and-thinking-models/PRIMER.md#5-thinking-models) reports what it did for R1 and Qwen3. This primer is
the mechanism behind those lines.

## 2. Soft targets, temperature and the choice of divergence

**Soft targets and dark knowledge** (Hinton, Vinyals & Dean 2015). A teacher's logits z give
p_T = softmax(z/T). At T = 1 a confident teacher's small probabilities are tiny; raising T flattens the
distribution and makes the ranking of wrong answers carry weight. Five tokens, teacher z = (4, 3, 1, 0, −1),
student v = (3, 3.5, 0, 0.5, −1) (`losses.softmax()`):

| | token 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| teacher, T = 1 | 0.6931 | 0.2550 | 0.0345 | 0.0127 | 0.0047 |
| teacher, T = 2 | 0.4885 | 0.2963 | 0.1090 | 0.0661 | 0.0401 |
| student, T = 1 | 0.3573 | 0.5891 | 0.0178 | 0.0293 | 0.0065 |

The hard label says "token 0"; the soft target also says token 1 is a far better second choice than tokens 3
and 4 — the dark knowledge. KL(p ‖ q) = 0.25650 nats at T = 1 (`losses.kl()`).

**The loss and its gradient.**

```
L = α·T²·KL(p_T ‖ q_T) + (1 − α)·CE(y, q_1)                                   losses.hinton()
∂KL(p_T ‖ q_T)/∂v_i = (q_T,i − p_T,i) / T        so   ∂(T²·KL)/∂v_i = T·(q_T,i − p_T,i)     losses.kd()
```

The derivation is one line: KL = Σ p log p − Σ p log q_T and ∂log q_T,j/∂v_i = (1[i = j] − q_T,i)/T. At T = 2 the
gradient of the KL is (−0.073543, 0.071047, −0.016410, 0.015853, 0.003053), which finite differences confirm (the
core's tests). Soft-target cross-entropy (`losses.soft_ce()`) has the same gradient; a hard label is the same loss
with p one-hot (`losses.hard_ce()`).

**Why T².** The KL itself shrinks roughly as 1/T² when T grows, and so does its gradient, so without the factor a
soft term at T = 4 is drowned by the hard-label term and α stops meaning what it says (`losses.kd(scale=False)`):

| T | 1 | 2 | 4 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|
| KL(p_T ‖ q_T) | 0.25650 | 0.06639 | 0.01596 | 0.00242 | 0.00002 | 0.00000 |
| T²·KL | 0.25650 | 0.26558 | 0.25542 | 0.24196 | 0.23129 | 0.23013 |

**The high-T limit is logit matching.** Expanding softmax(v/T) ≈ 1/N + (v_i − v̄)/(NT) gives
T²·KL → (1/2N)·Σ((v_i − v̄) − (z_i − z̄))², mean-squared error on centred logits (Caruana's model compression):
0.23000 for the example (`losses.logit_mse()`). TRL's distillation trainers run the divergence at T = 1 and apply
no T²; NVIDIA ModelOpt's `LogitsDistillationLoss` multiplies by T² (verify). Treat T and α as knobs to sweep.

**Why a soft target is worth more per example.** The gradient from a sampled hard label is onehot(y) − q; its
expectation over y ~ p is p − q, the soft-target gradient, and its extra variance is exactly 1 − Σp²
(`losses.label_noise()`): 0.34 per example for the toy language's 0.8/0.1/0.1. The soft target delivers the
expectation with no sampling noise — the §1 table is that variance at work.

**Forward against reverse KL.** A student too small to match the teacher must choose what to get wrong:

```
forward KL(p ‖ q) = Σ p·log(p/q)   paid wherever the teacher has mass the student lacks → mode-covering    (SFT, KD)
reverse KL(q ‖ p) = Σ q·log(q/p)   paid wherever the student has mass the teacher lacks → mode-seeking     (on-policy)
JSD(β) = β·KL(p ‖ m) + (1 − β)·KL(q ‖ m),  m = β·p + (1 − β)·q                                  divergences.jsd()
```

MiniLLM's argument (Gu et al.) is that for generation the reverse direction is right: a forward-KL student
"overestimates the low-probability regions of the teacher distribution" and samples text the teacher would never
write. The core fits a one-bump student (a discretised Gaussian) to a two-bump teacher on 11 tokens, modes at 2
and 8 (`divergences.fit_bump()`, notebook 02):

| Fit | μ, s | Divergence | Student mass where the teacher gives < 1% | Teacher mass left uncovered |
|---|---|---|---|---|
| forward KL | 5.0, 8.6 | 0.6425 | 0.454 | 0.00 |
| JSD, β = 0.5 | 5.0, 9.9 | 0.1789 | 0.454 | 0.00 |
| reverse KL | 2.0, 0.7 | 0.6931 = ln 2 | 0.019 | 0.51 |

The forward fit is nearly flat and puts 45.4% of its samples where the teacher almost never goes — mostly between
the modes; the reverse fit commits to one mode and gives up half the teacher's mass. With this family the JSD's fit
flips from covering to seeking between β = 0.6 and β = 0.7. For classification the choice barely matters (only
the argmax is used); for generation every sampled token conditions the next, so where the student puts its
low-probability mass is what it writes.

**TRL's convention.** `generalized_jsd_loss(student_logits, teacher_logits, beta)` computes exactly the table's
three rows: β = 0 is KL(teacher ‖ student), β = 1 is KL(student ‖ teacher), both exact at the endpoints, and the
interior is the mixture JSD (≤ ln 2) — so the loss scale jumps by ~1/β at the ends. On the five-token example:
0.256501 at β = 0, 0.023026 at 0.1, 0.064431 at 0.5, 0.024067 at 0.9, 0.271424 at β = 1 (`losses.gkd()`).
**Total variation** TV = ½Σ|p − q| is the fourth measure worth knowing: 1 − TV is a draft's acceptance rate (§7).
This is the same KL that RL uses as its penalty to a reference model
([RL primer §2](../rl-and-thinking-models/PRIMER.md#2-policy-gradients-over-token-sequences)), pointed the other
way: RL keeps the policy near the reference; distillation pulls it onto the teacher.

## 3. Sequence-level distillation: learning from the teacher's outputs

**SeqKD is SFT on text the teacher wrote** (Kim & Rush 2016). Generate completions for a set of prompts, keep the
good ones, and fine-tune the student on them with the ordinary cross-entropy (`seqkd.sft()`). It needs no logits,
so it works through any API, and it is the recipe behind the R1 distills and the off-policy stage of Qwen3's
strong-to-weak distillation — the recipe and its numbers are in the
[RL primer §1 and §5](../rl-and-thinking-models/PRIMER.md#5-thinking-models).

**The pipeline and its bookkeeping** (`seqkd.pipeline()`): generate n samples per prompt at a temperature →
verify → deduplicate → SFT. Every stage has a price. Twenty prompts × 8 samples × 12 tokens from the toy teacher:

| Sampling | Generated | Pass the verifier | Unique | Teacher tokens paid | Tokens used |
|---|---|---|---|---|---|
| T = 0 (greedy) | 160 | 160 | 20 | 1,920 | 240 |
| T = 1 | 160 | 16 | 11 | 1,920 | 132 |

At T = 0 every sample of a prompt is the same; at T = 1 a teacher that follows the rule with probability 0.8 per
token passes a 12-token check (0.8)^12 = 0.069 of the time, so each kept sample costs about 175 teacher tokens.
The data budget is therefore prompts × samples per prompt × tokens per sample at the teacher's price per token —
not the size of the kept set. Price it with the 06 layer's model
([scaling primer §3.4](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md#34-cost-per-conversation);
`cost.api_cost()` reproduces its `cost_per_call`).

**Filtering, deduplication, decontamination.** Filter by a verifier (rejection sampling — the same machinery as
best-of-n, [RL primer §6](../rl-and-thinking-models/PRIMER.md#6-test-time-compute)) and by length (§5);
deduplicate; and remove anything that overlaps your eval sets, because a teacher that saw a benchmark can write it
into the student's training data (§8).

**Exposure bias, measured.** The student trains on the teacher's prefixes and decodes on its own. Two 16-unit
students are trained on the greedy teacher's continuations of twenty prompts; each is scored by whether its top
token follows the rule, given the teacher's prefixes and given its own samples at T = 1 (`seqkd.exposure_bias()`,
notebook 02):

| Student | On the teacher's prefixes | Own prefixes, position 1 | position 4 | position 12 | Entropy (teacher 0.64) |
|---|---|---|---|---|---|
| SFT on the greedy text | 1.000 | 1.000 | 0.999 | 0.999 | 0.009 |
| supervised KD on the greedy text (GKD λ = 0) | 1.000 | 1.000 | 0.788 | 0.774 | 0.652 |

The SFT student never slips because it never varies: trained on one greedy output per prompt, it copied that
output, not the teacher. The KD student matches the teacher's distribution on the teacher's text, so at T = 1 it
wanders off the rule's cycle as often as the teacher would — into contexts no training example covered — and
from the first slip its accuracy falls to 0.788 by position 4. Per-token errors now compound with length. Two
cures: cover more of the states the student will visit (sample the teacher at T > 0, more samples per prompt,
more tokens paid), or train on the student's own states (§4).

**Rationales as extra supervision.** "Distilling step-by-step" (Hsieh et al.) trains a small T5 student to
predict both the label and the teacher's rationale, loss = α·label + (1 − α)·rationale with α = 0.5 recommended
(its README; verify); reasoning traces (§5) are the generative version of the same idea.

**Worked: the fixed cost of a SeqKD run.** 100,000 prompts × one 2,000-token completion = 2 × 10⁸ teacher tokens;
at $9.00 per million output tokens (the 06 lab's Gemini 3.5 Flash price, dated 2026-09-05 there, verify) that is
$1,800; self-hosting the 32B at its §9 roofline cost makes it $1,070.42. SFT of the 1.5B student for one epoch is
6·N·D = 1.85 × 10¹⁸ FLOPs ([transformer primer §6.2](../transformers/docs/transformer-primer.md#62-what-scale-means)),
1.300 GPU-hours on an H100 at 40% MFU, $14.30 at $11/GPU-hour (`cost.fixed_cost()`, `cost.training_flops()`,
`cost.gpu_hours()`). The teacher's tokens are the bill.

## 4. On-policy distillation

**The method** (Agarwal et al., GKD; Gu et al., MiniLLM). Sample a continuation from the *student*; ask the
teacher for its distribution (or just its log-probability of each sampled token) at every position; minimise a
divergence there. The student learns exactly on the prefixes it will produce, so exposure bias has nowhere to
hide. TRL's `GKDTrainer` mixes the data sources per batch — with probability λ the student generates (on-policy);
otherwise, with `seq_kd=True`, the teacher generates; otherwise the dataset's completion is used (supervised KD) —
and minimises JSD(β) (`onpolicy.gkd_train()` implements the same loop).

**Exposure bias removed.** Continue the §3 KD student with GKD at λ = 1 for 300 steps (notebook 02):

| Continue with | Own prefixes, position 4 | position 12 | All 121 contexts |
|---|---|---|---|
| (nothing: supervised KD only) | 0.788 | 0.774 | 0.198 |
| GKD λ = 1, β = 0 (forward) | 0.994 | 0.984 | 0.942 |
| GKD λ = 1, β = 0.5 | 0.957 | 0.831 | 0.636 |
| GKD λ = 1, β = 1 (reverse) | 0.930 | 0.754 | 0.504 |

On-policy data fixes it; the divergence decides how fast. The reverse end is slow here because the student starts
confidently wrong off the cycle, where reverse KL's gradient q_i·(log q_i − log p_i − KL) nearly vanishes: for a
student with almost all its mass on a token the teacher gives 0.0001, the reverse-KL gradient is 0.0055 of the
forward one (notebook 02, exercise 2.3; `losses.gkd()`). Start reverse-KL distillation from an SFT or KD student —
as the recipes do — or mix in forward KL; TRL's docs put it as "the optimal beta varied depending on the task".

**On-policy distillation is RL with a dense reward.** Reverse KL over whole sequences is
KL(π_S ‖ π_T) = E_{y~π_S}[log π_S(y) − log π_T(y)], and because E[∇log π_S] = 0 its gradient is REINFORCE:

```
−∇KL(π_S ‖ π_T) = E_{y~π_S}[ R(y)·∇log π_S(y) ],   R(y) = Σ_t r_t,   r_t = log π_T(y_t | y_<t) − log π_S(y_t | y_<t)
```

That is [RL primer §2](../rl-and-thinking-models/PRIMER.md#2-policy-gradients-over-token-sequences)'s KL penalty
with the teacher as the reference and no task reward — `rlcore.pg.reinforce_grad(policy, trajs, "mean",
ref=teacher, beta=1.0)` with zero rewards; `onpolicy.advantages()` uses the same sign, batch-mean baseline and 1/N
average, and the core's test checks it against rlcore's function on rlcore's own bracket task. Where GRPO's
verifier gives one number per sequence
([RL primer §4](../rl-and-thinking-models/PRIMER.md#4-rl-with-verifiable-rewards-and-grpo)), every token gets its
own. The core checks the identity exactly: over all 125 three-token continuations of a 5-token language, the
estimator's expectation (`onpolicy.exact_pg_grad()`) equals −∇KL by finite differences of `onpolicy.exact_seq_kl()`
to 9 digits. Tinker's recipe uses the per-token form — token t's advantage is r_t alone
(`kl_discount_factor=0`, verify) — which is what GKD's per-position β = 1 loss equals in expectation
(`onpolicy.token_pg_identity()`); on the same enumerable task that estimator is biased for the sequence KL (its
expectation differs by 31% of the gradient's norm, because it ignores how a token changes the states after it)
and quieter (total variance 1.86 against 3.15 per batch of 64).

**Compute per student token.** In on-policy distillation the teacher never generates: it scores the student's tokens in one prefill-shaped
forward pass, 2·N_T FLOPs per token, beside the student's 2·N_S to generate and 6·N_S to train
(`onpolicy.flops_per_prompt()`). For an 8B student and a 32B teacher on 4K-token completions, four scored samples
per prompt cost 2.10 × 10¹⁵ FLOPs against 4.19 × 10¹⁵ for GRPO's sixteen rollouts — and the dense signal needs
far fewer steps. Qwen3's report compares the two from the same off-policy-distilled 8B checkpoint: RL reached
AIME'24 67.6 in 17,920 GPU-hours, on-policy distillation 74.4 in 1,800 (verify). The tinker-cookbook recipe lists
SFT on OpenThoughts3 at AIME'24 ~65% and on-policy distillation after it at ~76.7% in 200 steps (verify). Both
compare against a teacher that already exists — its training cost is not in either number.

**The shared-tokenizer requirement.** Per-token scores need the teacher to read the student's tokens. TRL's GKD
raises if the two configs' `vocab_size` differ; Qwen2.5-7B and larger have 152,064 against 151,936 for the small
Qwen2.5 and all Qwen3 models, so a 0.5B cannot be GKD-distilled from a 7B even though the token ids match (verify;
§6 covers cross-tokenizer methods).

**TRL, concretely (1.14.0, verify).** `GKDConfig`/`GKDTrainer` are experimental — `from trl.experimental.gkd
import GKDConfig, GKDTrainer` — with defaults `lmbda=0.5`, `beta=0.5`, `temperature=0.9` (sampling only: the loss
runs at T = 1), `max_new_tokens=128`, `seq_kd=False`; λ is a per-batch coin flip. `DistillationTrainer` /
`DistillationConfig` are stable API (`from trl import …`): always on-policy, `beta=1.0` (reverse KL) by default,
`max_completion_length=512`, `temperature=1.0` applied to both logits, and its quick start is exactly
Qwen2.5-0.5B-Instruct ← Qwen2.5-1.5B-Instruct. A served teacher gives the needed log-probabilities through vLLM's
`prompt_logprobs` (capped by `--max-logprobs`, default 20; full-vocabulary distributions through an API are
impractical, so API teachers support the sampled-token reward, not full logit KD).

## 5. Distilling reasoning

**Traces are the data.** A thinking model's output — its reasoning and its answer — is the SFT target, so what a
small student inherits is the teacher's **procedure**: its format, its steps and how long it thinks. The core's toy
uses rlcore's ThinkTask formula: think L tokens, then answer, right with probability 1 − e0·(1 − q)^L (e0 = 0.8,
q = 0.1, at most 32 tokens). A policy is a stopping rule, and SFT on traces — maximum likelihood — has a closed form
for it: the fraction of traces that stopped at each length among those that reached it (`reasoning.LengthPolicy.fit()`).
The teacher is what 16,000 REINFORCE rollouts produced (`reasoning.reinforce()`, as in
[RL primer §2](../rl-and-thinking-models/PRIMER.md#2-policy-gradients-over-token-sequences)); 1,000 of its traces
distil into a student that copies it (`reasoning.distil()`, `LengthPolicy.expected()`, notebook 03):

| Policy | Accuracy | Mean thinking tokens | p90 | Tokens per correct answer |
|---|---|---|---|---|
| untrained student | 0.273 | 1.00 | 3 | 3.67 |
| teacher (RL, 16,000 rollouts) | 0.893 | 19.91 | 23 | 22.31 |
| student, SFT on 1,000 traces | 0.892 | 19.91 | 24 | 22.31 |

Nothing told the student how long to think; it copied that from the traces — so the serving workload of a thinking
model ([RL primer §7](../rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving): decode-heavy,
heavy-tailed, KV growing while it thinks) comes with the distillation. Fewer traces give a student that stops too
early: 10, 30, 100, 300 and 1,000 traces reach 0.710, 0.824, 0.864, 0.883 and 0.893 — length is learned from the
tail of the data.

**Filtering sets the accuracy–length trade.** From the same 1,000 traces:

| Keep | Traces kept | Accuracy | Mean length | Tokens per correct answer |
|---|---|---|---|---|
| all (plain SeqKD) | 1000 | 0.892 | 19.91 | 22.31 |
| correct only (rejection sampling) | 898 | 0.895 | 20.07 | 22.43 |
| correct and L ≤ 16 | 89 | 0.769 | 12.73 | 16.56 |
| correct and L ≤ 12 | 28 | 0.646 | 8.55 | 13.24 |

Correct traces are longer when thinking helps, so rejection sampling nudges thinking up; a length cap
(budget-aware distillation) buys a cheaper student at a price in accuracy — and discards traces the teacher was
paid to write. The cap is a product decision that the data makes for you.

**Distillation against RL on a small model.** R1's comparison — distilling the big model beat large-scale RL on the
same small base — is in the [RL primer §5](../rl-and-thinking-models/PRIMER.md#5-thinking-models). The toy version:
spend 1,000 samples on the student either way. SFT on 1,000 teacher traces reaches 0.892; REINFORCE on the student
with 1,000 rollouts reaches 0.445, with 4,000 0.699, with 16,000 0.884. A trace carries the whole behaviour; a
reward carries one bit. (The toy has no capability ceiling; R1's authors note that a small base with RL alone may
not reach distillation's result at all.)

**What does not transfer: knowledge.** Accuracy depends on the solver's own q. Give the student half the teacher's
q (0.05): it copies the teacher's 19.91-token thinking exactly and scores 0.706, not 0.893 — and its own best length
at a cost of 0.01 per token is 27.5 tokens, against the teacher's 20.2 (`ThinkToy.optimal_length()`). Distillation
copies the procedure the teacher needed, not the one the student needs; evaluate at the student's own best budget,
and consider a short RL stage after distillation. Long-tail facts are the same story at scale: the model-landscape
primer's Inkling-Small beat its parent on reasoning and lost on factuality
([§4.10](../model-landscape/open-weight-llms-primer.md#410-thinking-machines-lab-us)).

## 6. Feature distillation, pruning and vocabulary mismatch

**Matching features.** The encoder era distilled more than outputs: DistilBERT's "triple loss" (masked-LM,
distillation and a cosine loss on hidden states), TinyBERT's matching of attention maps and hidden states layer by
layer, MiniLM's matching of the last layer's self-attention distributions (verify). They need a layer mapping (which
student layer imitates which teacher layer) and projections where widths differ. Decoder LMs mostly distil logits
and sequences instead: the output distribution is what generation uses, it needs no mapping, and it works across
architectures (a dense student of an MoE teacher). Feature matching survives where the student is built from the
teacher — EAGLE's draft head reads the target's hidden states (§7).

**Pruning, then distillation.** Minitron prunes a trained model's embedding width, attention heads and MLP width
(and depth) by activation importance on a small calibration set, then repairs it with KD from the unpruned parent.
Its README reports Minitron 8B and 4B from Nemotron-4 15B with "up to 40x fewer training tokens per model", "compute
cost savings of 1.8x" for the family and "up to a 16% improvement in MMLU" over training from scratch (verify). The
core does it in miniature (`TinyLM.prune_width()`, notebook 01): the 64-unit teacher pruned to its 16 most active
units scores 0.355 rule accuracy; 100 KD steps from the parent take it to 0.868, where a fresh 16-unit student after
the same 100 steps reaches 0.843 — the surviving units already hold most of the table.

**Distillation at pretraining scale** (verify). Llama 3.2's 1B and 3B "incorporated logits from the Llama 3.1 8B
and 70B models into the pretraining stage … as token-level targets", and KD "was used after pruning to recover
performance" (model card); Gemma 2's 2B and 9B were trained with knowledge distillation, and Gemma 3's
instruction-tuned models were post-trained with KD and RL (transformers docs). Pretraining-scale logit KD needs the
teacher's forward pass over trillions of tokens — 2·N_T·D FLOPs on top of the student's 6·N_S·D — which is why it is
done by the teacher's owner.

**A dense student from an MoE teacher.** Nothing changes in the method; the teacher's cost does. An MoE teacher's
forward pass costs its *active* parameters per token, but its memory is its total, and its decode batch shape is
set by the experts a step touches
([MoE primer §5](../mixture-of-experts/PRIMER.md#5-moe-at-inference-which-experts-a-step-touches) and
[§7](../mixture-of-experts/PRIMER.md#7-sizing-and-cost)). Qwen3's small dense models were distilled from Qwen3-32B
and the Qwen3-235B-A22B MoE (verify).

**Different tokenizers.** Logit KD and per-token scoring need the same token ids and the same vocabulary size.
Across tokenizers: Universal Logit Distillation compares sorted probability vectors instead of aligned tokens, and
TRL's experimental GOLD trainer aligns text spans and merges their logits (`use_uld_loss`); vLLM's
`use_heterogeneous_vocab` lets a draft of another vocabulary propose greedily (all verify). Sequence-level
distillation sidesteps the problem entirely — text is text — which is one reason it dominates in practice.

**The checkpoint is an ordinary model.** A distilled student is served like any other model of its size — the
engine, quantization and routing of layers 04–06 apply unchanged, and they compound: quantizing a student
(quantization §10) multiplies the §9 savings.

## 7. A distilled draft for speculative decoding

**The draft is a student whose metric is acceptance.** Speculative decoding
([serving-engine primer §7](../../04-inference-engine/serving-engine/PRIMER.md#7-speculative-decoding)) accepts a
drafted token with probability min(1, p/q), so per position

```
α = Σ_v min(p(v), q(v)) = 1 − TV(p, q)                 draft.acceptance_rate()   (minengine.spec.acceptance_rate)
E[tokens per pass] = (1 − α^(k+1)) / (1 − α)            draft.expected_tokens()   (minengine.spec.expected_tokens)
speedup = E / (k·c + 1),  c = a draft step in target steps    draft.speedup()   (minengine.spec.speedup)
```

On that primer's own p = (0.5, 0.3, 0.15, 0.05) and q = (0.2, 0.2, 0.2, 0.4): α = 0.6, 2.3056 tokens per pass at
k = 4, and at c = 0.1 the best depth is k = 3 with a 1.6738× speedup (`draft.best_k()`); the core's tests
reproduce `minengine.spec` on these inputs and on that primer's α = 0.8 examples. A draft is good exactly when its
distribution is close to the target's **on the target's own text** — which is what distilling from the target
optimises. By Pinsker, TV ≤ √(KL/2): driving the KL down drives acceptance up.

**Distilled against off-the-shelf.** The core's target is a fine-tuned dialect of the toy language (its noise all
goes to one neighbour). Three 16-unit drafts, scored on the target's samples (`draft.acceptance_on_text()`,
notebook 04):

| Draft | α | Greedy acceptance | KL(p ‖ q) | Speedup at k = 4, c = 0.289 |
|---|---|---|---|---|
| off-the-shelf (trained on the base language) | 0.890 | 0.800 | 0.147 | 1.86× |
| SeqKD from the target (24,000 of its tokens) | 0.933 | 0.795 | 0.038 | 2.03× |
| logit KD from the target | 0.988 | 0.800 | 0.010 | 2.26× |

The off-the-shelf draft knows the family but not the fine-tune, and loses exactly where the target was changed;
samples estimate the target's distribution, soft targets hand it over. The same holds for real pairs: a same-family
small model drafts for a fine-tuned target worse than a draft trained on the target's own outputs, which is why
SpecForge's data preparation regenerates the assistant responses with the target "to better align the draft model
with the target model's output distribution" (verify).

**Greedy drafting caps acceptance.** vLLM's draft models propose their argmax by default
(`draft_sample_method="greedy"`, verify), and the acceptance is then p(argmax q), not Σ min(p, q): 0.05 against 0.6
on the primer's p and q (`draft.greedy_acceptance()`). When the target samples at T = 1, greedy acceptance can never
exceed the target's top-token probability — 0.800 for every draft above, and for a perfect one.

**Acceptance against draft size.** Logit-KD drafts of growing width against the same target (c = draft parameters ÷
target parameters, the memory-bound view of a decode step, `draft.draft_cost()`):

| Draft hidden units | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| c | 0.112 | 0.171 | 0.289 | 0.526 |
| α (logit KD) | 0.589 | 0.722 | 0.988 | 0.995 |
| speedup at k = 4 | 1.56× | 1.72× | 2.26× | 1.60× |

α saturates once the draft can hold the target; c keeps growing; the speedup peaks at the smallest draft that holds
the target's behaviour. Below that size the capacity gap, not the training data, sets α. For a real pair,
Qwen3-0.6B drafting for Qwen3-4B (same 151,936 vocabulary) has c ≈ 0.148 by weight bytes, so k = 4 gives 1.45×,
1.74× and 2.11× at α = 0.6, 0.7 and 0.8 — a model, before per-step overheads (measure c, as the serving primer says).

**Draft-as-student designs** (verify). EAGLE trains a one-layer head that extrapolates the target's
second-to-top-layer features; EAGLE-3 fuses low-, mid- and high-level target features and, in its training code,
minimises cross-entropy against the target's softmax over 7 unrolled steps — soft-target distillation from the
target. Its README reports 3× (EAGLE) and 5.6× (EAGLE-3) over vanilla decoding for a 13B Vicuna on 2×RTX 3090 in
fp16. Medusa adds decoding heads to the target itself (Medusa-1 trains only the heads; "self-distillation" when the
original data is unavailable), reporting 2.2–3.6×. vLLM 0.30.0 serves them through `--speculative-config` with
`"method"` one of `draft_model`, `eagle`, `eagle3`, `medusa`, `mtp`, `ngram` and others; `draft_model` requires
the draft's `vocab_size` to equal the target's (Qwen2.5-0.5B cannot draft for Qwen2.5-7B).

**Reading vLLM's counters** (`vllm:spec_decode_num_drafts`, `…_num_draft_tokens`, `…_num_accepted_tokens`,
`…_num_accepted_tokens_per_pos`, verify). "Mean acceptance length" = 1 + accepted ÷ drafts = E[tokens per pass]; α
is the position-0 rate; the logged "draft acceptance rate" is accepted ÷ drafted = (E − 1)/k — 0.3264 at α = 0.6,
k = 4 (`draft.vllm_view()`). Per-position rates that fall faster than α^(i+1) mean correlated acceptance, and the
i.i.d. formula over-predicts deep drafts. The lab measures all of this for a real pair under vLLM
([`distill-lab`](distill-lab/) notebook `04_a_distilled_draft_in_vllm`, T1).

## 8. Measuring a student

**Agreement with the teacher** needs no labels: mean KL(p_teacher ‖ p_student) per position, top-1 agreement, and
top-k overlap — agreement on the plausible set (`eval.kl()`, `eval.argmax_agreement()`, `eval.topk_overlap()`).
The definitions are those of [quantization §8](../../04-inference-engine/quantization/PRIMER.md#8-measuring-the-accuracy-you-pay)
(`quantcore.eval.kl`, `argmax_agreement`; reproduced in the core's tests) — a quantized model is a student too,
and quantization-aware distillation ([quantization §7](../../04-inference-engine/quantization/PRIMER.md#7-quantization-aware-training-and-qlora-in-brief))
uses exactly §2's loss. Over the toy's 121 contexts (notebook 05):

| Student | KL(teacher ‖ student) | Top-1 agreement | Top-3 overlap |
|---|---|---|---|
| KD on the teacher's text (§3) | 5.557 | 0.198 | 0.342 |
| + on-policy GKD (§4) | 0.185 | 0.942 | 0.923 |
| 8 units, KD on everything (§1's capacity gap) | 0.322 | 0.934 | 0.826 |

**Task accuracy with intervals, per slice.** What users feel is task accuracy, and a single number hides where a
student fails. Report it per slice with a Wilson interval — centre (p + z²/2n)/(1 + z²/n), half-width
z·√(p(1 − p)/n + z²/4n²)/(1 + z²/n) (`eval.wilson_interval()`, as the 07 platform lab's
[evals notebook](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/08_evals_trajectory_judge_gates.ipynb)
and [`memory-core`](../../07-application-agent-framework/agent-memory/memory-core/) compute it). The toy's items:
continue a prompt greedily for n tokens, correct if all follow the rule; "common" prompts are the 20 the students
were trained from, "rare" the other 101 (`eval.capability_gap()`):

| Student | Common, n = 2 or 8 | Rare, n = 2 | Rare, n = 8 |
|---|---|---|---|
| KD on the teacher's text | 20/20 (0.839–1.000) | 0.000 (0.000–0.037) | 0.000 (0.000–0.037) |
| + on-policy GKD | 20/20 (0.839–1.000) | 0.881 (0.804–0.931) | 0.683 (0.587–0.766) |
| 8 units, KD on everything | 20/20 (0.839–1.000) | 0.851 (0.769–0.908) | 0.525 (0.428–0.619) |

Every student is perfect on the slice a quick eval would use. The gap is in the tail — rare inputs, long outputs
(a long output visits many contexts, and an error rate of 6.6% per context compounds) — and 20 items cannot bound
anything tighter than 84–100%. Where students fail at scale follows the same pattern: long-tail facts and rare
knowledge, multi-step problems, instruction edge cases and safety behaviour the distillation data under-covers.
Compare student and teacher on the *same* items and count the flips (`eval.compare()`, McNemar's z as quantization
§8's `paired_z`).

**When agreement is the wrong metric.** A weak teacher (trained on only 605 tokens) is right on 0.950 of contexts.
A student trained on its samples *filtered by the verifier* is right on 0.975 — better than its teacher — while
agreeing with it less (KL 1.59, top-1 0.942) than a student of its unfiltered samples (0.934 right, KL 0.164, top-1
0.959). A release gate of "agree with the teacher" would ship the worse student. Gate on the task; use agreement as
the cheap regression signal.

**Two hazards of synthetic data.** Model collapse: training generation after generation on models' own outputs
narrows the distribution, losing its tails first — the §3 SFT student's entropy (0.009 against the teacher's 0.64)
is a one-step version. Contamination: a teacher that saw a benchmark can reproduce it in the distillation data, and
the student's score on that benchmark then measures memory, not skill — decontaminate the prompts and the outputs
(§3). For thinking students, evaluate with the full generation length: lm-eval's gsm8k defaults to greedy decoding
and 256 generated tokens, too few for a trace (verify; the lab's notebook 03).

## 9. The economics of a student

**Cost per token, teacher against student.** On one H100 at 2K context, the largest batch whose decode step meets a
30 ms ITL and fits in HBM (`cost.serving()`: `best_batch()`, `decode_step()`, `cost_per_million_tokens()`), at
~$11 per GPU-hour on demand (September 2026, verify —
[roofline primer §8](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#8-the-cost-of-a-token)):

| Model | Batch | Step | Tokens/s | $/M output tokens |
|---|---|---|---|---|
| Qwen2.5-32B (teacher) | 12 | 21.02 ms | 571 | $5.352 |
| Qwen2.5-1.5B (student) | 1173 | 21.49 ms | 54,575 | $0.0560 |
| Qwen2.5-0.5B (student) | 2821 | 21.50 ms | 131,218 | $0.0233 |

Both are HBM-bound at this ITL, and the student is 96× cheaper per token: 21× fewer weight bytes and 9× less KV per
token let it run 98× the batch at the same step time. At a 10 ms ITL the 32B cannot serve at all — batch 1 already
takes 19.3 ms — while the 1.5B still runs batch 517. The core's tests reproduce the roofline primer's §8.1 table
with the same functions (Llama-3.1-8B: batch 68, 9.93 ms, 6,847 tokens/s, $0.446 per million). These are bounds:
real engines reach a fraction of them, and the ratio is the number to carry.

**The fixed cost** (`cost.fixed_cost()`, §3's example): teacher generation $1,800 through an API at $9.00/M or
$1,070.42 self-hosted, plus the student's SFT, 1.300 GPU-hours ($14.30) at 40% MFU. The teacher's tokens are 99%
of it; evals, engineering and keeping a second model current are extra and not small.

**Break-even** (`cost.break_even()`) is that bill over the saving per token: with the self-hosted teacher,
$1,084.72 ÷ ($5.352 − $0.056 per million) — 4.1 days at 50 million output tokens a day ($264.81 a day saved), 41.0
days at 5 million, 204.8 days at 1 million — before evals and engineering, and longer than many models stay in
service. At low volume, distil for latency or for control, not for cost.

**The cascade.** Instead of replacing the teacher, route: the student answers, and a gate escalates what it would
get wrong (`cost.cascade()`). With 500-token answers at the costs above and illustrative accuracies — easy requests
(70%) student 0.95, teacher 0.97; hard ones (30%) student 0.30, teacher 0.85, the split the
[RL primer §7](../rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving) uses for its cost per correct
answer:

| Policy | To the teacher | Accuracy | $ per 1,000 correct answers |
|---|---|---|---|
| teacher only | 1.00 | 0.934 | 2.865 |
| student only | 0.00 | 0.755 | 0.037 |
| student first, gate catches 80% of hard, 10% false alarms | 0.31 | 0.888 | 0.965 |
| a perfect router up front | 0.30 | 0.920 | 0.894 |

The gate's recall on hard requests buys the accuracy; every false alarm is a full teacher call (22% of this
gate's bill). This is routing by cost at the
gateway ([06 scaling primer §3.4](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md#34-cost-per-conversation)
routes a third of calls to a Flash-Lite tier the same way).

**A decision table.**

| Situation | Prefer | Why |
|---|---|---|
| the large model is fine, the prompt or format is the problem | prompting / a system prompt | no training; move a long prompt into weights later (prompt distillation) only if its tokens dominate cost |
| a narrow task with labelled data, no stronger teacher | fine-tune the small model | nothing to distil from |
| a stronger teacher exists, high volume, a narrow-to-medium task | **distil** (SeqKD, then on-policy) | per-token saving × volume ≫ the teacher-token bill |
| the model is right-sized but memory- or bandwidth-bound | quantize ([quantization §10](../../04-inference-engine/quantization/PRIMER.md#10-choosing-a-scheme)) | a 2–4× byte saving, no training |
| an off-the-shelf small model already meets the easy slice | route (cascade) | no training; distil later if volume grows |
| latency, not cost, is the problem | a distilled draft (§7) | same outputs, fewer target passes |

## 10. Where to run it

| Tier | What runs | What it teaches |
|---|---|---|
| **T0** laptop / Colab CPU / CI, $0 | `distill-core` (numpy, seconds); the lab's tiny torch transformers — hard labels against logit KD, SeqKD and GKD — on a CPU; its fake OpenAI-compatible teacher (labelled simulated) and bundled traces (labelled illustrative) | every concept in this primer |
| **T1** free Colab/Kaggle T4 (fp16 only) | teacher completions from Qwen2.5-1.5B-Instruct in vLLM; SFT and logit KD of Qwen2.5-0.5B-Instruct with TRL; on-policy GKD; traces from Qwen3-1.7B or DeepSeek-R1-Distill-Qwen-1.5B | the real pipeline at small scale |
| **T1** a rented 24 GB GPU (an L4 or RTX 4090: RunPod or Vast.ai containers, Lambda VMs, a GCP `g2-standard-4` Spot L4) | Qwen3-4B as target with a Qwen3-0.6B draft (off-the-shelf, then distilled) under `--speculative-config`; 1.5–1.7B students | spec-decode acceptance and speedup, measured |
| **T3** GCP | teacher inference through the 04 serving lab's Cloud Run GPU or GKE deploys ([`vllm-serving-lab/deploy/gcp/`](../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/)); no new Terraform | the teacher as a served endpoint |

**Fitting a T4** (predicted from the fact sheet's arithmetic, 2026-09-27 — verify on hardware; the lab's
`hf/memory.py` computes them). Full fine-tuning with AdamW and an fp32 master copy is ~16 bytes per parameter: 7.90
GB for Qwen2.5-0.5B, which fits a T4's ~15 GiB with gradient checkpointing and short sequences; 24.7 GB for a 1.5B,
which does not. Logit KD adds the frozen teacher (1.5B in fp16: 3.09 GB) and its logits, and the logits dominate:
one fp32 `[tokens, 151,936]` tensor is 2.49 GB per 4,096 tokens — chunk the loss (TRL's distillation loss works in
256-position chunks) or use LoRA (r = 16 on every linear layer of the 0.5B: 8.8 M trainable parameters). A T4 has no
bf16: load the trainable model in fp32 and set `fp16=True, bf16=False` (TRL's configs default to bf16 unless
`fp16` is set; training fp16-loaded weights with `fp16=True` fails with "Attempting to unscale FP16 gradients").
Serve the teacher with `vllm serve … --dtype half`; the RL primer's
[§9](../rl-and-thinking-models/PRIMER.md#9-where-to-run-it) has the T4 vLLM recipe, and
[`COMPUTE.md`](../../COMPUTE.md) the prices and how to obtain each tier. GCP is one target, never a prerequisite.

---

## In a design review

**The two-minute walkthrough.** "We want a small model that behaves like our large one on this task, because decode
cost scales with the bytes a step streams: a 1.5B student of a 32B teacher serves ~100× the batch under the same
ITL and is ~96× cheaper per token on the roofline. We distil rather than train small from scratch because the
teacher's distribution carries far more per example than a label. The first stage is sequence-level: generate with
the teacher, verify, deduplicate, decontaminate, SFT — it works through any API, and its bill is teacher tokens,
including the ones the verifier throws away. SFT on teacher text has exposure bias — the student trains on the
teacher's prefixes and decodes on its own — so the second stage is on-policy: the student samples, the teacher
scores every token in one forward pass, and we minimise a divergence there; it is RL with a dense per-token
reward, needs a shared tokenizer, and starts from the SFT student because reverse KL stalls where the student is
confidently wrong. For a thinking model, traces copy the procedure and the thinking length — so we cap length in
the data if we need a cheaper student, and size serving for the tail we inherit. We gate the student on task
accuracy with intervals per slice, not on agreement with the teacher, and ship it behind a cascade if it is only
good enough on easy traffic. Break-even is the teacher-token bill over the per-token saving: days at tens of
millions of tokens a day, months below that."

**Drill questions**

1. *Why does a distilled student beat the same model fine-tuned on the same labelled data?* — The soft target is the
   teacher's whole distribution: it ranks the wrong answers, and a sampled label adds 1 − Σp² of gradient noise per
   example. In the core, two examples per context gave 0.876 rule accuracy with soft targets and 0.678 with labels.
2. *Our distilled model writes blends of two valid answers. What happened and what do you change?* — A student too
   small for the teacher's modes, trained with forward KL (SFT, KD), covers both and puts mass between them — 45% of
   the toy student's samples landed where the teacher gives under 1%. Use reverse KL or a JSD with β near 1 (on
   policy, from the SFT student), or a larger student.
3. *Teacher-forced validation loss is great; generations degrade after a few sentences. Why?* — Exposure bias: the
   metric is computed on the teacher's prefixes. Measure on the student's own samples; add on-policy data (λ > 0).
   In the core, supervised KD fell from 1.000 on the teacher's prefixes to 0.788 on its own by position 4; on-policy
   training held 0.994.
4. *We distilled a 1.5B from a thinking 32B and our serving bill went up per request. Why?* — SFT on traces copies
   the teacher's thinking length; the student thinks as long as the teacher (19.9 tokens against 19.9 in the toy).
   Filter traces by length (budget-aware distillation), add a thinking budget, and size `max_model_len` for the
   inherited tail (RL primer §7).
5. *The draft model from the same family accepts only 0.55 on our fine-tuned target. Fix?* — Acceptance is 1 − TV
   between the pair on *this* traffic; the off-the-shelf draft never saw the fine-tune. Distil the draft from the
   target's own outputs or logits (0.890 → 0.988 in the toy), draft probabilistically for sampled traffic (greedy
   caps it at the target's top-token probability), and pick the size where E/(k·c + 1) peaks.
6. *Is distilling worth it at 5M output tokens a day?* — Compute it: the one-off bill is mostly teacher tokens
   ($1,070 self-hosted, $1,800 through an API for 2 × 10⁸ tokens) against a saving of ~$5.30 per million — 41 days
   before evals and engineering. If an off-the-shelf small model covers the easy slice, route first and distil when
   volume grows.

---

## Glossary

| Term | Meaning |
|---|---|
| Teacher / student | the model whose behaviour is copied / the (usually smaller) model trained to copy it |
| Soft targets | the teacher's full output distribution, softmax(z/T), used as the training target |
| Dark knowledge | the relative probabilities a teacher assigns to wrong answers |
| Temperature T | divides logits before the softmax; T > 1 flattens, exposing small probabilities |
| T² factor | multiplies the soft loss so its gradient does not fade as 1/T² |
| Logit matching | MSE on centred logits: the T → ∞ limit of T²·KL |
| Forward KL, KL(p ‖ q) | teacher-weighted; mode-covering; what SFT and KD minimise |
| Reverse KL, KL(q ‖ p) | student-weighted; mode-seeking; what on-policy distillation minimises |
| Generalised JSD(β) | β·KL(p ‖ m) + (1 − β)·KL(q ‖ m), m = βp + (1 − β)q; TRL: β = 0 forward, β = 1 reverse |
| Total variation (TV) | ½Σ\|p − q\|; 1 − TV is a draft's acceptance rate |
| SeqKD | sequence-level KD: SFT on teacher-generated outputs |
| Supervised KD | the teacher's distribution as target on fixed (dataset or teacher) text: GKD with λ = 0 |
| Exposure bias | training on the teacher's prefixes, decoding on one's own: errors compound |
| On-policy distillation / GKD | the student samples; the teacher scores its tokens; minimise a divergence there |
| λ (lmbda) | GKD's fraction of batches generated by the student |
| Per-token reward | r_t = log π_T(y_t \| ·) − log π_S(y_t \| ·): reverse KL as a dense RL reward |
| Rejection sampling | keeping only the teacher samples a verifier accepts |
| Budget-aware distillation | training on traces capped in length to get a student that thinks less |
| Feature distillation | matching hidden states or attention maps, with a layer mapping |
| Width / depth pruning | removing hidden units, heads or layers by importance, then repairing with KD |
| ULD / GOLD | cross-tokenizer distillation by sorted probabilities / aligned spans |
| Draft model | a small model proposing tokens for speculative decoding; a student measured by acceptance |
| Acceptance rate α | Σ min(p, q) per position; vLLM's position-0 rate |
| Cascade | student first, a gate escalating hard requests to the teacher |
| Break-even | the one-off distillation bill divided by the per-token saving |

## Sources

Papers:

- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531) — soft targets, T².
- Buciluă, Caruana & Niculescu-Mizil, *Model Compression*, KDD 2006; Ba & Caruana, *Do Deep Nets Really Need to be
  Deep?* (arXiv:1312.6184) — logit matching.
- Kim & Rush, *Sequence-Level Knowledge Distillation* (arXiv:1606.07947) — SeqKD.
- Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (arXiv:2306.13649)
  — GKD; Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models* (arXiv:2306.08543) — reverse KL.
- Hsieh et al., *Distilling Step-by-Step!* (arXiv:2305.02301).
- Sanh et al., *DistilBERT* (arXiv:1910.01108); Jiao et al., *TinyBERT* (arXiv:1909.10351); Wang et al., *MiniLM*
  (arXiv:2002.10957).
- Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (arXiv:2407.14679) — Minitron.
- Boizard et al., *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss* (arXiv:2402.12030).
- Leviathan, Kalman & Matias (arXiv:2211.17192) and Chen et al. (arXiv:2302.01318) — speculative decoding; Li et al.,
  *EAGLE* (arXiv:2401.15077) and *EAGLE-3* (arXiv:2503.01840); Cai et al., *Medusa* (arXiv:2401.10774).
- DeepSeek-AI, *DeepSeek-R1* (arXiv:2501.12948); Qwen Team, *Qwen3 Technical Report* (arXiv:2505.09388).
- Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631, 2024.

Code and documentation (read 2026-09-27):

- TRL `github.com/huggingface/trl` v1.14.0: `trl/experimental/gkd/gkd_trainer.py`, `gkd_config.py`,
  `trl/trainer/distillation_trainer.py`, `distillation_config.py`, `sft_config.py`, `docs/source/gkd_trainer.md`,
  `minillm_trainer.md`, `gold_trainer.md`, `dataset_formats.md`.
- vLLM `github.com/vllm-project/vllm` v0.30.0: `vllm/config/speculative.py`, `vllm/v1/spec_decode/metrics.py`,
  `docs/features/speculative_decoding/`, the completions protocol (`prompt_logprobs`, `echo`), `--max-logprobs`,
  `--logprobs-mode`.
- `github.com/thinking-machines-lab/tinker-cookbook` (`tinker_cookbook/distillation/train_on_policy.py`,
  `recipes/distillation/`); `github.com/microsoft/LMOps` (`minillm/`); `github.com/NVIDIA/Model-Optimizer`
  (`modelopt/torch/distill/`, `examples/pruning/`); `github.com/NVlabs/Minitron`; `github.com/SafeAILab/EAGLE`;
  `github.com/FasterDecoding/Medusa`; `github.com/sgl-project/SpecForge`;
  `github.com/google-research/distilling-step-by-step`; `github.com/meta-llama/llama-models` (Llama 3.2 model card,
  licences); `github.com/deepseek-ai/DeepSeek-R1`; `github.com/QwenLM/Qwen3`; transformers model docs (DistilBERT,
  Gemma 2, Gemma 3).
- In this repo: [transformer primer](../transformers/docs/transformer-primer.md); [capacity primer](../gpu-capacity-planning/PRIMER.md)
  and [`capacity.py`](../gpu-capacity-planning/capacity.py); [roofline primer](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)
  and [`roofline-core`](../../01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/) (`roofline.llm`, `roofline.cost`);
  [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) and
  [`mini-engine-core`](../../04-inference-engine/serving-engine/mini-engine-core/) (`minengine.spec`);
  [quantization primer](../../04-inference-engine/quantization/PRIMER.md) and
  [`quant-core`](../../04-inference-engine/quantization/quant-core/) (`quantcore.eval`);
  [RL and thinking-models primer](../rl-and-thinking-models/PRIMER.md) and [`rl-core`](../rl-and-thinking-models/rl-core/)
  (`rlcore.pg`, `rlcore.tasks.ThinkTask`); [mixture-of-experts primer](../mixture-of-experts/PRIMER.md);
  [model-landscape primer](../model-landscape/open-weight-llms-primer.md);
  [agentic scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) (`scalelab`);
  the 07 [platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) (evals) and
  [agent-memory primer](../../07-application-agent-framework/agent-memory/PRIMER.md) (`memcore.harness.wilson_interval`).

## Verify list

Product facts in this primer, `distillcore/cost.py` and the notebooks, as of 2026-09-27 — re-check against the
versions you pin.

- **TRL 1.14.0:** `GKDTrainer`/`GKDConfig` under `trl.experimental.gkd` (`from trl import GKDTrainer` fails);
  defaults `lmbda` 0.5, `beta` 0.5, `temperature` 0.9 (sampling only; the loss at T = 1, no T²), `max_new_tokens`
  128, `seq_kd` False, a per-batch λ coin flip, the `vocab_size` check; `DistillationTrainer`/`DistillationConfig`
  stable, always on-policy, `beta` 1.0, `temperature` 1.0 on both logits, `max_completion_length` 512, 256-position
  loss chunks, its Qwen2.5-0.5B ← 1.5B quick start; GOLD (`use_uld_loss`) and MiniLLM experimental; `SFTConfig` and
  the other configs defaulting to bf16 unless `fp16` is set.
- **vLLM 0.30.0:** `--speculative-config` and its `method` values, `draft_sample_method="greedy"` by default,
  `draft_model`'s equal-vocab-size check, `use_heterogeneous_vocab`; the `vllm:spec_decode_*` counters and the
  logged "draft acceptance rate" = accepted ÷ drafted; `prompt_logprobs`, `--max-logprobs` default 20,
  `--logprobs-mode` default `raw_logprobs`; `--dtype half` on a T4.
- **Models:** Qwen2.5-0.5B/1.5B/32B, Qwen3-0.6B/4B and Llama-3.1-8B shapes in `cost.SHAPES`; config `vocab_size`
  151,936 for small Qwen2.5 and all Qwen3, 152,064 for Qwen2.5-7B and larger; licences (Qwen Apache 2.0,
  DeepSeek-R1 MIT with distillation allowed, the Llama naming clause); hosted-API terms.
- **Results quoted:** Qwen3 Table 21 (8B: RL 67.6 at 17,920 GPU-hours; on-policy distillation 74.4 at 1,800);
  tinker-cookbook (~65% → ~76.7% AIME'24); Minitron (40× fewer tokens, 1.8×, up to 16% MMLU); Llama 3.2 and Gemma
  2/3 distillation statements; EAGLE 3×, EAGLE-3 5.6× (13B, 2×RTX 3090), Medusa 2.2–3.6×; SpecForge's
  regeneration step; distilling step-by-step's α = 0.5; DistilBERT's triple loss; TinyBERT and MiniLM described,
  not quantified; lm-eval gsm8k defaults (greedy, 256 tokens).
- **Prices and hardware:** H100 ~$11/GPU-hour on demand (Spot ~$3.7), Gemini 3.5 Flash $9.00/M output (06 lab,
  2026-09-05), the device figures in `cost.GPUS` (roofline.specs'); T4 fits (16 bytes per parameter for AdamW with
  an fp32 master, 2.49 GB of fp32 logits per 4,096 tokens) are predictions — maintained in
  [`COMPUTE.md`](../../COMPUTE.md) and the lab.
