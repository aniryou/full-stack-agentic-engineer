# Reinforcement learning and thinking models: how post-training teaches a model to reason, and what that does to serving

*A primer for 00-foundations. Snapshot: September 2026. Product facts are dated and marked (verify); every formula
has a worked number and names the function in [`rl-core/`](rl-core/) (package `rlcore`) that computes it. Numbers
on the toy tasks are exact (the output space is enumerated) or seeded simulations; serving numbers are a model —
the capacity primer's formulas plus a roofline step — not measurements.*

This primer explains what happens to a language model after pretraining: supervised fine-tuning, learning from
preferences (reward models, RLHF, DPO) and reinforcement learning with verifiable rewards (GRPO and its fixes); how
that last stage produced *thinking models* that spend thousands of tokens before they answer; how to spend
inference compute at test time (think longer, sample more, vote, verify); and what all of this does to the serving
stack below it. It builds on the [transformer primer](../transformers/docs/transformer-primer.md) (§6 training, §7
inference), the [capacity primer](../gpu-capacity-planning/PRIMER.md) (weights, KV bytes, TTFT and TPOT) and the
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) (§5 prefix caching, §6 sampling, §7
speculative decoding, §11 measuring), and does not repeat them. Every concept is learnable at tier T0 with
[`rl-core/`](rl-core/); [`thinking-lab/`](thinking-lab/) trains a tiny transformer with GRPO in torch and serves a
real thinking model in vLLM (T1).

---

## The one-minute version

Pretraining teaches a model what text looks like; **SFT** teaches it a format by imitation; **RL** changes which of
the things it can already do it actually does. RL for LLMs is **sample, score, reweight**: generate completions with
an inference engine, score them (a verifier, a reward model), and take a gradient step that raises the
log-probability of completions that beat a **baseline** and lowers the rest. A **KL penalty** to the SFT model keeps
the policy near it; the optimum is the reference reweighted by exp(reward/β). **Preferences** become rewards through
Bradley–Terry; **DPO** skips the reward model by inverting that closed form. Where answers can be checked, **GRPO**
samples a group per prompt and uses the group mean as the baseline — no value model. Trained this way on math and
code, models learned to **think**: long chains of thought grew because they raised the reward. RL optimises exactly
the reward written down, so loopholes get exploited and anything uncharged — length — grows; loss-averaging details
bend the update too. At inference, thinking longer and sampling more are two budgets to allocate, and only a
verifier or a vote turns samples into accuracy. For serving, thinking makes traffic **decode-heavy and
heavy-tailed**: concurrency scales with output length, KV per session grows while the model thinks, memory and the
ITL SLO set the GPU count, `max_tokens` truncates answers where a thinking budget would not, old thinking is dropped
from multi-turn prompts, and the metric is **cost per correct answer**.

---

## 1. From pretraining to post-training

**Three stages, three objectives.**

| Stage | Data | Objective | What it changes |
|---|---|---|---|
| Pretraining | trillions of tokens of text and code | next-token cross-entropy; compute C ≈ 6·N·D ([transformer primer](../transformers/docs/transformer-primer.md) §6.1–6.2) | what the model knows and can continue |
| SFT | thousands to millions of demonstrations (prompt → good answer) | the same cross-entropy, on the answer tokens only: imitation | format, instruction following, a style of reasoning to start from |
| Preference / RL | the model's *own* samples, scored | raise the probability of what scored well: sample, score, reweight | which behaviours the model uses; how long it thinks; refusals; tool use |

The transformer primer's §6.3 makes the point that post-training is not architecture: the network is the same, the
training signal changes. SFT is maximum likelihood on someone else's outputs (`rlcore.pg.sft_step()`). RL is
maximum likelihood on the model's own outputs, **weighted by how they scored** — which is why it can only amplify
behaviours the model already samples, and why its first ingredient is an inference engine.

**The loop.**

```
          prompts
             │
             ▼
   ┌───────────────────┐  completions (G per prompt,     ┌──────────────────────┐
   │  rollout engine   │  up to tens of thousands of     │  reward              │
   │  vLLM / SGLang    │─────────── tokens each) ───────►│  verifier, or        │
   │  (inference)      │                                 │  reward model        │
   └───────────────────┘                                 └──────────┬───────────┘
             ▲                                                      │ scores
             │ new weights (NCCL broadcast,                         ▼
             │ or shared GPUs)                           ┌──────────────────────┐
             └───────────────────────────────────────────│  trainer             │
                                                         │  advantages → loss → │
                                                         │  one gradient step   │
                                                         └──────────────────────┘
```

In the core's toy, a weak SFT model (two steps of `pg.sft_step()` on the 14 balanced bracket strings) is right
29.7% of the time; 150 steps of REINFORCE with 16 samples each take it to 93.4% (notebook 01, worked example 3).

**Where the compute goes: rollouts.** A training step does a forward and backward pass over every generated token —
compute-bound, about 6·N FLOPs per token — but first those tokens must be *generated*, one decode step at a time,
memory-bound, and the batch lives until its longest completion finishes. `rlcore.workload.rl_step_time()` models one
synchronous step at DAPO's batch shape — 512 prompts × 16 samples = 8,192 completions — for a 7.6B policy on 64
H100s with lognormal lengths (median 4,000 tokens, capped at 20,480): generation 139 s, training 137 s, so rollouts
are 50% of the step, and the generation batch is only 25% occupied on average because the last few long completions
decode almost alone. That model is optimistic; verl reports ~70% of step time in rollouts for DAPO-32B (verify). The
rollout generator is an inference engine with all the concerns of the
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md): continuous batching (§2), KV capacity
and preemption (§4), and prefix caching (§5) — the G samples of one prompt share its prefix, so it is prefilled once.

## 2. Policy gradients over token sequences

**The model as a policy.** At each position the model picks a token a from π(a | s), where the state s is
everything before it. A completion y is a trajectory, and log π(y) = Σ_t log π(a_t | s_t) — the per-token
log-probabilities an engine returns as `logprobs`, summed (`Policy.token_logprobs()`, `Policy.seq_logprob()`). The
core replaces the network with a table of softmaxes, π(a | s) = softmax(θ[s])_a, which keeps one fact exact:

```
∂ log π(a | s) / ∂θ[s, b] = 1[a = b] − π(b | s)                          Policy.grad_logprob()
```

Raising one token's log-probability lowers all the others in proportion (the row sums to zero). Every method below
is a weighted sum of these per-token gradients; the methods differ only in the weights. Under the uniform policy an
8-token completion has log π(y) = 8·log(1/2) = −5.545.

**REINFORCE.** The gradient of the expected reward follows from ∇π = π·∇log π (Williams 1992):

```
∇θ E_{y~π}[R(y)] = E[R(y)·∇log π(y)] ≈ (1/N) Σ_i (R_i − b)·∇log π(y_i)       pg.reinforce_grad()
```

Sample, score, and push each completion's log-probability up or down by how it scored. A uniformly random policy
solves the core's bracket task (8 tokens, balanced) with probability Catalan(4)/2⁸ = 14/256 = 5.5%
(`SeqTask.random_success_rate()`); a few hundred steps of this estimator solve it.

**Variance, baselines and advantages.** Because E_π[∇log π(y)] = ∇Σπ(y) = 0, subtracting any baseline b that does
not depend on the sample leaves the gradient unbiased — and changes its variance. `pg.grad_variance()` at the start
of a ThinkTask run: 0.054 with no baseline, 0.029 with the batch mean. Add a constant +5 to every reward — no
information at all — and the no-baseline variance becomes 3.678 while the baselined one stays 0.029. R − b is the
**advantage**. The baselines in use:

| Baseline | Used by | Note |
|---|---|---|
| batch or group mean | GRPO (§4), `pg.advantages("mean")` | the group of G samples of one prompt is the batch |
| leave-one-out mean | RLOO | exactly n/(n − 1) times the mean-baseline advantage: same direction |
| learned value V(s), per token | PPO (§3) | a second network as large as the policy |

**The KL penalty to a reference model.** Maximise E[R] − β·KL(π ‖ π_ref), with π_ref the frozen SFT model. As reward
shaping it is R − β·(log π(y) − log π_ref(y)), which has exactly the gradient of −β·KL because E[∇log π] = 0 (the
PPO-RLHF form; GRPO puts it in the loss instead, §4). Over all distributions the optimum has a closed form:

```
π*(y) = π_ref(y)·exp(R(y)/β) / Z,     KL(π* ‖ π_ref) = E_π*[R]/β − log Z       pg.kl_optimal()
```

For the SFT reference and R = 1 if balanced (`pg.kl_optimal()` over all 256 strings):

| β | 10 | 1 | 0.3 | 0.1 |
|---|---|---|---|---|
| E[R] = P(balanced) | 0.319 | 0.535 | 0.922 | 1.000 |
| KL(π* ‖ π_ref), nats | 0.00 | 0.12 | 0.87 | 1.21 |

Why it exists: the reward is trustworthy only near the data it was built on, fluency and language come from the
reference, and it slows collapse onto one answer. What it implies: π* can only move mass among completions π_ref
already produces — a completion with π_ref(y) = 0 stays at 0 for every β. RL sharpens; it does not invent.

**Reward hacking.** RL optimises exactly the reward written down. The core's `buggy_verify` returns *pass* the moment
the bracket depth goes negative (a harness that counts an early exit as success). 200 of 256 strings pass it; 14 are
balanced. The SFT reference passes it 72.7% of the time and is right 29.7%. After 200 REINFORCE steps against it
(`pg.train_reinforce()`, notebook 01, worked example 6):

| Training | Passes the buggy check | Truly balanced | P(starts with `)`) | KL to π_ref |
|---|---|---|---|---|
| reference | 0.727 | 0.297 | 0.18 | 0 |
| β = 0 | 0.994 | 0.049 | 0.80 | 1.06 |
| β = 0.3 | 0.959 | 0.354 | 0.25 | 0.20 |

At depth 0, `)` is an instant pass from every state, so gradient ascent finds it. The KL-regularised optimum
multiplies every passing string by the same exp(1/β), so it keeps the reference's honest-to-hacked ratio: 40.9% as
β → 0 (notebook 01, exercise 1.6). The penalty is a leash, not a fix. The fixes are the verifier and evals of the
true objective. DeepSeek-R1 used rule-based rewards and avoided neural reward models partly for this reason (verify).

**Length bias.** RL also lengthens whatever the reward does not charge for. In `ThinkTask` the policy emits "think"
tokens until it answers, and P(correct | L) = 1 − e0·(1 − q)^L (§6 explains the form). With reward = correct − c·L
the optimum is where a token's marginal gain equals its cost:

```
e0·(−ln(1 − q))·(1 − q)^L* = c   ⇒   L* = ln(c / (e0·(−ln(1 − q)))) / ln(1 − q)       ThinkTask.optimal_length()
e0 = 0.8, q = 0.1, c = 0.01:  L* = 20.23
```

Starting from a policy that answers at once half the time (a mean of 1.0 thinking token), 300 REINFORCE steps
with e0 = 0.8, q = 0.15 grow thinking to 11.1 tokens with no cost and to 7.8 with c = 0.02 (L* = 11.5; the table
policy is still climbing). The same pressure acting on real models is §5's length growth; §3 and §4 add two
more sources of length bias — reward models and loss averaging.

## 3. Learning from preferences

**Bradley–Terry reward models.** When no program can grade an answer, people compare two. Bradley–Terry models a
comparison as a noisy difference of rewards,

```
P(A ≻ B) = σ(r(A) − r(B)),     loss = −E[log σ(r(y⁺) − r(y⁻))]                    pref.bt_prob(), pref.fit_bradley_terry()
```

so a reward model is logistic regression on (chosen − rejected). Fitting 4,000 synthetic comparisons generated by
r = 1.5·x₁ − 0.5·x₂ recovers (1.57, −0.54) (notebook 02). Only differences are identified: σ(2 − 1) = 0.7311 =
σ(12 − 11). A reward model has no zero point, which is why TRL's `RewardTrainer` offers
`center_rewards_coefficient` and why scores from different runs do not compare.

**RLHF with PPO, in brief.** The InstructGPT recipe: SFT, fit a reward model on human comparisons, then PPO against
it with a per-token KL penalty. PPO's pieces:

- the **clipped ratio** −min(ρA, clip(ρ, 1 − ε, 1 + ε)A), ρ = π/π_old, which lets several gradient steps reuse one
  batch of samples without moving far (§4 works it);
- a **value model** V(s) — typically as large as the policy — giving each token its own baseline through
  **generalised advantage estimation**:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t),    A_t = δ_t + γλ·A_{t+1}                        pref.gae()
rewards (0, 0, 1), values (0.5, 0.6, 0.8), γ = 1:   λ = 0 → (0.1, 0.2, 0.2);  λ = 0.5 → (0.25, 0.3, 0.2);  λ = 1 → (0.5, 0.4, 0.2)
```

λ = 0 trusts V (low variance, biased when V is wrong); λ = 1 trusts the sampled return. In RLHF the reward-model
score sits on the last token and −β·log(π/π_ref) on every token. Four networks are in memory — policy, reference,
reward model, value model — which is the cost DPO and GRPO each remove part of.

**DPO: the closed form, as a classification loss.** §2's optimum can be inverted:

```
π*(y|x) = π_ref(y|x)·exp(r(x,y)/β) / Z(x)
   ⇒  r(x, y) = β·log(π*(y|x) / π_ref(y|x)) + β·log Z(x)
   ⇒  P(y⁺ ≻ y⁻) = σ(r⁺ − r⁻) = σ(β·[log π*/π_ref(y⁺) − log π*/π_ref(y⁻)])           (Z(x) cancels)
L_DPO(θ) = −E log σ(β·[(log π_θ(y⁺) − log π_ref(y⁺)) − (log π_θ(y⁻) − log π_ref(y⁻))])      pref.dpo_loss()
```

Train the policy directly on pairs; no reward model and no sampling. TRL's `DPOTrainer` (`loss_type="sigmoid"`,
default β = 0.1, log-probabilities summed over completion tokens) computes exactly this. Worked number: β = 0.1, the
chosen log-ratio +1 and the rejected −1 give −log σ(0.2) = 0.598139; zero margin gives ln 2 = 0.693147, where every
run starts. The per-pair gradient weight is β·σ(−m) with m the scaled margin (`pref.dpo_step()`): mis-ranked pairs get
up to β, confidently ranked ones almost nothing. The **implicit reward** β·log(π/π_ref) is what TRL logs as
`rewards/chosen` and `rewards/rejected`.

Notebook 02 checks the claim end to end. Preferences over bracket strings come from a true reward r = 3·balanced,
4,096 pairs drawn from the SFT reference (59% are ties, labelled by a coin), β = 1:

| Path | P(balanced) |
|---|---|
| reference | 0.297 |
| closed form π* ∝ π_ref·exp(r/β) (`pg.kl_optimal()`) | 0.895 |
| RLHF: reward model r̂ = 2.94·balanced, then REINFORCE with β = 1 | 0.885 |
| DPO, 150 epochs (`pref.dpo_step()`), KL(π_DPO ‖ π*) = 0.0094 | 0.889 |

The implicit reward separates balanced from unbalanced strings by 2.94 (true gap 3). At the end TRL's metrics read
rewards/chosen −0.407, rewards/rejected −1.524, margins +1.118, accuracies 0.705: the chosen strings' log-ratios
*fell*.

**What DPO gives up.**

- **It is offline.** It learns from the pairs' support; it cannot explore, and what it does to completions outside
  the pairs is whatever the network's generalisation does. Iterated or "online" DPO resamples pairs from the current
  policy to close part of the gap.
- **It optimises the margin, not the likelihood.** Lowering both log-ratios — the rejected one faster — wins the
  loss, so rewards/chosen drifts negative (−0.407 above). A collapsing chosen log-probability is a sign of
  over-training.
- **Deterministic preferences push the margin without bound** — the logistic loss keeps paying for a wider margin.
  β is the only brake.
- **No reward model is left behind** to rerank samples, monitor drift or reuse for best-of-n.

**Three variants, one line each.** **IPO**: a squared loss toward a fixed margin 1/(2β) (5 at β = 0.1,
`pref.ipo_loss()`), so the margin cannot run away. **KTO**: unpaired thumbs-up/thumbs-down labels with a
prospect-theory utility and loss aversion; no pairs needed. **ORPO**: no reference model; the SFT loss plus a
log-odds-ratio penalty on the rejected answer, in one stage (experimental in TRL).

**Reward-model over-optimisation, and length.** Any bias in the annotators becomes the reward. `pref.response_catalogue()`
holds 40 answers, each in five versions padded with 0–800 filler tokens that lower true quality by 0.3 per 100
tokens; the reference rarely pads. Annotators prefer quality *and* length: P(A ≻ B) = σ(Δquality + 0.6·Δlength/100)
(`pref.length_biased_prefs()`). A reward model fit on 3,000 such pairs learns r̂ = 1.01·quality + 0.61·length/100 —
faithfully. Optimising it harder (smaller β, closed form):

| β | ref | 10 | 1 | 0.5 | 0.3 | 0.1 |
|---|---|---|---|---|---|---|
| KL, nats | 0 | 0.00 | 0.37 | 1.68 | 5.42 | 8.76 |
| reward-model score | 1.09 | 1.16 | 1.83 | 2.69 | 4.10 | 4.86 |
| true quality | −0.280 | −0.229 | 0.138 | 0.136 | −0.668 | −0.957 |
| mean length (tokens) | 223 | 227 | 275 | 416 | 778 | 950 |

The proxy rises monotonically; true quality peaks (best at β = 0.7, a KL of 0.78 nats, notebook 02, exercise 2.5)
and then falls below the reference while answers grow four times longer. Optimising the true quality instead reaches
1.435 at 153 tokens. That shape — proxy up, gold up then down — is Gao et al.'s over-optimisation, and length is its
most common form. Defences: a KL budget (stop early), length-controlled collection and judging, verifiable rewards
where they exist, and evals on the true objective.

## 4. RL with verifiable rewards and GRPO

**Verifiers.** RLVR replaces the reward model with a program:

| Task | Verifier | Watch for |
|---|---|---|
| math | extract the final answer (R1: a `\boxed{}` answer), check equivalence (TRL `accuracy_reward` uses `math-verify`) | extraction failures scored as wrong; answers that game the parser |
| code | run the tests in a sandbox | exit-code and harness loopholes (§2); run untrusted code isolated (`07-application-agent-framework/sandboxed-execution`) |
| format | a regex, e.g. TRL's `think_format_reward` `^<think>(?!.*<think>)(.*?)</think>.*$` | the format becoming the goal |

DeepSeek-R1-Zero used exactly two rule-based rewards, accuracy and format, and no neural reward model (verify). DAPO
scores +1 / −1 by answer equivalence. Rewards are sparse — one number per completion — and binary, which is what
makes GRPO's group statistics work.

**GRPO.** For each prompt, sample a group of G completions from π_old, score them, and give every token of
completion i the same advantage (DeepSeekMath; R1 eq. 1–3; TRL's `GRPOTrainer`):

```
A_i = (r_i − mean(r)) / (std(r) + 1e-4)          std with Bessel's correction      grpo.group_advantages()
per token t of completion i:
  ρ = π_θ(a_t | s_t) / π_old(a_t | s_t)
  loss = −min(ρ·A_i, clip(ρ, 1 − ε_low, 1 + ε_high)·A_i) + β·k3                     grpo.clipped_surrogate()
  k3 = π_ref/π_θ − log(π_ref/π_θ) − 1                                               grpo.k3()
```

The group is the baseline, so there is no critic "typically the same size as the policy model" (R1). Worked
advantages: rewards [1, 0, 0, 1] → ±0.865875; [1, 0, 0, 0] → [1.4997, −0.4999, −0.4999, −0.4999]; [1, 1, 1, 1] → 0.
**A group that is all right or all wrong teaches nothing**, however many tokens it cost; TRL logs the share as
`frac_reward_zero_std`. With TRL's default `num_iterations=1` the batch is used for one step with the weights that
sampled it, ρ ≡ 1, and the clip never binds; it matters with μ > 1 or stale, off-policy rollouts (§8).

**k3, the KL estimator.** GRPO estimates KL(π ‖ π_ref) per token from samples of π. The naive k1 = log(π/π_ref) is
unbiased but noisy and often negative; k3 is unbiased, never negative (e^d ≥ 1 + d) and quieter (Schulman 2020).
Worked values: log(π_ref/π) = 0.1 → 0.0051709; −0.1 → 0.0048374; 0.5 → 0.1487213. On two three-way distributions
with KL 0.1168, 100,000 samples give k1 mean 0.1186 with std 0.460 (minimum −0.693) and k3 mean 0.1163 with std
0.106 (`grpo.k1()`, `grpo.k3()`, notebook 03). TRL's default is β = 0: no reference model is loaded at all; the R1
paper's setting is β = 0.001 per TRL's docs (verify).

**The clip, clip-higher and entropy.** Clipping is multiplicative: in one update a token with π_old = 0.01 can rise
to at most 0.0120 with ε = 0.2 (0.0128 with ε_high = 0.28), while one at 0.5 can reach 0.6000. Rare tokens —
exploration — are capped hardest, so symmetric clipping drives entropy down; DAPO's **clip-higher** decouples ε_low =
0.2 from ε_high = 0.28. The toy shows the collapse itself: GRPO on the bracket task (μ = 4, `grpo.train_grpo()`)
takes P(correct) from 0.297 to 0.995 while the effective number of distinct correct answers, exp(entropy) over
them, falls from 13.9 to 2.4. Clip-higher's effect on real models' entropy is DAPO's measurement; the table policy
collapses with or without it. The other lever is an entropy bonus, −c·log π(y) added to the reward
(`pg.reinforce_grad(entropy_coef=…)`; TRL's `entropy_coef`, default 0.0, verify): 300 REINFORCE steps keep 12.8
effective correct answers with c = 0.05 against 6.5 without, at P(correct) 0.959 against 0.975.

**How the loss is averaged: the length bias.** TRL's `loss_type` decides what one token is worth
(`grpo.token_weights()`); for completions of 10 and 50 tokens in a group of two, with `max_len` 100:

| `loss_type` | Normaliser | Per-token weight, 10-token / 50-token completion |
|---|---|---|
| `"grpo"` | mean over each completion's tokens, then over completions: 1/(G·\|o_i\|) | 0.0500 / 0.0100 |
| `"dapo"` (TRL default) | all tokens in the batch alike: 1/Σ\|o_i\| | 0.0167 / 0.0167 |
| `"dr_grpo"` | a constant, 1/(G·max_len) | 0.0050 / 0.0050 |

Per-sequence averaging favours short correct answers and punishes long wrong ones *less per token* — including
truncated ones (Dr. GRPO's argument). `grpo.expected_update()` averages 600 one-group updates at a fixed ThinkTask
policy (answers with probability 0.1 per step; 18.5% of completions truncated at 16 tokens) and compares the
direction with the exact gradient of accuracy: cosine 0.67 for `"grpo"`, 0.99 for `"dapo"`, 1.00 for `"dr_grpo"`;
along the per-sequence update truncation *rises* and thinking lengthens two to four times faster, along the others
truncation falls. Dr. GRPO also drops the std division (`scale_rewards="none"`): the one miss on a prompt solved 7
times in 8 gets −2.47, a miss on a 50/50 prompt −0.94, so std scaling up-weights prompts that are nearly solved or
nearly hopeless (a "difficulty bias").

**DAPO's fixes.** DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) trains Qwen2.5-32B base with four
changes and no KL term:

| Technique | What it does | Core / TRL |
|---|---|---|
| clip-higher | ε_low 0.2, ε_high 0.28 | `GRPOConfig(epsilon_high=0.28)` |
| dynamic sampling | over-sample; drop groups with all-equal rewards; refill the batch | `grpo_step(..., dynamic_sampling)`; no TRL flag (it logs `frac_reward_zero_std`) |
| token-level loss | 1/Σ\|o_i\| | `loss_type="dapo"` |
| overlong shaping | mask truncated completions; soft penalty near the cap | `mask_truncated_completions=True`; `grpo.soft_overlong_penalty()`, TRL `get_soft_overlong_punishment` |

The soft penalty is 0 up to L_max − L_cache, falls linearly to −1 at L_max, and is −1 beyond: with L_max 100 and a
cache of 20, lengths 80, 90, 100, 101 score 0, −0.5, −1.0, −1. On a nine-prompt ThinkTask dataset with four prompts
per step, 59% of generated groups are silent without dynamic sampling; with it every trained group is informative,
at 8.2 groups generated per step instead of 4.0 (notebook 03). DAPO's ablation (AIME 2024 avg@32): naive GRPO 30 →
+ overlong filtering 36 → + clip-higher 38 → + soft overlong punishment 41 → + token-level loss 42 → + dynamic
sampling 50 (verify). Its run: 512 prompts × 16 responses, maximum generation 20,480 tokens (16,384 + a 4,096 cache).

**Compute.** Per step: G× generation, the dominant cost (§1); a forward pass for π_ref's log-probabilities if β > 0;
π_old's if μ > 1 (or taken from the rollout engine, which is where train–inference mismatch enters, §8); and one
forward + backward pass of the policy. In memory: the policy with gradients and optimizer state, the reference if
β > 0, and the rollout engine's weights and KV cache — on the same GPUs (colocated) or others.

**TRL's `GRPOTrainer` as the concrete reference** (v1.14.0 field defaults, verify):

| Field | Default | Paper setting |
|---|---|---|
| `num_generations` (G) | 8 | DAPO 16 |
| `max_completion_length` | 512 | DAPO 20,480 |
| `beta` | 0.0 (no reference model) | R1 0.001; DAPO 0 |
| `epsilon` / `epsilon_high` | 0.2 / None (= epsilon) | DAPO 0.2 / 0.28 |
| `loss_type` | `"dapo"` | R1 `"grpo"`; Dr. GRPO `"dr_grpo"` |
| `scale_rewards` | `"group"` | Dr. GRPO `"none"` |
| `num_iterations` (μ) | 1 | — |
| `mask_truncated_completions` | False | DAPO True |
| `use_vllm` / `vllm_mode` | False / `"colocate"` | `"server"` puts vLLM on separate GPUs |
| `bf16` | True unless `fp16` is set | a T4 has no bf16: `bf16=False, fp16=True` |

The defaults are neither the R1 paper's GRPO nor DAPO; set the fields explicitly when reproducing either (notebook 03,
exercise 3.6).

## 5. Thinking models

**Long chain-of-thought as a learned behaviour.** DeepSeek-R1-Zero applied GRPO directly to DeepSeek-V3-Base with
rule-based accuracy and format rewards and a template asking for reasoning inside `<think>` tags. AIME 2024 pass@1
rose from 15.6% to 71.0% (86.7% with majority voting) over thousands of RL steps, and the responses grew from
hundreds to thousands of reasoning tokens; reflection ("wait…") emerged on its own — the paper's "aha moment". It
also produced endless repetition, poor readability and language mixing (verify). Nobody rewarded length: longer
thinking raised the reward, so RL lengthened it — §2's ThinkTask in miniature (1.0 → 11.1 tokens).

**The DeepSeek-R1 recipe** (two SFT stages, two RL stages; R1 is 671B total / 37B activated, verify):

| Stage | Data | Purpose |
|---|---|---|
| 1. cold-start SFT | thousands of long-CoT examples | readable reasoning to start RL from |
| 2. reasoning RL | as R1-Zero, plus a language-consistency reward | reasoning ability |
| 3. rejection-sampling SFT | ~600k reasoning samples kept only if correct + ~200k non-reasoning = ~800k; 2 epochs on V3-Base | fold RL's gains into a clean SFT model; general skills |
| 4. RL for all scenarios | rule rewards for reasoning; reward models for helpfulness and harmlessness | alignment across tasks |

Stage 3 is itself a length pressure: correct answers are longer on average when thinking helps, so imitating only the
survivors lengthens thinking with no RL at all. On ThinkTask, three rounds of "sample 512, keep the correct ones, 20
SFT steps" (`pg.sft_step()`, notebook 05) take mean thinking from 1.0 to 1.49, 2.28 and 2.77 tokens and accuracy from
0.304 to 0.462.

**Distillation of traces into small models.** The R1 distills (Qwen2.5 1.5B–32B, Llama 8B and 70B) were trained by
SFT alone on the ~800k samples, with no RL stage. DeepSeek-R1-Distill-Qwen-1.5B reports AIME 2024 pass@1 28.9; and
distillation beat RL: RL on Qwen-32B-Base for over 10K steps reached 47.0 on AIME 2024, the distilled 32B 72.6
(verify). Qwen3's small dense models (0.6B–14B) likewise come from strong-to-weak distillation. In the toy, a fresh
student trained by SFT on 1,000 traces of the RL-trained ThinkTask teacher reaches accuracy 0.848 against the
teacher's 0.855, with the same mean thinking length, 11.1 tokens — behaviour copied from outputs alone.

**Hybrid thinking modes and chat-template switches.** Qwen3's original release is hybrid: the chat template's
`enable_thinking` switch (default True) makes it think or not. `enable_thinking=False` appends an empty
`<think>\n\n</think>\n\n` block to the generation prompt; the soft switches `/think` and `/no_think` in a message
follow the latest instruction. The 2507 releases split the modes into separate `-Instruct-2507` and `-Thinking-2507`
models. Through vLLM the switch is `chat_template_kwargs: {"enable_thinking": false}` per request, or
`--default-chat-template-kwargs` for the server (verify). The same templates drop earlier turns' thinking from the
prompt (§7).

**Thinking budgets and `reasoning_effort`.**

| Control | What it does | Watch for |
|---|---|---|
| `max_tokens` | caps reasoning + answer together | ends inside `<think>`: empty content, `finish_reason="length"` (§7) |
| vLLM `thinking_token_budget` (request) | when reasoning reaches the budget, forces the end-of-think string; needs `--reasoning-parser` (and optionally `--reasoning-config`); −1 = unlimited | vLLM-specific (verify) |
| Qwen's two-call recipe | call 1 with `max_tokens` = budget; if still thinking, append "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n" and continue | two requests; the second re-prefills |
| `reasoning_effort` | vLLM accepts none … max; for Qwen3 any value but `"none"` only sets `enable_thinking=True` | on Qwen3 it is a switch, not a length control (verify) |
| gpt-oss effort | low / medium / high written into the system prompt (Harmony format; default medium) | other values are rejected (verify) |

**A thinking token is billed as an output token.** `completion_tokens` counts reasoning plus answer; vLLM fills
`usage.completion_tokens_details.reasoning_tokens` when a reasoning parser is on, and `include_reasoning: false` hides
the reasoning but still generates it (verify). At the 06 scaling lab's example prices, a call with 5,000 input tokens
(2,700 cached) and 350 output tokens costs $0.007005; the same call with 3,500 output tokens costs $0.035355, 5.05×
(`workload.api_cost()`, reproducing that lab's `cost_per_call`). Recommended sampling differs too: Qwen3 thinking
mode temperature 0.6, top-p 0.95, top-k 20; greedy decoding "can lead to performance degradation and endless
repetitions"; DeepSeek-R1: temperature 0.6 and no system prompt (verify). Sampling mechanics: the serving-engine
primer §6.

## 6. Test-time compute

**Two kinds of hard.** `rlcore.ttc` reads P(correct | L) = 1 − e0·(1 − q)^L as "each thinking token cracks the
problem with probability q; an uncracked answer is a guess, right with probability 1 − e0". It adds a second axis:
with probability a an attempt starts on an approach that can work at all, so one sample is right with

```
P(correct | L) = 1 − e0·(1 − a·(1 − (1 − q)^L))                                       ttc.sample_accuracy()
```

and a = 1 recovers the ThinkTask formula. `ttc.question_set()` draws 400 questions with a ~ Beta(2, 1) (mean 0.65),
q lognormal (median crack length 1,337 tokens) and e0 = 1 (open-ended answers, no lucky guesses). **Sequential**
compute — thinking longer — helps questions that are slow to crack; **parallel** compute — more samples — helps
questions where an attempt can dead-end, *if* something can pick the right sample.

**Sequential: diminishing returns.** One sample's accuracy against thinking length (`ttc.accuracy()`):

| L (tokens) | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---|---|---|---|---|---|
| accuracy | 0.230 | 0.342 | 0.457 | 0.550 | 0.610 | 0.639 |
| gain per 1K tokens | +0.461 | +0.224 | +0.115 | +0.047 | +0.015 | +0.004 |

Accuracy plateaus near the 0.65 of attempts that start on a workable approach: thinking cannot rescue a dead end.
And a fixed budget over-thinks: if a model stopped when it cracked a question, 41% of a fixed 4,000-token think would
be spent after the answer was already found (notebook 04, exercise 4.6) — the case for adaptive budgets.

**Parallel: best-of-n, verifiers and votes.** With a perfect verifier, n samples give 1 − (1 − p)^n. A reward model
sees correctness through noise (`ttc.best_of_n_accuracy()`, p = 0.3): at n = 16, 0.996 with a verifier, 0.934 with
noise 0.5, 0.702 with noise 1.0 — more samples, more chances to be fooled; a *biased* scorer (§3) selects for its
bias. **Majority vote** (self-consistency) needs no checker, only that the right answer be the most common
(`ttc.majority_accuracy()`, exact). One question, a sample right with p = 0.4:

| Where the wrong 60% goes | n = 1 | n = 15 | n = 31 |
|---|---|---|---|
| one common misconception (0.42 / 0.18) | 0.400 | 0.449 | 0.447 |
| two equal wrong answers (0.3 / 0.3) | 0.400 | 0.534 | 0.621 |
| four scattered wrong answers (0.15 each) | 0.400 | 0.780 | 0.925 |

As n → ∞ the vote is right iff p exceeds every wrong answer's share. lm-eval's GSM8K self-consistency task reports
maj@64 from 64 samples at temperature 0.2 (verify).

**pass@k vs pass^k, and the unbiased estimator.** pass@k — at least one of k samples correct — must be estimated from
n ≥ k samples with c correct (Chen et al. 2021):

```
pass@k = 1 − C(n − c, k) / C(n, k) = 1 − Π_{i=n−c+1..n} (1 − k/i)                       ttc.pass_at_k()
n = 10, c = 3:  pass@1 = 0.3,  pass@5 = 0.916667,  pass@8 = 1.0 (n − c < k);   n = 64, c = 16: pass@8 = 0.914746
```

The plug-in 1 − (1 − c/n)^k is biased low, worst where pass@k is most informative: for p = 0.1, n = 10, k = 5 its
expectation is 0.3485 against a truth of 0.4095, which the unbiased estimator matches exactly
(`ttc.expected_estimate()`). **pass^k** — all k succeed, C(c, k)/C(n, k), τ-bench's reliability metric — is what an
agent flow is judged on: n = 16, c = 4 gives pass@4 = 0.728022 but pass^4 = 0.000549 (`ttc.pass_hat_k()`). Sampling
raises pass@k and does nothing for pass^k. Report either with confidence intervals — the Wilson interval of the 07
platform lab's evals notebook (`agentlab.evals.wilson_interval`,
[`08_evals_trajectory_judge_gates`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py)).

**Search with process reward models, in brief.** An outcome reward scores the final answer; a **process reward
model** (PRM) scores each step, which allows beam search or lookahead over partial solutions and reranking of whole
ones (Lightman et al. 2023; Snell et al. 2024). R1's authors report PRMs and MCTS as unsuccessful for large-scale
RL — steps are hard to define, labels are costly, reward hacking follows — while noting PRMs remain useful for
reranking and guided search (verify).

**Compute-optimal allocation.** Split a per-question budget as n·(L + 50) ≤ B (`ttc.allocate()`):

| Questions | Picker | Best (n, L) at 1K / 4K / 16K tokens |
|---|---|---|
| only slow to crack (a = 1) | verifier or vote | (1, 950) / (1, 3950) / (1, 15950) |
| slow or dead-end (a ~ Beta(2, 1)) | verifier | (2, 450) / (8, 450) / (16, 950) |
| slow or dead-end | majority vote | (1, 950) / (1, 3950) / (4, 3950) |

When length is the only difficulty, n short attempts equal one long one minus the answer overheads (the crack rate
is memoryless), so thinking wins; when attempts can dead-end, a verifier makes sampling the better buy as the budget
grows; a vote pays only once single samples usually win it — from 16,000 tokens here. **A smaller model with more
samples**: give a model that costs a fifth as much per token (and is weaker on both axes) five times the tokens.
With a verifier it beats the larger model at equal compute (0.586 vs 0.502 at 1K large-model tokens); with a vote
it loses (0.465 vs 0.479 at 1K, 0.573 vs 0.705 at 4K). Whether it wins is a property of the task and of having a
checker — measure it on your evals.

## 7. What thinking does to serving

**Output-heavy, decode-dominant, heavy-tailed.** A chat request is prompt-heavy; a thinking request spends most of
its life in decode, and its length depends on how hard the question is. A lognormal with median 1,500 thinking tokens
and σ = 1 has mean 2,473, p90 5,403 and p99 15,361 (`workload.lognormal_mean()`, `workload.lognormal_quantile()`):
the p99 is ten times the median. On a small model one trace's KV is the size of the model: Qwen3-0.6B holds 112 KiB
of KV per token in fp16 (114,688 B, `workload.kv_per_token_kb()`), so an 8K-token trace holds 0.94 GB against 1.19
GB of weights.

**The KV working set grows with the square of the output.** At decode step t a request holds P + t tokens of KV,
for L steps:

```
KV-token-steps = Σ_{t=1..L} (P + t) = P·L + L(L + 1)/2                                 workload.kv_token_steps()
P = 1,500:  L = 300 → 495,150;   L = 3,000 → 9,001,500 = 18.2×;   L = 1,500 → 6.8×;   L = 8,000 → 88.9×
```

Against a 300-token answer, memory × time per request grows 4–18× for 1,000–3,000 output tokens on a 1,500-token
prompt (4.0× at 1,000, 6.8× at 1,500, 18.2× at 3,000). Peak KV per request grows less (4,500 vs 1,800 tokens: 2.5×);
what explodes is how long it is held.

**The capacity primer's formulas with long outputs.** The [capacity primer](../gpu-capacity-planning/PRIMER.md) sizes
an internal assistant with [`capacity.py`](../gpu-capacity-planning/capacity.py): 8.33 requests/s, 1,500 tokens in,
300 out, 40 ms TPOT, Mistral Small 3 (24B) in FP8 on H100s. `workload.plan()` restates those formulas (Little's law,
then GPUs per constraint) and reproduces the primer's numbers in a test; it adds what `decode_aggregate` leaves out —
the batch per GPU is capped by HBM and by the ITL SLO, and the step is max(bytes/bandwidth, FLOPs/peak).

| | No thinking (the primer) | 2,700 thinking + 300 answer | Same, 20 ms ITL SLO |
|---|---|---|---|
| request lifetime | 12.07 s | 120.07 s | 60.07 s |
| live requests (Little's law) | 100.6 | 1,000.6 | 500.6 |
| average context; KV per session (FP8) | 1,650; 0.126 GB | 3,000; 0.229 GB | 3,000; 0.229 GB |
| sessions per GPU by HBM | 381.3 | 209.7 | 209.7 |
| batch per GPU within the ITL SLO | 824 | 480 | 187 |
| GPUs by memory / ITL / decode / prefill | 0.26 / 0.12 / 0.27 / 0.61 | 4.77 / 2.08 / 2.57 / 0.61 | 2.39 / 2.68 / 2.67 / 0.61 |
| GPUs needed (binding) | 1 (prefill) | 5 (memory) | 3 (ITL) |

Ten times the output needs eighteen times the GPUs for memory (4.77 vs 0.264): concurrency grows 10× and each live
session holds 1.8× the KV. The primer's `decode_aggregate` would put all 1,000 live requests in one batch — 229 GB
of KV on an 80 GB card. Decode *throughput* is not the binding constraint.

**ITL is the binding SLO; TTFT less so.** Thinking leaves the prompt, and so TTFT, unchanged — but the user waits for
the *answer*: with 2,700 thinking tokens at 40 ms each, the first answer token arrives 108.07 s after the request
(`workload.request_duration_s()` with the thinking tokens as output). Streaming the reasoning, or a summary of it,
is the UX answer; the capacity answer is that ITL, not tokens/s, governs how many sessions a GPU may hold. With a
20 ms ITL SLO a GPU holds 187 sessions (`workload.max_batch_for_itl()`), below the 209 that fit in HBM: ITL binds
first. Measure with the serving-engine primer's §11 method — open-loop load at realistic (here: heavy-tailed) output
lengths, percentiles of ITL and TPOT, goodput against the SLO — and watch `vllm:inter_token_latency_seconds`,
`vllm:kv_cache_usage_perc` and `vllm:num_preemptions`: a long-tail trace that outgrows the pool preempts the newest
request.

**Thinking is dropped from history: prefix-cache implications.** Qwen3's template (and gpt-oss's: "CoT is dropped
during all previous turns") renders earlier assistant turns without their reasoning (verify). Turn N's KV holds
prompt + thinking + answer; turn N+1's prompt contains only the answer, so the prefix-cache hit
([serving-engine primer §5](../../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching): full 16-token blocks
only) ends where turn N's prompt ended, and the answer is prefilled again (`workload.turn_prefills()`; 1,000-token
system prompt, turns of 100 user + 800 thinking + 200 answer tokens):

| Turn | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| thinking dropped: prompt / prefilled | 1,104 / 1,104 | 1,408 / 304 | 1,712 / 304 | 2,016 / 304 |
| thinking kept (one tool loop): prompt / prefilled | 1,104 / 1,104 | 2,208 / 112 | 3,312 / 112 | 4,416 / 112 |

Dropping thinking keeps prompts short, but the 800 thinking tokens' KV per turn is computed, held and never reused,
so multi-turn hit rates are lower than for the same conversation without thinking. Within one multi-step tool loop
the template keeps the thinking and the whole previous sequence is a hit.

**Streaming reasoning and vLLM's `--reasoning-parser`.** `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`
splits the output into `message.reasoning` and `message.content` (streamed as `delta.reasoning` and `delta.content`).
vLLM 0.30 names the field `reasoning`; SGLang, the DeepSeek API and Qwen's docs use `reasoning_content`, so a portable
client reads both. Parser names use underscores in vLLM (`qwen3`, `deepseek_r1`, `openai_gptoss`) and hyphens in
SGLang (`deepseek-r1`); `--enable-reasoning` no longer exists. Structured output applies after the end of thinking
unless `enable_in_reasoning` is set, and tool calls are parsed only from `content`. No Prometheus metric separates
reasoning from answer tokens — both are in `vllm:generation_tokens`; per request, read
`usage.completion_tokens_details.reasoning_tokens` (all verify).

**Budget enforcement: `max_tokens` vs budget forcing.** `workload.budget_outcome()`: each question needs L_req ~
lognormal(1,500, σ = 1) thinking tokens and the answer is 300; truncation returns no answer, budget forcing answers
with what it has (right 30% of the time when uncracked, e0 = 0.7):

| Limit (tokens) | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|
| `max_tokens`: accuracy / truncated | 0.561 / 0.439 | 0.823 / 0.177 | 0.952 / 0.048 | 0.991 / 0.009 |
| thinking budget (limit − 300): accuracy | 0.693 | 0.876 | 0.966 | 0.994 |
| mean output tokens (either) | 1,559 | 2,136 | 2,526 | 2,705 |

Same tokens, different outcome: at 4K one request in six gets no answer under `max_tokens`. Set `max_tokens` and
`max_model_len` for the tail — 17,408 (a multiple of 1,024) keeps truncation at or below 1% with 1,500-token prompts
(notebook 05, exercise 5.3) — and enforce cost with a budget.

**Cost per *correct* answer, and routing by effort.** Wrong answers are paid for too: cost per correct = cost per
request / accuracy (`workload.cost_per_correct()`). With the §5 prices ($0.007005 per call without thinking,
$0.035355 with) and illustrative accuracies — easy requests (70%) 0.95 off / 0.97 on, hard ones (30%) 0.30 off /
0.85 on — thinking everywhere reaches 0.934 at $0.037853 per correct answer, thinking only on hard requests 0.920 at
$0.016859, never thinking 0.755 at $0.009278. That is routing by effort at the gateway: the 06 layer's
[scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) prices
output (thinking included) at six times input on its model and sets thinking levels per task (§3.4, §5.5), and the
Mistral variant's reference architecture routes task × level × mode to a model, a `reasoning_effort` and an output
cap. It needs a classifier, or a cheap first pass, that knows which requests are hard.

**Speculative decoding on long outputs.** A decode-dominant workload is where speculation pays
([serving-engine primer §7](../../04-inference-engine/serving-engine/PRIMER.md#7-speculative-decoding)): one target
pass emits (1 − α^(k+1))/(1 − α) tokens — 3.36 at α = 0.8, k = 4 (`minengine.spec.expected_tokens()`) — and the output
distribution is unchanged. The limits are the same as there: at high concurrency the verify pass's extra positions
cost real compute, and acceptance on reasoning text must be measured, not assumed (verify per model).

## 8. The RL training stack in brief

**Rollouts with an inference engine inside the trainer.**

| Framework | Rollouts | Notes |
|---|---|---|
| TRL `GRPOTrainer` | `use_vllm=True`: `vllm_mode="colocate"` (vLLM in the trainer process, sharing the GPU; `vllm_gpu_memory_utilization` 0.3; sleep mode frees it during the optimizer step) or `"server"` (`vllm serve` on other GPUs, weights sent over NCCL) | train–inference log-prob mismatch corrected by truncated importance sampling (`vllm_importance_sampling_correction`, cap 3.0) (verify) |
| verl | `rollout.name`: `hf`, `vllm` or `sglang`; `rollout.n` samples per prompt; HybridFlow's `ActorRolloutRefWorker` colocates actor, rollout and reference on the same GPUs | `algorithm.adv_estimator=grpo`, `kl_loss_type=low_var_kl` (k3), `loss_agg_mode` (verify) |
| OpenRLHF | Ray + vLLM; PPO, GRPO, REINFORCE++ | named only here (unverified) |

**Weight sync.** After each optimizer step the rollout engine needs the new weights: shared memory when colocated, an
NCCL broadcast when disaggregated. verl reports that over 99% of BF16 weight bytes are unchanged step over step, so
delta sync is 1.3–21× faster (verify). With LoRA, the engine loads adapters instead of full weights — vLLM's
`--max-loras` must then cover every adapter version in flight (TRL's async guidance: at least `max_staleness` + 2;
the [vllm-internals primer §9](../../04-inference-engine/vllm-internals/vllm-internals-primer.md#9-multi-lora-multimodal-and-hybrid-models-in-brief)
explains why a low `--max-loras` starves requests).

**Async and off-policy RL.** A synchronous step idles the trainer while rollouts run and idles most of the rollout
batch while the long tail finishes (25% occupancy in §1's model). verl's one-step-off-policy trainer generates step
k + 1 while training on step k; its fully asynchronous trainer runs rollouts and training on separate GPUs and
reports 2.35–2.67× on Qwen2.5-7B with 128 GPUs; TRL's experimental `AsyncGRPOTrainer` streams completions from a vLLM
server and drops samples more than `max_staleness` (default 4) weight versions old (verify). The price is staleness:
samples come from an older policy, so the ratio ρ = π/π_old is no longer 1, and the clip (§4) and importance-sampling
corrections do real work.

**Why serving skills transfer.** The rollout side is a serving problem with a different SLO — throughput per
step instead of latency per user — and the same levers: KV capacity and preemption (a GRPO batch is G long samples
per prompt), prefix sharing (the G samples share their prompt), batching and the long tail, CUDA Graphs and weight
loading, and the metrics of the serving-engine primer §11.

**Agentic RL.** Multi-turn rollouts with tools make each trajectory a loop of generate → tool call → observation →
generate: environments must be reset and isolated per rollout (code execution belongs in a sandbox —
`07-application-agent-framework/sandboxed-execution`), tool latency adds its own tail to the step, and credit
assignment spans turns. Reward design and evals are the same discipline as 07's platform lab
([`08_evals_trajectory_judge_gates`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py):
golden sets, run-to-run noise, Wilson intervals, release gates): the training reward is a proxy and the eval is how
you notice it being gamed.

## 9. Where to run it

| To learn | T0 (laptop / Colab CPU) | T1: free Colab / Kaggle T4 | T1: rented 24 GB GPU | T3: GCP |
|---|---|---|---|---|
| §2–4 policy gradients, DPO, GRPO | `rl-core` notebooks 01–03 | `thinking-lab` 01: a tiny transformer trained from scratch with SFT then GRPO in torch (also runs on CPU, slower) | same, faster | — |
| §5, §7 a real thinking model | `rl-core` notebook 05; the lab's fake server (labelled simulated) | `Qwen/Qwen3-0.6B` in vLLM 0.30 with `--reasoning-parser qwen3 --dtype half`; `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with `deepseek_r1` | `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507` in bf16 | the 04 lab's [Cloud Run and GKE deploys](../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/) with a thinking model and `--reasoning-parser` |
| §6 test-time compute | `rl-core` notebook 04 | best-of-n and majority vote on Qwen3-0.6B (lab 03) | a 4B model | — |
| §8 one RL step with an engine | the lab's rollout bookkeeping on the toy | TRL 1.14 + vLLM 0.30, colocated, fp16, a 0.5–0.6B policy (lab 05; fit to 15 GB is verify) | same, with room | — |

A T4 has 15 GiB usable, no bf16 and no FP8; vLLM 0.30 needs compute capability 7.5 and uses the Triton attention
backend there (verify). The 04 lab's `servelab.sizing.size()` predicts 6,969 KV blocks for Qwen3-0.6B at
`max_model_len=8192` on a T4 — 13.6 concurrent 8K-token requests — and no room at all for Qwen3-8B in fp16. Kaggle's
free 2×T4 does not change the picture for thinking models; a rented 24 GB L4 or RTX 4090 runs a 4B thinking model
with room for long traces. GCP is one target, never a prerequisite: the lab adds no Terraform of its own and points at
the 04 serving lab's. Prices, quotas and obtainability: [`COMPUTE.md`](../../COMPUTE.md).

---

## In a design review

**The two-minute walkthrough.** "Post-training is three stages on the same network: SFT imitates demonstrations,
and RL raises the probability of the model's own samples in proportion to how they scored against a baseline. RL
can only reweight what the SFT model already samples — with a KL penalty the optimum is the reference times
exp(reward/β) — so we invest in the SFT model and in the reward. Where answers can be checked we use verifiable
rewards and GRPO: G samples per prompt, group-normalised advantages, no value model, a token-level loss, truncated
completions masked. Where they cannot, preferences through Bradley–Terry, with DPO when we want the KL-regularised
optimum without a reward model — and a length-controlled eval, because annotators' biases become the reward. Most of
an RL step is generation, so the rollout side is an inference-serving problem, and async rollouts trade throughput
for staleness. Trained this way, models learned to think, and that changes serving: output-heavy, heavy-tailed
traffic where concurrency scales with output length and KV per session grows while the model thinks, so memory and
the ITL SLO set the GPU count — eighteen times the GPUs for ten times the output in the capacity primer's example.
We size `max_model_len` for the p99, enforce cost with a thinking budget rather than `max_tokens`, expect lower
multi-turn prefix-cache hits because templates drop old thinking, and route effort per request class on cost per
correct answer."

**Drill questions**

1. *Training reward climbed to 99% and the held-out eval fell. What happened and what do you change?* — Reward
   hacking: the policy found where the reward and the objective disagree (in the core, a verifier's early exit took
   true accuracy from 29.7% to 4.9% while the reward reached 99.4%). Inspect high-reward failures, fix the verifier or
   reward model, and gate on the eval. A larger β only bounds the damage.
2. *Why can DPO skip the reward model, and what does it give up?* — The KL-regularised optimum gives r =
   β·log(π*/π_ref) + β·log Z; in the Bradley–Terry difference Z cancels, leaving a classification loss on the policy's
   own log-ratios. It gives up exploration (offline pairs only), optimises margins rather than likelihoods (the
   chosen log-ratio falls), and leaves no reward model to rerank or monitor with.
3. *Our GRPO run's responses keep getting longer and more hit `max_completion_length`. First checks?* — `loss_type`
   (per-sequence averaging under-weights long completions: in the core its update has cosine 0.67 with the true
   gradient and raises truncation), `mask_truncated_completions`, a soft overlong penalty, and whether the reward
   charges anything for length.
4. *We turned on thinking for our assistant at the same QPS. What happens to the fleet?* — Concurrency scales with
   output length and KV per session with average context, so memory binds: 10× output needed 18× the GPUs in the
   capacity primer's example (4.77 vs 0.26), and a tight ITL SLO caps the batch before HBM does. TTFT is unchanged;
   time to the first answer token is not.
5. *Users get empty answers from the thinking model. Why, and the fix?* — `max_tokens` counts reasoning; requests
   still thinking at the cap return `finish_reason="length"` with empty content (17.7% at a 4K cap in §7's model).
   Raise `max_tokens`/`max_model_len` for the tail and enforce cost with `thinking_token_budget` or a two-call budget.
6. *Best-of-16 with a reward model or one sample from a larger model, same budget?* — With a real verifier, sampling
   a cheaper model often wins (it covers dead ends); with a reward model the gain saturates below the verifier's and
   a biased reward model selects for its bias; with only a vote, the larger model's per-sample accuracy wins unless
   single samples are already usually right. Measure pass@1 and pass^k at equal cost on your evals.

---

## Glossary

| Term | Meaning |
|---|---|
| Policy | the model as a distribution over the next token given the context, π(a \| s) |
| Trajectory / completion | one sampled output; log π(y) is the sum of its tokens' log-probabilities |
| SFT | supervised fine-tuning: maximum likelihood on demonstrations |
| Reward model (RM) | a network trained on preference pairs to score completions (Bradley–Terry) |
| Verifier | a program that checks a completion (answer equivalence, unit tests, a format regex) |
| RLVR | reinforcement learning with verifiable rewards |
| REINFORCE | the score-function gradient E[(R − b)·∇log π(y)] |
| Baseline / advantage | b, subtracted from the reward without biasing the gradient / R − b |
| Value model | a learned V(s) giving per-token baselines (PPO's critic) |
| GAE | generalised advantage estimation: TD errors accumulated with factor γλ |
| KL penalty | −β·KL(π ‖ π_ref) in the objective; optimum π_ref·exp(R/β)/Z |
| k1 / k3 | per-sample KL estimators log(π/π_ref) and π_ref/π − log(π_ref/π) − 1 |
| PPO | policy optimisation with a clipped probability ratio ρ = π/π_old |
| RLHF | SFT, a reward model from human preferences, then PPO with a KL penalty |
| Bradley–Terry | P(A ≻ B) = σ(r_A − r_B) |
| DPO | direct preference optimisation: the KL-regularised optimum's closed form as a pairwise loss |
| Implicit reward | β·log(π/π_ref), DPO's reward (TRL's rewards/chosen, rewards/rejected) |
| IPO / KTO / ORPO | squared loss to a fixed margin / unpaired good–bad labels / reference-free SFT + odds-ratio loss |
| GRPO | group relative policy optimisation: G samples per prompt, group-normalised advantages, no critic |
| Clip-higher | a larger upper clip ε_high than lower ε_low, against entropy collapse (DAPO) |
| Dynamic sampling | dropping groups whose rewards are all equal and refilling the batch (DAPO) |
| Overlong shaping | masking truncated completions and a soft penalty near the length cap (DAPO) |
| Dr. GRPO | GRPO with a constant loss normaliser and no std scaling |
| Rollout | generating completions for training, with an inference engine |
| Reward hacking | the policy maximising the reward in a way the designer did not intend |
| Over-optimisation | proxy reward rising while the true objective falls as the policy moves away |
| Thinking model | a model that emits a reasoning block before its answer, trained (or distilled) to do so |
| Rejection-sampling SFT | fine-tuning on the model's own samples that passed a check |
| Distillation (of traces) | SFT of a smaller model on a stronger model's outputs |
| Thinking budget | a cap on reasoning tokens after which the model is made to answer |
| pass@k / pass^k | at least one of k samples correct / all k correct |
| Self-consistency | majority vote over sampled answers |
| PRM | process reward model: scores intermediate steps |
| KV-token-steps | Σ over decode steps of the tokens a request holds in KV: P·L + L(L + 1)/2 |
| Cost per correct answer | cost per request divided by accuracy |

## Sources

Papers:

- Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine
  Learning 8, 1992 — REINFORCE.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (arXiv:1506.02438);
  *Proximal Policy Optimization Algorithms* (arXiv:1707.06347); Schulman, *Approximating KL Divergence* (blog, 2020) —
  k1/k2/k3.
- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (arXiv:1706.03741); Ouyang et al., *Training
  language models to follow instructions with human feedback* (arXiv:2203.02155) — RLHF.
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs*, Biometrika, 1952.
- Rafailov et al., *Direct Preference Optimization* (arXiv:2305.18290); Azar et al., *A General Theoretical Paradigm
  to Understand Learning from Human Preferences* (arXiv:2310.12036) — IPO; Ethayarajh et al., *KTO* (arXiv:2402.01306);
  Hong et al., *ORPO* (arXiv:2403.07691).
- Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*
  (arXiv:2402.14740) — RLOO.
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (arXiv:2210.10760).
- Shao et al., *DeepSeekMath* (arXiv:2402.03300) — GRPO; DeepSeek-AI, *DeepSeek-R1* (arXiv:2501.12948).
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (arXiv:2503.14476); Liu et al.,
  *Understanding R1-Zero-Like Training: A Critical Perspective* (arXiv:2503.20783) — Dr. GRPO.
- Qwen Team, *Qwen3 Technical Report* (arXiv:2505.09388).
- Chen et al., *Evaluating Large Language Models Trained on Code* (arXiv:2107.03374) — unbiased pass@k; Yao et al.,
  *τ-bench* (arXiv:2406.12045) — pass^k; Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*
  (arXiv:2203.11171); Lightman et al., *Let's Verify Step by Step* (arXiv:2305.20050); Snell et al., *Scaling LLM
  Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (arXiv:2408.03314).
- Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (arXiv:2409.19256) — verl.

Code and documentation (read 2026-09-26):

- TRL `github.com/huggingface/trl` v1.14.0: `trl/trainer/grpo_config.py`, `grpo_trainer.py` (`_compute_advantages`,
  `_compute_loss`), `dpo_trainer.py`, `trl/rewards/`, `docs/source/grpo_trainer.md`, `async_grpo_trainer.md`,
  `reward_trainer.md`.
- vLLM `github.com/vllm-project/vllm` v0.30.0: `docs/features/reasoning_outputs.md`, `vllm/reasoning/`,
  `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/sampling_params.py` (`thinking_token_budget`),
  `vllm/config/reasoning.py`, `vllm/v1/metrics/loggers.py`.
- verl `github.com/volcengine/verl` docs: `algo/grpo.md`, `examples/config.rst`, `hybrid_flow.rst`,
  `advance/one_step_off.md`, `advance/fully_async.md`, `advance/delta_weight_sync.md`.
- DeepSeek-R1 `github.com/deepseek-ai/DeepSeek-R1` (README, paper); DAPO `github.com/BytedTsinghua-SIA/DAPO`; Qwen3
  `github.com/QwenLM/Qwen3` (README, docs: quickstart, thinking budget, vLLM deployment); gpt-oss
  `github.com/openai/gpt-oss`; SGLang docs (`separate_reasoning`); lm-evaluation-harness (`gsm8k-cot-self-consistency`);
  HumanEval `estimate_pass_at_k`.
- In this repo: [transformer primer](../transformers/docs/transformer-primer.md); [capacity primer](../gpu-capacity-planning/PRIMER.md)
  and [`capacity.py`](../gpu-capacity-planning/capacity.py); [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md);
  [vllm-internals primer](../../04-inference-engine/vllm-internals/vllm-internals-primer.md);
  [`vllm-serving-lab`](../../04-inference-engine/serving-engine/vllm-serving-lab/) (`servelab.sizing`, bench, fake
  server); [agentic scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md);
  the 07 [platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) (evals);
  `07-application-agent-framework/sandboxed-execution` (sandboxing tool calls and code).

## Verify list

Product facts in this primer, `rlcore/workload.py` and the notebooks, as of 2026-09-26 — re-check against the
versions you pin.

- **vLLM 0.30.0 reasoning support:** `--reasoning-parser` (and `--reasoning-parser-plugin`), with `--enable-reasoning`
  removed; parser names `qwen3`, `deepseek_r1`, `openai_gptoss` among ~35; the response field `reasoning` (renamed
  from `reasoning_content`) in `message` and `delta`; `include_reasoning`; `usage.completion_tokens_details.reasoning_tokens`;
  `chat_template_kwargs` and `--default-chat-template-kwargs`; `reasoning_effort` values none/minimal/low/medium/high/xhigh/max
  and its mapping to `enable_thinking` for Qwen3; `thinking_token_budget` (−1 = unlimited) with `--reasoning-config`;
  structured output after reasoning unless `enable_in_reasoning`; tool calls parsed from `content` only; no
  reasoning-specific Prometheus metric; the T4 facts (compute capability 7.5 minimum, Triton attention backend,
  `--dtype half`).
- **TRL v1.14.0:** `GRPOConfig` defaults in §4's table, the `loss_type` values (`grpo`, `dapo`, `dr_grpo`, `bnpo`,
  `cispo`, `sapo`, `luspo`, `vespo`), `scale_rewards` values, the +1e-4 and Bessel's correction in the advantage, the
  k3 KL, vLLM colocate/server modes, `vllm_gpu_memory_utilization` 0.3, the importance-sampling correction defaults;
  `DPOConfig.beta` 0.1 and `loss_type="sigmoid"`; ORPO under `trl.experimental`; `AsyncGRPOTrainer` defaults
  (`max_staleness` 4, `weight_sync_steps` 1); the claim that R1 used β = 0.001.
- **DeepSeek-R1:** R1-Zero AIME 2024 15.6% → 71.0% (86.7% majority); the four-stage recipe and the ~600k + ~200k
  sample counts; 671B total / 37B activated; distill results (Qwen-1.5B 28.9; Qwen-32B 72.6 vs RL-on-32B-base 47.0);
  rule-based rewards only; PRM and MCTS listed as unsuccessful; usage recommendations (temperature 0.6, no system prompt).
- **DAPO:** the ablation 30 → 36 → 38 → 41 → 42 → 50 (AIME 2024 avg@32); ε 0.2/0.28; 512 prompts × 16 responses;
  maximum generation 20,480 tokens; the soft overlong formula; no KL term.
- **verl:** rollouts ~70% of step time in DAPO-32B training; delta weight sync (>99% of bytes unchanged, 1.3–21×);
  fully async 2.35–2.67× on Qwen2.5-7B with 128 GPUs; config names in §8.
- **Qwen3:** model ids (0.6B, 1.7B, 4B, 8B; the 2507 Instruct/Thinking split); `enable_thinking` and the soft switches;
  the template dropping earlier turns' reasoning; recommended sampling (thinking 0.6/0.95/20, non-thinking 0.7/0.8/20);
  the thinking-budget phrase; post-training (GRPO on 3,995 query–verifier pairs; strong-to-weak distillation for
  small models). gpt-oss effort levels and the Harmony format.
- **Sizing inputs:** the GPU figures in `workload.GPUS` (H100 as the capacity primer's; L4 24 GB, 0.30 TB/s,
  121/242 TFLOP/s; T4 16 GB, 0.32 TB/s, 65 TFLOP/s); Qwen3-0.6B's config (28 layers, 8 KV heads, head_dim 128,
  0.596 B parameters); the 04 lab's T4 prediction (6,969 blocks, 13.6 × 8K); the 06 lab's example prices ($1.50 /
  $9.00 / $0.15 per 1M tokens, dated 2026-09-05 there).
- **Where to run:** Colab/Kaggle T4 availability, rented 24 GB GPU prices and GCP deploy details — maintained in
  [`COMPUTE.md`](../../COMPUTE.md) and the 04 lab's deploy READMEs.
