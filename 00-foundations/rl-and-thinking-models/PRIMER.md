# Reinforcement learning and thinking models: how post-training teaches a model to reason, and what that does to serving

*A primer for 00-foundations. Snapshot: September 2026. Each product fact has a date and the mark (verify). Each
formula has a worked number and names the function in [`rl-core/`](rl-core/) (package `rlcore`) that calculates it.*

*The numbers on the toy tasks are exact (the core goes through every possible output) or come from seeded
simulations. The serving numbers come from a model, not from measurements. That model is the formulas of the
capacity primer plus a roofline step.*

This primer explains four things:

- What occurs to a language model after pretraining. It goes through supervised fine-tuning, learning from
  preferences (reward models, RLHF, DPO) and reinforcement learning with verifiable rewards (GRPO and its
  corrections).
- How that last stage made *thinking models*. A thinking model spends thousands of tokens before it answers.
- How to spend inference compute at test time: think longer, sample more, vote, use a verifier.
- What all of this does to the serving stack below it.

This primer uses three other primers and does not repeat them:

- the [transformer primer](../transformers/docs/transformer-primer.md) (§6 training, §7 inference),
- the [capacity primer](../gpu-capacity-planning/PRIMER.md) (weights, KV bytes, TTFT and TPOT),
- the [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) (§5 prefix caching, §6 sampling,
  §7 speculative decoding, §11 measurement).

You can learn every concept at tier T0 with [`rl-core/`](rl-core/). [`thinking-lab/`](thinking-lab/) trains a small
transformer with GRPO in torch. It also serves a real thinking model in vLLM (T1).

---

## The one-minute version

Pretraining teaches a model what text looks like. **SFT** teaches it a format by imitation. The model can already do
many things, and **RL** changes which of these things it actually does.

- RL for LLMs is **sample, score, reweight**. The trainer generates completions with an inference engine. It scores
  them with a verifier or a reward model. Then it takes a gradient step. The step increases the log-probability of
  the completions that beat a **baseline** and decreases the log-probability of the others.
- A **KL penalty** to the SFT model keeps the policy near the SFT model. The optimum is the reference, reweighted by
  $\exp(\text{reward}/\beta)$.
- **Preferences** become rewards through Bradley–Terry. **DPO** inverts that closed form, and thus it does not use a
  reward model.
- Where you can check the answers, **GRPO** samples a group for each prompt. It uses the group mean as the baseline,
  and it has no value model.
- Models that got this training on math and code learned to **think**. Long chains of thought grew because they
  increased the reward.
- RL optimises exactly the reward that you write down. Thus the policy exploits loopholes, and anything that has no
  charge grows, for example length. The details of how the trainer averages the loss also bend the update.
- At inference, to think longer and to sample more are two budgets that you allocate. Only a verifier or a vote turns
  samples into accuracy.
- For serving, thinking makes traffic **decode-heavy and heavy-tailed**:
  - Concurrency scales with output length.
  - KV per session grows while the model thinks.
  - Memory and the ITL SLO set the GPU count.
  - `max_tokens` truncates answers in cases where a thinking budget does not truncate them.
  - Multi-turn prompts do not contain old thinking.
  - The metric is **cost per correct answer**.

---

## 1. From pretraining to post-training

**Three stages, three objectives.**

| Stage | Data | Objective | What it changes |
|---|---|---|---|
| Pretraining | trillions of tokens of text and code | next-token cross-entropy, with compute $C \approx 6 \cdot N \cdot D$ ([transformer primer](../transformers/docs/transformer-primer.md) §6.1–6.2) | what the model knows and can continue |
| SFT | thousands to millions of demonstrations (a prompt and a good answer) | the same cross-entropy, on the answer tokens only: imitation | format, how to obey instructions, a style of reasoning to start from |
| Preference / RL | the model's *own* samples, with their scores | increase the probability of the samples with good scores: sample, score, reweight | which behaviours the model uses, how long it thinks, refusals, tool use |

§6.3 of the transformer primer makes this point: post-training is not architecture. The network stays the same, and
the training signal changes. SFT is maximum likelihood on the outputs of someone else (`rlcore.pg.sft_step()`). RL
is maximum likelihood on the model's own outputs, **weighted by how they scored**. Thus RL can only amplify
behaviours that the model already samples. For the same reason, its first ingredient is an inference engine.

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

The core has a toy example. A weak SFT model (two steps of `pg.sft_step()` on the 14 balanced bracket strings) is
right 29.7% of the time. 150 steps of REINFORCE with 16 samples each take it to 93.4% (notebook 01, worked example
3).

**Where the compute goes: rollouts.** A training step does a forward and backward pass over every generated token.
This pass is compute-bound, at about $6 \cdot N$ FLOPs per token. But first the engine must *generate* those tokens,
one decode step at a time, and this part is memory-bound. The batch lives until its longest completion ends.

`rlcore.workload.rl_step_time()` calculates one synchronous step at the batch shape of DAPO: 512 prompts × 16
samples = 8,192 completions. The step is for a 7.6B policy on 64 H100s, with lognormal lengths (median 4,000
tokens, capped at 20,480). The result is generation 142 s, training 137 s, so rollouts are 51% of the step. The
generation batch is only 25% occupied on average. The cause is the last few long completions, which decode almost
alone. That model is optimistic. verl reports ~70% of step time in rollouts for DAPO-32B (verify).

The rollout generator is an inference engine. It has all the concerns of the
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md): continuous batching (§2), KV capacity
and preemption (§4), and prefix caching (§5). The $G$ samples of one prompt share its prefix, so the engine prefills
that prefix one time.

## 2. Policy gradients over token sequences

**The model as a policy.** At each position, the model selects a token $a$ from $\pi(a \mid s)$. The state $s$ is
everything before that position. A completion $y$ is a trajectory, and $\log \pi(y) = \sum_t \log \pi(a_t \mid s_t)$.
This is the sum of the per-token log-probabilities that an engine returns as `logprobs` (`Policy.token_logprobs()`,
`Policy.seq_logprob()`).

The core replaces the network with a table of softmaxes, $\pi(a \mid s) = \operatorname{softmax}(\theta[s])_a$. This
table keeps one fact exact (`Policy.grad_logprob()`):

$$
\frac{\partial \log \pi(a \mid s)}{\partial \theta[s, b]} = \mathbf{1}[a = b] - \pi(b \mid s)
$$

When the log-probability of one token increases, the log-probabilities of all the other tokens decrease in
proportion (the row sums to zero). Each training method in this primer is a weighted sum of these per-token
gradients. The methods differ only in the weights. Under the uniform policy, an 8-token completion has
log π(y) = 8·log(1/2) = −5.545.

**REINFORCE.** The gradient of the expected reward comes from $\nabla \pi = \pi \nabla \log \pi$ (Williams 1992,
`pg.reinforce_grad()`):

$$
\nabla_\theta \mathbb{E}_{y \sim \pi}[R(y)] = \mathbb{E}\big[R(y) \nabla \log \pi(y)\big] \approx \frac{1}{N} \sum_i (R_i - b) \nabla \log \pi(y_i)
$$

Sample, score, and push the log-probability of each completion up or down by its score. A uniformly random policy
solves the bracket task of the core (8 tokens, balanced) with probability Catalan(4)/2⁸ = 14/256 = 5.5%
(`SeqTask.random_success_rate()`). A few hundred steps of this estimator solve it.

**Variance, baselines and advantages.** $\mathbb{E}_\pi[\nabla \log \pi(y)] = \nabla \sum \pi(y) = 0$. Thus, when
you subtract any baseline $b$ that does not depend on the sample, the gradient stays unbiased. But its variance
changes. `pg.grad_variance()` at the start of a ThinkTask run gives 0.054 with no baseline, 0.029 with the batch
mean.

Add a constant +5 to every reward. This adds no information at all. Then the no-baseline variance becomes 3.678
while the baselined one stays 0.029. ${R - b}$ is the **advantage**. These are the baselines in use:

| Baseline | Used by | Note |
|---|---|---|
| batch or group mean | GRPO (§4), `pg.advantages("mean")` | the group of G samples of one prompt is the batch |
| leave-one-out mean | RLOO | exactly n/(n − 1) times the mean-baseline advantage, thus the same direction |
| learned value V(s), per token | PPO (§3) | a second network as large as the policy |

**The KL penalty to a reference model.** Maximise
$\mathbb{E}[R] - \beta \cdot \mathrm{KL}(\pi \parallel \pi_{\text{ref}})$. Here $\pi_{\text{ref}}$ is the frozen SFT
model. As reward shaping, the penalty is $R - \beta \cdot (\log \pi(y) - \log \pi_{\text{ref}}(y))$. This has exactly
the gradient of $-\beta \cdot \mathrm{KL}$, because $\mathbb{E}[\nabla \log \pi] = 0$ (the PPO-RLHF form). GRPO puts
the penalty in the loss instead (§4).

Over all distributions, the optimum has a closed form (`pg.kl_optimal()`):

$$
\pi^*(y) = \frac{\pi_{\text{ref}}(y) \exp\big(R(y)/\beta\big)}{Z}, \qquad \mathrm{KL}(\pi^* \parallel \pi_{\text{ref}}) = \frac{\mathbb{E}_{\pi^*}[R]}{\beta} - \log Z
$$

This is the result for the SFT reference, with R = 1 for a balanced string (`pg.kl_optimal()` over all 256
strings):

| β | 10 | 1 | 0.3 | 0.1 |
|---|---|---|---|---|
| E[R] = P(balanced) | 0.319 | 0.535 | 0.922 | 1.000 |
| KL(π* ‖ π_ref), nats | 0.00 | 0.12 | 0.87 | 1.21 |

The penalty exists for three reasons:

- The reward is trustworthy only near the data that it comes from.
- Fluency and language come from the reference.
- The penalty slows the collapse onto one answer.

What the penalty implies: $\pi^*$ can only move mass among the completions that $\pi_{\text{ref}}$ already produces.
A completion with $\pi_{\text{ref}}(y) = 0$ stays at 0 for every $\beta$. RL sharpens. It does not invent.

**Reward hacking.** RL optimises exactly the reward that you write down. The core's `buggy_verify` returns *pass* at
the moment when the bracket depth goes below zero. It is a harness that counts an early exit as a success. 200 of 256
strings pass it, and 14 strings have balanced brackets. The SFT reference passes it 72.7% of the time and is right
29.7%.

The table shows the result after 200 REINFORCE steps against it (`pg.train_reinforce()`, notebook 01, worked
example 6):

| Training | Passes the buggy check | Truly balanced | P(starts with `)`) | KL to $\pi_{\text{ref}}$ |
|---|---|---|---|---|
| reference | 0.727 | 0.297 | 0.18 | 0 |
| β = 0 | 0.994 | 0.049 | 0.80 | 1.06 |
| β = 0.3 | 0.959 | 0.354 | 0.25 | 0.20 |

At depth 0, `)` is an instant pass from every state, so gradient ascent finds it. The KL-regularised optimum
multiplies every string that passes by the same $\exp(1/\beta)$. Thus it keeps the reference's share of honest
strings among the strings that pass: 40.9% as β → 0 (notebook 01, exercise 1.6). The penalty is a leash, not a
correction. The corrections are the verifier and evals of the true objective. DeepSeek-R1 used rule-based rewards
and did not use neural reward models, partly for this reason (verify).

**Length bias.** RL also increases the length of anything that the reward does not charge for. In `ThinkTask`, the
policy emits "think" tokens until it answers, and $P(\text{correct} \mid L) = 1 - e_0 (1 - q)^L$ (§6 explains the
form). With $\text{reward} = \text{correct} - c \cdot L$, the optimum is where the marginal gain of a token equals
its cost (`ThinkTask.optimal_length()`):

$$
e_0 (-\ln(1 - q)) (1 - q)^{L^*} = c \quad\Rightarrow\quad L^* = \frac{\ln\big(c / (e_0 (-\ln(1 - q)))\big)}{\ln(1 - q)}
$$

e0 = 0.8, q = 0.1, c = 0.01: L* = 20.23

Start from a policy that answers at once half the time (a mean of 1.0 thinking token). Use e0 = 0.8 and q = 0.15. Then
300 REINFORCE steps grow thinking to 11.1 tokens with no cost and to 7.8 with c = 0.02 (L* = 11.5). At that point, the
table policy still climbs. The same pressure on real models is the length growth of §5. Also, §3 and §4 add two more
sources of length bias: reward models and the way the trainer averages the loss.

## 3. Learning from preferences

**Bradley–Terry reward models.** When no program can grade an answer, people compare two answers. Bradley–Terry
treats a comparison as a noisy difference of rewards (`pref.bt_prob()`, `pref.fit_bradley_terry()`):

$$
P(A \succ B) = \sigma\big(r(A) - r(B)\big), \qquad \text{loss} = -\mathbb{E}\big[\log \sigma\big(r(y^+) - r(y^-)\big)\big]
$$

Thus a reward model is logistic regression on (chosen − rejected). A fit to 4,000 synthetic comparisons from
$r = 1.5 x_1 - 0.5 x_2$ recovers (1.57, −0.54) (notebook 02). The comparisons identify only differences:
σ(2 − 1) = 0.7311 = σ(12 − 11). A reward model has no zero point. For this reason, TRL's `RewardTrainer` has
`center_rewards_coefficient`, and scores from different runs do not compare.

**RLHF with PPO, in brief.** The InstructGPT recipe is SFT, then a reward model fit on human comparisons, then PPO
against that reward model with a per-token KL penalty. PPO has these parts:

- the **clipped ratio** $-\min\big(\rho A, \operatorname{clip}(\rho, 1 - \varepsilon, 1 + \varepsilon) A\big)$,
  with $\rho = \pi/\pi_{\text{old}}$. It lets several gradient steps use one batch of samples again, and the policy
  does not move far (§4 gives a worked example).
- a **value model** ${V(s)}$, typically as large as the policy. It gives each token its own baseline through
  **generalised advantage estimation** (`pref.gae()`):

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad A_t = \delta_t + \gamma \lambda A_{t+1}
$$

rewards (0, 0, 1), values (0.5, 0.6, 0.8), γ = 1: λ = 0 → (0.1, 0.2, 0.2); λ = 0.5 → (0.25, 0.3, 0.2);
λ = 1 → (0.5, 0.4, 0.2)

$\lambda = 0$ trusts $V$ (low variance, but biased when $V$ is incorrect). $\lambda = 1$ trusts the sampled return.
In RLHF, the reward-model score sits on the last token, and $-\beta \log(\pi/\pi_{\text{ref}})$ sits on every token.
Four networks are in memory: the policy, the reference, the reward model and the value model. DPO and GRPO each
remove part of this cost.

**DPO: the closed form, as a classification loss.** You can invert the optimum of §2 (`pref.dpo_loss()`):

$$
\begin{aligned}
\pi^*(y \mid x) &= \pi_{\text{ref}}(y \mid x) \exp\big(r(x, y)/\beta\big) / Z(x) \\
\Rightarrow \quad r(x, y) &= \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) \\
\Rightarrow \quad P(y^+ \succ y^-) &= \sigma(r^+ - r^-) = \sigma\Big(\beta \Big[\log \frac{\pi^*(y^+)}{\pi_{\text{ref}}(y^+)} - \log \frac{\pi^*(y^-)}{\pi_{\text{ref}}(y^-)}\Big]\Big) \qquad \big(Z(x) \text{ cancels}\big) \\
L_{\text{DPO}}(\theta) &= -\mathbb{E} \log \sigma\Big(\beta \Big[\big(\log \pi_\theta(y^+) - \log \pi_{\text{ref}}(y^+)\big) - \big(\log \pi_\theta(y^-) - \log \pi_{\text{ref}}(y^-)\big)\Big]\Big)
\end{aligned}
$$

Train the policy directly on pairs, with no reward model and no sampling. TRL's `DPOTrainer`
(`loss_type="sigmoid"`, default β = 0.1, log-probabilities summed over completion tokens) calculates exactly this.
Worked number: with β = 0.1, the chosen log-ratio +1 and the rejected −1 give −log σ(0.2) = 0.598139; zero margin
gives ln 2 = 0.693147. Every run starts at that point.

The per-pair gradient weight is $\beta \cdot \sigma(-m)$, with $m$ the scaled margin (`pref.dpo_step()`). Mis-ranked
pairs get up to $\beta$. Pairs that the policy ranks with confidence get almost nothing. The **implicit reward**
$\beta \log(\pi/\pi_{\text{ref}})$ is what TRL logs as `rewards/chosen` and `rewards/rejected`.

Notebook 02 does an end-to-end check of the claim. The preferences over bracket strings come from a true reward
$r = 3 \cdot \text{balanced}$. There are 4,096 pairs from the SFT reference (59% are ties, labelled by a coin), with
β = 1:

| Path | P(balanced) |
|---|---|
| reference | 0.297 |
| closed form $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$ (`pg.kl_optimal()`) | 0.895 |
| RLHF: reward model r̂ = 2.94·balanced, then REINFORCE with β = 1 | 0.885 |
| DPO, 150 epochs (`pref.dpo_step()`), KL(π_DPO ‖ π*) = 0.0094 | 0.889 |

The implicit reward separates balanced from unbalanced strings by 2.94 (true gap 3). At the end, TRL's metrics read
rewards/chosen −0.407, rewards/rejected −1.524, margins +1.118, accuracies 0.705. Thus the log-ratios of the chosen
strings *decreased*.

**What DPO gives up.**

- **It is offline.** It learns from the support of the pairs. It cannot explore. Its effect on completions outside
  the pairs is whatever the generalisation of the network does. Iterated or "online" DPO samples new pairs from the
  current policy to close part of the gap.
- **It optimises the margin, not the likelihood.** The policy wins the loss when it decreases both log-ratios, and
  decreases the rejected one faster. Thus rewards/chosen drifts to negative values (−0.407 in the result before this
  list). A chosen log-probability that collapses is a sign of too much training.
- **Deterministic preferences push the margin without bound.** $-\log \sigma(\beta \cdot \text{margin})$ continues
  to pay for a wider margin at every $\beta$. Thus the loss drives $\pi(\text{rejected})$ towards 0, whatever $\beta$
  is, and the KL term no longer regularises. $\beta$ only rescales how fast the loss saturates. In practice, the
  brakes are an early stop, or the constant target margin of IPO (in the next list).
- **No reward model is left behind.** Thus you have no reward model to rerank samples, to monitor drift or to use
  again for best-of-n.

**Three variants, one line each.**

- **IPO**: a squared loss toward a constant margin 1/(2β) (5 at β = 0.1, `pref.ipo_loss()`). Thus the margin cannot
  grow without bound.
- **KTO**: unpaired thumbs-up/thumbs-down labels with a prospect-theory utility and loss aversion. It needs no pairs.
- **ORPO**: no reference model. It adds a log-odds-ratio penalty on the rejected answer to the SFT loss, in one stage
  (experimental in TRL).

**Reward-model over-optimisation, and length.** Any bias in the annotators becomes the reward.
`pref.response_catalogue()` holds 40 answers. Each answer has five versions, padded with 0–800 filler tokens that
decrease true quality by 0.3 per 100 tokens. The reference rarely pads. Annotators prefer quality *and* length:
$P(A \succ B) = \sigma(\Delta\text{quality} + 0.6 \cdot \Delta\text{length}/100)$ (`pref.length_biased_prefs()`).

A reward model fit on 3,000 such pairs learns r̂ = 1.01·quality + 0.61·length/100, and it learns it faithfully. The
table shows what occurs when you optimise that reward model harder (smaller β, closed form):

| β | ref | 10 | 1 | 0.5 | 0.3 | 0.1 |
|---|---|---|---|---|---|---|
| KL, nats | 0 | 0.00 | 0.37 | 1.68 | 5.42 | 8.76 |
| reward-model score | 1.09 | 1.16 | 1.83 | 2.69 | 4.10 | 4.86 |
| true quality | −0.280 | −0.229 | 0.138 | 0.136 | −0.668 | −0.957 |
| mean length (tokens) | 223 | 227 | 275 | 416 | 778 | 950 |

The proxy increases monotonically. True quality has a peak (best at β = 0.7, a KL of 0.78 nats, notebook 02,
exercise 2.5). Then true quality decreases below the reference, while the answers grow four times longer. If you
optimise the true quality instead, the policy reaches 1.435 at 153 tokens.

That shape is the over-optimisation of Gao et al.: the proxy goes up, and the gold goes up and then down. Length is
its most common form. The defences:

- a KL budget (stop early),
- length control when you collect data and when you judge outputs,
- verifiable rewards where they exist,
- evals on the true objective.

## 4. RL with verifiable rewards and GRPO

**Verifiers.** RLVR replaces the reward model with a program:

| Task | Verifier | Watch for |
|---|---|---|
| math | extract the final answer (R1: a `\boxed{}` answer), then check equivalence (TRL `accuracy_reward` uses `math-verify`) | extraction failures with the score "incorrect", answers that exploit the parser |
| code | run the tests in a sandbox | exit-code and harness loopholes (§2). Run untrusted code in isolation ([sandboxed-execution primer](../../07-application-agent-framework/sandboxed-execution/PRIMER.md)) |
| format | a regex, for example TRL's `think_format_reward` `^<think>(?!.*<think>)(.*?)</think>.*$` | the format becomes the goal |

DeepSeek-R1-Zero used exactly two rule-based rewards, accuracy and format, and no neural reward model (verify). DAPO
scores +1 / −1 by answer equivalence. The rewards are sparse (one number per completion) and binary. This is what
makes the group statistics of GRPO work. On-policy distillation is the dense-reward cousin: a teacher scores every
token that the student samples ([distillation §4](../distillation/PRIMER.md#4-on-policy-distillation)).

**GRPO.** For each prompt, sample a group of $G$ completions from $\pi_{\text{old}}$ and score them. Give every token
of completion $i$ the same advantage. This is the GRPO of DeepSeekMath, which R1 adopted, and TRL's `GRPOTrainer`.

DeepSeekMath writes the ratio per token and averages the tokens of each completion $(1/\lvert o_i \rvert)$. The R1
paper's eq. 1 writes one ratio per whole completion, $\pi_\theta(o_i)/\pi_{\text{old}}(o_i)$, averaged over the group
$(1/{G})$:

$$
\begin{aligned}
A_i &= \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + 10^{-4}} && \text{std with Bessel's correction} \\
\rho &= \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} && \text{per token } t \text{ of completion } i \\
\text{loss} &= -\min\big(\rho A_i, \operatorname{clip}(\rho, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}}) A_i\big) + \beta \cdot \text{k3} \\
\text{k3} &= \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1
\end{aligned}
$$

`grpo.group_advantages()`, `grpo.clipped_surrogate()` and `grpo.k3()` calculate these terms.

The group is the baseline. Thus there is no critic "typically the same size as the policy model" (R1).

Worked advantages: rewards [1, 0, 0, 1] → ±0.865875; [1, 0, 0, 0] → [1.4997, −0.4999, −0.4999, −0.4999];
[1, 1, 1, 1] → 0.

**A group that is all correct or all incorrect teaches nothing**, whatever the number of tokens that it cost. TRL
logs the share of such groups as `frac_reward_zero_std`. With TRL's default `num_iterations=1`, the trainer uses the
batch for one step with the weights that sampled it. Then $\rho \equiv 1$, and the clip never binds. The clip
matters with $\mu > 1$ or with stale, off-policy rollouts (§8).

**k3, the KL estimator.** GRPO estimates $\mathrm{KL}(\pi \parallel \pi_{\text{ref}})$ per token from samples of
$\pi$. The simple $\text{k1} = \log(\pi/\pi_{\text{ref}})$ has no bias, but it is noisy and often negative. The
estimator k3 has no bias, is never negative $(e^d \ge 1 + d)$ and is quieter (Schulman 2020).

Worked values: $\log(\pi_{\text{ref}}/\pi)$ = 0.1 → 0.0051709; −0.1 → 0.0048374; 0.5 → 0.1487213.

Take two three-way distributions with KL 0.1168. 100,000 samples give k1 mean 0.1186 with std 0.460 (minimum −0.693)
and k3 mean 0.1163 with std 0.106 (`grpo.k1()`, `grpo.k3()`, notebook 03). TRL's default is β = 0, and then TRL
loads no reference model at all. The R1 paper's setting is β = 0.001, as TRL's docs say (verify).

### Clipping, entropy and the length bias

**The clip, clip-higher and entropy.** The clip is multiplicative: in one update, a token with
$\pi_{\text{old}} = 0.01$ can rise to at most 0.0120 with ε = 0.2 (0.0128 with ε_high = 0.28). A token at 0.5 can
reach 0.6000. The clip caps rare tokens hardest, and rare tokens are the exploration. Thus a symmetric clip drives
entropy down. DAPO's **clip-higher** decouples $\varepsilon_{\text{low}} = 0.2$ from $\varepsilon_{\text{high}} = 0.28$.

The toy shows the collapse itself. GRPO on the bracket task ($\mu = 4$, `grpo.train_grpo()`) takes P(correct) from
0.297 to 0.995. In the same run, the effective number of distinct correct answers, $\exp(\text{entropy})$ over them,
falls from 13.9 to 2.4. The effect of clip-higher on the entropy of real models comes from the measurement of DAPO.
The table policy collapses with or without clip-higher.

The other lever is an entropy bonus. The core adds a sequence-level $-c \log \pi(y)$ to the reward
(`pg.reinforce_grad(entropy_coef=…)`). Thus the baseline applies to the bonus as it applies to the reward. 300
REINFORCE steps keep 12.8 effective correct answers with c = 0.05 against 6.5 without, at P(correct) 0.959 against
0.975.

TRL's `entropy_coef` (default 0.0, verify) has the same intent, but it adds a different term to the loss. The term
is the mean per-token entropy of the full next-token distribution, with no baseline and no loss-type rescale. It is a
different estimator on a different scale. Thus c = 0.05 here is not a TRL setting.

**How the loss is averaged: the length bias.** TRL's `loss_type` decides what one token is worth
(`grpo.token_weights()`). The table gives the weights for completions of 10 and 50 tokens in a group of two, with
`max_len` 100:

| `loss_type` | Normaliser | Per-token weight, 10-token / 50-token completion |
|---|---|---|
| `"grpo"` | the mean over the tokens of each completion, then over completions: $1/(G \cdot \lvert o_i \rvert)$ | 0.0500 / 0.0100 |
| `"dapo"` (TRL default) | all tokens in the batch have the same weight: $1/\sum \lvert o_i \rvert$ | 0.0167 / 0.0167 |
| `"dr_grpo"` | a constant, $1/(G \cdot \text{max_len})$ | 0.0050 / 0.0050 |

A per-sequence average favours short correct answers. It punishes long incorrect answers *less per token*, and
truncated answers too (the argument of Dr. GRPO).

`grpo.expected_update()` averages 600 one-group updates at a constant ThinkTask policy. The policy answers with
probability 0.1 per step, which gives 18.5% of completions truncated at 16 tokens. The function compares the
direction with the exact gradient of accuracy: cosine 0.67 for `"grpo"`, 0.99 for `"dapo"`, 1.00 for `"dr_grpo"`.
Along the per-sequence update, truncation *rises*, and thinking grows two to four times faster. Along the other
updates, truncation decreases.

Dr. GRPO also removes the division by the std (`scale_rewards="none"`). Take a prompt that the policy solves 7 times
in 8. The one miss gets −2.47, a miss on a 50/50 prompt −0.94. Thus the division by the std gives more weight to
prompts that the policy nearly always solves or nearly never solves (a "difficulty bias").

### DAPO's fixes and TRL's defaults

**DAPO's fixes.** DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) trains Qwen2.5-32B base with four
changes and no KL term:

| Technique | What it does | Core / TRL |
|---|---|---|
| clip-higher | $\varepsilon_{\text{low}}$ 0.2, $\varepsilon_{\text{high}}$ 0.28 | `GRPOConfig(epsilon_high=0.28)` |
| dynamic sampling | over-sample, drop the groups with all-equal rewards, refill the batch | `GRPOConfig(dynamic_sampling=True)` in the core. No TRL flag (TRL logs `frac_reward_zero_std`) |
| token-level loss | $1/\sum \lvert o_i \rvert$ | `loss_type="dapo"` |
| overlong shaping | mask truncated completions, and a soft penalty near the cap | `mask_truncated_completions=True`, `grpo.soft_overlong_penalty()`, TRL `get_soft_overlong_punishment` |

The soft penalty is 0 up to $L_{\max} - L_{\text{cache}}$. It decreases linearly to −1 at $L_{\max}$, and it is −1
after that. With $L_{\max}$ 100 and a cache of 20, lengths 80, 90, 100, 101 score 0, −0.5, −1.0, −1.

Take a nine-prompt ThinkTask dataset with four prompts per step. Without dynamic sampling, 59% of generated groups
are silent. With dynamic sampling, every group that the trainer uses is informative, at 8.2 groups generated per step
instead of 4.0 (notebook 03).

DAPO's ablation (AIME 2024 avg@32) adds one technique at a time:

- "naive GRPO": 30,
- plus overlong filtering: 36,
- plus clip-higher: 38,
- plus soft overlong punishment: 41,
- plus token-level loss: 42,
- plus dynamic sampling: 50 (verify).

The DAPO run used 512 prompts × 16 responses and a maximum generation of 20,480 tokens (16,384 + a 4,096 cache).

**Compute.** Each step has these costs:

- $G\times$ generation, the dominant cost (§1),
- a forward pass for the log-probabilities of $\pi_{\text{ref}}$, if $\beta > 0$,
- the same for $\pi_{\text{old}}$, if $\mu > 1$ (or the trainer takes them from the rollout engine, and this is
  where train–inference mismatch enters, §8),
- one forward + backward pass of the policy.

The memory holds the policy with gradients and optimizer state, the reference if $\beta > 0$, and the rollout
engine's weights and KV cache. The rollout engine is on the same GPUs (colocated) or on other GPUs.

**TRL's `GRPOTrainer` as the concrete reference** (v1.14.0 field defaults, verify):

| Field | Default | Paper setting |
|---|---|---|
| `num_generations` (G) | 8 | DAPO 16 |
| `max_completion_length` | 512 | DAPO 20,480 |
| `beta` | 0.0 (no reference model) | R1 0.001, DAPO 0 |
| `epsilon` / `epsilon_high` | 0.2 / None (= epsilon) | DAPO 0.2 / 0.28 |
| `loss_type` | `"dapo"` | DeepSeekMath (and R1) `"grpo"`, Dr. GRPO `"dr_grpo"` |
| `scale_rewards` | `"group"` | Dr. GRPO `"none"` |
| `num_iterations` (μ) | 1 | — |
| `mask_truncated_completions` | False | DAPO True |
| `use_vllm` / `vllm_mode` | False / `"colocate"` | `"server"` puts vLLM on separate GPUs |
| `bf16` | True unless you set `fp16` | a T4 has no bf16: `bf16=False, fp16=True` |

The defaults are not the GRPO of the R1 paper and not DAPO. When you reproduce one of them, set the fields
explicitly (notebook 03, exercise 3.6).

## 5. Thinking models

**Long chain-of-thought as a learned behaviour.** DeepSeek-R1-Zero applied GRPO directly to DeepSeek-V3-Base. It
used rule-based accuracy and format rewards, and a template that asks for reasoning inside `<think>` tags. Over
thousands of RL steps, AIME 2024 pass@1 rose from 15.6% to 71.0% (86.7% with a majority vote).

The responses grew from hundreds to thousands of reasoning tokens. Reflection ("wait…") appeared by itself. The
paper calls this the "aha moment". The training also produced endless repetition, poor readability and a mix of
languages (verify).

Nobody gave a reward for length. Longer thinking increased the reward, so RL made the thinking longer. This is the
ThinkTask of §2 at a small scale (from 1.0 to 11.1 tokens).

But not all growth is capability. A per-sequence loss average makes incorrect answers longer too (§4). Dr. GRPO
argues that part of the length growth in R1-Zero-like training is this artefact. Thus examine accuracy by length
before you give the credit to longer traces.

**The DeepSeek-R1 recipe** (two SFT stages and two RL stages, and R1 is 671B total / 37B activated, verify):

| Stage | Data | Purpose |
|---|---|---|
| 1. cold-start SFT | thousands of long-CoT examples | readable reasoning to start RL from |
| 2. reasoning RL | the same as R1-Zero, plus a language-consistency reward | reasoning ability |
| 3. rejection-sampling SFT | ~600k reasoning samples, kept only if correct, + ~200k samples without reasoning = ~800k. 2 epochs on V3-Base | move the gains of RL into a clean SFT model. General skills |
| 4. RL for all scenarios | rule rewards for reasoning, reward models for helpfulness and harmlessness | alignment across tasks |

Stage 3 is itself a length pressure. When thinking helps, correct answers are longer on average. Thus, if the model
imitates only the survivors, its thinking grows longer with no RL at all. On ThinkTask, each round is "sample 512,
keep the correct ones, 20 SFT steps" (`pg.sft_step()`, notebook 05). Three such rounds take mean thinking from 1.0
to 1.49, 2.28 and 2.77 tokens and accuracy from 0.304 to 0.462.

**Distillation of traces into small models.** The R1 distills (Qwen2.5 1.5B–32B, Llama 8B and 70B) got their
training from SFT alone on the ~800k samples, with no RL stage. DeepSeek-R1-Distill-Qwen-1.5B reports AIME 2024
pass@1 28.9. Also, distillation beat RL. RL on Qwen-32B-Base for over 10K steps reached 47.0 on AIME 2024, and the
distilled 32B reached 72.6 (verify). The small dense models of Qwen3 (0.6B–14B) also come from strong-to-weak
distillation.

In the toy, a new student learns by SFT on 1,000 traces of the RL-trained ThinkTask teacher. The student reaches
accuracy 0.848 against the teacher's 0.855, with the same mean thinking length, 11.1 tokens. It copies the behaviour
from the outputs alone. The [distillation primer](../distillation/PRIMER.md) works through this:

- sequence-level distillation and its exposure bias
  ([§3](../distillation/PRIMER.md#3-sequence-level-distillation-learning-from-the-teachers-outputs)),
- on-policy distillation ([§4](../distillation/PRIMER.md#4-on-policy-distillation)),
- what a distilled thinking model inherits ([§5](../distillation/PRIMER.md#5-distilling-reasoning)).

**Hybrid thinking modes and chat-template switches.** The original release of Qwen3 is hybrid. The `enable_thinking`
switch of the chat template (default True) makes it think or not think. `enable_thinking=False` appends an empty
`<think>\n\n</think>\n\n` block to the generation prompt. With the soft switches `/think` and `/no_think` in a
message, the model obeys the latest instruction.

The 2507 releases divide the modes into separate `-Instruct-2507` and `-Thinking-2507` models. Through vLLM, the
switch is `chat_template_kwargs: {"enable_thinking": false}` per request, or `--default-chat-template-kwargs` for
the server (verify). The same templates drop the thinking of earlier turns from the prompt (§7).

**Thinking budgets and `reasoning_effort`.**

| Control | What it does | Watch for |
|---|---|---|
| `max_tokens` | caps reasoning + answer together | the output ends inside `<think>`: empty content, `finish_reason="length"` (§7) |
| vLLM `thinking_token_budget` (request) | When reasoning reaches the budget, it forces the end-of-think string. It needs `--reasoning-parser` (and optionally `--reasoning-config`). −1 = unlimited | vLLM-specific (verify) |
| Qwen's two-call recipe | Call 1 with `max_tokens` = budget. If the model still thinks, append a text and continue. The text is "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n" | two requests, and the second request does the prefill again |
| `reasoning_effort` | vLLM accepts none … max. For Qwen3, any value but `"none"` only sets `enable_thinking=True` | on Qwen3 it is a switch, not a length control (verify) |
| gpt-oss effort | low / medium / high, written into the system prompt (Harmony format, default medium) | other values give an error (verify) |

**A thinking token is billed as an output token.** `completion_tokens` counts reasoning plus answer. vLLM fills
`usage.completion_tokens_details.reasoning_tokens` when a reasoning parser is on. `include_reasoning: false` hides
the reasoning, but the engine still generates it (verify).

The prices that follow are the example prices of the 06 scaling lab. A call with 5,000 input tokens (2,700 cached)
and 350 output tokens costs $0.007005; the same call with 3,500 output tokens costs $0.035355, 5.05×.
`workload.api_cost()` calculates these costs and reproduces that lab's `cost_per_call`.

The recommended sampling is also different:

- Qwen3 thinking mode: temperature 0.6, top-p 0.95, top-k 20. Greedy decoding "can lead to performance degradation
  and endless repetitions".
- DeepSeek-R1: temperature 0.6 and no system prompt (verify).

The serving-engine primer §6 explains the mechanics of sampling.

## 6. Test-time compute

**Two kinds of hard.** `rlcore.ttc` reads $P(\text{correct} \mid L) = 1 - e_0 (1 - q)^L$ in this way. Each thinking
token cracks the problem with probability $q$. An uncracked answer is a guess, and the guess is correct with
probability $1 - e_0$.

The same module adds a second axis (`ttc.sample_accuracy()`). With probability $a$, an attempt starts on an approach
that can work at all. Thus one sample is correct with this probability:

$$
P(\text{correct} \mid L) = 1 - e_0 \Big(1 - a \big(1 - (1 - q)^L\big)\Big)
$$

With ${a = 1}$, this formula gives the ThinkTask formula again. `ttc.question_set()` draws 400 questions with
$a \sim \mathrm{Beta}(2, 1)$ (mean 0.65), q lognormal (median crack length 1,337 tokens) and $e_0 = 1$, for
open-ended answers with no lucky guesses.

**Sequential** compute (a longer think) helps questions that are slow to crack. **Parallel** compute (more samples)
helps questions where an attempt can come to a dead end, *if* something can select the correct sample.

**Sequential: diminishing returns.** The accuracy of one sample against thinking length (`ttc.accuracy()`):

| L (tokens) | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---|---|---|---|---|---|
| accuracy | 0.230 | 0.342 | 0.457 | 0.550 | 0.610 | 0.639 |
| gain per 1K tokens | +0.461 | +0.224 | +0.115 | +0.047 | +0.015 | +0.004 |

Accuracy levels off near 0.65, the share of attempts that start on a workable approach. Thinking cannot rescue a
dead end. Also, a constant budget over-thinks. Compare it with a model that stops when it cracks a question. The
constant budget spends 41% of a fixed 4,000-token think after the model already has the answer (notebook 04,
exercise 4.6). This is the case for adaptive budgets.

**Parallel: best-of-n, verifiers and votes.** With a perfect verifier, $n$ samples give $1 - (1 - p)^n$. A reward
model sees correctness through noise (`ttc.best_of_n_accuracy()`, Monte Carlo, 20,000 trials, p = 0.3). At n = 16,
the result is 0.997 with a verifier (exactly 1 − 0.7^16), 0.934 with noise 0.5, 0.702 with noise 1.0. More samples
give the noise more chances to fool the selection. A *biased* scorer (§3) selects for its bias.

**Majority vote** (self-consistency) needs no checker. It needs only that the correct answer is the most common
answer (`ttc.majority_accuracy()`, exact). Take one question, where a sample is correct with p = 0.4:

| Where the incorrect 60% goes | n = 1 | n = 15 | n = 31 |
|---|---|---|---|
| one dominant misconception (0.5 / 0.1) | 0.400 | 0.340 | 0.278 |
| a narrow misconception (0.42 / 0.18) | 0.400 | 0.449 | 0.447 |
| two equal incorrect answers (0.3 / 0.3) | 0.400 | 0.534 | 0.621 |
| four scattered incorrect answers (0.15 each) | 0.400 | 0.780 | 0.925 |

As $n \to \infty$, the vote is correct if and only if $p$ is more than the share of each incorrect answer. But the
limit can be slow. Against a narrow misconception, the vote still helps at 15 votes. It drops below one sample's
accuracy only past about 130. A dominant misconception makes every extra vote cost accuracy. The GSM8K
self-consistency task of lm-eval reports maj@64 from 64 samples at temperature 0.2 (verify).

**pass@k vs pass^k, and the unbiased estimator.** pass@k is the probability that at least one of $k$ samples is
correct. You must estimate it from $n \ge k$ samples with $c$ correct (Chen et al. 2021, `ttc.pass_at_k()`):

$$
\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} = 1 - \prod_{i = n - c + 1}^{n} \Big(1 - \frac{k}{i}\Big)
$$

n = 10, c = 3: pass@1 = 0.3, pass@5 = 0.916667, pass@8 = 1.0 (n − c < k); n = 64, c = 16: pass@8 = 0.914746

The plug-in $1 - (1 - c/n)^k$ has a downward bias. The bias is worst where pass@k is most informative. For p = 0.1,
n = 10, k = 5, its expectation is 0.3485 against a truth of 0.4095. The unbiased estimator matches the truth exactly
(`ttc.expected_estimate()`).

**pass^k** is the probability that all $k$ samples succeed, $\binom{c}{k}/\binom{n}{k}$. It is the reliability
metric of τ-bench, and the metric for the judgement of an agent flow. For n = 16, c = 4, the result is
pass@4 = 0.728022 but pass^4 = 0.000549 (`ttc.pass_hat_k()`). More samples raise pass@k and do nothing for pass^k.

Report each metric with confidence intervals. Use the Wilson interval of the 07 platform lab's evals notebook
(`agentlab.evals.wilson_interval`,
[`08_evals_trajectory_judge_gates`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py)).

**Search with process reward models, in brief.** An outcome reward scores the final answer, and a **process reward
model** (PRM) scores each step. This lets you do a beam search or a lookahead over partial solutions, and rerank whole
solutions (Lightman et al. 2023, Snell et al. 2024).

R1's authors report PRMs and MCTS as unsuccessful for large-scale RL. Steps are hard to define, labels are high-cost,
and reward hacking is the result. But the authors also note that PRMs stay useful for reranking and guided search
(verify).

**Compute-optimal allocation.** Divide a per-question budget as $n \cdot (L + 50) \le B$ (`ttc.allocate()`):

| Questions | Picker | Best (n, L) at 1K / 4K / 16K tokens |
|---|---|---|
| only slow to crack (a = 1) | verifier or vote | (1, 950) / (1, 3950) / (1, 15950) |
| slow or dead-end (a ~ Beta(2, 1)) | verifier | (2, 450) / (8, 450) / (16, 950) |
| slow or dead-end | majority vote | (1, 950) / (1, 3950) / (4, 3950) |

When length is the only difficulty, n short attempts equal one long attempt minus the answer overheads (the crack
rate is memoryless). Thus thinking wins. When attempts can come to a dead end, a verifier makes sampling the better
buy as the budget grows. A vote pays only when single samples usually win the vote. Here, that starts at 16,000
tokens.

**A smaller model with more samples**: take a model that costs a fifth as much per token (and is weaker on both
axes). Give it five times the tokens. With a verifier, it beats the larger model at equal compute (0.586 vs 0.502 at 1K
large-model tokens). With a vote, it loses (0.465 vs 0.479 at 1K, 0.573 vs 0.705 at 4K). The task, and the presence
of a checker, decide if it wins. Measure it on your evals.

## 7. What thinking does to serving

**Output-heavy, decode-dominant, heavy-tailed.** A chat request is prompt-heavy. A thinking request spends most of
its life in decode, and its length depends on how hard the question is. A lognormal with median 1,500 thinking
tokens and $\sigma = 1$ has mean 2,473, p90 5,403 and p99 15,361 (`workload.lognormal_mean()`,
`workload.lognormal_quantile()`). The p99 equals ten times the median.

On a small model, the KV of one trace is the size of the model. Qwen3-0.6B holds 115 kB of KV per token in fp16
(114,688 B, `workload.kv_per_token_kb()`). Thus an 8K-token trace holds 0.94 GB against 1.19 GB of weights.

**The KV working set grows with the square of the output.** At decode step $t$, a request holds ${P + t}$ tokens of
KV, for $L$ steps (`workload.kv_token_steps()`):

$$
\text{KV-token-steps} = \sum_{t=1}^{L} (P + t) = P \cdot L + \frac{L(L + 1)}{2}
$$

P = 1,500: L = 300 → 495,150; L = 3,000 → 9,001,500 = 18.2×; L = 1,500 → 6.8×; L = 8,000 → 88.9×

Compare with a 300-token answer on a 1,500-token prompt. For 1,000–3,000 output tokens, memory × time per request
grows 4–18× (4.0× at 1,000, 6.8× at 1,500, 18.2× at 3,000). The peak KV per request grows less (4,500 against 1,800
tokens: 2.5×). The large increase is in the time for which the request holds that KV.

### Sizing a fleet: Little's law, HBM and the ITL SLO

**The capacity primer's formulas with long outputs.** The [capacity primer](../gpu-capacity-planning/PRIMER.md)
calculates the fleet for an internal assistant with [`capacity.py`](../gpu-capacity-planning/capacity.py). The inputs
are 8.33 requests/s, 1,500 tokens in, 300 out, 40 ms TPOT, Mistral Small 3 (24B) in FP8 on H100s. `workload.plan()`
restates those formulas (Little's law, then GPUs per constraint). It reproduces the primer's numbers in a test.

It also adds what `decode_aggregate` leaves out. HBM and the ITL SLO cap the batch per GPU, and the step is
$\max(\text{bytes}/\text{bandwidth}, \text{FLOPs}/\text{peak})$. Like the primer, it assumes that **every output
token takes the SLO's TPOT**. Thus a request lives TTFT + out × 40 ms, whatever the load is.

`workload.plan_steady()` drops that assumption. In it, the lifetime of a request comes from the step at which the
fleet actually runs. That step depends on the batch that the lifetime produces.

Read Little's law backwards: $N = \text{rps} \cdot (\text{TTFT} + \text{out} \cdot \text{step}(b))/b$ GPUs hold a
steady batch of $b$ per GPU (`workload.gpus_for_batch()`). $N$ decreases as $b$ grows. Thus the fewest GPUs that
each cap permits is $N$ at that cap. The model assumes that chunked prefill rides in the memory-bound decode steps.
Thus it calculates prefill separately, as in `plan()`.

| | No thinking (the primer) | 2,700 thinking + 300 answer | Same, 20 ms ITL SLO |
|---|---|---|---|
| **At the SLO's TPOT (`plan()`, the primer's convention)** | | | |
| request lifetime | 12.07 s | 120.07 s | 60.07 s |
| live requests (Little's law) | 100.6 | 1,000.6 | 500.6 |
| average context; KV per session (FP8) | 1,650; 0.135 GB | 3,000; 0.246 GB | 3,000; 0.246 GB |
| sessions per GPU by HBM | 355.1 | 195.3 | 195.3 |
| batch per GPU within the ITL SLO | 813 | 447 | 174 |
| GPUs by memory / ITL / decode / prefill | 0.28 / 0.12 / 0.28 / 0.61 | 5.12 / 2.24 / 2.75 / 0.61 | 2.56 / 2.88 / 2.86 / 0.61 |
| GPUs needed (binding) | 1 (prefill) | 6 (memory) | 3 (ITL) |
| **At the step the fleet runs at (`plan_steady()`)** | | | |
| GPUs by memory / ITL / prefill | 0.15 / 0.12 / 0.61 | 2.75 / 2.24 / 0.61 | 2.75 / 2.87 / 0.61 |
| GPUs needed (binding) | 1 (prefill) | 3 (memory) | 3 (ITL) |
| where it settles: batch per GPU; step; lifetime | 20.6; 8.0 ms; 2.47 s | 154.1; 18.5 ms; 55.49 s | 154.1; 18.5 ms; 55.49 s |

The convention has a paradox in its last column. A 20 ms SLO halves the lifetime that the convention assumes. Thus
it needs *fewer* GPUs (3) than the looser 40 ms SLO (6). A fleet does not run at its SLO.

At 3 GPUs, the batch settles at 154 per GPU and a step takes 18.5 ms, under both SLOs. Thus both SLOs need 3 GPUs.
The need of the tight SLO is the larger one (2.87 vs 2.75), as it should be. The convention calculates the fleet for
the slowest step that the SLO permits. This gives a safe margin for bursts, but it is the incorrect tool to compare
SLOs.

With either method, ten times the output needs eighteen times the GPUs for memory. The numbers are 5.12 vs 0.283 at
the SLO's TPOT, 2.75 vs 0.15 at the step the fleet runs at. Concurrency grows with the output, and each live session
holds 1.8× the KV. This is the 18.2× of KV-token-steps from earlier in this section, which shows through Little's
law.

The primer's `decode_aggregate` puts all 1,000 live requests of the convention in one batch. That is 246 GB of KV on
an 80 GB card. Decode *throughput* is not the binding constraint.

**ITL is the binding SLO; TTFT less so.** Thinking does not change the prompt, and thus TTFT stays the same. But the
user waits for the *answer*. With 2,700 thinking tokens at the SLO's 40 ms each, the first answer token arrives
108.07 s after the request. This is `workload.request_duration_s()` with the thinking tokens as output. The wait is
still 49.9 s at the 18.5 ms step that the 3-GPU fleet runs at.

The UX answer is to stream the reasoning, or a summary of it. The capacity answer is that ITL, not tokens/s, sets how
many sessions a GPU can hold. With a 20 ms ITL SLO, a GPU holds 174 sessions (`workload.max_batch_for_itl()`). This
is fewer than the 195 that fit in HBM. Thus ITL binds first (2.87 GPUs against memory's 2.75 in `plan_steady()`).

Measure with the method of the serving-engine primer §11:

- open-loop load at realistic (here: heavy-tailed) output lengths,
- percentiles of ITL and TPOT,
- goodput against the SLO.

Also monitor `vllm:inter_token_latency_seconds`, `vllm:kv_cache_usage_perc` and `vllm:num_preemptions`. A long-tail
trace that outgrows the pool preempts the newest request.

### Multi-turn prompts and the reasoning parser

**Thinking is dropped from history: prefix-cache implications.** Qwen3's template (and gpt-oss's: "CoT is dropped
during all previous turns") renders earlier assistant turns without their reasoning (verify). The KV of turn N holds
prompt + thinking + answer. The prompt of turn N+1 contains only the answer.

Thus the prefix-cache hit
([serving-engine primer §5](../../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching): full 16-token blocks
only) ends where the prompt of turn N ended. Then the engine prefills the answer again. The table uses `workload.turn_prefills()` with a
1,000-token system prompt, and turns of 100 user + 800 thinking + 200 answer tokens:

| Turn | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| thinking dropped: prompt / prefilled | 1,104 / 1,104 | 1,408 / 304 | 1,712 / 304 | 2,016 / 304 |
| thinking kept (one tool loop): prompt / prefilled | 1,104 / 1,104 | 2,208 / 112 | 3,312 / 112 | 4,416 / 112 |

When the template drops thinking, the prompts stay short. But the engine computes and holds the KV of the 800
thinking tokens of each turn, and never uses it again. Thus multi-turn hit rates are lower than for the same
conversation without thinking. In one multi-step tool loop, the template keeps the thinking, and the whole previous
sequence is a hit.

**Streaming reasoning and vLLM's `--reasoning-parser`.** `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`
divides the output into `message.reasoning` and `message.content` (streamed as `delta.reasoning` and
`delta.content`). vLLM 0.30 names the field `reasoning`. SGLang, the DeepSeek API and Qwen's docs use
`reasoning_content`. Thus a portable client reads both.

Parser names use underscores in vLLM (`qwen3`, `deepseek_r1`, `openai_gptoss`) and hyphens in SGLang
(`deepseek-r1`). `--enable-reasoning` no longer exists. Structured output applies after the end of thinking, unless
you set `enable_in_reasoning`. The engine parses tool calls only from `content`. No Prometheus metric separates
reasoning from answer tokens, and both are in `vllm:generation_tokens`. For each request, read
`usage.completion_tokens_details.reasoning_tokens` (all verify).

### Output limits, cost and speculation

**Budget enforcement: `max_tokens` vs budget forcing.** In `workload.budget_outcome()`, each question needs
$L_{\text{req}} \sim \operatorname{lognormal}(1{,}500, \sigma = 1)$ thinking tokens, and the answer is 300 tokens.
Truncation returns no answer. Budget forcing answers with what it has. It is correct 30% of the time when the think
has not cracked the question ($e_0 = 0.7$):

| Limit (tokens) | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|
| `max_tokens`: accuracy / truncated | 0.561 / 0.439 | 0.823 / 0.177 | 0.952 / 0.048 | 0.991 / 0.009 |
| thinking budget (limit − 300): accuracy | 0.693 | 0.876 | 0.966 | 0.994 |
| mean output tokens (either) | 1,559 | 2,136 | 2,526 | 2,705 |

The tokens are the same, but the outcome is different. At 4K, one request in six gets no answer under `max_tokens`.
Set `max_model_len` for the tail: prompt + p99 thinking + answer. With 1,500-token prompts, 17,408 (a multiple of
1,024) keeps truncation at or below 1% (notebook 05, exercise 5.3). Keep `max_tokens` at `max_model_len` minus the
prompt (15,908 here), because it counts output only. Enforce cost with a budget.

**Cost per *correct* answer, and routing by effort.** You also pay for incorrect answers: cost per correct = cost per
request / accuracy (`workload.cost_per_correct()`). Use the §5 prices ($0.007005 per call without thinking,
$0.035355 with thinking). Use these illustrative accuracies:

- easy requests (70%): 0.95 off / 0.97 on,
- hard requests (30%): 0.30 off / 0.85 on.

The results:

- thinking everywhere: 0.934 at $0.037853 per correct answer,
- thinking only on hard requests: 0.920 at $0.016859,
- no thinking: 0.755 at $0.009278.

That is routing by effort at the gateway. The 06 layer's
[scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) gives
output (thinking included) a price of six times input on its model. It sets thinking levels per task (§3.4, §5.5).
Its [Mistral primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/mistral/01-scaling-primer.md)
routes task × level × mode to a model, a `reasoning_effort` and an output cap (§5.3, §5.5). Routing by effort needs
a classifier, or a low-cost first pass, that knows which requests are hard.

**Speculative decoding on long outputs.** Speculation pays on a decode-dominant workload
([serving-engine primer §7](../../04-inference-engine/serving-engine/PRIMER.md#7-speculative-decoding)). One target
pass emits $(1 - \alpha^{k+1})/(1 - \alpha)$ tokens: 3.36 at α = 0.8, k = 4 (`minengine.spec.expected_tokens()`).
The output distribution does not change.

The limits are the same as in that primer. At high concurrency, the extra positions of the verify pass cost real
compute. You must measure the acceptance on reasoning text, and not assume it (verify per model).

## 8. The RL training stack in brief

**Rollouts with an inference engine inside the trainer.**

| Framework | Rollouts | Notes |
|---|---|---|
| TRL `GRPOTrainer` | `use_vllm=True`: `vllm_mode="colocate"` or `"server"`. The colocate mode puts vLLM in the trainer process, on the same GPU (`vllm_gpu_memory_utilization` 0.3), and sleep mode frees it during the optimizer step. The server mode puts `vllm serve` on other GPUs and sends the weights over NCCL | A sequence-level importance weight corrects the train–inference log-prob mismatch. By default, it masks the weight outside $[C_{\min}, 3.0]$ (`vllm_importance_sampling_mode="sequence_mask"`, and the `*_truncate` modes truncate instead) (verify) |
| verl | `rollout.name`: `hf`, `vllm` or `sglang`. `rollout.n` samples per prompt. HybridFlow's `ActorRolloutRefWorker` puts actor, rollout and reference on the same GPUs | `algorithm.adv_estimator=grpo`, `kl_loss_type=low_var_kl` (k3), `loss_agg_mode` (verify) |
| OpenRLHF | Ray + vLLM, with PPO, GRPO, REINFORCE++ | this primer only names it (unverified) |

**Weight sync.** After each optimizer step, the rollout engine needs the new weights. In a colocated setup, it gets
them through shared memory. In a disaggregated setup, it gets them through an NCCL broadcast. In the report of verl,
over 99% of BF16 weight bytes do not change from step to step. Thus delta sync is 1.3–21× faster (verify).

With LoRA, the engine loads adapters instead of full weights. Then vLLM's `--max-loras` must cover every adapter
version in flight. TRL's async guidance is at least `max_staleness` + 2. The
[vllm-internals primer §9](../../04-inference-engine/vllm-internals/vllm-internals-primer.md#9-multi-lora-multimodal-and-hybrid-models-in-brief)
explains why a low `--max-loras` starves requests.

**Async and off-policy RL.** A synchronous step keeps the trainer idle while rollouts run. It also keeps most of the
rollout batch idle while the long tail completes (25% occupancy in the model of §1).

verl's one-step-off-policy trainer generates step k + 1 while it trains on step k. Its fully asynchronous trainer
runs rollouts and training on separate GPUs and reports 2.35–2.67× on Qwen2.5-7B with 128 GPUs. TRL's experimental
`AsyncGRPOTrainer` streams completions from a vLLM server. It drops samples more than `max_staleness` (default 4)
weight versions old (verify).

The price is staleness. Samples come from an older policy, so the ratio $\rho = \pi/\pi_{\text{old}}$ is no
longer 1. Then the clip (§4) and the corrections by importance sampling do real work.

**Why serving skills transfer.** The rollout side is a serving problem with a different SLO: throughput per step
instead of latency per user. It has the same levers:

- KV capacity and preemption (a GRPO batch is G long samples per prompt),
- a shared prefix (the G samples share their prompt),
- batching and the long tail,
- CUDA Graphs and the load of the weights,
- the metrics of the serving-engine primer §11.

**Agentic RL.** Multi-turn rollouts with tools make each trajectory a loop: generate, tool call, observation,
generate. The consequences:

- You must reset and isolate the environment for each rollout. Code execution belongs in a sandbox (the
  [sandboxed-execution primer](../../07-application-agent-framework/sandboxed-execution/PRIMER.md)).
- Tool latency adds its own tail to the step.
- Credit assignment spans turns.

Reward design and evals are the same discipline as in the platform lab of 07
([`08_evals_trajectory_judge_gates`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py):
golden sets, run-to-run noise, Wilson intervals, release gates). The training reward is a proxy. The eval is how you
see that the policy exploits it.

## 9. Where to run it

| To learn | T0 (laptop / Colab CPU) | T1: free Colab / Kaggle T4 | T1: rented 24 GB GPU | T3: GCP |
|---|---|---|---|---|
| §2–4 policy gradients, DPO, GRPO | `rl-core` notebooks 01–03 | `thinking-lab` 01: a small transformer, trained from scratch with SFT then GRPO in torch. It takes about a minute on a laptop CPU, so it is T0 too | same, faster | — |
| §5, §7 a real thinking model | `rl-core` notebook 05, and the lab's fake server (labelled simulated) | `Qwen/Qwen3-0.6B` in vLLM 0.30 with `--reasoning-parser qwen3 --dtype half`, and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with `deepseek_r1` | `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507` in bf16 | the 04 lab's [Cloud Run and GKE deploys](../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/) with a thinking model and `--reasoning-parser` |
| §6 test-time compute | `rl-core` notebook 04 | best-of-n and majority vote on Qwen3-0.6B (lab 03) | a 4B model | — |
| §8 one RL step with an engine | the lab's rollout records on the small transformer | vLLM 0.30 generates the rollouts for `Qwen/Qwen2.5-0.5B-Instruct`, and transformers takes the step (lab 05). TRL 1.14's `GRPOTrainer` with colocated vLLM in fp16 is the packaged alternative (fit to 15 GB: verify) | same, with room | — |

A T4 has 15 GiB usable, no bf16 and no FP8. The minimum for vLLM 0.30 is compute capability 7.5, and on a T4 vLLM uses
the Triton attention backend (verify). The 04 lab's `servelab.sizing.size()` predicts 6,969 KV blocks for Qwen3-0.6B
at `max_model_len=8192` on a T4. That is 13.6 concurrent 8K-token requests. It predicts no room at all for Qwen3-8B in
fp16.

Kaggle's free 2×T4 does not change the picture for thinking models. A rented 24 GB L4 or RTX 4090 runs a 4B thinking
model with room for long traces. You can rent it as a container on RunPod or Vast.ai, or as a VM on Lambda or a GCP
`g2-standard-4` Spot L4. GCP is one target, never a prerequisite. The lab adds no Terraform of its own and points at
the Terraform of the 04 serving lab. For prices, quotas and obtainability, see [`COMPUTE.md`](../../COMPUTE.md).

---

## In a design review

**The two-minute walkthrough.** "Post-training is three stages on the same network. SFT imitates demonstrations.
RL increases the probability of the model's own samples in proportion to how they scored against a baseline. RL can
only reweight what the SFT model already samples: with a KL penalty, the optimum is the reference times
$\exp(\text{reward}/\beta)$. Thus we invest in the SFT model and in the reward.

"Where we can check the answers, we use verifiable rewards and GRPO. That means G samples per prompt,
group-normalised advantages, no value model, a token-level loss, and a mask on truncated completions. Where we cannot
check them, we use preferences through Bradley–Terry. We use DPO when we want the KL-regularised optimum without a
reward model. We also use a length-controlled eval, because the biases of the annotators become the reward.

"Most of an RL step is generation. Thus the rollout side is a problem of inference serving, and async rollouts trade
throughput for staleness.

"With this training, models learned to think, and that changes serving. The traffic is output-heavy and
heavy-tailed. Concurrency scales with output length, and KV per session grows while the model thinks. Thus memory
and the ITL SLO set the GPU count. In the capacity primer's example, ten times the output needs eighteen times the
GPUs for memory. This is true under both assumptions: a request takes the SLO's TPOT per token, or it takes the step
at which the fleet actually runs.

"We set `max_model_len` for the p99. We enforce cost with a thinking budget, not with `max_tokens`. We expect lower
multi-turn prefix-cache hits, because templates drop old thinking. We route effort per request class on cost per
correct answer."

**Drill questions**

1. *Training reward climbed to 99% and the held-out eval fell. What happened and what do you change?* Reward
   hacking: the policy found where the reward and the objective disagree. In the core, a verifier's early exit took
   true accuracy from 29.7% to 4.9%, while the reward reached 99.4%. Examine high-reward failures. Repair the
   verifier or the reward model. Use the eval as a gate. A larger β only limits the damage.
2. *Why can DPO skip the reward model, and what does it give up?* The KL-regularised optimum gives
   $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z$. In the Bradley–Terry difference, $Z$ cancels. The result
   is a classification loss on the policy's own log-ratios. DPO gives up exploration (offline pairs only). It
   optimises margins, not likelihoods (the chosen log-ratio decreases). It leaves no reward model to rerank or monitor
   with.
3. *Our GRPO run's responses keep getting longer and more hit `max_completion_length`. First checks?* First, examine
   `loss_type`. A per-sequence average gives less weight to long completions. In the core, its update has cosine
   0.67 with the true gradient and increases truncation. Then examine `mask_truncated_completions` and a soft
   overlong penalty. Also examine if the reward charges anything for length.
4. *We turned on thinking for our assistant at the same QPS. What happens to the fleet?* Concurrency scales with
   output length, and KV per session scales with average context. Thus memory binds. In the capacity primer's
   example, 10× output needed 18× the GPUs for memory. The numbers are 5.12 against 0.28 at the SLO's TPOT, 2.75
   against 0.15 at the step the fleet runs at. That is 3 H100s instead of 1. Also, a tight ITL SLO caps the batch
   before HBM does. Do not calculate the fleet at the SLO's TPOT when you compare SLOs. That method makes the looser
   SLO seem to cost more. TTFT does not change, but the time to the first answer token changes.
5. *Users get empty answers from the thinking model. Why, and the fix?* `max_tokens` counts reasoning. Some
   requests still think at the cap. They return `finish_reason="length"` with empty content (17.7% at a 4K cap in
   §7's model). Increase `max_tokens`/`max_model_len` for the tail. Enforce cost with `thinking_token_budget` or a
   two-call budget.
6. *Best-of-16 with a reward model or one sample from a larger model, same budget?* With a real verifier, more
   samples from a lower-cost model often win, because they cover dead ends. With a reward model, the gain saturates
   below the gain of the verifier. Also, a biased reward model selects for its bias. With only a vote, the per-sample
   accuracy of the larger model wins, unless single samples are already usually correct. Measure pass@1 and pass^k at
   equal cost on your evals.

---

## Glossary

| Term | Meaning |
|---|---|
| Policy | the model as a distribution over the next token given the context, $\pi(a \mid s)$ |
| Trajectory / completion | one sampled output. $\log \pi(y)$ is the sum of the log-probabilities of its tokens |
| SFT | supervised fine-tuning: maximum likelihood on demonstrations |
| Reward model (RM) | a network trained on preference pairs to score completions (Bradley–Terry) |
| Verifier | a program that checks a completion (answer equivalence, unit tests, a format regex) |
| RLVR | reinforcement learning with verifiable rewards |
| REINFORCE | the score-function gradient $\mathbb{E}[(R - b) \nabla \log \pi(y)]$ |
| Baseline / advantage | $b$, subtracted from the reward with no bias to the gradient / ${R - b}$ |
| Value model | a learned ${V(s)}$ that gives per-token baselines (PPO's critic) |
| GAE | generalised advantage estimation: TD errors accumulated with factor $\gamma\lambda$ |
| KL penalty | $-\beta \cdot \mathrm{KL}(\pi \parallel \pi_{\text{ref}})$ in the objective, with the optimum $\pi_{\text{ref}} \exp(R/\beta)/Z$ |
| k1 / k3 | per-sample KL estimators $\log(\pi/\pi_{\text{ref}})$ and $\pi_{\text{ref}}/\pi - \log(\pi_{\text{ref}}/\pi) - 1$ |
| PPO | policy optimisation with a clipped probability ratio $\rho = \pi/\pi_{\text{old}}$ |
| RLHF | SFT, a reward model from human preferences, then PPO with a KL penalty |
| Bradley–Terry | $P(A \succ B) = \sigma(r_A - r_B)$ |
| DPO | direct preference optimisation: the closed form of the KL-regularised optimum as a pairwise loss |
| Implicit reward | $\beta \log(\pi/\pi_{\text{ref}})$, DPO's reward (TRL's rewards/chosen, rewards/rejected) |
| IPO / KTO / ORPO | squared loss to a constant margin / unpaired good–bad labels / reference-free SFT + odds-ratio loss |
| GRPO | group relative policy optimisation: $G$ samples per prompt, group-normalised advantages, no critic |
| Clip-higher | a larger upper clip $\varepsilon_{\text{high}}$ than lower $\varepsilon_{\text{low}}$, against entropy collapse (DAPO) |
| Dynamic sampling | the trainer drops groups whose rewards are all equal and refills the batch (DAPO) |
| Overlong shaping | a mask on truncated completions and a soft penalty near the length cap (DAPO) |
| Dr. GRPO | GRPO with a constant loss normaliser and no division by the std |
| Rollout | the generation of completions for training, with an inference engine |
| Reward hacking | the policy maximises the reward in a way that the designer did not intend |
| Over-optimisation | the proxy reward increases while the true objective decreases, as the policy moves away |
| Thinking model | a model that emits a reasoning block before its answer, trained (or distilled) to do so |
| Rejection-sampling SFT | fine-tuning on the model's own samples that passed a check |
| Distillation (of traces) | SFT of a smaller model on a stronger model's outputs |
| Thinking budget | a cap on reasoning tokens. After the cap, the model must answer |
| pass@k / pass^k | at least one of $k$ samples correct / all $k$ correct |
| Self-consistency | majority vote over sampled answers |
| PRM | process reward model: it scores intermediate steps |
| KV-token-steps | $\sum$ over decode steps of the tokens a request holds in KV: $P \cdot L + L(L + 1)/2$ |
| Cost per correct answer | cost per request divided by accuracy |

## Sources

Papers:

- Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine
  Learning 8, 1992: REINFORCE.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (arXiv:1506.02438)
  and *Proximal Policy Optimization Algorithms* (arXiv:1707.06347). Schulman, *Approximating KL Divergence* (blog,
  2020): k1/k2/k3.
- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (arXiv:1706.03741), and Ouyang et al.,
  *Training language models to follow instructions with human feedback* (arXiv:2203.02155): RLHF.
- Bradley and Terry, *Rank Analysis of Incomplete Block Designs*, Biometrika, 1952.
- Rafailov et al., *Direct Preference Optimization* (arXiv:2305.18290). Azar et al., *A General Theoretical Paradigm
  to Understand Learning from Human Preferences* (arXiv:2310.12036): IPO. Ethayarajh et al., *KTO* (arXiv:2402.01306).
  Hong et al., *ORPO* (arXiv:2403.07691).
- Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*
  (arXiv:2402.14740): RLOO.
- Gao, Schulman and Hilton, *Scaling Laws for Reward Model Overoptimization* (arXiv:2210.10760).
- Shao et al., *DeepSeekMath* (arXiv:2402.03300): GRPO. DeepSeek-AI, *DeepSeek-R1* (arXiv:2501.12948).
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (arXiv:2503.14476). Liu et al.,
  *Understanding R1-Zero-Like Training: A Critical Perspective* (arXiv:2503.20783): Dr. GRPO.
- Qwen Team, *Qwen3 Technical Report* (arXiv:2505.09388).
- Chen et al., *Evaluating Large Language Models Trained on Code* (arXiv:2107.03374): unbiased pass@k. Yao et al.,
  *τ-bench* (arXiv:2406.12045): pass^k. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*
  (arXiv:2203.11171). Lightman et al., *Let's Verify Step by Step* (arXiv:2305.20050). Snell et al., *Scaling LLM
  Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (arXiv:2408.03314).
- Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (arXiv:2409.19256): verl.

Code and documentation (read 2026-09-26):

- TRL `github.com/huggingface/trl` v1.14.0: `trl/trainer/grpo_config.py`, `grpo_trainer.py` (`_compute_advantages`,
  `_compute_loss`), `dpo_trainer.py`, `trl/rewards/`, `docs/source/grpo_trainer.md`, `async_grpo_trainer.md`,
  `reward_trainer.md`.
- vLLM `github.com/vllm-project/vllm` v0.30.0: `docs/features/reasoning_outputs.md`, `vllm/reasoning/`,
  `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/sampling_params.py` (`thinking_token_budget`),
  `vllm/config/reasoning.py`, `vllm/v1/metrics/loggers.py`.
- verl `github.com/volcengine/verl` docs: `algo/grpo.md`, `examples/config.rst`, `hybrid_flow.rst`,
  `advance/one_step_off.md`, `advance/fully_async.md`, `advance/delta_weight_sync.md`.
- DeepSeek-R1 `github.com/deepseek-ai/DeepSeek-R1` (README, paper). DAPO `github.com/BytedTsinghua-SIA/DAPO`. Qwen3
  `github.com/QwenLM/Qwen3` (README, docs: quickstart, thinking budget, vLLM deployment). gpt-oss
  `github.com/openai/gpt-oss`. SGLang docs (`separate_reasoning`). lm-evaluation-harness (`gsm8k-cot-self-consistency`).
  HumanEval `estimate_pass_at_k`.
- In this repo:
  - the [transformer primer](../transformers/docs/transformer-primer.md),
  - the [capacity primer](../gpu-capacity-planning/PRIMER.md) and [`capacity.py`](../gpu-capacity-planning/capacity.py),
  - the [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md),
  - the [vllm-internals primer](../../04-inference-engine/vllm-internals/vllm-internals-primer.md),
  - [`vllm-serving-lab`](../../04-inference-engine/serving-engine/vllm-serving-lab/) (`servelab.sizing`, bench, fake
    server),
  - the [agentic scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md),
  - the 07 [platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) (evals),
  - the 07 [sandboxed-execution primer](../../07-application-agent-framework/sandboxed-execution/PRIMER.md) (sandboxes
    for tool calls and code).

## Verify list

These are the product facts in this primer, in `rlcore/workload.py` and in the notebooks, as of 2026-09-26. Examine
them again against the versions that you pin.

- **vLLM 0.30.0 reasoning support:**
  - `--reasoning-parser` (and `--reasoning-parser-plugin`), with `--enable-reasoning` removed,
  - the parser names `qwen3`, `deepseek_r1`, `openai_gptoss`, among ~35,
  - the response field `reasoning` (renamed from `reasoning_content`) in `message` and `delta`,
  - `include_reasoning`,
  - `usage.completion_tokens_details.reasoning_tokens`,
  - `chat_template_kwargs` and `--default-chat-template-kwargs`,
  - the `reasoning_effort` values none/minimal/low/medium/high/xhigh/max, and their mapping to `enable_thinking` for
    Qwen3,
  - `thinking_token_budget` (−1 = unlimited) with `--reasoning-config`,
  - structured output after reasoning, unless `enable_in_reasoning`,
  - tool calls parsed from `content` only,
  - no reasoning-specific Prometheus metric,
  - the T4 facts (compute capability 7.5 minimum, Triton attention backend, `--dtype half`).
- **TRL v1.14.0:**
  - the `GRPOConfig` defaults in §4's table,
  - the `loss_type` values (`grpo`, `dapo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo`),
  - the `scale_rewards` values,
  - the +1e-4 and Bessel's correction in the advantage,
  - the k3 KL,
  - the vLLM colocate/server modes, and `vllm_gpu_memory_utilization` 0.3,
  - the defaults of the importance sampling correction,
  - `DPOConfig.beta` 0.1 and `loss_type="sigmoid"`,
  - ORPO under `trl.experimental`,
  - the `AsyncGRPOTrainer` defaults (`max_staleness` 4, `weight_sync_steps` 1),
  - the claim that R1 used β = 0.001.
- **DeepSeek-R1:**
  - R1-Zero AIME 2024 from 15.6% to 71.0% (86.7% majority),
  - the four-stage recipe, and the ~600k + ~200k sample counts,
  - 671B total / 37B activated,
  - the distill results (Qwen-1.5B 28.9, and Qwen-32B 72.6 against RL-on-32B-base 47.0),
  - rule-based rewards only,
  - PRM and MCTS listed as unsuccessful,
  - the usage recommendations (temperature 0.6, no system prompt).
- **DAPO:**
  - the ablation 30, 36, 38, 41, 42 and 50 (AIME 2024 avg@32),
  - ε 0.2/0.28,
  - 512 prompts × 16 responses,
  - maximum generation 20,480 tokens,
  - the soft overlong formula,
  - no KL term.
- **verl:**
  - rollouts ~70% of step time in DAPO-32B training,
  - delta weight sync (>99% of bytes unchanged, 1.3–21×),
  - fully async 2.35–2.67× on Qwen2.5-7B with 128 GPUs,
  - the config names in §8.
- **Qwen3:**
  - the model ids (0.6B, 1.7B, 4B, 8B, and the 2507 Instruct/Thinking split),
  - `enable_thinking` and the soft switches,
  - the template that drops the reasoning of earlier turns,
  - the recommended sampling (thinking 0.6/0.95/20, non-thinking 0.7/0.8/20),
  - the thinking-budget phrase,
  - the post-training (GRPO on 3,995 query–verifier pairs, and strong-to-weak distillation for small models),
  - the gpt-oss effort levels and the Harmony format.
- **Sizing inputs:**
  - the GPU figures in `workload.GPUS`:
    - the H100 as in the capacity primer,
    - the L4: 24 GB, 0.30 TB/s, 121/242 TFLOP/s,
    - the T4: 16 GB nominal (15 GiB usable), 0.32 TB/s, 65 TFLOP/s fp16, no FP8,
  - the config of Qwen3-0.6B (28 layers, 8 KV heads, head_dim 128, 0.596 B parameters),
  - the T4 prediction of the 04 lab (6,969 blocks, 13.6 × 8K),
  - the example prices of the 06 lab ($1.50 / $9.00 / $0.15 per 1M tokens, dated 2026-09-05 there).
- **Where to run:** Colab/Kaggle T4 availability, rented 24 GB GPU prices and GCP deploy details.
  [`COMPUTE.md`](../../COMPUTE.md) and the 04 lab's deploy READMEs keep these facts up to date.
