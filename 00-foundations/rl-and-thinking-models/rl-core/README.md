# rl-core — policy gradients, DPO, GRPO and thinking workloads, small enough to compute exactly

After this core, you can do these things in `rlcore`:

- Derive what an RL post-training step does to a model, and write it in code. This covers REINFORCE, the KL
  penalty and its closed form, Bradley–Terry and DPO, and GRPO with the DAPO corrections.
- Predict how RL exploits a bug in a verifier, and why thinking becomes longer.
- Select between longer thinking and more samples.
- Calculate the size of a fleet for a thinking model.

`rlcore` is a standard-library-plus-numpy package of ~1,000 lines. In it, a table of softmaxes replaces the network.
Thus you can also calculate each expectation exactly.

## Start here

1. Read "The one-minute version" in [`../PRIMER.md`](../PRIMER.md). Then read §1 From pretraining to
   post-training and §2 Policy gradients over token sequences.
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 65 tests run in approximately
   50 s. Two of them are "DPO lands on the closed-form optimum π_ref·exp(r/β)" and "the capacity primer's bank
   example, number for number".
3. Open [`notebooks/01_policy_gradients_on_a_toy_task.ipynb`](notebooks/01_policy_gradients_on_a_toy_task.ipynb).
   See how RL finds the bug in a verifier.

## What you get

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network.*

Each notebook starts with "The one-minute version". Then it shows worked examples that use the code, and it sets
exercises. Each exercise has a check cell that prints ✅. Each notebook ends with "In a design review". The finished
versions are in [`solutions/`](solutions/). The notebooks and the primer take approximately 12 hours in total.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_policy_gradients_on_a_toy_task`](notebooks/01_policy_gradients_on_a_toy_task.ipynb) | Derive the softmax log-probability gradient and REINFORCE. Show why a baseline is important (without one, a constant reward is only noise). Calculate the KL-regularised optimum in closed form. See RL exploit a verifier with a bug (true accuracy goes from 29.7% to 4.9% while the reward gets to 99.4%). See RL also make thinking longer when thinking has no cost. Predict the optimal thinking length. | §1, §2 | ~2 h | T0 |
| [`02_preferences_reward_models_and_dpo`](notebooks/02_preferences_reward_models_and_dpo.ipynb) | Fit a Bradley–Terry reward model. Run RLHF (reward model + RL with a KL penalty) and DPO on the same pairs, and get both to the closed form. Read the DPO metrics of TRL (and why rewards/chosen decreases). Calculate GAE. See RL over-optimise a length-biased reward model, and select the KL budget. | §3 | ~2 h | T0 |
| [`03_grpo_with_verifiable_rewards`](notebooks/03_grpo_with_verifiable_rewards.ipynb) | Calculate three things by hand: the group advantages of TRL, k3 and the clipped surrogate. See RL sharpen (effective correct answers go from 13.9 to 2.4), and see an entropy bonus get diversity back. Measure the length bias of the per-sequence average against the exact gradient. Compare the cost of the extra rollouts of dynamic sampling with what they give. Estimate how much of an RL step is generation. Write the `GRPOConfig` of DAPO and of R1. | §1, §4, §8 | ~2.5 h | T0 |
| [`04_test_time_compute`](notebooks/04_test_time_compute.ipynb) | Estimate pass@k without bias, and use pass^k for reliability. Predict when majority vote helps and when it makes the result worse. Compare a verifier with a noisy reward model. Divide a token budget between samples and thinking. Tell when a smaller model with more samples is better. | §6 | ~1.5 h | T0 |
| [`05_thinking_models_and_the_serving_workload`](notebooks/05_thinking_models_and_the_serving_workload.ipynb) | Show how rejection-sampling SFT and distillation copy thinking behaviour. Calculate the size of a heavy-tailed thinking workload with the formulas of the capacity primer plus HBM and ITL caps. Select `max_model_len`, and select a thinking budget instead of `max_tokens`. Predict prefix-cache hits when templates remove old thinking. Route effort by cost per correct answer. | §5, §7 | ~2 h | T0 |

§9 (where to run it) has no notebook. The notebooks of the lab are the T1 half of §9.

## Run it

```bash
cd rl-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 65 tests, ~50 s
python3 -m jupyterlab notebooks                # do the exercises
```

The library itself needs only numpy:

```python
import numpy as np
from rlcore import Policy, SeqTask, pg

task = SeqTask("brackets", length=8)                       # a verifier: is the string balanced?
policy = Policy.for_task(task)                              # a table of softmaxes: uniform to start
print(task.random_success_rate())                           # 0.0546875 = 14/256
pg.train_reinforce(policy, task, np.random.default_rng(0), steps=200, batch=16, lr=0.5)
print(pg.expected(policy, task, task.verify))               # P(balanced), exactly, over all 256 strings
```

## The whole library

Read the modules in this order. Each module starts with a docstring that gives the one idea that it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`rlcore/tasks.py`](rlcore/tasks.py) | ~165 | Two verifiable toy environments. `SeqTask` has balanced brackets, with a true verifier and a verifier with a bug, or any reward function. `ThinkTask` has think tokens until the answer, P(correct \| L) = 1 − e0·(1 − q)^L, truncation against budget forcing, exact length distributions and the optimal length. The code selects the states so that the reward depends only on the final state. |
| [`rlcore/policy.py`](rlcore/policy.py) | ~85 | A policy as a table of softmaxes: seeded token-by-token sampling, per-token log-probabilities, the closed-form gradient `onehot − π`, explicit reference/old-policy copies, exact sequence probabilities. |
| [`rlcore/pg.py`](rlcore/pg.py) | ~105 | REINFORCE. Baselines (none, mean, leave-one-out) and gradient variance. The KL penalty as reward shaping. An entropy bonus. SFT. Exact expectations and KL by enumeration. The KL-regularised optimum π_ref·exp(R/β)/Z. |
| [`rlcore/pref.py`](rlcore/pref.py) | ~140 | The Bradley–Terry fit. The DPO loss, its gradient and the metric names of TRL. IPO, the implicit reward and GAE. A response catalogue and a length-biased annotator. |
| [`rlcore/grpo.py`](rlcore/grpo.py) | ~180 | The group advantages of TRL (Bessel, +1e-4), k1 and k3, the clipped surrogate and its gradient mask, and the three loss normalisers. DAPO's soft overlong penalty, and `GRPOConfig` with the field names of TRL. A GRPO step with μ iterations, KL, masking and dynamic sampling (its gradient is in `surrogate_grad`). The averaged update that the code uses to measure the length bias. |
| [`rlcore/ttc.py`](rlcore/ttc.py) | ~110 | The unbiased pass@k (the product form of HumanEval), pass^k, the biased plug-in and its exact expectation. Exact majority vote. Best-of-n with a noisy scorer. A question population with two kinds of difficulty. Budget allocation. |
| [`rlcore/workload.py`](rlcore/workload.py) | ~225 | The formulas of the capacity primer, stated again (weights, KV per token and session, TTFT, request lifetime, sessions per GPU, decode and prefill throughput). A roofline decode step. Batches with HBM and ITL caps. `plan()` (every token at the TPOT of the SLO, as the capacity primer does) and `plan_steady()` (at the step that the fleet runs at). KV-token-steps, lognormal lengths, `max_tokens` against budget forcing, and prefix reuse across turns. API cost and cost per correct answer. The time split of one synchronous RL step. |

## What the tests prove

`tests/` has one focused test per concept (57, plus 8 notebook-tooling checks). The tests run offline in ~50 s in
all. The tests in this list carry the correctness claims:

- **The gradients are right.** `grad_logprob` agrees with finite differences. The mean of 400 REINFORCE estimates
  has a correlation above 0.95 with the exact gradient of P(correct), which the test calculates by enumeration. The
  step of DPO agrees with finite differences of its loss. The hand-derived gradient of GRPO agrees with finite
  differences of the objective, with the clipped surrogate plus β·k3, every `loss_type`, and the clip active. The
  tests are in `test_tasks_policy.py`, `test_pg.py`, `test_pref.py` and `test_grpo.py`.
- **The closed forms hold.** `kl_optimal` gets a higher E[R] − β·KL than other policies, and its KL equals
  E[R]/β − log Z. DPO on 4,096 Bradley–Terry pairs gets to within 0.02 nats of π*. Its implicit reward finds the
  true gap of 3 to within 0.15. The tests are in `test_pg.py` and `test_pref.py`.
- **The failure modes are real, not staged.** RL on the verifier with the bug decreases true accuracy to below 8%
  while the reward goes above 99%. With β = 0.3, true accuracy stays above 30%. RL makes thinking longer when
  tokens have no cost, and less long when tokens have a cost. The gold quality of a length-biased reward model goes
  to a peak and then decreases, while its score increases. The update of the per-sequence average points away from
  the true gradient (cosine < 0.8) and increases truncation. But the update of the token-level average (> 0.95)
  decreases truncation (`test_pg.py`, `test_pref.py`, `test_grpo.py`).
- **Formulas pinned to hand-computed and reference values:** TRL's advantages (±0.865875 and 1.4997/−0.4999), k3 at
  ±0.1 and 0.5, the clip's gradient mask and the three normalisers. The tests also pin DAPO's overlong penalty
  (0, −0.5, −1.0, −1), the DPO loss 0.598139 and GAE. They also pin the unbiased pass@k (0.916667, 0.728022,
  0.914746), pass^k and exact majority votes. With a dominant misconception, each extra vote decreases accuracy.
  With a narrow misconception, this occurs only after ~130 votes. The tests are in `test_grpo.py`, `test_pref.py`
  and `test_ttc.py`.
- **Existing repo numbers reproduced.** The tests reproduce the bank example of the capacity primer: 12.07 s, 100.6
  live, 88.8 and 355.1 sessions per GPU, and 8,929 and 20,615 tokens/s. Every memory quantity is in GB = 10⁹ bytes.
  The tests compare these numbers as constants, and also with `capacity.py`, function by function. They also
  reproduce the `cost_per_call` of the 06 scaling lab ($0.007005, $0.035355). The tests are in `test_workload.py`.
- **The primer says what the code computes.** `test_primer_numbers.py` calculates again each computed number in
  `../PRIMER.md`. Each number must appear verbatim.

## Caveats: what the toys are and are not

A table of per-state softmaxes is a language model without generalisation. Each state learns only from its own
visits. This is why the thinking length increases slowly in the ThinkTask runs. It is also why the reach of DPO
beyond its pairs is exact here, but a property of the network in practice.

The toy shows these effects: reward hacking, length growth, over-optimisation, the length bias and diversity
collapse. The direction of each effect does not change across seeds (the tests examine this). But the magnitudes
are those of the toy. The entropy effect of clip-higher is a measurement by DAPO. This core does not reproduce it.

`ttc.question_set()` is a model of benchmark difficulty, not a benchmark. Each latency, GPU count and dollar figure
from `rlcore.workload` is a **model**, not a measurement. The model uses the formulas of the capacity primer with its
round datasheet numbers (verify), plus a roofline step. The lab measures a real vLLM server.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three commands and the tests. On Colab, the first cell of each notebook clones the repo
and installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../thinking-lab/`](../thinking-lab/). In the lab, you do these things:

- Train a small transformer with SFT, then with GRPO, in torch.
- Serve a real thinking model (Qwen3-0.6B) in vLLM with `--reasoning-parser` on a free T4.
- Run best-of-n and majority vote against that thinking model.
- Measure what long outputs do to ITL and KV usage.
- Run one GRPO step with vLLM as the rollout engine.

The engine below the lab is [`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md).
The lab builds on the sizing in [`gpu-capacity-planning`](../../gpu-capacity-planning/PRIMER.md). For where each tier
runs and what it costs, see [`COMPUTE.md`](../../../COMPUTE.md). The core has an MIT licence.
