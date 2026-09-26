# rl-core — policy gradients, DPO, GRPO and thinking workloads, small enough to compute exactly

After this you can derive and implement what an RL post-training step does to a model — REINFORCE, the KL
penalty and its closed form, Bradley–Terry and DPO, GRPO with the DAPO fixes — predict how a verifier gets hacked
and why thinking lengthens, choose between thinking longer and sampling more, and size a fleet for a thinking
model, all in `rlcore`, a standard-library-plus-numpy package of ~1,000 lines where a table of softmaxes stands in
for the network so every expectation can also be computed exactly.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md): "The one-minute version", then §1 From pretraining to post-training and
   §2 Policy gradients over token sequences.
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 54 tests in about 30 s, including "DPO
   lands on the closed-form optimum π_ref·exp(r/β)" and "the capacity primer's bank example, number for number".
3. Open [`notebooks/01_policy_gradients_on_a_toy_task.ipynb`](notebooks/01_policy_gradients_on_a_toy_task.ipynb)
   and watch RL find a verifier's bug.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network.* Each notebook opens with
"The one-minute version", works examples against the code, sets exercises with a check cell that prints ✅, and
ends with "In a design review". Finished versions are in [`solutions/`](solutions/). About 12 hours in all with
the primer.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_policy_gradients_on_a_toy_task`](notebooks/01_policy_gradients_on_a_toy_task.ipynb) | derive the softmax log-probability gradient and REINFORCE; show why a baseline matters (a constant reward is pure noise without one); compute the KL-regularised optimum in closed form; watch RL exploit a buggy verifier (true accuracy 29.7% → 4.9% while the reward hits 99.4%) and lengthen uncharged thinking; predict the optimal thinking length | §1, §2 | ~2 h | T0 |
| [`02_preferences_reward_models_and_dpo`](notebooks/02_preferences_reward_models_and_dpo.ipynb) | fit a Bradley–Terry reward model; run RLHF (reward model + RL with a KL penalty) and DPO on the same pairs and land both on the closed form; read TRL's DPO metrics (why rewards/chosen falls); compute GAE; watch a length-biased reward model get over-optimised and pick the KL budget | §3 | ~2 h | T0 |
| [`03_grpo_with_verifiable_rewards`](notebooks/03_grpo_with_verifiable_rewards.ipynb) | compute TRL's group advantages, k3 and the clipped surrogate by hand; see RL sharpen (13.9 → 2.4 effective correct answers) and an entropy bonus buy diversity back; measure the length bias of per-sequence averaging against the exact gradient; weigh dynamic sampling's extra rollouts; estimate how much of an RL step is generation; write DAPO's and R1's `GRPOConfig` | §1, §4, §8 | ~2.5 h | T0 |
| [`04_test_time_compute`](notebooks/04_test_time_compute.ipynb) | estimate pass@k without bias and use pass^k for reliability; predict when majority vote helps or hurts; compare a verifier with a noisy reward model; split a token budget between samples and thinking; say when a smaller model with more samples wins | §6 | ~1.5 h | T0 |
| [`05_thinking_models_and_the_serving_workload`](notebooks/05_thinking_models_and_the_serving_workload.ipynb) | show rejection-sampling SFT and distillation copying thinking behaviour; size a heavy-tailed thinking workload with the capacity primer's formulas plus HBM and ITL caps; choose `max_model_len` and a thinking budget over `max_tokens`; predict prefix-cache hits when templates drop old thinking; route effort by cost per correct answer | §5, §7 | ~2 h | T0 |

§9 (where to run it) has no notebook; the lab's notebooks are its T1 half.

## Run it

```bash
cd rl-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 54 tests, ~30 s
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

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`rlcore/tasks.py`](rlcore/tasks.py) | ~165 | two verifiable toy environments: `SeqTask` (balanced brackets, with a true and a buggy verifier, or any reward function) and `ThinkTask` (think tokens until answering; P(correct \| L) = 1 − e0·(1 − q)^L; truncation vs budget forcing; exact length distributions, optimal length); states chosen so the reward depends only on the final state |
| [`rlcore/policy.py`](rlcore/policy.py) | ~85 | a policy as a table of softmaxes: seeded token-by-token sampling, per-token log-probabilities, the closed-form gradient `onehot − π`, explicit reference/old-policy copies, exact sequence probabilities |
| [`rlcore/pg.py`](rlcore/pg.py) | ~105 | REINFORCE; baselines (none, mean, leave-one-out) and gradient variance; the KL penalty as reward shaping; an entropy bonus; SFT; exact expectations and KL by enumeration; the KL-regularised optimum π_ref·exp(R/β)/Z |
| [`rlcore/pref.py`](rlcore/pref.py) | ~140 | Bradley–Terry fitting; the DPO loss, its gradient and TRL's metric names; IPO; the implicit reward; GAE; a response catalogue and a length-biased annotator |
| [`rlcore/grpo.py`](rlcore/grpo.py) | ~180 | TRL's group advantages (Bessel, +1e-4), k1 and k3, the clipped surrogate and its gradient mask, the three loss normalisers, DAPO's soft overlong penalty, `GRPOConfig` with TRL's field names, a GRPO step with μ iterations, KL, masking and dynamic sampling, and the averaged update used to measure the length bias |
| [`rlcore/ttc.py`](rlcore/ttc.py) | ~110 | the unbiased pass@k (HumanEval's product form), pass^k, the biased plug-in and its exact expectation; exact majority vote; best-of-n with a noisy scorer; a question population with two kinds of difficulty; budget allocation |
| [`rlcore/workload.py`](rlcore/workload.py) | ~200 | the capacity primer's formulas restated (weights, KV per token and session, TTFT, request lifetime, sessions per GPU, decode and prefill throughput); a roofline decode step; batches capped by HBM and ITL; `plan()`; KV-token-steps; lognormal lengths; `max_tokens` vs budget forcing; prefix reuse across turns; API cost, cost per correct answer; the time split of one synchronous RL step |

## What the tests prove

`tests/` has one focused test per concept (54, offline, ~30 s). The ones that carry the correctness claims:

- **The gradients are right.** `grad_logprob` matches finite differences; the mean of 400 REINFORCE estimates
  correlates above 0.95 with the exact gradient of P(correct) computed by enumeration; DPO's step matches finite
  differences of its loss (`test_tasks_policy.py`, `test_pg.py`, `test_pref.py`).
- **The closed forms hold.** `kl_optimal` beats other policies on E[R] − β·KL and its KL equals E[R]/β − log Z;
  DPO on 4,096 Bradley–Terry pairs lands within 0.02 nats of π* and its implicit reward recovers the true gap of 3
  within 0.15 (`test_pg.py`, `test_pref.py`).
- **The failure modes are real, not staged.** RL on the buggy verifier drives true accuracy below 8% while the
  reward exceeds 99%, and β = 0.3 holds it above 30%; RL lengthens thinking with no token cost and less with one;
  a length-biased reward model's gold quality peaks and then falls while its score rises; per-sequence averaging's
  update points away from the true gradient (cosine < 0.8) and raises truncation while token-level averaging's
  (> 0.95) lowers it (`test_pg.py`, `test_pref.py`, `test_grpo.py`).
- **Formulas pinned to hand-computed and reference values:** TRL's advantages (±0.865875; 1.4997/−0.4999), k3 at
  ±0.1 and 0.5, the clip's gradient mask, the three normalisers, DAPO's overlong penalty (0, −0.5, −1.0, −1), the
  DPO loss 0.598139, GAE, the unbiased pass@k (0.916667, 0.728022, 0.914746) and pass^k, exact majority votes
  (`test_grpo.py`, `test_pref.py`, `test_ttc.py`).
- **Existing repo numbers reproduced.** The capacity primer's bank example — 12.07 s, 100.6 live, 95.3 and 381.3
  sessions per GPU, 9,156 and 20,615 tokens/s — both as constants and against `capacity.py` function by function,
  and the 06 scaling lab's `cost_per_call` ($0.007005, $0.035355) (`test_workload.py`).
- **The primer says what the code computes.** Every computed number in `../PRIMER.md` is recomputed and must
  appear verbatim (`test_primer_numbers.py`).

## Caveats: what the toys are and are not

A table of per-state softmaxes is a language model without generalisation: each state learns only from its own
visits, which is why thinking length grows slowly in the ThinkTask runs and why DPO's reach beyond its pairs is
exact here but a property of the network in practice. The directions of every effect — reward hacking, length
growth, over-optimisation, the length bias, diversity collapse — are robust across seeds (the tests check them);
the magnitudes are the toy's. Clip-higher's entropy effect is DAPO's measurement and is not reproduced here.
`ttc.question_set()` is a model of benchmark difficulty, not a benchmark. Every latency, GPU count and dollar figure
from `rlcore.workload` is a **model** — the capacity primer's formulas with its round datasheet numbers (verify),
plus a roofline step — not a measurement; the lab measures a real vLLM server.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs
this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../thinking-lab/`](../thinking-lab/) to train a tiny transformer with SFT then GRPO in torch, serve a real
thinking model (Qwen3-0.6B) in vLLM with `--reasoning-parser` on a free T4, run best-of-n and majority vote against
it, measure what long outputs do to ITL and KV usage, and drive one GRPO step with vLLM as the rollout engine. The
engine underneath is [`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md);
the sizing it builds on is [`gpu-capacity-planning`](../../gpu-capacity-planning/PRIMER.md). Where each tier runs
and what it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
