# distill-core — distillation small enough to compute exactly

After this you can derive and implement what distilling a model does — soft targets and the T² factor, forward and
reverse KL and TRL's JSD, sequence-level distillation and its exposure bias, on-policy distillation as policy
gradient, trace distillation against RL, a distilled draft's acceptance — and put a number on whether a student is
worth it, all in `distillcore`, a standard-library-plus-numpy package of ~1,050 lines where a toy language with a
known truth and tiny networks with manual gradients let every claim be checked exactly.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md): "The one-minute version", then §1 Why distil and §2 Soft targets,
   temperature and the choice of divergence.
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 86 tests in about 30 s, including "the
   REINFORCE estimator equals −∇KL by enumeration" and "the roofline primer's §8.1 table, number for number".
3. Open [`notebooks/01_soft_targets_and_temperature.ipynb`](notebooks/01_soft_targets_and_temperature.ipynb) and
   watch a student learn more from soft targets than from labels on the same examples.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network.* Each notebook opens with
"The one-minute version", works examples against the code, sets exercises with a check cell that prints ✅, and
ends with "In a design review". Finished versions are in [`solutions/`](solutions/). About 11 hours in all with the
primer.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_soft_targets_and_temperature`](notebooks/01_soft_targets_and_temperature.ipynb) | compute soft targets at a temperature, derive the KD gradient T·(q_T − p_T) and why T² is there, find the logit-matching limit; show soft targets beating hard labels at two examples per context (0.876 against 0.678); see the capacity gap; prune a teacher by activation importance, repair it by distillation, and see over five seeds that pruning buys a head start in steps, not a better student | §1, §2, §6 | ~1.5 h | T0 |
| [`02_forward_reverse_kl_and_on_policy_distillation`](notebooks/02_forward_reverse_kl_and_on_policy_distillation.ipynb) | fit a one-mode student to a two-mode teacher under forward KL, reverse KL and JSD(β); price a SeqKD pipeline in teacher tokens; measure exposure bias (1.000 on the teacher's prefixes, 0.788 on its own per position, and only 0.190 of its outputs right through position 12); remove it with GKD, and see reverse KL converge slowly from every start because it barely lifts a token the student gives little probability; check that on-policy distillation is REINFORCE with a dense reward, exactly, and what it costs per token against GRPO | §2, §3, §4 | ~2 h | T0 |
| [`03_distilling_reasoning_traces`](notebooks/03_distilling_reasoning_traces.ipynb) | distil an RL-trained thinker from its traces by maximum likelihood in closed form; see the student inherit its thinking-length distribution; trade accuracy for tokens with trace filters; beat RL on the student at equal samples (0.892 against 0.445); show why knowledge does not transfer | §3, §5 | ~1.5 h | T0 |
| [`04_a_distilled_draft_for_speculative_decoding`](notebooks/04_a_distilled_draft_for_speculative_decoding.ipynb) | compute acceptance, tokens per pass and speedup; show a draft distilled from a fine-tuned target beating an off-the-shelf one (α 0.988 against 0.890); see greedy drafting capped by the target's top-token probability; pick a draft size; read vLLM's counters | §7 | ~1.5 h | T0 |
| [`05_measuring_a_student_and_the_economics`](notebooks/05_measuring_a_student_and_the_economics.ipynb) | measure agreement (KL, top-1, top-k) and accuracy with Wilson intervals per slice; find a student that beats its teacher while agreeing less; cost teacher and student on the roofline (~16× per token against the teacher on two H100s; 96× against one, a capacity-starved baseline); compute the fixed cost, break-even and a cascade's cost per correct answer | §1, §8, §9 | ~2 h | T0 |

§10 (where to run it) has no notebook; the lab's notebooks are its T1 half.

## Run it

```bash
cd distill-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 86 tests (one skips by design), ~30 s
python3 -m jupyterlab notebooks                # do the exercises
```

The library itself needs only numpy:

```python
import numpy as np
from distillcore import ModLang, TinyLM, train, losses, eval
from distillcore.tinylm import fit_language

lang = ModLang(11, 0.2)                                   # (a + b) mod 11 with probability 0.8, neighbours 0.1
teacher = fit_language(lang, H=64)                        # knows the language: KL to the truth ≈ 0.0004 nats
C = lang.contexts()
ctx = C[np.random.default_rng(1).integers(0, 121, 242)]   # two examples per context
student = TinyLM(11, 16, seed=1)
train(student, ctx, lambda z, i: losses.kd(z, teacher.logits(ctx)[i], T=1.0), steps=400)
print(eval.vs_truth(student, lang))                       # rule accuracy 0.876 (hard labels: 0.678)
```

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`distillcore/tasks.py`](distillcore/tasks.py) | ~110 | `ModLang`, a next-token language with a known true distribution, a verifier, its rule's cycles and enumerable continuations (and a fine-tuned "dialect"); `ThinkToy`, rlcore's ThinkTask formula 1 − e0·(1 − q)^L |
| [`distillcore/losses.py`](distillcore/losses.py) | ~100 | every loss as (loss, dlogits): hard CE, soft-target CE, T²·KL and its gradient T·(q_T − p_T), Hinton's α-mix, logit MSE (the T → ∞ limit), the label noise 1 − Σp², and TRL's generalised JSD(β) with its closed-form gradient |
| [`distillcore/tinylm.py`](distillcore/tinylm.py) | ~145 | a context-embedding → tanh → logits model with manual backprop; Adam; sampling with temperature; per-token log-probs and ∇ of weighted log-probs; Minitron-style width pruning; `fit_language` for teachers |
| [`distillcore/divergences.py`](distillcore/divergences.py) | ~70 | forward and reverse KL, JSD(β), TV; a one-bump student fitted to a two-bump teacher under each — mass between the modes against a dropped mode |
| [`distillcore/seqkd.py`](distillcore/seqkd.py) | ~85 | generate → verify → deduplicate with the token bill; SFT on teacher text; exposure bias as accuracy on the teacher's prefixes against the student's own, per position and per output |
| [`distillcore/onpolicy.py`](distillcore/onpolicy.py) | ~110 | TRL's GKD loop (λ, β); per-token rewards; the sequence and per-token REINFORCE forms with rlcore's advantage convention; the exact expectation by enumeration; compute per token and per prompt against GRPO |
| [`distillcore/reasoning.py`](distillcore/reasoning.py) | ~100 | a stopping-rule policy; SFT on traces as a closed-form MLE; trace filters (correct only, length caps); the REINFORCE baseline that also makes the teacher |
| [`distillcore/draft.py`](distillcore/draft.py) | ~65 | `minengine.spec`'s acceptance, tokens per pass, speedup and best k, restated; greedy acceptance; what vLLM's counters show; acceptance on the target's own text |
| [`distillcore/eval.py`](distillcore/eval.py) | ~85 | KL, top-1 agreement and top-k overlap (quantcore's definitions); accuracy against the truth, and how sure a student is where it is wrong; Wilson intervals; paired flips; the capability gap by slice |
| [`distillcore/cost.py`](distillcore/cost.py) | ~185 | model shapes; a roofline decode step, the batch an ITL allows and $/M tokens on one GPU or an ideal tensor-parallel group (`roofline.llm`, `roofline.cost`); capacity.py's memory view; API prices; 6·N·D → GPU-hours; the fixed cost, break-even and a cascade |

## What the tests prove

`tests/` has one focused test per concept (78, plus 8 notebook-tooling checks; offline, ~30 s in all). The ones that
carry the correctness claims:

- **The gradients are right.** Every loss's gradient — soft CE, T²·KL, Hinton's mix, logit MSE and TRL's JSD at five
  values of β — matches finite differences; so do the network's backward pass and ∇ of weighted log-probabilities
  (`test_losses.py`, `test_tinylm.py`).
- **The formulas hold, pinned to hand-computed values.** The five-token example's soft targets, KL 0.25650, the T²
  table and its limit 0.23; the gradient (q − p)/T at T = 2; label noise 0.34; the JSD at β = 0.01…0.99 including
  TRL's exact endpoints; the bimodal fits (forward μ = 5, s = 8.6; reverse μ = 2, s = 0.7, KL = ln 2) and the flip
  between β = 0.6 and 0.7; the empirical-hazard MLE; Wilson intervals (`test_losses.py`, `test_divergences.py`,
  `test_reasoning.py`, `test_eval.py`).
- **On-policy distillation is policy gradient, exactly.** The per-token score-function estimate equals the
  analytic reverse-KL gradient; over all 125 continuations of a small language, the REINFORCE expectation equals
  −∇KL by finite differences; the sampled estimator is unbiased; the per-token form is biased (31% of the norm)
  and quieter (`test_onpolicy.py`).
- **The effects are real, not staged.** Soft targets beat hard labels; exposure bias appears under supervised KD,
  flat per position and compounding per output, and on-policy training removes it; reverse KL is slower than
  forward from every start tried; a pruned student's head start holds at every seed at 20 steps and is gone by 100;
  a like-for-like teacher (two H100s) is 16× the student's cost, not 96×; on-policy distillation costs twice GRPO
  per token with a 32B teacher; a distilled draft accepts more than an off-the-shelf one; traces beat RL at equal
  samples; a verifier-filtered student beats its weak teacher while agreeing less (`test_seqkd.py`,
  `test_primer_numbers.py`, `test_cost.py`, `test_onpolicy.py`, `test_draft.py`, `test_reasoning.py`,
  `test_eval.py`).
- **Existing repo numbers reproduced** (`test_repo_numbers.py`, each test named for whose numbers): the serving
  primer's §7 (`minengine.spec`: α = 0.6 from its p and q; at α = 0.8, k = 4 gives 3.36 tokens per pass, and at
  draft cost c = 0.1 the best k is 6, at 2.47×), the roofline primer's §3.3 and §8.1 tables and `max_batch_by_memory` over 2 and 4 devices (Llama-3.1-8B: batch 68, 9.93 ms, 6,847 tokens/s, $0.446; FP8 193; 4.52 and 50.5 ms), the capacity
  primer's bank example (88.8 and 355.1 sessions per GPU), quantization §8 (±0.0268, z = −2.9), rlcore's ThinkTask
  (L* = 20.23; 0.304 at a mean of 1.0 tokens) and `reinforce_grad`'s convention with the teacher as reference, the
  memory primer's and the platform lab's Wilson intervals, and the 06 lab's `cost_per_call` ($0.007005) — as
  constants, and function by function against the originals when they are in the checkout (loaded by path, no
  cache left behind).
- **The primer says what the code computes.** Every computed number in `../PRIMER.md` is recomputed and must appear
  verbatim (`test_primer_numbers.py`). A closed-form number must match to the digit; a trained or sampled number is
  one CPU's run (numpy's matrix kernel differs by microarchitecture, and a few hundred Adam steps grow the last-bit
  difference into a slightly different model), so it is pinned to the reference run within a tolerance measured across
  kernels — `tests/pins.py`, with `tools/host_sensitivity.py` to remeasure after a change — and the claim behind it is
  asserted on whatever machine runs the test.

## Caveats: what the toys are and are not

`ModLang` is a lookup table dressed as a language: its rule has no smooth structure, which makes width a clean
capacity knob and the truth exactly known, but it is not text. Its greedy continuations stay on the rule's cycles,
which is what makes exposure bias visible here; in real text the same mechanism is spread over far more states.
Reverse KL is slow here from every start tried (KD, SFT, fresh): it lifts a token only in proportion to the
student's own probability of it, and on a lookup table students are often confidently wrong. The recipes start
on-policy distillation from an SFT checkpoint because on a real model it already writes in the teacher's format;
the toy neither shows nor refutes that. The reasoning toy's teacher is narrow (it was made by RL on one task), not heavy-tailed like a real thinking
model. Every serving cost is an ideal roofline bound with list prices marked (verify), not a measurement; the lab
measures real engines.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs this
package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../distill-lab/`](../distill-lab/) to distil a tiny transformer four ways in torch, generate teacher data
from a served model and train a 0.5–0.6B student with TRL on a free T4, distil reasoning traces from a thinking
model, and measure a distilled draft under vLLM's speculative decoding. The theory underneath is
[`rl-and-thinking-models`](../../rl-and-thinking-models/README.md) (post-training, REINFORCE, thinking workloads)
and [`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md) (speculative
decoding); the cost model is layer 01's [roofline](../../../01-hardware-gpu-fabric/roofline-and-fabric/README.md).
Where each tier runs and what it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
