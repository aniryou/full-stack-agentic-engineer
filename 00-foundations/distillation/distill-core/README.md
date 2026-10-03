# distill-core — distillation small enough to compute exactly

After this, you can derive and implement what the distillation of a model does. The topics are these:

- Soft targets and the T² factor.
- Forward and reverse KL, and TRL's JSD.
- Sequence-level distillation and its exposure bias.
- On-policy distillation as policy gradient.
- Trace distillation against RL.
- The acceptance of a distilled draft.

You can also calculate a number that shows if a student is worth it. All of this is in `distillcore`, a package of ~1,050 lines
that uses only the standard library and numpy. In it, a toy language with a known truth and small networks with
manual gradients let you check every claim exactly.

## Start here

1. Read "The one-minute version" in [`../PRIMER.md`](../PRIMER.md). Then read §1 Why distil and §2 Soft targets,
   temperature and the choice of divergence.
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 86 tests run in about 30 s. They
   include "the REINFORCE estimator equals −∇KL by enumeration" and "the roofline primer's §8.1 table, number for
   number".
3. Open [`notebooks/01_soft_targets_and_temperature.ipynb`](notebooks/01_soft_targets_and_temperature.ipynb). See
   how a student learns more from soft targets than from labels on the same examples.

## What you get

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network.* Each notebook
opens with "The one-minute version" and works examples against the code. Then it sets exercises with a check cell
that prints ✅, and it ends with "In a design review". The finished versions are in [`solutions/`](solutions/). With
the primer, the work takes about 11 hours in all.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_soft_targets_and_temperature`](notebooks/01_soft_targets_and_temperature.ipynb) | Calculate soft targets at a temperature. Derive the KD gradient T·(q_T − p_T), and why T² is there. Find the logit-matching limit. Show that soft targets do better than hard labels at two examples per context (0.876 against 0.678). See the capacity gap. Prune a teacher by activation importance, and repair it by distillation. Over five seeds, see that pruning gives a head start in steps, not a better student. | §1, §2, §6 | ~1.5 h | T0 |
| [`02_forward_reverse_kl_and_on_policy_distillation`](notebooks/02_forward_reverse_kl_and_on_policy_distillation.ipynb) | Fit a one-mode student to a two-mode teacher under forward KL, reverse KL and JSD(β). Calculate the cost of a SeqKD pipeline in teacher tokens. Measure exposure bias: 1.000 on the prefixes of the teacher, and 0.788 on the prefixes of the student per position. Only 0.190 of the outputs of the student are correct through position 12. Remove the exposure bias with GKD. See reverse KL converge slowly from every start. The cause is that reverse KL increases the probability of a token by only a small quantity when the student gives it a low probability. Show that on-policy distillation is exactly REINFORCE with a dense reward, and find what it costs per token against GRPO. | §2, §3, §4 | ~2 h | T0 |
| [`03_distilling_reasoning_traces`](notebooks/03_distilling_reasoning_traces.ipynb) | Distil an RL-trained thinker from its traces by maximum likelihood in closed form. See the student take the thinking-length distribution of the thinker. Exchange accuracy for tokens with trace filters. On the student, get better results than RL at equal samples (0.892 against 0.445). Show why knowledge does not transfer. | §3, §5 | ~1.5 h | T0 |
| [`04_a_distilled_draft_for_speculative_decoding`](notebooks/04_a_distilled_draft_for_speculative_decoding.ipynb) | Calculate acceptance, tokens per pass and speedup. Show that a draft distilled from a fine-tuned target does better than an off-the-shelf one (α 0.988 against 0.890). See how the top-token probability of the target sets a limit on greedy acceptance. Select a draft size. Read the counters of vLLM. | §7 | ~1.5 h | T0 |
| [`05_measuring_a_student_and_the_economics`](notebooks/05_measuring_a_student_and_the_economics.ipynb) | Measure agreement (KL, top-1, top-k) and accuracy with Wilson intervals per slice. Find a student that does better than its teacher but agrees with it less. Calculate the cost of the teacher and the student on the roofline. The result is ~16× per token against the teacher on two H100s, and 96× against one. The teacher on one H100 is a baseline with insufficient capacity. Calculate the fixed cost, the break-even and the cost per correct answer of a cascade. | §1, §8, §9 | ~2 h | T0 |

§10 (where to run it) has no notebook. The notebooks of the lab are its T1 half.

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

Read the modules in this order. Each module opens with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`distillcore/tasks.py`](distillcore/tasks.py) | ~110 | `ModLang`: a next-token language with a known true distribution, a verifier, the cycles of its rule and enumerable continuations (and a fine-tuned "dialect"). `ThinkToy`: the ThinkTask formula of rlcore, 1 − e0·(1 − q)^L. |
| [`distillcore/losses.py`](distillcore/losses.py) | ~100 | Every loss as (loss, dlogits): hard CE, soft-target CE, T²·KL and its gradient T·(q_T − p_T), and Hinton's α-mix. Also logit MSE (the limit when T goes to ∞), the label noise 1 − Σp², and TRL's generalised JSD(β) with its closed-form gradient. |
| [`distillcore/tinylm.py`](distillcore/tinylm.py) | ~145 | A model with manual backprop: a context embedding, then tanh, then logits. Also Adam, sampling with temperature, per-token log-probs and ∇ of weighted log-probs, Minitron-style width pruning, and `fit_language` for teachers. |
| [`distillcore/divergences.py`](distillcore/divergences.py) | ~70 | Forward and reverse KL, JSD(β) and TV. A one-bump student fitted to a two-bump teacher under each divergence: mass between the modes against a dropped mode. |
| [`distillcore/seqkd.py`](distillcore/seqkd.py) | ~85 | Generate, then verify, then deduplicate, with the token bill. SFT on teacher text. Exposure bias as accuracy on the prefixes of the teacher against the prefixes of the student, per position and per output. |
| [`distillcore/onpolicy.py`](distillcore/onpolicy.py) | ~110 | The GKD loop of TRL (λ, β). Per-token rewards. The sequence and per-token REINFORCE forms, with the advantage convention of rlcore. The exact expectation by enumeration. The compute cost per token and per prompt, against GRPO. |
| [`distillcore/reasoning.py`](distillcore/reasoning.py) | ~100 | A stopping-rule policy, SFT on traces as a closed-form MLE, and trace filters (correct only, length caps). The REINFORCE baseline, which also makes the teacher. |
| [`distillcore/draft.py`](distillcore/draft.py) | ~65 | The acceptance, tokens per pass, speedup and best k of `minengine.spec`, stated again. Greedy acceptance. What the counters of vLLM show. Acceptance on the text of the target itself. |
| [`distillcore/eval.py`](distillcore/eval.py) | ~85 | KL, top-1 agreement and top-k overlap (the definitions of quantcore). Accuracy against the truth, and how sure a student is where it is incorrect. Wilson intervals, paired flips and the capability gap by slice. |
| [`distillcore/cost.py`](distillcore/cost.py) | ~185 | Model shapes. A roofline decode step, the batch that an ITL permits, and $/M tokens on one GPU or an ideal tensor-parallel group (`roofline.llm`, `roofline.cost`). The memory view of capacity.py. API prices. The conversion from 6·N·D to GPU-hours. The fixed cost, break-even and a cascade. |

## What the tests prove

`tests/` has one focused test per concept (78 tests plus 8 notebook-tooling checks, all offline, ~30 s in total). The tests
that carry the correctness claims are these:

- **The gradients are correct.** The gradient of each loss matches finite differences. The losses are soft CE, T²·KL,
  Hinton's mix, logit MSE and TRL's JSD at five values of β. The backward pass of the network and ∇ of weighted
  log-probabilities also match finite differences (`test_losses.py`, `test_tinylm.py`).
- **The formulas hold, pinned to hand-computed values.** These values are the soft targets of the five-token
  example, KL 0.25650, and the T² table and its limit 0.23. They also include the gradient (q − p)/T at T = 2 and
  the label noise 0.34. They include the JSD at β = 0.01…0.99, with the exact endpoints of TRL. They include the
  forward bimodal fit (μ = 5, s = 8.6) and the reverse bimodal fit (μ = 2, s = 0.7, KL = ln 2). They also include the flip between
  β = 0.6 and 0.7, the empirical-hazard MLE and Wilson intervals (`test_losses.py`, `test_divergences.py`,
  `test_reasoning.py`, `test_eval.py`).
- **On-policy distillation is policy gradient, exactly.** The per-token score-function estimate equals the analytic
  reverse-KL gradient. Over all 125 continuations of a small language, the REINFORCE expectation equals −∇KL by
  finite differences. The sampled estimator is unbiased. The per-token form is biased (31% of the norm) and has less
  noise (`test_onpolicy.py`).
- **The effects are real, not staged.** The tests show these effects:
  - Soft targets do better than hard labels.
  - Exposure bias appears under supervised KD. It is flat per position and compounds per output, and on-policy
    training removes it.
  - Reverse KL is slower than forward KL from every start that the tests tried.
  - The head start of a pruned student holds at every seed at 20 steps, and it is gone by 100 steps.
  - A like-for-like teacher (two H100s) has 16× the cost of the student, not 96×.
  - With a 32B teacher, on-policy distillation costs two times as much as GRPO per token.
  - A distilled draft accepts more than an off-the-shelf one.
  - Traces do better than RL at equal samples.
  - A verifier-filtered student does better than its weak teacher but agrees with it less.

  The tests are `test_seqkd.py`, `test_primer_numbers.py`, `test_cost.py`, `test_onpolicy.py`, `test_draft.py`,
  `test_reasoning.py` and `test_eval.py`.
- **Numbers from other parts of the repo, reproduced** (`test_repo_numbers.py`). The name of each test says whose
  numbers it reproduces. The numbers are these:
  - §7 of the serving primer (`minengine.spec`): α = 0.6 from its p and q. At α = 0.8, k = 4 gives 3.36 tokens per
    pass. At draft cost c = 0.1, the best k is 6, at 2.47×.
  - The §3.3 and §8.1 tables of the roofline primer, and `max_batch_by_memory` over 2 and 4 devices. The values
    for Llama-3.1-8B are batch 68, 9.93 ms, 6,847 tokens/s, $0.446 · FP8 193 · 4.52 and 50.5 ms.
  - The bank example of the capacity primer (88.8 and 355.1 sessions per GPU).
  - Quantization §8 (±0.0268, z = −2.9).
  - The ThinkTask of rlcore (L* = 20.23, and 0.304 at a mean of 1.0 tokens), and the convention of `reinforce_grad`
    with the teacher as reference.
  - The Wilson intervals of the memory primer and of the platform lab.
  - The `cost_per_call` of the 06 lab ($0.007005).

  The tests hold these numbers as constants. When the originals are in the checkout, the tests also compare
  function by function against them. They load the originals by path and leave no cache behind.
- **The primer says what the code computes.** The tests recompute every computed number in `../PRIMER.md`
  (`test_primer_numbers.py`). The exact numbers must appear verbatim. They include closed forms on given logits,
  enumerations, parameter counts, roofline costs and Wilson intervals of given counts.

  A number from a trained or sampled toy model is one seeded run on one CPU. This is important because the matmul
  kernels of OpenBLAS and the SIMD loops of numpy round differently in the last bit on each CPU family. Training
  makes that difference larger. Most such numbers move in the third digit (an AVX2-only AMD runner gets α = 0.889
  where the primer says 0.890). A few chaotic ones move by 0.1–0.2 (the on-policy student on rare inputs, the 4- and
  8-unit drafts).

  The tests compare those numbers, here and in the other test files, with a tolerance. The tolerance is about three
  times the largest deviation measured across 15 x86 kernel and SIMD variants. The qualitative claims stay exact.
  `OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" python3 -m pytest -q`
  reproduces the numbers of that AMD runner on an AVX-512 Intel machine.

## Caveats: what the toys are and are not

`ModLang` is a lookup table in the form of a language. Its rule has no smooth structure. Thus width is a clean
capacity knob, and you know the truth exactly. But `ModLang` is not text. Its greedy continuations stay on the
cycles of the rule, and this makes exposure bias visible here. In real text, the same mechanism spreads over far
more states.

Reverse KL is slow here from every start that the tests tried (KD, SFT, fresh). The cause is that reverse KL increases
the probability of a token only in proportion to the probability that the student gives that token. Also, on a lookup
table, students are often confidently incorrect. The recipes start on-policy distillation from an SFT checkpoint,
because on a real model that checkpoint already writes in the format of the teacher. The toy does not show if this
reason is true or false.

The teacher of the reasoning toy is narrow (RL made it on one task). It is not heavy-tailed like a real thinking
model. Every serving cost is an ideal roofline bound with list prices marked (verify), not a measurement. The lab
measures real engines.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three and the tests. On Colab, the first cell of each notebook clones the repo and installs
this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../distill-lab/`](../distill-lab/). There, you distil a small transformer four ways in torch. You generate
teacher data from a served model, and train a 0.5–0.6B student with TRL on a free T4. You also distil reasoning
traces from a thinking model, and measure a distilled draft under the speculative decoding of vLLM.

The theory that the core and the lab use is in [`rl-and-thinking-models`](../../rl-and-thinking-models/README.md) (post-training,
REINFORCE, thinking workloads) and [`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md)
(speculative decoding). The cost model is the [roofline](../../../01-hardware-gpu-fabric/roofline-and-fabric/README.md)
of layer 01. For where each tier runs and what it costs, see [`COMPUTE.md`](../../../COMPUTE.md). The licence is
MIT.
