# distillation — teaching a small model what a large one knows, and what the student saves in serving

After this topic you can explain why a student learns more from a teacher's distribution than from labels,
choose between logit, sequence-level and on-policy distillation and the divergence each minimises, predict
exposure bias and what a distilled thinking model inherits, train a draft model for speculative decoding, measure
a student honestly, and decide with numbers whether distilling pays for itself.

## Start here

1. Read [PRIMER.md](PRIMER.md): "The one-minute version", then §1 Why distil and §2 Soft targets, temperature and
   the choice of divergence (40 min).
2. `cd distill-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 84 tests in about
   30 s; then open [`01_soft_targets_and_temperature`](distill-core/notebooks/01_soft_targets_and_temperature.ipynb)
   and watch soft targets beat hard labels on the same examples.
3. With torch on a laptop (or a free Colab CPU), distil a tiny transformer four ways:
   [`distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb`](distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain distillation in ten sections — §1 why distil · §2 soft targets, temperature and the choice of divergence · §3 sequence-level distillation · §4 on-policy distillation · §5 distilling reasoning · §6 feature distillation, pruning and vocabulary mismatch · §7 a distilled draft for speculative decoding · §8 measuring a student · §9 the economics of a student · §10 where to run it — each formula with a worked number and the core function that computes it; then "In a design review", a glossary, sources and a dated Verify list | ~2.5 h, read alongside the core | — |
| [`distill-core/`](distill-core/) | build it yourself in `distillcore` (standard library + numpy, ~1,000 lines): a toy language whose truth is known, tiny teachers and students with manual gradients, every KD loss and its gradient, forward/reverse KL and TRL's JSD, the SeqKD pipeline and exposure bias, GKD and its policy-gradient form checked by enumeration, trace distillation against RL, distilled drafts, agreement and Wilson intervals, and roofline serving costs and break-even; five fill-in notebooks | ~9 h | T0 |
| [`distill-lab/`](distill-lab/) | distil a tiny transformer four ways in torch; generate teacher data from a served model and train a 0.5–0.6B student with TRL; distil reasoning traces; measure a distilled draft under vLLM's speculative decoding; compute whether a student pays for itself from measured throughput (`distillab`; a fake teacher server and bundled outputs make every notebook run at T0) | ~10 h | T0 → T1 (T3 via the 04 lab's deploys) |

### Work it in this order

Read the primer sections, do the core notebook (T0), then the lab notebook — at T0 first, then on a GPU if you have
one.

| Step | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| Soft targets and three routes to a small model | §1 Why distil · §2 Soft targets, temperature and the choice of divergence · §6 Feature distillation, pruning and vocabulary mismatch | [`01_soft_targets_and_temperature`](distill-core/notebooks/01_soft_targets_and_temperature.ipynb) | [`01_kd_on_a_tiny_transformer`](distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb) | T0 (torch for the lab) |
| Divergences, SeqKD and on-policy distillation | §2 · §3 Sequence-level distillation: learning from the teacher's outputs · §4 On-policy distillation | [`02_forward_reverse_kl_and_on_policy_distillation`](distill-core/notebooks/02_forward_reverse_kl_and_on_policy_distillation.ipynb) | [`02_teacher_data_and_a_real_student`](distill-lab/notebooks/02_teacher_data_and_a_real_student.ipynb) | T0 → T1 |
| Distilling reasoning | §5 Distilling reasoning | [`03_distilling_reasoning_traces`](distill-core/notebooks/03_distilling_reasoning_traces.ipynb) | [`03_distilling_reasoning_traces_for_real`](distill-lab/notebooks/03_distilling_reasoning_traces_for_real.ipynb) | T0 → T1 |
| A distilled draft | §7 A distilled draft for speculative decoding | [`04_a_distilled_draft_for_speculative_decoding`](distill-core/notebooks/04_a_distilled_draft_for_speculative_decoding.ipynb) | [`04_a_distilled_draft_in_vllm`](distill-lab/notebooks/04_a_distilled_draft_in_vllm.ipynb) | T0 → T1 (24 GB) |
| Measuring a student, and whether it pays | §8 Measuring a student · §9 The economics of a student · §10 Where to run it | [`05_measuring_a_student_and_the_economics`](distill-core/notebooks/05_measuring_a_student_and_the_economics.ipynb) | [`05_is_the_student_worth_it`](distill-lab/notebooks/05_is_the_student_worth_it.ipynb) | T0 → T1 |

Each notebook ends with "In a design review" — the two-minute explanation and its drills. The primer's own
design-review section covers the whole topic.

## Run it

```bash
cd distill-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 84 tests (one skips by design), ~30 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../distill-lab
python3 -m pip install -e ".[dev,torch]"       # the CPU build of torch is enough at T0; ".[dev]" without it
python3 -m pytest -q                           # offline; torch paths skip when torch is absent
python3 -m jupyterlab notebooks
```

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | every core notebook; the lab's tiny-transformer distillation on CPU (torch), its fake OpenAI-compatible teacher (labelled simulated) and its bundled traces and curves (labelled illustrative) | laptop, Colab CPU or CI — $0 |
| **T1** | Qwen2.5-1.5B-Instruct as the teacher in vLLM on a T4 with `--dtype half`, SFT and logit KD of Qwen2.5-0.5B-Instruct with TRL, on-policy GKD, reasoning traces from Qwen3-1.7B or DeepSeek-R1-Distill-Qwen-1.5B; Qwen3-4B with a Qwen3-0.6B draft on a 24 GB card | Colab/Kaggle T4 (free; fp16 only), any 24 GB GPU (~$0.3–0.7/hr, verify) |
| **T3** | teacher inference behind the 04 serving lab's Cloud Run GPU or GKE deploy (no new Terraform here) | GCP, pay per use; see that lab's `deploy/` READMEs for cleanup |

Prices, free tiers and how to obtain GPUs on GCP and elsewhere: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`rl-and-thinking-models`](../rl-and-thinking-models/README.md) (primer §1, §2, §4, §5, §7); [`transformers`](../transformers/) (§6 training) | where distillation sits in post-training, the KL penalty and REINFORCE that on-policy distillation reuses, the R1 and Qwen3 distills, and the thinking workload a distilled student inherits; 6·N·D |
| before | [`gpu-capacity-planning`](../gpu-capacity-planning/PRIMER.md); [`01 roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/README.md) (§3, §8) | weights, KV bytes and sessions per GPU; the decode step and $/M tokens the economics reuse |
| beside | [`04 serving-engine`](../../04-inference-engine/serving-engine/README.md) (§7); [`04 quantization`](../../04-inference-engine/quantization/README.md) (§7, §8, §10) | speculative decoding, which a distilled draft feeds; the agreement metrics and the other way to shrink a model |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | cost per call and per conversation, and routing by cost — where a student and a cascade pay off |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/) — the platform lab's evals notebook | evals with intervals, the release gate for a student |

## Caveats

- **Toys, labelled.** The core's teachers and students are one-hidden-layer networks on a toy language whose truth
  is known; every effect's direction is pinned by tests, but magnitudes are the toy's. The reasoning toy is
  rlcore's ThinkTask formula, not a language model.
- **Bounds, not measurements.** Serving costs are an ideal roofline decode step (the arithmetic of layer 01's
  `roofline.llm` and `roofline.cost`, reproduced in tests); real engines reach a fraction of them. Measured
  throughput, real teachers and real students are the lab's T1 runs; its fake server and bundled outputs are
  labelled simulated or illustrative.
- **Dated facts.** TRL 1.14.0's distillation trainers and defaults, vLLM 0.30.0's speculative-decoding flags and
  metrics, the Qwen3, DeepSeek-R1, Minitron and EAGLE results, licences and all prices are as of September 2026
  and marked `(verify)`; the primer's [Verify list](PRIMER.md#verify-list) collects them.
