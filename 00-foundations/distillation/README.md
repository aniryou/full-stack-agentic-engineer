# distillation — teaching a small model what a large one knows, and what the student saves in serving

After this topic, you can do these things:

- Explain why a student learns more from the distribution of a teacher than from labels.
- Select between logit, sequence-level and on-policy distillation, and the divergence that each one minimises.
- Predict exposure bias, and what a distilled thinking model gets from its teacher.
- Train a draft model for speculative decoding.
- Measure a student honestly.
- Decide with numbers if distillation pays for itself.

## Start here

1. Read "The one-minute version" in [PRIMER.md](PRIMER.md). Then read §1 Why distil and §2 Soft targets,
   temperature and the choice of divergence. This takes 40 min.
2. Run `cd distill-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 86 tests run in
   about 30 s. Then open [`01_soft_targets_and_temperature`](distill-core/notebooks/01_soft_targets_and_temperature.ipynb).
   See how soft targets do better than hard labels on the same examples.
3. With torch on a laptop (or on a free Colab CPU), distil a small transformer four ways:
   [`distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb`](distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb).

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, and it is optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain distillation in ten sections. The first three are §1 why distil · §2 soft targets, temperature and the choice of divergence · §3 sequence-level distillation. The next three are §4 on-policy distillation · §5 distilling reasoning · §6 feature distillation, pruning and vocabulary mismatch. The next two are §7 a distilled draft for speculative decoding · §8 measuring a student. The last two are §9 the economics of a student · §10 where to run it. Each formula has a worked number and the core function that computes it. After the ten sections come "In a design review", a glossary, sources and a dated Verify list. | ~2.5 h. Read it together with the core. | Not applicable |
| [`distill-core/`](distill-core/) | Build it yourself in `distillcore` (standard library + numpy, ~1,050 lines). It has a toy language with a known truth, and small teachers and students with manual gradients. It has every KD loss and its gradient, forward/reverse KL and TRL's JSD. It has the SeqKD pipeline and exposure bias, and GKD with an enumeration check of its policy-gradient form. It also has trace distillation against RL, distilled drafts, agreement and Wilson intervals, and roofline serving costs and break-even. There are five fill-in notebooks. | ~8.5 h | T0 |
| [`distill-lab/`](distill-lab/) | Distil a small transformer four ways in torch. Generate teacher data from a served model, and train a 0.5–0.6B student with TRL. Distil reasoning traces. Measure a distilled draft under the speculative decoding of vLLM. From measured throughput, calculate if a student pays for itself. The package is `distillab`. A fake teacher server and bundled outputs let every notebook run at T0. | ~10 h | T0 to T1 (T3 through the deploys of the 04 lab) |

### Work it in this order

Read the primer sections. Then do the core notebook (T0). Then do the lab notebook at T0 first. If you have a GPU,
do the lab notebook on the GPU after that.

| Step | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| Soft targets and three routes to a small model | §1 Why distil · §2 Soft targets, temperature and the choice of divergence · §6 Feature distillation, pruning and vocabulary mismatch | [`01_soft_targets_and_temperature`](distill-core/notebooks/01_soft_targets_and_temperature.ipynb) | [`01_kd_on_a_tiny_transformer`](distill-lab/notebooks/01_kd_on_a_tiny_transformer.ipynb) | T0 (torch for the lab) |
| Divergences, SeqKD and on-policy distillation | §2 · §3 Sequence-level distillation: learning from the teacher's outputs · §4 On-policy distillation | [`02_forward_reverse_kl_and_on_policy_distillation`](distill-core/notebooks/02_forward_reverse_kl_and_on_policy_distillation.ipynb) | [`02_teacher_data_and_a_real_student`](distill-lab/notebooks/02_teacher_data_and_a_real_student.ipynb) | T0 to T1 |
| Distillation of reasoning | §5 Distilling reasoning | [`03_distilling_reasoning_traces`](distill-core/notebooks/03_distilling_reasoning_traces.ipynb) | [`03_distilling_reasoning_traces_for_real`](distill-lab/notebooks/03_distilling_reasoning_traces_for_real.ipynb) | T0 to T1 |
| A distilled draft | §7 A distilled draft for speculative decoding | [`04_a_distilled_draft_for_speculative_decoding`](distill-core/notebooks/04_a_distilled_draft_for_speculative_decoding.ipynb) | [`04_a_distilled_draft_in_vllm`](distill-lab/notebooks/04_a_distilled_draft_in_vllm.ipynb) | T0 to T1 (24 GB) |
| The measurement of a student, and if it pays | §8 Measuring a student · §9 The economics of a student · §10 Where to run it | [`05_measuring_a_student_and_the_economics`](distill-core/notebooks/05_measuring_a_student_and_the_economics.ipynb) | [`05_is_the_student_worth_it`](distill-lab/notebooks/05_is_the_student_worth_it.ipynb) | T0 to T1 |

Each notebook ends with "In a design review". That section has the two-minute explanation and its drills. The
design-review section of the primer covers the whole topic.

## Run it

```bash
cd distill-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 86 tests (one skips by design), ~30 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../distill-lab
python3 -m pip install -e ".[dev]"            # numpy, aiohttp; dev: pytest, jupyter, matplotlib
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: the CPU build is enough at T0
python3 -m pytest -q                           # offline; torch paths skip when torch is absent
python3 -m jupyterlab notebooks
```

On Colab, the first cell of each notebook clones the repo and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | Every core notebook. The lab runs the distillation of a small transformer on CPU (torch), and its fake OpenAI-compatible teacher (labelled simulated). It also has bundled traces and curves (labelled illustrative). | Laptop, Colab CPU or CI. The cost is $0. |
| **T1** | Qwen2.5-1.5B-Instruct as the teacher in vLLM on a T4 with `--dtype half`. SFT and logit KD of Qwen2.5-0.5B-Instruct with TRL. On-policy GKD. Reasoning traces from Qwen3-1.7B or DeepSeek-R1-Distill-Qwen-1.5B. Qwen3-4B with a Qwen3-0.6B draft on a 24 GB card. | Colab/Kaggle T4 (free, fp16 only). Any 24 GB GPU (~$0.3–0.7/hr, verify). |
| **T3** | Teacher inference behind the Cloud Run GPU or GKE deploy of the 04 serving lab. This topic adds no new Terraform. | GCP, pay per use. For cleanup, see the `deploy/` READMEs of that lab. |

For prices, free tiers and how to get GPUs on GCP and elsewhere, see [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`rl-and-thinking-models`](../rl-and-thinking-models/README.md) (primer §1, §2, §4, §5, §7) and [`transformers`](../transformers/) (§6 training) | From `rl-and-thinking-models`: where distillation sits in post-training, and the KL penalty and REINFORCE that on-policy distillation uses again. Also from it: the R1 and Qwen3 distills, and the thinking workload that a distilled student gets from its teacher. From `transformers`: 6·N·D. |
| before | [`gpu-capacity-planning`](../gpu-capacity-planning/PRIMER.md) and [`01 roofline-and-fabric`](../../01-hardware-gpu-fabric/roofline-and-fabric/README.md) (§3, §8) | From `gpu-capacity-planning`: weights, KV bytes and sessions per GPU. From `roofline-and-fabric`: the decode step and $/M tokens, which the economics use again. |
| beside | [`04 serving-engine`](../../04-inference-engine/serving-engine/README.md) (§7) and [`04 quantization`](../../04-inference-engine/quantization/README.md) (§7, §8, §10) | From `serving-engine`: speculative decoding, which uses a distilled draft. From `quantization`: the agreement metrics, and the other way to make a model smaller. |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | Cost per call and per conversation, and routing by cost, where a student and a cascade pay for themselves. |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/): the evals notebook of the platform lab | Evals with intervals. These are the release gate for a student. |

## Caveats

- **Toys, labelled.** The teachers and students of the core are one-hidden-layer networks on a toy language with a
  known truth. Tests pin the direction of each effect, but the magnitudes are those of the toy. The reasoning toy
  is the ThinkTask formula of rlcore, not a language model.
- **Bounds, not measurements.** The serving costs are those of an ideal roofline decode step. This is the
  arithmetic of `roofline.llm` and `roofline.cost` in layer 01, and the tests reproduce it. Real engines reach a
  fraction of these costs. The measured throughput, the real teachers and the real students are in the T1 runs of
  the lab. The fake server and the bundled outputs of the lab are labelled simulated or illustrative.
- **Dated facts.** The facts that follow are as of September 2026, and each one has the mark `(verify)`. They are
  the distillation trainers and defaults of TRL 1.14.0, and the speculative-decoding flags and metrics of vLLM
  0.30.0. They are also the Qwen3, DeepSeek-R1, Minitron and EAGLE results, the licences and all prices. The
  [Verify list](PRIMER.md#verify-list) of the primer collects them.
