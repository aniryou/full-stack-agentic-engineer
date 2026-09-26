# How to use this site

This site is a readable copy of the [full-stack-agentic-engineer](https://github.com/aniryou/full-stack-agentic-engineer)
repository: a study guide to the LLM inference and serving stack, from the GPU and its fabric at the bottom to the
agent application at the top. Every page here is built from the repository's Markdown and notebooks, so the site and
the repo always say the same thing. The code itself lives on GitHub; links to code files open there.

## The layers

The material is organised as eight layers, bottom-up: model-level foundations (00), hardware and fabric (01), CUDA
and NCCL (02), Kubernetes for GPUs (03), the inference engine (04), the orchestrator above the engines (05), the
gateway in front of the models (06), and the agent application (07). The [layers overview](../layers/index.md) has
one line per layer. You do not have to read them in order: the [curriculum](curriculum.md) suggests a path that
starts at the foundations, goes to the engine, down to the hardware that explains it, and back up.

## Three pieces per topic

The main topics come in three pieces. Use them in this order.

| Piece | What it is | How to use it |
|---|---|---|
| **Primer** | A document: the concepts in numbered sections, each formula with a worked number, a glossary, sources and a list of facts to re-check. | Read the sections a notebook asks for before opening it. |
| **Minimal core** | A small implementation from scratch, mostly standard-library Python, that runs on a laptop in seconds. | Where the concept is learned. Do every exercise. |
| **Detailed lab** | The fuller version: real GPU code paths, a real engine or cluster, deployment recipes. Each has an offline fallback. | Run it on a laptop first, then again on whatever hardware you have. |

Nine topics have all three: `roofline-and-fabric` (01), `cuda-and-nccl` (02), `gpu-scheduling` (03),
`serving-engine` and `quantization` (04), `serving-orchestration` (05), `mixture-of-experts` and
`rl-and-thinking-models` (00) and `sandboxed-execution` (07). The others are shaped differently: layer 01's
`gpu-primer/` and `gpu-deployment/` are a primer with written exercises and no code; the rest are usually a primer
plus practice notebooks or a lab of their own. Each topic's README says what it has.

## Run tiers and what they cost

Every notebook says which tier it needs. Every concept can be learned at T0.

| Tier | In plain words | Cost |
|---|---|---|
| **T0** | Your laptop, a free Colab CPU session, or CI. Simulators, calculators and from-scratch code. | Free |
| **T1** | One small GPU: a free Colab or Kaggle T4, or a rented 24 GB card. Real kernels and a real small model. | Free, or about $0.3–0.7 an hour (verify) |
| **T2** | A machine with several GPUs, ideally linked by NVLink, rented for about an hour. Collectives and tensor parallelism. | Free on Kaggle's 2×T4 (no NVLink); rented, about $1–6 for an hour on two NVLink GPUs, up to about $25 for eight (verify) |
| **T3** | Google Cloud managed services, deployed with Terraform with cheap defaults (L4, Spot, scale to zero). Optional. | Pay per use; tear it down afterwards |

Notebooks at T1 and above look for the hardware they need. When it is not there they run a clearly labelled T0 path:
simulator output says "simulated", bundled tool output says "sample output (illustrative)". [Compute](compute.md)
has the current options and prices, and the habits that keep a paid session from turning into a bill.

## How the labs check your understanding

- **Exercises are committed blank.** Exercise notebooks have `# YOUR CODE HERE` gaps.
- **A check cell follows the exercise** in most notebooks (all of those in the nine primer, core and lab
  topics). It prints ✅ when your answer is right, or fails with what is
  off; some older checks are lighter, and the 06 scaling notebooks print "not attempted" until you fill one in.
- **Worked answers are separate.** They usually sit in `solutions/` (shown under "Solutions" in the navigation). Try
  the exercise first.
- **Start again at any time.** On Colab, reopen the notebook from its badge. Locally, `git restore <notebook>` returns
  it to the blank version; labs whose notebooks are generated from sources rebuild them with
  `python3 tools/build_notebooks.py`.
- **Most modules end with a design review**: a short spoken explanation and a few drills, with answers.

## Colab or local

- **Colab:** exercise notebooks on this site have an "Open in Colab" button. The notebook's first cell clones the
  repo and installs that lab's dependencies; nothing else to set up. See [Running in Colab](colab.md).
- **Locally:** each lab is independent, with its own `requirements.txt` or `pyproject.toml`.

```bash
git clone https://github.com/aniryou/full-stack-agentic-engineer
cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core
python3 -m pip install -e .
python3 -m pytest -q          # most labs ship tests
jupyter lab notebooks/
```

## Where the code lives

Notebooks and documents are rendered here. Python packages, tests, Terraform, Kubernetes manifests and scripts are
not: links to them open the file on GitHub. The site shows notebooks as committed, without running them, so
exercise notebooks appear blank and solution notebooks show their saved output. The Colab setup cell at the top of
each notebook is left off these pages; it is still the first cell when you open the notebook in Colab.
