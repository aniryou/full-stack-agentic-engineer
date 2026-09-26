# full-stack-agentic-engineer — the LLM serving stack, from GPUs to agents

**Read it as a site:** <https://aniryou.github.io/full-stack-agentic-engineer/> (same content, with search and rendered notebooks).

A learning repository for the LLM serving stack, from the GPUs and fabric at the bottom, through the runtime,
Kubernetes, the inference engine, the orchestrator and the gateway, to the agent application at the top. It is
organised as eight layers and, within each layer, by topic: 347 notebooks (exercise and solution versions), all of
which run on a laptop or in Google Colab ([`COLAB.md`](COLAB.md)). Nine topics come as a primer, a small
from-scratch implementation and a fuller lab: `roofline-and-fabric` (01), `cuda-and-nccl` (02), `gpu-scheduling`
(03), `serving-engine` and `quantization` (04), `serving-orchestration` (05), `mixture-of-experts` and
`rl-and-thinking-models` (00) and `sandboxed-execution` (07). The other topics vary in shape: layer 01's
`gpu-primer/` and `gpu-deployment/` are a primer with written exercises and no code; the rest are usually a primer
plus practice notebooks or a lab; each topic's or lab's README says what it contains.

## Start here

1. **Pick a layer** from the [stack map](#the-stack) below — start where your question lives, or at the bottom
   and work up.
2. **Read its primer.** Most topics have a primer (`PRIMER.md`, a `*-primer.md` or a lab's `docs/primer.md`) that
   explains the concepts with worked numbers; `agent-fundamentals/` teaches through its lab notebooks instead. The
   layer's `README.md` links it and says which sections to read first, with the time and tier of each topic.
3. **Run its core notebooks**, in Colab (the links at the end of each layer README) or locally (see [Run it](#run-it)).
   Most exercises are followed by a check cell — all of them in the primer, core and lab topics — that prints ✅
   when your answer is right; some older checks are lighter, and the 06 scaling notebooks print "not attempted"
   until you fill an exercise in.

**Learning path:** [`CURRICULUM.md`](CURRICULUM.md) gives the order to work the whole repo, module by module, with
hours, tiers, shorter routes and cross-layer design drills.

**Compute and cost:** [`COMPUTE.md`](COMPUTE.md) says where each tier runs — laptop, free Colab or Kaggle GPUs,
rented GPUs, Google Cloud — what it costs, and which lab needs which tier.

## The stack

Read bottom-up: each layer is built on the one below it.

```
  ┌─ 07-application-agent-framework   build the agent: loop, tools, sandboxes, state, durability, evals, RAG
  │  06-gateway                       decide who may run what: identity, policy, rate limits, admission, cost
  │  05-orchestrator                  run many engine replicas as one service: routing, autoscaling, P/D split
  │  04-inference-engine              run one model fast: attention kernels, the KV cache, batching, quantization
  │  03-kubernetes-gpu                make GPUs schedulable: device plugin, scheduler, gangs, quotas, capacity
  │  02-cuda-nccl-runtime             get from a container to a GPU: driver, CUDA, NCCL, container runtime
  └─ 01-hardware-gpu-fabric           read the machine: spec sheets, the roofline, fabrics, cost of a token
     00-foundations                   the model itself: transformer internals, capacity math, MoE, RL and thinking
```

## What is inside

One entry per layer: what you will be able to do, then its topic folders.

- **00 · Foundations** — [`00-foundations/`](00-foundations/README.md). Build a tiny GPT; size memory and bandwidth
  for a model; predict what MoE and thinking models do to serving; implement REINFORCE, DPO and GRPO.
  Topics: `transformers/`, `gpu-capacity-planning/`, `model-landscape/`,
  [`mixture-of-experts/`](00-foundations/mixture-of-experts/README.md),
  [`rl-and-thinking-models/`](00-foundations/rl-and-thinking-models/README.md).
- **01 · Hardware and fabric** — [`01-hardware-gpu-fabric/`](01-hardware-gpu-fabric/README.md). Read a spec sheet
  and say whether a step is compute- or memory-bound; price a collective; compute $/M tokens; measure your machine.
  Topics: `gpu-primer/`, `gpu-deployment/`,
  [`roofline-and-fabric/`](01-hardware-gpu-fabric/roofline-and-fabric/README.md).
- **02 · CUDA, NCCL and runtime** — [`02-cuda-nccl-runtime/`](02-cuda-nccl-runtime/README.md). Predict coalescing,
  occupancy and all-reduce cost; run Numba kernels; measure collectives; see how a container gets a GPU.
  Topic: [`cuda-and-nccl/`](02-cuda-nccl-runtime/cuda-and-nccl/README.md).
- **03 · Kubernetes and GPU scheduling** — [`03-kubernetes-gpu/`](03-kubernetes-gpu/README.md). Explain why a GPU pod is
  Pending; place gangs; set Kueue quotas; choose Spot, flex-start or reservations.
  Topic: [`gpu-scheduling/`](03-kubernetes-gpu/gpu-scheduling/README.md).
- **04 · Inference engine** — [`04-inference-engine/`](04-inference-engine/README.md). Build an engine's step loop,
  scheduler and prefix cache; size a KV cache; tune a real vLLM against an SLO; choose a quantization scheme.
  Topics: `kv-cache/`, `paged-attention/`, `flash-attention/`,
  [`serving-engine/`](04-inference-engine/serving-engine/README.md),
  [`quantization/`](04-inference-engine/quantization/README.md),
  [`vllm-internals/`](04-inference-engine/vllm-internals/README.md).
- **05 · Orchestrator** — [`05-orchestrator/`](05-orchestrator/README.md). Route on prefix affinity and load;
  autoscale on the right signals; size a prefill/decode split — in a simulator, then a real router.
  Topic: [`serving-orchestration/`](05-orchestrator/serving-orchestration/README.md).
- **06 · Gateway** — [`06-gateway/`](06-gateway/README.md). Give agents identities, exchange tokens, enforce
  policy with an audit trail; plan capacity, find the provisioned-throughput break-even, add admission control.
  Topics: [`identity-security/`](06-gateway/identity-security/README.md),
  [`scaling-admission-cost/`](06-gateway/scaling-admission-cost/README.md).
- **07 · Agents and applications** — [`07-application-agent-framework/`](07-application-agent-framework/README.md). Write
  an agent loop; make long-running agents durable; build RAG and vector indexes; sandbox model-written code.
  Topics: `agent-fundamentals/`, `long-running-durable/`, `retrieval-rag/`,
  [`sandboxed-execution/`](07-application-agent-framework/sandboxed-execution/README.md).

Several topics are worked on more than one provider (Google Cloud, Mistral) so the same concept can be compared
across stacks. Each layer `README.md` has its topics with time and tier, where to start, how it fits with the layers around it and
the Colab links.

## How the labs work

The nine primer + core + lab topics (listed at the top of this page) follow the same pattern, so once you have done
one you know how to do the rest. The other topics are shaped differently — a primer with written exercises
(`gpu-primer/`, `gpu-deployment/`), a primer plus practice notebooks (for example `kv-cache/`, `paged-attention/`,
`flash-attention/`), or a lab of its own (most of 06 and 07) — and their READMEs say what they have.

| Piece | What it is |
|---|---|
| **Primer** | the concepts, with worked numbers; read it first, one section at a time |
| **Core** | a minimal implementation, usually standard-library Python, plus fill-in notebooks that *predict* what the real system does |
| **Lab** | the detailed version: real tools, benchmarks and deploy recipes that *run* or *measure* the same ideas |

**Tiers** say what hardware a notebook or recipe needs, and the nine primer + core + lab topics mark every step
with one ([`COMPUTE.md`](COMPUTE.md) has the details and prices):

- **T0** — a laptop, Colab CPU or CI. Free. Every concept is learnable here.
- **T1** — one small GPU: a free Colab or Kaggle T4, or a rented card.
- **T2** — a multi-GPU box, rented for an hour (or Kaggle's free 2×T4).
- **T3** — the Google Cloud deployment via Terraform, with the cheapest defaults. Optional; never a prerequisite.

Notebooks that need a GPU detect what they have and fall back to a clearly labelled T0 path.

**Exercises** sit in `notebooks/` (or `exercises/`, or `practice/`) with `# YOUR CODE HERE`; in the primer, core and
lab topics each is followed by a check cell that prints ✅ when your answer is right, and worked answers are in
`solutions/` (older labs keep them beside the exercise, for example `*_solution.ipynb`). Exercises are committed blank, so
`git restore <notebook>` returns one to its starting state (for labs built from `notebooks_src/`, re-run that
lab's `python3 tools/build_notebooks.py`).

## Run it

**In Colab:** open any notebook from its link in its layer README. There is nothing to set up: the first cell
clones this repo and installs that lab's dependencies. [`COLAB.md`](COLAB.md) has the details, including how
to keep your edits.

**Locally:** each lab is independent and carries its own `requirements.txt` / `pyproject.toml`; there is no
repo-wide environment.

```bash
git clone https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/07-application-agent-framework/agent-fundamentals/agent-core
pip install -e .          # or: pip install -r requirements.txt
python3 -m pytest -q      # most labs ship tests
# open notebooks/ in Jupyter, or use the Colab links in this layer's README
```

## Layout

```
.
├── 00-foundations/ … 07-application-agent-framework/   one folder per layer; labs in topic sub-folders
├── CURRICULUM.md          the learning path: modules, order, hours, tiers, design drills
├── COMPUTE.md             where to run each tier, what it costs, which lab needs which tier
├── COLAB.md               running notebooks in Colab
├── LICENSE                MIT, for everything not otherwise licensed (see Licence)
├── site/, mkdocs.yml      the guide site: hand-written pages and theme; the site configuration
├── requirements-site.txt  what building the site needs
├── .github/workflows/     builds and publishes the site on every push to main
├── tools/                 Colab bootstrap and link generators; site/ (the site's page generator)
├── tools/orchestration/   notes on how the topics were built and reviewed — maintainer material, not lessons
├── raw/                   inbox for new material (gitignored except its README)
├── CLAUDE.md              instructions for the agent that maintains the repo
└── README.md              this file
```

## Licence

Everything in this repository that does not carry its own licence — the primers, the curriculum, the compute
guide, the site's text and the code outside the labs — is under the MIT licence in [`LICENSE`](LICENSE). A lab
with its own `LICENSE` file keeps it: most are MIT as well, and two are Apache 2.0 —
[`agentic-identity-gcp-lab`](06-gateway/identity-security/agentic-identity-gcp-lab/LICENSE) (06) and
[`lra-gcp`](07-application-agent-framework/long-running-durable/lra/lra-gcp/LICENSE) (07). Third-party names and
products are trademarks of their owners and are mentioned only to explain how they work.

## Maintaining the repo

New material arrives in [`raw/`](raw/README.md) and is moved, never copied, into the layer and topic folder it
belongs to; [`raw/README.md`](raw/README.md) says what a good drop looks like. [`CLAUDE.md`](CLAUDE.md) holds the
maintainer and agent instructions: the layer rules, the reorganisation checklist and a log of decisions.
Corrections are welcome as [GitHub issues](https://github.com/aniryou/full-stack-agentic-engineer/issues).
