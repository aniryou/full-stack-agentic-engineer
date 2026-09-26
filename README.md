# full-stack-agentic-engineer — the LLM serving stack, from GPUs to agents

A hands-on course in how an LLM request is served, layer by layer: the GPUs and fabric at the bottom, the
runtime, Kubernetes, the inference engine, the orchestrator and the gateway in the middle, and the agent
application at the top. It is for engineers who build or run LLM systems and want to explain each design
choice with numbers. After working through a layer you can predict how it behaves, run it yourself on a
laptop or a free GPU, and walk a colleague through the design in a review.

**267 notebooks**, every one runnable in Google Colab with one click.

## Start here

1. **Pick a layer** from the [stack map](#the-stack) below — start where your question lives, or at the bottom
   and work up.
2. **Read its primer.** Each topic has a `PRIMER.md` (or a `*-primer.md`) that explains the concepts with
   worked numbers; the layer's `README.md` links it and says which sections to read first.
3. **Run its core notebooks**, in Colab (the badges in each layer README) or locally (see [Run it](#run-it)).
   Each exercise has a check cell that prints ✅ when your answer is right.

**Learning path:** [`CURRICULUM.md`](CURRICULUM.md) gives the order to work the whole repo, module by module, with
hours, tiers, shorter routes and cross-layer design drills.

**Compute and cost:** [`COMPUTE.md`](COMPUTE.md) says where each tier runs — laptop, free Colab or Kaggle GPUs,
rented GPUs, Google Cloud — what it costs, and which lab needs which tier.

## The stack

Read bottom-up: each layer is built on the one below it.

```
  ┌─ 07-application-agent-framework   build the agent: loop, tools, state, durability, evals, RAG
  │  06-gateway                       decide who may run what: identity, policy, rate limits, admission, cost
  │  05-orchestrator                  run many engine replicas as one service: routing, autoscaling, P/D split
  │  04-inference-engine              run one model fast: attention kernels, the KV cache, batching
  │  03-kubernetes-gpu                make GPUs schedulable: device plugin, scheduler, gangs, quotas, capacity
  │  02-cuda-nccl-runtime             get from a container to a GPU: driver, CUDA, NCCL, container runtime
  └─ 01-hardware-gpu-fabric           read the machine: spec sheets, the roofline, fabrics, cost of a token
     00-foundations                   the model itself: transformer internals, capacity math, model landscape
```

## What is inside

| Layer | Topics | What you will be able to do | Where |
|---|---|---|---|
| **00** foundations | transformer internals; GPU capacity planning; the open-weight model landscape | build attention, a transformer block and a tiny GPT; size memory and bandwidth for a model and budget TTFT/TPOT; compare open-weight models on cost and routing | [`00-foundations/`](00-foundations/README.md): `transformers/`, `gpu-capacity-planning/`, `model-landscape/` |
| **01** hardware and fabric | why a GPU is shaped the way it is; scale-up vs scale-out fabrics; rooflines, fabrics and the cost of a token | read a GPU spec sheet and predict whether an LLM step is compute- or memory-bound; price a collective on NVLink vs InfiniBand; budget a cold start and a failure rate; compute $/M tokens — then measure your own machine (CPU, a free T4, or a Spot L4 on Google Cloud) | [`01-hardware-gpu-fabric/`](01-hardware-gpu-fabric/README.md): `gpu-primer/`, `gpu-deployment/`, [`roofline-and-fabric/`](01-hardware-gpu-fabric/roofline-and-fabric/README.md) |
| **02** CUDA, NCCL, runtime | why a kernel is fast or slow; collectives and NCCL; how a container gets a GPU; sharing and monitoring a GPU | predict memory coalescing, occupancy and what an all-reduce costs with numpy simulators; run Numba CUDA kernels in the simulator, then on a GPU; measure collectives like nccl-tests with an α-β fit; see what a container sees of its GPU; choose MIG, MPS or time-slicing and alert on DCGM and XIDs — on a laptop, a free Kaggle 2×T4, any GPU box, or GKE | [`02-cuda-nccl-runtime/`](02-cuda-nccl-runtime/README.md): [`cuda-and-nccl/`](02-cuda-nccl-runtime/cuda-and-nccl/README.md) |
| **03** Kubernetes for GPUs | device plugin and DRA; the scheduling cycle and GPU fragmentation; gangs and topology-aware placement; Kueue quotas; getting capacity; sharing | explain why a GPU pod is Pending and fix it; place gangs without deadlock; set up Kueue quotas with borrowing and reclaim; choose Spot, flex-start or reservations — on a laptop simulator, a kind cluster with fake GPUs, one GPU VM, or GKE | [`03-kubernetes-gpu/`](03-kubernetes-gpu/README.md): [`gpu-scheduling/`](03-kubernetes-gpu/gpu-scheduling/README.md) |
| **04** inference engine | attention kernels, KV cache, paging; the engine itself — continuous batching, chunked prefill, prefix caching, sampling, speculation, quantization; vLLM in its source | implement FlashAttention and paged attention in miniature and defend a kernel choice with byte counts; build an engine's step loop, scheduler and prefix cache in numpy; size a model's KV cache before paying for a GPU; measure and tune a real vLLM against an SLO (a fake vLLM on a laptop, a free T4, Cloud Run GPU or GKE); follow a request through vLLM's source | [`04-inference-engine/`](04-inference-engine/README.md): `kv-cache/`, `paged-attention/`, `flash-attention/` (primer and deep dive), [`serving-engine/`](04-inference-engine/serving-engine/README.md), [`vllm-internals/`](04-inference-engine/vllm-internals/README.md) |
| **05** orchestrator | replica routing, flow control, autoscaling, prefill/decode disaggregation, KV-cache tiers | route on prefix affinity and load like the llm-d endpoint picker; autoscale on the right signal; size a prefill/decode split — in a fleet simulator, then as a real router on kind or GKE Inference Gateway | [`05-orchestrator/`](05-orchestrator/README.md): [`serving-orchestration/`](05-orchestrator/serving-orchestration/README.md) |
| **06** gateway | identity and security for agents; scaling, admission control and cost | give agents SPIFFE identities, exchange tokens and enforce policy (including MCP authorization and A2A) with an audit trail; plan capacity, find the provisioned-throughput break-even, and add admission control | [`06-gateway/`](06-gateway/README.md): `identity-security/`, `scaling-admission-cost/` |
| **07** agent application | the agent loop, tools, state, multi-agent, durable execution, evals, retrieval | write an agent loop from scratch, then build multi-agent workflows with sessions, MCP, OAuth, evals and tracing; make long-running agents durable; build RAG and vector indexes (IVF, PQ, HNSW, GraphRAG) from scratch | [`07-application-agent-framework/`](07-application-agent-framework/README.md): `agent-fundamentals/`, `long-running-durable/`, `retrieval-rag/` |

Several topics are worked on more than one provider (Google Cloud, Mistral) so the same concept can be compared
across stacks. Each layer `README.md` has the full scope, the current contents and the Colab links.

## How the labs work

Most topics follow the same pattern, so once you have done one you know how to do the rest.

| Piece | What it is |
|---|---|
| **Primer** | the concepts, with worked numbers; read it first, one section at a time |
| **Core** | a minimal implementation, usually standard-library Python, plus fill-in notebooks that *predict* what the real system does |
| **Lab** | the detailed version: real tools, benchmarks and deploy recipes that *run* or *measure* the same ideas |

**Tiers** say what hardware a notebook or recipe needs, and the topics built for layers 01–05 mark every step with
one ([`COMPUTE.md`](COMPUTE.md) has the details and prices):

- **T0** — a laptop, Colab CPU or CI. Free. Every concept is learnable here.
- **T1** — one small GPU: a free Colab or Kaggle T4, or a rented card.
- **T2** — a multi-GPU box, rented for an hour (or Kaggle's free 2×T4).
- **T3** — the Google Cloud deployment via Terraform, with the cheapest defaults. Optional; never a prerequisite.

Notebooks that need a GPU detect what they have and fall back to a clearly labelled T0 path.

**Exercises** sit in `notebooks/` (or `exercises/`) with `# YOUR CODE HERE` and a check cell that prints ✅ when
your answer is right; worked answers are in `solutions/`. Exercises are committed blank, so
`git restore <notebook>` returns one to its starting state (for labs built from `notebooks_src/`, re-run that
lab's `python3 tools/build_notebooks.py`).

## Run it

**In Colab:** open any notebook from the badge in its layer README. There is nothing to set up: the first cell
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
├── raw/            inbox for new material (gitignored except its README); sorted into a layer, never copied
├── tools/          inject_colab_bootstrap.py, gen_colab_index.py; orchestration/ (build spec, facts, status, README style guide)
├── CURRICULUM.md   the learning path: modules, order, hours, tiers, design drills
├── COMPUTE.md      where to run each tier, what it costs, which lab needs which tier
├── COLAB.md        running notebooks in Colab
├── CLAUDE.md       how the repo is organised and how new content gets sorted
└── README.md       this file
```

## Adding material

Drop it in [`raw/`](raw/) and follow [`raw/README.md`](raw/README.md). The sorting rules (which layer, which topic
folder) and the decisions log are in [`CLAUDE.md`](CLAUDE.md); READMEs follow
[`tools/orchestration/README-STYLE.md`](tools/orchestration/README-STYLE.md).
