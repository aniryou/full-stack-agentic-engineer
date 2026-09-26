# full-stack-agentic-engineer

A hands-on knowledge base for the **LLM inference / serving stack**, organised bottom-up: from the
GPUs and fabric at the bottom (`01`) to the agent application at the top (`07`), with the model-level
foundations underneath (`00`). Each layer holds self-contained labs: a primer, runnable Python,
worked notebooks, and fill-in-the-blank exercises with solutions. Several topics are worked on more
than one vendor stack (Google Cloud, Mistral) so the same concepts can be compared across providers.

**227 notebooks**, all runnable in Google Colab with one click.

## How to use it

- **In Colab:** every layer `README.md` lists its notebooks with an "Open in Colab" badge. Nothing to
  set up: the notebook's first cell clones this repo and installs that lab's dependencies.
  [`COLAB.md`](./COLAB.md) has the details.
- **Locally:** each lab is independent and carries its own `requirements.txt` / `pyproject.toml`.
  ```bash
  cd 07-application-agent-framework/agent-fundamentals/agent-core
  pip install -e .          # or: pip install -r requirements.txt
  python3 -m pytest -q      # most labs ship tests
  # open notebooks/ in Jupyter, or use the Colab links in this layer's README
  ```
- **Exercises** live in `notebooks/` (or `exercises/`) with `# YOUR CODE HERE` and a check cell that
  prints ✅ when your answer is right; worked answers are in `solutions/`. To retry an exercise after
  editing it: `git restore <notebook>`.
- **Primers** are the `*.md` documents next to each lab. Read the primer first, then work
  the notebooks against the code.

## The stack

```
  ┌─ 07-application-agent-framework   the agent: loop, tools, state, durability, evals, RAG
  │  06-gateway                       identity, policy, rate limits, admission, cost, observability
  │  05-orchestrator                  Dynamo, llm-d, Ray Serve (routing, autoscaling, disaggregation)
  │  04-inference-engine              vLLM / SGLang / TensorRT-LLM: attention kernels, KV cache, batching
  │  03-kubernetes-gpu                Kubernetes + GPU Operator + scheduling
  │  02-cuda-nccl-runtime             container runtime, NCCL, CUDA, driver
  └─ 01-hardware-gpu-fabric           GPUs, NVLink, NICs, storage, cooling
     00-foundations                   transformer internals, GPU capacity planning, model landscape
```

## What is inside

| Layer | Topics | Labs |
|---|---|---|
| **`00-foundations/`** | Transformer internals, GPU capacity planning, the open-weight model landscape | `transformers/` (attention → block → tiny GPT, primer + practice), `gpu-capacity-planning/` (memory and bandwidth constraints, TTFT/TPOT, sizing formulas), `model-landscape/` (open-weight LLM primer, Mistral cost/routing exercises) |
| **`01-hardware-gpu-fabric/`** | Why a GPU is shaped the way it is; scale-up vs scale-out fabrics; rooflines, fabrics and the cost of a token | `gpu-primer/`, `gpu-deployment/` (NVLink vs InfiniBand/RoCE and what it means for serving). `roofline-and-fabric/`: a primer (spec sheets, the roofline, LLM inference on the roofline, the memory hierarchy, fabrics and collective cost, cold start, reliability, cost per token, the accelerator landscape), `roofline-core/` (the calculators, standard library only), `gpu-bench-lab/` (measure the machine you have — GEMM, memory bandwidth, transfers, P2P, weight loading — on a laptop CPU or a GPU; Docker recipes and a Spot L4 VM on Google Cloud via Terraform) |
| **`02-cuda-nccl-runtime/`** | Driver, CUDA, NCCL collectives, container runtime, MIG | _scaffolded, awaiting content_ |
| **`03-kubernetes-gpu/`** | GPU Operator, device plugins, gang and topology-aware scheduling | _scaffolded, awaiting content_ |
| **`04-inference-engine/`** | Attention kernels, KV cache and paging; the engine itself — continuous batching, chunked prefill, prefix caching, sampling, speculation, quantization; vLLM read in its source | `kv-cache/`, `paged-attention/`, `flash-attention/` (minimal implementations + practice notebooks; a FlashAttention deep dive with calculators, a companion notebook and a Triton forward pass). `serving-engine/`: a primer (the step loop, continuous batching, chunked prefill, KV management, prefix caching, sampling and structured output, speculative decoding, quantization, parallelism, multi-LoRA, measuring an engine), `mini-engine-core/` (a numpy "nano-vLLM" with a roofline simulator), `vllm-serving-lab/` (size a model before serving, a load generator, `/metrics`, knob sweeps against an SLO, a fake vLLM for laptops; any GPU box, Cloud Run GPU via Terraform, GKE). `vllm-internals/`: a source-level primer, a source map with a reading plan, and a notebook |
| **`05-orchestrator/`** | Replica routing, flow control, autoscaling, prefill/decode disaggregation, KV-cache tiers | `serving-orchestration/`: a primer (why round-robin fails for LLMs, routing signals and algorithms, the llm-d endpoint picker, flow control and priorities, the HPA algorithm and which signal to scale on, prefill/decode disaggregation, KV cache beyond HBM, multi-LoRA routing, the Kubernetes-native stack), `orchestrator-core/` (a discrete-event fleet simulator, standard library only), `inference-gateway-lab/` (a real async router that makes the endpoint picker's decisions in front of emulated or real vLLM, an HPA recommender; docker compose, kind with the llm-d Router, any GPU box, GKE Inference Gateway via Terraform) |
| **`06-gateway/`** | Identity & security for agents; scaling, admission control and cost | `identity-security/`: `agentic-identity-core/` (the idea in one file), `agentic-identity-gcp-lab/` (SPIFFE principals, token exchange, policy enforcement, MCP resource server, A2A, audit on Google Cloud), `agentic-identity-core-mistral/`. `scaling-admission-cost/`: `agentic-scaling-lab/` (capacity math, provisioned-throughput break-even, resilience, admission control, a Cloud Run + Gemini reference architecture), `agentic-scaling-lab-mistral/` (hosted vs self-hosted on vLLM) |
| **`07-application-agent-framework/`** | The agent loop, tools, state, multi-agent, durable execution, evals, retrieval | `agent-fundamentals/`: `agent-core/` (a fake model, a tool, and the loop in ~200 lines), `gcp-agent-platform-lab/` (workflows, multi-agent, sessions, context engineering, MCP, OAuth, evals, tracing, reliability, estimation, a capstone), `mistral-agent-core/`. `long-running-durable/`: durable cores (`long-running-agents-core/`, `lra-core/`), full Google Cloud labs (`long-running-agentic/`, `lra/`), `long-running-agents-mistral/`. `retrieval-rag/`: `embeddings-lab/`, `rag-from-scratch/`, `vector_stores/` (IVF, PQ, HNSW, GraphRAG from scratch), a vector-databases primer |

Each layer's `README.md` has the full scope, its current contents, and the Colab links.

## Layout

```
.
├── 00-foundations/ … 07-application-agent-framework/   one folder per layer; labs in topic sub-folders
├── raw/            inbox for new material (gitignored except its README); sorted into a layer, never copied
├── tools/          inject_colab_bootstrap.py, gen_colab_index.py; orchestration/ (build spec, facts, status)
├── COLAB.md        running notebooks in Colab
├── CLAUDE.md       how the repo is organised and how new content gets sorted
└── README.md       this file
```

Adding material: drop it in [`raw/`](./raw/) and follow [`raw/README.md`](./raw/README.md); the
sorting rules and the decisions log are in [`CLAUDE.md`](./CLAUDE.md).
