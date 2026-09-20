# full-stack-agentic-engineer

A hands-on learning mono-repo that mirrors the **LLM inference / serving stack**, from GPUs at the
bottom (`01`) to the agent application at the top (`07`). Each layer holds self-contained labs —
runnable Python, worked notebooks, and fill-in-the-blank exercises.

- **Run any notebook in Google Colab:** each layer's `README.md` has "Open in Colab" links for its notebooks. One-time setup (connect Colab to GitHub + add a `GH_TOKEN` secret) is in [`COLAB.md`](./COLAB.md); then every notebook's first cell clones the repo and installs that lab's dependencies.
- **Sorting workflow & layer definitions:** [`CLAUDE.md`](./CLAUDE.md).
- **Adding new content:** drop it in [`raw/`](./raw/) and follow [`raw/README.md`](./raw/README.md).

## Layout

```
.
├── 00-foundations/            transformer internals · GPU capacity · model landscape
├── 01-hardware-gpu-fabric/     GPUs, NVLink, NICs, storage, cooling            gpu-primer, gpu-deployment
├── 02-cuda-nccl-runtime/       container runtime, NCCL, CUDA, driver           (awaiting content)
├── 03-kubernetes-gpu/          Kubernetes + GPU Operator + scheduler           (awaiting content)
├── 04-inference-engine/        vLLM / SGLang / TensorRT-LLM                    flash/paged attn, kv-cache
├── 05-orchestrator/            Dynamo, llm-d, Ray Serve                        (awaiting content)
├── 06-gateway/                 auth, rate limits, routing, quotas, cost
│   ├── identity-security/          agentic-identity-core, agentic-identity-gcp-lab
│   └── scaling-admission-cost/     agentic-scaling-lab
├── 07-application-agent-framework/
│   ├── agent-fundamentals/         agent-core, gcp-agent-platform-lab
│   ├── long-running-durable/       long-running-agents-core, lra-core, long-running-agentic, lra
│   └── retrieval-rag/              vector_stores
├── raw/                        inbox — drop new content here (gitignored except its README)
├── tools/                      inject_colab_bootstrap.py, gen_colab_index.py
├── COLAB.md                    one-time Colab setup (per-notebook links live in each layer README)
├── CLAUDE.md                   how the repo is organized and how to sort new content
└── README.md                   this file
```

## Working with a lab

Each lab is independent and carries its own `requirements.txt` / `pyproject.toml`.

```bash
cd 07-application-agent-framework/agent-fundamentals/agent-core
pip install -e .          # or: pip install -r requirements.txt
# open notebooks/ locally, or use the Colab links in this layer's README
```

Exercise notebooks live in `notebooks/` (blank, with `# YOUR CODE HERE`); worked answers are in
`solutions/`. To retry an exercise after editing it: `git restore <notebook>`.

Layers 02, 03 and 05 are scaffolded and awaiting content; `00-foundations/` holds cross-cutting model-level material.
