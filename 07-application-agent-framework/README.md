# 07 · Application / agent framework

The top of the stack: the agent itself — its loop, tools, memory, durability, the retrieval it
stands on, and the sandbox its code-running tools execute in. What the end user interacts with.

**Covers:** the agent loop, tool calling, state/sessions/checkpoints, context engineering &
caching, multi-agent workflows, durable/long-running execution, evals, retrieval-augmented
generation and vector search, sandboxed execution of model-generated code (the isolation ladder,
the execution contract, egress and secrets, sandboxes on Kubernetes, warm pools and cost per
action), capstone applications.

**Signal keywords:** agent loop, tool calling, ADK, LangGraph, multi-agent, workflow, durable
execution, long-running, checkpoint, saga, human-in-the-loop, RAG, retrieval, vector store,
embeddings, eval, trajectory, sandbox, code execution, `run_code`, gVisor, Firecracker, Kata,
RuntimeClass, Pod Security, NetworkPolicy, egress proxy, warm pool.

## Current contents

### `agent-fundamentals/`
- **`agent-core/`** — the minimal agent core: loop, tools, state & control, a mini support agent.
- **`gcp-agent-platform-lab/`** — the full agent-platform lab (workflows, multi-agent, state,
  context engineering, evals, reliability, resource estimation, capstone). Cross-refs **06-gateway**.
- **`mistral-agent-core/`** — Mistral variant of the agent core (`agentcore` + a `mistral_llm`
  adapter and a "going live on Mistral" lesson).

### [`sandboxed-execution/`](sandboxed-execution/README.md)
Run the model's code without handing it your keys: say what a `run_code` tool can reach when the model
is hijacked, which control bounds each risk, how much isolation you need and what a pool of sandboxes
costs — then render and check the Kubernetes that enforces it. Builds on `agent-core`'s tool contract
(07.1) and the identity primer's §6.2 (code execution); module 07.5 in [`CURRICULUM.md`](../CURRICULUM.md).
*T0 = laptop or Colab CPU, free; T0 + Docker = the container rungs and kind on your own machine; T3 = the
Google Cloud deployment, optional. No GPU anywhere.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](sandboxed-execution/PRIMER.md) | explain the threat model, the isolation ladder (process → container → gVisor → microVM), the execution contract, egress and secrets, sandboxes on Kubernetes, latency and cost per action, and audit — then walk the design in a review | read with the core | — |
| [`sandbox-core/`](sandboxed-execution/sandbox-core/README.md) | build every part of a safe `run_code` in standard-library Python: attack probes, a process sandbox with rlimits, a wall-clock kill and its own UID, the execution contract, an allowlisting egress proxy that injects a credential the code never holds, policy rendered to Kubernetes YAML, and warm-pool sizing with Erlang C; 5 notebooks | ~4 h with the primer | T0 |
| [`sandbox-lab/`](sandboxed-execution/sandbox-lab/README.md) | the same controls on a network namespace, a hardened Docker container, pod-per-execution on kind and a GKE Sandbox (gVisor) node pool, with an agent whose `run_code`/`fetch_url` tools fail closed under prompt injection; 5 notebooks | ~8 h | T0, T0 + Docker, T3 |

```bash
cd sandboxed-execution/sandbox-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q   # 81 tests, ~30 s
cd ../sandbox-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q                     # 112 tests, ~25 s, offline
python3 -m sandboxlab env      # which isolation levels this machine can actually run
```

### `long-running-durable/`
Durable, long-running agentic workflows — several fidelities kept side by side (not merged).
- **`long-running-agents-core/`**, **`lra-core/`** — minimal durable cores.
- **`long-running-agentic/`**, **`lra/`** — full GCP labs (Cloud Run, Tasks, Pub/Sub, Workflows, Firestore, ADK 2).
- **`long-running-agents-mistral/`** — Mistral variant (durable core + a local-Temporal workflow + Mistral model adapter).
- **`00_primer.md`** — the shared primer (identical to the copy inside `long-running-agentic/`).

### `retrieval-rag/`
- **`vector_stores/`** — from-scratch retrieval: `minifaiss` (IVF, PQ, HNSW, LSH, k-means in numpy),
  `minigraphrag` (GraphRAG), a homemade TF-IDF, Gutenberg demos.
- **`embeddings-lab/`** — embeddings from scratch in NumPy: counts→vectors, contrastive bi-encoder,
  geometry, vector search, a retrieval pipeline, superposition (worked notebooks + exercises + solutions).
- **`rag-from-scratch/`** — a RAG pipeline built from scratch.
- **`vector-databases-primer.md`** — first-principles-to-production primer on vector databases.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`agent-fundamentals/agent-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/notebooks/01_the_agent_loop.ipynb) `01_the_agent_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/notebooks/02_tools.ipynb) `02_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/notebooks/03_state_and_control.ipynb) `03_state_and_control.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/notebooks/04_mini_support_agent.ipynb) `04_mini_support_agent.ipynb`

**`agent-fundamentals/agent-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/solutions/01_the_agent_loop.ipynb) `01_the_agent_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/solutions/02_tools.ipynb) `02_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/solutions/03_state_and_control.ipynb) `03_state_and_control.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/agent-core/solutions/04_mini_support_agent.ipynb) `04_mini_support_agent.ipynb`

**`agent-fundamentals/gcp-agent-platform-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/00_setup_and_fake_llm.ipynb) `00_setup_and_fake_llm.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/01_agent_loop_and_tools.ipynb) `01_agent_loop_and_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/02_workflows_and_multi_agent.ipynb) `02_workflows_and_multi_agent.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/03_state_sessions_checkpoints.ipynb) `03_state_sessions_checkpoints.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/04_context_engineering_and_caching.ipynb) `04_context_engineering_and_caching.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/05_mcp_server_client_gateway.ipynb) `05_mcp_server_client_gateway.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/06_oauth_identity_propagation.ipynb) `06_oauth_identity_propagation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/07_agent_api_streaming_tasks.ipynb) `07_agent_api_streaming_tasks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/08_evals_trajectory_judge_gates.ipynb) `08_evals_trajectory_judge_gates.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/09_tracing_and_metrics.ipynb) `09_tracing_and_metrics.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/10_reliability_retries_breakers.ipynb) `10_reliability_retries_breakers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/11_security_prompt_injection.ipynb) `11_security_prompt_injection.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/12_resource_estimation.ipynb) `12_resource_estimation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/13_code_review_exercises.ipynb) `13_code_review_exercises.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/14_capstone_bank_agent.ipynb) `14_capstone_bank_agent.ipynb`

**`agent-fundamentals/gcp-agent-platform-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/00_setup_and_fake_llm.ipynb) `00_setup_and_fake_llm.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/01_agent_loop_and_tools.ipynb) `01_agent_loop_and_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/02_workflows_and_multi_agent.ipynb) `02_workflows_and_multi_agent.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/03_state_sessions_checkpoints.ipynb) `03_state_sessions_checkpoints.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/04_context_engineering_and_caching.ipynb) `04_context_engineering_and_caching.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/05_mcp_server_client_gateway.ipynb) `05_mcp_server_client_gateway.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/06_oauth_identity_propagation.ipynb) `06_oauth_identity_propagation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/07_agent_api_streaming_tasks.ipynb) `07_agent_api_streaming_tasks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/08_evals_trajectory_judge_gates.ipynb) `08_evals_trajectory_judge_gates.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/09_tracing_and_metrics.ipynb) `09_tracing_and_metrics.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/10_reliability_retries_breakers.ipynb) `10_reliability_retries_breakers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/11_security_prompt_injection.ipynb) `11_security_prompt_injection.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/12_resource_estimation.ipynb) `12_resource_estimation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/13_code_review_exercises.ipynb) `13_code_review_exercises.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/solutions/14_capstone_bank_agent.ipynb) `14_capstone_bank_agent.ipynb`

**`agent-fundamentals/mistral-agent-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/notebooks/01_the_agent_loop.ipynb) `01_the_agent_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/notebooks/02_tools.ipynb) `02_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/notebooks/03_state_and_control.ipynb) `03_state_and_control.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/notebooks/04_mini_support_agent.ipynb) `04_mini_support_agent.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/notebooks/05_going_live_on_mistral.ipynb) `05_going_live_on_mistral.ipynb`

**`agent-fundamentals/mistral-agent-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/solutions/01_the_agent_loop.ipynb) `01_the_agent_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/solutions/02_tools.ipynb) `02_tools.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/solutions/03_state_and_control.ipynb) `03_state_and_control.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/solutions/04_mini_support_agent.ipynb) `04_mini_support_agent.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/agent-fundamentals/mistral-agent-core/solutions/05_going_live_on_mistral.ipynb) `05_going_live_on_mistral.ipynb`

**`long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/01_durable_loop_practice.ipynb) `01_durable_loop_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/01_durable_loop_worked.ipynb) `01_durable_loop_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/02_fan_out_fan_in_practice.ipynb) `02_fan_out_fan_in_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/02_fan_out_fan_in_worked.ipynb) `02_fan_out_fan_in_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/03_hitl_saga_practice.ipynb) `03_hitl_saga_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/03_hitl_saga_scheduled_worked.ipynb) `03_hitl_saga_scheduled_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/04_adk_workflow_practice.ipynb) `04_adk_workflow_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks/04_adk_workflow_worked.ipynb) `04_adk_workflow_worked.ipynb`

**`long-running-durable/long-running-agents-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agents-core/notebooks/01_worked.ipynb) `01_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agents-core/notebooks/02_practice.ipynb) `02_practice.ipynb`

**`long-running-durable/long-running-agents-mistral/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agents-mistral/notebooks/01_worked.ipynb) `01_worked.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/long-running-agents-mistral/notebooks/02_practice.ipynb) `02_practice.ipynb`

**`long-running-durable/lra-core/lra-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/01_durable_loop.ipynb) `01_durable_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/02_wait_and_resume.ipynb) `02_wait_and_resume.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/03_retry_and_reaper.ipynb) `03_retry_and_reaper.ipynb`

**`long-running-durable/lra-core/lra-core/notebooks/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/solutions/01_durable_loop.ipynb) `01_durable_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/solutions/02_wait_and_resume.ipynb) `02_wait_and_resume.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra-core/lra-core/notebooks/solutions/03_retry_and_reaper.ipynb) `03_retry_and_reaper.ipynb`

**`long-running-durable/lra/lra-gcp/notebooks/practice/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/practice/00_core_idea.ipynb) `00_core_idea.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/practice/01_durable_execution.ipynb) `01_durable_execution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/practice/02_human_in_the_loop.ipynb) `02_human_in_the_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/practice/03_fanout_saga_reflection.ipynb) `03_fanout_saga_reflection.ipynb`

**`long-running-durable/lra/lra-gcp/notebooks/worked/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/worked/00_core_idea.ipynb) `00_core_idea.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/worked/01_durable_execution.ipynb) `01_durable_execution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/worked/02_human_in_the_loop.ipynb) `02_human_in_the_loop.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/worked/03_fanout_saga_reflection.ipynb) `03_fanout_saga_reflection.ipynb`

**`retrieval-rag/embeddings-lab/exercises/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex01.ipynb) `ex01.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex02.ipynb) `ex02.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex03.ipynb) `ex03.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex04.ipynb) `ex04.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex05.ipynb) `ex05.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/exercises/ex06.ipynb) `ex06.ipynb`

**`retrieval-rag/embeddings-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/01_counts_to_vectors.ipynb) `01_counts_to_vectors.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/02_contrastive_bi_encoder.ipynb) `02_contrastive_bi_encoder.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/03_geometry.ipynb) `03_geometry.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/04_vector_search.ipynb) `04_vector_search.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/05_retrieval_pipeline.ipynb) `05_retrieval_pipeline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/notebooks/06_superposition.ipynb) `06_superposition.ipynb`

**`retrieval-rag/embeddings-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex01_solutions.ipynb) `ex01_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex02_solutions.ipynb) `ex02_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex03_solutions.ipynb) `ex03_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex04_solutions.ipynb) `ex04_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex05_solutions.ipynb) `ex05_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/embeddings-lab/solutions/ex06_solutions.ipynb) `ex06_solutions.ipynb`

**`retrieval-rag/rag-from-scratch/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/00_setup_and_corpus.ipynb) `00_setup_and_corpus.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/01_minimal_rag.ipynb) `01_minimal_rag.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/02_chunking.ipynb) `02_chunking.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/03_hybrid_search.ipynb) `03_hybrid_search.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/04_reranking.ipynb) `04_reranking.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/05_evaluation.ipynb) `05_evaluation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/notebooks/06_iterative_rag.ipynb) `06_iterative_rag.ipynb`

**`retrieval-rag/rag-from-scratch/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/00_setup_and_corpus.ipynb) `00_setup_and_corpus.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/01_minimal_rag.ipynb) `01_minimal_rag.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/02_chunking.ipynb) `02_chunking.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/03_hybrid_search.ipynb) `03_hybrid_search.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/04_reranking.ipynb) `04_reranking.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/05_evaluation.ipynb) `05_evaluation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/retrieval-rag/rag-from-scratch/solutions/06_iterative_rag.ipynb) `06_iterative_rag.ipynb`

**`sandboxed-execution/sandbox-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/01_the_threat_model.ipynb) `01_the_threat_model.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/02_a_process_sandbox.ipynb) `02_a_process_sandbox.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/03_the_execution_contract_and_policies.ipynb) `03_the_execution_contract_and_policies.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/04_egress_and_secrets.ipynb) `04_egress_and_secrets.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/05_pools_latency_and_cost.ipynb) `05_pools_latency_and_cost.ipynb`

**`sandboxed-execution/sandbox-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/solutions/01_the_threat_model.ipynb) `01_the_threat_model.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/solutions/02_a_process_sandbox.ipynb) `02_a_process_sandbox.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/solutions/03_the_execution_contract_and_policies.ipynb) `03_the_execution_contract_and_policies.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/solutions/04_egress_and_secrets.ipynb) `04_egress_and_secrets.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-core/solutions/05_pools_latency_and_cost.ipynb) `05_pools_latency_and_cost.ipynb`

**`sandboxed-execution/sandbox-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/01_hardened_containers.ipynb) `01_hardened_containers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/02_pod_per_execution_on_kind.ipynb) `02_pod_per_execution_on_kind.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/03_egress_proxy_and_secret_brokering.ipynb) `03_egress_proxy_and_secret_brokering.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/04_an_agent_with_a_sandbox_tool.ipynb) `04_an_agent_with_a_sandbox_tool.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/05_gke_sandbox_with_gvisor.ipynb) `05_gke_sandbox_with_gvisor.ipynb`

**`sandboxed-execution/sandbox-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/solutions/01_hardened_containers.ipynb) `01_hardened_containers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/solutions/02_pod_per_execution_on_kind.ipynb) `02_pod_per_execution_on_kind.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/solutions/03_egress_proxy_and_secret_brokering.ipynb) `03_egress_proxy_and_secret_brokering.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/solutions/04_an_agent_with_a_sandbox_tool.ipynb) `04_an_agent_with_a_sandbox_tool.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/07-application-agent-framework/sandboxed-execution/sandbox-lab/solutions/05_gke_sandbox_with_gvisor.ipynb) `05_gke_sandbox_with_gvisor.ipynb`
<!-- colab-links:end -->
