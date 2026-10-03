# Agent Platform Lab (Google Cloud)

This lab is a small, readable agent runtime and fifteen practice notebooks. The notebooks turn every topic
of the design and the review of an agent on Google Cloud into code that you write yourself. Everything runs
**offline**. A scripted `FakeLLM` takes the place of Gemini. Thus you practise the mechanisms (tool contracts,
loops, state, identity, MCP, evaluation, tracing, reliability, cost), not the infrastructure code of an API.
When you have a key, one import puts the real model in place of `FakeLLM`.

    83 exercises · 15 notebooks · 6.5k lines of library · 149 unit tests · every solution notebook executes clean

**Time and tier:** ~20 h. This is module 07.2 in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 is a laptop or a Colab CPU, at no cost. At T0, a scripted `FakeLLM` takes the place of the model, and you need no GPU and no cloud project. A Gemini key is optional. With the key, the real model takes the place of `FakeLLM`.

## Quick start

```bash
git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
make setup            # pip install -e ".[dev]" + a Jupyter kernel
make lab              # opens JupyterLab in notebooks/
```

If you do not have `make`, run `pip install -e ".[dev]"`. Then run `python -m jupyterlab notebooks`.
The notebooks also run when you do not install the package. A bootstrap cell puts the repo on `sys.path`.

## How to work through it

Open `notebooks/NN_*.ipynb` in sequence. Each notebook is a short course on one topic. It starts with
worked examples that you run and read. Then come **Exercise** cells with `# YOUR CODE HERE`, which you
fill in. A **Check** cell comes after each Exercise cell. The Check cell fails loudly until your solution
is correct, and it prints ✅ when your solution is correct.

The fully solved version of every notebook is in `solutions/`. Try the exercise first, then compare. Each
notebook ends with *The one-minute version*: how to explain that topic in a design review.

| # | Notebook | Deeper in this repo | What you build |
|---|----------|--------|----------------|
| 00 | Setup and the fake model | — | Control a function-calling model in three ways. Look at caching and usage. |
| 01 | Agent loop and tools | [sandbox](../../sandboxed-execution/PRIMER.md) §3 | Tool contracts from signatures, structured errors, idempotency, the loop with budgets and parallel calls |
| 02 | Workflows and multi-agent | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1.7 | Sequential/parallel/loop agents, delegation, when one more agent is worth its cost |
| 03 | State, sessions, checkpoints | [durable](../../long-running-durable/PRIMER.md) §2–§3 | The event log and the working state, compare-and-set, durable tasks that resume after a crash, pause/approve |
| 04 | Context engineering and caching | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.5 | Cache-friendly layout, compaction, the shape of tool results, token budgets |
| 05 | MCP server, client, gateway | [MCP revisions](docs/MCP_REVISIONS.md), [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §7 | A teaching subset of MCP (2026-07-28 shape): stateless requests, mirrored headers, MRTR, Tasks. Also a policy gateway. |
| 06 | OAuth and identity propagation | [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §3.5, §7.1 | PKCE, resource indicators, audience-bound tokens, step-up, token exchange, the confused deputy |
| 07 | The agent's own API | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.3, §5.7 | Sessions, SSE-style event streams, 202 + task handles, idempotency keys, rate limits |
| 08 | Evals: trajectory, judge, gates | — | Golden sets, trajectory metrics, judge calibration (kappa), release gates with confidence intervals |
| 09 | Tracing and metrics | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.10 | Spans with `gen_ai.*` attributes, cost per conversation, p95 for each step, TTFT and tokens/s, alerts |
| 10 | Reliability | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.2 | Backoff with jitter, idempotent retries, circuit breakers, bulkheads, deadlines, fallbacks |
| 11 | Security and prompt injection | [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §6, [sandbox](../../sandboxed-execution/PRIMER.md) §1 | An indirect injection demo, provenance-tagged data blocks, screening, allowlists, redaction |
| 12 | Resource estimation | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §3 | Calculate the worked numbers again: cost, peak TPM, Little's law, latency waterfall, vectors. |
| 13 | Code evaluation | [vectors](../../retrieval-rag/vector-databases-primer.md) §12 | Two full programs with bugs, and six drills. Their tests pass only when your repairs are correct. |
| 14 | Capstone: the bank agent | all of the above | All the parts, put together along the spine, with a gate, traces and a production cost estimate |

A reasonable pace is one or two notebooks each evening. Give notebooks 13 and 14 a full session each.
For each concept, [`docs/PRIMER_MAP.md`](docs/PRIMER_MAP.md) gives its notebooks, modules and key exercises.
It also gives the primers in other parts of this repo that go deeper.

## The library, in one screen

```
agentlab/
  llm/            FakeLLM (scripted / policy / KeywordPlanner), message + usage types, optional GeminiLLM
  agents/         tools (schema from signatures, side-effect classes, idempotency), Budget, LlmAgent loop,
                  Sequential/Parallel/Loop agents, AgentTool delegation, Session/Event log + stores,
                  ContextBuilder, Runner with pause/approve
  mcp/            McpServer, transports (in-process, local HTTP), McpClient + McpToolset, Gateway with policy
  auth/           toy OAuth 2.1 authorization server: PKCE, RFC 8707 resource, RFC 8693 exchange, RFC 9207 iss
  evals/          GoldenSet, trajectory metrics, RubricJudge + calibration, run_eval + Gate, SafetySuite
  observability/  Tracer (contextvar nesting), PriceTable, TraceSummary, percentiles, TTFT/tps, alerts
  reliability/    retry/backoff, IdempotentCall, CircuitBreaker, Bulkhead, Deadline, FallbackChain, GracefulTool
  security/       DataBlock provenance, screen(), redact, ActionPolicy + GuardedTool, indirect_injection_demo
  estimation/     Scenario cost/throughput/concurrency, latency_budget, vector sizing, CI, compounded reliability
```

The names agree with the names of ADK, by intention (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`,
`FunctionTool`, `AgentTool`, `Runner`, session/memory services, callbacks-as-hooks). Thus, what you learn here
also applies to ADK. `docs/LAB_TO_ADK.md` gives the mapping. The MCP and OAuth modules have the label "teaching
subsets". They are faithful to the concepts and to the shape of the current spec. They are not conformant
implementations.

## Verifying the whole thing

```bash
make test        # unit tests (~3 s)
make notebooks   # regenerate notebooks/ and solutions/ from notebooks_src/
make check       # rebuild, unit tests, and execute every solution notebook end to end (~40 s)
python tools/run_notebooks.py notebooks --expect-fail   # exercise variants stop at the first unsolved cell
```

The builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent-format sources with
`### BEGIN SOLUTION` blocks). Edit the sources, not the notebooks. See `notebooks_src/README.md`.

## Using a real model

Run `pip install -e ".[gemini]"`. Set `GOOGLE_API_KEY` (or the Vertex environment variables). Then, in any
notebook, replace `FakeLLM(...)` with `GeminiLLM(model="gemini-3-flash")` (the model string as of
September 2026, verify). The author wrote the adapter against the `google-genai` 1.x surface. The SDK is
on 2.x since May 2026. Thus, *make sure that the adapter works before you trust it*.

The details are in [`docs/GEMINI_ADAPTER.md`](docs/GEMINI_ADAPTER.md). The mapping to Google's Agent
Development Kit (2.x on PyPI since May 2026, verify) is in [`docs/LAB_TO_ADK.md`](docs/LAB_TO_ADK.md).

## Honesty notes

- The prices and model names in `estimation/` and `observability/` are illustrative. Their date is
  September 2026. Before you quote them, compare them with the official pricing page.
- The `FakeLLM` is not intelligent, by intention. The lab teaches the harness: validation, budgets,
  identity, evaluation and tracing. Agent systems in production succeed or fail in the harness. The
  behaviour of a real model (and its real susceptibility to prompt injection) is out of scope.
- The MCP of this lab has the shape of the 2026-07-28 revision. That is, it has stateless per-request
  metadata, mirrored headers, embedded server-to-client interactions and the Tasks extension. The lab
  compared this shape with the spec repository on 2026-09-26 (verify). Many deployed servers still use
  the 2025 revisions. [`docs/MCP_REVISIONS.md`](docs/MCP_REVISIONS.md) tells what is different from
  2025-03-26, 2025-06-18 and 2025-11-25. It also tells where the subset of the lab is different from the
  spec.

## Layout

```
agentlab/           library            tests/          pytest (unit + slow notebook execution)
notebooks_src/    notebook sources   tools/          build_notebooks.py, run_notebooks.py
notebooks/        exercises          solutions/      solved notebooks
docs/             concept map, MCP revisions, ADK mapping, Gemini adapter, contributing
```

MIT licensed.
