# Agent Platform Lab (Google Cloud)

A small, readable agent runtime plus fifteen practice notebooks that turn every topic of
designing and reviewing an agent on Google Cloud into code you write yourself. Everything runs **offline** — a scripted `FakeLLM` stands in for
Gemini — so you practise the mechanisms (tool contracts, loops, state, identity, MCP, evaluation,
tracing, reliability, cost) rather than API plumbing. When you have a key, one import swaps in the
real model.

    83 exercises · 15 notebooks · 6.5k lines of library · 147 unit tests · every solution notebook executes clean

## Quick start

```bash
git clone <this repo> && cd gcp-agent-platform-lab
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
make setup            # pip install -e ".[dev]" + a Jupyter kernel
make lab              # opens JupyterLab in notebooks/
```

No `make`? `pip install -e ".[dev]"` then `python -m jupyterlab notebooks`.
Notebooks also run without installing the package: a bootstrap cell puts the repo on `sys.path`.

## How to work through it

Open `notebooks/NN_*.ipynb` in order. Each notebook is a short course on one topic: worked examples
you run and read, then **Exercise** cells with `# YOUR CODE HERE` that you fill in, each followed by
a **Check** cell that fails loudly until your solution is right and prints ✅ when it is. The fully
solved version of every notebook is in `solutions/` — try the exercise first, then compare. Each
notebook ends with *The one-minute version*: how to explain that topic in a design review.

| # | Notebook | Deeper in this repo | What you build |
|---|----------|--------|----------------|
| 00 | Setup and the fake model | — | drive a function-calling model three ways; see caching and usage |
| 01 | Agent loop and tools | [sandbox](../../sandboxed-execution/PRIMER.md) §3 | tool contracts from signatures, structured errors, idempotency, the loop with budgets and parallel calls |
| 02 | Workflows and multi-agent | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1.7 | sequential/parallel/loop agents, delegation, when an extra agent earns its keep |
| 03 | State, sessions, checkpoints | [durable](../../long-running-durable/00_primer.md) §2–§3 | event log vs working state, compare-and-set, durable tasks that resume after a crash, pause/approve |
| 04 | Context engineering and caching | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.5 | cache-friendly layout, compaction, tool-result shaping, token budgets |
| 05 | MCP server, client, gateway | [MCP revisions](docs/MCP_REVISIONS.md); [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §7 | a teaching subset of MCP (2026-07-28 shape): stateless requests, mirrored headers, MRTR, Tasks; a policy gateway |
| 06 | OAuth and identity propagation | [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §3.5, §7.1 | PKCE, resource indicators, audience-bound tokens, step-up, token exchange, the confused deputy |
| 07 | The agent's own API | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.3, §5.7 | sessions, SSE-style event streams, 202 + task handles, idempotency keys, rate limits |
| 08 | Evals: trajectory, judge, gates | — | golden sets, trajectory metrics, judge calibration (kappa), release gates with confidence intervals |
| 09 | Tracing and metrics | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.10 | spans with `gen_ai.*` attributes, cost per conversation, p95 by step, TTFT and tokens/s, alerts |
| 10 | Reliability | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.2 | backoff with jitter, idempotent retries, circuit breakers, bulkheads, deadlines, fallbacks |
| 11 | Security and prompt injection | [identity](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §6; [sandbox](../../sandboxed-execution/PRIMER.md) §1 | indirect injection demo, provenance-tagged data blocks, screening, allowlists, redaction |
| 12 | Resource estimation | [scaling](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §3 | worked numbers reproduced: cost, peak TPM, Little's law, latency waterfall, vectors |
| 13 | Code evaluation | [vectors](../../retrieval-rag/vector-databases-primer.md) §12 | two full buggy programs and six drills with tests that pass only when your fixes are right |
| 14 | Capstone: the bank agent | all of the above | everything assembled along the spine, with a gate, traces, and a production cost estimate |

A reasonable pace is one or two notebooks per evening; 13 and 14 deserve a full session each.
[`docs/PRIMER_MAP.md`](docs/PRIMER_MAP.md) maps each concept to its notebooks, modules, key exercises and
the primers elsewhere in this repo that go deeper.

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

The names line up with ADK's on purpose (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`,
`FunctionTool`, `AgentTool`, `Runner`, session/memory services, callbacks-as-hooks) so that what you
learn here transfers; `docs/LAB_TO_ADK.md` gives the mapping. The MCP and OAuth modules are labelled
teaching subsets: faithful to the concepts and the current spec shape, not conformant implementations.

## Verifying the whole thing

```bash
make test        # unit tests (~3 s)
make notebooks   # regenerate notebooks/ and solutions/ from notebooks_src/
make check       # rebuild, unit tests, and execute every solution notebook end to end (~40 s)
python tools/run_notebooks.py notebooks --expect-fail   # exercise variants stop at the first unsolved cell
```

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent-format sources with
`### BEGIN SOLUTION` blocks). Edit the sources, not the notebooks; see `notebooks_src/README.md`.

## Using a real model

`pip install -e ".[gemini]"`, set `GOOGLE_API_KEY` (or the Vertex environment variables), then in any
notebook replace `FakeLLM(...)` with `GeminiLLM(model="gemini-3-flash")`. The adapter follows the
`google-genai` 1.x surface and is marked *verify before relying on it*; details and a porting guide
to ADK are in `docs/GEMINI_ADAPTER.md`.

## Honesty notes

- Prices and model names in `estimation/` and `observability/` are illustrative and dated September
  2026; verify against the official pricing page before quoting them.
- The `FakeLLM` is deliberately dumb. The lab teaches the harness — validation, budgets, identity,
  evaluation, tracing — which is where production agent systems succeed or fail. Real-model behaviour
  (and real prompt-injection susceptibility) is out of scope.
- MCP here follows the 2026-07-28 revision's shape (stateless per-request metadata, mirrored headers,
  embedded server-to-client interactions, the Tasks extension), checked against the spec repository on
  2026-09-26 (verify). Many deployed servers still speak the 2025 revisions;
  [`docs/MCP_REVISIONS.md`](docs/MCP_REVISIONS.md) says what differs from 2025-03-26, 2025-06-18 and
  2025-11-25, and where the lab's subset departs from the spec.

## Layout

```
agentlab/           library            tests/          pytest (unit + slow notebook execution)
notebooks_src/    notebook sources   tools/          build_notebooks.py, run_notebooks.py
notebooks/        exercises          solutions/      solved notebooks
docs/             concept map, MCP revisions, ADK mapping, Gemini adapter, contributing
```

MIT licensed.
