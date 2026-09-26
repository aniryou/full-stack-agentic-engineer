# Concept map: where each idea is practised

This lab is self-contained: every concept below is taught in a notebook and implemented in
a library module you can read. The last column points at the primers elsewhere in this repo
that go deeper on the same idea. (A dedicated agent-platform primer is not written yet; until
it is, this map and the notebooks' own introductions are the reading.)

Primers referenced below, relative to this file:

- **Durable** — [long-running agentic workflows](../../../long-running-durable/00_primer.md)
- **Identity** — [identity and security for agentic systems](../../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
- **Scaling** — [scaling agentic solutions](../../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
- **Sandbox** — [sandboxed execution](../../../sandboxed-execution/PRIMER.md)
- **Vectors** — [vector databases](../../../retrieval-rag/vector-databases-primer.md)

## The map

| Concept | Notebook(s) | Library | Exercises that matter most | Deeper in this repo |
|---|---|---|---|---|
| How a design is judged: requirements, the spine (channels → runtime → tools and gateway → systems of record), what is not in v1 | 00, 14 §1 and §9 | — | 14 §1, §8 (requirements, not-in-v1, rollout) | Scaling §8 |
| The single-agent loop and its budget | 00, 01 | `agents/loop.py`, `agents/budget.py` | 01 (c) recover from invalid args, (d) budget stops the loop | — |
| Workflows vs model routing: control flow in code | 02 | `agents/workflow.py` | 02 (a) 3-stage pipeline, (b) loop exit in code | Durable §4 (P8) |
| Multi-agent patterns, and when they earn their keep | 02 | `agents/loop.py` (`AgentTool`) | 02 (c) coordinator + specialists, (d) compounded reliability, (e) collapse to one agent | Scaling §1.7, §5.11 |
| State, durable execution, human in the loop | 03, 07 | `agents/state.py`, `agents/runner.py` | 03 (a) resume without re-applying steps, (b) compare-and-set retry, (c) saga, (d) session lock | Durable §2–§3, §4 (P3, P4) |
| Context engineering and prompt caching | 00, 04 | `agents/context.py`, `llm/fake.py` (`PrefixCache`) | 04 (b) ≥ 60 % cached share, (c) tool-set scoping, (d) token budget | Scaling §1.4, §5.5; Durable §3.5 |
| Framework map (ADK / LangGraph / CrewAI) | — | [`LAB_TO_ADK.md`](LAB_TO_ADK.md) | — | — |
| Tool design for models: contracts, side-effect classes, errors | 01 | `agents/tools.py` | 01 (a) contract design, (b) not_found semantics, (e) result limits | Identity §4.2; Sandbox §3 |
| MCP: server, client, transports, Tasks | 05 | `mcp/*` | 05 long-running tool (Tasks), policy rule with condition, task polling loop | [MCP revisions](MCP_REVISIONS.md); Durable §4 (P7) |
| OAuth and identity propagation: audience, token exchange, step-up, the confused deputy | 06, 14 §4 | `auth/oauth.py`, `mcp/gateway.py` | 06 audience check, token exchange, step-up, ACL in the system of record | Identity §3.5, §4.3, §7.1 |
| The policy gateway in front of tools | 05, 06 | `mcp/gateway.py` | 05 policy with condition; 14 E1 gateway rules | Identity §4.1 |
| Legacy integration: read model, façade | 14 §2–§3 | `mcp/server.py` | 14 E1 gateway rules, E5 staleness note | — |
| The agent's own API: sessions, streams, task handles, idempotency, rate limits | 07 | `agents/runner.py` (`stream`) | 07 idempotency dedupe, SSE serialisation, job polling, token bucket | Scaling §5.3, §5.7 |
| Class hierarchies worth reading | 01, 02 | whole `agents/` package | — | — |
| The evaluation flywheel: golden sets, trajectories, judges, gates | 08, 14 §6 | `evals/*` | 08 in-order match, Wilson interval, kappa, gate thresholds, case from a failure | Vectors §14 (retrieval evals) |
| Tracing with `gen_ai.*` attributes, redaction | 09 | `observability/tracing.py` | 09 redaction rule, percentile, TTFT/tps | Scaling §5.10; Sandbox §8 |
| LLM-native metrics and cost levers | 09, 12 | `observability/metrics.py`, `estimation/calc.py` | 12 lever ordering with cumulative saving | Scaling §3.4 |
| Reliability: retries, breakers, bulkheads, deadlines, fallbacks | 10 | `reliability/*` | 10 backoff schedule, half-open transition, fallback rule, GracefulTool | Scaling §5.2; Durable §3.2 |
| Safety and prompt injection | 11, 14 §6 | `security/injection.py`, `evals/safety.py` | 11 screening regexes, DataBlock escaping, ActionPolicy check | Identity §6; Sandbox §1 |
| Rollout and change | 14 §8 | — | — | Scaling §7 |
| The numbers: token cost, peak TPM, Little's law, latency budgets, vector sizing | 12, 09 | `estimation/calc.py`, `observability/metrics.py` (`PriceTable`) | 12 token_cost with cache share, peak TPM + concurrency, latency budget, vector sizing | Scaling §3; Vectors §7 |
| Google's stack | [`LAB_TO_ADK.md`](LAB_TO_ADK.md), [`GEMINI_ADAPTER.md`](GEMINI_ADAPTER.md) | `llm/gemini.py` | — | Identity §10 |
| Reading code for defects | 13 | — | 13 Exercise A (agent loop), Exercise B (retrieval), six drills | Vectors §12 |
| Worked design: a bank customer-service agent | 14 | everything | 14 E1–E6 | — |
| Worked design: an MCP layer over legacy systems of record | 05, 14 §3 | `mcp/*` | 05 policy with condition; 14 E1 | Identity §7 |
| Worked design: taking an eval suite to a release gate | 08 | `evals/gate.py` | 08 gate thresholds, regression vs baseline | — |
| Presenting a design: the deep dives to offer and the trade-offs to name | 14 §8–§9 | — | 14 E6 | Scaling §8 |

## Suggested order and pace

| Evening | Notebooks | Focus |
|---|---|---|
| 1 | 00, 01 | the loop and tool contracts until they are reflex |
| 2 | 02, 03 | control flow and state — the two questions a design review asks first |
| 3 | 04, 12 | context and numbers — the cost conversation |
| 4 | 05, 06 | MCP and identity — how an agent acts on a user's behalf without holding their keys |
| 5 | 07, 10 | the API surface and reliability |
| 6 | 08, 09, 11 | evaluation, tracing, security |
| 7 | 13 | reading code for defects: about 20 minutes per exercise, explaining each finding as you go |
| 8 | 14 | the capstone, presented end to end as a 45-minute design review |
