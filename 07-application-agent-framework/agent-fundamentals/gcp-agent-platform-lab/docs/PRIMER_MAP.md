# Primer ↔ lab map

The companion primer (*Agent platform on Google Cloud*, not included in this repo) is organised in Parts 0–9. This table says where each part is
practised: which notebook, which library module, and the exercises that make the point stick.

| Primer section | Topic | Notebook(s) | Library | Exercises that matter most |
|---|---|---|---|---|
| 0 | How a design is judged | 00, 14 §9 | — | 14 E6 (the three deep dives you offer) |
| 1.1–1.6 | The design method, spine, simplicity, limitations | 14 (design-review structure) | — | 14 §1, §8 (requirements, not-in-v1, rollout) |
| 2.1 | The single-agent loop | 01 | `agents/loop.py`, `agents/budget.py` | 01 (c) recover from invalid args, (d) budget stops the loop |
| 2.2 | Workflows vs model routing | 02 | `agents/workflow.py` | 02 (a) 3-stage pipeline, (b) loop exit in code |
| 2.3 | Multi-agent patterns, when they earn their keep | 02 | `agents/loop.py` (`AgentTool`) | 02 (c) coordinator + specialists, (d) compounded reliability, (e) collapse to one agent |
| 2.4 | State management, durable execution, HITL | 03, 07 | `agents/state.py`, `agents/runner.py` | 03 (a) resume without re-applying steps, (b) compare-and-set retry, (c) saga, (d) session lock |
| 2.5 | Context engineering | 04 | `agents/context.py`, `llm/fake.py` (`PrefixCache`) | 04 (b) ≥ 60 % cached share, (c) tool-set scoping, (d) token budget |
| 2.6 | Framework map (ADK / LangGraph / CrewAI) | — | `docs/LAB_TO_ADK.md` | — |
| 3.1 | Tool design for models | 01 | `agents/tools.py` | 01 (a) contract design, (b) not_found semantics, (e) result limits |
| 3.2 | MCP as the spec stands | 05 | `mcp/*` | 05 long-running tool (Tasks), policy rule with condition, task polling loop |
| 3.3 | OAuth and identity propagation | 06, 14 §4 | `auth/oauth.py`, `mcp/gateway.py` | 06 audience check, token exchange, step-up, ACL in the system of record |
| 3.4 | Legacy integration patterns | 14 §2–3 (read model, façade) | `mcp/server.py` | 14 E1 gateway rules, E5 staleness note |
| 3.5 | The agent's own API | 07 | `agents/runner.py` (`stream`) | 07 idempotency dedupe, SSE serialisation, job polling, token bucket |
| 3.6 | Class hierarchies | 01, 02 (read `agents/*.py`) | whole `agents/` package | — |
| 4.1 | Evaluation flywheel | 08, 14 §6 | `evals/*` | 08 in-order match, Wilson interval, kappa, gate thresholds, case from a failure |
| 4.2 | Observability and tracing | 09 | `observability/tracing.py` | 09 redaction rule, percentile, TTFT/tps |
| 4.3 | LLM-native metrics and levers | 09, 12 | `observability/metrics.py`, `estimation/calc.py` | 12 lever ordering with cumulative saving |
| 4.4 | Reliability | 10 | `reliability/*` | 10 backoff schedule, half-open transition, fallback rule, GracefulTool |
| 4.5 | Safety and security | 11, 14 §6 | `security/injection.py`, `evals/safety.py` | 11 screening regexes, DataBlock escaping, ActionPolicy check |
| 4.6 | Rollout and change | 14 §8 | — | — |
| 5.1–5.4 | Numbers | 12, 09 | `estimation/calc.py`, `observability/metrics.py` (`PriceTable`) | 12 token_cost with cache share, peak TPM + concurrency, latency budget, vector sizing |
| 6 | Google's stack | `docs/LAB_TO_ADK.md`, `docs/GEMINI_ADAPTER.md` | `llm/gemini.py` | — |
| 7 | Code evaluation | 13 | — | 13 Exercise A (agent loop), Exercise B (retrieval), six drills |
| 8.1 | Bank customer-service agent | 14 | everything | 14 E1–E6 |
| 8.2 | MCP layer over SAP / Salesforce / mainframe | 05 + 14 §3 | `mcp/*` | 05 policy with condition; 14 E1 |
| 8.3 | Claims triage: evals to GA | 08 | `evals/gate.py` | 08 gate thresholds, regression vs baseline |
| 9 | The Senior Staff layer | 14 §8–9 | — | 14 E6 |

## Suggested order and pace

| Evening | Notebooks | Focus |
|---|---|---|
| 1 | 00, 01 | the loop and tool contracts until they are reflex |
| 2 | 02, 03 | control flow and state — the two most common deep dives |
| 3 | 04, 12 | context and numbers — the cost conversation |
| 4 | 05, 06 | MCP and identity — the JD's "secure agentic workflows" line |
| 5 | 07, 10 | the API surface and reliability |
| 6 | 08, 09, 11 | evaluation, tracing, security |
| 7 | 13 | code evaluation, timed: 20 minutes per exercise, narrate aloud |
| 8 | 14 | the capstone as a 45-minute whiteboard rehearsal |
