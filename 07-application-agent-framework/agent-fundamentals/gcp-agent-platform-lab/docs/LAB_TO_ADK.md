# From the lab to ADK (and back)

The lab's runtime has the shape of Google's Agent Development Kit on purpose, so that the
concepts transfer. This page is the translation table. Use it when you rebuild a notebook on ADK,
or when a reader uses ADK vocabulary. ADK changes fast: `google-adk` 2.0.0 reached PyPI on
2026-05-19 and 2.10.0 was current on 2026-09-25 (verify).

The ADK and Google Cloud names in the right-hand columns are as of September 2026 (verify).
These names are Agent Runtime (old name: Agent Engine), Memory Bank, Agent Registry, Agent
Gateway, Agent Identity and the `gemini-3-flash` model string. Treat them as "the name to look
up", not as the final truth. Examine the current docs before you rely on them.

| Concept | Lab (`agentlab`) | ADK | Notes |
|---|---|---|---|
| Model-driven agent | `agents.LlmAgent(name, llm, instruction, tools, sub_agents, output_key)` | `Agent` / `LlmAgent(name, model, instruction, tools, sub_agents, output_key)` | ADK puts `{state_key}` values into instructions in the same way. Here, `render_template` does this. |
| Custom control flow | subclass `agents.BaseAgent`, implement `run(ctx) -> events` | subclass `BaseAgent`, implement `_run_async_impl`, which yields events | |
| Deterministic workflows | `SequentialAgent`, `ParallelAgent`, `LoopAgent(max_iterations, until)` | `SequentialAgent`, `ParallelAgent`, `LoopAgent(max_iterations)`. The loop exits through escalate or conditions. 2.x has a native DAG `Workflow` (verify). | ADK's loop exits when a sub-agent escalates. Here, `until(session)` is code. |
| Function tools | `@tool` / `FunctionTool(fn, side_effect=..., required_scope=...)`. The schema comes from the signature, through pydantic. | `FunctionTool(func)`. The schema comes from the signature and the docstring. | Side-effect classes and scopes are lab additions. ADK has tool confirmation for HITL. |
| Delegation | `AgentTool(agent)`, with a child session and a shared budget | `AgentTool(agent)`, and the LLM-driven `transfer_to_agent` for a hand-off | The distinction between a hand-off (same session) and a delegation (child session) is the same in both. |
| MCP tools | `mcp.McpToolset(client)` gives remote tools | `MCPToolset(connection_params=...)` | ADK uses the current MCP revision through the official SDK. |
| OpenAPI tools | — | `OpenAPIToolset(spec)` | The lab's façade servers do the same job. |
| Session and events | `agents.Session` (event log + `state`), `InMemorySessionStore`, `JsonFileSessionStore` | `Session` with `events` and `state`. The session services are `InMemorySessionService`, `DatabaseSessionService` and `VertexAiSessionService` (Agent Runtime). | The state key prefixes `app:` `user:` `temp:` match the scopes of ADK. |
| Long-term memory | `ContextBuilder.memory_provider` | `MemoryService` (in-memory, Agent Runtime Memory Bank) | |
| Runner | `agents.Runner(agent, store, tracer).run / stream / approve` | `Runner(agent, app_name, session_service).run_async(...)` yields events | Here, approve and resume simulate ADK's tool confirmation and its pause and resume. |
| Guardrail hooks | `security.GuardedTool`, `reliability.GracefulTool`, `ContextBuilder` | callbacks, for example `before_model_callback` and `after_tool_callback` | ADK attaches cross-cutting logic as callbacks on the agent. The lab wraps the tools. |
| Budgets | `agents.Budget(max_steps, max_tokens, max_seconds, max_depth)` shared through `InvocationContext` | `RunConfig` limits (for example, the maximum number of LLM calls), and `RetryConfig` in 2.x (verify) | |
| Evaluation | `evals.run_eval`, `Gate`, trajectory metrics, judges | `adk eval` with eval sets (tool trajectory and response criteria), and user and environment simulation | The metrics have the same shape: a trajectory match and a rubric score for the response. |
| Tracing | `observability.Tracer` with `gen_ai.*` attributes | OpenTelemetry integration, and Agent Observability / Cloud Trace on the platform | |
| Model | `llm.FakeLLM` / `llm.GeminiLLM` | A model string (`gemini-3-flash`, September 2026, verify) or a `Gemini(...)` model object. LiteLLM for other models. | |
| Dev loop | `make lab`, `tools/run_notebooks.py` | `adk web`, `adk run`, `adk api_server`, `adk eval`, `adk deploy` | |
| Deployment | None. Notebook 07 gives a sketch of the API. | Agent Runtime (old name: Agent Engine, verify), Cloud Run, GKE | |
| Governance | `mcp.Gateway` with `Policy` rules, `X-Agent-Identity` | Agent Registry + Agent Gateway + Agent Identity (SPIFFE, mTLS/DPoP), Model Armor | The lab's gateway is a substitute for the platform's egress governance. |

## Porting a notebook to ADK in an evening

1. Run `pip install google-adk`. Then set `GOOGLE_API_KEY` (or the Vertex environment variables).
2. Declare the notebook's tools again as plain, typed Python functions with docstrings. ADK gets the schema from them, as the lab does.
3. Replace `LlmAgent(..., llm=FakeLLM(...))` with `Agent(model="gemini-3-flash", ...)`, or with the current Flash model string (verify). Keep the instruction and `output_key`.
4. Replace `Runner(agent).run(session_id, message)` with ADK's `Runner` and a session service. Then iterate over the events.
5. Move the `GuardedTool` logic into `before_tool_callback` / `after_tool_callback`. Move the `GracefulTool` retries into `RetryConfig`.
6. Rebuild the golden set as an ADK eval set. Run `adk eval`. Compare its trajectory report with the report of notebook 08.
7. Deploy with `adk deploy` to Agent Runtime. Read the trace in Agent Observability. Record two friction points to report as product feedback.
