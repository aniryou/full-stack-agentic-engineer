# From the lab to ADK (and back)

The lab's runtime is deliberately shaped like Google's Agent Development Kit so the concepts
transfer. This is the translation table to use when you rebuild a notebook on ADK, or when an
reader uses ADK vocabulary. ADK moves quickly (2.0 in 2026) — treat the right-hand column as
"the name to look up", not gospel, and check the current docs before relying on it.

| Concept | Lab (`agentlab`) | ADK | Notes |
|---|---|---|---|
| Model-driven agent | `agents.LlmAgent(name, llm, instruction, tools, sub_agents, output_key)` | `Agent` / `LlmAgent(name, model, instruction, tools, sub_agents, output_key)` | ADK templates `{state_key}` into instructions the same way (`render_template` here) |
| Custom control flow | subclass `agents.BaseAgent`, implement `run(ctx) -> events` | subclass `BaseAgent`, implement `_run_async_impl` yielding events | |
| Deterministic workflows | `SequentialAgent`, `ParallelAgent`, `LoopAgent(max_iterations, until)` | `SequentialAgent`, `ParallelAgent`, `LoopAgent(max_iterations)`; exit via escalate/conditions; native DAG `Workflow` in 2.0 | ADK's loop exits when a sub-agent escalates; here `until(session)` is code |
| Function tools | `@tool` / `FunctionTool(fn, side_effect=..., required_scope=...)` — schema from the signature via pydantic | `FunctionTool(func)` — schema from the signature and docstring | side-effect classes and scopes are lab additions; ADK has tool confirmation for HITL |
| Delegation | `AgentTool(agent)`; child session, shared budget | `AgentTool(agent)`; LLM-driven `transfer_to_agent` for hand-off | hand-off (same session) vs delegation (child session) is the same distinction in both |
| MCP tools | `mcp.McpToolset(client)` → remote tools | `MCPToolset(connection_params=...)` | ADK speaks the current MCP revision through the official SDK |
| OpenAPI tools | — | `OpenAPIToolset(spec)` | the lab's façade servers play the same role |
| Session and events | `agents.Session` (event log + `state`), `InMemorySessionStore`, `JsonFileSessionStore` | `Session` with `events` and `state`; `InMemorySessionService`, `DatabaseSessionService`, `VertexAiSessionService` (Agent Runtime) | state key prefixes `app:` `user:` `temp:` mirror ADK's scoping |
| Long-term memory | `ContextBuilder.memory_provider` | `MemoryService` (in-memory, Agent Runtime Memory Bank) | |
| Runner | `agents.Runner(agent, store, tracer).run / stream / approve` | `Runner(agent, app_name, session_service).run_async(...)` yields events | approve/resume here models ADK's tool confirmation + pause/resume |
| Guardrail hooks | `security.GuardedTool`, `reliability.GracefulTool`, `ContextBuilder` | callbacks: `before_model_callback`, `after_tool_callback`, etc. | ADK attaches cross-cutting logic as callbacks on the agent; the lab wraps tools |
| Budgets | `agents.Budget(max_steps, max_tokens, max_seconds, max_depth)` shared through `InvocationContext` | `RunConfig` limits (e.g. max LLM calls), `RetryConfig` in 2.0 | |
| Evaluation | `evals.run_eval`, `Gate`, trajectory metrics, judges | `adk eval` with eval sets (tool trajectory + response criteria), user/environment simulation | the metrics have the same shape: trajectory match and rubric response scoring |
| Tracing | `observability.Tracer` with `gen_ai.*` attributes | OpenTelemetry integration; Agent Observability / Cloud Trace on the platform | |
| Model | `llm.FakeLLM` / `llm.GeminiLLM` | model string (`gemini-3-flash`) or a `Gemini(...)` model object; LiteLLM for others | |
| Dev loop | `make lab`, `tools/run_notebooks.py` | `adk web`, `adk run`, `adk api_server`, `adk eval`, `adk deploy` | |
| Deployment | — (notebook 07 sketches the API) | Agent Runtime (formerly Agent Engine), Cloud Run, GKE | |
| Governance | `mcp.Gateway` with `Policy` rules, `X-Agent-Identity` | Agent Registry + Agent Gateway + Agent Identity (SPIFFE, mTLS/DPoP), Model Armor | the lab's gateway is a stand-in for the platform's egress governance |

## Porting a notebook to ADK in an evening

1. `pip install google-adk` and set `GOOGLE_API_KEY` (or Vertex environment variables).
2. Re-declare the notebook's tools as plain typed Python functions with docstrings (ADK derives the schema as the lab does).
3. Replace `LlmAgent(..., llm=FakeLLM(...))` with `Agent(model="gemini-3-flash", ...)`; keep the instruction and `output_key`.
4. Replace `Runner(agent).run(session_id, message)` with ADK's `Runner` + a session service, iterating events.
5. Move `GuardedTool` logic into `before_tool_callback` / `after_tool_callback`; move `GracefulTool` retries into `RetryConfig`.
6. Rebuild the golden set as an ADK eval set and run `adk eval` — compare its trajectory report with notebook 08's.
7. Deploy with `adk deploy` to Agent Runtime and read the trace in Agent Observability; note two friction points you would file as product feedback.
