# ADK 2 + Agent Engine: the managed-runtime path

Same pipeline as `src/lra/examples/research_pipeline.py`, expressed with ADK 2's
`Workflow` graph and deployable to Vertex AI Agent Engine.

| File | What it is |
|---|---|
| `workflow_agent.py` | The graph: plan → parallel research → synthesize → critique/revise loop → **review interrupt** → publish |
| `run_local.py` | Start → pause at review → resume **from a new process** (SQLite sessions) |
| `agent_engine_app.py` | `AdkApp` wrapper + `agent_engines.create(...)` deploy |

```bash
pip install -e ".[adk]"
python examples/adk_agent_engine/run_local.py "durable execution for agents"   # pauses at review
python examples/adk_agent_engine/run_local.py approve                          # resumes, publishes
```

## When to choose this over the Cloud Run engine

| | `lra` engine (Cloud Run + Cloud Tasks + Firestore) | ADK 2 `Workflow` on Agent Engine |
|---|---|---|
| Durability unit | every step is a queued task; state in Firestore | node outputs/state persisted in the session; replayed on resume |
| Fan-out | **child runs** — distributed across replicas, individually retried/budgeted | `parallel_worker` — asyncio inside one invocation |
| Retries | per-step policy, recorded in run history, survive restarts | `RetryConfig`, in-process; retry count not persisted across resume |
| Waiting on humans | `Wait` → no task exists; Cloud Tasks timer for timeouts | interrupt via `long_running_tool_ids`; resume with `FunctionResponse` |
| Budgets | first-class (`Budget`) | build with plugins/callbacks |
| Ops | you own queues, reaper, dashboards | managed runtime, sessions, Memory Bank, tracing |
| Best for | agents that **act** across systems, need sagas, hours–weeks, high fan-out | conversational + workflow agents, fast path to production, Gemini-native tooling |

They compose: an Agent Engine tool can `POST /runs` to start an engine run and
resume the ADK invocation when the run's completion event arrives.
