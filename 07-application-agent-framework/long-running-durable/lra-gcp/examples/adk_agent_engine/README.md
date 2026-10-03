# ADK 2 + Agent Engine: the managed-runtime path

This example is the same pipeline as `src/lra/examples/research_pipeline.py`. It uses the
`Workflow` graph of ADK 2, and you can deploy it to Vertex AI Agent Engine.

| File | What it is |
|---|---|
| `workflow_agent.py` | The graph. Its nodes come in this sequence: plan, parallel research, synthesize, the critique and revise loop, the **review interrupt**, publish. |
| `run_local.py` | The first call starts the workflow and pauses at the review. A second call resumes the workflow **from a new process** (SQLite sessions). |
| `agent_engine_app.py` | The `AdkApp` wrapper and the `agent_engines.create(...)` deploy. |

```bash
pip install -e ".[adk]"
python examples/adk_agent_engine/run_local.py "durable execution for agents"   # pauses at review
python examples/adk_agent_engine/run_local.py approve                          # resumes, publishes
```

## When to choose this over the Cloud Run engine

| | `lra` engine (Cloud Run + Cloud Tasks + Firestore) | ADK 2 `Workflow` on Agent Engine |
|---|---|---|
| Durability unit | Each step is a task in a queue. The state is in Firestore. | The session keeps the outputs of the nodes and the state. ADK replays them on resume. |
| Fan-out | The engine uses **child runs**. The engine sends them to different replicas. Each child run has its own retries and its own budget. | ADK uses `parallel_worker`, with asyncio in one invocation. |
| Retries | Each step has a retry policy. The run history records the retries. The retry count stays after a restart. | ADK uses `RetryConfig` in the process. ADK does not keep the retry count across a resume. |
| Waits for a person | The step returns `Wait`, and then no task exists. A Cloud Tasks timer controls the timeouts. | The workflow stops at an interrupt through `long_running_tool_ids`. A `FunctionResponse` resumes it. |
| Budgets | The engine has a built-in budget type (`Budget`). | You make them with plugins or callbacks. |
| Operations | You operate the queues, the reaper and the dashboards. | A managed runtime, sessions, Memory Bank, tracing. |
| Best for | Agents that **act** across systems, need sagas, run for hours–weeks, have a high fan-out. | Conversational agents and workflow agents, a fast path to production, Gemini-native tools. |

You can use the two together. An Agent Engine tool can send `POST /runs` to start a run of the engine. It can then
resume the ADK invocation when the completion event of the run arrives.
