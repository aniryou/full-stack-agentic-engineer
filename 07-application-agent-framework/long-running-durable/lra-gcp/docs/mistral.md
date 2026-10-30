# Mistral as the model provider, and Mistral Workflows as the orchestrator

The engine of the lab does not depend on the model that makes the decisions.
[`lra.adapters.mistral`](../src/lra/adapters/mistral/__init__.py) puts Mistral in at two seams.
[`lra.adapters.mistral.workflow`](../src/lra/adapters/mistral/workflow.py) shows the same tool loop on Mistral
Workflows. There, the platform supplies the durability that the engine builds by hand. The notebook
[`05_tool_loop_and_mistral`](../notebooks/05_tool_loop_and_mistral.ipynb) shows all of it offline.

```bash
pip install -e ".[dev,mistral]"          # mistralai; on Python 3.12-3.14 also mistralai-workflows
python -m pytest tests/test_mistral_adapter.py tests/test_mistral_workflow.py -q
python -m lra.adapters.mistral.demo live       # the lra tool loop with a real Mistral model (export MISTRAL_API_KEY)
python -m lra.adapters.mistral.demo workflow   # Mistral Workflows on a local Temporal dev server (Python 3.12+)
```

**Python:** The adapter and its tests run on 3.11+. They use a fake client, so they need no key and no network.
Mistral Workflows needs **Python 3.12–3.14**, because `mistralai-workflows` 3.15 declares
`Requires-Python >=3.12,<3.15` (checked 2026-09-26, verify).

On 3.11, the `mistral` extra does not install `mistralai-workflows`. `tests/test_mistral_workflow.py` skips with that
reason, and `demo workflow` exits with the same message. On first use, the workflow tests and the demo download a
Temporal dev server. Thus they need network access to `temporal.download`.

## The two seams

| Seam | Class | What it does |
|---|---|---|
| the `LLM` port of the engine | `MistralLLM` | It supplies `generate(prompt, system=..., json_mode=...)` on chat completions. The usage becomes cost for the `Budget` of the run. The prices are only examples, so set the real prices. |
| the decider of the tool loop | `MistralDecider` | It changes the journal of the run into a chat. The chat has the goal first, then each earlier decision as an assistant tool call with its recorded result. Then the decider asks for the next call with function calling (`tool_choice="auto"`, one call at a time). |

A tool-call id must have exactly nine alphanumeric characters. Thus the adapter hashes the idempotency key of the
journal (`run_ab12cd:3`) to a stable nine-character id. The SDK is a namespace package: `from mistralai.client import Mistral`.

## The five invariants, and who provides them

| # | Invariant (PRIMER §3) | The `lra` engine | Mistral Workflows |
|---|---|---|---|
| 1 | **Durable state** | the run document, with a checkpoint after each step | The event history of the execution. A replacement worker replays it. |
| 2 | **Intent before act** | `ctx.effect("decide:N")` records the decision. `ctx.effect("act:N")` records the call. Downstream, the key is `run_id:N`. | The platform records each **activity** and retries it on failure. It never runs a complete activity again. The model call is also an activity. The side effect still carries `execution_id:N` as its idempotency key. |
| 3 | **Exclusive progress** | a lease with a TTL, the reaper | One worker for each task. If the worker misses heartbeats, the platform schedules the task again. |
| 4 | **Bounded execution** | `Budget(max_steps=...)`, with a check in the code | A loop limit in deterministic workflow code, and also `execution_timeout`. |
| 5 | **Wait in the store, do not hold a process** | `Wait` on an approval key, `engine.resume` | `workflow.wait_condition(...)` and `@workflow.signal`. The execution waits at zero cost until the signal or the timeout. |

One property of the platform is not obvious. An *unexpected* exception in workflow code fails only the workflow
**task**. The platform retries that task until you repair the code and deploy it again. Thus a bug never loses a run.
To fail an execution on purpose, raise `WorkflowError`.

## The Mistral pieces

The product facts in this table (APIs, SDK names, preview status, limits) are as of 2026-09-26 (verify).

| Need | Mistral piece | Here |
|---|---|---|
| A model that selects the next step | Chat Completions with function calling, through the `mistralai` SDK (`chat.complete(..., tools=..., tool_choice="auto")`). Aliases: `mistral-medium-latest` (the default here), `mistral-small-latest` for low-cost steps, `mistral-large-latest`. | `MistralDecider`, `MistralLLM` |
| Durable orchestration: state, retries, waits, timeouts | **Mistral Workflows** (in Studio, public preview since April 2026). Its base is Temporal. Hybrid mode: Mistral hosts the orchestrator, and your workers run in your infrastructure and connect outbound. You can also host the orchestrator yourself. The Python SDK `mistralai-workflows` has `@workflows.activity(start_to_close_timeout, retry_policy_max_attempts, heartbeat_timeout)`, `@workflows.workflow.define(name, execution_timeout)`, `@workflow.signal` and `workflow.wait_condition`. It also has schedules, child workflows, a 2 MB payload limit with offload to your blob storage, and SDK-side encryption. | `workflow.py`, `local_temporal.py` |
| An agent loop *inside* a workflow, with server-side tools | The Durable Agents plugin (`mistralai-workflows[mistralai]`): `Agent(model, instructions, tools=[activities], handoffs=[...])`, `Runner.run(agent, inputs, session, max_turns)`, `RemoteSession` (Agents API) or `LocalSession` (chat completions, on-prem). Built-in tools run server-side (web search, code interpreter, image generation, document library). MCP servers connect through stdio, SSE or streamable HTTP. | the next level after the hand-written loop in `workflow.py` |
| Server-side conversation state | Agents API: `beta.conversations.start(agent_id, inputs)` gives a `function.call` entry with a `tool_call_id`. That entry is the pending intent. `beta.conversations.append(conversation_id, inputs=[FunctionResultEntry(tool_call_id, result)])` is the resume. | only this file describes it |
| Input from a person | `wait_condition` and a signal (above), or `InteractiveWorkflow.wait_for_input()` for forms and confirmations in chat style | `workflow.py` |
| Schedules | `schedules=[...]` on `workflow.define` (calendar or interval, overlap policy) | none |
| Observability | The event history of the execution and OpenTelemetry traces in Studio. `execution_timeout` can be up to months. | none |
| Deployment | Hosted (La Plateforme, EU/US regional endpoints), dedicated, hybrid, or self-hosted open-weight models on your GPUs (vLLM). Only one line changes: the client of the adapter. | none |

On 2026-09-19, we compared these facts with docs.mistral.ai and with SDK introspection (`mistralai` 2.10.1,
`mistralai-workflows` 3.15.0). The docs pages were: Workflows overview, core concepts, activities, waiting for
conditions, durable agents. Before you rely on the model aliases and the preview status of Workflows, examine them
again.

## What was verified, and what wasn't

* **Verified offline:** the adapter tests (fake client) and notebook 05. When we built the Mistral lab, we ran the
  workflow tests and `demo workflow` end to end on a Temporal dev server, and they passed. They examined these cases:
  * An approval signal causes one charge with the key `execution_id:1`.
  * The worker crashes after the charge. The retried activity converges to one charge, and the workflow does not ask
    the model again.
  * The workflow gives a rejection back to the model.
  * An approval timeout gives `approval expired`.
  * The budget stops the run.

  These tests run in CI on Python 3.12 (`lra-gcp-mistral-py312` in `tools/ci/labs.json`).
* **Not verified:** a live model call (`demo live`, which needs `MISTRAL_API_KEY`) and a worker against the hosted
  orchestrator of Mistral (`workflows.run_worker([InvoiceAgent])`).
