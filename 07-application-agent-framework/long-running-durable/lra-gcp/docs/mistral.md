# Mistral as the model provider, and Mistral Workflows as the orchestrator

The lab's engine does not care which model decides. [`lra.adapters.mistral`](../src/lra/adapters/mistral/__init__.py)
swaps Mistral in at two seams, and [`lra.adapters.mistral.workflow`](../src/lra/adapters/mistral/workflow.py) shows the
same tool loop on Mistral Workflows, where the platform provides the durability the engine builds by hand. Notebook
[`05_tool_loop_and_mistral`](../notebooks/05_tool_loop_and_mistral.ipynb) walks all of it offline.

```bash
pip install -e ".[dev,mistral]"          # mistralai; on Python 3.12-3.14 also mistralai-workflows
python -m pytest tests/test_mistral_adapter.py tests/test_mistral_workflow.py -q
python -m lra.adapters.mistral.demo live       # the lra tool loop with a real Mistral model (export MISTRAL_API_KEY)
python -m lra.adapters.mistral.demo workflow   # Mistral Workflows on a local Temporal dev server (Python 3.12+)
```

**Python:** the adapter and its tests run on 3.11+ (with a fake client, so no key and no network). Mistral Workflows
needs **Python 3.12–3.14**: `mistralai-workflows` 3.15 declares `Requires-Python >=3.12,<3.15` (checked 2026-09-26,
verify). On 3.11 the `mistral` extra skips it, `tests/test_mistral_workflow.py` skips with that reason and
`demo workflow` exits with the same message. The workflow tests and demo download a Temporal dev server on first
use, so they need network access to `temporal.download`.

## The two seams

| Seam | Class | What it does |
|---|---|---|
| the engine's `LLM` port | `MistralLLM` | `generate(prompt, system=..., json_mode=...)` on chat completions; usage becomes cost for the run's `Budget` (illustrative prices; set real ones) |
| the tool loop's decider | `MistralDecider` | turns the run's journal into a chat (the goal, then every earlier decision as an assistant tool call with its recorded result) and asks for the next call with function calling (`tool_choice="auto"`, one call at a time) |

Tool-call ids must be exactly nine alphanumeric characters, so the journal's idempotency key (`run_ab12cd:3`) is
hashed to a stable nine-character id. The SDK is a namespace package: `from mistralai.client import Mistral`.

## The five invariants, and who provides them

| # | Invariant (PRIMER §3) | The `lra` engine | Mistral Workflows |
|---|---|---|---|
| 1 | **Durable state** | the run document, checkpointed after every step | the execution's event history; a replacement worker replays it |
| 2 | **Intent before act** | `ctx.effect("decide:N")` records the decision, `ctx.effect("act:N")` the call, keyed `run_id:N` downstream | every **activity** is recorded and retried on failure, never re-run once complete; the model call is an activity too; the side effect still carries `execution_id:N` as its idempotency key |
| 3 | **Exclusive progress** | a lease with a TTL, the reaper | one worker per task; missed heartbeats → the task is rescheduled |
| 4 | **Bounded execution** | `Budget(max_steps=...)` checked in code | a loop bound in deterministic workflow code, plus `execution_timeout` |
| 5 | **Park, don't wait** | `Wait` on an approval key, `engine.resume` | `workflow.wait_condition(...)` + `@workflow.signal`: suspended at zero cost until the signal or the timeout |

One platform subtlety: an *unexpected* exception in workflow code fails only the workflow **task**, which the platform
retries until the code is fixed and redeployed, so a bug never loses a run. To fail an execution on purpose, raise
`WorkflowError`.

## The Mistral pieces

Product facts in this table (APIs, SDK names, preview status, limits) are as of 2026-09-26 (verify).

| Need | Mistral piece | Here |
|---|---|---|
| A model that picks the next step | Chat Completions with function calling via the `mistralai` SDK (`chat.complete(..., tools=..., tool_choice="auto")`). Aliases: `mistral-medium-latest` (the default here), `mistral-small-latest` for cheap steps, `mistral-large-latest`. | `MistralDecider`, `MistralLLM` |
| Durable orchestration: state, retries, waits, timeouts | **Mistral Workflows** (Studio; public preview since April 2026), built on Temporal. Hybrid mode: Mistral hosts the orchestrator, your workers run in your infrastructure and connect outbound; the orchestrator can also be self-hosted. Python SDK `mistralai-workflows`: `@workflows.activity(start_to_close_timeout, retry_policy_max_attempts, heartbeat_timeout)`, `@workflows.workflow.define(name, execution_timeout)`, `@workflow.signal`, `workflow.wait_condition`, schedules, child workflows, a 2 MB payload limit with offloading to your blob storage, SDK-side encryption. | `workflow.py`, `local_temporal.py` |
| An agent loop *inside* a workflow, with server-side tools | Durable Agents plugin (`mistralai-workflows[mistralai]`): `Agent(model, instructions, tools=[activities], handoffs=[...])`, `Runner.run(agent, inputs, session, max_turns)`, `RemoteSession` (Agents API) or `LocalSession` (chat completions; on-prem). Built-in tools run server-side (web search, code interpreter, image generation, document library); MCP servers via stdio, SSE or streamable HTTP. | the step up from `workflow.py`'s hand-written loop |
| Server-side conversation state | Agents API: `beta.conversations.start(agent_id, inputs)` → a `function.call` entry with a `tool_call_id` is the pending intent; `beta.conversations.append(conversation_id, inputs=[FunctionResultEntry(tool_call_id, result)])` is the resume. | described here only |
| Human input | `wait_condition` + a signal (above), or `InteractiveWorkflow.wait_for_input()` for chat-style forms and confirmations | `workflow.py` |
| Scheduling | `schedules=[...]` on `workflow.define` (calendar or interval, overlap policy) | none |
| Observability | the execution's event history and OpenTelemetry traces in Studio; `execution_timeout` up to months | none |
| Deployment | hosted (La Plateforme, EU/US regional endpoints), dedicated, hybrid, or self-hosted open-weight models on your GPUs (vLLM): the adapter's client is the only line that changes | none |

Checked on 2026-09-19 against docs.mistral.ai (Workflows overview, core concepts, activities, waiting for conditions,
durable agents) and SDK introspection (`mistralai` 2.10.1, `mistralai-workflows` 3.15.0). Re-check the model aliases
and the preview status of Workflows before relying on them.

## What was verified, and what wasn't

* **Verified offline:** the adapter tests (fake client) and notebook 05. The workflow tests and `demo workflow`
  were verified end to end on a Temporal dev server when the Mistral lab was built (approval signal → one charge with
  key `execution_id:1`; the worker dies after charging → the retried activity converges to one charge and the model
  is not re-asked; rejection fed back; approval timeout → `approval expired`; budget stop); they run in CI on
  Python 3.12 (`lra-gcp-mistral-py312` in `tools/ci/labs.json`).
* **Not verified:** a live model call (`demo live`, needs `MISTRAL_API_KEY`) and a worker against Mistral's hosted
  orchestrator (`workflows.run_worker([InvoiceAgent])`).
