# Long-running agents — the core idea, Mistral edition

> **A long-running agent is not a long-running process.**
> It wakes up, does *one* step, checkpoints, and goes back to sleep. Waiting for a human, a slow job or the world means **parking** the run — nothing runs, nothing costs — until something resumes it.

Three files carry the idea:

| File | What it is | Depends on |
|---|---|---|
| `durable.py` | the five rules in ~170 lines: journal + write-ahead intent + idempotency key, expiring lease, budget, park/resume | stdlib only |
| `mistral_model.py` | the model swap: a Mistral model decides the next step via function calling over the journal | `mistralai` |
| `mistral_workflow.py` | **the same agent as a Mistral Workflow** — Temporal underneath provides the five rules, so the plumbing disappears | `mistralai-workflows[mistralai]` |

**Time and tier:** ~2 h (rough); module 07.3, the Mistral Workflows version in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: the durable core and the adapter tests run offline with a fake client. The workflow half needs Python 3.12–3.14 and a local Temporal dev server (still free); a `MISTRAL_API_KEY` adds the live model.

## The five rules, and who provides them

| # | Rule | `durable.py` | Mistral Workflows |
|---|---|---|---|
| 1 | **Durable state** | `Store.save()` after every step | the execution's event history; a replacement worker replays it |
| 2 | **Intent → act** | journal the call *with a key*, save, then act; a retry repeats the same call, never re-asks the model | every **activity** is recorded and retried on failure, never re-run once complete — the model call is an activity too; the side effect still carries `execution_id:step` as its idempotency key |
| 3 | **Lease** | `acquire_lease` with a TTL | one worker per task; missed heartbeats → task rescheduled |
| 4 | **Budget** | `max_steps` checked in code | a loop bound in deterministic workflow code + `execution_timeout` |
| 5 | **Park, don't wait** | `WAITING` + token, `resume()` | `workflow.wait_condition(...)` + `@workflow.signal` — suspended at zero cost, resumes on the signal or times out |

One platform subtlety worth knowing: an *unexpected* exception in workflow code fails only the workflow **task**, which the platform retries until the code is fixed and redeployed — a bug never loses a run. To fail an execution on purpose, raise `WorkflowError`.

## Run it

**Python:** the stdlib core (`durable.py`), the adapter (`mistral_model.py`) and their 15 tests run on Python 3.11+.
The Workflows edition needs **Python 3.12–3.14**: `mistralai-workflows` 3.15 declares `Requires-Python >=3.12,<3.15`
(checked 2026-09-26, verify). On 3.11, `pip install -r requirements.txt` skips it, `tests/test_workflow.py` skips with
that reason, and `python demo.py workflow` exits with the same message. The workflow tests and demo also download a
Temporal dev server on first use, so they need network access to `temporal.download`.

```bash
pip install -r requirements.txt       # mistralai, pytest, and on Python 3.12+ mistralai-workflows[mistralai] (the core needs nothing)

python demo.py            # stdlib core: happy path · crash after the charge + retry · human gate · slow tool
python demo.py kill       # charges the card, then dies (exit 137) before the checkpoint
python demo.py resume     # a *different process* finds the run in runs.json and finishes it — one charge
python demo.py workflow   # the same agent on Mistral Workflows, on a local Temporal dev server (auto-downloaded)
python demo.py live       # durable.py with a real Mistral model deciding — export MISTRAL_API_KEY first

python -m pytest tests -q # 19 tests: 10 core · 5 adapter (offline, fake client) · 4 workflow (Python 3.12+, local dev server, ~10 s each)
jupyter lab notebooks/    # 01_worked.ipynb (executed) · 02_practice.ipynb (6 graded exercises); their workflow sections need Python 3.12+
```

## The Mistral setup, piece by piece

Product facts in this table (APIs, SDK names, preview status, limits) are as of 2026-09-26 (verify).

| Need | Mistral piece | In this repo |
|---|---|---|
| A model that picks the next step | Chat Completions with function calling via the `mistralai` SDK (`from mistralai.client import Mistral`; `chat.complete(..., tools=..., tool_choice="auto")`). Aliases: `mistral-medium-latest` (default here), `mistral-small-latest` for cheap steps, `mistral-large-latest`. Tool-call ids must be 9 alphanumeric characters. | `mistral_model.py` |
| Durable orchestration: state, retries, waits, timeouts | **Mistral Workflows** (Studio; public preview since Apr 2026), built on Temporal. Hybrid mode: Mistral hosts the orchestrator, your workers run in your infrastructure and connect outbound; enterprise can self-host the orchestrator. Python SDK `mistralai-workflows`: `@workflows.activity(start_to_close_timeout, retry_policy_max_attempts, heartbeat_timeout)`, `@workflows.workflow.define(name, execution_timeout)`, `@workflow.signal`, `workflow.wait_condition`, schedules, child workflows, 2 MB payload limit with offloading to your blob storage, SDK-side encryption. | `mistral_workflow.py`, `local_temporal.py` |
| An agent loop *inside* a workflow, with server-side tools | Durable Agents plugin (`mistralai-workflows[mistralai]`): `Agent(model, instructions, tools=[activities], handoffs=[...])`, `Runner.run(agent, inputs, session, max_turns)`, `RemoteSession` (Agents API) or `LocalSession` (chat completions; on-prem). Built-in tools run server-side: web search, code interpreter, image generation, document library; MCP servers via stdio / SSE / streamable HTTP. | the step-up from `mistral_workflow.py`'s hand-written loop |
| Server-side conversation state | Agents API: `beta.conversations.start(agent_id, inputs)` → a `function.call` entry with a `tool_call_id` is the pending intent; `beta.conversations.append(conversation_id, inputs=[FunctionResultEntry(tool_call_id, result)])` is the resume. | described in notebook 01 §8 |
| Human input | `wait_condition` + signal (above), or `InteractiveWorkflow.wait_for_input()` for chat-style forms/confirmations | `mistral_workflow.py` |
| Scheduling | `schedules=[...]` on `workflow.define` (calendar/interval, overlap policy) | — |
| Observability | execution event history and OpenTelemetry traces in Studio Observability; `execution_timeout` up to months | — |
| Deployment | hosted (La Plateforme, EU/US regional endpoints), dedicated, hybrid, or self-hosted open-weight models on your GPUs (vLLM) — the adapter's client is the only line that changes | — |

Verified on 19 Sep 2026 against docs.mistral.ai (Workflows overview, core concepts, activities, waiting for conditions, durable agents) and SDK introspection (`mistralai` 2.10.1, `mistralai-workflows` 3.15.0). Re-check model aliases and the preview status of Workflows before relying on them.

## What was verified, and what wasn't

* **Verified locally:** all 19 tests; `demo.py workflow` end to end on a Temporal dev server (approval signal → one charge with key `execution_id:1`; worker dies after charging → the retried activity converges to one charge and the model is not re-asked; rejection fed back; 2-second approval timeout → `approval expired`; budget stop); both notebooks (practice fails only at its TODOs and passes when filled with the reference code).
* **Not verified:** a live model call (`python demo.py live`, needs `MISTRAL_API_KEY`) and running the worker against Mistral's hosted orchestrator (`workflows.run_worker([InvoiceAgent])`).

## Where the GCP editions went

`long-running-agents-core` is the provider-neutral original of `durable.py`; `long-running-agents-gcp` is the full step-up (Cloud Run + Cloud Tasks + Firestore, fan-out/fan-in, sagas, ADK 2). Same five rules everywhere — Mistral's difference is that the orchestration layer is part of the product.
