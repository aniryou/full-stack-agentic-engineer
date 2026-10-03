# lra-gcp — a durable-execution engine for long-running agents, run locally and on Google Cloud

After this lab, you can run a long-running agent from start to end, break it and repair it. The agent and its engine have these items:

- child runs that fan out and fan in
- a human gate that waits for days
- sagas that compensate their own steps
- budgets that fail closed
- scheduled ticks
- slow tools
- a loop in which the model selects the next tool
- the same engine on Firestore, Cloud Tasks, Pub/Sub and Cloud Run

## Start here

1. Read the topic primer [`../PRIMER.md`](../PRIMER.md) §2–§4 (40 min). Then read the design notes of this lab,
   [`docs/primer.md`](docs/primer.md) §2–§3. These two sections of the design notes give the three invariants of the engine and its list of patterns.
2. Run `python3 -m pip install -e ".[dev,services]" && python3 scripts/local_demo.py`. The demo runs in a few seconds.
   It describes these stages: a fan-out, a crash, the reaper, a 3-day wait, an approval, a saga rollback.
3. Open [`notebooks/00_core_idea.ipynb`](notebooks/00_core_idea.ipynb). Do the notebooks from 00 to 05 in sequence.
   The worked answers are in [`solutions/`](solutions/), with the same names.

## What you get

*T0 is a laptop or a Colab CPU, at no cost. It uses in-memory adapters with the same semantics, a scripted model, no
key and no cloud project. T3 is the Google Cloud deployment (Terraform in `infra/terraform/`). It is optional, and
Google Cloud bills it per use.*

Time: after [`lra-core`](../lra-core/README.md), notebooks 00–03 and 05 and the docs
take about 6 h. With the optional ADK path and a read of the deploy, the total is 8 h. These times are rough. Module 07.3
in [`CURRICULUM.md`](../../../CURRICULUM.md) gives 10 h to the primer, the core and this lab. Each blank in
`notebooks/` stops at its first exercise until you write your answer in it.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| `00_core_idea` | make the store, the named-task queue and the worker yourself, with no code to start from. Then break them with a crash and a duplicate delivery. | 45 min | T0 |
| `01_durable_execution` | operate the engine one task at a time, with explicit retries, chaos hooks at each crash window and leases between two workers. | 1 h | T0 |
| `02_human_in_the_loop` | make a run wait for days and resume it by key as an idempotent operation. You can also reject the approval, let the reaper apply the gate timeout, auto-approve, or cancel the run. | 45 min | T0 |
| `03_fanout_saga_reflection` | spawn child runs in a fan-out, handle a partial failure as data and write the fan-in counter. Compensate a saga in reverse sequence and set a limit on a reflection loop. Fail closed on the budget and on the deadline. | 1 h | T0 |
| `04_adk_workflow` | run the same shapes on the `Workflow` of ADK 2 (interrupts, resume, routing, the staleness guard, the new-invocation mistake), then make the graph yourself. | 45 min | T0 with the `adk` extra |
| `05_tool_loop_and_mistral` | write the decision of a model to the journal before the loop acts on it, and approve exactly the action that runs. Send a call to a tool that does not exist, or a tool error, back to the model as an observation. Then replace the model with Mistral (function calling, the `LLM` port) and see the loop on Mistral Workflows. | 1 h | T0 |
| [`docs/`](docs/) | [`primer.md`](docs/primer.md): the design notes of the engine, with 13 patterns and their GCP mappings, a reference architecture, scale and cost, security, tests. [`code-evaluation-drills.md`](docs/code-evaluation-drills.md). [`runbook.md`](docs/runbook.md). [`gcp-cheatsheet.md`](docs/gcp-cheatsheet.md): delivery semantics, limits, short CLI and SDK examples, IAM. [`mistral.md`](docs/mistral.md). | 2 h | T0 |
| `src/lra/` | The package has `core/` (engine, models, workflow DSL, ports) and three adapters: `adapters/memory`, `adapters/gcp` (Firestore, Cloud Tasks, Pub/Sub, Gemini) and `adapters/mistral` (optional). It also has `patterns/` (hitl, reflection, orchestrator_worker, saga, scheduled, async_tool) and `examples/` (research pipeline, procurement saga, tool agent). | — | T0 |
| `services/`, `workflows/`, `infra/terraform/` | the Cloud Run `api` and `worker`, the same flow in Cloud Workflows (`parallel`, callbacks, compensation), Terraform for all of these. | 1 h to read, deploy ~15 min | T3 |
| `examples/adk_agent_engine/`, `examples/adk_ticket_queue/` | ADK 2 `Workflow` graphs, an agent that uses a model and has a long-running tool, a Cloud Run `/wake` endpoint, an Agent Engine deploy. | 1 h | T0 with the `adk` extra, T3 |

## Run it

```bash
python3 -m pip install -e ".[dev,services]"
python3 -m pytest -q          # 73 tests: 66 pass, 7 skip (Google Cloud clients, ADK 2, Mistral Workflows), ~15 s
python3 scripts/local_demo.py
make notebooks                # the solutions run clean; each blank stops at its first exercise
```

The optional extras add paths. They never add requirements:

- `pip install -e ".[dev,services,gcp]"` runs the four tests of the Google Cloud adapters against fake clients. These
  tests need no credentials.
- `".[adk]"` adds ADK 2 and the Google Cloud clients (about 220 MB, measured 2026-09-26, verify). It runs the two ADK
  test files, notebook 04 and `make adk-demo`.
- `".[mistral]"` adds the `mistralai` SDK. On Python 3.12–3.14, it also adds Mistral Workflows (see
  [`docs/mistral.md`](docs/mistral.md)). The five tests of Mistral Workflows start a local Temporal dev server. When the
  tests use the server for the first time, they download it from `temporal.download`.

With all the extras (`gcp`, `adk`, `mistral`) on Python 3.12, and with access to that download, 81 tests run.

## The engine in one picture

```
client ─▶ Cloud Run api ─▶ Firestore agent_runs (checkpoints, leases, version)
                │ enqueue (run, step, attempt)              ▲
                ▼                                            │ CAS
          Cloud Tasks ──OIDC──▶ Cloud Run worker ──▶ Gemini / tools
                ▲                     │ progress ─▶ Pub/Sub agent-events ─▶ BigQuery / UI / DLQ
        Cloud Scheduler ─ reap ───────┘
```

The engine has three invariants (primer §2):

- **The checkpoint comes before the enqueue.**
- **`(run, step, attempt)` identifies the work.**
- **The engine records each effect before the checkpoint, with the intent as its key.**

The tests in `tests/test_durability.py` crash the worker at each crash window. They make sure that exactly one recovery
occurs and that no side effect occurs more than one time.

## Writing a workflow

```python
from lra import Workflow, Next, Done, Wait, Budget
from lra.patterns.hitl import request_approval, approval_decision

wf = Workflow("expense", version="1", default_budget=Budget(max_steps=20, max_cost_usd=1.0))

@wf.step(start=True)
def draft(ctx):
    ctx.state["summary"] = ctx.llm("Summarise: " + ctx.input["text"]).text     # budgeted model call
    return Next("gate")

@wf.step()
def gate(ctx):
    return request_approval(ctx, then="pay", gate="manager", timeout=timedelta(days=3))   # sleeps for free

@wf.step(compensate=lambda ctx: ctx.effect("refund", refund_fn))
def pay(ctx):
    approval_decision(ctx, gate="manager")                                          # raises on reject/timeout
    rec = ctx.effect("pay", lambda: payments.charge(...))                           # at most once, ever
    return Done({"tx": rec["id"]})
```

Add the workflow to `lra.examples.ALL_WORKFLOWS`, or give it to `Engine(workflows=[...])`. Then send
`POST /runs {"workflow": "expense", ...}`.

## Deploying to GCP

```bash
export PROJECT_ID=... REGION=asia-southeast1
./scripts/deploy.sh                       # Cloud Build image + terraform apply (see infra/terraform/variables.tf)
gcloud workflows run research-approval --data='{"goal":"...","subtopics":["a","b"]}'
```

Set `LRA_BACKEND=gcp` and the variables in `.env.example`. The two services read these variables. Set
`LRA_GEMINI_MODEL` to one specific model that is turned on in your project. Set real prices in `GeminiLLM` before you
trust the values of `cost_usd`.

## Caveats

- The engine, the adapters, the services, the patterns, the examples and the notebooks have local tests. The Google
  Cloud adapters have unit tests against fake clients. The Terraform passes validation, but no one applied it
  to a real project during the work on this lab. Before you run `terraform apply`, examine the names, the quotas and the
  org policies.
- The ADK and Agent Engine code had offline checks. The ticket-queue graph and its grader had a check against
  `google-adk` 2.10, and the Agent Engine example had a check against 2.8. The deploy surface of Agent Engine changes between
  releases. Thus, compare `AdkApp` and `agent_engines.create` with the current docs (verify).
  `examples/adk_ticket_queue/main.py` and `agent.py` use the ADK docs as their reference. No one ran them live during
  the work on this lab.
- The notebooks show simulated costs. `FakeLLM`, `GeminiLLM` and `MistralLLM` use illustrative prices for each
  token. Set real prices in place of these prices before you trust the values of `cost_usd`.
