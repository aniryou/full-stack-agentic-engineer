# GCP cheat sheet for long-running agents

Numbers as verified on 5 Sep 2026 from the linked docs. Quote the semantics; re-check the digits before relying on them.

## Delivery semantics (the thing to say first)

| Mechanism | Delivery | What that forces on you |
|---|---|---|
| Cloud Tasks | at-least-once | idempotent handler; named tasks to collapse double enqueues |
| Pub/Sub push/pull | at-least-once (exactly-once only on pull subscriptions, regional) | idempotent consumer; dead-letter topic; ack within deadline |
| Cloud Scheduler | at-least-once (may double-fire) | due-time check + lease |
| Cloud Workflows retries | re-run the step | idempotent HTTP targets |
| Cloud Run request retry | client-dependent | return 2xx only after durable commit |
| ADK `ResumabilityConfig` | at-least-once on resume | idempotent tools; no `temp:` reliance |

## Cloud Run
* Services: request timeout default 300 s, **max 3600 s** (`--timeout`). Container keeps running after a 504; track the deadline and return early.
* Jobs: `--task-timeout` default 10 min, **up to 168 h** (GPU tasks: 1 h); `--tasks N --parallelism P`; per-task retries; `gcloud run jobs execute JOB --update-env-vars=RUN_ID=…` to pass parameters.
* Deploy pattern: `gcloud run deploy agent --source . --no-allow-unauthenticated --service-account=$RUN_SA --set-env-vars=...`
* Docs: cloud.google.com/run/docs/configuring/request-timeout · /run/docs/configuring/task-timeout

## Cloud Tasks
* Named task ⇒ de-dup: `ALREADY_EXISTS` if the name was used recently (docs quote ~1 h for API-created queues, up to 24 h on the quotas page; use hashed, non-sequential ids).
* `schedule_time` up to **30 days** ahead; retention 31 days; 500 dispatches/s/queue; task ≤ 1 MiB; batch create ≤ 100.
* Queue config: `--max-dispatches-per-second`, `--max-concurrent-dispatches`, `--max-attempts`, `--min-backoff`, `--max-backoff`, `--max-doublings`.
* HTTP target auth: `oidc_token{service_account_email, audience}`; handler checks `X-CloudTasks-TaskRetryCount` for observability.
* Python (see `core/transport.py`): `tasks_v2.CloudTasksClient().create_task(request=CreateTaskRequest(parent=queue_path, task=Task(name=task_path(...), http_request=HttpRequest(...), schedule_time=..., dispatch_deadline=...)))`.
* Docs: cloud.google.com/tasks/docs/quotas · /tasks/docs/common-pitfalls

## Pub/Sub
* Retention up to 31 days (default 7); ack deadline 10–600 s; message ≤ 10 MB; ordering keys (per-key throughput limit); filtering by attributes.
* Push subscription → Cloud Run with `--push-auth-service-account`; the endpoint must respond within the ack deadline — hand long work to Cloud Tasks.
* Dead-letter: `--dead-letter-topic --max-delivery-attempts 5..100`.
* Exactly-once delivery: pull subscriptions only; not for push.

## Cloud Scheduler
* `gcloud scheduler jobs create pubsub tick --schedule="*/5 * * * *" --topic=agent-ticks --message-body='{"kind":"tick"}' --time-zone=Asia/Singapore`
* Or HTTP target with `--oidc-service-account-email`. Pause/resume jobs; a paused job refuses `jobs run`.

## Cloud Workflows
* Execution max duration **1 year**; `events.await_callback` default timeout 43 200 s (12 h) — set it explicitly; callbacks are idempotent.
* `parallel` with `shared:` variables and `concurrency_limit`; `for … in`; `retry: predicate: ${http.default_retry_predicate}` with `backoff`; `sys.sleep` for polling loops (each step is billed — prefer callbacks).
* Connectors poll GCP long-running operations for you (timeout default 1800 s, up to 1 year for LROs).
* Quotas: 10 000 concurrent executions per region (backlog beyond), 10 000 workflows per project.
* Docs: cloud.google.com/workflows/docs/creating-callback-endpoints · /workflows/quotas

## Firestore
* Transactions: `@firestore.transactional`, optimistic, retried on contention; ≤ 500 writes per transaction; keep external side effects out.
* Document ≤ 1 MiB; keep sustained writes per document ≈ 1/s; use subcollections for big journals; TTL policies for idempotency keys.
* Emulator: `gcloud emulators firestore start --host-port=localhost:8080` + `FIRESTORE_EMULATOR_HOST`.

## ADK 2 (Python, `google-adk` 2.x)
```python
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.tools import LongRunningFunctionTool
from google.adk.workflow import START, Workflow, node
from google.adk.events.request_input import RequestInput

app = App(name="x", root_agent=root, resumability_config=ResumabilityConfig(is_resumable=True),
          events_compaction_config=EventsCompactionConfig(compaction_interval=10, overlap_size=1))

@node(rerun_on_resume=True)          # re-check the world on every wake-up
def check(ctx):
    if not ready(): return RequestInput(interrupt_id="wake", message="not yet")
    ctx.route = "ready"

@node(rerun_on_resume=False)         # side effect: never repeat on resume
def act(ctx): ...

wf = Workflow(name="w", edges=[(START, check, {"ready": act, "wait": park})])
# resume: runner.run_async(user_id, session_id, invocation_id=inv, new_message=Content(parts=[Part(function_response=FunctionResponse(id="wake", name="adk_request_input", response={...}))]))
```
* Session stores by URI: `sqlite+aiosqlite:///sessions.db`, `postgresql+asyncpg://user:pw@/db?host=/cloudsql/PROJ:REGION:INST`, `agentengine://RESOURCE_ID`; artifacts `file://`, `gs://`; memory `rag://`, `agentengine://`.
* State prefixes: `user:` (cross-session), `app:` (all users), `temp:` (one invocation), none (session).
* `get_fast_api_app(agents_dir=..., web=False, trigger_sources=["pubsub"], trace_to_cloud=True)`; the built-in Pub/Sub trigger creates a new session per message — add your own `/wake` to resume an existing one.
* `runner.rewind_async(user_id=..., session_id=..., rewind_before_invocation_id=...)` rolls state back before a bad decision.
* Deploy to Agent Runtime: `vertexai.Client(project, location).agent_engines.create(agent_engine=AdkApp(agent=root), config={"staging_bucket": "gs://…", "requirements": ["google-cloud-aiplatform[agent_engines,adk]"]})`; sessions via `VertexAiSessionService(project, location, agent_engine_id)`; memory via `VertexAiMemoryBankService(...)`.

## Gemini via google-genai (Vertex, ADC)
```python
from google import genai
from google.genai import types
client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
resp = client.models.generate_content(model="gemini-3.8-flash", contents=[...],
        config=types.GenerateContentConfig(system_instruction=..., tools=[types.Tool(function_declarations=[...])],
                                           response_mime_type="application/json"))
```
Models as of Sep 2026: Gemini 3.8 Flash (workhorse), 3.1 Pro (flagship), 3.5 Flash-Lite (cheapest); `-latest` aliases are AI Studio only — on Vertex use `gemini-latest-flash`-style aliases or pin ids.

## IAM sketch
* `sa-agent-run` (Cloud Run runtime): `roles/datastore.user`, `roles/cloudtasks.enqueuer`, `roles/pubsub.publisher`, `roles/aiplatform.user`, `roles/secretmanager.secretAccessor`.
* `sa-tasks-invoker` (Cloud Tasks / Pub/Sub push identity): `roles/run.invoker` on the service only.
* `sa-workflows`: `roles/run.invoker`; workflows call Cloud Run with `auth: type: OIDC`.
* Cloud Run: `--no-allow-unauthenticated`; human endpoints behind IAP.
