# GCP cheat sheet for long-running agents

This sheet gives the limits, the delivery semantics, the CLI and SDK snippets and the IAM roles that §5 of the topic [`PRIMER.md`](../../PRIMER.md) uses. On 5 Sep 2026, we made sure that the numbers agreed with the linked docs (verify). Quote the semantics. Before you rely on the numbers, examine them again.

## Delivery semantics (the thing to say first)

| Mechanism | Delivery | What that forces on you |
|---|---|---|
| Cloud Tasks | at-least-once | An idempotent handler. Named tasks, so that two enqueues of the same task become one task. |
| Pub/Sub push/pull | at-least-once (exactly-once only on pull subscriptions, regional) | An idempotent consumer, a dead-letter topic, an ack within the deadline. |
| Cloud Scheduler | at-least-once (it can send the tick two times) | A check of the due time, and a lease. |
| Cloud Workflows retries | the step runs again | Idempotent HTTP targets. |
| Cloud Run request retry | the client decides | Return 2xx only after a durable commit. |
| ADK `ResumabilityConfig` | at-least-once on a resume | Idempotent tools. Do not depend on `temp:`. |

## Cloud Run
* Services: the request timeout has a default of 300 s and a **maximum of 3600 s** (`--timeout`). After a 504, the container continues to run. Monitor the deadline and return early.
* Jobs: `--task-timeout` has a default of 10 min and can be **up to 168 h** (for GPU tasks, 1 h). Set the tasks and the parallelism with `--tasks N --parallelism P`. Each task has its own retries. To pass parameters, use `gcloud run jobs execute JOB --update-env-vars=RUN_ID=…`.
* The deploy pattern: `gcloud run deploy agent --source . --no-allow-unauthenticated --service-account=$RUN_SA --set-env-vars=...`
* Docs: cloud.google.com/run/docs/configuring/request-timeout · /run/docs/configuring/task-timeout

## Cloud Tasks
* A named task removes duplicates. If a task used the name recently, Cloud Tasks returns `ALREADY_EXISTS`. The docs give approximately 1 h for queues that the API created, and up to 24 h on the quotas page. Use hashed ids that are not sequential.
* `schedule_time` can be up to **30 days** in the future. The retention is 31 days. The limit is 500 dispatches/s/queue. A task is ≤ 1 MiB. A batch create has ≤ 100 tasks.
* The queue configuration: `--max-dispatches-per-second`, `--max-concurrent-dispatches`, `--max-attempts`, `--min-backoff`, `--max-backoff`, `--max-doublings`.
* HTTP target auth: `oidc_token{service_account_email, audience}`. For observability, the handler examines `X-CloudTasks-TaskRetryCount`.
* Python (refer to `core/transport.py`): `tasks_v2.CloudTasksClient().create_task(request=CreateTaskRequest(parent=queue_path, task=Task(name=task_path(...), http_request=HttpRequest(...), schedule_time=..., dispatch_deadline=...)))`.
* Docs: cloud.google.com/tasks/docs/quotas · /tasks/docs/common-pitfalls

## Pub/Sub
* The retention is up to 31 days (the default is 7). The ack deadline is 10–600 s. A message is ≤ 10 MB. Ordering keys are available, and each key has a throughput limit. You can filter messages by attributes.
* A push subscription sends messages to Cloud Run with `--push-auth-service-account`. The endpoint must respond within the ack deadline. Give long work to Cloud Tasks.
* For a dead-letter topic, use `--dead-letter-topic --max-delivery-attempts 5..100`.
* Exactly-once delivery is available only on pull subscriptions. It is not available for push.

## Cloud Scheduler
* `gcloud scheduler jobs create pubsub tick --schedule="*/5 * * * *" --topic=agent-ticks --message-body='{"kind":"tick"}' --time-zone=Asia/Singapore`
* Or use an HTTP target with `--oidc-service-account-email`. You can pause and resume jobs. A paused job does not accept `jobs run`.

## Cloud Workflows
* The maximum duration of an execution is **1 year**. The default timeout of `events.await_callback` is 43 200 s (12 h). Set this timeout explicitly. Callbacks are idempotent.
* Use `parallel` with `shared:` variables and `concurrency_limit`. Use `for … in`. Use `retry: predicate: ${http.default_retry_predicate}` with `backoff`. Use `sys.sleep` for poll loops. But Cloud Workflows bills each step, so use callbacks in place of poll loops when you can.
* Connectors poll GCP long-running operations for you. The default timeout is 1800 s, and for LROs it can be up to 1 year.
* Quotas: 10 000 concurrent executions per region (more executions go into a backlog), and 10 000 workflows per project.
* Docs: cloud.google.com/workflows/docs/creating-callback-endpoints · /workflows/quotas

## Firestore
* Transactions: `@firestore.transactional`. A transaction is optimistic, and Firestore retries it on contention. A transaction has ≤ 500 writes. Keep external side effects out of a transaction.
* A document is ≤ 1 MiB. Keep the sustained writes to one document at ≈ 1/s. Use subcollections for large journals. Use TTL policies for idempotency keys.
* Emulator: `gcloud emulators firestore start --host-port=localhost:8080` and `FIRESTORE_EMULATOR_HOST`.

## ADK 2 (Python, `google-adk` 2.x)
```python
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.tools import LongRunningFunctionTool
from google.adk.workflow import START, Workflow, node
from google.adk.events.request_input import RequestInput

app = App(name="x", root_agent=root, resumability_config=ResumabilityConfig(is_resumable=True),
          events_compaction_config=EventsCompactionConfig(compaction_interval=10, overlap_size=1))

@node(rerun_on_resume=True)          # this node interrupts: re-run it (re-check the world) on every wake-up;
def check(ctx):                      # False would mark it done and take the resume input as its output
    if not ready(): return RequestInput(interrupt_id="wake", message="not yet")
    ctx.route = "ready"

@node                                # side effect: finished nodes are never replayed by a resume of the same
def act(ctx): ...                    # invocation (a new invocation replays them), and it carries an idempotency key

wf = Workflow(name="w", edges=[(START, check, {"ready": act, "wait": park})])
# resume: runner.run_async(user_id, session_id, invocation_id=inv, new_message=Content(parts=[Part(function_response=FunctionResponse(id="wake", name="adk_request_input", response={...}))]))
```
* Session stores by URI: `sqlite+aiosqlite:///sessions.db`, `postgresql+asyncpg://user:pw@/db?host=/cloudsql/PROJ:REGION:INST`, `agentengine://RESOURCE_ID`. Artifacts: `file://`, `gs://`. Memory: `rag://`, `agentengine://`.
* State prefixes: `user:` (across sessions), `app:` (all users), `temp:` (one invocation), none (session).
* `get_fast_api_app(agents_dir=..., web=False, trigger_sources=["pubsub"], trace_to_cloud=True)`. The built-in Pub/Sub trigger creates a new session for each message. To resume a session that exists, add your own `/wake`.
* `runner.rewind_async(user_id=..., session_id=..., rewind_before_invocation_id=...)` rolls the state back to a point before a bad decision.
* Deploy to Agent Runtime: `vertexai.Client(project, location).agent_engines.create(agent_engine=AdkApp(agent=root), config={"staging_bucket": "gs://…", "requirements": ["google-cloud-aiplatform[agent_engines,adk]"]})`. Sessions: `VertexAiSessionService(project, location, agent_engine_id)`. Memory: `VertexAiMemoryBankService(...)`.

## Gemini via google-genai (Vertex, ADC)
```python
from google import genai
from google.genai import types
client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
resp = client.models.generate_content(model="gemini-3.8-flash", contents=[...],
        config=types.GenerateContentConfig(system_instruction=..., tools=[types.Tool(function_declarations=[...])],
                                           response_mime_type="application/json"))
```
The models as of Sep 2026: Gemini 3.8 Flash (the model for most work), 3.1 Pro (the top model), 3.5 Flash-Lite (the lowest cost). The `-latest` aliases are available only in AI Studio. On Vertex, use aliases of the `gemini-latest-flash` type, or use exact ids.

## IAM sketch
* `sa-agent-run` (Cloud Run runtime): `roles/datastore.user`, `roles/cloudtasks.enqueuer`, `roles/pubsub.publisher`, `roles/aiplatform.user`, `roles/secretmanager.secretAccessor`.
* `sa-tasks-invoker` (the push identity of Cloud Tasks and Pub/Sub): `roles/run.invoker` on the service only.
* `sa-workflows`: `roles/run.invoker`. Workflows call Cloud Run with `auth: type: OIDC`.
* Cloud Run: `--no-allow-unauthenticated`. Put the endpoints that persons use behind IAP.
