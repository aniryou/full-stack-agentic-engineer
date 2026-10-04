# deploy/gcp — consolidation as a Cloud Run job on Cloud Scheduler, and where the model and the store live (T3)

**Tier:** T3 (Google Cloud, optional). **Everything here is printed, not run by the lab**: notebook 04 and
`python -m memlab gcp-commands --project <id>` print the commands, and you run them. This lab has no Terraform.
The GPU and Cloud Run infrastructure already has Terraform in the 04 serving lab.

```
Cloud Scheduler "memlab-consolidate-weekly"  (cron 17 3 * * 1, UTC: Mondays)
   │  POST https://run.googleapis.com/v2/projects/<p>/locations/<r>/jobs/memlab-consolidate:run
   │  OAuth access token of memlab-scheduler@<p>.iam.gserviceaccount.com (roles/run.invoker on the job)
   ▼
Cloud Run job "memlab-consolidate"  (--tasks N; each task takes the users whose hash lands on its index)
   │  runs as memlab-job@<p>: roles/secretmanager.secretAccessor on memlab-pg-dsn, roles/cloudsql.client
   │  python -m memlab consolidate   → the previous ISO week, Monday 00:00 to Monday 00:00 UTC
   │  one durable run per (tenant, user, ISO week): lease row, checkpoints, idempotent writes
   ▼  --set-cloudsql-instances <p>:<r>:memlab-pg  (socket at /cloudsql/<p>:<r>:memlab-pg)
Postgres + pgvector (Cloud SQL, verify) ◀── the job's tables; a memory service on Postgres is yours to port
   ▲                                        (memlab's service runs on SQLite only, see §2)
   │  the agent's model: vLLM on Cloud Run with one L4 (the 04 serving lab), tool calling on
```

**Weekly, on purpose.** There is one run id per (tenant, user, ISO week). Thus the trigger is weekly too. If a
daily trigger has a weekly id, six of seven starts are no-ops ("run already done"). Also, the facts arrive up to a
week late, but the trigger calls itself daily.

The data that the next turn needs comes from extraction after the turn (PRIMER §2), not from this job. A daily job
needs a per-day run id and a one-day window (`--window-days 1` and an id by date).

## 1. The model: reuse the serving lab's Cloud Run GPU service

[`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/)
deploys `vllm/vllm-openai:v0.30.0` on one L4 with scale to zero (Terraform or `deploy.sh`). For this lab, make
these changes:

| Setting | Value | Why |
|---|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct` | It can make tool calls. It has 3.1 GB of BF16 weights, on a 24 GB L4. |
| extra vLLM args | `--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details` | They give tool calls, and `cached_tokens` for each request, which notebook 03 uses. |
| `max_instances` | 1 | This is a lab, not a fleet. |

Then run `gcloud run services proxy <service> --region <r> --port 8000 &`. After that, run
`export MEMLAB_LLM_URL=http://127.0.0.1:8000`. The README of the serving lab has the details and the cold-start
arithmetic.

## 2. The store: Postgres + pgvector, never a SQLite file on an instance

The disk of a Cloud Run instance is temporary, and SQLite has no network protocol. Thus on GCP, the store is
Postgres with the `vector` extension. Cloud SQL for PostgreSQL and AlloyDB both give pgvector (verify the versions
against 0.8.6, because the SQL of this lab is for 0.8.6).

`memlab.store.pgvector` holds every statement. The tests
examine each statement offline with the parser of Postgres. These statements include the lease and checkpoint
tables of the job. The consolidation job runs on it (`python -m memlab consolidate --pg-dsn` or `MEMLAB_PG_DSN`).

**The memory service (`memlab serve`) is SQLite-only**, because its idempotency table uses SQLite SQL. Thus on GCP,
this lab deploys only the job. The service needs a Postgres port of its own before it can share these tables.

The job connects to the instance through the Unix socket of the Cloud SQL connector. `--set-cloudsql-instances
<project>:<region>:<instance>` mounts `/cloudsql/<project>:<region>:<instance>`, and the DSN is:

```
host=/cloudsql/<project>:<region>:memlab-pg dbname=memlab user=memlab password=...
```

(verify the flag on `gcloud run jobs create`.) A private-IP instance needs Direct VPC egress, `--network/--subnet`,
and a `host=<private IP>` DSN instead. Export the DSN as `MEMLAB_PG_DSN` in your shell. The printed commands pipe it
into Secret Manager as `memlab-pg-dsn`, and they do not write it to a file. The job then reads it back as
`MEMLAB_PG_DSN`.

## 3. The job and its schedule

```bash
python -m memlab gcp-commands --project my-project --region us-central1 --tasks 4
```

This command prints the commands in the list, in this order. Run them from `memory-lab/`, which is the build
context of the image.

1. **the image**: the commands make an Artifact Registry repository `memlab` and run `gcloud auth configure-docker`.
   Then they run `docker build -f
   deploy/local/Dockerfile` and `docker push` of `memlab/memlab:0.1.0` (Docker on your machine, or Cloud Build).
2. **the job's identity**: a service account `memlab-job`, never the default compute account. The commands make the
   secret `memlab-pg-dsn` from `$MEMLAB_PG_DSN`. They give the account `roles/secretmanager.secretAccessor` on that
   secret and `roles/cloudsql.client` on the project.
3. **the job**: the commands run `gcloud run jobs create` with `--service-account memlab-job@…`,
   `--set-cloudsql-instances`, `--tasks`, `--max-retries 3`, `--set-secrets MEMLAB_PG_DSN=memlab-pg-dsn:latest` and
   the command.
4. **the trigger**: the commands make a service account `memlab-scheduler` and give it `roles/run.invoker` on the
   job. Then they run `gcloud scheduler
   jobs create http memlab-consolidate-weekly` with `--oauth-service-account-email`. Last, a manual
   `gcloud run jobs execute --wait` tries the job one time.

Why **OAuth, not OIDC**: the scheduler calls the Cloud Run Admin API (`run.googleapis.com`), and this API accepts
OAuth access tokens. OIDC identity tokens are for calls to the URL of your own service. The lra-gcp reaper makes
such calls. See
[§3.3 of its primer](../../../../long-running-durable/lra-gcp/docs/primer.md#33-leases-and-the-reaper-crash-recovery) and
the `oidc_token` of `google_cloud_scheduler_job.reaper` in its
[Terraform](../../../../long-running-durable/lra-gcp/infra/terraform/main.tf).

The Terraform sample of Google for scheduled jobs uses this v2 URI. The gcloud samples of Google use the v1 form
`https://<region>-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/<project>/jobs/<job>:run` (verify
which form your gcloud prefers). Also verify `--task-timeout`, `--set-secrets` and `--command/--args` on
`gcloud run jobs create`. Also verify if `roles/run.invoker` on the job is sufficient to run it (the sample of
Google gives `roles/run.developer` on the project).

If the schedule starts the job two times, no damage occurs. Both executions calculate the same window (the
previous ISO week) and the same run id for each user. One execution takes the lease. The other execution finds the
lease held or the run done (notebook 04).

## Managed memory stores (dated 2026-09-26; verify before choosing)

| Service | What its SDK shows | Scope | Deletion / TTL |
|---|---|---|---|
| Vertex AI Agent Engine Memory Bank | ADK's `VertexAiMemoryBankService`: `memories.generate` / `ingest_events` / `create`, `retrieve` with a similarity search | `{app_name, user_id}` | It keeps revisions with `revision_ttl`. It has a consolidation toggle (verify quotas, regions, price). |
| Amazon Bedrock AgentCore Memory | strategies `semanticMemoryStrategy`, `summaryMemoryStrategy`, `userPreferenceMemoryStrategy`, `episodicMemoryStrategy`, `customMemoryStrategy` | for each memory resource and actor (verify) | short-term events expire after 90 days by default (`event_expiry_days`) |
| Anthropic memory tool | The tool type is `memory_20250818`. The commands are `view`, `create`, `str_replace`, `insert`, `delete`, `rename`. | the **client** stores the files | what your storage does |
| mem0 Platform · Zep Cloud · Letta Cloud | mem0 2.x is ADD-only (`/v3/memories/add/`). The open-source server of Zep moved to `legacy/`, and Graphiti is the engine of Zep. The server of Letta moved to `letta-code`. | vendor-defined | Examine what "delete" removes. The open-source `delete` of mem0 keeps the text in its SQLite history table. |

Ask each of these services the questions of this lab:

- Does it take the scope from a verified principal?
- Does it apply screening to writes, and does it keep provenance?
- Can a forget reach derived facts, indexes and backups?
- Can you prove it?

## Cost

This README gives no dated prices. See [`COMPUTE.md`](../../../../../COMPUTE.md) and the pricing pages of Cloud
Run, Cloud Scheduler and Cloud SQL (verify). The shape of the cost:

- The job runs for minutes per night on CPU. The bill is per task-second.
- The scheduler is a few jobs per month.
- Cloud SQL bills per hour while the instance exists, also when it is idle. For a lab, this is the largest line.
- The L4 service bills per second while an instance is up, and it scales to zero.

## Cleanup

`python -m memlab gcp-commands` prints the same list after the setup commands:

```bash
gcloud scheduler jobs delete memlab-consolidate-weekly --project my-project --location us-central1 --quiet
gcloud run jobs delete memlab-consolidate --project my-project --region us-central1 --quiet
gcloud iam service-accounts delete memlab-scheduler@my-project.iam.gserviceaccount.com --project my-project --quiet
gcloud projects remove-iam-policy-binding my-project --member serviceAccount:memlab-job@my-project.iam.gserviceaccount.com --role roles/cloudsql.client --quiet
gcloud secrets delete memlab-pg-dsn --project my-project --quiet
gcloud iam service-accounts delete memlab-job@my-project.iam.gserviceaccount.com --project my-project --quiet
gcloud artifacts repositories delete memlab --project my-project --location us-central1 --quiet
# the Cloud SQL instance and the serving lab's Cloud Run service: see their own cleanup steps
```

When you delete the Cloud SQL instance, you delete the memories. Its automated backups obey the backup retention
setting of the instance (verify). This is the "backups" line of the deletion checklist in notebook 05.
