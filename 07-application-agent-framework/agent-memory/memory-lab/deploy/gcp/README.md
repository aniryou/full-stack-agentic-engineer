# deploy/gcp — consolidation as a Cloud Run job on Cloud Scheduler, and where the model and the store live (T3)

**Tier:** T3 (Google Cloud, optional). **Everything here is printed, not run by the lab**: notebook 04 and
`python -m memlab gcp-commands --project <id>` print the commands; you run them. No Terraform in this lab —
the GPU and Cloud Run infrastructure already has Terraform in the 04 serving lab.

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

**Weekly, on purpose.** The run id is one per (tenant, user, ISO week), so the trigger is weekly too: a daily
trigger with a weekly id would make six of seven firings no-ops ("run already done") and land facts up to a week
late while calling itself daily. What the next turn needs comes from extraction after the turn (PRIMER §2), not from
this job. A daily job needs a per-day run id and a one-day window (`--window-days 1` and an id by date).

## 1. The model: reuse the serving lab's Cloud Run GPU service

[`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/)
deploys `vllm/vllm-openai:v0.30.0` on one L4 with scale to zero (Terraform or `deploy.sh`). For this lab change:

| Setting | Value | Why |
|---|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct` | follows tool calls; 3.1 GB of BF16 weights on a 24 GB L4 |
| extra vLLM args | `--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details` | tool calls, and `cached_tokens` per request for notebook 03 |
| `max_instances` | 1 | a lab, not a fleet |

Then `gcloud run services proxy <service> --region <r> --port 8000 &` and `export MEMLAB_LLM_URL=http://127.0.0.1:8000`
(the serving lab's README has the details and the cold-start arithmetic).

## 2. The store: Postgres + pgvector, never a SQLite file on an instance

A Cloud Run instance's disk is ephemeral and SQLite has no network protocol, so on GCP the store is
Postgres with the `vector` extension — Cloud SQL for PostgreSQL or AlloyDB both offer pgvector (verify the
versions against 0.8.6, which this lab's SQL is written for). `memlab.store.pgvector` holds every statement
(checked offline by Postgres's parser in the tests), including the job's lease and checkpoint tables; the
consolidation job runs on it (`python -m memlab consolidate --pg-dsn` or `MEMLAB_PG_DSN`). **The memory service
(`memlab serve`) is SQLite-only** — its idempotency table uses SQLite SQL — so on GCP the job is what this lab
deploys; the service needs a Postgres port of its own before it can share these tables.

The job reaches the instance through the Cloud SQL connector's Unix socket: `--set-cloudsql-instances
<project>:<region>:<instance>` mounts `/cloudsql/<project>:<region>:<instance>`, and the DSN is

```
host=/cloudsql/<project>:<region>:memlab-pg dbname=memlab user=memlab password=...
```

(verify the flag on `gcloud run jobs create`; a private-IP instance needs Direct VPC egress, `--network/--subnet`,
and a `host=<private IP>` DSN instead). Export it as `MEMLAB_PG_DSN` in your shell: the printed commands pipe it into
Secret Manager as `memlab-pg-dsn` without writing it to a file, and the job reads it back as `MEMLAB_PG_DSN`.

## 3. The job and its schedule

```bash
python -m memlab gcp-commands --project my-project --region us-central1 --tasks 4
```

prints, in order (run them from `memory-lab/`, the image's build context):

1. **the image** — an Artifact Registry repository `memlab`, `gcloud auth configure-docker`, then `docker build -f
   deploy/local/Dockerfile` and `docker push` of `memlab/memlab:0.1.0` (Docker on your machine; or Cloud Build);
2. **the job's identity** — a service account `memlab-job` (never the default compute account), the secret
   `memlab-pg-dsn` created from `$MEMLAB_PG_DSN`, `roles/secretmanager.secretAccessor` on that secret and
   `roles/cloudsql.client` on the project for it;
3. **the job** — `gcloud run jobs create` with `--service-account memlab-job@…`, `--set-cloudsql-instances`,
   `--tasks`, `--max-retries 3`, `--set-secrets MEMLAB_PG_DSN=memlab-pg-dsn:latest` and the command;
4. **the trigger** — a service account `memlab-scheduler`, `roles/run.invoker` on the job for it, `gcloud scheduler
   jobs create http memlab-consolidate-weekly` with `--oauth-service-account-email`, and a manual
   `gcloud run jobs execute --wait` to try it once.

Why **OAuth, not OIDC**: the scheduler calls the Cloud Run Admin API (`run.googleapis.com`), which accepts OAuth
access tokens; OIDC identity tokens are for invoking your own service's URL (the lra-gcp reaper does that,
see `07-application-agent-framework/long-running-durable/lra-gcp/docs/primer.md` §3.13). Google's Terraform
sample for scheduled jobs uses this v2 URI; its gcloud samples use the v1 form
`https://<region>-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/<project>/jobs/<job>:run` (verify
which your gcloud prefers). Also verify: `--task-timeout`, `--set-secrets` and `--command/--args` on
`gcloud run jobs create`, and whether `roles/run.invoker` on the job is enough to run it (Google's sample
grants `roles/run.developer` on the project).

A double-fired schedule is harmless: both executions compute the same window (the previous ISO week) and the same
run id per user; one takes the lease, the other finds it held or the run done (notebook 04).

## Managed memory stores (dated 2026-09-26; verify before choosing)

| Service | What its SDK shows | Scope | Deletion / TTL |
|---|---|---|---|
| Vertex AI Agent Engine Memory Bank | ADK's `VertexAiMemoryBankService`: `memories.generate` / `ingest_events` / `create`, `retrieve` with a similarity search | `{app_name, user_id}` | revisions with `revision_ttl`; consolidation toggle (verify quotas, regions, price) |
| Amazon Bedrock AgentCore Memory | strategies `semanticMemoryStrategy`, `summaryMemoryStrategy`, `userPreferenceMemoryStrategy`, `episodicMemoryStrategy`, `customMemoryStrategy` | per memory resource and actor (verify) | short-term events expire after 90 days by default (`event_expiry_days`) |
| Anthropic memory tool | tool type `memory_20250818`; commands `view`, `create`, `str_replace`, `insert`, `delete`, `rename` | the **client** stores the files | whatever your storage does |
| mem0 Platform · Zep Cloud · Letta Cloud | mem0 2.x is ADD-only (`/v3/memories/add/`); Zep's open-source server moved to `legacy/`, Graphiti is its engine; Letta's server moved to `letta-code` | vendor-defined | check what "delete" removes — mem0's open-source `delete` keeps the text in its SQLite history table |

The questions to ask any of them are this lab's: is scope taken from a verified principal, are writes
screened and provenance kept, can a forget reach derived facts, indexes and backups — and can you prove it?

## Cost

No prices are dated here: see [`COMPUTE.md`](../../../../../COMPUTE.md) and the Cloud Run, Cloud Scheduler
and Cloud SQL pricing pages (verify). The shape: the job runs minutes per night on CPU (billed per
task-second); the scheduler is a few jobs per month; Cloud SQL bills per hour while it exists even when
idle — the largest line for a lab; the L4 service bills per second while an instance is up and scales to zero.

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

Deleting the Cloud SQL instance deletes the memories; its automated backups follow the instance's backup
retention setting (verify) — the "backups" line of notebook 05's deletion checklist.
