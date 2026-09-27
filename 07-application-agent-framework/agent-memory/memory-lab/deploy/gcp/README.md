# deploy/gcp — consolidation as a Cloud Run job on Cloud Scheduler, and where the model and the store live (T3)

**Tier:** T3 (Google Cloud, optional). **Everything here is printed, not run by the lab**: notebook 04 and
`python -m memlab gcp-commands --project <id>` print the commands; you run them. No Terraform in this lab —
the GPU and Cloud Run infrastructure already has Terraform in the 04 serving lab.

```
Cloud Scheduler "memlab-consolidate-nightly"  (cron 17 3 * * *, UTC)
   │  POST https://run.googleapis.com/v2/projects/<p>/locations/<r>/jobs/memlab-consolidate:run
   │  OAuth access token of memlab-scheduler@<p>.iam.gserviceaccount.com
   ▼
Cloud Run job "memlab-consolidate"  (--tasks N; each task takes the users whose hash lands on its index)
   │  python -m memlab consolidate --window-days 7
   │  one durable run per (tenant, user, ISO week): lease row, checkpoints, idempotent writes
   ▼
Postgres + pgvector (Cloud SQL, verify) ◀── the memory service and the agent read and write the same tables
   ▲
   │  the agent's model: vLLM on Cloud Run with one L4 (the 04 serving lab), tool calling on
```

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
(checked offline by Postgres's parser in the tests), including the job's lease and checkpoint tables. Put the
DSN in Secret Manager as `memlab-pg-dsn`; the job reads it as `MEMLAB_PG_DSN`.

## 3. The job and its schedule

```bash
python -m memlab gcp-commands --project my-project --region us-central1 --tasks 4
```

prints, in order: `gcloud run jobs create` (image, `--tasks`, `--max-retries 3`, the secret, the command),
a service account for the scheduler, `roles/run.invoker` on the job for it, `gcloud scheduler jobs create
http` with `--oauth-service-account-email`, and a manual `gcloud run jobs execute --wait`. Build and push the
image first (`deploy/local/Dockerfile`, to Artifact Registry `memlab/memlab:0.1.0`).

Why **OAuth, not OIDC**: the scheduler calls the Cloud Run Admin API (`run.googleapis.com`), which accepts OAuth
access tokens; OIDC identity tokens are for invoking your own service's URL (the lra-gcp reaper does that,
see `07-application-agent-framework/long-running-durable/lra-gcp/docs/primer.md` §3.13). Google's Terraform
sample for scheduled jobs uses this v2 URI; its gcloud samples use the v1 form
`https://<region>-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/<project>/jobs/<job>:run` (verify
which your gcloud prefers). Also verify: `--task-timeout`, `--set-secrets` and `--command/--args` on
`gcloud run jobs create`, and whether `roles/run.invoker` on the job is enough to run it (Google's sample
grants `roles/run.developer` on the project).

A double-fired schedule is harmless: both executions compute the same run id per user and week; one takes
the lease, the other finds it held or the run done (notebook 04).

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

```bash
gcloud scheduler jobs delete memlab-consolidate-nightly --location us-central1 --quiet
gcloud run jobs delete memlab-consolidate --region us-central1 --quiet
gcloud iam service-accounts delete memlab-scheduler@my-project.iam.gserviceaccount.com --quiet
gcloud secrets delete memlab-pg-dsn --quiet
gcloud artifacts docker images delete us-central1-docker.pkg.dev/my-project/memlab/memlab:0.1.0 --quiet
# the Cloud SQL instance and the serving lab's Cloud Run service: see their own cleanup steps
```

Deleting the Cloud SQL instance deletes the memories; its automated backups follow the instance's backup
retention setting (verify) — the "backups" line of notebook 05's deletion checklist.
