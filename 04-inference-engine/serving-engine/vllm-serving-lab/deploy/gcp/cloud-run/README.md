# deploy/gcp/cloud-run — vLLM on Cloud Run with one L4 (T3)

**Tier:** T3 (GCP, pay per use). Everything here can be *read and planned* offline; notebook
[`06_deploy_on_cloud_run_gpu`](../../../notebooks_src/06_deploy_on_cloud_run_gpu.py) walks through
it and computes the cold-start and cost arithmetic at T0.

```
client ──HTTPS + identity token──▶ Cloud Run service "vllm-l4"
                                   ├─ instance 0..max_instances (scale to zero)
                                   │   container: vllm/vllm-openai  (ENTRYPOINT `vllm serve`)
                                   │   args: <model> --max-model-len --gpu-memory-utilization ...
                                   │   1 × NVIDIA L4 (24 GB), 8 vCPU, 32 GiB
                                   │   startup probe GET /health (200 once KV blocks are allocated)
                                   └─ weights: Hugging Face at start-up  |  gs://bucket mounted at /models
```

Two equivalent paths:

| Path | Command | When |
|---|---|---|
| Terraform | `cd terraform && cp terraform.tfvars.example terraform.tfvars && terraform init && terraform apply` | repeatable, reviewable, IAM included |
| gcloud | `PROJECT_ID=... ./deploy.sh` (`DRY_RUN=1` prints the commands) | quickest first run |

## Before you start

* A **paid** billing account (GPUs are not available on the Free Trial) and **Cloud Run L4 quota**
  in your region; quota for GPUs often starts at 0 — request it first (verify the quota name for
  "L4 without zonal redundancy" in the Cloud Run GPU docs).
* Size the engine for 24 GB before you pay for it:
  `python -m servelab size --model qwen2.5-1.5b-instruct --gpu L4 --max-model-len 8192`.
* Gated models (Llama) need a Hugging Face token in Secret Manager:
  `printf %s "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-`, then set
  `hf_token_secret_id = "hf-token"` (Terraform) or `HF_SECRET=hf-token` (script). The token never
  enters Terraform state.

## Measure it

The service is private (`--no-allow-unauthenticated`); either send an identity token or open a
local authenticated proxy (verify: `gcloud run services proxy` is available in your gcloud):

```bash
gcloud run services proxy vllm-l4 --region us-central1 --port 8080 &      # then:
python -m servelab bench --url http://127.0.0.1:8080 --rate 2 -n 60 --slo-ttft-ms 1000 --slo-tpot-ms 60
python -m servelab metrics --url http://127.0.0.1:8080 --window 30
SERVELAB_URL=http://127.0.0.1:8080 jupyter lab ../../../notebooks        # the notebooks now measure the L4
```

The first request after the service scaled to zero pays the **cold start**: instance start + image
pull (the vLLM image is several GB) + weights (download or GCS read) + engine init (profiling run,
KV allocation, CUDA-graph capture). Notebook 06 turns each term into seconds from bytes and
bandwidth; the startup log (`gcloud run services logs read vllm-l4`) shows the real ones —
`python -c "from servelab.sizing import parse_startup_log; ..."` reads them.

## The knobs that matter here

| Knob | Where | What it trades |
|---|---|---|
| `concurrency` (`max_instance_request_concurrency`) | Cloud Run | requests sent to one instance; set it to the batch that still meets your SLO (notebook 06), not to a default |
| `min_instances` | Cloud Run | 0 = pay nothing idle, pay cold starts; 1 = warm, pay every second |
| `max_instances` | Cloud Run | upper bound on L4s (and on your bill); quota caps it too |
| `max_model_len`, `gpu_memory_utilization`, `extra_args` | vLLM | KV blocks and concurrency (notebook 01), batching (notebook 03) |
| `model_source = "gcs"` | both | faster, repeatable cold starts; nothing written to the in-memory filesystem |

## Cost and cleanup

Cloud Run bills GPU instances per second while they exist (with CPU always allocated), and nothing
while scaled to zero. The L4 rate is roughly that of a `g2-standard` L4 hour (~$0.7/hr on Compute
Engine on-demand, us-central1) plus vCPU and memory — check the Cloud Run pricing page (verify)
and `COMPUTE.md` at the repo root. An idle service with `min_instances = 0` costs nothing
but the image in Artifact Registry (none here: the image comes from Docker Hub).

```bash
terraform destroy                                  # or:
PROJECT_ID=... ./deploy.sh delete
gcloud secrets delete hf-token                     # if you created it
```

Items marked `# VERIFY:` in the Terraform are product details (GPU regions, quota type, startup-probe
limits, CPU allocation for GPU services) to re-check against the current Cloud Run GPU docs.
