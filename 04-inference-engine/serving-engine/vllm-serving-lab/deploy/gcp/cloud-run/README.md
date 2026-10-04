# deploy/gcp/cloud-run — vLLM on Cloud Run with one L4 (T3)

**Tier:** T3 (GCP, pay per use). You can *read and plan* everything here offline. Notebook
[`06_deploy_on_cloud_run_gpu`](../../../notebooks_src/06_deploy_on_cloud_run_gpu.py) goes through
it. It also does the cold-start and cost arithmetic at T0.

```
client ──HTTPS + identity token──▶ Cloud Run service "vllm-l4"
                                   ├─ instance 0..max_instances (scale to zero)
                                   │   container: vllm/vllm-openai  (ENTRYPOINT `vllm serve`)
                                   │   args: <model> --max-model-len --gpu-memory-utilization ...
                                   │   1 × NVIDIA L4 (24 GB), 8 vCPU, 32 GiB
                                   │   startup probe GET /health (200 once KV blocks are allocated)
                                   └─ weights: Hugging Face at start-up  |  gs://bucket mounted at /models
```

There are two equivalent paths:

| Path | Command | When |
|---|---|---|
| Terraform | `cd terraform && cp terraform.tfvars.example terraform.tfvars && terraform init && terraform apply` | You can repeat it and review it. It includes IAM. |
| gcloud | `PROJECT_ID=... ./deploy.sh` (`DRY_RUN=1` prints the commands) | The fastest first run. |

## Before you start

* Get a **paid** billing account. GPUs are not available on the Free Trial. Also get **Cloud Run L4
  quota** in your region. Because quota for GPUs often starts at 0, request it first. Find the quota
  name for "L4 without zonal redundancy" in the Cloud Run GPU docs (verify).
* Calculate the engine size for 24 GB before you pay for it:
  `python -m servelab size --model qwen2.5-1.5b-instruct --gpu L4 --max-model-len 8192`.
* Gated models (Llama) need a Hugging Face token in Secret Manager. Put the token there with
  `printf %s "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-`. Then set
  `hf_token_secret_id = "hf-token"` (Terraform) or `HF_SECRET=hf-token` (script). The token never
  goes into the Terraform state.

## Measure it

The service is private (`--no-allow-unauthenticated`). Send an identity token, or open a local
authenticated proxy (verify: `gcloud run services proxy` is available in your gcloud):

```bash
gcloud run services proxy vllm-l4 --region us-central1 --port 8080 &      # then:
python -m servelab bench --url http://127.0.0.1:8080 --rate 2 -n 60 --slo-ttft-ms 1000 --slo-tpot-ms 60
python -m servelab metrics --url http://127.0.0.1:8080 --window 30
SERVELAB_URL=http://127.0.0.1:8080 jupyter lab ../../../notebooks        # the notebooks now measure the L4
```

The first request after a scale to zero pays the **cold start**. The cold start is the sum of these
terms:

* the instance start,
* the image pull (the vLLM image is several GB),
* the weights (a download or a GCS read),
* the engine init (the profiling run, the KV allocation and the CUDA-graph capture).

Notebook 06 calculates each term in seconds from bytes and bandwidth. The startup log
(`gcloud run services logs read vllm-l4`) shows the real values.
`python -c "from servelab.sizing import parse_startup_log; ..."` reads them.

## The knobs that matter here

| Knob | Where | What it trades |
|---|---|---|
| `concurrency` (`max_instance_request_concurrency`) | Cloud Run | The requests that go to one instance. Set it to the batch that still meets your SLO (notebook 06), not to a default. |
| `min_instances` | Cloud Run | At 0, you pay nothing while idle, but you pay for cold starts. At 1, the instance is warm, and you pay for every second. |
| `max_instances` | Cloud Run | The upper limit on L4s (and on your bill). The quota also limits it. |
| `max_model_len`, `gpu_memory_utilization`, `extra_args` | vLLM | KV blocks and concurrency (notebook 01), batching (notebook 03). |
| `model_source = "gcs"` | both | Faster cold starts, with the same duration each time. Nothing goes into the in-memory filesystem. |

## Cost and cleanup

Cloud Run charges for GPU instances per second while they exist (with CPU always allocated). It
charges nothing while the service is scaled to zero. The L4 rate is approximately the rate of a
`g2-standard` L4 hour (~$0.7/hr on Compute Engine on-demand, us-central1), plus vCPU and memory.
Examine the Cloud Run pricing page (verify) and [`COMPUTE.md`](../../../../../../COMPUTE.md). An idle
service with `min_instances = 0` costs nothing, except for the image in Artifact Registry (none
here: the image comes from Docker Hub).

```bash
terraform destroy                                  # or:
PROJECT_ID=... ./deploy.sh delete
gcloud secrets delete hf-token                     # if you created it
```

The items with the mark `# VERIFY:` in the Terraform are product details (GPU regions, quota type,
startup-probe limits, CPU allocation for GPU services). Make sure that they agree with the
current Cloud Run GPU docs.
