# deploy/gcp — the 04 serving lab's Google Cloud deploys, serving a teacher (T3)

**Tier:** T3 (GCP, pay per use, optional). This lab adds **no Terraform**. A teacher on Google Cloud uses the
deploy of the 04 lab with different engine flags. The students train wherever you have a GPU
([`../any-gpu/`](../any-gpu/)). The one file here holds those flags, and `tests/test_deploy.py` examines it offline:

| File | For | What changes against the 04 lab's defaults |
|---|---|---|
| [`cloud-run-teacher.tfvars.example`](cloud-run-teacher.tfvars.example) | the Cloud Run Terraform of the 04 lab, [`../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/) | `Qwen/Qwen2.5-7B-Instruct` as the teacher, `max_model_len` 4096, `--max-logprobs 20`, `--max-num-seqs 64`, `concurrency` 64 for a batch job, `request_timeout` 900 s, scale to zero |

## Cloud Run (one L4, scale to zero)

```bash
# from this directory (distill-lab/deploy/gcp)
LAB04=../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform
cp cloud-run-teacher.tfvars.example "$LAB04/teacher.tfvars"      # edit project_id, invoker_members
(cd "$LAB04" && terraform init && terraform apply -var-file=teacher.tfvars)   # a subshell: you stay here
gcloud run services proxy vllm-teacher-l4 --region us-central1 --port 8080 &      # (verify the command in your gcloud)
export DISTILLAB_URL=http://127.0.0.1:8080
cd ../..                                                          # the lab directory
python -m distillab teacher-data --problems 5000 -n 4 --out _run_outputs/teacher
```

Read the [Cloud Run README](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/README.md)
of the 04 lab first. It covers a paid billing account, L4 quota, a private service, and the cold-start arithmetic.
Two things are different for a teacher:

* **It is a batch job, not an interactive service.** Send many requests at the same time (`concurrency` 64
  against `--max-num-seqs 64`). Measure the deploy in tokens per second and dollars per million tokens (notebook
  05), not in time to first token.
* **The teacher's vocabulary decides what you can do with it.** A 7B Qwen2.5 teacher produces text for
  sequence-level distillation into any student. Logit KD, GKD and draft models need the `vocab_size` of the
  student (PRIMER §6 "Feature distillation, pruning and vocabulary mismatch").

## GKE (L4 Spot)

The 04 lab's [`gke/vllm.yaml`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/vllm.yaml)
serves `Qwen/Qwen2.5-1.5B-Instruct`, the default teacher here. Add `--max-logprobs=20` to its `args`. Apply the manifest
to the 04 lab's cluster (`./cluster.sh`), or to the Terraform clusters of layer 03 or 05. Examine the service name
and port in that manifest. Then run `kubectl port-forward svc/vllm 8000:8000`. After that, run
`export DISTILLAB_URL=http://127.0.0.1:8000`.

## Cost and cleanup

You pay for Cloud Run per second while an instance exists. With `min_instances = 0`, an idle service costs
nothing. But the next request pays for a cold start (image pull plus 15 GB of weights for the 7B).

The generation of 20,000 samples takes minutes to an hour of L4 time (verify with the throughput cell of notebook
05). To remove the service, run `terraform destroy
-var-file=teacher.tfvars` in `$LAB04`. On GKE, delete the Deployment and then the cluster. You pay for the cluster
until you delete it. Prices are in [`COMPUTE.md`](../../../../../COMPUTE.md).
