# deploy/gcp — the 04 serving lab's Google Cloud deploys, serving a teacher (T3)

**Tier:** T3 (GCP, pay per use; optional). This lab adds **no Terraform**. Serving a teacher on Google Cloud is
the 04 lab's deploy with different engine flags, and the students train wherever you have a GPU
([`../any-gpu/`](../any-gpu/)). The one file here is those flags, checked offline by `tests/test_deploy.py`:

| File | For | What changes against the 04 lab's defaults |
|---|---|---|
| [`cloud-run-teacher.tfvars.example`](cloud-run-teacher.tfvars.example) | the 04 lab's Cloud Run Terraform, [`../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/) | `Qwen/Qwen2.5-7B-Instruct` as the teacher, `max_model_len` 4096, `--max-logprobs 20`, `--max-num-seqs 64`, `concurrency` 64 for a batch job, `request_timeout` 900 s, scale to zero |

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

Read the 04 lab's [Cloud Run README](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/README.md)
first: a paid billing account, L4 quota, a private service, and the cold-start arithmetic. Two things differ
for a teacher:

* **It is a batch job, not an interactive service.** Send many requests at once (`concurrency` 64 against
  `--max-num-seqs 64`) and judge the deploy by tokens per second and dollars per million tokens (notebook 05),
  not by time to first token.
* **The teacher's vocabulary decides what you can do with it.** A 7B Qwen2.5 teacher produces text for
  sequence-level distillation into any student. Logit KD, GKD and draft models need the student's `vocab_size`
  (PRIMER §6 "Feature distillation, pruning and vocabulary mismatch").

## GKE (L4 Spot)

The 04 lab's [`gke/vllm.yaml`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/vllm.yaml)
serves `Qwen/Qwen2.5-1.5B-Instruct`, the default teacher here. Add `--max-logprobs=20` to its `args` and apply
it to the 04 lab's cluster (`./cluster.sh`), or to layer 03's or 05's Terraform clusters, then
`kubectl port-forward svc/vllm 8000:8000` and `export DISTILLAB_URL=http://127.0.0.1:8000` (check the service
name and port in that manifest).

## Cost and cleanup

Cloud Run bills per second while an instance exists. With `min_instances = 0` an idle service costs nothing,
but the next request pays a cold start (image pull plus 15 GB of weights for the 7B). Generating 20,000 samples
is minutes to an hour of L4 time (verify with notebook 05's throughput cell). `terraform destroy
-var-file=teacher.tfvars`, run in `$LAB04`, removes it. On GKE, delete the Deployment and then the cluster; the cluster keeps
billing until you delete it. Prices are in [`COMPUTE.md`](../../../../../COMPUTE.md).
