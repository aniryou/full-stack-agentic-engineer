# deploy — a real teacher, a real student and a real draft, wherever you have a GPU

Every target runs the same `distillab` code as the notebooks. At T0 the fake teacher answers
(`python -m distillab fake`); from T1 on, `DISTILLAB_URL` points at a real vLLM and the notebooks measure it.
This lab adds **no Terraform**: teacher inference on Google Cloud is the 04 serving lab's Cloud Run or GKE
deploy with a teacher's settings.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m distillab fake` | T0 | a fake vLLM teacher: generated problems answered with scratchpads, log-probs, usage (simulated) | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `serve_teacher.sh` (vLLM with `--max-logprobs`, a reasoning parser for thinking teachers), `train_student.sh` (teacher data, then SFT, logit KD or GKD of a 0.5–0.6B student), `serve_with_draft.sh` (a target with a draft model) | free on Colab or Kaggle (T4); ~$0.3–0.7/hr for a 24 GB card | Ctrl-C; terminate rented machines |
| [`gcp/`](gcp/) | T3 | the 04 lab's Cloud Run service (one L4, scale to zero) or GKE Deployment, serving a larger teacher for data generation | per second while an instance runs | `terraform destroy` / `kubectl delete` |

Prices and GPU availability move; the dated table is [`COMPUTE.md`](../../../../COMPUTE.md).
