# deploy/any-gpu — a teacher, a student and a draft on whatever NVIDIA GPU you have (T1)

**Tier:** T1, one small GPU: a Colab or Kaggle T4 for free, or a rented 24 GB card for about $0.3–0.7/hr
(verify). T1 swaps the lab's *simulated* teacher for a real one. Point `DISTILLAB_URL` at the server and
the notebooks run the same code against it.

| Script | What it does |
|---|---|
| [`serve_teacher.sh`](serve_teacher.sh) | `vllm serve` a teacher with `--max-logprobs 20` (vLLM v0.30.0's cap on top log-probabilities per request), a reasoning parser picked from the model name for thinking teachers, and `--dtype half` below compute capability 8.0; docker if a daemon is reachable, else pip. `DRY_RUN=1` prints the command |
| [`train_student.sh`](train_student.sh) | prints the memory plan for the batch and length it will train at (`BATCH`, `MAX_LEN`), generates teacher data through `DISTILLAB_URL`, then trains the student with `METHOD=sft` (TRL `SFTTrainer`), `kd` (`distillab.hf.kd`: teacher in-process, chunked KL) or `gkd` (TRL `GKDTrainer`); on one card run `STAGE=data`, stop vLLM, then `STAGE=train` (the default `STAGE=auto` stops for you when the teacher is on this machine's only GPU); `LORA_R=16` for adapters; `INSTALL=1` installs the T1 stack |
| [`serve_with_draft.sh`](serve_with_draft.sh) | `vllm serve Qwen/Qwen3-4B` with `--speculative-config` for a draft model (off the shelf, or a local distilled directory, mounted into the container when it runs in docker), for notebook 04's acceptance and speedup measurements |

Everything is pinned to **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`) and **TRL 1.14.0**, the versions the
repo uses; flags and field names were checked against their sources on 2026-09-27, and the lab's `SFTConfig`,
`GKDConfig` and `DistillationConfig` arguments were built against TRL 1.14.0 with transformers 5.17.0, which is what
pip resolves for TRL's `transformers>=4.56.2` (a CPU check, not a training run; verify when you move).

## Which models on which card

| GPU | Role | Model | Weights (16-bit) | Notes |
|---|---|---|---|---|
| T4 16 GB | teacher | `Qwen/Qwen2.5-1.5B-Instruct` | 3.09 GB | the default pair with the student below: same `vocab_size` (151,936), same template, Apache-2.0 |
| T4 16 GB | student | `Qwen/Qwen2.5-0.5B-Instruct` | 0.99 GB | full SFT fits (7.9 GB of weights, gradients and AdamW + activations, predicted); logit KD needs `--chunk` and LoRA or batch 1–2 |
| T4 16 GB | thinking teacher | `Qwen/Qwen3-1.7B` (thinking on) or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.44 / 3.55 GB | parsers `qwen3` / `deepseek_r1`; sample at temperature 0.6 |
| T4 16 GB | student of a thinking teacher | `Qwen/Qwen3-0.6B` | 1.19 GB | SFT on `reasoning_content` rows (notebook 03) |
| 24 GB (L4, RTX 4090) | target + draft | `Qwen/Qwen3-4B` with `Qwen/Qwen3-0.6B` | 8.04 + 1.19 GB | equal `vocab_size`; not Qwen2.5-0.5B for Qwen2.5-7B (151,936 against 152,064) |

Parameter counts come from the bundled configs (`distillab/assets/configs/`, verify). The memory verdicts are
`python -m distillab memory`'s predictions. Qwen models are trained in bf16 and a T4 has only fp16, so watch
the first losses and outputs for `nan` or garbage (verify per model).

## Colab or Kaggle (free T4)

```python
!nvidia-smi --query-gpu=name,compute_cap --format=csv      # want compute_cap >= 7.5 (Kaggle: "GPU T4 x2", not P100)
!pip install -q "vllm==0.30.0"                             # several minutes; in its own process, not beside TRL
import subprocess, os
subprocess.Popen("vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 "
                 "--gpu-memory-utilization 0.85 --max-logprobs 20 --port 8000 > vllm.log 2>&1", shell=True)
from distillab.env import wait_healthy
assert wait_healthy("http://127.0.0.1:8000", timeout_s=900), open("vllm.log").read()[-3000:]
os.environ["DISTILLAB_URL"] = "http://127.0.0.1:8000"      # notebooks 02-05 now use the real teacher
```

Then generate the data (`python -m distillab teacher-data --problems 2000 -n 4`), **stop vLLM** so that the
trainer has the card, and train:

```bash
pip install -q "trl==1.14.0" "transformers>=4.56.2" peft datasets
python -m distillab.hf.sft --model Qwen/Qwen2.5-0.5B-Instruct --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-sft
python -m distillab.hf.kd --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-kd --lora-r 16 --chunk 256
python -m distillab.hf.gkd --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-gkd
```

## The settings that matter

| Setting | Why |
|---|---|
| `fp16=True, bf16=False` (T4) | a T4 has no bf16, and TRL's `SFTConfig`, `GKDConfig` and `DistillationConfig` turn `bf16` on unless `fp16` is set; the lab's builders set both |
| trainable weights in fp32 | loading fp16 weights and training with `fp16=True` fails ("Attempting to unscale FP16 gradients."); `hf/sft.py` and `hf/kd.py` load fp32 (full) or fp16 base + fp32 adapters (LoRA) |
| a chunked KD loss | at 151,936 tokens one fp32 `[4,096 tokens × vocab]` tensor is 2.49 GB; `--chunk 256` computes the KL 256 positions at a time |
| `--max-logprobs` | vLLM rejects `top_logprobs` above it (default 20); `-1` means the whole vocabulary and may run out of memory |
| `from trl.experimental.gkd import GKDTrainer` | GKD is experimental in TRL 1.x; `from trl import GKDTrainer` fails. `GKDConfig.temperature` only sets sampling, the loss runs at T = 1 |
| same `vocab_size` | TRL's GKD and vLLM's draft model both check it; Qwen2.5 ≥ 7B has 152,064, the small ones 151,936 |
| transformers 5.x | `TrainingArguments.warmup_ratio` is gone and `warmup_steps` below 1 is read as a ratio; `hf.sft.for_config` translates the lab's kwargs for whichever major is installed |
| data for `DistillationTrainer` | it needs a `prompt` column and generates its own completions: `python -m distillab.hf.gkd --trainer distillation --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-od` (a `messages` file is converted) |

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod and Vast.ai** give you a container. Choose `vllm/vllm-openai:v0.30.0`, put the model and flags from
  `serve_teacher.sh` (or `serve_with_draft.sh`) in the container arguments, expose port 8000 and set
  `--api-key` (the endpoint is public). A 24 GB RTX 4090 costs roughly $0.3–0.4/hr (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)). Train in a second container with the TRL stack, or on the same
  one after stopping vLLM.
* **Lambda** (and GCP Compute Engine) give you a VM: install Docker and the NVIDIA Container Toolkit
  (layer 02) or `pip install vllm`, then run the scripts here.
* Run the notebooks on the same machine (`DISTILLAB_URL=http://127.0.0.1:8000`) unless you want network
  latency in your throughput numbers.

## Cost and cleanup

`Ctrl-C` (or `docker stop`) the server. Colab and Kaggle cost only quota. Rented machines bill until you
*terminate* them, not when vLLM stops, so terminate the pod or instance when the session ends. Budget the
session: generating 2,000 × 4 samples from a 1.5B teacher on a T4 takes minutes, and SFT of a 0.5B student on
them takes tens of minutes (verify on your card).
