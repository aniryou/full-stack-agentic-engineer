# deploy/any-gpu — a teacher, a student and a draft on whatever NVIDIA GPU you have (T1)

**Tier:** T1, one small GPU: a Colab or Kaggle T4 for free, or a rented 24 GB card for about $0.3–0.7/hr
(verify). T1 replaces the *simulated* teacher of the lab with a real one. Point `DISTILLAB_URL` at the server.
Then the notebooks run the same code against it.

| Script | What it does |
|---|---|
| [`serve_teacher.sh`](serve_teacher.sh) | Runs `vllm serve` for a teacher with `--max-logprobs 20` (the limit of vLLM v0.30.0 on top log-probabilities per request). For thinking teachers, it adds a reasoning parser that it selects from the model name. Below compute capability 8.0, it adds `--dtype half`. It uses docker if a daemon is reachable, and pip if not. `DRY_RUN=1` prints the command. |
| [`train_student.sh`](train_student.sh) | Prints the memory plan for the batch and length that the training will use (`BATCH`, `MAX_LEN`). Then it generates teacher data through `DISTILLAB_URL`. Then it trains the student with `METHOD=sft` (TRL `SFTTrainer`), `kd` (`distillab.hf.kd`: teacher in-process, chunked KL) or `gkd` (TRL `GKDTrainer`). On one card, run `STAGE=data`, stop vLLM, then run `STAGE=train`. With the default `STAGE=auto`, the script stops after the data step when the teacher is on the only GPU of this machine. `LORA_R=16` selects adapters, and `INSTALL=1` installs the T1 stack. |
| [`serve_with_draft.sh`](serve_with_draft.sh) | Runs `vllm serve Qwen/Qwen3-4B` with `--speculative-config` for a draft model. The draft is off the shelf, or a local distilled directory. When the script runs in docker, it mounts that directory into the container. Notebook 04 uses this server for its acceptance and speedup measurements. |

The lab pins everything to **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`) and **TRL 1.14.0**, the versions that
the repo uses. On 2026-09-27, a check compared the flags and field names with their sources. The lab also built its
`SFTConfig`, `GKDConfig` and `DistillationConfig` arguments against TRL 1.14.0 with transformers 5.17.0. Transformers
5.17.0 is the version that pip resolves for the `transformers>=4.56.2` requirement of TRL. That build was a CPU check, not a training run.
When you move to other versions, do the check again.

## Which models on which card

| GPU | Role | Model | Weights (16-bit) | Notes |
|---|---|---|---|---|
| T4 16 GB | teacher | `Qwen/Qwen2.5-1.5B-Instruct` | 3.09 GB | the default pair with the student in the next row: same `vocab_size` (151,936), same template, Apache-2.0 |
| T4 16 GB | student | `Qwen/Qwen2.5-0.5B-Instruct` | 0.99 GB | Full SFT fits (7.9 GB of weights, gradients and AdamW + activations, predicted). Logit KD needs `--chunk`, and LoRA or batch 1–2. |
| T4 16 GB | thinking teacher | `Qwen/Qwen3-1.7B` (thinking on) or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.44 / 3.55 GB | Parsers `qwen3` / `deepseek_r1`. Sample at temperature 0.6. |
| T4 16 GB | student of a thinking teacher | `Qwen/Qwen3-0.6B` | 1.19 GB | SFT on `reasoning_content` rows (notebook 03) |
| 24 GB (L4, RTX 4090) | target + draft | `Qwen/Qwen3-4B` with `Qwen/Qwen3-0.6B` | 8.04 + 1.19 GB | Equal `vocab_size`. Do not use Qwen2.5-0.5B as the draft for Qwen2.5-7B (151,936 against 152,064). |

Parameter counts come from the bundled configs (`distillab/assets/configs/`, verify). The memory verdicts are
`python -m distillab memory`'s predictions. The makers of the Qwen models trained them in bf16, but a T4 has
only fp16. Thus, monitor the first losses and outputs for `nan` or for text with no meaning (verify per model).

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

Then generate the data (`python -m distillab teacher-data --problems 2000 -n 4`). After that, **stop vLLM** so
that the trainer has the card. Then train:

```bash
pip install -q "trl==1.14.0" "transformers>=4.56.2" peft datasets
python -m distillab.hf.sft --model Qwen/Qwen2.5-0.5B-Instruct --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-sft
python -m distillab.hf.kd --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-kd --lora-r 16 --chunk 256
python -m distillab.hf.gkd --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-gkd
```

## The settings that matter

| Setting | Why |
|---|---|
| `fp16=True, bf16=False` (T4) | A T4 has no bf16. TRL's `SFTConfig`, `GKDConfig` and `DistillationConfig` turn `bf16` on, unless you set `fp16`. The builders of the lab set both. |
| trainable weights in fp32 | If you load fp16 weights and train with `fp16=True`, the run fails ("Attempting to unscale FP16 gradients."). `hf/sft.py` and `hf/kd.py` load fp32 (full) or an fp16 base + fp32 adapters (LoRA). |
| a chunked KD loss | At 151,936 tokens, one fp32 `[4,096 tokens × vocab]` tensor is 2.49 GB. `--chunk 256` calculates the KL for 256 positions at a time. |
| `--max-logprobs` | vLLM rejects `top_logprobs` above it (default 20). `-1` means the whole vocabulary. With `-1`, it is possible that the server runs out of memory. |
| `from trl.experimental.gkd import GKDTrainer` | GKD is experimental in TRL 1.x. `from trl import GKDTrainer` fails. `GKDConfig.temperature` applies only when the trainer samples. But the loss runs at T = 1. |
| same `vocab_size` | The GKD of TRL and the draft model of vLLM both do a check on it. Qwen2.5 ≥ 7B has 152,064, and the small models have 151,936. |
| transformers 5.x | `TrainingArguments.warmup_ratio` is gone. Version 5.x reads a `warmup_steps` below 1 as a ratio. `hf.sft.for_config` translates the kwargs of the lab for the major version that you have. |
| data for `DistillationTrainer` | It needs a `prompt` column, and it generates its own completions: `python -m distillab.hf.gkd --trainer distillation --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-od` (the command converts a `messages` file). |

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod and Vast.ai** give you a container. Select `vllm/vllm-openai:v0.30.0`. Put the model and flags from
  `serve_teacher.sh` (or `serve_with_draft.sh`) in the container arguments. Expose port 8000. Set `--api-key`,
  because the endpoint is public. A 24 GB RTX 4090 costs approximately $0.3–0.4/hr (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)).

  Train in a second container with the TRL stack, or on the same one after you stop vLLM.
* **Lambda** (and GCP Compute Engine) give you a VM. Install Docker and the NVIDIA Container Toolkit
  (layer 02), or run `pip install vllm`. Then run the scripts here.
* Run the notebooks on the same machine (`DISTILLAB_URL=http://127.0.0.1:8000`), unless you want network
  latency in your throughput numbers.

## Cost and cleanup

Stop the server with `Ctrl-C` (or `docker stop`). Colab and Kaggle cost only quota. You pay for rented machines
until you *terminate* them, not until vLLM stops. Thus, terminate the pod or instance when the session ends.

Make a budget for the session. The generation of 2,000 × 4 samples from a 1.5B teacher on a T4 takes minutes. SFT
of a 0.5B student on them takes tens of minutes (verify on your card).
