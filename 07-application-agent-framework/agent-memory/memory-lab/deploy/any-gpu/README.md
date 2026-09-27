# deploy/any-gpu — a real model with tool calls, and a real embedder, on one GPU (T1)

**Tier:** T1 — one small GPU: a free Colab or Kaggle T4, or any rented 24 GB card. **Cost:** free on
Colab/Kaggle; about $0.3–0.7 per hour for a rented 24 GB GPU (verify; [`COMPUTE.md`](../../../../../COMPUTE.md)).

This lab adds no server of its own: it reuses the 04 serving lab's
[`serve.sh`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/serve.sh)
(vLLM `v0.30.0`, `DRY_RUN=1` prints the command) with the flags memory needs, then points the notebooks at it.
Read its [README](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md)
first for drivers, compute capability (7.5 or newer: a T4 works, a P100 does not) and the Colab recipe.

## 1. A chat model that calls tools, with per-request cached tokens

```bash
cd 04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu     # from the repo root
MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.6 \
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details" ./serve.sh
export MEMLAB_LLM_URL=http://127.0.0.1:8000
```

| Flag | Why this lab needs it |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser hermes` | lets the model emit `remember` / `recall` / `forget` calls; Qwen2.5's chat template uses Hermes-style tool calls (vLLM tool-calling docs, v0.30.0) |
| `--enable-prompt-tokens-details` | adds `usage.prompt_tokens_details.cached_tokens` per request — notebook 03's measured column; without it only `/metrics` has hits |
| `GPU_MEM_UTIL=0.6` | leaves room on the same GPU for the embedder below (two vLLM processes, one model each) |

`Qwen/Qwen2.5-1.5B-Instruct` is 1.54 B parameters, about 3.1 GB in BF16 (FP16 on a T4) with 28 KiB of KV per
token — it fits a T4 with room for long agent prompts (verify the licence on its model card).
`Qwen/Qwen3-1.7B` also works with `--reasoning-parser qwen3` added (verify the tool-call parser for Qwen3).

## 2. An embedder on vLLM's pooling runner (notebook 05's paraphrase subset)

```bash
MODEL=BAAI/bge-small-en-v1.5 PORT=8001 MAX_MODEL_LEN=512 GPU_MEM_UTIL=0.2 EXTRA_ARGS="--runner pooling" ./serve.sh
export MEMLAB_EMBED_URL=http://127.0.0.1:8001
curl -s http://127.0.0.1:8001/v1/embeddings -H 'Content-Type: application/json' \
     -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["Where am I based these days?"]}' | head -c 200
```

vLLM v0.30.0 has no `--task` flag; `--runner pooling` selects the embedding path (auto-detection also picks it
for a Sentence-Transformers checkpoint). `bge-small-en-v1.5` is a 384-dimension BERT encoder of about 33 M
parameters with 512 positions — hence `MAX_MODEL_LEN=512` (verify its licence on the model card). Its vectors
have 384 dimensions, not the hashing embedder's 1,024: start a **fresh** store for it (notebook 05's T1 cell
does), never mix dimensions in one table.

## 3. Run the notebooks against them

```bash
cd 07-application-agent-framework/agent-memory/memory-lab
python -m memlab env                  # tier T1 when either URL answers and is not the fake server
python -m memlab cachebench --url $MEMLAB_LLM_URL      # notebook 03's table, measured
jupyter lab notebooks                  # 02 (real tool calls), 03 (measured cached tokens), 05 (real embedder)
```

On Colab: run `serve.sh`'s pip path in a cell (see the serving lab's README), then set the two environment
variables with `os.environ[...]` before running the notebook's first cell.

## Cost

A free T4 costs nothing but session time (Colab and Kaggle limit weekly GPU hours; verify). A rented 24 GB
card at about $0.3–0.7 per hour runs notebooks 02, 03 and 05 in well under an hour.

## Cleanup

Stop both `vllm serve` processes (Ctrl-C, or `docker stop` if `serve.sh` used Docker), then terminate the
rented instance — a stopped-but-allocated GPU instance keeps billing on most providers (verify yours).
The Hugging Face cache (`~/.cache/huggingface`) holds the weights; delete it if the disk is shared.
