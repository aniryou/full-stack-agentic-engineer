# deploy/any-gpu — a real model with tool calls, and a real embedder, on one GPU (T1)

**Tier:** T1, that is one small GPU: a free Colab or Kaggle T4, or any rented 24 GB card. **Cost:** free on
Colab/Kaggle. A rented 24 GB GPU costs about $0.3–0.7 per hour (verify, and see
[`COMPUTE.md`](../../../../../COMPUTE.md)).

This lab adds no server of its own. It uses the
[`serve.sh`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/serve.sh) that the 04
serving lab already has (vLLM `v0.30.0`, and `DRY_RUN=1` prints the command). It adds the flags that memory needs. Then it
points the notebooks at that server. First, read the
[README](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md) of that
folder. It tells you about drivers, compute capability (7.5 or newer: a T4 works, a P100 does not) and the Colab
recipe.

## 1. A chat model that calls tools, with per-request cached tokens

```bash
cd 04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu     # from the repo root
MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.6 \
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details" ./serve.sh
export MEMLAB_LLM_URL=http://127.0.0.1:8000
```

| Flag | Why this lab needs it |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser hermes` | It lets the model make `remember` / `recall` / `forget` calls. The chat template of Qwen2.5 uses Hermes-style tool calls (vLLM tool-calling docs, v0.30.0). |
| `--enable-prompt-tokens-details` | It adds `usage.prompt_tokens_details.cached_tokens` for each request. This is the measured column of notebook 03. It is the only measured column: the prefill times stay a roofline estimate, and the dollars stay a price table. Without this flag, only `/metrics` has hits. |
| `GPU_MEM_UTIL=0.6` | It leaves space on the same GPU for the embedder in §2 (two vLLM processes, one model each). |

`Qwen/Qwen2.5-1.5B-Instruct` has 1.54 B parameters. It is about 3.1 GB in BF16 (FP16 on a T4), with 28 KiB of KV
per token. It fits a T4 with space for long agent prompts (verify the licence on its model card).
`Qwen/Qwen3-1.7B` also works if you add `--reasoning-parser qwen3` (verify the tool-call parser for Qwen3).

## 2. An embedder on vLLM's pooling runner (notebook 05's paraphrase subset)

```bash
MODEL=BAAI/bge-small-en-v1.5 PORT=8001 MAX_MODEL_LEN=512 GPU_MEM_UTIL=0.2 EXTRA_ARGS="--runner pooling" ./serve.sh
export MEMLAB_EMBED_URL=http://127.0.0.1:8001
curl -s http://127.0.0.1:8001/v1/embeddings -H 'Content-Type: application/json' \
     -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["Where am I based these days?"]}' | head -c 200
```

vLLM v0.30.0 has no `--task` flag. `--runner pooling` selects the embedding path. Auto-detection also selects it
for a Sentence-Transformers checkpoint. `bge-small-en-v1.5` is a 384-dimension BERT encoder of about 33 M
parameters with 512 positions. Thus the command sets `MAX_MODEL_LEN=512` (verify its licence on the model card).

Its vectors have 384 dimensions, not the 1,024 of the hashing embedder. Start a **fresh** store for it (the T1 cell
of notebook 05 does this). Never mix dimensions in one table.

## 3. Run the notebooks against them

```bash
cd 07-application-agent-framework/agent-memory/memory-lab
python -m memlab env                  # tier T1 when either URL answers and is not the fake server
python -m memlab cachebench --url $MEMLAB_LLM_URL --gpu T4   # notebook 03's table: cached tokens measured,
                                                             # prefill ms modelled for --gpu/--llm, $ at list prices
jupyter lab notebooks                  # 02 (real tool calls), 03 (measured cached tokens), 05 (real embedder)
```

On Colab, run the pip path of `serve.sh` in a cell (see the README of the serving lab). Then set the two
environment variables with `os.environ[...]`. Do this before you run the first cell of the notebook.

## Cost

A free T4 costs only session time (Colab and Kaggle limit weekly GPU hours, verify). A rented 24 GB card at about
$0.3–0.7 per hour runs notebooks 02, 03 and 05 in much less than an hour.

## Cleanup

Stop both `vllm serve` processes (Ctrl-C, or `docker stop` if `serve.sh` used Docker). Then terminate the rented
instance. Most providers continue to bill for a GPU instance that is stopped but still allocated (verify yours).
The Hugging Face cache (`~/.cache/huggingface`) holds the weights. If you share the disk, delete the cache.
