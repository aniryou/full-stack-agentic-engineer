# 00 · Foundations — model internals, capacity math, model landscape

Concepts that underpin the whole stack but aren't one of the seven infrastructure layers:
how the models work, how to size hardware for them, and what the model landscape looks like.

**Covers:** transformer architecture (attention, MLP blocks, a tiny GPT from scratch), GPU
capacity planning for LLM serving (memory vs bandwidth, TTFT/TPOT, fleet sizing), and the
open-weight / commercial model landscape.

**Signal keywords:** transformer, attention, MLP, embeddings, tiny GPT, scaling laws, capacity
planning, TTFT, TPOT, HBM vs bandwidth, open-weight, model landscape, Mistral/Llama/Qwen.

## Current contents
- **`transformers/`** — build a Transformer from nothing (attention → block → tiny GPT), a
  primer (`docs/transformer-primer.md`), practice notebooks, plus extra walkthrough/exercise
  notebooks under `notebooks/`.
- **`gpu-capacity-planning/`** — sizing GPU fleets for LLM serving: two constraints (memory,
  bandwidth), two SLOs (TTFT, TPOT), ~8 plain-Python formulas, worked examples, practice.
- **`model-landscape/`** — `open-weight-llms-primer.md` (the open-weight frontier) and
  `mistral-primer-exercises.md` (Mistral model cost/routing exercises).

_New model-level material (architectures, model families, scaling laws) belongs here._

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`gpu-capacity-planning/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice.ipynb) `01_capacity_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice_solved.ipynb) `01_capacity_practice_solved.ipynb`

**`transformers/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/notebooks/01_transformer_walkthrough.ipynb) `01_transformer_walkthrough.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/notebooks/02_transformer_exercises.ipynb) `02_transformer_exercises.ipynb`

**`transformers/practice/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/practice/attention_practice.ipynb) `attention_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/practice/attention_solutions.ipynb) `attention_solutions.ipynb`
<!-- colab-links:end -->
