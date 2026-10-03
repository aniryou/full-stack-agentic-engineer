# Transformer lessons

This folder holds:

- three short Python files that build the core of a Transformer from nothing and that you can run,
- a fill-in-the-blank practice notebook,
- a walkthrough and exercise pair for the original encoder-decoder.

It is the companion to `docs/transformer-primer.md`.

This is the core concept, in one line: **attention moves information between positions. The MLP processes it within a
position. A Transformer is that pair, stacked.** Each file that this README lists shows one part of that sentence in
printed numbers.

**Time and tier:** ~5 h with the primer. This folder is module 00.1 in [`CURRICULUM.md`](../../CURRICULUM.md). T0 is a laptop or Colab CPU, at no cost. Lessons 1–2 and the practice notebook need only numpy. Lesson 3 and the two `notebooks/0*_transformer_*` notebooks need CPU PyTorch (T0 + torch, and Colab has it). You need no GPU and no key.

## Setup

```bash
pip install -r requirements.txt      # numpy for lessons 1-2 and the practice notebook; torch (CPU is fine) for the rest
```

**Tiers.** `lessons/01_attention.py`, `lessons/02_block.py` and the attention practice notebook need only numpy. Their
tier is T0 (a laptop, Colab CPU or CI, with no GPU, $0). `lessons/03_tiny_gpt.py`, `notebooks/01_transformer_walkthrough.ipynb` and
`notebooks/02_transformer_exercises.ipynb` need PyTorch. A CPU build is sufficient, and Colab has PyTorch preinstalled.
No file here needs a GPU.

## Lessons — read top to bottom, then run

| File | What it builds | What to look for in the output |
|---|---|---|
| `lessons/01_attention.py` | Scaled dot-product attention in ~15 lines of numpy | The weights always add up to 1. Thus, the result is an average, not a selection. Without the `1/sqrt(d_k)`, the softmax saturates. If you change the order of the tokens, the output rows change order in the same way. Thus, attention knows nothing about order until you add positions. The causal mask puts exact zeros above the diagonal. |
| `lessons/02_block.py` | A full block: multi-head attention + MLP + residuals + norm | The output has the same shape as the input. `block(x) == x + attention_update + mlp_update`. Make a small change to token 2. Then the MLP changes only row 2, causal attention changes rows 2–5, and unmasked attention changes everything. The parameter count from `12·d²` reproduces the 124M of GPT-2 small. |
| `lessons/03_tiny_gpt.py` | A 26k-parameter GPT trained on a copy task, ~15 s on CPU | The loss starts at `ln(vocab)`. The loss on the unpredictable half of each sequence never improves. The loss on the copyable half decreases to zero. Greedy decoding reproduces the prompt. One head puts ~80% of its weight on "the token six positions back". Nobody wrote that rule. |

The primer sections that go with the lessons are §2 for lesson 1, §3 and §8 for lesson 2, and §5–7 for lesson 3.

## Practice

`notebooks/attention_practice.ipynb` has seven exercises with `...` blanks. They come in this sequence: softmax,
attention, causal mask, heads, multi-head attention, the block and the parameter count. Each exercise has a check cell.

The check cell prints ✅ when the exercise passes. It tells you what is incorrect when the exercise does not pass. It
stops with `NotImplementedError` while the exercise still has `...` blanks. A `...` used as a numpy index, as in
`x[..., 0]`, is not a blank.

An eighth section, an order-blindness experiment, has nothing to fill in. Run it. Then answer the question after it.

`solutions/attention_practice.ipynb` is the filled-in version. `tools/build_notebooks.py` generates both notebooks.
`tests/test_practice.py` makes sure that the solutions run, that each blank stops at its own check, and that an answer
which indexes with `...` passes. The test needs only numpy and pytest.

Open either notebook in Jupyter or VS Code, or upload it to Colab.

`notebooks/01_transformer_walkthrough.ipynb` builds the original Transformer in PyTorch. It trains the model to put a
digit sequence in reverse order. In `notebooks/02_transformer_exercises.ipynb`, you write the attention, multi-head
attention, positional encoding, causal mask and decoder layer of that Transformer. A TEST cell follows each of them.

## Where to go next

- Make the copy length different for each example in lesson 3 (`K` random per example). A positional lookup can
  solve a constant offset. A variable offset needs an *induction head*, that is, two heads that work together. Find out
  if the model still trains.
- Add a KV cache to `generate()` in lesson 3. Then make sure that the output is identical.
- Replace `nn.LayerNorm` with RMSNorm, the position table with RoPE, and the MLP with SwiGLU. With these three
  changes, the lesson-3 model becomes a Llama-style model.
- Karpathy's `nanoGPT` is the same architecture, trained on real text. After these three files, it will probably
  look familiar.
