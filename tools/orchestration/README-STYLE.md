# README style guide — write for a human who has 90 seconds

Every README in this repo (root, layer, topic, core, lab) is read by someone deciding *whether* and *how* to spend an
evening here. Optimise for that reader: what is this, what will I be able to do afterwards, where do I start, what does
it cost me. Structure first, detail second. No orchestration jargon (builder, fixer, T0 without definition).

## The shape of every README

1. **Title + one-sentence promise.** What this is and what you will be able to explain or do afterwards. One sentence, no list.
2. **Start here** — three numbered steps at most (read this → run this → open this notebook), each a link and one line.
   Include the fastest possible win (a command that prints something meaningful in under a minute).
3. **What you get** — a short table: path → what it teaches → time (e.g. "45 min") → tier.
   Define the tier once per README where it is first used, in plain words: *T0 = laptop or Colab CPU, free; T1 = one small GPU
   (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box, rented for an hour; T3 = the Google Cloud deployment, optional.*
4. **Run it** — the exact commands (install, tests, notebooks), copy-pasteable, verified.
5. **How it fits** — one short paragraph: what to read before, what this unlocks after, with links to the neighbouring layers
   and the primer sections it relies on. Only for topic/layer READMEs.
6. **Going further / caveats** — what is simulated vs measured, what needs real hardware or a cloud account and roughly what it costs,
   dated facts marked `(verify)`. Keep it honest and short.

Root and layer READMEs also get a **map**: the stack diagram (root) or the layer's place in it (layer), then one row per topic.

## Rules of thumb

- Lead with the outcome, not the mechanism: "Predict decode latency from a spec sheet" beats "Implements the roofline model".
- One idea per paragraph; paragraphs of at most four lines; tables for anything with three or more parallel items.
- Prefer a numbered path over a bulleted inventory. Inventories go in a table at the end.
- Say how long things take and what they cost. "Free, 20 minutes on a laptop" is information; "lightweight" is not.
- Name files as links (`[PRIMER.md](PRIMER.md)`), never as prose that the reader has to find.
- No marketing adjectives (powerful, comprehensive, seamless, production-grade). No emojis except ✅ in check output.
- Numbers only where they are computed by code in the repo or marked `(verify)` with a date.
- The **root README** must be regenerated to match the tree whenever content lands: the stack diagram, the "What is inside" table
  (one row per layer: topics → what you learn → paths), the notebook count, links to `CURRICULUM.md` and `COMPUTE.md` (once they
  are on the same branch), and the layout section. It is the front door: someone landing there must find the learning path
  in one click and any lab in two.
- **Layer READMEs** keep the `<!-- colab-links -->` markers; `tools/gen_colab_index.py` owns that section — never hand-edit it.
- Check every relative link (`python3 tools/orchestration/mdlinks.py <paths>`) before committing.

## A model to copy

```
# roofline-and-fabric — read a GPU spec sheet and predict what a model will do on it

After this topic you can look at a GPU, a model and a fabric and say how fast decode and prefill will run, why, and what breaks first.

## Start here
1. Read [PRIMER.md](PRIMER.md) §1–§3 (40 min): spec-sheet literacy and the roofline.
2. `cd roofline-core && python3 -m pytest -q` — 58 tests, ~1 s; then open `notebooks/01_spec_sheets_and_the_roofline.ipynb`.
3. When you have any GPU (even a free Colab T4): `gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb` measures the real thing.

## What you get
| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| ... | ... | ... | ... |
```
