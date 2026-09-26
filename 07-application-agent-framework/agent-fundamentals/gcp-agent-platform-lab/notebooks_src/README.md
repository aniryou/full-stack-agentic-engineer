# Authoring notebooks

Sources here are Python files in the *percent* cell format; `tools/build_notebooks.py`
turns each into an exercise notebook (`notebooks/`) and a solution notebook (`solutions/`).

Cell markers:

    # %% [markdown]      prose (each line prefixed with "# ")
    # %%                 ordinary code cell
    # %% exercise        code cell containing ### BEGIN SOLUTION / ### END SOLUTION blocks
    # %% check           self-test cell that asserts the exercise is correct (kept in both variants)

Rules that keep the pipeline honest:

1. A solution block must be one or more **complete statements** (a function body, an assignment,
   a class), never a fragment inside an expression — the exercise variant replaces the block with
   `raise NotImplementedError(...)` at the same indentation.
2. Every exercise cell is followed by a check cell that fails loudly until the exercise is solved
   and prints a ✅ line when it passes.
3. Notebooks run offline, in under ~30 s, with no network and no API keys. Use `FakeLLM`.
4. Top-level `await` is fine (Jupyter and nbclient support it).
5. Start with a heading, a **Concept map** line (docs/PRIMER_MAP.md plus the in-repo primer that goes deeper), and a 3-bullet "in this notebook you will".
   End with a **The one-minute version** cell: how to explain this topic in a design review.
6. `make check` (or `python tools/run_notebooks.py solutions`) must pass: every solution notebook
   executes clean end to end. `python tools/run_notebooks.py notebooks --expect-fail` verifies the
   exercise variants stop at the first unsolved exercise.
