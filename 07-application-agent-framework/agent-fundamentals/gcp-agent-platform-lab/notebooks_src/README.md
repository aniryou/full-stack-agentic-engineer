# Authoring notebooks

The sources in this folder are Python files in the *percent* cell format.
`tools/build_notebooks.py` turns each source into an exercise notebook (`notebooks/`) and a
solution notebook (`solutions/`).

Cell markers:

    # %% [markdown]      prose (each line prefixed with "# ")
    # %%                 ordinary code cell
    # %% exercise        code cell containing ### BEGIN SOLUTION / ### END SOLUTION blocks
    # %% check           self-test cell that asserts the exercise is correct (kept in both variants)

These rules keep the pipeline honest:

1. A solution block must be one or more **complete statements** (a function body, an assignment,
   a class). It must never be a fragment inside an expression. This is because the exercise
   variant replaces the block with `raise NotImplementedError(...)` at the same indentation.
2. Put a check cell after every exercise cell. The check cell fails loudly until the solution to
   the exercise is correct. When the check passes, it prints a ✅ line.
3. Notebooks run offline, in under ~30 s, with no network and no API keys. Use `FakeLLM`.
4. You can use top-level `await` (Jupyter and nbclient support it).
5. Start with a heading, a **Concept map** line and a 3-bullet "in this notebook you will".
   The **Concept map** line names docs/PRIMER_MAP.md and the in-repo primer that goes deeper.
   End with a **The one-minute version** cell. That cell tells how to explain this topic in a design review.
6. `make check` (or `python tools/run_notebooks.py solutions`) must pass. That is, every solution
   notebook executes clean end to end. `python tools/run_notebooks.py notebooks --expect-fail`
   makes sure that the exercise variants stop at the first unsolved exercise.
