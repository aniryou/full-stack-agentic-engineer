"""Execute every SOLUTION notebook's code cells with stubbed models.

This can't judge answer quality (no real embedder here), but it proves the
solutions run without error and that the shape/maths self-checks pass — which
is what the blanks are graded on. The real embedding model only sharpens the
observational prints.
"""
import zlib
import numpy as np

import ragkit.embed as embed
from ragkit.corpus import tokenize
import tools_build_notebooks as tb


class StubEmbedder:
    """Deterministic bag-of-words hashing embedder (env-independent hash)."""
    dim = 1024

    def __init__(self):
        self.model_name = "stub-bow"

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, batch_size=64):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")


class StubCrossEncoder:
    """.predict([(q, passage), ...]) -> token-overlap score per pair."""
    def predict(self, pairs):
        out = []
        for q, p in pairs:
            qs, ps = set(tokenize(q)), set(tokenize(p))
            out.append(float(len(qs & ps)))
        return np.asarray(out, dtype="float32")


def run_notebook(name, cells):
    # patch model factories before exec; notebook `from ragkit.embed import ...`
    # binds these patched callables.
    embed.get_embedder = lambda *a, **k: StubEmbedder()
    embed.get_cross_encoder = lambda *a, **k: StubCrossEncoder()

    code = "\n\n".join(
        c.get("sol", c["src"]) for c in cells if c["t"] == "code"
    )
    ns = {}
    exec(compile(code, f"<{name}>", "exec"), ns)


if __name__ == "__main__":
    for name, cells in tb.NOTEBOOKS.items():
        run_notebook(name, cells)
        print("ok:", name)
    print(f"\nAll {len(tb.NOTEBOOKS)} solution notebooks executed cleanly.")
