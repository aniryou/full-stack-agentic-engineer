"""The kv-cache notebooks, read statically (executing them is `make check`, ~10 s more):

- the worked notebook is committed executed, error-free, and its saved numbers are what kerncore computes;
- the practice notebook is committed blank: every blank raises NotImplementedError in a `# YOUR CODE HERE` cell;
- neither needs torch: `import torch` appears only in 01's optional cell, inside try/except ImportError.
"""
import json
import re
from pathlib import Path

import pytest

from kerncore import kv

KV = Path(__file__).resolve().parents[2] / "kv-cache"
WORKED, PRACTICE = KV / "01_kv_cache_worked.ipynb", KV / "02_kv_cache_practice.ipynb"


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def src(cell):
    return "".join(cell["source"])


def text(path):
    out = []
    for c in cells(path):
        for o in c.get("outputs", []):
            out.append("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")))
    return "\n".join(out)


def test_worked_notebook_is_committed_executed_and_matches_kerncore():
    code = [c for c in cells(WORKED) if c["cell_type"] == "code"]
    printing = [c for c in code if "print(" in src(c) or "plt.show" in src(c)]
    assert all(c.get("outputs") for c in printing), "01 must be committed with its outputs (make notebooks)"
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
    out = text(WORKED)
    assert "IDENTICAL: True" in out and "same logits as kerncore.kv.TinyDecoder: True" in out
    for tokens, batch in ((8192, 1), (128_000, 1), (8192, 32)):
        assert kv.fmt_bytes(kv.kv_cache_bytes(32, 8, 128, tokens, batch)) in out
    assert "131,072 bytes = 128 KiB" in out and "at most 59" in out and "at most 119" in out
    assert "cached:    271 token-passes" in out and "cached does ~136x less token work" in out
    assert any("image/png" in o.get("data", {}) for c in code for o in c["outputs"]), "the cost plot is missing"


def test_practice_notebook_is_committed_blank_and_every_blank_stops():
    code = [c for c in cells(PRACTICE) if c["cell_type"] == "code"]
    assert not any(c.get("outputs") for c in code), "02 is committed without outputs"
    blanks = [c for c in code if "raise NotImplementedError(\"BLANK" in src(c)]
    names = re.findall(r'NotImplementedError\("BLANK ([A-D]):', "\n".join(src(c) for c in blanks))
    assert sorted(names) == ["A", "B", "C", "D"]
    assert all("# YOUR CODE HERE" in src(c) for c in blanks)


@pytest.mark.parametrize("path", [WORKED, PRACTICE], ids=lambda p: p.name)
def test_torch_only_in_the_optional_guarded_cell(path):
    for c in cells(path):
        s = src(c)
        if c["cell_type"] != "code" or not re.search(r"\bimport torch\b|\btorch\.[A-Za-z_]", s):
            continue
        assert path == WORKED and "OPTIONAL" in s, f"{path.name}: torch outside the optional cell"
        assert re.search(r"try:\s*\n\s*import torch\s*\nexcept ImportError:", s), "the torch import must be guarded"
    assert cells(path)[0]["metadata"].get("tags") == ["colab-bootstrap"]
