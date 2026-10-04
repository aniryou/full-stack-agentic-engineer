# LLM GPU Capacity Planning — primer + code + practice

This is a primer about how to size GPU fleets for LLM serving. Its examples are
Mistral Small 3 (24B dense) and Mistral Large 3 (675B MoE).

**The whole idea in three lines:**
- There are two constraints: **memory** (what fits) and **bandwidth/compute** (how fast).
- There are two SLOs: **TTFT** (prefill, compute-bound) and **TPOT** (decode, bandwidth-bound).
- The GPU count is the max over each constraint. Then add utilisation headroom + N+1.

**Time and tier:** ~2 h with the primer. This is module 00.2 in [`CURRICULUM.md`](../../CURRICULUM.md). The tier is T0: a laptop or Colab CPU, free. The code uses the standard library only, with no GPU and no key.

## Files
| File | What it is |
|---|---|
| `PRIMER.md` | The reference. It has the mental model, ~8 formulas, a worked example, MoE and a cheat-sheet. |
| `capacity.py` | Each formula as a small plain-Python function. It does not use numpy. |
| `worked_example.py` | It runs all of the examples: Mistral Small on H100, the bank and Mistral Large 3. |
| `test_capacity.py` | It pins the numbers that `PRIMER.md` quotes (units, the bank example, the attention term of prefill). It also does checks on the practice notebooks: the solution runs, and each check cell fails an incorrect answer. |
| `notebooks/01_capacity_practice.ipynb` | Fill-in-the-blank. Implement the 6 core functions. The checks compare them with `capacity.py` on several inputs. |
| `solutions/01_capacity_practice.ipynb` | The worked answers. This is the same notebook with all blanks filled in. |

## Run
```bash
python worked_example.py          # prints the numbers in PRIMER.md
python -m pytest -q test_capacity.py   # checks them (needs pytest)
jupyter notebook notebooks/       # do the practice (pure stdlib, no install); answers in solutions/
```

Start with `PRIMER.md`. Then run `worked_example.py`. Then do the practice notebook
from memory. All memory is in GB = 10⁹ bytes (weights, HBM and KV cache). The numbers
(layers, kv_heads, TFLOPS) come from the example that this topic uses. In practice, read
them from the `config.json` of the model and from the GPU spec sheet.
