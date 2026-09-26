# LLM GPU Capacity Planning — primer + code + practice

A primer for sizing GPU fleets for LLM serving, built around
Mistral Small 3 (24B dense) and Mistral Large 3 (675B MoE).

**The whole idea in three lines:**
- Two constraints: **memory** (what fits) and **bandwidth/compute** (how fast).
- Two SLOs: **TTFT** (prefill, compute-bound) and **TPOT** (decode, bandwidth-bound).
- GPU count = max over each constraint, then add utilisation headroom + N+1.

**Time and tier:** ~2 h with the primer; module 00.2 in [`CURRICULUM.md`](../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: standard library only, no GPU, no key.

## Files
| File | What it is |
|---|---|
| `PRIMER.md` | The reference. Mental model, ~8 formulas, worked example, MoE, cheat-sheet. |
| `capacity.py` | Every formula as a small plain-Python function. No numpy. |
| `worked_example.py` | Runs it all: Mistral Small on H100, the bank, Mistral Large 3. |
| `test_capacity.py` | Pins the numbers `PRIMER.md` quotes (units, the bank example, prefill's attention term) and checks the practice notebooks: the solution runs and each check cell fails a wrong answer. |
| `notebooks/01_capacity_practice.ipynb` | Fill-in-the-blank. Implement the 6 core functions; the checks compare them with `capacity.py` on several inputs. |
| `notebooks/01_capacity_practice_solved.ipynb` | Solutions. |

## Run
```bash
python worked_example.py          # prints the numbers in PRIMER.md
python -m pytest -q test_capacity.py   # checks them (needs pytest)
jupyter notebook notebooks/       # do the practice (pure stdlib, no install)
```

Start with `PRIMER.md`, run `worked_example.py`, then do the practice notebook
from memory. Memory is in GB = 10⁹ bytes throughout (weights, HBM and KV cache).
Numbers (layers, kv_heads, TFLOPS) are the running example — in real life read
them from the model's `config.json` and the GPU spec sheet.
