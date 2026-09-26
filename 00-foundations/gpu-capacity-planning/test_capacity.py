"""Pins the numbers PRIMER.md quotes and checks the practice notebooks.
Run: python -m pytest -q test_capacity.py (standard library + pytest)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity as c  # noqa: E402

SMALL, H100 = c.MISTRAL_SMALL, c.GPUS["H100"]
RPS = 10_000 * 0.10 * 0.5 / 60


def test_one_unit_for_memory_kv_is_in_the_same_gb_as_weights_and_hbm():
    """KV used to come out in GiB (÷1024³) and was subtracted from decimal-GB HBM: ~7% too many sessions."""
    kv_bytes = 2 * 40 * 8 * 128 * 2
    assert kv_bytes == 163_840 and c.kv_per_token_kb(SMALL) == pytest.approx(163.84)
    assert c.kv_per_session_gb(SMALL, 8000) * 1e9 == pytest.approx(kv_bytes * 8000)
    assert c.weight_memory_gb(24, "bf16") * 1e9 == 48e9
    assert [round(c.kv_per_session_gb(SMALL, n), 2) for n in (8000, 32000, 128000)] == [1.31, 5.24, 20.97]
    assert round(c.kv_per_token_kb(SMALL, "fp8"), 2) == 81.92


def test_the_bank_example_sessions_per_gpu():
    spare = {dt: c.usable_hbm_gb(H100) - c.weight_memory_gb(24, dt) for dt in ("bf16", "fp8")}
    per = {dt: c.max_concurrent_sessions(spare[dt], SMALL, 1650, dt) for dt in spare}
    assert round(per["bf16"], 1) == 88.8 and round(per["fp8"], 1) == 355.1       # were 95.3 / 381.3 with GiB KV
    dur = c.request_duration_s(24, 1500, 300, H100)
    live = c.concurrency(RPS, dur)
    assert round(dur, 2) == 12.07 and round(live, 1) == 100.6
    assert math.ceil(live / per["bf16"]) == 2 and math.ceil(live / per["fp8"]) == 1
    agg, _ = c.decode_aggregate(SMALL, H100, int(live), 1650, "fp8")
    assert round(agg) == 8929 and round(c.prefill_tok_s(24, H100)) == 20615


def test_prefill_attention_term():
    """2·P·S leaves out causal attention: +22% at 32K and ~90% at 128K for Mistral Small."""
    extra = {n: c.attention_flops(SMALL, n) / c.prefill_flops(24, n) for n in (1500, 2048, 8192, 32768, 131072)}
    assert [round(100 * extra[n], 1) for n in extra] == [1.0, 1.4, 5.6, 22.4, 89.5]
    assert c.prefill_flops(24, 2000) == 96e12                                      # weights only by default
    assert c.prefill_flops(24, 2000, SMALL) == 96e12 + 2 * 40 * 32 * 128 * 2000 * 2001
    assert round(c.prefill_flops(24, 32000, SMALL) / 1e15, 2) == 1.87
    assert round(c.ttft_s(24, 2000, H100), 3) == 0.097 and round(c.ttft_s(24, 32000, H100, model=SMALL), 2) == 1.89
    assert c.prefill_flops(41, 2000, c.MISTRAL_LARGE) == c.prefill_flops(41, 2000)   # no q_heads: weights only


# --- the practice notebooks: the solution runs, and each check cell fails a wrong answer -------------------------
NB = Path(__file__).resolve().parent / "notebooks"
WRONG = {  # old answers that used to pass a single range assert, plus the old GiB units
    "decode_tok_s": "def decode_tok_s(weight_gb_, gpu):\n    return 70\n",
    "ttft_s": "def ttft_s(active_b, prompt_tokens, gpu, mfu=0.5):\n    return prompt_tokens / 20000\n",
    "concurrency": "def concurrency(rps, active_b, in_tok, out_tok, gpu, tpot_ms=40):\n    return 100\n",
    "kv_per_token_kb": "def kv_per_token_kb(m, dtype):\n"
                       "    return 2 * m['layers'] * m['kv_heads'] * m['head_dim'] * BYTES[dtype] / 1024\n",
    "max_sessions": "def max_sessions(spare_gb, m, context_tokens, dtype):\n"
                    "    return spare_gb / (kv_per_token_kb(m, dtype) * context_tokens / (1024 * 1024))\n",
}


def _run_notebook(name, monkeypatch, replace=None):
    """Execute a notebook's code cells in-process (they are standard library only); return the namespace."""
    import json
    monkeypatch.chdir(NB)
    monkeypatch.setattr(sys, "path", list(sys.path))
    cells = [("".join(cell["source"])) for cell in json.loads((NB / name).read_text())["cells"] if cell["cell_type"] == "code"]
    ns = {}
    for i, src in enumerate(cells):
        fn = next((f for f in (replace or {}) if src.startswith(f"def {f}(")), None)
        try:
            exec(compile(replace[fn] if fn else src, name, "exec"), ns)
        except Exception as e:
            e.cell = i                         # which code cell stopped, so a test can say it was the right one
            raise
    return ns


def _def_cell(name, fn):
    import json
    cells = [("".join(c["source"])) for c in json.loads((NB / name).read_text())["cells"] if c["cell_type"] == "code"]
    return next(i for i, src in enumerate(cells) if src.startswith(f"def {fn}("))


def test_practice_solution_runs_and_reproduces_the_bank(monkeypatch, capsys):
    ns = _run_notebook("01_capacity_practice_solved.ipynb", monkeypatch)
    assert (round(ns["per_gpu_fp8"], 1), round(ns["per_gpu_bf16"], 1)) == (355.1, 88.8)
    assert "163.84 kB (bf16)" in capsys.readouterr().out


@pytest.mark.parametrize("fn", sorted(WRONG))
def test_practice_checks_fail_a_wrong_answer(fn, monkeypatch):
    """The exercise's own check (the cell right after its def) fails it, not some later cell."""
    name = "01_capacity_practice_solved.ipynb"
    with pytest.raises(AssertionError) as err:
        _run_notebook(name, monkeypatch, {fn: WRONG[fn]})
    assert err.value.cell == _def_cell(name, fn) + 1


def test_practice_blank_stops_at_the_first_exercise(monkeypatch):
    with pytest.raises(NotImplementedError) as err:
        _run_notebook("01_capacity_practice.ipynb", monkeypatch)
    assert err.value.cell == _def_cell("01_capacity_practice.ipynb", "weight_gb") + 1   # its check, not its def
