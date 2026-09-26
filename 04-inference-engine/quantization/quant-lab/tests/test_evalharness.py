"""lm-eval commands and results; the statistics that decide whether a drop is real."""
import math

import pytest

from quantlab import evalharness as E


def test_lm_eval_commands():
    v = E.lm_eval_command("m", tasks=["gsm8k"], num_fewshot=5, limit=250)
    assert v[:4] == ["lm_eval", "--model", "vllm", "--model_args"] and "add_bos_token=True" in v[4]
    assert v[v.index("--limit") + 1] == "250" and v[v.index("--num_fewshot") + 1] == "5"
    s = E.lm_eval_command("m", backend="local-completions", base_url="http://127.0.0.1:8000/v1/completions")
    assert "base_url=http://127.0.0.1:8000/v1/completions" in s[4] and "tokenized_requests=False" in s[4]
    with pytest.raises(ValueError):
        E.lm_eval_command("m", backend="local-completions")


def test_parse_sample_results_and_table():
    rows = E.parse_results(E.sample_results("bf16"))
    strict = next(r for r in rows if r["filter"] == "strict-match")
    assert strict["value"] == 0.3 and strict["stderr"] == pytest.approx(E.stderr(0.3, 250), abs=1e-4) and strict["n"] == 250
    table = E.parse_table((E.SAMPLES / "lm_eval_table_bf16.txt").read_text())
    assert {(r["filter"], r["value"]) for r in table} == {(r["filter"], r["value"]) for r in rows}


def test_a_three_point_drop_on_250_questions_is_within_noise():
    cmp = E.compare(E.parse_results(E.sample_results("bf16")), E.parse_results(E.sample_results("w4a16")))
    strict = next(c for c in cmp if c["filter"] == "strict-match")
    assert strict["delta"] == pytest.approx(-0.028) and strict["verdict"] == "within noise" and abs(strict["z"]) < 1


def test_statistics():
    assert E.stderr(0.768, 250) == pytest.approx(0.0268, abs=2e-4)          # vLLM's FP8 gsm8k example: 0.768 +- 0.0268
    lo, hi = E.wilson(500, 500)
    assert hi == 1.0 and lo > 0.99
    assert E.mcnemar_p(0, 0) == 1.0 and E.mcnemar_p(10, 0) < 0.01 and E.mcnemar_p(5, 5) == 1.0
    assert E.logit_kl([[0.0, 0.0]], [[0.0, 0.0]]) == 0.0
    p, q = [[math.log(0.9), math.log(0.1)]], [[math.log(0.5), math.log(0.5)]]
    assert E.logit_kl(p, q) == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))
