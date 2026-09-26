"""evalharness.py — measure the accuracy a scheme costs: lm-evaluation-harness at T1, a mini-eval at T0.

One idea: "accuracy" is three different measurements, and quantization can move them apart.
*Distribution distance* (mean KL between the reference and quantized next-token distributions,
argmax agreement) is cheap, sensitive and task-blind; *perplexity* is task-blind too; *task
accuracy* (exact match on held-out problems) is what users feel but is noisy — a 250-question
subset of gsm8k carries a standard error of about +-2.7 points, so small drops are invisible
unless you compare the same questions (paired flips, McNemar). This module builds the
``lm_eval`` command lines (lm-eval 0.4.13, verify; ``add_bos_token=True`` because quantized
models can be sensitive to it, ``--limit`` for smoke tests only), parses the results it writes,
and runs the same three measurements offline on the bundled tiny model (PRIMER §8 "Measuring
the accuracy you pay").
"""
from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .tinymodel import TASKS, TinyLM, make_task

SAMPLES = Path(__file__).parent / "data" / "samples"
LM_EVAL_VERSION = "0.4.13"


# ---------------------------------------------------------------------------------------------
# lm-evaluation-harness (T1)
# ---------------------------------------------------------------------------------------------
def lm_eval_command(model: str, *, backend: str = "vllm", tasks=("gsm8k",), num_fewshot: int | None = None,
                    limit: int | float | None = None, batch_size: str = "auto", base_url: str | None = None,
                    dtype: str = "auto", gpu_memory_utilization: float = 0.8, add_bos_token: bool = True,
                    num_concurrent: int = 4, output_path: str | None = "lm_eval_out", log_samples: bool = False,
                    apply_chat_template: bool = False, extra_model_args: dict | None = None) -> list:
    """argv for ``lm_eval``. Backends: ``vllm`` (in-process engine), ``hf`` (transformers), or
    ``local-completions`` (an already running OpenAI-compatible server, e.g. ``vllm serve``)."""
    if backend == "vllm":
        margs = {"pretrained": model, "tensor_parallel_size": 1, "dtype": dtype,
                 "gpu_memory_utilization": gpu_memory_utilization}
    elif backend == "hf":
        margs = {"pretrained": model, "dtype": dtype}
    elif backend == "local-completions":
        if not base_url:
            raise ValueError("local-completions needs base_url, e.g. http://127.0.0.1:8000/v1/completions")
        margs = {"model": model, "base_url": base_url, "num_concurrent": num_concurrent, "max_retries": 3,
                 "tokenized_requests": False}
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if add_bos_token:
        margs["add_bos_token"] = True
    margs.update(extra_model_args or {})
    cmd = ["lm_eval", "--model", backend, "--model_args", ",".join(f"{k}={v}" for k, v in margs.items()),
           "--tasks", ",".join(tasks), "--batch_size", str(batch_size)]
    if backend == "hf":
        cmd += ["--device", "cuda:0"]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if output_path:
        cmd += ["--output_path", output_path]
    if log_samples:
        cmd += ["--log_samples"]
    if apply_chat_template:
        cmd += ["--apply_chat_template"]
    return cmd


def run_lm_eval(cmd: list, timeout: float = 7200) -> subprocess.CompletedProcess:
    """T1: run it (needs ``pip install "lm_eval[vllm]"`` or ``"lm_eval[api]"`` and a GPU or a server)."""
    if not shutil.which(cmd[0]):
        raise RuntimeError(f"{cmd[0]} is not installed; run on a GPU box: pip install \"lm_eval[vllm]=={LM_EVAL_VERSION}\"")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)


def parse_results(obj_or_path) -> list:
    """Rows ``{task, metric, filter, value, stderr, n}`` from lm-eval's results JSON: ``results`` maps each
    task to ``{"<metric>,<filter>": value, "<metric>_stderr,<filter>": se, "alias": ...}``."""
    obj = obj_or_path if isinstance(obj_or_path, dict) else json.loads(Path(obj_or_path).read_text())
    rows = []
    for task, metrics in obj["results"].items():
        n = (obj.get("n-samples", {}).get(task) or {}).get("effective")
        for key, val in metrics.items():
            if key == "alias" or "_stderr" in key or not isinstance(val, (int, float)):
                continue
            metric, _, flt = key.partition(",")
            se = metrics.get(f"{metric}_stderr,{flt}")
            rows.append({"task": task, "metric": metric, "filter": flt or "none", "value": float(val),
                         "stderr": float(se) if isinstance(se, (int, float)) else None, "n": n})
    return rows


def parse_table(text: str) -> list:
    """Rows from the markdown table ``lm_eval`` prints (``|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|``)."""
    rows, task = [], None
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] in ("Tasks", "Groups") or (cells[0] and set(cells[0]) <= {"-", ":"}):
            continue
        task = cells[0] or task
        try:
            value = float(cells[6])
        except ValueError:
            continue
        se = float(cells[8]) if cells[8] not in ("", "N/A") else None
        rows.append({"task": task.lstrip(" -"), "metric": cells[4], "filter": cells[2], "value": value, "stderr": se,
                     "n": None})
    return rows


def compare(base: list, test: list) -> list:
    """Per (task, metric, filter): the change, the combined standard error ``sqrt(se_a^2 + se_b^2)``
    and a verdict. Unpaired, so conservative; paired flips on the same questions are sharper."""
    idx = {(r["task"], r["metric"], r["filter"]): r for r in base}
    out = []
    for r in test:
        b = idx.get((r["task"], r["metric"], r["filter"]))
        if b is None:
            continue
        d = r["value"] - b["value"]
        se = math.hypot(b["stderr"] or 0, r["stderr"] or 0)
        z = d / se if se else (math.copysign(math.inf, d) if d else 0.0)
        verdict = "within noise" if abs(z) < 2 else ("regression" if d < 0 else "improvement")
        out.append({"task": r["task"], "metric": r["metric"], "filter": r["filter"], "base": b["value"],
                    "test": r["value"], "delta": d, "se": se, "z": z, "verdict": verdict})
    return out


def sample_results(name: str) -> dict:
    """Bundled lm-eval results files — *sample output in the documented format (illustrative)*."""
    return json.loads((SAMPLES / f"lm_eval_{name}.json").read_text())


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def stderr(p: float, n: int) -> float:
    """Standard error of an accuracy ``p`` on ``n`` questions (lm-eval's ``acc_stderr`` uses n-1)."""
    return math.sqrt(p * (1 - p) / max(n - 1, 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for ``k`` successes in ``n`` — behaves at 0% and 100%, unlike p +- 2 se."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = z * z / n
    mid = (p + c / 2) / (1 + c)
    half = z * math.sqrt(p * (1 - p) / n + c / (4 * n)) / (1 + c)
    return (0.0 if k == 0 else max(0.0, mid - half), 1.0 if k == n else min(1.0, mid + half))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs: ``b`` right->wrong, ``c`` wrong->right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def logit_kl(ref_logits, test_logits) -> float:
    """Mean KL(p_ref || p_test) in nats over all positions (the last axis is the vocabulary)."""
    def logsm(z):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
    lp, lq = logsm(ref_logits), logsm(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


# ---------------------------------------------------------------------------------------------
# The offline mini-eval (T0)
# ---------------------------------------------------------------------------------------------
def mini_eval(model: TinyLM, ref: TinyLM | None = None, *, n: int = 500, seed: int = 1000,
              act_quant=None, kv_quant=None, ref_cache: dict | None = None) -> dict:
    """Per task: exact-match accuracy with a Wilson interval, answer perplexity, and — against
    ``ref`` — mean KL, argmax agreement and the paired flips. ``ref_cache`` avoids recomputing ``ref``."""
    out = {}
    for task in TASKS:
        prompts, answers = make_task(task, n, seed)
        logits = model.answer_logits(prompts, answers, act_quant=act_quant, kv_quant=kv_quant)
        gen = model.generate(prompts, answers.shape[1], act_quant=act_quant, kv_quant=kv_quant)
        right = (gen == answers).all(1)
        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        nll = -np.take_along_axis(logp, answers[..., None], -1).mean()
        row = {"accuracy": float(right.mean()), "ci95": wilson(int(right.sum()), n), "ppl": float(np.exp(nll)), "n": n}
        if ref is not None:
            key = (task, n, seed)
            if ref_cache is None or key not in ref_cache:
                r_logits = ref.answer_logits(prompts, answers)
                r_right = (ref.generate(prompts, answers.shape[1]) == answers).all(1)
                if ref_cache is not None:
                    ref_cache[key] = (r_logits, r_right)
            else:
                r_logits, r_right = ref_cache[key]
            b, c = int((r_right & ~right).sum()), int((~r_right & right).sum())
            row.update({"kl": logit_kl(r_logits, logits),
                        "argmax_agree": float((r_logits.argmax(-1) == logits.argmax(-1)).mean()),
                        "right_to_wrong": b, "wrong_to_right": c, "mcnemar_p": mcnemar_p(b, c)})
        out[task] = row
    return out


def table(results: dict) -> str:
    """``{scheme: mini_eval(...)}`` -> a markdown table, one row per scheme."""
    head = "| scheme | " + " | ".join(f"{t} acc | {t} KL | {t} argmax" for t in TASKS) + " |"
    rows = [head, "|---|" + "---|---|---|" * len(TASKS)]
    for name, r in results.items():
        cells = []
        for t in TASKS:
            x = r[t]
            cells += [f"{x['accuracy']:.1%}", f"{x.get('kl', 0):.2e}" if "kl" in x else "-",
                      f"{x['argmax_agree']:.1%}" if "argmax_agree" in x else "-"]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def shell(cmd: list) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
