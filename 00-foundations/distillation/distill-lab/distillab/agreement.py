"""agreement.py — measuring a student against its teacher: agreement, task accuracy with intervals, the gap.

One idea: two families of numbers answer two different questions (PRIMER §8 "Measuring a student"):

    agreement      does the student *copy* the teacher?   mean KL(p_teacher ‖ p_student) per position, top-1
                                                           agreement, top-k overlap — on the same positions
    task accuracy  does the student *solve* the task?     verifier accuracy on held-out problems, with a Wilson
                                                           interval, per difficulty (where the gap hides)

They can disagree: a student trained on the teacher's *verified* outputs can beat its teacher on the task while
agreeing with it less (the tiny run in notebook 01 shows it). The definitions of ``kl`` and ``argmax_agreement``
are the quantization topic's (``quantcore.eval.kl`` and ``quantcore.granularity.argmax_agreement``) and
``wilson_interval`` is the one the 07 agent lab and ``memory-core`` use; ``tests/test_repo_numbers.py`` checks all
three against those files. numpy only; the Hugging Face path (T1) imports lazily.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, float)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kl(ref_logits, test_logits) -> float:
    """Mean KL(p_ref ‖ p_test) in nats per position (rows)."""
    lp, lq = log_softmax(ref_logits), log_softmax(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def argmax_agreement(ref_logits, test_logits) -> float:
    """Share of positions whose top-1 token is the same."""
    return float((np.argmax(ref_logits, -1) == np.argmax(test_logits, -1)).mean())


def topk_overlap(ref_logits, test_logits, k: int = 5) -> float:
    """Mean |top-k(ref) ∩ top-k(test)| / k over positions: agreement on the candidates, not just the winner."""
    r = np.argsort(-np.asarray(ref_logits, float), -1)[..., :k]
    t = np.argsort(-np.asarray(test_logits, float), -1)[..., :k]
    r2, t2 = r.reshape(-1, k), t.reshape(-1, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(r2, t2)]))


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a pass rate: sane at 0/n and n/n, unlike p ± 1.96·SE."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = passes / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired(teacher_ok: list, student_ok: list) -> dict:
    """The same items scored for both: accuracies, the flips behind the difference, and McNemar's z
    (|z| < 2: the difference is within noise). Only the flips carry information about the gap."""
    lost = sum(t and not s for t, s in zip(teacher_ok, student_ok))
    gained = sum(s and not t for t, s in zip(teacher_ok, student_ok))
    n = len(teacher_ok)
    flips = lost + gained
    return {"n": n, "teacher": sum(teacher_ok) / n, "student": sum(student_ok) / n, "lost": lost, "gained": gained,
            "mcnemar_z": 0.0 if flips == 0 else (gained - lost) / math.sqrt(flips)}


def gap_table(records: list, key: str = "difficulty") -> list:
    """Records ``{key, "teacher": bool, "student": bool}`` → per ``key``: n, each accuracy with its Wilson
    interval, and the gap. The average gap is a weighted mean of these rows; the tail rows are where it lives."""
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    rows = []
    for g in sorted(groups):
        rs = groups[g]
        n, t, s = len(rs), sum(r["teacher"] for r in rs), sum(r["student"] for r in rs)
        tl, th = wilson_interval(t, n)
        sl, sh = wilson_interval(s, n)
        rows.append({key: g, "n": n, "teacher": round(t / n, 3), "teacher 95% CI": f"{tl:.2f}-{th:.2f}",
                     "student": round(s / n, 3), "student 95% CI": f"{sl:.2f}-{sh:.2f}", "gap": round((t - s) / n, 3)})
    return rows


def lm_eval_command(model: str, tasks: str = "gsm8k", *, backend: str = "vllm", limit: int | None = 250,
                    num_fewshot: int = 5, thinking: bool = False, dtype: str = "half") -> list:
    """An lm-evaluation-harness (0.4.13, verify) command line. gsm8k is greedy with 256 new tokens by default —
    wrong for thinking models (pass ``thinking=True``: chat template, ``enable_thinking``, ``think_end_token``
    and a larger ``max_gen_toks``). ``limit`` is for testing: report the stderr lm-eval prints next to it."""
    args = f"pretrained={model},dtype={dtype}"
    if thinking:
        args += ",enable_thinking=True,think_end_token=</think>"
    cmd = ["lm_eval", "--model", backend, "--model_args", args, "--tasks", tasks, "--num_fewshot", str(num_fewshot),
           "--batch_size", "auto"]
    if thinking:
        cmd += ["--apply_chat_template", "--gen_kwargs", "max_gen_toks=4096,temperature=0.6,top_p=0.95"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_lm_eval(model: str, tasks: str = "gsm8k", *, output_path: str = "_run_outputs/lm_eval", **kw) -> dict | None:
    """T1: run lm-evaluation-harness when it is installed and return its results JSON (the newest file it wrote);
    otherwise print the command and return None. ``kw`` goes to :func:`lm_eval_command`."""
    import json
    import shutil
    import subprocess
    from pathlib import Path
    cmd = lm_eval_command(model, tasks, **kw) + ["--output_path", output_path]
    if not shutil.which("lm_eval"):
        print("lm_eval is not installed (pip install 'lm-eval[vllm]==0.4.13'); on a GPU run:\n  " + " ".join(cmd))
        return None
    subprocess.run(cmd, check=True)
    files = sorted(Path(output_path).rglob("results*.json"), key=lambda f: f.stat().st_mtime)
    return json.loads(files[-1].read_text()) if files else None


def hf_position_logits(model_id: str, texts: list, device: str = "cuda", dtype: str = "float16", max_len: int = 512):
    """T1: next-token logits of a Hugging Face causal LM on ``texts`` (list of ``(B, T, V)`` numpy arrays, one
    per text; lazy imports). Teacher and student must share a tokenizer for the positions to line up."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out.append(model(**ids).logits[0].float().cpu().numpy())
    return out


def hf_agreement(teacher_id: str, student_id: str, texts: list, *, k: int = 5, device: str = "cuda") -> dict:
    """T1: teacher-student agreement over every position of ``texts`` (held-out prompts with answers): mean KL,
    top-1 agreement and top-k overlap. The two models must share a tokenizer, or the positions do not line up."""
    import numpy as _np
    t = hf_position_logits(teacher_id, texts, device)
    s = hf_position_logits(student_id, texts, device)
    if any(a.shape != b.shape for a, b in zip(t, s)):
        raise ValueError("teacher and student logits differ in shape: different tokenizers or vocab_size")
    T, S = _np.concatenate(t), _np.concatenate(s)
    return {"positions": len(T), "kl": kl(T, S), "top1": argmax_agreement(T, S), f"top{k}": topk_overlap(T, S, k)}
