"""gkd.py — on-policy distillation at T1 with TRL: ``GKDTrainer`` (experimental) and ``DistillationTrainer``.

One idea: in TRL 1.14.0 on-policy distillation is configuration, and the configuration hides three facts that
change what you train (PRIMER §4 "On-policy distillation"; read against ``trl/experimental/gkd/gkd_trainer.py``):

* **Where it lives.** ``from trl.experimental.gkd import GKDConfig, GKDTrainer`` — ``from trl import GKDTrainer``
  fails on 1.x. ``from trl import DistillationConfig, DistillationTrainer`` is the stable, always-on-policy
  alternative (the ``trl.experimental.distillation`` path is a deprecated shim).
* **What the knobs mean.** ``lmbda`` (default 0.5) is a per-*batch* coin flip: with that probability the student
  generates the batch's completions (on-policy); otherwise the dataset's completions are used, or the teacher's
  when ``seq_kd=True``. ``beta`` (0.5) picks the divergence: 0 = forward KL(teacher ‖ student), 1 = reverse KL
  (student ‖ teacher), in between the generalised JSD. ``DistillationConfig.beta`` defaults to 1.0 (reverse KL).
* **Temperature and T².** ``GKDConfig.temperature`` (0.9) only sets *sampling*; its loss runs at T = 1.
  ``DistillationConfig.temperature`` (1.0) divides both logits inside the loss. Nothing in TRL multiplies by T².

* **Data.** ``GKDTrainer`` reads conversational rows (``teacher_msgs.jsonl``); ``DistillationTrainer`` needs a
  ``prompt`` column and generates the completions itself, so pass ``teacher_pc.jsonl`` (its ``completion`` column
  is ignored) or a messages file, which :func:`train` converts with :func:`prompt_only`.

Both trainers check ``vocab_size`` equality and raise ``ValueError`` otherwise ("GKD compares the teacher's full
next-token distribution, which requires a shared vocabulary"); ``check_vocab`` does the same before anything
downloads. On a T4 set ``fp16=True, bf16=False`` (both configs default ``bf16=True`` when ``fp16`` is unset).
The kwargs builders are plain dictionaries; :func:`train` imports TRL only when called. Pins: ``trl==1.14.0``
(verify; its GKD trainer and config are byte-identical on main as of 2026-09-26).

    python -m distillab.hf.gkd --teacher Qwen/Qwen2.5-1.5B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct \\
        --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-gkd --lmbda 0.5 --beta 0.5
    python -m distillab.hf.gkd --trainer distillation --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-od
"""
from __future__ import annotations

import argparse
import sys

from .memory import BF16_OK
from .sft import for_config

GKD_DEFAULTS = {"temperature": 0.9, "lmbda": 0.5, "beta": 0.5, "max_new_tokens": 128, "disable_dropout": True,
                "seq_kd": False}                        # TRL 1.14.0 GKDConfig (verify)
DISTILLATION_DEFAULTS = {"learning_rate": 1e-6, "max_completion_length": 512, "temperature": 1.0, "beta": 1.0,
                         "use_vllm": False}             # TRL 1.14.0 DistillationConfig (verify)


def gkd_kwargs(*, out: str, teacher: str, gpu: str = "T4", lmbda: float = 0.5, beta: float = 0.5,
               temperature: float = 0.9, max_new_tokens: int = 128, seq_kd: bool = False, lr: float = 5e-5,
               batch_size: int = 2, grad_accum: int = 8, max_length: int = 512, epochs: float = 1.0) -> dict:
    """``GKDConfig`` arguments (an ``SFTConfig`` subclass). ``lmbda`` and ``beta`` must lie in [0, 1]."""
    for name, v in (("lmbda", lmbda), ("beta", beta)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1] (GKDConfig.__post_init__ raises otherwise)")
    bf16 = BF16_OK.get(gpu, True)
    return {"output_dir": out, "teacher_model_name_or_path": teacher, "lmbda": lmbda, "beta": beta,
            # the frozen teacher in 16-bit (GKDTrainer converts this string with getattr(torch, ...))
            "teacher_model_init_kwargs": {"dtype": "bfloat16" if bf16 else "float16"},
            "temperature": temperature, "max_new_tokens": max_new_tokens, "seq_kd": seq_kd,
            "learning_rate": lr, "per_device_train_batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
            "max_length": max_length, "num_train_epochs": epochs, "gradient_checkpointing": True,
            "bf16": bf16, "fp16": not bf16, "logging_steps": 10, "save_strategy": "no", "report_to": "none"}


def distillation_kwargs(*, out: str, teacher: str, gpu: str = "T4", beta: float = 1.0, max_completion_length: int = 256,
                        lr: float = 1e-6, batch_size: int = 2, grad_accum: int = 8, use_vllm: bool = False) -> dict:
    """``DistillationConfig`` arguments: always on-policy, ``beta`` picks the divergence (1.0 = reverse KL).
    Its dataset is prompt-only: ``{"prompt": [{"role": "user", "content": ...}]}``."""
    bf16 = BF16_OK.get(gpu, True)
    return {"output_dir": out, "teacher_model_name_or_path": teacher, "beta": beta,
            "teacher_model_init_kwargs": {"dtype": "bfloat16" if bf16 else "float16"},
            "max_completion_length": max_completion_length, "learning_rate": lr,
            "per_device_train_batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
            "use_vllm": use_vllm, "bf16": bf16, "fp16": not bf16, "report_to": "none"}


def prompt_only(row: dict) -> dict:
    """A conversational row as ``DistillationTrainer`` wants it: the messages before the last assistant turn."""
    msgs = row["messages"]
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=len(msgs))
    if last == 0:
        raise ValueError("a row with no message before its assistant turn has no prompt")
    return {"prompt": msgs[:last]}


def check_vocab(teacher_config: dict, student_config: dict) -> None:
    t, s = teacher_config.get("vocab_size"), student_config.get("vocab_size")
    if t != s:
        raise ValueError(f"student vocab_size {s} != teacher vocab_size {t}: GKD compares the teacher's full "
                         "next-token distribution, which requires a shared vocabulary (TRL's GOLD trainer handles "
                         "different tokenizers)")


def train(student: str, data: str, out: str, *, trainer: str = "gkd", **kw):
    """T1: run TRL's GKDTrainer on conversational rows, or DistillationTrainer on prompt-only rows."""
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    teacher = kw["teacher"]
    check_vocab(AutoConfig.from_pretrained(teacher).to_dict(), AutoConfig.from_pretrained(student).to_dict())
    tok = AutoTokenizer.from_pretrained(student)
    ds = load_dataset("json", data_files=data, split="train")
    bf16 = BF16_OK.get(kw.get("gpu", "T4"), True)
    model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16 if bf16 else torch.float32)
    if trainer == "gkd":
        from trl.experimental.gkd import GKDConfig, GKDTrainer
        args = GKDConfig(**for_config(GKDConfig, gkd_kwargs(out=out, **kw)))
        t = GKDTrainer(model=model, teacher_model=teacher, args=args, train_dataset=ds, processing_class=tok)
    else:
        from trl import DistillationConfig, DistillationTrainer
        if "prompt" not in ds.column_names:
            if "messages" not in ds.column_names:
                raise ValueError(f"{data}: DistillationTrainer needs a 'prompt' or a 'messages' column")
            ds = ds.map(prompt_only, remove_columns=ds.column_names)
        keep = {k: v for k, v in kw.items() if k in ("teacher", "gpu", "beta", "max_completion_length", "lr",
                                                    "batch_size", "grad_accum", "use_vllm")}
        args = DistillationConfig(**for_config(DistillationConfig, distillation_kwargs(out=out, **keep)))
        t = DistillationTrainer(model=model, teacher_model=teacher, args=args, train_dataset=ds, processing_class=tok)
    t.train()
    t.save_model(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m distillab.hf.gkd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer", choices=["gkd", "distillation"], default="gkd")
    ap.add_argument("--gpu", default="T4", choices=sorted(BF16_OK))
    ap.add_argument("--lmbda", type=float, default=0.5)
    ap.add_argument("--beta", type=float, help="0 forward KL, 1 reverse (default: 0.5 for gkd, 1.0 for distillation)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seq-kd", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512, help="GKD: prompt + completion tokens per row")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    beta = a.beta if a.beta is not None else (GKD_DEFAULTS if a.trainer == "gkd" else DISTILLATION_DEFAULTS)["beta"]
    kw = dict(teacher=a.teacher, gpu=a.gpu, beta=beta, batch_size=a.batch_size)
    if a.trainer == "gkd":
        kw.update(lmbda=a.lmbda, temperature=a.temperature, max_new_tokens=a.max_new_tokens, seq_kd=a.seq_kd,
                  max_length=a.max_length)
        print("GKDConfig:", gkd_kwargs(out=a.out, **kw))
    else:
        print("DistillationConfig:", distillation_kwargs(out=a.out, **kw))
    if a.dry_run:
        return 0
    from ..env import gpu_name, hf_stack
    if not (gpu_name() and hf_stack()["trl"] and hf_stack()["transformers"]):
        print("gkd.py is T1: it needs a CUDA GPU and pip install 'trl==1.14.0' 'transformers>=4.56.2' datasets. "
              "At T0, notebook 01 trains GKD on the tiny transformers.", file=sys.stderr)
        return 2
    train(a.student, a.data, a.out, trainer=a.trainer, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
