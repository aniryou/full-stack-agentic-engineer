"""sft.py — sequence-level distillation at T1: SFT of a small Hugging Face model on a teacher's completions (TRL).

One idea: SeqKD is ordinary supervised fine-tuning; everything distillation-specific happened upstream, in
choosing, filtering and formatting the teacher's outputs (``distillab.teacher``). What remains are the settings
that decide whether it fits and trains on the card you have (PRIMER §3 and §10):

* **Format.** Prompt-completion rows (``{"prompt": [user], "completion": [assistant]}``) so the loss is on the
  completion only (TRL 1.14.0 ``SFTConfig.completion_only_loss=None`` does that for this format; verify).
* **Precision on a T4.** No bf16 on Turing, and TRL's configs turn ``bf16`` on unless ``fp16`` is set — so set
  ``fp16=True, bf16=False``, and keep the *trainable* weights in fp32: loading fp16 weights and training with
  ``fp16=True`` fails in torch's GradScaler ("Attempting to unscale FP16 gradients."). Full fine-tuning of a
  0.5B model is then 16 B/param = 7.9 GB before activations (``distillab.hf.memory``; predicted, verify).
* **LoRA** when memory is short (``peft_config``): r = 16 on all linear layers trains 1.8 % of Qwen2.5-0.5B.

* **Versions.** ``trl==1.14.0`` asks for ``transformers>=4.56.2``, and pip resolves that to transformers 5.x, which
  removed ``warmup_ratio`` (its ``warmup_steps`` reads a value below 1 as a ratio). :func:`for_config` adapts the
  kwargs to whatever config class is installed and names any field it does not know, before training starts.

``sft_kwargs`` and ``lora_kwargs`` are plain dictionaries (tested without TRL installed); :func:`train` imports
torch, transformers, datasets, trl and peft only when called. Pins: ``trl==1.14.0``, ``transformers>=4.56.2``
(5.17.0 when checked, 2026-09-27), ``peft`` 0.21.0 (verify).

    python -m distillab.hf.sft --model Qwen/Qwen2.5-0.5B-Instruct --data _run_outputs/teacher_pc.jsonl \\
        --out _run_outputs/student-sft [--lora-r 16] [--gpu T4]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

from .memory import BF16_OK


def for_config(config_cls, kwargs: dict) -> dict:
    """The builders' kwargs, adapted to the installed TRL/transformers ``config_cls`` (a dataclass).

    ``warmup_ratio`` becomes ``warmup_steps`` only where the field is gone (transformers 5.x, which reads
    ``warmup_steps`` < 1 as a ratio); on 4.x a float ``warmup_steps`` of 0.03 would mean no warmup at all. Any other
    field the installed class lacks raises here, naming it, instead of deep inside ``__init__``."""
    fields = {f.name for f in dataclasses.fields(config_cls)}
    kw = dict(kwargs)
    if "warmup_ratio" in kw and "warmup_ratio" not in fields:
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    unknown = sorted(set(kw) - fields)
    if unknown:
        raise TypeError(f"the installed {config_cls.__name__} has no field(s) {unknown}: check the trl/transformers pins")
    return kw


def sft_kwargs(*, out: str, gpu: str = "T4", lora: bool = False, epochs: float = 1.0, batch_size: int = 4,
               grad_accum: int = 4, max_length: int = 1024, lr: float | None = None, seed: int = 0) -> dict:
    """``SFTConfig`` arguments (TRL 1.14.0 names). Learning rate: TRL's 2e-5 default for full fine-tuning,
    1e-4 for LoRA (the tinker-cookbook's SFT guidance; verify for your data)."""
    bf16 = BF16_OK.get(gpu, True)
    return {"output_dir": out, "num_train_epochs": epochs, "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum, "learning_rate": lr or (1e-4 if lora else 2e-5),
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "max_length": max_length,
            "gradient_checkpointing": True, "bf16": bf16, "fp16": not bf16, "logging_steps": 10,
            "save_strategy": "no", "report_to": "none", "seed": seed}


def lora_kwargs(r: int = 16, alpha: int | None = None, target="all-linear") -> dict:
    """``peft.LoraConfig`` arguments. PEFT's own default targets only q_proj and v_proj for Qwen and Llama."""
    return {"r": r, "lora_alpha": alpha or 2 * r, "lora_dropout": 0.0, "target_modules": target,
            "task_type": "CAUSAL_LM"}


def train(model_id: str, data: str, out: str, *, gpu: str = "T4", lora_r: int = 0, **kw):
    """Run the SFT (T1: needs a GPU, torch, transformers, datasets, trl, and peft for LoRA)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id)
    bf16 = BF16_OK.get(gpu, True)
    # trainable weights in fp32 on a T4 (fp16 autocast); a frozen LoRA base can be 16-bit (verify that PEFT keeps
    # the adapters in fp32: peft's autocast_adapter_dtype, default True in recent releases)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if lora_r else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    peft_config = None
    if lora_r:
        from peft import LoraConfig
        peft_config = LoraConfig(**lora_kwargs(lora_r))
    ds = load_dataset("json", data_files=data, split="train")
    args = SFTConfig(**for_config(SFTConfig, sft_kwargs(out=out, gpu=gpu, lora=bool(lora_r), **kw)))
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m distillab.hf.sft", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="prompt-completion JSONL (distillab.teacher.to_prompt_completion)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="T4", choices=sorted(BF16_OK))
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true", help="print the configuration and exit")
    a = ap.parse_args(argv)
    kw = dict(epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum, max_length=a.max_length)
    print("SFTConfig:", sft_kwargs(out=a.out, gpu=a.gpu, lora=bool(a.lora_r), **kw))
    if a.lora_r:
        print("LoraConfig:", lora_kwargs(a.lora_r))
    if a.dry_run:
        return 0
    from ..env import hf_stack
    missing = [m for m, ok in hf_stack().items() if not ok and m in ("transformers", "trl", "datasets")]
    if missing:
        print(f"missing {missing}: pip install 'trl==1.14.0' 'transformers>=4.56.2' datasets peft", file=sys.stderr)
        return 2
    train(a.model, a.data, a.out, gpu=a.gpu, lora_r=a.lora_r, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
