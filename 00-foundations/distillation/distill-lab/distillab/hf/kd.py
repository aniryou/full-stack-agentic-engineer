"""kd.py — logit distillation of a Hugging Face student from a Hugging Face teacher (T1; torch + transformers).

One idea: logit KD needs the teacher's *full* next-token distribution at every position the student learns, which
an API does not return (151,936 numbers per token for Qwen) — so the teacher runs in-process, frozen, beside
the student, and the loss is Hinton's (PRIMER §2 "Soft targets, temperature and the choice of divergence"):

    loss = α · T² · KL(softmax(z_teacher / T) ‖ softmax(z_student / T)) + (1 − α) · CE(labels, softmax(z_student))

on the assistant tokens only (prompt positions carry label −100). Per step: the teacher's forward under
``torch.no_grad()`` in 16-bit, the student's forward with gradients, :func:`distillab.losses.kd_loss` computed
``chunk`` rows at a time so the float32 softmaxes of two ``[tokens, 151,936]`` tensors never sit in memory at
once (PRIMER §10), fp16 autocast with a GradScaler on a T4 (fp32 trainable weights; bf16 elsewhere).

Requirements the code checks before training: the same ``vocab_size`` (the positions of the two distributions
must mean the same tokens — Qwen2.5-0.5B/1.5B share 151,936; Qwen2.5-7B has 152,064) and one tokenizer and chat
template for both (the rows are rendered once, with the *student's* template). Memory on a T4: full KD of a
0.5B student from a 1.5B teacher at 4 × 512 tokens does not fit unchunked (17.4 GB predicted); LoRA, B = 1–2
or chunking does (``distillab.hf.memory.plan``; verify).

    python -m distillab.hf.kd --teacher Qwen/Qwen2.5-1.5B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct \\
        --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-kd --temperature 2 --alpha 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .memory import BF16_OK


@dataclass
class KDConfig:
    teacher: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data: str = "_run_outputs/teacher_msgs.jsonl"     # conversational rows: {"messages": [user, assistant]}
    out: str = "_run_outputs/student-kd"
    gpu: str = "T4"
    temperature: float = 2.0
    alpha: float = 0.9                                 # weight on the soft (teacher) term
    lr: float = 2e-5
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    max_length: int = 512
    chunk: int = 256                                   # rows of the KD loss materialised at once
    lora_r: int = 0                                    # 0 = full fine-tuning
    gradient_checkpointing: bool = True
    seed: int = 0


def check_pair(teacher_config: dict, student_config: dict) -> list:
    """Reasons the pair cannot be logit-distilled (empty = fine)."""
    out = []
    if teacher_config.get("vocab_size") != student_config.get("vocab_size"):
        out.append(f"vocab_size differs ({teacher_config.get('vocab_size')} vs {student_config.get('vocab_size')}): "
                   "the two distributions are over different token sets")
    if teacher_config.get("model_type") != student_config.get("model_type"):
        out.append(f"model families differ ({teacher_config.get('model_type')} vs {student_config.get('model_type')}): "
                   "check that tokenizer and chat template are identical before trusting the positions to line up")
    return out


def _ids(out) -> list:
    """``apply_chat_template(..., tokenize=True)`` returns a list of ids in transformers 4.x and a ``BatchEncoding``
    in 5.x (MIGRATION_GUIDE_V5.md §4); accept both."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def encode(tokenizer, row: dict, max_length: int) -> dict:
    """input_ids and labels for one conversation: −100 on the prompt, the assistant turn's tokens as labels."""
    msgs = row["messages"]
    prompt_ids = _ids(tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = _ids(tokenizer.apply_chat_template(msgs, tokenize=True))[:max_length]
    n = min(len(prompt_ids), len(full_ids))
    return {"input_ids": full_ids, "labels": [-100] * n + full_ids[n:]}


def _collate(batch: list, pad_id: int):
    import torch
    w = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), w), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), w), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), w), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    return ids, labels, attn


def train(cfg: KDConfig) -> dict:
    """The KD loop (T1). Returns the logged losses; saves the student (LoRA merged) to ``cfg.out``."""
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..losses import kd_loss

    torch.manual_seed(cfg.seed)
    bf16 = BF16_OK.get(cfg.gpu, True)
    half = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(cfg.student)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher, dtype=half).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(cfg.student, dtype=half if (bf16 or cfg.lora_r) else torch.float32)
    problems = check_pair(teacher.config.to_dict(), student.config.to_dict())
    if any("vocab_size" in p for p in problems):
        raise ValueError(problems)
    if cfg.lora_r:
        from peft import LoraConfig, get_peft_model

        from .sft import lora_kwargs
        student = get_peft_model(student, LoraConfig(**lora_kwargs(cfg.lora_r)))
        for p in student.parameters():            # trainable adapters in fp32 (the GradScaler needs it on a T4)
            if p.requires_grad:
                p.data = p.data.float()
    if cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        if cfg.lora_r:
            student.enable_input_require_grads()
    student.cuda().train()
    rows = [json.loads(line) for line in Path(cfg.data).read_text().splitlines() if line.strip()]
    data = [encode(tok, r, cfg.max_length) for r in rows]
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    log, step = [], 0
    rng = random.Random(cfg.seed)
    for _ in range(cfg.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), cfg.batch_size):
            ids, labels, attn = (t.cuda() for t in _collate(data[i:i + cfg.batch_size], tok.pad_token_id or 0))
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=attn).logits[:, :-1]
            with torch.autocast("cuda", dtype=half):
                s_logits = student(input_ids=ids, attention_mask=attn).logits[:, :-1]
            y = labels[:, 1:]
            mask = y != -100
            loss = kd_loss(s_logits, t_logits, y, mask, cfg.temperature, cfg.alpha, cfg.chunk)
            scaler.scale(loss / cfg.grad_accum).backward()
            if (i // cfg.batch_size + 1) % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "loss": round(loss.item(), 4)})
                if step % 10 == 0:
                    print(f"step {step}  kd loss {loss.item():.4f}", flush=True)
    if cfg.lora_r:
        student = student.merge_and_unload()
    student.save_pretrained(cfg.out)
    tok.save_pretrained(cfg.out)
    Path(cfg.out, "kd_log.json").write_text(json.dumps({"config": asdict(cfg), "log": log}, indent=1))
    return {"steps": step, "log": log}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m distillab.hf.kd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = KDConfig()
    for f, v in asdict(d).items():
        kind = type(v) if not isinstance(v, bool) else None
        if kind is None:
            ap.add_argument("--" + f.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument("--" + f.replace("_", "-"), type=kind, default=v)
    ap.add_argument("--dry-run", action="store_true", help="print the configuration and the memory plan, then exit")
    a = vars(ap.parse_args(argv))
    dry = a.pop("dry_run")
    cfg = KDConfig(**a)
    print("KDConfig:", asdict(cfg))
    if dry:
        return 0
    from ..env import gpu_name, has_torch, hf_stack
    if not (has_torch() and hf_stack()["transformers"] and gpu_name()):
        print("kd.py is T1: it needs a CUDA GPU, torch and transformers (pip install 'transformers>=4.56.2' peft). "
              "At T0, notebook 01 trains the same loss on the tiny transformers.", file=sys.stderr)
        return 2
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
