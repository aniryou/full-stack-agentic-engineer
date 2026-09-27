"""train.py — a tiny teacher, and a narrower student trained four ways on the same budget (torch; a CPU is fine).

One idea: the four ways differ only in *what the student is asked to match* and *on whose samples*:

    method   targets                                            sequences
    hard     the labelled tokens (one-hot)                      the labelled set (an answer key: 80% skip the working)
    kd       α·T²·KL(teacher_T ‖ student_T) + (1−α)·CE          the labelled set, with the teacher's logits at every position
    seqkd    the teacher's sampled tokens (one-hot)             the teacher's completions for the labelled prompts,
                                                                 kept only when the verifier passes them
             (``seqkd_all``: every teacher completion, unfiltered — the recipe for a speculative draft, §7)
    gkd      generalised JSD(β) to the teacher's distribution   the student's own samples with probability λ,
                                                                 else the labelled set (TRL's GKDTrainer, seq_kd=False)

Every student has the same architecture, the same random initialisation, the same optimizer, the same number of
steps and the same batch size. The labelled set, the prompts and the evaluation problems are shared. What the
run measures, every ``eval_every`` steps, on held-out problems:

    accuracy      one sample per problem at temperature 1, checked by the verifier
    full          the share of samples that write the whole scratchpad (the behaviour the teacher shows)
    length        mean completion length in tokens (4 = answer only, K + 4 = full scratchpad)
    agree         top-1 agreement with the teacher, teacher-forced on held-out *teacher* samples
    kl            mean per-token KL(teacher ‖ student) on those teacher samples (forward, off-policy)
    rkl           mean per-token KL(student ‖ teacher) on the student's *own* samples (reverse, on-policy)
    accept        mean Σ_v min(p_teacher, q_student) per position of the teacher samples: the student's
                  acceptance rate α as a speculative draft for the teacher (PRIMER §7)
    accept_greedy mean p_teacher(argmax q_student): the acceptance when the draft proposes its argmax
                  (vLLM v0.30.0's default ``draft_sample_method="greedy"``)

Everything returned is what *this run* measured on *this machine*; ``distillab.tinylm.curves`` holds a recorded
copy for machines without torch (labelled illustrative). PRIMER §2–§5 explain each method; notebook 01 runs it.
"""
from __future__ import annotations

import copy
import platform
import random
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from ..losses import generalized_jsd, kd_loss
from .model import TinyLM, completion_logits, sample
from .task import PAD, VOCAB, SumTask, data_mix, parse, teacher_mix, verify

METHODS = ("hard", "kd", "seqkd", "gkd")
ALL_METHODS = METHODS + ("seqkd_all",)


@dataclass
class DistillConfig:
    k: int = 6                        # digits per problem
    base: int = 5                     # the answer is the sum mod base (chance = 1/base)
    # the teacher: wider, trained long on demonstrations that mostly show the working
    teacher_d: int = 64
    teacher_layers: int = 2
    teacher_heads: int = 4
    teacher_steps: int = 700
    teacher_batch: int = 128
    teacher_lr: float = 1e-3
    teacher_full: float = 0.7         # share of teacher demonstrations with the full scratchpad
    teacher_none: float = 0.1         # ... with none (the rest are partial)
    # the student: narrower, a fixed labelled set and a fixed budget, whatever the method
    student_d: int = 32
    student_layers: int = 2
    student_heads: int = 2
    student_steps: int = 1500
    student_batch: int = 64
    student_lr: float = 3e-3
    n_labelled: int = 1000            # labelled problems the student may use (prompts for seqkd and gkd too)
    data_full: float = 0.2            # share of labelled answers that show the working
    # the methods' knobs (PRIMER §2-§4)
    kd_temperature: float = 2.0       # T
    kd_alpha: float = 0.9             # weight on the soft term (1 = pure distillation)
    seqkd_filter: bool = True         # keep only teacher samples the verifier accepts
    gkd_lmbda: float = 0.5            # probability that a batch is the student's own samples
    gkd_beta: float = 0.5             # 0 = forward KL, 1 = reverse KL (TRL's convention)
    gkd_temperature: float = 1.0      # sampling temperature for the on-policy batches
    methods: tuple = METHODS
    eval_every: int = 100
    eval_prompts: int = 256
    seed: int = 0
    threads: int = 2                  # torch CPU threads (the machine may be shared)
    device: str = "auto"              # "auto" = cuda when available (T1), else cpu (T0)


_DEVICE = "cpu"


def _tensor(rows, dtype=torch.long):
    return torch.tensor(rows, dtype=dtype, device=_DEVICE)


def _setup(cfg: DistillConfig):
    global _DEVICE
    _DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    return random.Random(cfg.seed), torch.Generator(device=_DEVICE).manual_seed(cfg.seed)


def _batch(task: SumTask, pairs: list):
    """(prompts, completions padded with PAD, mask of real completion tokens)."""
    w = task.max_completion
    prompts = _tensor([p.prompt for p, _ in pairs])
    comps = _tensor([c + [PAD] * (w - len(c)) for _, c in pairs])
    mask = _tensor([[1.0] * len(c) + [0.0] * (w - len(c)) for _, c in pairs], torch.float32)
    return prompts, comps, mask


def new_model(cfg: DistillConfig, task: SumTask, role: str, seed: int | None = None) -> TinyLM:
    if seed is not None:
        torch.manual_seed(seed)
    if role == "teacher":
        m = TinyLM(VOCAB, cfg.teacher_d, cfg.teacher_layers, cfg.teacher_heads, task.seq_len)
    else:
        m = TinyLM(VOCAB, cfg.student_d, cfg.student_layers, cfg.student_heads, task.seq_len)
    return m.to(_DEVICE)


_TEACHERS: dict = {}     # teacher settings -> (state_dict, curve): the teacher is seeded, so this only saves time


def _teacher_key(cfg: DistillConfig) -> tuple:
    return (cfg.k, cfg.base, cfg.teacher_d, cfg.teacher_layers, cfg.teacher_heads, cfg.teacher_steps,
            cfg.teacher_batch, cfg.teacher_lr, cfg.teacher_full, cfg.teacher_none, cfg.seed, cfg.threads, cfg.device)


def train_teacher(cfg: DistillConfig, task: SumTask, rng: random.Random, log=print):
    """Cross-entropy on completion tokens of fresh demonstrations from the teacher mix, every step. A teacher with
    the same settings trained earlier in this process is reused (it would come out identical)."""
    key = _teacher_key(cfg)
    if key in _TEACHERS:
        state, curve, rng_state = _TEACHERS[key]
        model = new_model(cfg, task, "teacher", cfg.seed)
        model.load_state_dict(copy.deepcopy(state))
        rng.setstate(rng_state)
        log("  teacher: reusing the one trained earlier in this process")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, curve
    model = new_model(cfg, task, "teacher", cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.teacher_lr)
    mix = teacher_mix(task.k, cfg.teacher_full, cfg.teacher_none)
    curve = []
    model.train()
    for step in range(cfg.teacher_steps + 1):
        P, C, M = _batch(task, task.demos(cfg.teacher_batch, rng, mix))
        logits = completion_logits(model, P, C)
        loss = (F.cross_entropy(logits.reshape(-1, VOCAB), C.reshape(-1), reduction="none") * M.reshape(-1)).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.teacher_steps // 7) == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4)})
            log(f"  teacher step {step:4d}  loss {loss.item():.3f}")
    _TEACHERS[key] = (copy.deepcopy(model.state_dict()), curve, rng.getstate())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, curve


@torch.no_grad()
def teacher_samples(teacher: TinyLM, task: SumTask, problems: list, gen: torch.Generator,
                    temperature: float = 1.0, keep_verified: bool = True) -> tuple:
    """SeqKD's data: one teacher completion per problem; with ``keep_verified`` only the ones the verifier
    passes. Returns ``(pairs, stats)`` with the kept share and the full-scratchpad share before and after."""
    comps, _, mask = sample(teacher, _tensor([p.prompt for p in problems]), task.max_completion, temperature, gen)
    rows = [(p, c[: int(sum(m))]) for p, c, m in zip(problems, comps.tolist(), mask.tolist())]
    full = lambda rs: sum(parse(c)["scratch"] == task.k for _, c in rs) / max(1, len(rs))  # noqa: E731
    kept = [(p, c) for p, c in rows if verify(p, c)] if keep_verified else rows
    is_full = [parse(c)["scratch"] == task.k for _, c in rows]
    ok = [verify(p, c) for p, c in rows]
    acc = lambda sel: sum(o for o, f in zip(ok, is_full) if f == sel) / max(1, sum(f == sel for f in is_full))  # noqa: E731
    return kept, {"generated": len(rows), "kept": len(kept), "kept_share": round(len(kept) / max(1, len(rows)), 4),
                  "acc_full": round(acc(True), 4), "acc_other": round(acc(False), 4),
                  "full_before": round(full(rows), 4), "full_after": round(full(kept), 4),
                  "mean_len_before": round(sum(len(c) for _, c in rows) / max(1, len(rows)), 3),
                  "mean_len_after": round(sum(len(c) for _, c in kept) / max(1, len(kept)), 3)}


@torch.no_grad()
def evaluate(model: TinyLM, teacher: TinyLM, task: SumTask, problems: list, teacher_seqs: tuple,
             gen: torch.Generator) -> dict:
    """The metrics in the module docstring, for one model (the teacher itself gives agree 1, kl 0, accept 1)."""
    was = model.training
    model.eval()
    P = _tensor([p.prompt for p in problems])
    comps, _, own_mask = sample(model, P, task.max_completion, 1.0, gen)
    rows = [c[: int(sum(m))] for c, m in zip(comps.tolist(), own_mask.tolist())]
    parsed = [parse(c) for c in rows]
    acc = sum(verify(p, c) for p, c in zip(problems, rows)) / len(problems)
    full = sum(r["scratch"] == task.k for r in parsed) / len(problems)
    length = sum(r["length"] for r in parsed) / len(problems)
    # reverse KL on the model's own samples
    s_own = F.log_softmax(completion_logits(model, P, comps).float(), -1)
    t_own = F.log_softmax(completion_logits(teacher, P, comps).float(), -1)
    rkl = ((s_own.exp() * (s_own - t_own)).sum(-1) * own_mask).sum() / own_mask.sum()
    # teacher-forced on the teacher's own samples: agreement, forward KL, draft acceptance
    TP, TC, TM = teacher_seqs
    s = F.log_softmax(completion_logits(model, TP, TC).float(), -1)
    t = F.log_softmax(completion_logits(teacher, TP, TC).float(), -1)
    agree = ((s.argmax(-1) == t.argmax(-1)).float() * TM).sum() / TM.sum()
    kl = ((t.exp() * (t - s)).sum(-1) * TM).sum() / TM.sum()
    accept = (torch.minimum(s.exp(), t.exp()).sum(-1) * TM).sum() / TM.sum()
    greedy = (t.exp().gather(-1, s.argmax(-1, keepdim=True)).squeeze(-1) * TM).sum() / TM.sum()
    model.train(was)
    return {"accuracy": round(acc, 4), "full": round(full, 4), "length": round(length, 3),
            "agree": round(agree.item(), 4), "kl": round(kl.item(), 5), "rkl": round(rkl.item(), 5),
            "accept": round(accept.item(), 4), "accept_greedy": round(greedy.item(), 4)}


@torch.no_grad()
def exposure(model: TinyLM, teacher: TinyLM, task: SumTask, problems: list, teacher_seqs: tuple,
             gen: torch.Generator) -> list:
    """Exposure bias, measured: per completion position, the model's top-1 agreement with the teacher when the
    prefix is the *teacher's* sample (how it was trained, for SeqKD and KD) and when it is the model's *own*
    sample (how it is used). Rows: position, agreement on each, and how many sequences reach that position."""
    P = _tensor([p.prompt for p in problems])
    comps, _, own_mask = sample(model, P, task.max_completion, 1.0, gen)
    rows = []
    for (pp, cc, mm), name in ((teacher_seqs, "teacher"), ((P, comps, own_mask), "own")):
        s = completion_logits(model, pp, cc).argmax(-1)
        t = completion_logits(teacher, pp, cc).argmax(-1)
        hit = ((s == t).float() * mm).sum(0)
        rows.append((name, hit / mm.sum(0).clamp(min=1), mm.sum(0)))
    out = []
    for i in range(task.max_completion):
        out.append({"position": i, "on teacher prefixes": round(rows[0][1][i].item(), 4),
                    "on own prefixes": round(rows[1][1][i].item(), 4),
                    "n teacher": int(rows[0][2][i].item()), "n own": int(rows[1][2][i].item())})
    return out


def train_student(method: str, teacher: TinyLM, task: SumTask, cfg: DistillConfig, labelled: list,
                  seqkd_pairs: dict, eval_problems: list, teacher_seqs: tuple, log=print, loss_fn=None) -> tuple:
    """One student, one method, the shared budget. ``seqkd_pairs`` maps ``"seqkd"`` / ``"seqkd_all"`` to the
    teacher's (filtered / unfiltered) completions. ``loss_fn(student_logits, teacher_logits, labels, mask, cfg)``
    replaces the ``kd`` loss (notebook 01 trains with the learner's own). Returns ``(model, curve,
    on-policy batches)``."""
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}")
    rng = random.Random(cfg.seed + 101)
    gen = torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 101)
    egen = torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 202)
    model = new_model(cfg, task, "student", cfg.seed + 1)           # the same initial weights for every method
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.student_lr)
    pool = seqkd_pairs[method] if method.startswith("seqkd") else labelled
    curve, onpolicy_batches = [], 0
    for step in range(cfg.student_steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.student_steps:
            row = {"step": step, **evaluate(model, teacher, task, eval_problems, teacher_seqs, egen)}
            curve.append(row)
            log(f"  {method:5s} step {step:4d}  acc {row['accuracy']:.3f}  full {row['full']:.2f}  "
                f"agree {row['agree']:.3f}  kl {row['kl']:.3f}")
        if step == cfg.student_steps:
            break
        model.train()
        batch = [pool[rng.randrange(len(pool))] for _ in range(cfg.student_batch)]
        P, C, M = _batch(task, batch)
        if method == "gkd" and rng.random() <= cfg.gkd_lmbda:        # on-policy: the student's own samples
            C, _, M = sample(model, P, task.max_completion, cfg.gkd_temperature, gen)
            model.train()
            onpolicy_batches += 1
        logits = completion_logits(model, P, C)
        if method in ("hard", "seqkd", "seqkd_all"):
            loss = (F.cross_entropy(logits.reshape(-1, VOCAB), C.reshape(-1), reduction="none") * M.reshape(-1)).sum() / M.sum()
        else:
            with torch.no_grad():
                t_logits = completion_logits(teacher, P, C)
            if method == "kd":
                fn = loss_fn or (lambda s, t, y, m, c: kd_loss(s, t, y, m, c.kd_temperature, c.kd_alpha))
                loss = fn(logits, t_logits, C, M, cfg)
            else:
                loss = generalized_jsd(logits, t_logits, cfg.gkd_beta, M)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, curve, onpolicy_batches


def sft_student(pairs: list, cfg: DistillConfig | None = None, steps: int = 800, eval_n: int = 256, log=print) -> dict:
    """Train a fresh student by plain SFT on ``pairs`` (``(Problem, completion)``: a served teacher's answers
    converted with ``task.from_teacher_text``, for instance) and report accuracy, full-scratchpad share and length
    on held-out problems. No teacher model is involved, so there are no agreement numbers."""
    cfg = cfg or DistillConfig()
    _, gen = _setup(cfg)
    task = SumTask(cfg.k, cfg.base)
    if not pairs or any(len(p.digits) != task.k for p, _ in pairs):
        raise ValueError(f"pairs must be {task.k}-digit problems")
    rng = random.Random(cfg.seed + 101)
    model = new_model(cfg, task, "student", cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.student_lr)
    t0 = time.perf_counter()
    for step in range(steps):
        P, C, M = _batch(task, [pairs[rng.randrange(len(pairs))] for _ in range(cfg.student_batch)])
        logits = completion_logits(model, P, C)
        loss = (F.cross_entropy(logits.reshape(-1, VOCAB), C.reshape(-1), reduction="none") * M.reshape(-1)).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    problems = [task.sample(random.Random(cfg.seed + 9000 + i)) for i in range(eval_n)]
    comps, _, mask = sample(model.eval(), _tensor([p.prompt for p in problems]), task.max_completion, 1.0, gen)
    rows = [c[: int(sum(m))] for c, m in zip(comps.tolist(), mask.tolist())]
    return {"source": "measured", "pairs": len(pairs), "steps": steps,
            "accuracy": round(sum(verify(p, c) for p, c in zip(problems, rows)) / eval_n, 4),
            "full": round(sum(parse(c)["scratch"] == task.k for c in rows) / eval_n, 4),
            "length": round(sum(parse(c)["length"] for c in rows) / eval_n, 3),
            "seconds": round(time.perf_counter() - t0, 1)}


def length_cap_experiment(cfg: DistillConfig | None = None, caps: tuple = (None, 9), n_prompts: int = 1000,
                          steps: int = 1500, log=print) -> dict:
    """Budget-aware trace distillation on the tiny task (PRIMER §5): the teacher answers ``n_prompts`` fresh
    problems once each; for every cap, a fresh student is trained by SFT on the *verified* answers of at most
    ``cap`` tokens (None = no cap; K + 4 is the full scratchpad, so a cap of K + 3 keeps none of it). Returns the
    teacher's sample statistics and one row per cap: what was kept, and the student's accuracy and length."""
    cfg = cfg or DistillConfig()
    rng, _ = _setup(cfg)
    task = SumTask(cfg.k, cfg.base)
    teacher, _ = train_teacher(cfg, task, rng, log)
    problems = [task.sample(random.Random(cfg.seed + 20_000 + i)) for i in range(n_prompts)]
    pairs, stats = teacher_samples(teacher, task, problems, torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 4),
                                   1.0, keep_verified=True)
    rows = []
    for cap in caps:
        kept = [(p, c) for p, c in pairs if cap is None or len(c) <= cap]
        r = sft_student(kept, cfg, steps=steps, log=log) if kept else {"accuracy": 0.0, "full": 0.0, "length": 0.0}
        rows.append({"cap": "none" if cap is None else cap, "kept": len(kept),
                     "kept full share": round(sum(parse(c)["scratch"] == task.k for _, c in kept) / max(1, len(kept)), 4),
                     "student accuracy": r["accuracy"], "student full": r["full"], "student length": r["length"]})
    return {"source": "measured", "teacher": stats, "rows": rows}


def run(cfg: DistillConfig | None = None, log=print, keep_models: bool = False) -> dict:
    """Teacher → SeqKD data → the four students → a summary. Returns everything measured (JSON-serialisable;
    with ``keep_models`` also the torch models under ``"models"``)."""
    cfg = cfg or DistillConfig()
    rng, gen = _setup(cfg)
    task = SumTask(cfg.k, cfg.base)
    t0 = time.perf_counter()
    teacher, teacher_curve = train_teacher(cfg, task, rng, log)
    t1 = time.perf_counter()
    labelled = task.demos(cfg.n_labelled, random.Random(cfg.seed + 7), data_mix(task.k, cfg.data_full))
    eval_problems = [task.sample(random.Random(cfg.seed + 9000 + i)) for i in range(cfg.eval_prompts)]
    held = [task.sample(random.Random(cfg.seed + 5000 + i)) for i in range(cfg.eval_prompts)]
    tgen = torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 3)
    held_pairs, _ = teacher_samples(teacher, task, held, tgen, 1.0, keep_verified=False)
    teacher_seqs = _batch(task, held_pairs)
    all_pairs, all_stats = teacher_samples(teacher, task, [p for p, _ in labelled], tgen, 1.0, keep_verified=False)
    kept = [(p, c) for p, c in all_pairs if verify(p, c)] if cfg.seqkd_filter else all_pairs
    full = lambda rs: round(sum(parse(c)["scratch"] == task.k for _, c in rs) / max(1, len(rs)), 4)  # noqa: E731
    seqkd_stats = {**all_stats, "kept": len(kept), "kept_share": round(len(kept) / len(all_pairs), 4),
                   "full_after": full(kept), "mean_len_after": round(sum(len(c) for _, c in kept) / max(1, len(kept)), 3)}
    seqkd_pairs = {"seqkd": kept, "seqkd_all": all_pairs}
    teacher_eval = evaluate(teacher, teacher, task, eval_problems, teacher_seqs,
                            torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 202))
    log(f"teacher: {teacher.num_params():,} parameters, accuracy {teacher_eval['accuracy']:.3f}, "
        f"full scratchpad {teacher_eval['full']:.2f}")
    students, models, timing = {}, {"teacher": teacher}, {"teacher": round(t1 - t0, 1)}
    for method in cfg.methods:
        ts = time.perf_counter()
        model, curve, onp = train_student(method, teacher, task, cfg, labelled, seqkd_pairs, eval_problems,
                                          teacher_seqs, log)
        timing[method] = round(time.perf_counter() - ts, 1)
        xgen = torch.Generator(device=_DEVICE).manual_seed(cfg.seed + 303)
        students[method] = {"curve": curve, "final": curve[-1], "onpolicy_batches": onp,
                            "exposure": exposure(model, teacher, task, eval_problems, teacher_seqs, xgen)}
        models[method] = model
    labelled_full = sum(parse(c)["scratch"] == task.k for _, c in labelled) / len(labelled)
    out = {"source": "measured", "config": asdict(cfg),
           "params": {"teacher": teacher.num_params(), "student": models[cfg.methods[0]].num_params()},
           "teacher": {"curve": teacher_curve, "eval": teacher_eval},
           "data": {"labelled": len(labelled), "labelled_full": round(labelled_full, 4), "seqkd": seqkd_stats},
           "students": students, "timing_s": timing,
           "machine": (f"{torch.cuda.get_device_name(0)} / torch {torch.__version__}" if _DEVICE == "cuda" else
                       f"{platform.machine()} CPU / torch {torch.__version__} / {cfg.threads} threads")}
    if keep_models:
        out["models"] = models
        out["task"] = task
        out["teacher_seqs"] = teacher_seqs
        out["eval_problems"] = eval_problems
    return out
