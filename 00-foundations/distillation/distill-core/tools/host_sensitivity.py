#!/usr/bin/env python3
"""How far this lab's trained and sampled numbers move from one CPU to another — the evidence behind tests/pins.py.

A seeded run of a tiny numpy model is one CPU's run: numpy's bundled OpenBLAS picks a GEMM kernel per
microarchitecture, its exp, tanh and reductions dispatch per SIMD level, the kernels round differently in the last
bit, and soft-target training, the drafts and on-policy GKD grow that into a slightly different model (hard-label
training and the reasoning toy's table policy do not). This tool recomputes every trained or sampled quantity the
tests pin, exactly as the tests do, under other kernels and levels, and prints how far each one moved:

    python3 tools/host_sensitivity.py run OUT_DIR              # ~3 minutes on 4 cores: 5 kernels × 4 SIMD levels
                                                               # + 16 last-bit-perturbation runs, then the table
    python3 tools/host_sensitivity.py collect out.json [--ulp SEED]   # one run, in this process (what `run` spawns)
    python3 tools/host_sensitivity.py summarize OUT_DIR         # the table again: reference, min, max, max |dev|

The kernels come from OPENBLAS_CORETYPE (Prescott → the Katmai kernel, Nehalem, Sandybridge, Haswell — what AMD
Zen runs too — and the host's own, SkylakeX on the reference machine) and the levels from NPY_DISABLE_CPU_FEATURES
(everything on; X86_V4 only; X86_V3 only, an AMD EPYC runner; the X86_V2 baseline). A kernel the host's OpenBLAS
lacks is reported and skipped, so run this on an x86-64 machine with AVX-512 to cover all five. `--ulp SEED` flips
the last bit of every np.exp / np.tanh result at random: a stand-in for kernels not on this machine (an ARM
laptop, a future runner). A tolerance in tests/pins.py is at least twice the largest deviation in the table.

Measured 2026-09-27 on the reference machine (numpy 2.4.6, Python 3.11), 36 runs, largest |deviation| from the
reference run — the numbers PRIMER.md quotes:
    §1 soft-target student, 242 contexts: rule accuracy 0.876 ± 0.025, KL 0.617 within ×1.7 (0.50–1.02);
       605 contexts: 0.992 ± 0.025, KL 0.108 within ×2.8; hard-label students and the 8/16-unit fits: 0
    §3 T = 1 pipeline: verified 16 (14–17), unique 11 (10–13); KD student on its own prefixes 0.788 ± 0.013 at
       position 4, 0.774 ± 0.027 at 12; whole output right through 4: 0.650 ± 0.015, through 12: 0.190 ± 0.016;
       the SFT student: 0
    §4 GKD from the KD student, own prefixes at 4 / 12 / rule accuracy: β = 0 ± 0.007 / 0.036 / 0.050,
       β = 0.5 ± 0.024 / 0.069 / 0.066, β = 1 ± 0.027 / 0.080 / 0.074; the starts table's β = 1 column up to
       ± 0.14 (fresh), mass on wrong/right tokens up to ± 0.23; the 5-token identities (31%, 1.86 / 3.15): 0
    §5 the reasoning toy: 0 everywhere
    §6 prune vs fresh, mean of five draws ± 0.018, a single draw (a min or max cell) ± 0.050
    §7 drafts: off-the-shelf α 0.890 ± 0.003, SeqKD 0.933 ± 0.039, logit KD 16 units 0.988 ± 0.024, 32 units
       0.995 ± 0.006, 4 units 0.589 ± 0.054, 8 units 0.722 ± 0.139; speedups follow (16 units: 2.26 ± 0.11,
       8 units: 1.72 ± 0.39); small KLs within ×2.7
    §8 KD student: KL 5.557 ± 0.087, top-1 0.198 ± 0.017, top-3 0.342 ± 0.028; on-policy student: KL 0.185
       within ×2.4, top-1 0.942 ± 0.050, top-3 0.923 ± 0.163, right on the rare slice 89 ± 11 (n = 2) and
       69 ± 19 (n = 8) of 101; 8-unit student ± 0.003; the weak-teacher comparison: 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

KERNELS = {"host": None, "Katmai": "Prescott", "Nehalem": "Nehalem", "Sandybridge": "Sandybridge", "Haswell": "Haswell"}
LEVELS = {"full": None, "v4": "AVX512_ICL AVX512_SPR", "v3": "X86_V4 AVX512_ICL AVX512_SPR",
          "v2": "X86_V3 X86_V4 AVX512_ICL AVX512_SPR"}
ULP_RUNS = 16


def install_ulp_noise(seed: int) -> None:
    """Flip the last bit of every float64 np.exp / np.tanh result at random (before anything is computed)."""
    rng = np.random.default_rng(seed)

    def wrap(f):
        def g(x, *a, **k):
            y = np.asarray(f(x, *a, **k))
            if y.dtype == np.float64 and y.size:
                flip = rng.integers(0, 3, y.shape)
                y = np.where(flip == 0, np.nextafter(y, -np.inf), np.where(flip == 2, np.nextafter(y, np.inf), y))
            return y
        return g
    np.exp, np.tanh = wrap(np.exp), wrap(np.tanh)


def collect() -> dict:
    """Every trained or sampled quantity the tests pin, computed as the tests compute it (same seeds, same order)."""
    from distillcore import (ModLang, ThinkToy, TinyLM, divergences as D, draft as Dr, eval as E, losses as L,
                             onpolicy as op, reasoning as R, seqkd, train)
    from distillcore.tinylm import fit_language
    out: dict = {}
    lang = ModLang(11, 0.2)
    teacher = fit_language(lang, 64)
    o = lang.orbits()
    prompts = np.array(o[0] + o[1])
    greedy = seqkd.teacher_data(teacher, prompts, 1, 12, np.random.default_rng(1), T=0.0)
    kd = TinyLM(11, 16, 8, seed=1)
    op.gkd_train(kd, teacher, prompts, 12, 300, lam=0.0, beta=0.0, data=greedy)
    C = lang.contexts()
    rng5 = lambda: np.random.default_rng(5)

    t = E.vs_truth(teacher, lang)
    out["teacher.kl"], out["teacher.rule_acc"] = t["kl"], t["rule_acc"]
    # -- §1 soft against hard, capacity
    for N in (242, 605):
        rng = np.random.default_rng(1)
        ctx = C[rng.integers(0, 121, N)]
        y, zt = lang.sample_next(ctx, rng), teacher.logits(ctx)
        hard, soft = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
        train(hard, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
        train(soft, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 400)
        for name, m in (("hard", hard), ("soft", soft)):
            v = E.vs_truth(m, lang)
            out[f"s1.{name}{N}.rule_acc"], out[f"s1.{name}{N}.kl"] = v["rule_acc"], v["kl"]
    for H in (4, 8, 16):
        v = E.vs_truth(fit_language(lang, H), lang)
        out[f"s1.fit{H}.rule_acc"], out[f"s1.fit{H}.kl"] = v["rule_acc"], v["kl"]
    # -- §3 pipeline, entropy, exposure bias
    for T in (0.0, 1.0):
        _, st = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=T)
        for k, v in st.items():
            out[f"s3.pipe.T{T:.0f}.{k}"] = v
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, greedy, 400)
    ctx_g, _ = teacher.positions(greedy)
    ent = lambda m: float(-(m.probs(ctx_g) * np.log(m.probs(ctx_g))).sum(1).mean())
    out["s3.ent.teacher"], out["s3.ent.sft"], out["s3.ent.kd"] = ent(teacher), ent(sft), ent(kd)
    for name, m in (("teacher", teacher), ("sft", sft), ("kd", kd)):
        eb = seqkd.exposure_bias(m, lang, greedy, prompts, rng5())
        own, allr, rule = eb["own_prefixes"], eb["all_right_so_far"], eb["sampled_on_rule"]
        out[f"s3.eb.{name}.tf_mean"] = float(eb["teacher_prefixes"].mean())
        for i in (0, 3, 11):
            out[f"s3.eb.{name}.own{i}"], out[f"s3.eb.{name}.all{i}"], out[f"s3.eb.{name}.rule{i}"] = float(own[i]), float(allr[i]), float(rule[i])
        out[f"s3.eb.{name}.rule_min"], out[f"s3.eb.{name}.rule_max"] = float(rule.min()), float(rule.max())
        out[f"s3.eb.{name}.own_ptp_3on"], out[f"s3.eb.{name}.all_diff_max"] = float(np.ptp(own[3:])), float(np.diff(allr).max())
        out[f"s3.eb.{name}.gap_last"], out[f"s3.eb.{name}.rule_3on_max"] = float(eb["gap_last"]), float(rule[3:].max())
    out["s3.sft.own_last"] = float(seqkd.own_accuracy(sft, lang, prompts, 12, rng5())[-1])
    # -- §4 GKD rows and the starts table
    v = E.vs_truth(kd, lang)
    out["s4.kd.rule_acc"], out["s4.kd.wrong_right_q"], out["s4.kd.wrong_top_q"] = v["rule_acc"], v["wrong_right_q"], v["wrong_top_q"]
    own = seqkd.own_accuracy(kd, lang, prompts, 12, rng5())
    out["s4.kd.own3"], out["s4.kd.own11"] = float(own[3]), float(own[11])
    onp = None
    for beta in (0.0, 0.5, 1.0):
        s = kd.copy()
        op.gkd_train(s, teacher, prompts, 12, 300, lam=1.0, beta=beta, data=greedy, seed=1)
        own = seqkd.own_accuracy(s, lang, prompts, 12, rng5())
        v = E.vs_truth(s, lang)
        out[f"s4.gkd.b{beta}.own3"], out[f"s4.gkd.b{beta}.own11"], out[f"s4.gkd.b{beta}.rule_acc"] = float(own[3]), float(own[11]), v["rule_acc"]
        if beta == 0.0:
            onp = s
    for label, start in (("kd", kd), ("sft", sft), ("fresh", TinyLM(11, 16, 8, seed=1))):
        v = E.vs_truth(start, lang)
        out[f"s4.start.{label}.rule_acc"], out[f"s4.start.{label}.wrong_right_q"] = v["rule_acc"], v["wrong_right_q"]
        for beta in (0.0, 1.0):
            s = start.copy()
            op.gkd_train(s, teacher, prompts, 12, 300, lam=1.0, beta=beta, data=greedy, seed=1)
            a = E.vs_truth(s, lang)
            out[f"s4.start.{label}.b{beta}.rule_acc"] = a["rule_acc"]
            out[f"s4.start.{label}.b{beta}.wrong_top_q"], out[f"s4.start.{label}.b{beta}.wrong_right_q"] = a["wrong_top_q"], a["wrong_right_q"]
    # -- §4 the identities on the 5-token world
    t5, s5, prompt = fit_language(ModLang(5, 0.2), 16, steps=500), TinyLM(5, 4, 3, seed=1), np.array([1, 2])
    flat = lambda d: np.concatenate([d[k].ravel() for k in sorted(d)])
    ex_seq, ex_tok = flat(op.exact_pg_grad(s5, t5, prompt, 3)), flat(op.exact_pg_grad(s5, t5, prompt, 3, per_token=True))
    out["s4.id.bias"] = float(np.linalg.norm(ex_tok - ex_seq) / np.linalg.norm(ex_seq))
    rng, A, B = np.random.default_rng(0), [], []
    for _ in range(200):
        s = s5.sample(np.tile(prompt, (64, 1)), 3, rng)
        A.append(flat(op.pg_grad(s5, t5, s)))
        B.append(flat(op.pg_grad(s5, t5, s, per_token=True)))
    A, B = np.array(A), np.array(B)
    out["s4.id.var_tok"], out["s4.id.var_seq"] = float(B.var(0).sum()), float(A.var(0).sum())
    out["s4.id.mc_err"] = float(np.linalg.norm(A.mean(0) - ex_seq) / np.linalg.norm(ex_seq))
    # -- §5 the reasoning toy
    task = ThinkToy(0.8, 0.1, 32)
    tt = R.reinforce(task, np.random.default_rng(0), 16000)
    for k, v in tt.expected(task).items():
        out[f"s5.teacher.{k}"] = v
    L_, ok = R.traces(tt, task, np.random.default_rng(1), 1000)
    student = R.distil(L_, ok)
    for k, v in student.expected(task).items():
        out[f"s5.student.{k}"] = v
    for n in (10, 30, 100, 300, 1000):
        Ln, okn = R.traces(tt, task, np.random.default_rng(2), n)
        out[f"s5.sweep{n}.accuracy"] = R.distil(Ln, okn).expected(task)["accuracy"]
    for label, keep, ml in (("all", "all", None), ("correct", "correct", None), ("le16", "correct", 16), ("le12", "correct", 12)):
        out[f"s5.filt.{label}.kept"] = int(((ok if keep == "correct" else np.ones_like(ok)) & (L_ <= (ml or 99)) & (L_ < 32)).sum())
        for k, v in R.distil(L_, ok, keep=keep, max_len=ml).expected(task).items():
            out[f"s5.filt.{label}.{k}"] = v
    for n in (1000, 4000, 16000):
        out[f"s5.rl{n}.accuracy"] = R.reinforce(task, np.random.default_rng(3), n).expected(task)["accuracy"]
    weak = student.expected(task, q=0.05)
    out["s5.weak.accuracy"], out["s5.weak.length"] = weak["accuracy"], weak["length"]
    # -- §6 prune against fresh
    for steps in (0, 20, 100):
        pruned, fresh = [], []
        for d in range(1, 6):
            rng = np.random.default_rng(d)
            ctx = C[rng.integers(0, 121, 242)]
            lang.sample_next(ctx, rng)
            zt = teacher.logits(ctx)
            for m, dest in [(teacher.prune_width(ctx, 16), pruned)] + [(TinyLM(11, 16, 8, seed=s_), fresh) for s_ in range(1, 5)]:
                if steps:
                    train(m, ctx, lambda z, i: L.kd(z, zt[i], 1.0), steps)
                dest.append(E.vs_truth(m, lang)["rule_acc"])
        for name, a in (("pruned", np.array(pruned)), ("fresh", np.array(fresh))):
            out[f"s6.{name}{steps}.mean"], out[f"s6.{name}{steps}.min"], out[f"s6.{name}{steps}.max"] = float(a.mean()), float(a.min()), float(a.max())
            out[f"s6.{name}{steps}.std"] = float(a.std())
    # -- §7 drafts
    base, dialect = ModLang(11, 0.2), ModLang(11, 0.2, skew=1.0)
    target = fit_language(dialect, 64)
    text = target.sample(C[np.random.default_rng(1).integers(0, 121, 500)], 12, np.random.default_rng(2))
    corpus = target.sample(C[np.random.default_rng(3).integers(0, 121, 2000)], 12, np.random.default_rng(4))
    PT = target.probs(C)
    drafts = {"off": fit_language(base, 16, seed=3)}
    for H in (4, 8, 16, 32):
        d_ = TinyLM(11, H, 8, seed=3)
        train(d_, C, lambda z, i: L.soft_ce(z, PT[i]), 1500)
        drafts[f"kd{H}"] = d_
    sk = TinyLM(11, 16, 8, seed=3)
    seqkd.sft(sk, corpus, 1500, batch=512)
    drafts["seqkd"] = sk
    c16 = Dr.draft_cost(drafts["kd16"].n_params, target.n_params)
    for name, m in drafts.items():
        r = Dr.acceptance_on_text(target, m, text)
        for k in ("alpha", "greedy", "kl", "tv"):
            out[f"s7.{name}.{k}"] = r[k]
        out[f"s7.{name}.speedup_c16"] = Dr.speedup(r["alpha"], 4, c16)
        out[f"s7.{name}.speedup_own_c"] = Dr.speedup(r["alpha"], 4, Dr.draft_cost(m.n_params, target.n_params))
    # -- §8 measuring a student
    small = fit_language(lang, 8)
    zt = teacher.logits(C)
    common = set(map(tuple, prompts))
    rare = np.array([c for c in C if tuple(c) not in common])
    okf = lambda m, x, n: lang.verify(m.sample(x, n, None, 0.0))
    for name, s in (("kd", kd), ("onp", onp), ("small", small)):
        zs = s.logits(C)
        out[f"s8.{name}.kl"], out[f"s8.{name}.agree"], out[f"s8.{name}.top3"] = E.kl(zt, zs), E.argmax_agreement(zt, zs), E.topk_overlap(zt, zs, 3)
        for label, x, n in (("common2", prompts, 2), ("rare2", rare, 2), ("rare8", rare, 8)):
            out[f"s8.{name}.{label}.k"] = int(okf(s, x, n).sum())
        out[f"s8.{name}.common8.all"] = bool(okf(s, prompts, 8).all())
    ctx = C[np.random.default_rng(7).integers(0, 121, 605)]
    y = lang.sample_next(ctx, np.random.default_rng(8))
    weak = TinyLM(11, 64, 8, seed=2)
    train(weak, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
    data = seqkd.teacher_data(weak, C, 20, 1, np.random.default_rng(9))
    filt, raw = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
    seqkd.sft(filt, seqkd.keep_verified(lang, data), 600)
    seqkd.sft(raw, data, 600)
    zw = weak.logits(C)
    out["s8.beat.weak.acc"] = E.vs_truth(weak, lang)["rule_acc"]
    for name, m in (("filt", filt), ("raw", raw)):
        out[f"s8.beat.{name}.acc"] = E.vs_truth(m, lang)["rule_acc"]
        out[f"s8.beat.{name}.kl"], out[f"s8.beat.{name}.agree"] = E.kl(zw, m.logits(C)), E.argmax_agreement(zw, m.logits(C))
    # -- the bimodal fits are a grid search: exact unless a tie moves
    p = D.bimodal()
    for name, kw in (("forward", dict(divergence="forward")), ("jsd", dict(divergence="jsd", beta=0.5)), ("reverse", dict(divergence="reverse"))):
        f = D.fit_bump(p, **kw)
        for k in ("mu", "s", "value", "mass_where_teacher_is_empty", "teacher_mass_uncovered"):
            out[f"div.{name}.{k}"] = f[k]
    out["div.jsd06.mu"], out["div.jsd07.mu"] = D.fit_bump(p, "jsd", 0.6)["mu"], D.fit_bump(p, "jsd", 0.7)["mu"]
    return {k: (v if isinstance(v, (bool, int, str)) else float(v)) for k, v in out.items()}


def summarize(out_dir: Path) -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    if not runs:
        sys.exit(f"no runs in {out_dir}")
    ref = runs.get("host_full") or next(iter(runs.values()))
    print(f"{len(runs)} runs: {', '.join(sorted(runs))}\n")
    print(f"{'quantity':40s} {'ref':>10s} {'min':>10s} {'max':>10s} {'max|dev|':>9s} {'distinct':>8s}  furthest run")
    for k, r in ref.items():
        vals = {n: run[k] for n, run in runs.items() if k in run}
        if isinstance(r, (bool, str)):
            distinct = sorted(set(str(v) for v in vals.values()))
            print(f"{k:40s} {str(r):>10s} {'':>10s} {'':>10s} {'':>9s} {len(distinct):8d}  {'' if len(distinct) == 1 else distinct}")
            continue
        arr = np.array(list(vals.values()), float)
        dev = float(np.abs(arr - r).max())
        worst = max(vals, key=lambda n: abs(vals[n] - r)) if dev > 0 else ""
        print(f"{k:40s} {r:10.4f} {arr.min():10.4f} {arr.max():10.4f} {dev:9.4f} {len(set(np.round(arr, 6))):8d}  {worst}")


def run_all(out_dir: Path, jobs: int) -> None:
    """collect() in a subprocess per kernel × level and per --ulp seed, `jobs` at a time, then the table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [(f"{k}_{l}", {**({"OPENBLAS_CORETYPE": core} if core else {}), **({"NPY_DISABLE_CPU_FEATURES": off} if off else {})}, [])
            for k, core in KERNELS.items() for l, off in LEVELS.items()]
    plan += [(f"ulp{s}", {}, ["--ulp", str(s)]) for s in range(1, ULP_RUNS + 1)]

    def one(job):
        name, env, extra = job
        full = {**os.environ, **env, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, __file__, "collect", str(out_dir / f"{name}.json"), *extra],
                              env=full, capture_output=True, text=True)
        note = " ".join(l for l in proc.stderr.splitlines() if "Core not found" in l or "not supported" in l)
        return f"{name:20s} {'ok' if proc.returncode == 0 else 'FAILED: ' + proc.stderr[-300:]} {note}"

    with ThreadPoolExecutor(jobs) as pool:
        for line in pool.map(one, plan):
            print(line, flush=True)
    print()
    summarize(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "collect", "summarize"])
    ap.add_argument("target", help="a directory for run/summarize, a JSON file for collect")
    ap.add_argument("--ulp", type=int, default=None, help="collect: flip the last bit of exp/tanh results at random, with this seed")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="run: parallel subprocesses")
    args = ap.parse_args()
    if args.cmd == "collect":
        if args.ulp is not None:
            install_ulp_noise(args.ulp)
        Path(args.target).write_text(json.dumps(collect(), indent=1))
    elif args.cmd == "run":
        run_all(Path(args.target), args.jobs)
    else:
        summarize(Path(args.target))
