"""Every computed number the topic's PRIMER.md quotes is recomputed here.

Exact numbers must appear verbatim (`present`): closed forms on fixed logits, enumerations, parameter counts, the
roofline's bytes, FLOPs, costs and break-even, Wilson intervals of fixed counts. Numbers that come from training or
sampling a toy model — downstream of `fit_language`, `train`, `gkd_train`, `seqkd.pipeline`, `exposure_bias`,
pruning, the drafts' acceptance, the REINFORCE teacher — are one seeded run on one CPU. Another CPU family's
OpenBLAS kernels and numpy SIMD loops round the tiny models' matmuls and exponentials differently in the last bit,
training amplifies that, and the third digit moves (an AVX2-only runner gives 0.889 where the primer says 0.890).
Those are checked with `near`: the text around them verbatim, each number within a tolerance of the one stated.
The tolerances were measured: OpenBLAS's SkylakeX, Haswell (= Zen), Sandybridge, Nehalem and Prescott kernels under
numpy's AVX-512, AVX2 and SSE4.2 loops (15 variants; one or four threads changes nothing). A number that came out
bit-identical on all 15 gets one unit of its last printed digit; any other, about three times its largest
deviation from the primer, rounded up. Where the loosest of those windows could hide a claim the primer draws from
the numbers (soft beats hard, the speedup peaks at 16 units), the claim is asserted outright.

If a formula, a seed or an experiment changes, the primer fails this test until it is updated. Cited facts (TRL
and vLLM defaults, the R1, Qwen3, Minitron and EAGLE results, prices, the T4 memory predictions) are quoted and
marked (verify) in the primer; they are not checked here.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from distillcore import (ModLang, ThinkToy, TinyLM, cost as K, divergences as D, draft as Dr, eval as E,
                         losses as L, onpolicy as op, reasoning as R, seqkd, train)
from distillcore.tinylm import fit_language


def _norm(text: str) -> str:
    return " ".join(text.split())


PRIMER = _norm((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8"))


def present(*fragments):
    missing = [f for f in fragments if _norm(f) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


NUMBER = r"([−-]?\d[\d,]*(?:\.\d+)?)"


def stated(template: str) -> list[list[float]]:
    """Each place PRIMER.md says `template` — its text verbatim (whitespace-normalised, as in present()), every '#'
    one number — and the numbers it states there, one list per place."""
    pattern = NUMBER.join(re.escape(part) for part in _norm(template).split("#"))
    found = [[float(g.replace(",", "").replace("−", "-")) for g in m.groups()] for m in re.finditer(pattern, PRIMER)]
    assert found, f"PRIMER.md no longer says: {template}"
    return found


def near(template: str, *computed, tol):
    """Trained or sampled numbers: the primer's text around them verbatim, and each computed value within `tol` (one
    for every '#', or one per '#', in order) of the number stated, wherever the primer says it."""
    tols = tol if isinstance(tol, tuple) else (tol,) * len(computed)
    assert template.count("#") == len(computed) == len(tols), template
    for want in stated(template):
        off = [(w, round(float(c), 4), t) for w, c, t in zip(want, computed, tols) if not abs(float(c) - w) <= t]
        assert not off, f"PRIMER.md says {template!r} with {want}; outside the tolerance (stated, computed, tol): {off}"


def neg(s: str) -> str:
    return s.replace("-", "−")


Z = np.array([[4.0, 3.0, 1.0, 0.0, -1.0]])
V = np.array([[3.0, 3.5, 0.0, 0.5, -1.0]])
H100 = K.GPUS["h100"]


def test_s1_the_roofline_and_capacity_view():
    rows = {}
    for key, label in (("qwen2.5-32b", "32B (teacher)"), ("qwen2.5-1.5b", "1.5B (student)")):
        m = K.SHAPES[key]
        w, kv = K.weight_gb(m.params()), m.kv_bytes_per_token()
        rows[key] = (m, w, kv)
        present(f"| {label} | {m.params() / 1e9:.2f} B | {w:.2f} GB | {kv:,.0f} B | {2 * m.params() / 1e9:.2f} GFLOP | "
                f"{K.decode_step(m, H100, 1, 1024) * 1e3:.3f} ms | {K.sessions_per_gpu(80, w, kv / 1e3, 2048):.2f} |")
    t, s = rows["qwen2.5-32b"][0], rows["qwen2.5-1.5b"][0]
    present(f"The weights are {t.params() / s.params():.1f}× smaller and the KV per token "
            f"{t.kv_bytes_per_token() / s.kv_bytes_per_token():.1f}× smaller")
    w, kv = rows["qwen2.5-32b"][1], rows["qwen2.5-32b"][2] / 1e3
    tp2, one = K.sessions_per_gpu(160, w, kv, 2048), K.sessions_per_gpu(80, rows["qwen2.5-1.5b"][1], rows["qwen2.5-1.5b"][2] / 1e3, 2048)
    present(f"leave for KV ({80 * 0.9 - w:.2f} GB of the usable {80 * 0.9:.0f})",
            f"the 32B holds {tp2:.1f} such sessions, {tp2 / 2:.1f} per GPU, and the per-GPU ratio is {one / (tp2 / 2):.0f}×")


def test_s1_soft_targets_beat_hard_labels_and_capacity(lang, teacher):
    C = lang.contexts()
    for N in (242, 605):
        rng = np.random.default_rng(1)
        ctx = C[rng.integers(0, 121, N)]
        y, zt = lang.sample_next(ctx, rng), teacher.logits(ctx)
        hard, soft = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
        train(hard, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
        train(soft, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 400)
        h, s = E.vs_truth(hard, lang), E.vs_truth(soft, lang)
        # the KD student's KL is dominated by the contexts its 242 draws never covered: 0.57–1.02 across CPUs
        near(f"| {N} ({round(N / 121)}) | # / # nats | # / # nats |", h["rule_acc"], h["kl"], s["rule_acc"], s["kl"],
             tol=(0.001, 0.001, 0.08, 1.5 if N == 242 else 0.4))
        assert s["rule_acc"] > h["rule_acc"] and s["kl"] < h["kl"]  # soft targets beat hard labels
        if N == 242:
            near("two examples per context gave # rule accuracy with soft targets and # with labels", s["rule_acc"],
                 h["rule_acc"], tol=(0.08, 0.001))
    r8, r16 = E.vs_truth(fit_language(lang, 8), lang), E.vs_truth(fit_language(lang, 16), lang)
    near("tops out at # rule accuracy", r8["rule_acc"], tol=0.001)
    near("where 16 units reach #", r16["rule_acc"], tol=0.001)
    near("an error rate of #% per context", (1 - r8["rule_acc"]) * 100, tol=0.1)


def test_s2_soft_targets_temperature_and_the_limit():
    for label, z, T in (("teacher, T = 1", Z, 1), ("teacher, T = 2", Z, 2), ("student, T = 1", V, 1)):
        present(f"| {label} | " + " | ".join(f"{x:.4f}" for x in L.softmax(z, T)[0]) + " |")
    present(f"KL(p ‖ q) = {L.kl(L.softmax(Z), L.softmax(V))[0]:.5f} nats at T = 1")
    g = L.kd(V, Z, 2.0, scale=False)[1][0]
    present(neg("(" + ", ".join(f"{x:.6f}" for x in g) + ")"))
    Ts = (1, 2, 4, 10, 100, 1000)
    present("| KL(p_T ‖ q_T) | " + " | ".join(f"{L.kd(V, Z, T, scale=False)[0]:.5f}" for T in Ts) + " |",
            "| T²·KL | " + " | ".join(f"{L.kd(V, Z, T)[0]:.5f}" for T in Ts) + " |")
    present(f"{L.logit_mse(V, Z)[0]:.5f} for the example", f"{L.label_noise(np.array([0.8, 0.1, 0.1]))[0]:.2f} per example")
    p = L.softmax(Z)
    vals = {b: L.gkd(V, p, b)[0] for b in (0.0, 0.1, 0.5, 0.9, 1.0)}
    present(f"{vals[0.0]:.6f} at β = 0, {vals[0.1]:.6f} at 0.1, {vals[0.5]:.6f} at 0.5, {vals[0.9]:.6f} at 0.9, "
            f"{vals[1.0]:.6f} at β = 1")


def test_s2_forward_against_reverse_kl():
    p = D.bimodal()
    f, j, r = D.fit_bump(p, "forward"), D.fit_bump(p, "jsd", 0.5), D.fit_bump(p, "reverse")
    for label, x, val in (("forward KL", f, f"{f['value']:.4f}"), ("JSD, β = 0.5", j, f"{j['value']:.4f}"),
                          ("reverse KL", r, f"{r['value']:.4f} = ln 2")):
        present(f"| {label} | {x['mu']:.1f}, {x['s']:.1f} | {val} | {x['mass_where_teacher_is_empty']:.3f} | "
                f"{x['teacher_mass_uncovered']:.2f} |")
    present(f"puts {f['mass_where_teacher_is_empty']:.1%} of its samples")
    assert D.fit_bump(p, "jsd", 0.6)["mu"] == 5.0 and D.fit_bump(p, "jsd", 0.7)["mu"] == 2.0
    present("flips from covering to seeking between β = 0.6 and β = 0.7",
            f"{round(f['mass_where_teacher_is_empty'] * 100)}% of the toy student's samples")


def test_s3_pipeline_bookkeeping_and_exposure_bias(lang, teacher, bench):
    # greedy decoding of a teacher right at every context (test_tinylm) is exact: 8 identical samples per prompt
    _, st = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=0.0)
    present(f"| T = 0 (greedy) | {st['generated']} | {st['verified']} | {st['unique']} | {st['tokens_paid']:,} | "
            f"{st['tokens_used']} |")
    _, st = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=1.0)
    near(f"| T = 1 | {st['generated']} | # | # | {st['tokens_paid']:,} | # |", st["verified"], st["unique"],
         st["tokens_used"], tol=(6, 6, 72))
    assert st["tokens_used"] == 12 * st["unique"]
    present(f"(0.8)^12 = {0.8 ** 12:.3f}", f"about {12 / 0.8 ** 12:.0f} teacher tokens")
    greedy, kd = bench["greedy"], bench["kd"]
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, greedy, 400)
    ctx, _ = teacher.positions(greedy)
    ent = lambda m: float(-(m.probs(ctx) * np.log(m.probs(ctx))).sum(1).mean())
    near("Entropy (teacher #)", ent(teacher), tol=0.015)
    ebs = {}
    for label, m in (("SFT on the greedy text", sft), ("supervised KD on the greedy text (GKD λ = 0)", kd)):
        eb = ebs[label] = seqkd.exposure_bias(m, lang, greedy, bench["prompts"], np.random.default_rng(5))
        o, a = eb["own_prefixes"], eb["all_right_so_far"]
        near(f"| {label} | # | # | # | # | # | # | # |", eb["teacher_prefixes"].mean(), o[0], o[3], o[11], a[3], a[11],
             ent(m), tol=0.001 if m is sft else (0.001, 0.001, 0.02, 0.04, 0.015, 0.03, 0.02))
    near("entropy (# against the teacher's #)", ent(sft), ent(teacher), tol=(0.001, 0.015))
    k, t = ebs["supervised KD on the greedy text (GKD λ = 0)"], seqkd.exposure_bias(teacher, lang, greedy, bench["prompts"], np.random.default_rng(5))
    o, a, r, tr = k["own_prefixes"], k["all_right_so_far"], k["sampled_on_rule"], t["sampled_on_rule"]
    near("its samples follow the rule # of the time at position 1, like the teacher's, then # by position 4, where the "
         "teacher's stay between # and #", r[0], r[3], tr.min(), tr.max(), tol=(0.015, 0.04, 0.02, 0.02))
    near("per-position accuracy drops to # by position 4 and then stays there (# at position 12)", o[3], o[11],
         tol=(0.02, 0.04))
    near("falls to # by position 4 and # by position 12, where the teacher's stays at #", a[3], a[11],
         t["all_right_so_far"][-1], tol=(0.015, 0.03, 0.001))
    near("only # of its outputs were right all the way to position 12", a[11], tol=0.03)
    assert abs(r[0] - tr[0]) < 0.02 and r[3] < tr.min() - 0.1       # it slips more often than the teacher after position 1
    assert np.ptp(o[3:]) < 0.05 and np.all(np.diff(a) < 0)          # flat per position; compounding per output


def test_s3_and_s9_the_fixed_cost():
    s = K.SHAPES["qwen2.5-1.5b"]
    api = K.fixed_cost(100_000, 1, 2000, 9.00, s.params(), H100, 11, mfu=0.4)
    t_cost = K.serving(K.SHAPES["qwen2.5-32b"], H100, 11, 2048, 0.030, n_gpus=2)["usd_per_m"]
    own = K.fixed_cost(100_000, 1, 2000, t_cost, s.params(), H100, 11, mfu=0.4)
    present(f"that is ${api['generation_usd']:,.0f}", f"at §9's roofline cost, makes it ${own['generation_usd']:,.2f}",
            f"6·N·D = {api['train_flops'] / 1e18:.2f} × 10¹⁸ FLOPs", f"{api['gpu_hours']:.3f} GPU-hours on an H100 at 40% MFU, "
            f"${api['train_usd']:.2f} at $11/GPU-hour", f"{own['gpu_hours']:.3f} GPU-hours (${own['train_usd']:.2f})",
            f"The teacher's tokens are {own['generation_usd'] / own['total_usd']:.0%} of it self-hosted and "
            f"{api['generation_usd'] / api['total_usd']:.0%} through the API",
            f"${own['generation_usd']:,.2f} on the 32B self-hosted at TP = 2")
    present(f"(${own['generation_usd']:,.0f} on a self-hosted teacher, ${api['generation_usd']:,.0f} through an API")


def test_s4_on_policy_removes_exposure_bias(lang, teacher, bench):
    C = lang.contexts()
    rows = [("(nothing: supervised KD only)", bench["kd"])]
    for lam, beta, label in ((1.0, 0.0, "GKD λ = 1, β = 0 (forward)"), (1.0, 0.5, "GKD λ = 1, β = 0.5"),
                             (1.0, 1.0, "GKD λ = 1, β = 1 (reverse)")):
        s = bench["kd"].copy()
        op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=lam, beta=beta, data=bench["greedy"], seed=1)
        rows.append((label, s))
    tols = {"(nothing: supervised KD only)": (0.02, 0.04, 0.05), "GKD λ = 1, β = 0 (forward)": (0.025, 0.15, 0.15),
            "GKD λ = 1, β = 0.5": (0.08, 0.15, 0.15), "GKD λ = 1, β = 1 (reverse)": (0.08, 0.2, 0.15)}
    for label, m in rows:
        o = seqkd.own_accuracy(m, lang, bench["prompts"], 12, np.random.default_rng(5))
        near(f"| {label} | # | # | # |", o[3], o[11], E.vs_truth(m, lang)["rule_acc"], tol=tols[label])
    o_fwd = seqkd.own_accuracy(rows[1][1], lang, bench["prompts"], 12, np.random.default_rng(5))
    near("on-policy training held #", o_fwd[3], tol=0.025)
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, bench["greedy"], 400)
    after, at_start = {}, {}
    for label, start, tol in (("the KD student (§3)", bench["kd"], (0.05, 0.025, 0.15, 0.15)),
                              ("the SFT student (§3)", sft, (0.001, 0.001, 0.08, 0.3)),
                              ("a fresh 16-unit student", TinyLM(11, 16, 8, seed=1), (0.001, 0.001, 0.025, 0.25))):
        v = at_start[label] = E.vs_truth(start, lang)
        accs = []
        for beta in (0.0, 1.0):
            s = start.copy()
            op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=1.0, beta=beta, data=bench["greedy"], seed=1)
            after[(label, beta)] = E.vs_truth(s, lang)
            accs.append(after[(label, beta)]["rule_acc"])
        near(f"| {label} | # | # | # | # |", v["rule_acc"], v["wrong_right_q"], accs[0], accs[1], tol=tol)
        assert accs[1] < accs[0] - 0.3                              # reverse KL is slow from every start here
    kd_q, sft_q = at_start["the KD student (§3)"]["wrong_right_q"], at_start["the SFT student (§3)"]["wrong_right_q"]
    near("they give the right token # and #, so starting from them does not help here", kd_q, sft_q, tol=(0.025, 0.001))
    f1 = after[("a fresh 16-unit student", 1.0)]
    near("it puts # on a wrong token and # on the right one", f1["wrong_top_q"], f1["wrong_right_q"], tol=(0.15, 0.08))
    p_t, v_wrong = np.array([0.8, 0.1, 0.0999, 0.0001]), np.array([0.0, 0.0, 0.0, 9.0])
    ratio = np.linalg.norm(L.gkd(v_wrong[None], p_t[None], 1.0)[1]) / np.linalg.norm(L.softmax(v_wrong) - p_t)
    present(f"the reverse-KL gradient is {ratio:.4f} of the forward one")


def test_s4_the_policy_gradient_identities():
    t5, s5, prompt = fit_language(ModLang(5, 0.2), 16, steps=500), TinyLM(5, 4, 3, seed=1), np.array([1, 2])
    _, _, seqs = op.exact_seq_kl(s5, t5, prompt, 3)
    g = op.exact_pg_grad(s5, t5, prompt, 3)
    dirs = {k: np.random.default_rng(0).normal(size=a.shape) for k, a in s5.p.items()}
    sp, sm = s5.copy(), s5.copy()
    for k in dirs:
        sp.p[k] += 1e-5 * dirs[k]
        sm.p[k] -= 1e-5 * dirs[k]
    fd = -(op.exact_seq_kl(sp, t5, prompt, 3)[0] - op.exact_seq_kl(sm, t5, prompt, 3)[0]) / 2e-5
    an = sum(float((g[k] * dirs[k]).sum()) for k in dirs)
    assert abs(fd - an) / abs(an) < 1e-8
    present(f"over all {len(seqs)} three-token continuations of a 5-token language", "to 9 digits")
    flat = lambda d: np.concatenate([d[k].ravel() for k in sorted(d)])
    ex_seq, ex_tok = flat(g), flat(op.exact_pg_grad(s5, t5, prompt, 3, per_token=True))
    rng, A, B = np.random.default_rng(0), [], []
    for _ in range(200):
        s = s5.sample(np.tile(prompt, (64, 1)), 3, rng)
        A.append(flat(op.pg_grad(s5, t5, s)))
        B.append(flat(op.pg_grad(s5, t5, s, per_token=True)))
    near("differs by #% of the gradient's norm", 100 * np.linalg.norm(ex_tok - ex_seq) / np.linalg.norm(ex_seq), tol=1)
    near("total variance # against # per batch of 64", np.array(B).var(0).sum(), np.array(A).var(0).sum(), tol=0.01)
    g = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False)
    gr = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False, reference_params=8e9)
    d = op.flops_per_prompt(8e9, 32e9, 4096, samples=16)
    present(f"that is {d['per_token'] / 1e9:.0f} GFLOP per student token, against {g['per_token'] / 1e9:.0f} for GRPO "
            f"({gr['per_token'] / 1e9:.0f} with a reference model's forward pass",
            "makes each on-policy token twice as dear",
            f"that is {d['total'] / 1e15:.2f} × 10¹⁵ FLOPs against {g['total'] / 1e15:.2f} × 10¹⁵ "
            f"({gr['total'] / 1e15:.2f} × 10¹⁵ with the reference)")
    assert d["per_token"] == 2 * g["per_token"] and d["per_token"] == 2 * 8e9 + 6 * 8e9 + 2 * 32e9


def test_s5_distilling_reasoning(think):
    task, teacher = think
    # the RL teacher and its traces are sampled, but no matmul is in the loop: identical on every CPU tried
    e = R.LengthPolicy(32).expected(task)                          # untrained: a closed form
    present(f"| untrained student | {e['accuracy']:.3f} | {e['length']:.2f} | {e['p90']} | "
            f"{e['tokens_per_correct']:.2f} |")
    L_, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
    student = R.distil(L_, ok)
    for label, pol in (("teacher (RL, 16,000 rollouts)", teacher), ("student, SFT on 1,000 traces", student)):
        e = pol.expected(task)
        near(f"| {label} | # | # | # | # |", e["accuracy"], e["length"], e["p90"], e["tokens_per_correct"],
             tol=(0.001, 0.01, 1, 0.01))
    accs = []
    for n in (10, 30, 100, 300, 1000):
        Ln, okn = R.traces(teacher, task, np.random.default_rng(2), n)
        accs.append(R.distil(Ln, okn).expected(task)["accuracy"])
    near("1,000 traces reach #, #, #, # and #", *accs, tol=0.001)
    for label, keep, ml in (("all (plain SeqKD)", "all", None), ("correct only (rejection sampling)", "correct", None),
                            ("correct and L ≤ 16", "correct", 16), ("correct and L ≤ 12", "correct", 12)):
        kept = int(((ok if keep == "correct" else np.ones_like(ok)) & (L_ <= (ml or 99)) & (L_ < 32)).sum())
        e = R.distil(L_, ok, keep=keep, max_len=ml).expected(task)
        near(f"| {label} | # | # | # | # |", kept, e["accuracy"], e["length"], e["tokens_per_correct"],
             tol=(1, 0.001, 0.01, 0.01))
    rl = [R.reinforce(task, np.random.default_rng(3), n).expected(task)["accuracy"] for n in (1000, 4000, 16000)]
    near("SFT on 1,000 teacher traces reaches #; REINFORCE on the student with 1,000 rollouts reaches #, with 4,000 #, "
         "with 16,000 #", student.expected(task)["accuracy"], *rl, tol=0.001)
    weak = student.expected(task, q=0.05)
    near("copies the teacher's #-token thinking exactly and scores #, not #", weak["length"], weak["accuracy"],
         teacher.expected(task)["accuracy"], tol=(0.01, 0.001, 0.001))
    near(f"is {task.optimal_length(0.01, q=0.05):.1f} tokens, not the # it copied", student.expected(task)["length"],
         tol=0.1)
    near("its # is where 16,000 rollouts left it", teacher.expected(task)["length"], tol=0.1)
    near("(# tokens against # in the toy)", student.expected(task)["length"], teacher.expected(task)["length"], tol=0.1)


def prune_vs_fresh(lang, teacher, steps, draws=range(1, 6), inits=range(1, 5)):
    """Rule accuracy of the pruned teacher and of fresh 16-unit students after `steps` KD steps on 242 contexts,
    over several data draws (draw 1 is notebook 01's worked example)."""
    C, pruned, fresh = lang.contexts(), [], []
    for d in draws:
        rng = np.random.default_rng(d)
        ctx = C[rng.integers(0, 121, 242)]
        lang.sample_next(ctx, rng)                                # the same draws as notebook 01
        zt = teacher.logits(ctx)
        for m, out in [(teacher.prune_width(ctx, 16), pruned)] + [(TinyLM(11, 16, 8, seed=s), fresh) for s in inits]:
            if steps:
                train(m, ctx, lambda z, i: L.kd(z, zt[i], 1.0), steps)
            out.append(E.vs_truth(m, lang)["rule_acc"])
    return np.array(pruned), np.array(fresh)


def test_s6_prune_then_distil(lang, teacher):
    runs = {n: prune_vs_fresh(lang, teacher, n) for n in (0, 20, 100)}
    tols = {"pruned from the teacher": (0.06, 0.06, 0.06, 0.04, 0.08, 0.06, 0.04, 0.025, 0.06),
            "fresh": (0.001, 0.001, 0.001, 0.015, 0.06, 0.05, 0.015, 0.08, 0.1)}
    for i, label in enumerate(tols):                                  # mean (min–max) at 0, 20 and 100 steps
        near(f"| {label} | # (#–#) | # (#–#) | # (#–#) |",
             *(x for n in runs for x in (runs[n][i].mean(), runs[n][i].min(), runs[n][i].max())), tol=tols[label])
    (p20, f20), (p100, f100) = runs[20], runs[100]
    assert p20.min() > f20.max()                                  # a head start at every seed
    assert abs(p100.mean() - f100.mean()) < f100.std() + p100.std()   # no better destination: within seed noise


@pytest.fixture(scope="module")
def draft_world():
    base, dialect = ModLang(11, 0.2), ModLang(11, 0.2, skew=1.0)
    C = base.contexts()
    target = fit_language(dialect, 64)
    text = target.sample(C[np.random.default_rng(1).integers(0, 121, 500)], 12, np.random.default_rng(2))
    corpus = target.sample(C[np.random.default_rng(3).integers(0, 121, 2000)], 12, np.random.default_rng(4))
    PT = target.probs(C)

    def kd_draft(H):
        d = TinyLM(11, H, 8, seed=3)
        train(d, C, lambda z, i: L.soft_ce(z, PT[i]), 1500)
        return d

    sk = TinyLM(11, 16, 8, seed=3)
    seqkd.sft(sk, corpus, 1500, batch=512)
    return target, text, {"off": fit_language(base, 16, seed=3), "seqkd": sk}, {H: kd_draft(H) for H in (4, 8, 16, 32)}


def test_s7_the_formulas_on_the_serving_primers_numbers():
    P, Q = np.array([0.5, 0.3, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.4])
    a = Dr.acceptance_rate(P, Q)
    kb = Dr.best_k(a, 0.1)
    present(f"α = {a:.1f}, {Dr.expected_tokens(a, 4):.4f} tokens per pass at k = 4, and at c = 0.1 the best depth is "
            f"k = {kb} with a {Dr.speedup(a, kb, 0.1):.4f}× speedup",
            f"{Dr.greedy_acceptance(P, Q):.2f} against {a:.1f} on the primer's p and q",
            f"{Dr.vllm_view(a, 4)['draft_acceptance_rate']:.4f} at α = 0.6, k = 4")


def test_s7_distilled_against_off_the_shelf_and_size(draft_world):
    target, text, d, kd = draft_world
    c16 = Dr.draft_cost(kd[16].n_params, target.n_params)
    for label, m, tol in (("off-the-shelf (trained on the base language)", d["off"], (0.008, 0.0025, 0.008, 0.025)),
                          ("SeqKD from the target (24,000 of its tokens)", d["seqkd"], (0.15, 0.02, 0.2, 0.5)),
                          ("logit KD from the target", kd[16], (0.06, 0.02, 0.03, 0.25))):
        r = Dr.acceptance_on_text(target, m, text)
        near(f"| {label} | # | # | # | #× |", r["alpha"], r["greedy"], r["kl"], Dr.speedup(r["alpha"], 4, c16),
             tol=tol)
    present(f"k = 4, c = {c16:.3f}")                                # parameter counts: exact
    cs = {H: Dr.draft_cost(m.n_params, target.n_params) for H, m in kd.items()}
    al = {H: Dr.acceptance_on_text(target, m, text)["alpha"] for H, m in kd.items()}
    present("| c | " + " | ".join(f"{cs[H]:.3f}" for H in kd) + " |")
    sp = {H: Dr.speedup(al[H], 4, cs[H]) for H in kd}                # the 4- and 8-unit drafts are the chaotic ones
    near("| α (logit KD) | # | # | # | # |", *al.values(), tol=(0.2, 0.5, 0.06, 0.02))
    near("| speedup at k = 4 | #× | #× | #× | #× |", *sp.values(), tol=(0.5, 1.5, 0.25, 0.08))
    assert max(sp, key=sp.get) == 16                                  # the smallest draft that holds the target
    off, kd16 = Dr.acceptance_on_text(target, d["off"], text)["alpha"], al[16]
    near("(# → # in the toy)", off, kd16, tol=(0.008, 0.06))
    c = K.SHAPES["qwen3-0.6b"].params() / K.SHAPES["qwen3-4b"].params()
    present(f"c ≈ {c:.3f} by weight bytes, so k = 4 gives {Dr.speedup(0.6, 4, c):.2f}×, {Dr.speedup(0.7, 4, c):.2f}× and "
            f"{Dr.speedup(0.8, 4, c):.2f}× at α = 0.6, 0.7 and 0.8")


def test_s8_measuring_a_student(lang, teacher, bench):
    C, prompts = lang.contexts(), bench["prompts"]
    onp = bench["kd"].copy()
    op.gkd_train(onp, teacher, prompts, 12, 300, lam=1.0, beta=0.0, data=bench["greedy"], seed=1)
    students = (("KD on the teacher's text (§3)", "KD on the teacher's text", bench["kd"]),
                ("+ on-policy GKD (§4)", "+ on-policy GKD", onp),
                ("8 units, KD on everything (§1's capacity gap)", "8 units, KD on everything", fit_language(lang, 8)))
    zt = teacher.logits(C)
    common = set(map(tuple, prompts))
    rare = np.array([c for c in C if tuple(c) not in common])
    ok = lambda m, x, n: lang.verify(m.sample(x, n, None, 0.0))
    # the on-policy student samples its own training data, so it is the most CPU-sensitive: its rare-input accuracy
    # at n = 8 ran 0.50–0.87 across the 15 variants, where the other two students' are identical on all of them
    agree_tol = ((0.25, 0.05, 0.08), (0.4, 0.15, 0.25), (0.008, 0.001, 0.01))
    rare_tol = (0.001, (0.4, 0.4, 0.3, 0.6, 0.8, 0.5), 0.001)
    for (label1, label2, s), atol, rtol in zip(students, agree_tol, rare_tol):
        zs = s.logits(C)
        near(f"| {label1} | # | # | # |", E.kl(zt, zs), E.argmax_agreement(zt, zs), E.topk_overlap(zt, zs, 3), tol=atol)
        assert ok(s, prompts, 8).all()                              # so n = 2 too: every student is perfect here
        m = len(prompts)
        lo, hi = E.wilson_interval(m, m)
        cells = []
        for n in (2, 8):                                            # the rare slice: a count and its Wilson interval
            k = int(ok(s, rare, n).sum())
            cells += [k / len(rare), *E.wilson_interval(k, len(rare))]
        assert cells[0] < 1                                         # the gap is in the tail
        near(f"| {label2} | {m}/{m} ({lo:.3f}–{hi:.3f}) | # (#–#) | # (#–#) |", *cells, tol=rtol)
    lo, _ = E.wilson_interval(20, 20)
    present(f"anything tighter than {lo * 100:.0f}–100%")


def test_s8_a_student_that_beats_its_teacher(lang):
    C = lang.contexts()
    ctx = C[np.random.default_rng(7).integers(0, 121, 605)]
    y = lang.sample_next(ctx, np.random.default_rng(8))
    weak = TinyLM(11, 64, 8, seed=2)
    train(weak, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
    data = seqkd.teacher_data(weak, C, 20, 1, np.random.default_rng(9))
    filt, raw = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
    seqkd.sft(filt, seqkd.keep_verified(lang, data), 600)
    seqkd.sft(raw, data, 600)
    zw = weak.logits(C)
    acc = lambda m: E.vs_truth(m, lang)["rule_acc"]
    near("is right on # of contexts", acc(weak), tol=0.001)     # the same to the printed digit on every CPU tried
    near("is right on # — better than its teacher", acc(filt), tol=0.001)
    near("(KL #, top-1 #)", E.kl(zw, filt.logits(C)), E.argmax_agreement(zw, filt.logits(C)), tol=(0.01, 0.001))
    near("(# right, KL #, top-1 #)", acc(raw), E.kl(zw, raw.logits(C)), E.argmax_agreement(zw, raw.logits(C)),
         tol=0.001)


def test_s9_serving_break_even_and_the_cascade():
    serve = {}
    for key, n, label in (("qwen2.5-32b", 1, "Qwen2.5-32B (teacher), one GPU"), ("qwen2.5-32b", 2, "Qwen2.5-32B (teacher), TP = 2"),
                          ("qwen2.5-1.5b", 1, "Qwen2.5-1.5B (student)"), ("qwen2.5-0.5b", 1, "Qwen2.5-0.5B (student)")):
        s = serve[(key, n)] = K.serving(K.SHAPES[key], H100, 11, 2048, 0.030, n_gpus=n)
        usd = f"${s['usd_per_m']:.3f}" if s["usd_per_m"] > 0.1 else f"${s['usd_per_m']:.4f}"
        present(f"| {label} | {n} | {s['batch']} | {s['step_s'] * 1e3:.2f} ms | {s['tok_s']:,.0f} | {usd} |")
    t1, t, s = serve[("qwen2.5-32b", 1)], serve[("qwen2.5-32b", 2)], serve[("qwen2.5-1.5b", 1)]
    t4 = K.serving(K.SHAPES["qwen2.5-32b"], H100, 11, 2048, 0.030, n_gpus=4)
    mt, ms = K.SHAPES["qwen2.5-32b"], K.SHAPES["qwen2.5-1.5b"]
    present(f"the 32B's weights leave {80 * 0.9 - K.weight_gb(mt.params()):.2f} GB of the usable 72 for KV, room for {t1['batch']}",
            f"and the {t1['usd_per_m'] / s['usd_per_m']:.0f}× it gives is an artefact of that",
            f"it runs batch {t['batch']} at ${t['usd_per_m']:.3f} per million, and the student is "
            f"{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token: {mt.params() / ms.params():.0f}× fewer weight bytes and "
            f"{mt.kv_bytes_per_token() / ms.kv_bytes_per_token():.0f}× less KV per token let it run "
            f"{s['batch'] / (t['batch'] / 2):.0f}× the batch per GPU",
            f"At TP = 4 the teacher reaches ${t4['usd_per_m']:.3f} and the ratio {t4['usd_per_m'] / s['usd_per_m']:.0f}×",
            f"batch 1 already takes {K.decode_step(mt, H100, 1, 2048) * 1e3:.1f} ms — while the 1.5B still runs batch "
            f"{K.serving(ms, H100, 11, 2048, 0.010)['batch']}",
            f"~{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token on the roofline than a 32B teacher served on two H100s "
            f"(the {t1['usd_per_m'] / s['usd_per_m']:.0f}× you get against one H100",
            f"a 1.5B student serves {s['batch'] / (t['batch'] / 2):.0f}× the batch per GPU under the same ITL and is "
            f"~{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token on the roofline ({t1['usd_per_m'] / s['usd_per_m']:.0f}× against one H100")
    assert K.serving(mt, H100, 11, 2048, 0.010)["batch"] == 0      # one H100 misses a 10 ms ITL at batch 1
    ll, H = K.SHAPES["llama-3.1-8b"], H100
    b = K.best_batch(ll, H, 2048, 0.010)
    step = K.decode_step(ll, H, b, 2048)
    present(f"batch {b}, {step * 1e3:.2f} ms, {b / step:,.0f} tokens/s, ${K.cost_per_million_tokens(11, b / step):.3f} per million")
    own = K.fixed_cost(100_000, 1, 2000, t["usd_per_m"], ms.params(), H, 11)
    api = K.fixed_cost(100_000, 1, 2000, 9.00, ms.params(), H, 11)
    saving = t["usd_per_m"] - s["usd_per_m"]
    b50, b5, b1 = (K.break_even(own["total_usd"], t["usd_per_m"], s["usd_per_m"], v) for v in (50e6, 5e6, 1e6))
    a50, a5 = (K.break_even(api["total_usd"], t["usd_per_m"], s["usd_per_m"], v) for v in (50e6, 5e6))
    present(f"${own['total_usd']:,.2f} ÷ (${t['usd_per_m']:.3f} − ${s['usd_per_m']:.3f} per million) is "
            f"{b50['tokens'] / 1e6:.1f} million tokens: {b50['days']:.1f} days at 50 million output tokens a day "
            f"(${b50['saving_per_day']:.2f} a day saved), {b5['days']:.1f} days at 5 million, {b1['days']:.1f} days at 1 million",
            f"makes it ${api['total_usd']:,.2f} ÷ ${saving:.3f}, {a50['tokens'] / 1e6:,.1f} million tokens: "
            f"{a50['days']:.1f} days at 50 million a day, {a5['days']:.1f} at 5 million",
            f"a saving of ~${saving:.2f} per million over the teacher on two H100s — {b5['days']:.0f} days with self-hosted "
            f"data, {a5['days']:.0f} with API-bought data")
    assert abs(b50["tokens"] - 2e8) / 2e8 < 0.2                    # ≈ the volume the teacher wrote for the student
    c_t, c_s = 500 * t["usd_per_m"] / 1e6, 500 * s["usd_per_m"] / 1e6
    gate = K.cascade(c_s, c_t, (0.95, 0.30), (0.97, 0.85), 0.3, catch=0.8, false_alarm=0.1)
    present(f"every false alarm is a full teacher call ({0.7 * 0.1 * c_t / gate['cost']:.0%} of this gate's bill)")
    a_s, a_t = (0.95, 0.30), (0.97, 0.85)
    for label, kw in (("teacher only", dict(catch=1.0, false_alarm=1.0, student_first=False)),
                      ("student only", dict(catch=0.0, false_alarm=0.0)),
                      ("student first, gate catches 80% of hard, 10% false alarms", dict(catch=0.8, false_alarm=0.1)),
                      ("a perfect router up front", dict(catch=1.0, false_alarm=0.0, student_first=False))):
        r = K.cascade(c_s, c_t, a_s, a_t, 0.3, **kw)
        present(f"| {label} | {r['to_teacher']:.2f} | {r['accuracy']:.3f} | {r['cost_per_correct'] * 1e3:.3f} |")
