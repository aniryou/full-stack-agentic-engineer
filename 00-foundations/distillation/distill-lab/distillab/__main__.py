"""python -m distillab <command> — the lab from a terminal.

    env                                   what this machine can run (tier detection)
    fake [--port 8000] [--profile P]      serve the simulated teacher (a fake vLLM; every answer simulated)
    tinylm [--steps 1500]                 teacher + four students on the tiny task (needs torch; a CPU is fine)
    teacher-data [--url U] [-n 4]         teacher samples -> verifier -> dedup -> JSONL for TRL (fake teacher by default)
    memory [--student S] [--teacher T]    will this training run fit on a T4 / L4 / ... (predicted)
    cost [--teacher T] [--student S]      serving $/M for both, the fixed cost, break-even (roofline; predicted)
    spec-config [--target T] [--draft D]  the vllm serve command for a target with a draft model
"""
from __future__ import annotations

import argparse
import json
import sys

from . import env
from .report import table


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="distillab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    f = sub.add_parser("fake")
    f.add_argument("--port", type=int, default=8000)
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--profile", default="teacher", choices=["teacher", "student", "thinker"])
    t = sub.add_parser("tinylm")
    t.add_argument("--steps", type=int, help="student steps (default 1500)")
    t.add_argument("--methods", default="hard,kd,seqkd,gkd")
    t.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("teacher-data")
    d.add_argument("--url", help="an OpenAI-compatible teacher (default: DISTILLAB_URL, else the in-process fake)")
    d.add_argument("--problems", type=int, default=100)
    d.add_argument("-n", type=int, default=4)
    d.add_argument("--temperature", type=float, default=0.7)
    d.add_argument("--max-tokens", type=int, default=512)
    d.add_argument("--out", default="_run_outputs/teacher")
    m = sub.add_parser("memory")
    m.add_argument("--student", default="qwen2.5-0.5b-instruct")
    m.add_argument("--teacher", help="a frozen teacher beside the student (logit KD / GKD)")
    m.add_argument("--gpu", default="T4")
    m.add_argument("--regime", default="full", choices=["full", "pure_bf16", "lora"])
    m.add_argument("--batch", type=int, default=4)
    m.add_argument("--seq", type=int, default=512)
    m.add_argument("--chunk", type=int)
    c = sub.add_parser("cost")
    c.add_argument("--teacher", default="qwen2.5-32b-instruct")
    c.add_argument("--student", default="qwen2.5-1.5b-instruct")
    c.add_argument("--gpu", default="H100")
    c.add_argument("--context", type=int, default=2048)
    c.add_argument("--itl-ms", type=float, default=30.0)
    c.add_argument("--teacher-tokens", type=float, default=2e8, help="tokens the teacher generates for the data")
    c.add_argument("--teacher-price", type=float, default=9.0, help="$ per 1M teacher output tokens")
    c.add_argument("--tokens-per-day", type=float, default=1e9)
    s = sub.add_parser("spec-config")
    s.add_argument("--target", default="Qwen/Qwen3-4B")
    s.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    s.add_argument("-k", type=int, default=4)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    if a.cmd == "env":
        print(env.describe())
    elif a.cmd == "fake":
        from .fakeserver import FakeTeacher
        FakeTeacher(a.profile, host=a.host, port=a.port).serve_forever()
    elif a.cmd == "tinylm":
        from .tinylm.curves import load_recorded, show
        if not env.has_torch():
            print("torch is not available (pip install torch; the CPU build is enough). The recorded run:")
            print(show(load_recorded()))
            return 0
        from .tinylm.train import DistillConfig, run
        cfg = DistillConfig(seed=a.seed, methods=tuple(a.methods.split(",")))
        if a.steps:
            cfg.student_steps = a.steps
        print(show(run(cfg)))
    elif a.cmd == "teacher-data":
        from . import data as D
        from . import teacher as TE
        from .client import Client
        url = a.url or env.server_url()
        tgt = env.Target(url, env.is_simulated(url, env.auth_headers()), "T1/T3", "given URL",
                         env.first_model(url, env.auth_headers()), headers=env.auth_headers()) if url else env.connect()
        train, _ = D.decontaminate(D.make_set(a.problems, seed=0, split="train"), D.make_set(200, seed=0))
        samples = TE.generate(Client(tgt.url, tgt.model, tgt.headers), train, n=a.n, temperature=a.temperature,
                              max_tokens=a.max_tokens)
        kept, rows = TE.funnel(samples)
        print(table(rows, title=f"[{tgt.label}] teacher data from {tgt.model}"))
        print(table([TE.token_bill(samples)], title="the bill (every generated token, kept or not)"))
        p1 = TE.write_jsonl(TE.to_messages(kept), a.out + "_msgs.jsonl")
        p2 = TE.write_jsonl(TE.to_prompt_completion(kept), a.out + "_pc.jsonl")
        print(f"wrote {p1} (GKD / KD) and {p2} (SFT)")
        tgt.stop()
    elif a.cmd == "memory":
        from .cost import load_config
        from .hf.memory import plan
        r = plan(load_config(a.student), gpu=a.gpu, regime=a.regime, batch=a.batch, seq=a.seq,
                 teacher=load_config(a.teacher) if a.teacher else None, chunk=a.chunk)
        print(table(r["rows"], title=f"PREDICTED: {a.student} ({a.regime}) on {a.gpu}, batch {a.batch} x {a.seq}"))
        print(f"total {r['total GB']} GB of {r['usable GB']} GB usable -> {'fits' if r['fits'] else 'does NOT fit'}")
    elif a.cmd == "cost":
        from . import cost as C
        gpu = C.GPUS[a.gpu]
        tm, sm = C.shape(a.teacher), C.shape(a.student)
        rows = [C.serving(x, gpu, context=a.context, itl_s=a.itl_ms / 1e3) for x in (tm, sm)]
        print(table(rows, title=f"PREDICTED (roofline bound): {a.gpu} at ${gpu.price_hr}/GPU-h, {a.context} context, "
                                f"ITL {a.itl_ms} ms"))
        fixed = C.fixed_cost(teacher_tokens=a.teacher_tokens, teacher_price_per_m=a.teacher_price,
                             student_params=sm.params(), train_tokens=a.teacher_tokens, gpu=gpu)
        print(json.dumps({k: round(v, 3) for k, v in fixed.items()}, indent=1))
        if rows[0]["tok/s"] and rows[1]["tok/s"]:
            be = C.break_even(fixed["total $"], rows[0]["$/M"], rows[1]["$/M"], a.tokens_per_day)
            print(f"break-even after {be['million tokens']:,.1f} M tokens = {be['days']:.2f} days at "
                  f"{a.tokens_per_day:,.0f} tokens/day")
    elif a.cmd == "spec-config":
        from .draft import serve_args, speculative_config
        import shlex
        print(shlex.join(serve_args(a.target, speculative_config(a.draft, a.k))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
