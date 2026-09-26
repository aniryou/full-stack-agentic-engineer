"""python -m thinklab <command> — the lab from a terminal.

    env                         what this machine can run (tier detection)
    fake [--port 8000]          serve the simulated thinking model (a fake vLLM; every number simulated)
    ask "question" [--url U]    one streamed answer: reasoning, content, tokens, TTFT and time-to-answer
    eval [--url U] [-n 20]      the generated eval set with thinking off / on / budgeted; accuracy and tokens
    shape [--profile P]         the serving shape of thinking vs not (virtual-time engine; simulated)
    tinyrl [--rl-steps 30]      SFT + GRPO on the tiny transformer (needs torch; CPU is fine)
    rl-step [--model M]         T1: one GRPO step with vLLM generating the rollouts (GPU + vllm + transformers)
"""
from __future__ import annotations

import argparse
import sys

from . import env
from .report import table


def _target(url):
    if url:
        return env.Target(url, env.is_simulated(url, env.auth_headers()), "T1/T3", "given URL", headers=env.auth_headers())
    return env.connect(time_scale=0.05)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="thinklab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    f = sub.add_parser("fake")
    f.add_argument("--port", type=int, default=8000)
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--profile", default="t4-qwen3-0.6b")
    f.add_argument("--time-scale", type=float, default=1.0)
    f.add_argument("--reasoning-field", default="reasoning", choices=["reasoning", "reasoning_content"])
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--url")
    a.add_argument("--no-think", action="store_true")
    a.add_argument("--budget", type=int)
    a.add_argument("--max-tokens", type=int)
    e = sub.add_parser("eval")
    e.add_argument("--url")
    e.add_argument("-n", type=int, default=20)
    e.add_argument("--budgets", default="256,1024")
    s = sub.add_parser("shape")
    s.add_argument("--profile", default="t4-qwen3-0.6b")
    s.add_argument("--rate", type=float, default=2.0)
    s.add_argument("-n", type=int, default=300)
    t = sub.add_parser("tinyrl")
    t.add_argument("--sft-steps", type=int)
    t.add_argument("--rl-steps", type=int)
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("rl-step")
    r.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    r.add_argument("--prompts", type=int, default=8)
    r.add_argument("-g", type=int, default=8)
    args = ap.parse_args(argv)

    if args.cmd == "env":
        print(env.describe())
    elif args.cmd == "fake":
        from .fakeserver import FakeServer
        FakeServer(args.profile, host=args.host, port=args.port, time_scale=args.time_scale,
                   reasoning_field=args.reasoning_field).serve_forever()
    elif args.cmd == "ask":
        from .thinking.client import ThinkingClient
        tgt = _target(args.url)
        c = ThinkingClient(tgt.url, headers=tgt.headers).chat(
            [{"role": "user", "content": args.question}], stream=True, thinking=not args.no_think,
            budget=args.budget, max_tokens=args.max_tokens)
        print(f"[{tgt.label}] reasoning ({c.reasoning_tokens} tokens): {(c.reasoning or '')[:300]}")
        print(f"content: {c.content}")
        print(f"finish_reason {c.finish_reason} | TTFT {c.ttft * 1e3:.0f} ms | answer starts {c.ttfc * 1e3:.0f} ms | "
              f"total {c.latency:.2f} s | completion tokens {c.completion_tokens}")
        tgt.stop()
    elif args.cmd == "eval":
        from .thinking import budget as B
        from .thinking.client import ThinkingClient
        from .thinking.evalset import make_evalset
        tgt = _target(args.url)
        cl = ThinkingClient(tgt.url, headers=tgt.headers)
        probs = make_evalset(args.n)
        rows = []
        for name, kw in [("thinking off", {"thinking": False}), ("thinking on", {})]:
            comps = cl.chat_many([p.messages() for p in probs], max_tokens=4096, **kw)
            rows.append({"mode": name, **B.summarize(probs, comps, name, None).as_dict()})
        for b in [int(x) for x in args.budgets.split(",") if x]:
            rows.append({"mode": f"budget {b}", **B.sweep(cl, probs, [b])[0].as_dict()})
        print(table(rows, ["mode", "accuracy", "mean_reasoning_tokens", "mean_output_tokens", "cut_in_thinking"],
                    f"[{tgt.label}] {len(probs)} generated problems"))
        tgt.stop()
    elif args.cmd == "shape":
        from .engine import profile
        from .thinking.evalset import make_evalset
        from .workload import simulate_modes
        qs = [p.prompt for p in make_evalset(args.n, seed=1)]
        rows = simulate_modes(profile(args.profile), qs, {"no thinking": {"thinking": False}, "thinking": {},
                                                          "budget 512": {"budget": 512}}, rate=args.rate)
        print(table(rows, title=f"SIMULATED: {args.n} requests at {args.rate}/s on {args.profile}"))
    elif args.cmd == "tinyrl":
        if not env.has_torch():
            print("torch is not available (pip install torch; the CPU build is enough). The recorded run:")
            from .tinyrl.curves import load_recorded, show
            print(show(load_recorded()))
            return 0
        from .tinyrl.curves import show
        from .tinyrl.train import TinyRLConfig, run
        cfg = TinyRLConfig(seed=args.seed)
        if args.sft_steps is not None:
            cfg.sft_steps = args.sft_steps
        if args.rl_steps is not None:
            cfg.rl_steps = args.rl_steps
        print(show(run(cfg)))
    elif args.cmd == "rl-step":
        from .rollout import one_grpo_step
        one_grpo_step(args.model, n_prompts=args.prompts, n=args.g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
