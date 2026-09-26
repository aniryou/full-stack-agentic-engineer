"""python -m moelab — the lab's quick answers from the command line.

    python -m moelab env                                         # what this machine can run (tier)
    python -m moelab models                                      # total / active / KV per token
    python -m moelab fit --model olmoe-1b-7b --gpu T4            # what fits on one GPU, and why
    python -m moelab touched --experts 64 --top-k 8 --batch 1 8 64 256
    python -m moelab stream --model olmoe-1b-7b --dense qwen2.5-1.5b --gpu L4    # ITL vs batch (simulated)
    python -m moelab ep --model olmoe-1b-7b --gpu T4 --link pcie-2xT4            # TP vs EP on 2 GPUs (simulated)
    python -m moelab trace                                       # utilisation of the bundled router traces
    python -m moelab bench-parse FILE                            # read a `vllm bench serve` result
    python -m moelab.tinymoe                                     # train the tiny MoE (needs torch)
"""
from __future__ import annotations

import argparse
import sys

from . import configs, env, ep, hooks, offload, stream


def _models(_):
    for m in configs.MODELS.values():
        print(configs.summary(m))


def _fit(a):
    m, g = configs.get(a.model), configs.gpu(a.gpu)
    for scheme in a.scheme:
        print(offload.fit(m, g, scheme, max_model_len=a.max_model_len).line())
    if m.is_moe:
        need = offload.min_offload_gib(m, g, "fp16", kv_tokens=4 * a.max_model_len)
        print(f"   16-bit with 4 x {a.max_model_len} tokens of KV needs --cpu-offload-gb {need:g} "
              f"--cpu-offload-params experts: +{offload.offload_step_s(need, g.pcie_gbs) * 1e3:.0f} ms per step "
              f"at {g.pcie_gbs:g} GB/s PCIe (simulated)")


def _touched(a):
    print(f"E={a.experts} k={a.top_k}: batch -> experts touched per layer (uniform closed form | Zipf s={a.zipf} Monte Carlo)")
    for b in a.batch:
        mc, hot = stream.touched_mc(a.experts, a.top_k, b, s=a.zipf, trials=100)
        print(f"  {b:6d}  {stream.experts_touched(a.experts, a.top_k, b):7.1f}   {mc:7.1f}  (hottest expert {hot:.1%} of assignments)")


def _stream(a):
    g = configs.gpu(a.gpu)
    names = [a.model] + ([a.dense] if a.dense else [])
    print(f"decode step (ms) vs batch on {g.name}, context {a.context} [simulated: roofline x SimParams()]")
    print("batch " + "".join(f"{n:>18s}" for n in names))
    for b in a.batch:
        print(f"{b:5d} " + "".join(f"{stream.simulate_itl(configs.get(n), g, b, a.context) * 1e3:18.2f}" for n in names))


def _ep(a):
    rows = ep.compare_layouts(configs.get(a.model), configs.gpu(a.gpu), ep.LINKS[a.link], tuple(a.batch), a.context, s=a.zipf)
    print(ep.format_rows(rows))
    for lay in ep.LAYOUTS:
        print(f"  {lay:6s} vllm serve {configs.get(a.model).hf_id} {' '.join(ep.vllm_flags(lay))}")


def _trace(_):
    ts = hooks.load_fixture()
    print(ts.label)
    util = hooks.utilisation(ts.stacked(), ts.n_experts)
    bal = hooks.balancedness(util)
    div = hooks.domain_divergence(ts)
    print("layer  balancedness  hottest expert (share)  domain JS divergence")
    for l in range(util.shape[0]):
        e, share = hooks.hot_experts(util[l:l + 1], 1)[0][0]
        print(f"{l:5d}  {bal[l]:12.2f}  {e:6d} ({share:5.1%})         {div[l]:.3f}")


def _bench_parse(a):
    with open(a.file) as f:
        for k, v in ep.parse_bench_serve(f.read()).items():
            print(f"{k:45s} {v}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m moelab", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env").set_defaults(fn=lambda _: print(env.describe()))
    sub.add_parser("models").set_defaults(fn=_models)
    f = sub.add_parser("fit")
    f.add_argument("--model", default="olmoe-1b-7b")
    f.add_argument("--gpu", default="T4")
    f.add_argument("--scheme", nargs="+", default=["fp16", "int4"])
    f.add_argument("--max-model-len", type=int, default=4096)
    f.set_defaults(fn=_fit)
    t = sub.add_parser("touched")
    t.add_argument("--experts", type=int, default=64)
    t.add_argument("--top-k", type=int, default=8)
    t.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    t.add_argument("--zipf", type=float, default=1.0)
    t.set_defaults(fn=_touched)
    s = sub.add_parser("stream")
    s.add_argument("--model", default="olmoe-1b-7b")
    s.add_argument("--dense", default="qwen2.5-1.5b")
    s.add_argument("--gpu", default="L4")
    s.add_argument("--context", type=int, default=512)
    s.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    s.set_defaults(fn=_stream)
    e = sub.add_parser("ep")
    e.add_argument("--model", default="olmoe-1b-7b")
    e.add_argument("--gpu", default="T4")
    e.add_argument("--link", default="pcie-2xT4", choices=list(ep.LINKS))
    e.add_argument("--context", type=int, default=512)
    e.add_argument("--zipf", type=float, default=0.0)
    e.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128])
    e.set_defaults(fn=_ep)
    sub.add_parser("trace").set_defaults(fn=_trace)
    b = sub.add_parser("bench-parse")
    b.add_argument("file")
    b.set_defaults(fn=_bench_parse)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
