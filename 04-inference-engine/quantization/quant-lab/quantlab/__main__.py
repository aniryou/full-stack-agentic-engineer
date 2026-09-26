"""python -m quantlab <command> — the lab from a shell.

    plan      --gpu L4 [--scheme w4a16] [--kv-cache-dtype fp8]   which schemes run natively, flags, kernel
    compress  --scheme W4A16 --algo gptq --out out/tiny-W4A16    quantize the bundled tiny model (T0)
    compress  --model Qwen/Qwen2.5-0.5B-Instruct --scheme FP8_DYNAMIC --print   the llm-compressor script (T1)
    eval      [--schemes FP8_DYNAMIC,W4A16]                       the offline mini-eval on the tiny model
    kv        --model llama-3.1-8b-instruct --gpu L4               KV blocks and sessions per weight/KV dtype
    fake      --model qwen2.5-1.5b-instruct --gpu L4 --scheme fp8 --port 8000   a fake vLLM (simulated)
    bench     [--url http://127.0.0.1:8000] [--schemes bf16,fp8,w4a16]         measure a server, or simulate
    lm-eval   --model <id> [--backend vllm] [--tasks gsm8k] [--limit 250] [--run]   the lm_eval command
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m quantlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--scheme")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--model")
    c = sub.add_parser("compress")
    c.add_argument("--scheme", default="W4A16")
    c.add_argument("--algo", default="", help="comma-separated: gptq, awq, smoothquant (empty = RTN)")
    c.add_argument("--out", default="out/tiny")
    c.add_argument("--model", help="a Hugging Face id: print (or, with --run, execute) the llm-compressor script")
    c.add_argument("--print", action="store_true")
    c.add_argument("--run", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--schemes", default="FP8_DYNAMIC,W8A8,W4A16")
    e.add_argument("-n", type=int, default=500)
    k = sub.add_parser("kv")
    k.add_argument("--model", default="llama-3.1-8b-instruct")
    k.add_argument("--gpu", default="L4")
    k.add_argument("--typical-len", type=int, default=2000)
    f = sub.add_parser("fake")
    f.add_argument("--model", default="qwen2.5-1.5b-instruct")
    f.add_argument("--gpu", default="L4")
    f.add_argument("--scheme", default="bf16")
    f.add_argument("--kv-cache-dtype", default="auto")
    f.add_argument("--port", type=int, default=8000)
    b = sub.add_parser("bench")
    b.add_argument("--url")
    b.add_argument("--served-model")
    b.add_argument("--model", default="qwen2.5-1.5b-instruct")
    b.add_argument("--gpu", default="L4")
    b.add_argument("--schemes", default="bf16,fp8,w4a16")
    b.add_argument("--users", type=int, default=4)
    b.add_argument("-n", type=int, default=16)
    b.add_argument("--prompt-len", type=int, default=512)
    b.add_argument("--output-len", type=int, default=64)
    le = sub.add_parser("lm-eval")
    le.add_argument("--model", required=True)
    le.add_argument("--backend", default="vllm")
    le.add_argument("--tasks", default="gsm8k")
    le.add_argument("--limit", type=float)
    le.add_argument("--num-fewshot", type=int)
    le.add_argument("--base-url")
    le.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        from . import serve
        if not a.scheme:
            print(serve.matrix(gpus=(a.gpu,)))
            return 0
        pl = serve.plan(a.scheme, a.gpu, model=a.model, kv_cache_dtype=a.kv_cache_dtype)
        print(f"{pl.scheme} on {pl.gpu}: {'runs' if pl.supported else 'does NOT run'} — {pl.compute}")
        print(f"kernel (verify in the startup log): {pl.kernel}")
        for n in pl.notes:
            print("note:", n)
        print(pl.command())
        print(pl.docker())
        return 0
    if a.cmd == "compress":
        from . import compress as C, tinymodel as tm
        recipe = C.Recipe(a.scheme, tuple(x for x in a.algo.split(",") if x))
        if a.model:
            src = C.llmcompressor_script(recipe, a.model)
            if a.run:
                C.run_llmcompressor(recipe, a.model, a.out)
            else:
                print(src)
            return 0
        rep = C.quantize_checkpoint(tm.TINY_DIR, a.out, recipe)
        print(f"{rep['recipe']}: {rep['dense_bytes']:,} -> {rep['bytes']:,} bytes at {rep['path']}")
        probs = C.validate_checkpoint(a.out)
        print("validate:", "ok" if not probs else probs)
        return 0 if not probs else 1
    if a.cmd == "eval":
        from . import compress as C, evalharness as E, tinymodel as tm
        ref = tm.load()
        res, cache = {"bf16 (reference)": E.mini_eval(ref, ref, n=a.n)}, {}
        for sch in a.schemes.split(","):
            qm = C.quantize_model(ref, C.Recipe(sch))
            res[sch] = E.mini_eval(qm.model(), ref, n=a.n, act_quant=qm.act_quant(), ref_cache=cache)
        print("measured on the bundled tiny model (T0):")
        print(E.table(res))
        return 0
    if a.cmd == "kv":
        from . import kv
        print(kv.table(a.model, a.gpu, a.typical_len))
        return 0
    if a.cmd == "fake":
        import time
        from . import bench
        from .fakeserver import FakeServer
        prof = bench.profile(a.model, a.gpu, a.scheme, kv_cache_dtype=a.kv_cache_dtype)
        srv = FakeServer(prof, model=a.model, port=a.port)
        print(f"fake vLLM (SIMULATED) on {srv.start()}: {prof.describe()}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            srv.stop()
        return 0
    if a.cmd == "bench":
        from . import bench, env
        if a.url:
            model = a.served_model or env.served_model(a.url)
            r = bench.run_http(a.url, model, users=a.users, n_requests=a.n, prompt_len=a.prompt_len,
                               output_len=a.output_len, headers=env.auth_headers())
            print(json.dumps(r, indent=2))
            return 0
        from . import report
        rows = bench.compare(a.schemes.split(","), a.model, a.gpu, users=a.users, n_requests=a.n,
                             prompt_len=a.prompt_len, output_len=a.output_len)
        print(report.markdown(rows, ["scheme", "ttft_ms_mean", "tpot_ms_mean", "output_tok_s", "source"]))
        return 0
    if a.cmd == "lm-eval":
        from . import evalharness as E
        limit = int(a.limit) if a.limit and a.limit >= 1 else a.limit
        cmd = E.lm_eval_command(a.model, backend=a.backend, tasks=a.tasks.split(","), limit=limit,
                                num_fewshot=a.num_fewshot, base_url=a.base_url)
        print(E.shell(cmd))
        if a.run:
            print(E.run_lm_eval(cmd).stdout)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
