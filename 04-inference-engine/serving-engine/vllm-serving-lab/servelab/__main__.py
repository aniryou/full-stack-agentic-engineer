"""servelab command line: ``python -m servelab {fake,bench,size,metrics,env}``.

    python -m servelab fake --profile t4-qwen2.5-0.5b --port 8000          # a fake vLLM (T0)
    python -m servelab bench --url http://127.0.0.1:8000 --rate 5 -n 100    # measure any OpenAI-compatible server
    python -m servelab bench --url ... --workload agent --sessions 20       # shared-prefix agent sessions
    python -m servelab size --model llama-3.1-8b-instruct --gpu L4 --max-model-len 16384 --quantization fp8
    python -m servelab metrics --url http://127.0.0.1:8000 --window 10      # engine snapshot over 10 s
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time


def _fake(a) -> int:
    from .fake_engine import PROFILES, EngineConfig
    from .fakeserver import FakeServer
    cfg = EngineConfig(max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                       enable_prefix_caching=not a.no_enable_prefix_caching,
                       num_speculative_tokens=a.spec_tokens, spec_acceptance=a.spec_acceptance)
    if a.profile not in PROFILES and a.profile != "tiny":
        print(f"unknown profile; choose from {', '.join(PROFILES)}", file=sys.stderr)
        return 2
    FakeServer(a.profile, cfg, host=a.host, port=a.port, model=a.served_model_name).serve_forever()
    return 0


def _bench(a) -> int:
    from .bench import SLO, Lengths, agent_sessions, random_requests, run_closed_loop, run_open_loop, run_sessions
    from .bench.report import save_json
    from .env import auth_headers
    headers = auth_headers()
    slo = SLO(a.slo_ttft_ms, a.slo_tpot_ms, a.slo_e2el_ms) if (a.slo_ttft_ms or a.slo_tpot_ms or a.slo_e2el_ms) else None
    if a.workload == "agent":
        run = run_sessions(a.url, agent_sessions(a.sessions, turns=a.turns, layout=a.layout, seed=a.seed),
                           a.rate, model=a.model, headers=headers)
    else:
        reqs = random_requests(a.num_requests, Lengths.fixed(a.input_len), Lengths.fixed(a.output_len),
                               prefix_tokens=a.prefix_len, seed=a.seed)
        if a.concurrency:
            run = run_closed_loop(a.url, reqs, a.concurrency, model=a.model, headers=headers, warmup=a.warmup)
        else:
            run = run_open_loop(a.url, reqs, a.rate, burstiness=a.burstiness, seed=a.seed, model=a.model,
                                headers=headers, warmup=a.warmup)
    print(run.report(slo))
    if a.json:
        print("saved", save_json(run, a.json))
    return 0


def _size(a) -> int:
    from .sizing import size
    r = size(a.model, a.gpu, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
             block_size=a.block_size, dtype=a.dtype, quantization=a.quantization, kv_cache_dtype=a.kv_cache_dtype,
             tensor_parallel_size=a.tensor_parallel_size, typical_len=a.typical_len,
             gpu_memory_bytes=int(a.gpu_memory_gib * 1024**3) if a.gpu_memory_gib else None)
    print(json.dumps(r.as_dict(), indent=1, default=str) if a.json else r.summary())
    return 0 if r.fits else 1


def _metrics(a) -> int:
    from . import metrics
    from .env import auth_headers
    h = auth_headers()
    first = metrics.scrape(a.url, headers=h)
    if a.window:
        time.sleep(a.window)
        print(metrics.snapshot(metrics.scrape(a.url, headers=h), first).table())
    else:
        print(metrics.snapshot(first).table())
    return 0


def _env(a) -> int:
    from . import env
    print(env.describe())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="servelab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fake", help="run the fake vLLM (simulated)")
    f.add_argument("--profile", default="t4-qwen2.5-0.5b")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8000)
    f.add_argument("--served-model-name", default=None)
    f.add_argument("--max-num-seqs", type=int, default=256)
    f.add_argument("--max-num-batched-tokens", type=int, default=2048)
    f.add_argument("--no-enable-prefix-caching", action="store_true")
    f.add_argument("--spec-tokens", type=int, default=0)
    f.add_argument("--spec-acceptance", type=float, default=0.6)
    f.set_defaults(fn=_fake)

    b = sub.add_parser("bench", help="benchmark an OpenAI-compatible server")
    b.add_argument("--url", required=True)
    b.add_argument("--model", default=None, help="default: first id from /v1/models")
    b.add_argument("--workload", choices=["random", "agent"], default="random")
    b.add_argument("-n", "--num-requests", type=int, default=100)
    b.add_argument("--input-len", type=int, default=512)
    b.add_argument("--output-len", type=int, default=128)
    b.add_argument("--prefix-len", type=int, default=0)
    b.add_argument("--rate", type=float, default=math.inf, help="open-loop req/s (sessions/s for --workload agent)")
    b.add_argument("--burstiness", type=float, default=1.0)
    b.add_argument("--concurrency", type=int, default=0, help="closed loop with N users instead of --rate")
    b.add_argument("--sessions", type=int, default=20)
    b.add_argument("--turns", type=int, default=4)
    b.add_argument("--layout", default="stable", choices=["stable", "timestamp_first", "shuffled_tools"])
    b.add_argument("--warmup", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--slo-ttft-ms", type=float, default=None)
    b.add_argument("--slo-tpot-ms", type=float, default=None)
    b.add_argument("--slo-e2el-ms", type=float, default=None)
    b.add_argument("--json", default=None, help="save per-request results here")
    b.set_defaults(fn=_bench)

    s = sub.add_parser("size", help="predict KV blocks and max concurrency")
    s.add_argument("--model", required=True, help="bundled name or path to config.json")
    s.add_argument("--gpu", default=None)
    s.add_argument("--gpu-memory-gib", type=float, default=None, help="instead of --gpu")
    s.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    s.add_argument("--max-model-len", type=int, default=None)
    s.add_argument("--block-size", type=int, default=16)
    s.add_argument("--dtype", default="auto")
    s.add_argument("--quantization", default=None)
    s.add_argument("--kv-cache-dtype", default="auto")
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--typical-len", type=int, default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_size)

    m = sub.add_parser("metrics", help="scrape /metrics and summarize the engine")
    m.add_argument("--url", required=True)
    m.add_argument("--window", type=float, default=0.0, help="seconds between two scrapes")
    m.set_defaults(fn=_metrics)

    e = sub.add_parser("env", help="what is available on this machine")
    e.set_defaults(fn=_env)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
