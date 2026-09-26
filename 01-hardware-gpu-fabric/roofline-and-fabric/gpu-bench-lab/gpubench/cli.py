"""Command line: ``python -m gpubench {info,run,topo,inventory,show}``.

    python -m gpubench info                         # what is this machine?
    python -m gpubench run --out results            # the suite (auto backend), JSON + Markdown
    python -m gpubench run --backend torch --full   # bigger sizes, on a GPU
    python -m gpubench topo [FILE]                  # analyse `nvidia-smi topo -m` (runs it if no FILE)
    python -m gpubench inventory [FILE]             # parse `nvidia-smi --query-gpu ... --format=csv`
    python -m gpubench show results/x.json          # re-render a saved report
"""
from __future__ import annotations

import argparse
import json
import sys


def _read(path):
    return sys.stdin.read() if path == "-" else open(path).read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpubench", description="Measure the machine you have.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="describe this machine (CPU, GPUs, backend)")
    r = sub.add_parser("run", help="run the benchmark suite and write a report")
    r.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    r.add_argument("--full", action="store_true", help="larger sizes (minutes, not seconds)")
    r.add_argument("--tiny", action="store_true", help="plumbing check for CI: tiny sizes, meaningless numbers")
    r.add_argument("--suite", default=",".join(("inventory", "gemm", "stream", "transfer", "p2p", "load")),
                   help="comma-separated subset of: inventory,gemm,stream,transfer,p2p,load")
    r.add_argument("--out", default="results", help="directory for the JSON and Markdown report")
    r.add_argument("--workdir", default=None, help="where to write the synthetic checkpoint: put it on the disk you want measured "
                        "(default: a temp dir on a real disk — /tmp is skipped if it is tmpfs)")
    t = sub.add_parser("topo", help="analyse `nvidia-smi topo -m` output")
    t.add_argument("file", nargs="?", help="saved output ('-' for stdin); omit to run nvidia-smi")
    t.add_argument("--tp", type=int, default=0, help="also suggest the best group of this many GPUs")
    i = sub.add_parser("inventory", help="parse `nvidia-smi --query-gpu=... --format=csv` output")
    i.add_argument("file", nargs="?", help="saved output ('-' for stdin); omit to run nvidia-smi")
    s = sub.add_parser("show", help="print a saved JSON report as Markdown")
    s.add_argument("file")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        from .backends import get_backend
        from .inventory import query_gpus
        be = get_backend("auto")
        print(json.dumps(be.describe(), indent=1, default=str))
        rows, raw = query_gpus()
        print(raw if rows else f"(no GPUs: {raw})")
        return 0
    if a.cmd == "run":
        from .suite import run_suite
        rep = run_suite(a.backend, quick=not a.full, suites=tuple(a.suite.split(",")), workdir=a.workdir,
                        tiny=a.tiny)
        jp, mp = rep.save(a.out)
        print(rep.to_markdown())
        print(f"\nwrote {jp}\nwrote {mp}")
        return 0
    if a.cmd == "topo":
        from . import topo
        text = _read(a.file) if a.file else topo.run_nvidia_smi()
        if not text:
            print("no nvidia-smi here; pass a saved `nvidia-smi topo -m` output file", file=sys.stderr)
            return 1
        tp = topo.parse(text)
        print(tp.summary())
        if a.tp:
            print(f"best {a.tp}-GPU group: {', '.join(topo.best_group(tp, a.tp))}")
        return 0
    if a.cmd == "inventory":
        from .inventory import QUERY, health, parse_csv, query_gpus
        if a.file:
            rows = parse_csv(_read(a.file))
        else:
            rows, raw = query_gpus()
            if rows is None:
                print(f"{raw}. Save the output of: {' '.join(QUERY)}", file=sys.stderr)
                return 1
        for row in rows:
            print({k: v for k, v in row.items() if k != "_units"})
        for f in health(rows):
            print(f)
        return 0
    if a.cmd == "show":
        from .report import Report
        print(Report.load(a.file).to_markdown())
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
