"""Run the lab router as a process.

    python -m igwlab.router --config default-weighted \
        --backend a=http://127.0.0.1:8001 --backend b=http://127.0.0.1:8002 --port 9000 \
        --objective premium=100 --objective batch=-10

`--config` takes a preset name (see `python -m igwlab.router --list-presets`), a YAML file, or
`-` for stdin. Backends are `name=url`; the name is what the router reports in the
`x-gateway-destination-endpoint` response header.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_config, preset_names
from .datalayer import Endpoint
from .server import Router, RouterSettings


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m igwlab.router", description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="default-weighted", help="preset name, YAML path, or - for stdin")
    ap.add_argument("--backend", action="append", default=[], help="name=url (repeatable)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--objective", action="append", default=[], help="InferenceObjective name=priority (repeatable)")
    ap.add_argument("--scrape-interval", type=float, default=0.05, help="seconds (default 0.05 = llm-d base tick)")
    ap.add_argument("--list-presets", action="store_true")
    return ap.parse_args(argv)


async def serve(args) -> None:
    src = sys.stdin.read() if args.config == "-" else args.config
    cfg = load_config(src)
    for w in cfg.warnings:
        print("config warning:", w, flush=True)
    eps = []
    for spec in args.backend:
        name, _, url = spec.partition("=")
        if not url:
            raise SystemExit(f"--backend must be name=url, got {spec!r}")
        eps.append(Endpoint(name=name, url=url.rstrip("/")))
    objectives = {}
    for spec in args.objective:
        name, _, prio = spec.partition("=")
        objectives[name] = int(prio)
    router = Router(cfg, eps, RouterSettings(scrape_interval_s=args.scrape_interval, objectives=objectives))
    url = await router.start(args.host, args.port)
    print(f"router listening on {url} with {len(eps)} backend(s)\n{cfg.describe()}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await router.stop()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_presets:
        print("\n".join(preset_names()))
        return 0
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
