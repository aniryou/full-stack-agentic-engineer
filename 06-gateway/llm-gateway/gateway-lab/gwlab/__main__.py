"""Command line: run the gateway, a fake provider, or the whole stack; issue a key; read a ledger.

    python -m gwlab stack --port 8080                      # fakes + gateway in one process (T0), a demo key printed
    python -m gwlab fake --name acme --dialect openai --port 8101 [--fail-rate 0.3] [--tpm 20000]
    python -m gwlab gateway --config lab --port 8080       # provider URLs from ACME_URL / BOLT_URL / GWLAB_VLLM_URL
    python -m gwlab key --url http://127.0.0.1:8080 --tenant team-a
    python -m gwlab ledger --db gateway.db
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _serve_gateway(args):
    from .gateway import Gateway, load_config
    cfg = load_config(args.config)
    gw = Gateway(cfg)

    async def main():
        url = await gw.start(host=args.host, port=args.port)
        print(f"gwlab gateway on {url} (config {cfg.source}); providers: " +
              ", ".join(f"{p.name}={p.base_url}" for p in cfg.providers.values()), flush=True)
        await asyncio.Event().wait()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass



def _serve_fake(args):
    from .fakes import FakeProvider, FakeSpec
    kw = dict(rpm=args.rpm, tpm=args.tpm, fail_rate=args.fail_rate, seed=args.seed, minute_s=args.minute_s)
    if args.output_tokens:
        kw["output_tokens"] = args.output_tokens
    if args.keys:
        kw["keys"] = tuple(args.keys.split(","))
    spec = FakeSpec.anthropic(args.name, **kw) if args.dialect == "anthropic" else FakeSpec(name=args.name, **kw)
    FakeProvider(spec, host=args.host, port=args.port).serve_forever()


def _stack(args):
    import time

    from .stack import LocalStack
    with LocalStack(gateway_port=args.port) as s:
        key = s.issue_key("team-a")
        print(f"gateway   {s.url}   (admin token: {s.admin_token})")
        for n, f in s.fakes.items():
            print(f"provider  {n}: {f.url} ({f.spec.dialect} dialect, SIMULATED)")
        print(f"demo key for team-a (shown once): {key}")
        print(f"try: curl -s {s.url}/v1/chat/completions -H 'Authorization: Bearer {key}' "
              "-H 'Content-Type: application/json' -d '{\"model\":\"chat\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def _key(args):
    from .client import post_json
    body = {"tenant": args.tenant}
    if args.aliases:
        body["aliases"] = args.aliases.split(",")
    if args.budget_usd is not None:
        body["budget_usd"] = args.budget_usd
    r = post_json(args.url.rstrip("/") + "/admin/keys", body, token=args.admin_token)
    if r.status != 200:
        print(r.status, r.json, file=sys.stderr)
        return 1
    print(r.json["key"])
    return 0


def _ledger(args):
    from . import report
    from .gateway.store import Store
    rows = Store(args.db).query("SELECT * FROM ledger ORDER BY ts")
    print(report.totals_table(rows, args.by))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gwlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gateway")
    g.add_argument("--config", default=os.environ.get("GWLAB_CONFIG", "lab"))
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8080)
    f = sub.add_parser("fake")
    f.add_argument("--name", default="acme")
    f.add_argument("--dialect", choices=["openai", "anthropic"], default="openai")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8101)
    f.add_argument("--keys", default="")
    f.add_argument("--rpm", type=int, default=0)
    f.add_argument("--tpm", type=int, default=0)
    f.add_argument("--minute-s", dest="minute_s", type=float, default=60.0)
    f.add_argument("--fail-rate", dest="fail_rate", type=float, default=0.0)
    f.add_argument("--output-tokens", dest="output_tokens", type=int, default=0)
    f.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("stack")
    s.add_argument("--port", type=int, default=8080)
    k = sub.add_parser("key")
    k.add_argument("--url", default=os.environ.get("GWLAB_URL", "http://127.0.0.1:8080"))
    k.add_argument("--admin-token", default=os.environ.get("GWLAB_ADMIN_TOKEN", "dev-admin-token"))
    k.add_argument("--tenant", required=True)
    k.add_argument("--aliases", default="")
    k.add_argument("--budget-usd", dest="budget_usd", type=float, default=None)
    ld = sub.add_parser("ledger")
    ld.add_argument("--db", required=True)
    ld.add_argument("--by", default="tenant", choices=["tenant", "target", "alias", "key_id"])
    a = p.parse_args(argv)
    return {"gateway": _serve_gateway, "fake": _serve_fake, "stack": _stack, "key": _key, "ledger": _ledger}[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
