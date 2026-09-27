"""The ``memlab`` command line: ``python -m memlab <command>``.

    env                      what this machine can run (tier, SQLite features, Docker, URLs)
    demo                     write, search and forget in a SQLite file, then look for the bytes on disk
    serve                    the memory service (aiohttp); the token key from MEMLAB_TOKEN_KEY
    token                    mint a stand-in bearer token for a tenant and user
    fake                     the fake OpenAI-compatible server (chat + embeddings, vLLM-named metrics)
    cachebench               prefix-cache hits per memory layout against a server (fake or real)
    harness                  the planted-facts eval across memory modes
    consolidate              run the consolidation job over every partition with episodes in the window
    gcp-commands             print the Cloud Run job + Cloud Scheduler commands (T3; nothing is run)
    residue                  count occurrences of a string in a SQLite file and its WAL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _key() -> bytes:
    key = os.environ.get("MEMLAB_TOKEN_KEY")
    if not key:
        print("MEMLAB_TOKEN_KEY is unset: using a throwaway local key (tokens die with this process)", file=sys.stderr)
        key = os.urandom(16).hex()
        os.environ["MEMLAB_TOKEN_KEY"] = key
    return key.encode()


def cmd_env(a) -> int:
    from .env import banner, describe
    print(banner())
    print(json.dumps(describe(), indent=2))
    return 0


def cmd_demo(a) -> int:
    from .deletion import residue
    from .memory import LocalMemory
    from .store.sqlite import SQLiteMemoryStore
    path = Path(a.db or Path(tempfile.mkdtemp(prefix="memlab-")) / "demo.db")
    store = SQLiteMemoryStore(path, fts_secure_delete=not a.no_secure_delete)
    mem = LocalMemory(store, "acme", "u1")
    mem.remember("Home address: the user lives at 12 Rua das Flores.", slot="address", value="12 Rua das Flores",
                 importance=9)
    mem.remember("Pet: the user has a dog named Rex.", slot="pet", value="dog named Rex")
    for item in mem.recall("what is my home address?", k=2):
        print(f"recall  {item['score']:.4f}  {item['text']}")
    print("on disk before forget:", residue(path, ["Flores"]))
    rep = mem.forget("address", mode=a.mode, needles=["Flores"])
    print(rep.table())
    print("database:", path)
    return 0 if rep.clean or a.mode == "logical" else 1


def cmd_serve(a) -> int:
    from .audit import AuditLog
    from .service import MemoryService, TokenVerifier
    from .store.sqlite import SQLiteMemoryStore
    store = SQLiteMemoryStore(a.db)
    svc = MemoryService(store, TokenVerifier(_key()), audit=AuditLog(a.audit), host=a.host, port=a.port)
    url = svc.start()
    print(f"memory service on {url} (db {a.db}, audit {a.audit}); Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        svc.stop()
    return 0


def cmd_token(a) -> int:
    from .service import SCOPES, TokenVerifier
    scopes = list(SCOPES) + (["memory.review"] if a.reviewer else [])
    print(TokenVerifier(_key()).mint(a.tenant, a.user, a.agent, scopes, a.ttl))
    return 0


def cmd_fake(a) -> int:
    from .fakeserver import FakeLLMServer
    srv = FakeLLMServer(host=a.host, port=a.port, enable_prompt_tokens_details=a.prompt_tokens_details)
    url = srv.start()
    print(f"fake OpenAI-compatible server (SIMULATED) on {url}; prompt_tokens_details={a.prompt_tokens_details}",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()
    return 0


def cmd_cachebench(a) -> int:
    from . import cachebench as cb
    from .fakeserver import FakeLLMServer
    if a.url:
        runs = {layout: cb.run_layout(a.url, layout, turns=a.turns, api_key=os.environ.get("MEMLAB_API_KEY"))
                for layout in cb.LAYOUTS}
    else:
        with FakeLLMServer(enable_prompt_tokens_details=True) as url:
            runs = {layout: cb.run_layout(url, layout, turns=a.turns) for layout in cb.LAYOUTS}
    for r in runs.values():
        print(r.table(), "\n")
    print(cb.summary(runs))
    return 0


def cmd_harness(a) -> int:
    from . import harness as H
    ds = H.generate(a.seed, users=a.users)
    results = {m: H.run(ds, m, budget_tokens=a.budget) for m in a.modes.split(",")}
    print(H.summary(results))
    if a.out:
        from .report import harness_markdown, save
        print("saved", save("harness", {m: r.rows for m, r in results.items()},
                            harness_markdown(results, "computed: scripted model, hashing embedder"), a.out,
                            "computed (T0, deterministic)"))
    return 0


def cmd_consolidate(a) -> int:
    from .consolidate import ConsolidationJob, due_partitions, shard, window
    if a.pg_dsn or os.environ.get("MEMLAB_PG_DSN"):
        from .store.pgvector import PgVectorStore
        store = PgVectorStore(a.pg_dsn or os.environ["MEMLAB_PG_DSN"])
        store.init()
    else:
        from .store.sqlite import SQLiteMemoryStore
        store = SQLiteMemoryStore(a.db)
    end = time.time() if a.end is None else a.end
    start, end = window(end, a.window_days)
    index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
    parts = shard(due_partitions(store, start, end), index, count)
    print(f"task {index}/{count}: {len(parts)} partitions with episodes in the window")
    for tenant, user in parts:
        print(ConsolidationJob(store, worker=f"task-{index}-{os.getpid()}").run(tenant, user, start, end).line())
    return 0


def cmd_gcp(a) -> int:
    from .consolidate import cleanup_commands, gcp_commands
    print("# T3, printed only (verify each flag against the current gcloud reference before running)")
    for c in gcp_commands(a.project, a.region, tasks=a.tasks):
        print(c)
    print("# cleanup")
    for c in cleanup_commands(a.project, a.region):
        print(c)
    return 0


def cmd_residue(a) -> int:
    from .deletion import residue
    print(json.dumps(residue(a.db, a.needle), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    s = sub.add_parser("demo")
    s.add_argument("--db")
    s.add_argument("--mode", choices=("purge", "logical"), default="purge")
    s.add_argument("--no-secure-delete", action="store_true", help="leave FTS5 secure-delete off (optimize instead)")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve")
    s.add_argument("--db", default="memory.db")
    s.add_argument("--audit", default="audit.jsonl")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("token")
    s.add_argument("--tenant", required=True)
    s.add_argument("--user", required=True)
    s.add_argument("--agent", default="agent:memlab")
    s.add_argument("--ttl", type=float, default=900)
    s.add_argument("--reviewer", action="store_true")
    s.set_defaults(fn=cmd_token)
    s = sub.add_parser("fake")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--prompt-tokens-details", action="store_true")
    s.set_defaults(fn=cmd_fake)
    s = sub.add_parser("cachebench")
    s.add_argument("--url", help="a server to measure (default: an in-process fake server)")
    s.add_argument("--turns", type=int, default=8)
    s.set_defaults(fn=cmd_cachebench)
    s = sub.add_parser("harness")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--users", type=int, default=6)
    s.add_argument("--budget", type=int, default=128)
    s.add_argument("--modes", default="none,full_history,tools,implicit,pinned")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_harness)
    s = sub.add_parser("consolidate")
    s.add_argument("--db", default="memory.db")
    s.add_argument("--pg-dsn")
    s.add_argument("--window-days", type=float, default=float(os.environ.get("MEMLAB_WINDOW_DAYS", "7")))
    s.add_argument("--end", type=float, help="window end, unix seconds (default: now)")
    s.set_defaults(fn=cmd_consolidate)
    s = sub.add_parser("gcp-commands")
    s.add_argument("--project", default="PROJECT_ID")
    s.add_argument("--region", default="us-central1")
    s.add_argument("--tasks", type=int, default=1)
    s.set_defaults(fn=cmd_gcp)
    s = sub.add_parser("residue")
    s.add_argument("--db", required=True)
    s.add_argument("needle", nargs="+")
    s.set_defaults(fn=cmd_residue)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
