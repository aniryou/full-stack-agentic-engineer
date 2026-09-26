"""``python3 -m sandboxlab <command>`` — the lab from a shell.

    env                         which isolation levels this machine can run
    probes [--level L] [--quick] run the attack probes through a level (none, process, process+netns,
                                docker:default, docker:runc, docker:runsc); Docker levels without a
                                daemon print the command and the bundled sample verdicts (illustrative)
    docker-cmd [--runtime runsc] the hardened docker run command, and what each flag stops
    render [--check]            write (or check) the generated kind/GKE manifests and seccomp profile
    admit FILE [--target T]     predict what admission says about each object in a YAML file
    bench [--n N]               start-up latency by level (measured here; samples for the rest)
    pool --rate R --exec S --cold S   warm-pool mean occupancy (Little's law) and size (Erlang C)
    gke-review                  the GKE Sandbox Terraform checklist, read offline
    report [--out DIR]          probes + bench into JSON and Markdown
"""
from __future__ import annotations

import argparse
import sys


def _executor(level: str, host):
    from .process import Budgets, ProcessSandbox
    from .probes import unsandboxed_for
    b = Budgets(cpu_s=1, wall_s=3)
    if level == "none":
        return unsandboxed_for(host)
    if level == "process":
        return ProcessSandbox(b, netns=False)
    if level == "process+netns":
        return ProcessSandbox(b, netns=True)
    raise ValueError(level)


def cmd_probes(a) -> int:
    from . import docker as D
    from .probes import QUICK, load_sample_runs, run_suite, sample_label, standin_host, verdict_table
    names = QUICK if a.quick else None
    if a.level.startswith("docker:"):
        if not D.docker_available() or (a.level == "docker:runsc" and not __import__("sandboxlab.env").env.has_runsc()):
            spec = D.DockerSandbox.naive() if a.level == "docker:default" else D.DockerSandbox.hardened(
                "runsc" if a.level == "docker:runsc" else None)
            print(f"# {D.env_summary()}\n# the command:\n{spec.shell()}\n")
            runs = load_sample_runs()
            print(f"[{sample_label()}]")
            print(verdict_table({a.level: runs[a.level]}))
            return 0
        spec = D.DockerSandbox.naive() if a.level == "docker:default" else D.DockerSandbox.hardened(
            "runsc" if a.level == "docker:runsc" else None)
        D.ensure_seccomp_file()
        with standin_host(bind="0.0.0.0", listener_hosts=D.listener_hosts_for(spec)) as host:
            res = run_suite(D.with_host_gateway(spec), host, names)
        print(verdict_table({a.level: res}))
        return 0
    with standin_host() as host:
        res = run_suite(_executor(a.level, host), host, names)
    print(verdict_table({a.level: res}))
    for r in res:
        print(f"  {r.probe:<16} {r.verdict:<10} {r.evidence}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sandboxlab", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    p = sub.add_parser("probes")
    p.add_argument("--level", default="process+netns")
    p.add_argument("--quick", action="store_true")
    p = sub.add_parser("docker-cmd")
    p.add_argument("--runtime", default=None)
    p = sub.add_parser("render")
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("admit")
    p.add_argument("file")
    p.add_argument("--target", default="kind", choices=["kind", "gke"])
    p = sub.add_parser("bench")
    p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("pool")
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--exec", dest="exec_s", type=float, required=True)
    p.add_argument("--cold", type=float, required=True)
    sub.add_parser("gke-review")
    p = sub.add_parser("report")
    p.add_argument("--out", default="reports")
    a = ap.parse_args(argv)

    if a.cmd == "env":
        from . import env
        caps = env.capabilities()
        print(caps.describe())
        print("measurable here:", ", ".join(caps.levels()))
    elif a.cmd == "probes":
        return cmd_probes(a)
    elif a.cmd == "docker-cmd":
        from . import docker as D
        print(D.DockerSandbox.hardened(a.runtime).shell())
        print()
        print(D.explain())
    elif a.cmd == "render":
        from .k8s import render
        if a.check:
            bad = render.stale()
            print("\n".join(bad) if bad else "✅ generated deploy files are up to date")
            return 1 if bad else 0
        for path in render.write():
            print("wrote", path.relative_to(render.LAB_ROOT))
    elif a.cmd == "admit":
        from .k8s import admission, manifests as m, policy as P
        pol = P.SandboxPolicy.kind() if a.target == "kind" else P.SandboxPolicy.gke()
        rcs = {pol.runtime_class: P.runtime_class(pol)}
        for o in m.load_all(a.file):
            if o.get("kind") not in ("Pod", "Job"):
                print(f"{o.get('kind')}/{o['metadata']['name']}: not a Pod or Job (skipped)")
                continue
            r = admission.admit(o, runtime_classes=rcs, allowed_runtime_classes=pol.allowed_runtime_classes,
                                node_handlers={pol.runtime_handler})
            print(f"{o['kind']}/{o['metadata']['name']}: {r.report()}")
    elif a.cmd == "bench":
        from . import bench
        for row in bench.ladder(bench.measure(n=a.n)):
            print(row.row())
    elif a.cmd == "pool":
        from . import bench
        p = bench.littles_law_pool(a.rate, a.exec_s, a.cold)
        print(f"busy {p['busy']:.1f} + warming {p['warming']:.1f} = {p['total']:.1f} slots on average "
              "(Little's law: the floor, not the size)")
        hold = a.exec_s + a.cold
        c = bench.replace_after_use_slots(a.rate, a.exec_s, a.cold, 0.2)
        print(f"replace-after-use: {c} slots keep P(wait for a warm sandbox) <= 0.2 "
              f"(Erlang C on a = {a.rate * hold:.1f}; E[Wq] {bench.mean_wait_s(a.rate, hold, c) * 1000:.0f} ms)")
        r = bench.servers_for(a.rate, a.exec_s, max_p_wait=0.2)
        print(f"reuse pool (state carries between executions): {r} slots for the same target (a = {a.rate * a.exec_s:.1f})")
    elif a.cmd == "gke-review":
        from . import gke
        for c in gke.review():
            print(f"{'✅' if c.ok else 'MISSING'}  {c.name:<40} {c.evidence}")
    elif a.cmd == "report":
        from . import bench, report
        from .probes import QUICK, run_suite, standin_host
        with standin_host() as host:
            v = {lv: run_suite(_executor(lv, host), host, QUICK) for lv in ("none", "process")}
        j, md = report.write(report.build(v, bench.ladder(bench.measure(n=10))), a.out)
        print("wrote", j, "and", md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
