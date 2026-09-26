"""Command line: ``python -m k8sgpu <command>`` (or ``k8sgpu`` once installed).

  lint FILE... [--machine g2-standard-48] [--load-seconds 300]
  pending --list | --fixture NAME | --live POD -n NAMESPACE | POD.json [--events EVENTS.json]
  capacity --machine a3-highgpu-8g --nodes 8 --hours 10 [--checkpoint-every 1] [--deadline 48] [--serving]
  kind status | predict S [--step N] | run S [--dry-run] [--no-reset] | observe S [--step N] | reset
  render [--check]
  gke plan [--tfvars FILE] | gke gcloud [--tfvars FILE]
"""
from __future__ import annotations

import argparse
import json
import sys


def _lint(a) -> int:
    from . import lint
    findings = []
    for f in a.files:
        findings += lint.lint_file(f, machine=a.machine, expected_load_s=a.load_seconds)
    print(lint.format_findings(findings))
    return 1 if lint.errors(findings) else 0


def _pending(a) -> int:
    from . import pending
    if a.list:
        for n in pending.fixture_names():
            print(f"{n:28} {pending.load_fixture(n)['description']}")
        return 0
    if a.fixture:
        bundle = pending.load_fixture(a.fixture)
        print(pending.fixture_label(bundle))
        print(pending.diagnose(bundle))
        return 0
    if a.live:
        from .kindlab import KubectlMissing
        try:
            print(pending.diagnose_live(a.live, a.namespace))
        except (KubectlMissing, RuntimeError) as e:
            print(f"cannot read the live pod: {e}", file=sys.stderr)
            return 2
        return 0
    if not a.pod:
        print("give --fixture, --live or a pod JSON file", file=sys.stderr)
        return 2
    bundle = {"pod": json.load(open(a.pod))}
    if a.events:
        ev = json.load(open(a.events))
        bundle["events"] = ev.get("items", ev) if isinstance(ev, dict) else ev
    print(pending.diagnose(bundle))
    return 0


def _capacity(a) -> int:
    from . import capacity
    need = capacity.Need("cli", a.machine, a.nodes, a.hours, checkpoint_every_h=a.checkpoint_every,
                         deadline_h=a.deadline, serving=a.serving)
    print(capacity.format_evaluations(capacity.evaluate(need)))
    return 0


def _kind(a) -> int:
    from . import kindlab, kindsim
    if a.action == "status":
        ok, msg = kindlab.cluster_status()
        print(("ready: " if ok else "not ready: ") + msg)
        return 0 if ok else 1
    if a.action == "predict":
        from .scenarios import scenario
        sc = scenario(a.scenario)
        steps = kindsim.predict_all_steps(a.scenario)[: a.step or len(sc.steps)]
        print(f"== {sc.id}: {sc.title} - predicted by k8sgpu.kindsim (simulated, not observed)")
        for i, (st, out) in enumerate(zip(sc.steps, steps), start=1):
            print(f"\n-- after step {i}/{len(sc.steps)}: {st.action} {st.file or st.target}"
                  + (f" - {st.note}" if st.note else ""))
            print(kindsim.describe(out))
        return 0
    if a.action in ("reset", "observe") and not a.dry_run:
        ok, msg = kindlab.cluster_status()
        if not ok:
            print(f"no usable cluster ({msg}); `k8sgpu kind predict {a.scenario}` is the T0 path", file=sys.stderr)
            return 2
    if a.action == "reset":
        kindlab.reset(kindlab.Kubectl(dry_run=a.dry_run))
        return 0
    if a.action == "observe":
        from .scenarios import EXPECTED, scenario
        sc = scenario(a.scenario)
        k = kindlab.Kubectl(echo=lambda s: None)
        obs = kindlab.observe(k, sc)
        print(kindsim.describe(obs))
        step = a.step or len(sc.steps)
        diffs = kindlab.compare(obs, kindsim.predict(a.scenario, upto=step,
                                                     nodes=kindsim.nodes_from_k8s(k.get_json("nodes")["items"])))
        print("\n" + ("matches the prediction" if not diffs else "\n".join(diffs)))
        if step in EXPECTED.get(sc.id, {}):
            print("answer key: " + ("OK" if not kindsim.check_expectation(obs, EXPECTED[sc.id][step]) else "differs"))
        return 0 if not diffs else 1
    if a.action == "run":
        ok, msg = kindlab.cluster_status()
        dry = a.dry_run or not ok
        if not ok and not a.dry_run:
            print(f"(no usable cluster: {msg}) - dry run: commands and predictions only\n")
        report = kindlab.run_scenario(a.scenario, dry_run=dry, do_reset=not a.no_reset)
        if dry:
            return 0
        bad = [s for s in report["steps"] if s.get("differences")]
        return 1 if bad else 0
    return 2


def _render(a) -> int:
    from . import render
    if a.check:
        stale = render.stale()
        print("\n".join(f"stale: {s}" for s in stale) or "all generated manifests up to date")
        return 1 if stale else 0
    for p in render.write():
        print("wrote", p)
    return 0


def _gke(a) -> int:
    from . import gke
    v = gke.effective_vars(gke.read_tfvars(a.tfvars) if a.tfvars else None)
    if a.action == "plan":
        print(gke.plan_summary(v))
        print("\nresources:")
        for f, t, n in gke.tf_resources():
            print(f"  {f:18} {t}.{n}")
    else:
        print("\n\n".join(gke.gcloud_equivalents(v)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k8sgpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lint")
    s.add_argument("files", nargs="+")
    s.add_argument("--machine")
    s.add_argument("--load-seconds", type=float)
    s.set_defaults(fn=_lint)
    s = sub.add_parser("pending")
    s.add_argument("pod", nargs="?")
    s.add_argument("--events")
    s.add_argument("--fixture")
    s.add_argument("--live")
    s.add_argument("-n", "--namespace", default="default")
    s.add_argument("--list", action="store_true")
    s.set_defaults(fn=_pending)
    s = sub.add_parser("capacity")
    s.add_argument("--machine", default="a3-highgpu-8g")
    s.add_argument("--nodes", type=int, default=1)
    s.add_argument("--hours", type=float, default=10)
    s.add_argument("--checkpoint-every", type=float, default=1.0)
    s.add_argument("--deadline", type=float)
    s.add_argument("--serving", action="store_true")
    s.set_defaults(fn=_capacity)
    s = sub.add_parser("kind")
    s.add_argument("action", choices=["status", "predict", "run", "observe", "reset"])
    s.add_argument("scenario", nargs="?", default="s1")
    s.add_argument("--step", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-reset", action="store_true")
    s.set_defaults(fn=_kind)
    s = sub.add_parser("render")
    s.add_argument("--check", action="store_true")
    s.set_defaults(fn=_render)
    s = sub.add_parser("gke")
    s.add_argument("action", choices=["plan", "gcloud"])
    s.add_argument("--tfvars")
    s.set_defaults(fn=_gke)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
