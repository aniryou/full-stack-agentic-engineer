"""seccomp.py — a tighter syscall filter for code sandboxes, derived from Docker's default.

One idea: Docker's default seccomp profile is an *allowlist* (``defaultAction: SCMP_ACT_ERRNO``,
EPERM for anything not listed) written to keep almost every containerised app working; it blocks
~44 of 300+ syscalls (namespaces, mount, kexec, bpf, keyrings, io_uring, ...). A sandbox for
model-generated Python can afford to be stricter, and each removal below closes a specific
route that the other controls leave open (PRIMER §2 "The isolation ladder", containers):

* ``memfd_create`` / ``memfd_secret`` / ``execveat`` — with a read-only root and a ``noexec``
  workspace, an anonymous memory file plus ``execveat`` is how code still runs a binary it wrote.
* ``ptrace``, ``process_vm_readv`` / ``process_vm_writev`` — reading or steering other
  processes of the same UID (Docker allows them on kernels >= 4.8).
* ``socket`` families beyond ``AF_UNIX``, ``AF_INET``, ``AF_INET6`` — no netlink (routing and
  interface tables), no ``AF_PACKET`` raw frames, no vsock to the host.
* ``socketcall`` — the 32-bit x86 multiplexer: it takes the family inside a pointer that seccomp
  cannot read, so as long as it is allowed a 32-bit binary skips the family filter above.

The base profile is ``moby/profiles`` ``seccomp/default.json`` (Apache-2.0), bundled as
``data/moby-seccomp-default.json`` and fetched on 2026-09-26; ``render.py`` writes the result to
``deploy/docker/seccomp-sandbox.json``. Kubernetes pods keep ``seccompProfile: RuntimeDefault``
(the runtime's default, equivalent in spirit); a ``Localhost`` profile needs the file on every
node, which is why the lab does not ship one for kind or GKE.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BASE = Path(__file__).parent / "data" / "moby-seccomp-default.json"
REMOVED = ("memfd_create", "memfd_secret", "execveat", "socketcall")
REMOVED_RULE = {"process_vm_readv", "process_vm_writev", "ptrace"}   # Docker's minKernel 4.8 rule
AF = {"AF_UNIX": 1, "AF_INET": 2, "AF_INET6": 10, "AF_NETLINK": 16, "AF_PACKET": 17, "AF_ALG": 38,
      "AF_VSOCK": 40}
ALLOWED_FAMILIES = ("AF_UNIX", "AF_INET", "AF_INET6")


def load_base() -> dict:
    return json.loads(BASE.read_text())


def _is_socket_rule(rule: dict) -> bool:
    return rule.get("names") == ["socket"]


def sandbox_profile(base: dict | None = None, families: tuple[str, ...] = ALLOWED_FAMILIES) -> dict:
    """Docker's default profile with the removals above; everything else is kept as-is."""
    prof = copy.deepcopy(base or load_base())
    rules = []
    for rule in prof["syscalls"]:
        if set(rule.get("names", [])) == REMOVED_RULE:
            continue
        if _is_socket_rule(rule):
            continue                       # replaced below by one rule per allowed family
        if rule.get("action") == "SCMP_ACT_ALLOW" and not rule.get("args") and not rule.get("includes") \
                and not rule.get("excludes"):
            rule = {**rule, "names": [n for n in rule["names"] if n not in REMOVED]}
        rules.append(rule)
    for fam in families:
        rules.append({"names": ["socket"], "action": "SCMP_ACT_ALLOW",
                      "args": [{"index": 0, "value": AF[fam], "op": "SCMP_CMP_EQ"}]})
    prof["syscalls"] = rules
    return prof


def allowed_unconditionally(profile: dict) -> set[str]:
    """Syscalls allowed with no argument, capability, kernel or architecture condition."""
    out: set[str] = set()
    for r in profile["syscalls"]:
        if r.get("action") == "SCMP_ACT_ALLOW" and not r.get("args") and not r.get("includes") and not r.get("excludes"):
            out |= set(r["names"])
    return out


def socket_families(profile: dict) -> set[int]:
    """Address families ``socket(2)`` may create under this profile (the lab's rules and Docker's)."""
    fams: set[int] = set()
    for r in profile["syscalls"]:
        if "socket" not in r.get("names", []) or r.get("action") != "SCMP_ACT_ALLOW":
            continue
        if not r.get("args"):
            return set(range(64))
        for a in r["args"]:
            if a["index"] != 0:
                continue
            if a["op"] == "SCMP_CMP_EQ":
                fams.add(a["value"])
            elif a["op"] == "SCMP_CMP_LT":
                fams |= set(range(a["value"]))
    return fams


def socket_allowed(profile: dict, family: str | int) -> bool:
    fam = AF[family] if isinstance(family, str) else family
    return fam in socket_families(profile)


def diff(base: dict, derived: dict) -> list[str]:
    """What the derived profile takes away, in words."""
    lines = []
    gone = sorted(allowed_unconditionally(base) - allowed_unconditionally(derived))
    lines.append(f"no longer allowed: {', '.join(gone)}")
    ptrace_rule = any(set(r.get("names", [])) == REMOVED_RULE for r in base["syscalls"])
    if ptrace_rule:
        lines.append("dropped Docker's kernel>=4.8 rule for ptrace, process_vm_readv, process_vm_writev")
    lost = sorted(socket_families(base) - socket_families(derived))
    names = {v: k for k, v in AF.items()}
    named = [names[f] for f in lost if f in names]
    lines.append("socket families removed: " + ", ".join(named) + f" and {len(lost) - len(named)} others")
    lines.append(f"socket families kept: {', '.join(names[f] for f in sorted(socket_families(derived)))}")
    return lines


def to_json(profile: dict) -> str:
    return json.dumps(profile, indent=2) + "\n"
