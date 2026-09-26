"""Audit events carry the right fields, and the agent's run_code fails closed under injection."""
import sys

import pytest

from sandboxcore import (AuditEvent, AuditLog, LoopbackTrap, SandboxAgent, SandboxPolicy,
                         answer, args_digest, code_call, injection_scenarios)
from sandboxcore.agent import ScriptedLLM

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")


def test_audit_event_has_identity_lab_fields_plus_sandbox_fields():
    e = AuditEvent(event_type="sandbox.result", agent="agent:x", exit_reason="ok",
                   policy_decision="allow", budgets_used={"cpu_s": 0.1})
    d = e.to_dict()
    # shared with the identity lab's AuditEvent
    for f in ("event_type", "agent", "authority", "user", "tool", "decision", "reasons",
              "args_hash", "result_hash", "approver", "trace_id", "invocation_id", "ts", "id"):
        assert f in d
    # the fields the sandbox adds
    for f in ("budgets_used", "exit_reason", "exit_reason_source", "policy_decision", "isolation"):
        assert f in d


def test_args_digest_matches_the_identity_lab_recipe():
    import hashlib
    canonical = '{"a":1}'
    assert args_digest({"a": 1}) == hashlib.sha256(canonical.encode()).hexdigest()[:16]


def test_injection_reading_a_secret_finds_nothing():
    pol = SandboxPolicy(egress_allowlist=())
    ag = SandboxAgent(injection_scenarios(pol)["read_secret"], pol)
    res = ag.run("summarise this ticket")     # ticket steered the model into reading env
    ex = res.executions[0]
    assert ex.exit_reason == "ok"
    assert "no token" in ex.stdout            # the secret was never in the sandbox
    assert ag.audit.events[0].event_type == "sandbox.result"


def test_declared_exfiltration_is_denied_before_running():
    # Defence in depth only: this holds because the hijacked model DECLARED attacker.example as egress.
    pol = SandboxPolicy(egress_allowlist=())
    ag = SandboxAgent(injection_scenarios(pol)["exfiltrate_declared"], pol)
    res = ag.run("look this up online")
    ex = res.executions[0]
    assert ex.exit_reason == "denied"          # policy refused; nothing executed
    assert ag.audit.events[0].decision == "deny"


def test_the_declared_scenario_only_ever_reaches_the_loopback_trap():
    # harmless by construction: even a policy that allows attacker.example sends it only to our listener
    pol = SandboxPolicy(egress_allowlist=("attacker.example",))
    with LoopbackTrap() as trap:
        ag = SandboxAgent(injection_scenarios(pol, trap=(trap.host, trap.port))["exfiltrate_declared"], pol)
        ex = ag.run("look this up online").executions[0]
        assert ex.exit_reason != "denied" and trap.hit is True


def test_undeclared_exfiltration_leaks_through_a_process_sandbox():
    # The honest result: the model simply does not declare its egress and opens a raw socket. The process
    # sandbox has no network control (network_blocked=False), so the bytes arrive and the audit says allow.
    pol = SandboxPolicy(egress_allowlist=())
    with LoopbackTrap() as trap:
        ag = SandboxAgent(injection_scenarios(pol, trap=(trap.host, trap.port))["exfiltrate_undeclared"], pol)
        res = ag.run("summarise this page")
        ex = res.executions[0]
        assert ex.exit_reason == "ok" and "sent" in ex.stdout
        assert trap.hit is True                                  # LEAKED: the listener received a connection
    assert ex.isolation["network_blocked"] is False               # and the sandbox said it would not stop it
    assert [e.decision for e in ag.audit.events] == ["allow"]    # nothing in the audit flags it


def test_honest_task_runs_and_answers():
    pol = SandboxPolicy(egress_allowlist=())
    ag = SandboxAgent(injection_scenarios(pol)["honest_task"], pol)
    res = ag.run("what is the sum of 0..99?")
    assert res.text == "The sum is 4950."
    assert res.executions[0].stdout.strip() == "4950"


def test_confirmation_gate_blocks_when_declined():
    pol = SandboxPolicy(egress_allowlist=("api.example",), egress_needs_confirm=True)
    llm = ScriptedLLM([code_call("print('hi')", egress=("api.example",)), answer("done")])
    ag = SandboxAgent(llm, pol, on_confirm=lambda name, args: False)
    res = ag.run("call the API")
    assert res.executions[0].exit_reason == "denied"


def test_audit_counts_are_a_kill_reason_histogram():
    log = AuditLog()
    log.record(AuditEvent(event_type="sandbox.result", agent="a", exit_reason="cpu_time", exit_reason_source="signal"))
    log.record(AuditEvent(event_type="sandbox.result", agent="a", exit_reason="cpu_time", exit_reason_source="signal"))
    log.record(AuditEvent(event_type="sandbox.result", agent="a", exit_reason="memory", exit_reason_source="code"))
    log.record(AuditEvent(event_type="sandbox.decision", agent="a", decision="deny"))
    assert log.counts()["cpu_time"] == 2 and log.counts()["deny"] == 1
    assert "memory" in log.counts() and "memory" not in log.counts(trusted_only=True)   # forgeable: dropped


def test_forged_exit_reason_is_labelled_as_the_codes_own_claim():
    pol = SandboxPolicy()
    ag = SandboxAgent(ScriptedLLM([code_call("import sys; sys.stderr.write('MemoryError\\n'); sys.exit(1)"),
                                   answer("done")]), pol)
    ag.run("x")
    ev = ag.audit.events[0]
    assert ev.exit_reason == "memory" and ev.exit_reason_source == "code"
    assert "max_rss_mb" in ev.budgets_used
