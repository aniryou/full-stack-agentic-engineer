"""The agent loop keeps agent-core's contracts and adds tiers, turn budgets, idempotency keys and audit
events whose fields are the identity lab's AuditEvent plus the sandbox's own."""
import dataclasses
import hashlib
import json
import sys

import pytest

from sandboxlab.agent import Agent, ScriptedLLM, TurnBudget, call, idempotency_key, run_code_tool, text
from sandboxlab.agent.loop import Tool
from sandboxlab.audit import AuditEvent, AuditLog, args_digest, detect, execution_event, summary
from sandboxlab.process import Budgets, ExecResult, ProcessSandbox

IDENTITY_LAB_FIELDS = ["event_type", "agent", "authority", "user", "tool", "decision", "reasons", "args_hash",
                       "args_redacted", "result_hash", "approver", "provenance", "trace_id", "invocation_id",
                       "session_id", "latency_ms", "ts", "id", "extra"]


def test_audit_event_is_the_identity_lab_shape_plus_sandbox_fields():
    names = [f.name for f in dataclasses.fields(AuditEvent)]
    assert names[: len(IDENTITY_LAB_FIELDS)] == IDENTITY_LAB_FIELDS
    assert names[len(IDENTITY_LAB_FIELDS):] == ["isolation", "budgets_used", "exit_reason", "policy_decision",
                                                  "idempotency_key"]


def test_args_digest_recipe():
    args = {"b": 1, "a": "x"}
    want = hashlib.sha256(b'{"a":"x","b":1}').hexdigest()[:16]
    assert args_digest(args) == want == args_digest({"a": "x", "b": 1})


def test_idempotency_key_recipe():
    k = idempotency_key(3, 2, 1, "run_code", {"code": "print(1)"})
    assert k == f"turn3:step2:call1:run_code:{args_digest({'code': 'print(1)'})}"


class CountingExecutor:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def run(self, code, budgets=None):
        self.calls += 1
        timeout = "while True" in code
        return ExecResult("cpu_limit" if timeout else "ok", -24 if timeout else 0, "" if timeout else "ok\n", "",
                          wall_s=1.0 if timeout else 0.01, cpu_s=1.0 if timeout else 0.01, isolation="fake")


def test_budgets_and_tiers_stop_the_loop_not_the_model():
    ex = CountingExecutor()
    llm = ScriptedLLM([call("run_code", code="while True: pass"), call("run_code", code="while True: pass"),
                       call("run_code", code="while True: pass"), call("delete_everything"),
                       call("admin", x=1), text("done")])
    admin = Tool("admin", "an admin tool", {"x": {"type": "integer"}}, lambda a, c: {"ok": True, "data": 1}, tier="write")
    agent = Agent(llm, [run_code_tool(ex), admin], budget=TurnBudget(max_run_code=2))
    r = agent.run("go")
    results = [json.loads(m["content"]) for m in r.messages if m["role"] == "tool"]
    assert [x.get("error") for x in results] == ["cpu_limit", "cpu_limit", "budget_exceeded", "unknown_tool", "denied"]
    assert ex.calls == 2 and r.text == "done"
    decisions = [e.decision for e in agent.audit.of_type("tool.decision")]
    assert decisions == ["allow", "allow", "deny", "deny", "deny"]
    assert [e.exit_reason for e in agent.audit.of_type("sandbox.execution")] == ["cpu_limit", "cpu_limit"]


def test_redelivered_step_replays_instead_of_rerunning():
    ex = CountingExecutor()
    tool = run_code_tool(ex)
    log = AuditLog()
    from sandboxlab.agent.loop import Context
    ctx = Context("a", "s", "i", "turn1:step1:call0:run_code:abc", log, TurnBudget())
    first, again = tool.fn({"code": "print(1)"}, ctx), tool.fn({"code": "print(1)"}, ctx)
    assert first == again and ex.calls == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux")
def test_execution_events_feed_detection():
    sb = ProcessSandbox(Budgets(cpu_s=1, wall_s=2), netns=False)
    log = AuditLog()
    for code in ["while True: pass"] * 3 + ["import sys; sys.stdout.write('x' * 2**21)"]:
        log.emit(execution_event(sb.run(code), agent="a", session_id="s1", invocation_id="i", code=code))
    log.from_proxy([{"event_type": "egress", "agent": "sandbox", "authority": "own", "decision": "deny",
                     "reasons": ["host 'attacker.net' is not on the egress allowlist"], "extra": {"host": "attacker.net"}}],
                   session_id="s1")
    s = summary(log.events)
    assert s["executions"] == 4 and s["exit_reasons"] == {"cpu_limit": 3, "output_limit": 1}
    rules = {a.rule for a in detect(log.events)}
    assert rules == {"repeated-cpu-kills", "output-flood", "egress-denied"}
    line = json.loads(log.jsonl().splitlines()[0])
    assert line["exit_reason"] == "cpu_limit" and line["budgets_used"]["cpu_s"] >= 0.9
