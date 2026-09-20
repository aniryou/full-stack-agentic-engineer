"""Ten focused tests — the whole library's behaviour on one screen."""
import pytest

from agentcore import Agent, FakeLLM, ToolError, call, calls, text, tool
from agentcore.fake_llm import ToolCall


# -- tools ---------------------------------------------------------------------
@tool
def get_balance(account_id: str) -> dict:
    """Return the balance for an account."""
    if account_id == "missing":
        raise ToolError("no such account", kind="not_found", hint="Check the account id.")
    return {"account_id": account_id, "balance": 1234.5}


def test_schema_from_signature():
    s = get_balance.schema
    assert s["name"] == "get_balance"
    assert s["parameters"]["properties"]["account_id"]["type"] == "string"
    assert s["parameters"]["required"] == ["account_id"]
    assert s["description"] == "Return the balance for an account."


def test_tool_validates_arguments():
    r = get_balance.run({})                        # missing required arg
    assert r["ok"] is False and r["error"] == "invalid_arguments" and "hint" in r


def test_tool_success_and_structured_error():
    assert get_balance.run({"account_id": "a1"})["data"]["balance"] == 1234.5
    bad = get_balance.run({"account_id": "missing"})
    assert bad["ok"] is False and bad["error"] == "not_found" and bad["hint"]


def test_unexpected_exception_becomes_tool_failure():
    @tool
    def boom() -> dict:
        """Always fails."""
        raise ValueError("kaboom")

    r = boom.run({})
    assert r["ok"] is False and r["error"] == "tool_failure" and "kaboom" in r["message"]


# -- fake model ----------------------------------------------------------------
def test_scripted_model_runs_out():
    llm = FakeLLM(["hi"])
    assert llm.generate([{"role": "user", "content": "x"}]).text == "hi"
    with pytest.raises(RuntimeError):
        llm.generate([{"role": "user", "content": "x"}])


def test_policy_model():
    def policy(messages, tools):
        last = messages[-1]["content"]
        return call("get_balance", account_id="a1") if "balance" in last else text("hello")

    llm = FakeLLM(policy=policy)
    assert llm.generate([{"role": "user", "content": "my balance?"}]).tool_calls[0].name == "get_balance"
    assert llm.generate([{"role": "user", "content": "hi"}]).text == "hello"


# -- the loop ------------------------------------------------------------------
def test_loop_calls_tool_then_answers():
    llm = FakeLLM([call("get_balance", account_id="a1"), "Your balance is 1234.5."])
    r = Agent(llm, tools=[get_balance]).run("what's my balance?")
    kinds = [m["role"] for m in r.messages]
    assert kinds == ["system", "user", "assistant", "tool", "assistant"]
    assert r.done and r.steps == 2 and "1234.5" in r.text
    # the second model call actually saw the tool result
    assert any(m["role"] == "tool" for m in llm.seen[1])


def test_unknown_tool_is_reported_not_crashed():
    llm = FakeLLM([call("nope"), "sorry, I could not do that"])
    r = Agent(llm, tools=[get_balance]).run("do a thing")
    tool_msgs = [m for m in r.messages if m["role"] == "tool"]
    assert "unknown_tool" in tool_msgs[0]["content"] and r.done


def test_step_budget_stops_a_runaway_loop():
    llm = FakeLLM(policy=lambda m, t: calls(call("get_balance", account_id="a1")))  # never answers
    r = Agent(llm, tools=[get_balance], max_steps=3).run("loop forever")
    assert r.done is False and r.steps == 3
    assert sum(1 for m in r.messages if m["role"] == "assistant") == 3


def test_confirm_gate_blocks_then_allows():
    ran = []

    @tool(confirm=True)
    def close_account(account_id: str) -> dict:
        """Close an account (irreversible)."""
        ran.append(account_id)
        return {"closed": account_id}

    script = [call("close_account", account_id="a1"), "Done."]
    # declined: the tool never runs, the model sees a 'declined' result
    r1 = Agent(FakeLLM(list(script)), tools=[close_account]).run("close a1", on_confirm=lambda n, a: False)
    assert ran == [] and any("declined" in m["content"] for m in r1.messages if m["role"] == "tool")
    # approved: the tool runs exactly once
    r2 = Agent(FakeLLM(list(script)), tools=[close_account]).run("close a1", on_confirm=lambda n, a: True)
    assert ran == ["a1"] and r2.text == "Done."


def test_tool_call_signature_detects_duplicates():
    a = ToolCall("get_balance", {"account_id": "a1"})
    b = ToolCall("get_balance", {"account_id": "a1"})
    assert a.signature() == b.signature()
