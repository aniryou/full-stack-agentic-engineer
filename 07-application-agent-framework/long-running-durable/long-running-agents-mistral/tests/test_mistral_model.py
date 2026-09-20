"""The Mistral adapter, offline: a fake client that speaks the SDK's response shape."""

import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from durable import Agent, Crash, FakeClock, PaymentAPI, Queue, Store
from mistral_model import TOOL_SPECS, MistralModel, function_tools, to_messages, tool_call_id


class FakeMistral:
    """Returns scripted chat completions; records the request so tests can inspect the prompt."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = self

    def complete(self, **kw):
        self.requests.append(kw)
        item = self.script.pop(0)
        if "tool" in item:
            msg = NS(content="", tool_calls=[NS(id="abc123def", type="function",
                                                  function=NS(name=item["tool"], arguments=json.dumps(item["args"])))])
        else:
            msg = NS(content=item["final"], tool_calls=None)
        return NS(choices=[NS(message=msg)])


def test_tool_call_ids_are_9_alphanumeric_and_stable():
    a, b = tool_call_id("run_ab12cd:3"), tool_call_id("run_ab12cd:3")
    assert a == b and len(a) == 9 and a.isalnum()
    assert tool_call_id("run_ab12cd:5") != a


def test_journal_becomes_tool_calls_and_results():
    journal = [{"type": "decision", "tool": "charge", "args": {"amount": 42}},
               {"type": "intent", "tool": "charge", "args": {"amount": 42}, "key": "run_x:1", "done": True,
                "approved": True, "result": {"charge_id": "run_x:1", "amount": 42}}]
    m = to_messages("pay invoice 42", journal)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool"]
    assert m[2]["tool_calls"][0]["id"] == m[3]["tool_call_id"] == tool_call_id("run_x:1")
    assert json.loads(m[2]["tool_calls"][0]["function"]["arguments"]) == {"amount": 42}
    assert json.loads(m[3]["content"])["charge_id"] == "run_x:1"


def test_function_tools_shape_hides_the_key_parameter():
    t = function_tools(TOOL_SPECS)
    assert t[0]["type"] == "function" and t[0]["function"]["name"] == "charge"
    assert "key" not in t[0]["function"]["parameters"]["properties"]


def test_decide_parses_tool_call_then_final():
    fake = FakeMistral([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
    model = MistralModel(TOOL_SPECS, model="mistral-small-latest", client=fake)
    assert model.decide("pay 42", []) == {"tool": "charge", "args": {"amount": 42}}
    assert model.decide("pay 42", []) == {"final": "Charged 42."}
    assert fake.requests[0]["model"] == "mistral-small-latest" and fake.requests[0]["tool_choice"] == "auto"


def test_adapter_inside_the_durable_loop_survives_a_crash_without_reasking():
    fake = FakeMistral([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
    path = "/tmp/lra_mistral_runs.json"
    if os.path.exists(path):
        os.remove(path)
    pay, clock = PaymentAPI(), FakeClock()
    agent = Agent(Store(path), Queue(), MistralModel(TOOL_SPECS, model="x", client=fake), {"charge": pay}, clock=clock)
    run = agent.start("pay invoice 42")
    agent.crash_at.add("after_side_effect")
    try:
        agent.queue.deliver_one(agent.handle)
    except Crash:
        pass
    clock.advance(61)
    agent.queue.drain(agent.handle)
    final = agent.store.get(run.id)
    assert final.status == "DONE" and list(pay.charges.values()) == [42]
    assert len(fake.requests) == 2                          # the retry re-executed the intent, never re-asked
    # the second prompt carried the recorded tool result, not the model's memory
    assert fake.requests[1]["messages"][-1]["role"] == "tool"
    assert json.loads(fake.requests[1]["messages"][-1]["content"])["amount"] == 42


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
