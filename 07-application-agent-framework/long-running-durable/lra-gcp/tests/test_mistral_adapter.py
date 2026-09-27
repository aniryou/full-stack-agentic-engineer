"""The Mistral provider path, offline: a fake client that speaks the SDK's response shape. No key, no network,
and no ``mistralai`` install needed (the adapter imports the SDK only when it has to build a real client)."""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from lra import Engine, Event, RunStatus, SimulatedCrash, Workflow, Done
from lra.adapters.memory import FakeClock, InMemoryEventBus, InMemoryStateStore, InMemoryTaskQueue, LocalRunner
from lra.adapters.mistral import TOOL_SPECS, MistralDecider, MistralLLM, function_tools, to_messages, tool_call_id
from lra.examples.tool_agent import PaymentAPI, make_tool_agent


class FakeMistral:
    """Returns scripted chat completions; records each request so tests can inspect the prompt."""

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
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=1000, completion_tokens=100))


def test_tool_call_ids_are_9_alphanumeric_and_stable():
    a, b = tool_call_id("run_ab12cd:3"), tool_call_id("run_ab12cd:3")
    assert a == b and len(a) == 9 and a.isalnum()
    assert tool_call_id("run_ab12cd:5") != a


def test_journal_becomes_tool_calls_and_results():
    journal = [{"type": "decision", "tool": "charge", "args": {"amount": 42}},
               {"type": "intent", "tool": "charge", "args": {"amount": 42}, "key": "run_x:1", "done": True,
                "result": {"charge_id": "run_x:1", "amount": 42}}]
    m = to_messages("pay invoice 42", journal)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool"]
    assert m[2]["tool_calls"][0]["id"] == m[3]["tool_call_id"] == tool_call_id("run_x:1")
    assert json.loads(m[2]["tool_calls"][0]["function"]["arguments"]) == {"amount": 42}
    assert json.loads(m[3]["content"])["charge_id"] == "run_x:1"


def test_a_pending_intent_is_reported_as_pending():
    journal = [{"type": "decision", "tool": "charge", "args": {"amount": 1}},
               {"type": "intent", "tool": "charge", "args": {"amount": 1}, "key": "r:1", "done": False}]
    assert json.loads(to_messages("g", journal)[-1]["content"]) == {"pending": True}


def test_function_tools_shape_hides_the_key_parameter():
    t = function_tools(TOOL_SPECS)
    assert t[0]["type"] == "function" and t[0]["function"]["name"] == "charge"
    assert "key" not in t[0]["function"]["parameters"]["properties"]


def test_decide_parses_tool_call_then_final():
    fake = FakeMistral([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
    decider = MistralDecider(TOOL_SPECS, model="mistral-small-latest", client=fake)
    assert decider.decide("pay 42", []) == {"tool": "charge", "args": {"amount": 42}}
    assert decider.decide("pay 42", []) == {"final": "Charged 42."}
    assert fake.requests[0]["model"] == "mistral-small-latest" and fake.requests[0]["tool_choice"] == "auto"


def test_mistral_llm_implements_the_engine_port_and_charges_the_budget():
    fake = FakeMistral([{"final": "billing"}])
    llm = MistralLLM(model="mistral-small-latest", client=fake, pricing_per_1m=(1.0, 3.0))
    wf = Workflow("classify")

    @wf.step(start=True)
    def classify(ctx):
        return Done(ctx.llm("Classify: I was charged twice", system="one word", json_mode=True).text)

    clock = FakeClock()
    store, queue = InMemoryStateStore(), InMemoryTaskQueue(clock)
    engine = Engine(store=store, queue=queue, bus=InMemoryEventBus(), llm=llm, clock=clock, workflows=[wf])
    run = engine.start("classify", {})
    LocalRunner(engine, queue, clock).run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and r.result == "billing"
    assert r.budget.tokens_used == 1100 and r.budget.cost_usd == pytest.approx(1000 / 1e6 * 1.0 + 100 / 1e6 * 3.0)
    req = fake.requests[0]
    assert req["messages"][0] == {"role": "system", "content": "one word"} and req["response_format"] == {"type": "json_object"}


def test_the_adapter_inside_the_durable_loop_survives_a_crash_without_reasking():
    fake = FakeMistral([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
    crashed = []

    def chaos(point, run):
        if point == "after_step_before_commit" and run.current_step == "act" and not crashed:
            crashed.append(1)
            raise SimulatedCrash()

    pay, clock = PaymentAPI(), FakeClock()
    store, queue = InMemoryStateStore(), InMemoryTaskQueue(clock)
    wf = make_tool_agent(MistralDecider(TOOL_SPECS, model="x", client=fake), {"charge": pay})
    engine = Engine(store=store, queue=queue, bus=InMemoryEventBus(), llm=None, clock=clock, workflows=[wf],
                    lease_ttl=timedelta(seconds=60), chaos=chaos)
    runner = LocalRunner(engine, queue, clock)
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    runner.run_until_idle()                                  # parks on the approval gate
    engine.resume(Event(run_id=run.run_id, key=f"tool-1:{run.run_id}", payload={"decision": "approve", "by": "cfo"}))
    with pytest.raises(SimulatedCrash):
        runner.run_until_idle()
    clock.advance(seconds=61)
    engine.reap()
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and list(pay.charges.values()) == [42.0] and pay.calls == 1
    assert len(fake.requests) == 2                          # the retry re-executed the intent, never re-asked
    # the second prompt carried the recorded tool result, not the model's memory
    assert fake.requests[1]["messages"][-1]["role"] == "tool"
    assert json.loads(fake.requests[1]["messages"][-1]["content"])["amount"] == 42
