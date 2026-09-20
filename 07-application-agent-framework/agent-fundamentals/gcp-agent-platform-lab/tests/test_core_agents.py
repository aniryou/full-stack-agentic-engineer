import asyncio
import time

import pytest

from agentlab.agents import (AgentTool, Budget, BudgetExceeded, ContextBuilder, Identity, IdempotencyStore,
                           InMemorySessionStore, InvocationContext, LlmAgent, LoopAgent, ParallelAgent, Runner,
                           SequentialAgent, Session, SessionStatus, SideEffect, ToolContext, ToolPermanentError,
                           ToolTransientError, VersionConflict, tool)
from agentlab.agents.state import Event
from agentlab.llm import FakeLLM, FakeLLMExhausted, KeywordPlanner, Rule, call, calls, scripted, text


# ---------------------------------------------------------------- fixtures
@tool
def get_balance(account_id: str) -> dict:
    """Current balance for an account."""
    if account_id == "missing":
        raise ToolPermanentError("no such account", type="not_found", hint="Ask the user to confirm the account id.")
    return {"account_id": account_id, "balance": 1234.5, "currency": "SGD"}


@tool
async def slow_lookup(q: str) -> dict:
    """A slow tool (50 ms)."""
    await asyncio.sleep(0.05)
    return {"q": q}


def make_agent(llm, tools=None, **kw):
    return LlmAgent("assistant", llm, "You are a bank assistant.", tools=tools or [get_balance, slow_lookup], **kw)


def new_session(msg="What is my balance for acc-1?"):
    s = Session(id="s1", tenant="t1", user="u1")
    s.append(Event(kind="user", payload={"content": msg}))
    return s


# ---------------------------------------------------------------- FakeLLM
async def test_scripted_llm_runs_out():
    llm = scripted("hello")
    r = await llm.generate([{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.usage.input_tokens > 0
    with pytest.raises(FakeLLMExhausted):
        await llm.generate([{"role": "user", "content": "hi"}])


async def test_prefix_cache_reports_cached_tokens():
    llm = FakeLLM(responses=["a", "b"])
    system = {"role": "system", "content": "policy " * 100}
    r1 = await llm.generate([system, {"role": "user", "content": "one"}])
    r2 = await llm.generate([system, {"role": "user", "content": "two"}])
    assert r1.usage.cached_tokens == 0
    assert r2.usage.cached_tokens > 0 and r2.usage.cached_tokens <= r2.usage.input_tokens


async def test_keyword_planner_calls_tools_then_answers():
    planner = KeywordPlanner([Rule(r"balance", "get_balance", lambda t: {"account_id": "acc-1"})])
    llm = FakeLLM(policy=planner)
    r = await llm.generate([{"role": "user", "content": "balance please"}], tools=[{"name": "get_balance"}])
    assert r.tool_calls and r.tool_calls[0].name == "get_balance"
    r2 = await llm.generate([{"role": "user", "content": "balance please"}, {"role": "tool", "name": "get_balance", "content": "{...}"}])
    assert r2.text and "get_balance" in r2.text


# ------------------------------------------------------------------ tools
def test_function_tool_schema_and_validation():
    spec = get_balance.spec
    assert spec.name == "get_balance" and "account_id" in spec.input_schema["properties"]
    assert spec.input_schema["required"] == ["account_id"]
    assert spec.side_effect == SideEffect.READ and not spec.requires_confirmation


async def test_function_tool_error_contract():
    ctx = ToolContext()
    bad = await get_balance.run({"account": "x"}, ctx)
    assert not bad.ok and bad.error.type == "invalid_arguments" and bad.error.hint
    nf = await get_balance.run({"account_id": "missing"}, ctx)
    assert nf.error.type == "not_found" and nf.error.retryable is False
    ok = await get_balance.run({"account_id": "acc-1"}, ctx)
    assert ok.ok and ok.data["balance"] == 1234.5
    assert '"ok":true' in ok.to_content()


async def test_idempotent_write_tool():
    store = IdempotencyStore()
    calls_made = []

    @tool(side_effect=SideEffect.REVERSIBLE, idempotency=store)
    def hold_shipment(order_id: str) -> dict:
        """Place a hold."""
        calls_made.append(order_id)
        return {"held": order_id}

    ctx = ToolContext(idempotency_key="s1:step1:hold")
    r1 = await hold_shipment.run({"order_id": "o-1"}, ctx)
    r2 = await hold_shipment.run({"order_id": "o-1"}, ctx)
    assert r1.ok and r2.ok and r2.from_idempotency_cache and calls_made == ["o-1"]


async def test_scope_enforced():
    @tool(required_scope="cards:write")
    def block_card(card_id: str) -> dict:
        """Block a card."""
        return {"blocked": card_id}

    denied = await block_card.run({"card_id": "c"}, ToolContext(user=Identity("u", scopes=set())))
    allowed = await block_card.run({"card_id": "c"}, ToolContext(user=Identity("u", scopes={"cards:write"})))
    assert denied.error.type == "forbidden" and allowed.ok


async def test_transient_error_marked_retryable():
    @tool
    def flaky() -> dict:
        """Flaky."""
        raise ToolTransientError("503 from upstream")

    r = await flaky.run({}, ToolContext())
    assert not r.ok and r.error.retryable is True and r.error.type == "transient"


# ------------------------------------------------------------------- loop
async def test_agent_loop_end_to_end():
    llm = scripted(call("get_balance", account_id="acc-1"), "Your balance is SGD 1,234.50.")
    agent = make_agent(llm)
    s = new_session()
    ctx = InvocationContext(session=s)
    events = await agent.run_to_completion(ctx)
    kinds = [e.kind for e in events]
    assert kinds == ["model", "tool_call", "tool_result", "model", "final"]
    assert s.last_final_text().startswith("Your balance")
    # the second model call saw the tool result
    assert any(m["role"] == "tool" for m in llm.calls[1]["messages"])
    assert s.usage().input_tokens > 0


async def test_parallel_tool_execution_is_concurrent():
    llm = scripted(calls(call("slow_lookup", q="a"), call("slow_lookup", q="b"), call("slow_lookup", q="c")), "done")
    agent = make_agent(llm)
    s = new_session("look up a b c")
    t0 = time.perf_counter()
    await agent.run_to_completion(InvocationContext(session=s))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.12, f"tool calls ran sequentially ({elapsed:.3f}s)"


async def test_unknown_tool_and_duplicate_detection():
    llm = scripted(call("nope", x=1),
                   call("get_balance", account_id="acc-1"),
                   call("get_balance", account_id="acc-1"),
                   call("get_balance", account_id="acc-1"),
                   "final")
    agent = make_agent(llm, max_repeated_calls=2)
    s = new_session()
    await agent.run_to_completion(InvocationContext(session=s, budget=Budget(max_steps=10)))
    errors = [e.payload["error"] for e in s.events if e.kind == "tool_result"]
    assert errors[0] == "unknown_tool"
    assert errors.count("duplicate_call") == 1


async def test_budget_stops_runaway_loop():
    llm = FakeLLM(policy=lambda m, t: calls(call("get_balance", account_id=str(len(m)))))
    agent = make_agent(llm)
    s = new_session()
    with pytest.raises(BudgetExceeded):
        await agent.run_to_completion(InvocationContext(session=s, budget=Budget(max_steps=3)))
    assert sum(1 for e in s.events if e.kind == "model") == 3


async def test_model_timeout_is_enforced():
    class HangingLLM(FakeLLM):
        async def generate(self, messages, tools=None, **o):
            await asyncio.sleep(1)
            return text("late")

    agent = make_agent(HangingLLM())
    with pytest.raises(asyncio.TimeoutError):
        await agent.run_to_completion(InvocationContext(session=new_session(), model_timeout_s=0.05))


# ------------------------------------------------------------- approvals
async def test_runner_pauses_for_confirmation_and_resumes():
    executed = []

    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def issue_refund(order_id: str, amount: float) -> dict:
        """Refund an order (irreversible)."""
        executed.append((order_id, amount))
        return {"refunded": amount}

    llm = scripted(call("issue_refund", order_id="o-9", amount=20.0), "Refund of 20 issued.")
    agent = LlmAgent("refunds", llm, "Handle refunds.", tools=[issue_refund])
    runner = Runner(agent)
    r1 = await runner.run("s-ref", "refund order o-9 for $20")
    assert r1.paused and r1.session.status == SessionStatus.AWAITING_APPROVAL
    assert executed == []
    with pytest.raises(RuntimeError):
        await runner.run("s-ref", "hello again")
    r2 = await runner.approve("s-ref", True)
    assert not r2.paused and executed == [("o-9", 20.0)] and r2.text.startswith("Refund")


async def test_runner_declined_approval_feeds_model():
    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def issue_refund(order_id: str, amount: float) -> dict:
        """Refund."""
        return {"refunded": amount}

    llm = scripted(call("issue_refund", order_id="o-9", amount=20.0), "Understood, no refund issued.")
    runner = Runner(LlmAgent("refunds", llm, "Handle refunds.", tools=[issue_refund]))
    await runner.run("s-dec", "refund o-9")
    r = await runner.approve("s-dec", False)
    assert r.text.startswith("Understood")
    last_tool = [e for e in r.session.events if e.kind == "tool_result"][-1]
    assert last_tool.payload["error"] == "declined"


async def test_confirm_hook_short_circuits_pause():
    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def send_email(to: str) -> dict:
        """Send."""
        return {"sent": to}

    async def auto_yes(tc, spec):
        return True

    llm = scripted(call("send_email", to="a@b.c"), "sent")
    agent = LlmAgent("mailer", llm, "Mail.", tools=[send_email])
    s = new_session("email a@b.c")
    events = await agent.run_to_completion(InvocationContext(session=s, confirm=auto_yes))
    assert any(e.kind == "approval" and e.payload["approved"] for e in s.events)
    assert s.last_final_text() == "sent"


# ---------------------------------------------------------------- stores
def test_store_compare_and_set():
    store = InMemorySessionStore()
    s = store.create(Session(id="x"))
    a = store.get("x")
    b = store.get("x")
    a.set_state("k", 1)
    store.put(a)
    b.set_state("k", 2)
    with pytest.raises(VersionConflict):
        store.put(b)
    assert store.get("x").state["k"] == 1


# ------------------------------------------------------------- workflows
async def test_sequential_pipeline_passes_state():
    extract = LlmAgent("extract", scripted("ORDER-42"), "Extract the order id.", output_key="order_id")
    answer = LlmAgent("answer", FakeLLM(policy=lambda m, t: text(f"Status for {m[0]['content'].split('Order: ')[-1]}")),
                      "Order: {order_id}", output_key="answer")
    pipe = SequentialAgent("pipeline", [extract, answer])
    s = new_session("where is my order?")
    await pipe.run_to_completion(InvocationContext(session=s))
    assert s.state["order_id"] == "ORDER-42"
    assert s.state["answer"] == "Status for ORDER-42"


async def test_parallel_branches_merge_state():
    a = LlmAgent("a", scripted("A-result"), "A", output_key="a")
    b = LlmAgent("b", scripted("B-result"), "B", output_key="b")
    par = ParallelAgent("fanout", [a, b])
    s = new_session("go")
    await par.run_to_completion(InvocationContext(session=s))
    assert s.state["a"] == "A-result" and s.state["b"] == "B-result"


async def test_loop_agent_exits_on_condition():
    counter = {"n": 0}

    def policy(m, t):
        counter["n"] += 1
        return text("ok" if counter["n"] >= 2 else "retry")

    worker = LlmAgent("worker", FakeLLM(policy=policy), "Work.", output_key="out")
    loop = LoopAgent("loop", [worker], max_iterations=5, until=lambda s: s.state.get("out") == "ok")
    s = new_session("go")
    await loop.run_to_completion(InvocationContext(session=s, budget=Budget(max_steps=20)))
    notes = [e for e in s.events if e.kind == "note"]
    assert notes[-1].payload == {"loop_exit": "condition met", "iterations": 2}


# ------------------------------------------------------------ delegation
async def test_agent_tool_delegation():
    specialist = LlmAgent("cards", scripted("Card ending 1234 is now blocked."), "Cards specialist.", description="Card blocks and replacements")
    coordinator = LlmAgent("coordinator", scripted(call("cards", request="block card 1234"), "Done: card blocked."),
                           "Route to specialists.", sub_agents=[specialist])
    s = new_session("block my card")
    await coordinator.run_to_completion(InvocationContext(session=s))
    deleg = [e for e in s.events if e.kind == "delegation"]
    assert deleg and deleg[0].payload["answer"].startswith("Card ending")
    assert s.last_final_text() == "Done: card blocked."


async def test_delegation_depth_is_bounded():
    # a → a → a ... every level shares one budget; max_depth stops the recursion and max_steps ends the run
    llm = FakeLLM(policy=lambda m, t: calls(call("self_agent", request="again")) if t else text("leaf"))
    a = LlmAgent("self_agent", llm, "Recurse.")
    a.registry.add(AgentTool(a))
    s = new_session("go")
    with pytest.raises(BudgetExceeded):
        await a.run_to_completion(InvocationContext(session=s, budget=Budget(max_steps=12, max_depth=2)))
    # no child session ever went deeper than max_depth
    deleg = [e for e in s.events if e.kind == "delegation"]
    assert all(e.payload["child_session"].count("/") <= 2 for e in deleg)
    contents = [e.payload["content"] for e in s.events if e.kind == "tool_result"]
    assert any("budget_exceeded" in c for c in contents)


# ---------------------------------------------------------------- context
def test_context_builder_compacts_and_truncates():
    s = Session(id="c")
    for i in range(30):
        s.append(Event(kind="user", payload={"content": f"question {i}"}))
        s.append(Event(kind="tool_result", payload={"name": "big", "id": str(i), "content": "x" * 5000}))
        s.append(Event(kind="model", payload={"content": f"answer {i}"}))
    cb = ContextBuilder(instruction="Be helpful. Tier: {tier}", static_context=["policy text"], max_recent_turns=3, max_tool_result_chars=100)
    s.state["tier"] = "gold"
    msgs = cb.build(s)
    assert msgs[0]["role"] == "system" and "Tier: gold" in msgs[0]["content"] and "policy text" in msgs[0]["content"]
    assert msgs[1]["role"] == "system" and "Summary of earlier" in msgs[1]["content"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 3 and all("truncated" in m["content"] for m in tool_msgs)
    assert msgs[-1]["content"] == "answer 29"


# ------------------------------------------------ mixed batches and templating
async def test_pause_runs_independent_calls_first_and_resumes_cleanly():
    ran = []

    @tool
    def read_profile(customer_id: str) -> dict:
        """Read."""
        ran.append(("read", customer_id))
        return {"tier": "gold"}

    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def close_account(customer_id: str) -> dict:
        """Irreversible."""
        ran.append(("close", customer_id))
        return {"closed": True}

    llm = scripted(calls(call("read_profile", customer_id="c1"), call("close_account", customer_id="c1")),
                   "Profile read and account closed.")
    runner = Runner(LlmAgent("ops", llm, "Ops.", tools=[read_profile, close_account]))
    r1 = await runner.run("s-mix", "close c1")
    assert r1.paused
    assert ran == [("read", "c1")], "independent call in the same batch should run before pausing"
    r2 = await runner.approve("s-mix", True)
    assert ran == [("read", "c1"), ("close", "c1")]
    assert r2.text.startswith("Profile read")
    # both results are in the log exactly once
    names = [e.payload["name"] for e in r2.session.events if e.kind == "tool_result"]
    assert sorted(names) == ["close_account", "read_profile"]


async def test_second_confirmation_in_same_batch_is_deferred():
    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def a(x: int) -> dict:
        """A."""
        return {"a": x}

    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def b(x: int) -> dict:
        """B."""
        return {"b": x}

    llm = scripted(calls(call("a", x=1), call("b", x=2)), "done")
    runner = Runner(LlmAgent("ops", llm, "Ops.", tools=[a, b]))
    r = await runner.run("s-two", "do both")
    assert r.paused and r.pending["tool_call"]["name"] == "a"
    deferred = [e for e in r.session.events if e.kind == "tool_result" and e.payload["error"] == "deferred"]
    assert len(deferred) == 1 and deferred[0].payload["name"] == "b"


def test_render_template_supports_scoped_keys():
    from agentlab.agents.context import render_template
    out = render_template("Tier {user:tier}; policy {app:policy}; missing {nope}; plain {k}", {"user:tier": "gold", "app:policy": "v3", "k": 1})
    assert out == "Tier gold; policy v3; missing {nope}; plain 1"


async def test_delegation_usage_counted_in_parent():
    specialist = LlmAgent("spec", scripted("specialist answer " * 20), "Specialist.")
    coordinator = LlmAgent("coord", scripted(call("spec", request="help"), "final"), "Coordinate.", sub_agents=[specialist])
    s = new_session("go")
    await coordinator.run_to_completion(InvocationContext(session=s))
    own = sum((e.usage.total for e in s.events if e.kind == "model" and e.usage), 0)
    assert s.usage().total > own, "delegated tokens should be included via the delegation event"


async def test_scope_is_checked_before_confirmation():
    @tool(side_effect=SideEffect.IRREVERSIBLE, required_scope="cards:write")
    def block_card(card_id: str) -> dict:
        """Block."""
        return {"blocked": card_id}

    llm = scripted(call("block_card", card_id="c1"), "You are not permitted to block cards; I can raise a case.")
    runner = Runner(LlmAgent("cards", llm, "Cards.", tools=[block_card]))
    r = await runner.run("s-scope", "block my card", user=Identity("bob", scopes={"accounts:read"}))
    assert not r.paused, "a user without the scope must not be asked to confirm"
    results = [e for e in r.session.events if e.kind == "tool_result"]
    assert results[0].payload["error"] == "forbidden"
    assert not any(e.kind == "approval_required" for e in r.session.events)


async def test_pause_inside_delegate_surfaces_as_needs_approval():
    @tool(side_effect=SideEffect.IRREVERSIBLE)
    def wire_funds(amount: float) -> dict:
        """Wire."""
        return {"wired": amount}

    specialist = LlmAgent("payments", scripted(call("wire_funds", amount=5.0), "unreachable"), "Payments.", tools=[wire_funds])
    coordinator = LlmAgent("coord", scripted(call("payments", request="wire 5"), "Please confirm the wire and I'll do it."),
                           "Coordinate.", sub_agents=[specialist])
    runner = Runner(coordinator)
    r = await runner.run("s-deleg", "wire 5 dollars")
    assert not r.paused
    results = [e.payload for e in r.session.events if e.kind == "tool_result"]
    assert results[0]["error"] == "needs_approval"
    assert r.text.startswith("Please confirm")
