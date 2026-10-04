# %% [markdown]
# # 01 · The agent loop and tool contracts
#
# An "agent" is a loop. The loop does these steps:
# 1. It builds the context.
# 2. It calls the model.
# 3. It validates and executes the tool calls that the model asked for.
# 4. It appends structured results.
# 5. It repeats, until a final answer or a budget stops it.
#
# The model supplies the judgement. **The harness supplies every guarantee**: schemas, side-effect classes, idempotency, timeouts and budgets.
# This notebook divides that loop into its parts with the lab's own `LlmAgent`. It examines one guarantee at a time.
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md). For more depth in this repo, see the [sandbox primer](../../../sandboxed-execution/PRIMER.md) §3 (the execution contract).
#
# In this notebook, you will:
# 1. Write tools that get their schema, validation and error contract from a typed Python signature.
# 2. Watch the loop run through its event log: parallel tool execution, unknown and duplicate calls, budgets and timeouts.
# 3. Design a tool contract, a model policy that corrects its own errors, and a tool wrapper. You can draw each of them in a design review.

# %%
import asyncio
import json
import time

from agentlab.llm import FakeLLM, call, calls, scripted, text
from agentlab.agents import (Budget, BudgetExceeded, Event, IdempotencyStore, Identity, InvocationContext, LlmAgent,
                           Runner, Session, SideEffect, ToolContext, ToolPermanentError, ToolTransientError, tool)


def new_session(message: str, session_id: str = "s1") -> Session:
    """A session whose log already holds the user's turn — exactly what the Runner does before calling an agent."""
    s = Session(id=session_id, tenant="t1", user="u1")
    s.append(Event(kind="user", payload={"content": message}))
    return s


def kinds(events) -> list[str]:
    return [e.kind for e in events]

# %% [markdown]
# ## 1. A tool is a contract, and the contract comes from the signature
#
# `@tool` changes a typed function into a `FunctionTool`.
# It makes the JSON schema from the parameters (with pydantic inside). It takes the description from the docstring.
# If you declare a `ctx: ToolContext` parameter, the runtime injects it, and the model never sees it.
# The model sees exactly `spec.to_model_schema()`. The model pays for this schema on every turn.

# %%
ORDERS = {"ORD-1": {"order_id": "ORD-1", "status": "shipped", "eta": "2026-09-08"}}


@tool
def lookup_order(order_id: str) -> dict:
    """Current fulfilment status of one order. Use when the user asks where an order is or when it will arrive."""
    if order_id not in ORDERS:
        raise ToolPermanentError(f"no order {order_id}", type="not_found", hint="Ask the user to check the order number.")
    return ORDERS[order_id]


print(json.dumps(lookup_order.spec.to_model_schema(), indent=1))

# %% [markdown]
# The runtime validates the arguments **before** the function runs. A malformed call does not raise an exception.
# It returns a structured result that the model can act on.
# The result has an error `type`, a message that names the field, and a `hint` that tells the model what to do next.

# %%
ctx = ToolContext()
bad = await lookup_order.run({"order": "ORD-1"}, ctx)            # wrong argument name — the function never ran
print("ok:", bad.ok, "| type:", bad.error.type, "| retryable:", bad.error.retryable)
print("what the model reads:", bad.to_content())
good = await lookup_order.run({"order_id": "ORD-1"}, ctx)
print("what the model reads:", good.to_content())

# %% [markdown]
# ### Side-effect classes
#
# Every tool declares its effect on the world. The three classes are `READ`, `REVERSIBLE` (you can undo it, or it is idempotent by key)
# and `IRREVERSIBLE` (money moved, email sent). By default, an irreversible tool **must get a confirmation**. The loop pauses for a person
# (Notebook 03 shows the machinery for the pause and the resume). `required_scope` connects a tool to the identity of the caller.
# Thus, whatever text the model writes, it cannot get an action that the user has no permission to do.

# %%
@tool(side_effect=SideEffect.REVERSIBLE)
def hold_shipment(order_id: str) -> dict:
    """Put a shipment on hold; it can be released later."""
    return {"held": order_id}


@tool(side_effect=SideEffect.IRREVERSIBLE, required_scope="refunds:write")
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund an order. Money leaves the account."""
    return {"refunded": amount}


for t in (lookup_order, hold_shipment, issue_refund):
    s = t.spec
    print(f"{s.name:14s} side_effect={s.side_effect.value:12s} requires_confirmation={str(s.requires_confirmation):5s} scope={s.required_scope}")

denied = await issue_refund.run({"order_id": "ORD-1", "amount": 10.0}, ToolContext(user=Identity("u1", scopes=set())))
print("without the scope →", denied.to_content())

# %% [markdown]
# ### Errors are part of the contract
#
# A tool function raises one of two exceptions. The runtime maps the exception to a result that the model can reason about:
#
# | raised inside the tool | `error.type` | `retryable` | who acts |
# |---|---|---|---|
# | `ToolPermanentError(msg, type=..., hint=...)` | your type (`not_found`, …) | `False` | the model does what the hint says |
# | `ToolTransientError(msg)` | `transient` | `True` | the runtime can retry. The model can tell the user. |
# | anything else | `tool_failure` | `False` | the model explains. **No stack trace leaks.** |

# %%
@tool
def flaky_inventory(sku: str) -> dict:
    """Stock level for a SKU (the upstream is flaky today)."""
    raise ToolTransientError("inventory service returned 503")


@tool
def buggy_inventory(sku: str) -> dict:
    """A tool with a bug in it."""
    return {"stock": 1 / 0}


for t, args in ((lookup_order, {"order_id": "ORD-404"}), (flaky_inventory, {"sku": "A1"}), (buggy_inventory, {"sku": "A1"})):
    r = await t.run(args, ctx)
    print(f"{t.spec.name:16s} → {r.to_content()}")

# %% [markdown]
# ### Idempotent writes
#
# When the runtime retries a write, the write must not post two times. Give a non-read tool an `IdempotencyStore`.
# The caller can supply an `idempotency_key`. The loop makes one from the session, the agent, the step and the call signature.
# When a second execution has the same key, the store serves the result and marks it `from_idempotency_cache`.
# The function body runs once.

# %%
idem = IdempotencyStore()
payouts: list[str] = []


@tool(side_effect=SideEffect.REVERSIBLE, idempotency=idem)
def reserve_payout(claim_id: str, amount: float) -> dict:
    """Reserve a payout for a claim."""
    payouts.append(claim_id)
    return {"reserved": amount, "claim_id": claim_id}


key_ctx = ToolContext(idempotency_key="s1:claims:step3:reserve_payout:CLM-7")
first = await reserve_payout.run({"claim_id": "CLM-7", "amount": 120.0}, key_ctx)
second = await reserve_payout.run({"claim_id": "CLM-7", "amount": 120.0}, key_ctx)
print("executions:", len(payouts), "| second served from cache:", second.from_idempotency_cache, "| same data:", first.data == second.data)

# %% [markdown]
# ## 2. The loop, observed through its event log
#
# `LlmAgent.run` yields an `Event` for each thing that it does. It also appends the same events to the session.
# With a scripted model, you can predict all of the trajectory before you run it:
# 1. The model asks for a tool.
# 2. The loop records the call.
# 3. The loop records the result.
# 4. The model answers.
# 5. A `final` marker comes last.

# %%
llm = scripted(call("lookup_order", order_id="ORD-1"), "ORD-1 has shipped and arrives on 8 September.")
agent = LlmAgent("orders", llm, "You are an order-tracking assistant.", tools=[lookup_order, hold_shipment])
session = new_session("Where is order ORD-1?")
events = await agent.run_to_completion(InvocationContext(session=session))
for e in events:
    print(f"step {e.step}  {e.kind:12s} {json.dumps(e.payload)[:95]}")

# %% [markdown]
# The second model call must contain the tool result. If not, the model answers with no information.
# `FakeLLM.calls` records every request that it received. Thus you can prove that the result went back:

# %%
for m in llm.calls[1]["messages"]:
    extra = f"  tool_calls={[tc['name'] for tc in m['tool_calls']]}" if m.get("tool_calls") else ""
    print(f"{m['role']:10s} {str(m.get('content'))[:80]!r}{extra}")

# %% [markdown]
# ### Independent tool calls run in parallel
#
# When the model asks for several tools in one turn, the loop runs them at the same time.
# It uses a semaphore, and each tool has its own timeout. Three 50 ms tools cost about 50 ms, not 150.
# This is why it is worth the effort to prompt the model to batch independent calls.

# %%
@tool
async def slow_lookup(q: str) -> dict:
    """A 50 ms lookup."""
    await asyncio.sleep(0.05)
    return {"q": q}


par_llm = scripted(calls(call("slow_lookup", q="a"), call("slow_lookup", q="b"), call("slow_lookup", q="c")), "All three done.")
par_agent = LlmAgent("par", par_llm, "Look things up.", tools=[slow_lookup])
t0 = time.perf_counter()
await par_agent.run_to_completion(InvocationContext(session=new_session("look up a, b and c")))
elapsed = time.perf_counter() - t0
print(f"three 50 ms tools finished in {elapsed * 1000:.0f} ms")
assert elapsed < 0.15, "tool calls ran sequentially?"

# %% [markdown]
# ### Unknown and repeated calls are answered, not crashed
#
# When the model hallucinates a tool name, the result is `unknown_tool`, with the real names in the hint.
# If the model repeats the same call more than `max_repeated_calls` times, the result is `duplicate_call`.
# This is the lowest-cost possible brake on a "call, ignore the result, call again" spiral.

# %%
dup_llm = scripted(call("track_parcel", id="ORD-1"),            # no such tool
                   call("lookup_order", order_id="ORD-1"),
                   call("lookup_order", order_id="ORD-1"),
                   call("lookup_order", order_id="ORD-1"),       # the third identical call
                   "ORD-1 has shipped.")
dup_agent = LlmAgent("orders", dup_llm, "Track orders.", tools=[lookup_order], max_repeated_calls=2)
dup_session = new_session("Where is ORD-1?")
await dup_agent.run_to_completion(InvocationContext(session=dup_session, budget=Budget(max_steps=10)))
for e in dup_session.events:
    if e.kind == "tool_result":
        print(f"step {e.step}: {e.payload['content'][:150]}")

# %% [markdown]
# ### Budgets stop a runaway loop
#
# A model that calls tools forever is a cost incident.
# `Budget` sets limits on every invocation in three units: model calls (`max_steps`), tokens and wall-clock seconds.
# Every delegated sub-agent shares the same `Budget` (Notebook 02). Thus a delegation cannot escape it.
#
# Here, a policy *always* calls a tool, with new arguments each time. Thus the duplicate detection cannot help us.
# The budget stops the policy after three model calls.

# %%
runaway = FakeLLM(policy=lambda messages, tools: calls(call("lookup_order", order_id=f"ORD-{len(messages)}")))
run_agent = LlmAgent("orders", runaway, "Track orders.", tools=[lookup_order])
run_session = new_session("Find all my orders")
try:
    await run_agent.run_to_completion(InvocationContext(session=run_session, budget=Budget(max_steps=3)))
except BudgetExceeded as e:
    print("stopped:", e)
    print("reason:", e.reason, "| model calls made:", kinds(run_session.events).count("model"))

# %% [markdown]
# ### The model call has a deadline too
#
# `model_timeout_s` sets a limit on each model call. The seconds that the budget has left can make this limit shorter.
# The raw loop raises `asyncio.TimeoutError`. The `Runner` is the boundary that owns persistence.
# The `Runner` records an `error` event and marks the session failed. It does not leak the exception to the caller.

# %%
class HangingLLM(FakeLLM):
    async def generate(self, messages, tools=None, **options):
        await asyncio.sleep(1.0)              # a provider that does not come back in time
        return text("too late")


try:
    await LlmAgent("slow", HangingLLM(), "Help.").run_to_completion(InvocationContext(session=new_session("hi"), model_timeout_s=0.05))
except asyncio.TimeoutError:
    print("raw loop: asyncio.TimeoutError after 50 ms")

r = await Runner(LlmAgent("slow", HangingLLM(), "Help."), model_timeout_s=0.05).run("s-timeout", "hi")
print("via Runner: error =", repr(r.error), "| session status =", r.session.status.value, "| last event:", r.session.events[-1].kind)

# %% [markdown]
# ## 3. Exercises
#
# ### Exercise 3.1 — design a tool contract
#
# Define `get_order_status(order_id: str, include_items: bool = False)` with `@tool`.
# The docstring is the only guidance that the model has about **when** to use the tool. Thus, say this clearly.
# Also say what the tool is *not* for (refunds go to a different tool).
#
# Return a dict with `order_id` and `status`. Add an `items` list only when the caller asks for it.
# Keep the description under 400 characters. The runtime sends it again on every model call.

# %% exercise
### BEGIN SOLUTION
@tool
def get_order_status(order_id: str, include_items: bool = False) -> dict:
    """Current fulfilment status and ETA of one order.

    Use this when the user asks where an order is, whether it has shipped or when it will arrive.
    Do not use it for refunds or cancellations. Set include_items=True only when the user asks what the order contains.
    """
    status = {"order_id": order_id, "status": "shipped", "eta": "2026-09-08"}
    if include_items:
        status["items"] = [{"sku": "SHOE-42", "qty": 1}]
    return status
### END SOLUTION

# %% check
spec = get_order_status.spec
schema = spec.input_schema
assert schema["type"] == "object" and set(schema["properties"]) == {"order_id", "include_items"}, schema
assert schema["properties"]["order_id"]["type"] == "string", schema["properties"]["order_id"]
assert schema["properties"]["include_items"] == {"type": "boolean", "default": False}, schema["properties"]["include_items"]
assert schema["required"] == ["order_id"], schema.get("required")
assert spec.side_effect == SideEffect.READ and not spec.requires_confirmation, "a lookup is a READ and needs no confirmation"
desc = spec.description.lower()
assert "when" in desc and ("use" in desc or "call" in desc), "say WHEN to use the tool in the docstring"
assert "refund" in desc, "say what it is NOT for (refunds)"
assert len(spec.description) <= 400, f"description is {len(spec.description)} chars — the model pays for it every turn"
r1 = await get_order_status.run({"order_id": "ORD-1"}, ToolContext())
assert r1.ok and r1.data["order_id"] == "ORD-1" and "status" in r1.data and "items" not in r1.data, r1
r2 = await get_order_status.run({"order_id": "ORD-1", "include_items": True}, ToolContext())
assert r2.ok and isinstance(r2.data.get("items"), list), r2
missing = await get_order_status.run({}, ToolContext())
assert not missing.ok and missing.error.type == "invalid_arguments", missing
print("✅ contract the model sees:", json.dumps(spec.to_model_schema())[:110], "…")

# %% [markdown]
# ### Exercise 3.2 — a not-found the model can act on
#
# Write `find_customer(email: str)` over the `CUSTOMERS` table.
# For an unknown email, raise `ToolPermanentError` with `type="not_found"` and a `hint`.
# The hint tells the model what to do next: ask the user to confirm the address, or offer to create a profile.
# Do not return `None` for "not found". A silent empty result causes models to invent customers.

# %% exercise
CUSTOMERS = {"anil@example.com": {"id": "C-1", "name": "Anil", "tier": "gold"}}


@tool
def find_customer(email: str) -> dict:
    """Find a customer profile by email. Use before any account-specific action."""
    ### BEGIN SOLUTION
    customer = CUSTOMERS.get(email.strip().lower())
    if customer is None:
        raise ToolPermanentError(f"no customer with email {email}", type="not_found",
                                 hint="Ask the user to confirm the email address, or offer to create a profile.")
    return customer
    ### END SOLUTION

# %% check
found = await find_customer.run({"email": "anil@example.com"}, ToolContext())
assert found.ok and found.data["id"] == "C-1", found
nf = await find_customer.run({"email": "nobody@example.com"}, ToolContext())
assert nf.ok is False and nf.error is not None and nf.data is None, nf
assert nf.error.type == "not_found", f"expected error type not_found, got {nf.error}"
assert nf.error.retryable is False, "not-found is permanent: retrying the same call cannot help"
assert nf.error.hint and len(nf.error.hint) > 15, "give the model a next step in the hint"
assert json.loads(nf.to_content())["hint"] == nf.error.hint, nf.to_content()
print("✅ the model reads:", nf.to_content())

# %% [markdown]
# ### Exercise 3.3 — a model that reads its own error
#
# Real models make malformed calls. A good harness gives back a structured error, and a good model recovers.
# Write `self_correcting_policy(messages, tools)`. In this exercise, you play the part of the model.
# For the tool `lookup_order`, the policy does these steps:
#
# 1. On its first step, it calls `lookup_order` with the incorrect argument name: `order="ORD-1"`. It does this intentionally and carelessly.
# 2. When the latest tool result is an `invalid_arguments` error, it reads the tool schema from `tools`
#    (`input_schema["required"]`). Then it sends the call again, with the correct argument name and the same value.
# 3. When the latest tool result is `ok`, it answers `"Order ORD-1 is <status>."`.
#
# Tool results come as `{"role": "tool", "content": <json>}` messages after the last user turn. Use `json.loads` on the content.

# %% exercise
def self_correcting_policy(messages, tools):
    ### BEGIN SOLUTION
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    results = [m for m in messages[last_user + 1:] if m.get("role") == "tool"]
    if not results:
        return call("lookup_order", order="ORD-1")                       # the sloppy first attempt
    last = json.loads(results[-1]["content"])
    if not last["ok"] and last["error"] == "invalid_arguments":
        schema = next(t for t in tools if t["name"] == "lookup_order")["input_schema"]
        field = schema["required"][0]
        return call("lookup_order", **{field: "ORD-1"})                  # corrected from the contract, not guessed
    if last["ok"]:
        return text(f"Order ORD-1 is {last['data']['status']}.")
    return text("I could not look that order up.")
    ### END SOLUTION

# %% check
sc_llm = FakeLLM(policy=self_correcting_policy)
sc_agent = LlmAgent("orders", sc_llm, "Track orders.", tools=[lookup_order])
sc_session = new_session("Where is ORD-1?")
sc_events = await sc_agent.run_to_completion(InvocationContext(session=sc_session, budget=Budget(max_steps=5)))
assert kinds(sc_events) == ["model", "tool_call", "tool_result", "model", "tool_call", "tool_result", "model", "final"], kinds(sc_events)
tool_calls = [e.payload for e in sc_events if e.kind == "tool_call"]
tool_results = [e.payload for e in sc_events if e.kind == "tool_result"]
assert tool_calls[0]["args"] == {"order": "ORD-1"}, tool_calls[0]
assert tool_results[0]["error"] == "invalid_arguments", tool_results[0]
assert tool_calls[1]["args"] == {"order_id": "ORD-1"}, tool_calls[1]
assert tool_results[1]["ok"] is True, tool_results[1]
assert sc_session.last_final_text() == "Order ORD-1 is shipped.", sc_session.last_final_text()
print("✅ trajectory: bad call → structured error → corrected call → answer, in", sc_llm.call_count, "model calls")

# %% [markdown]
# ### Exercise 3.4 — bound the loop, then fail gracefully
#
# `looping_policy` in the next cell never answers. Write `guard_budget()`.
# It gives a `Budget` under which such a run makes **exactly four** model calls.
#
# Then write `run_bounded(agent, session, budget)`. It runs the agent and returns the `BudgetExceeded.reason` string
# (for example `"max_steps=4"`). It does not raise. If the run finished normally, it returns `None`.
#
# This is what the `Runner` does at its boundary. A budget stop is an expected result to record, not a crash.

# %% exercise
looping_policy = lambda messages, tools: calls(call("lookup_order", order_id=f"ORD-{len(messages)}"))


def guard_budget() -> Budget:
    ### BEGIN SOLUTION
    return Budget(max_steps=4)
    ### END SOLUTION


async def run_bounded(agent, session, budget) -> str | None:
    ### BEGIN SOLUTION
    try:
        await agent.run_to_completion(InvocationContext(session=session, budget=budget))
        return None
    except BudgetExceeded as e:
        return e.reason
    ### END SOLUTION

# %% check
b_session = new_session("find everything")
b_agent = LlmAgent("orders", FakeLLM(policy=looping_policy), "Track orders.", tools=[lookup_order])
reason = await run_bounded(b_agent, b_session, guard_budget())
model_calls = kinds(b_session.events).count("model")
assert model_calls == 4, f"expected exactly 4 model calls, got {model_calls}"
assert reason is not None and reason.startswith("max_steps"), reason
ok_session = new_session("Where is ORD-1?")
ok_agent = LlmAgent("orders", scripted(call("lookup_order", order_id="ORD-1"), "Shipped."), "Track orders.", tools=[lookup_order])
assert await run_bounded(ok_agent, ok_session, guard_budget()) is None, "a normal run must complete and return None"
print(f"✅ stopped after {model_calls} model calls with reason {reason!r}; a normal run still completes")

# %% [markdown]
# ### Exercise 3.5 — compose tools: a result-size guard
#
# Tool output is the largest uncontrolled input to your context window. Write `with_result_limit(tool, max_chars)`.
# It returns an object that satisfies the `Tool` protocol (a `spec` attribute and `async run(args, ctx)`):
#
# * `spec` is the **same** `ToolSpec` object as the spec of the wrapped tool. The contract of the model does not change.
# * `run` delegates to the wrapped tool. If the result is `ok` and its compact JSON (`json.dumps(result.data, default=str)`)
#   is longer than `max_chars`, replace `data`. Put the first `max_chars` characters, then the marker
#   `…[truncated N chars; call again with a narrower request]`. `N` is the number of characters that you cut.
# * Errors, small results and latency pass through with no change.
#
# With wrappers like this, you can add caching, redaction or rate limits. You do not touch the tool, and you do not touch the model.

# %% exercise
@tool
def list_transactions(account_id: str) -> list[dict]:
    """All transactions on an account (can be thousands)."""
    return [{"id": f"tx-{i}", "amount": round(i * 1.5, 2), "memo": "coffee"} for i in range(200)]


### BEGIN SOLUTION
class _LimitedTool:
    def __init__(self, inner, max_chars: int):
        self.inner, self.max_chars = inner, max_chars
        self.spec = inner.spec                      # same contract object: the model notices nothing

    async def run(self, args, ctx):
        result = await self.inner.run(args, ctx)
        if not result.ok:
            return result
        payload = json.dumps(result.data, default=str)
        if len(payload) <= self.max_chars:
            return result
        cut = len(payload) - self.max_chars
        result.data = payload[: self.max_chars] + f"…[truncated {cut} chars; call again with a narrower request]"
        return result


def with_result_limit(tool, max_chars: int):
    return _LimitedTool(tool, max_chars)
### END SOLUTION

# %% check
limited = with_result_limit(list_transactions, 300)
assert hasattr(limited, "spec") and callable(getattr(limited, "run", None)), "expose .spec and an async .run(args, ctx)"
assert limited.spec is list_transactions.spec, "the model's contract must be the untouched original spec"
full = json.dumps((await list_transactions.run({"account_id": "acc-1"}, ToolContext())).data, default=str)
res = await limited.run({"account_id": "acc-1"}, ToolContext())
assert res.ok and isinstance(res.data, str), res
assert res.data.startswith(full[:300]) and "truncated" in res.data and "narrower request" in res.data, res.data[-120:]
assert len(res.data) < 300 + 80, len(res.data)
small = with_result_limit(lookup_order, 300)
untouched = await small.run({"order_id": "ORD-1"}, ToolContext())
assert untouched.ok and untouched.data == ORDERS["ORD-1"], "small results must pass through unchanged"
err = await small.run({"order_id": "ORD-404"}, ToolContext())
assert not err.ok and err.error.type == "not_found", "errors must pass through unchanged"
w_llm = scripted(call("list_transactions", account_id="acc-1"), "You have 200 transactions; the list was truncated.")
w_session = new_session("show my transactions")
await LlmAgent("bank", w_llm, "You are a bank assistant.", tools=[limited]).run_to_completion(InvocationContext(session=w_session))
sent = [m for m in w_llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"]
assert "truncated" in sent and len(sent) < 500, len(sent)
print(f"✅ {len(full)} chars of tool output became {len(sent)} chars in the model's context")

# %% [markdown]
# ### Exercise 3.6 — say it in one paragraph
#
# In `why_structured_errors`, write two to four sentences. Explain why a tool error must reach the model as a **structured
# result** (`type`, `message`, `retryable`, `hint`). Explain why it must never reach the model as a stack trace.
# Say what each field lets the model or the runtime do.

# %% exercise
### BEGIN SOLUTION
why_structured_errors = (
    "A stack trace tells the model nothing it can act on and leaks internals (paths, hosts, versions) into a context "
    "that may be logged or echoed to the user. A structured error separates the decisions: the retryable flag tells the "
    "runtime whether an automatic retry is safe, the type lets policies and evals classify failures, and the hint tells "
    "the model what to do next — fix the arguments, ask the user, or stop — so the loop converges instead of repeating "
    "the same broken call."
)
### END SOLUTION

# %% check
_t = why_structured_errors.lower()
assert len(why_structured_errors) > 120, "write two to four real sentences"
assert "retry" in _t, "mention what `retryable` is for"
assert "hint" in _t or "next" in _t, "mention what the hint is for"
assert "stack" in _t or "trace" in _t or "leak" in _t, "say what is wrong with stack traces"
print("✅", why_structured_errors[:100], "…")

# %% [markdown]
# ## The one-minute version
#
# When a person asks *"walk me through what happens when the agent calls a tool"*, describe the loop that you watched:
# 1. The model returns a typed call.
# 2. The runtime **validates it against the schema**. A bad call becomes an `invalid_arguments` result, not an exception.
# 3. The runtime examines the **scope** and the **side-effect class**. An irreversible tool causes a pause for approval.
# 4. The runtime executes the call **in parallel, with a timeout and an idempotency key**.
# 5. It appends a **structured result** that the model can act on.
# 6. It repeats under a **budget of steps, tokens and seconds**.
#
# Then make the design point. The model is probabilistic. Thus every guarantee that the customer cares about is in the harness:
# no double refunds, no runaway spend, no leaked stack traces. You can point to the exact line that enforces each guarantee.
