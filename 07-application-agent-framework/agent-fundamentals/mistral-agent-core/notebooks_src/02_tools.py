# %% [markdown]
# # 02 · Tools are contracts
#
# A tool is a function the model may call — but the model is a reader that cannot ask
# follow-up questions and will happily guess. So a good tool removes the need to guess:
# a clear schema, arguments validated **before** running, and errors the model can act
# on instead of stack traces.
#
# In this notebook you write tools the right way and see how the loop reacts to each
# kind of result.

# %%
from agentcore import Agent, FakeLLM, ToolError, call, tool

@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    orders = {"ORD-1": {"status": "shipped", "total": 42.0}}
    if order_id not in orders:
        raise ToolError(f"no order {order_id}", kind="not_found",
                        hint="Ask the customer to confirm the order id.")
    return {"order_id": order_id, **orders[order_id]}

print("schema     :", get_order.schema)
print("good call  :", get_order.run({"order_id": "ORD-1"}))
print("not found  :", get_order.run({"order_id": "ORD-9"}))
print("bad args   :", get_order.run({}))                      # missing required argument

# %% [markdown]
# Notice the three result shapes the model can receive:
# `{"ok": True, "data": ...}`, a **structured error** with a `hint`, and an
# `invalid_arguments` error. None of them is a crash. That is what lets the model
# recover — ask for the missing id, apologise for the not-found — instead of the whole
# turn failing.

# %% [markdown]
# ## Exercise 2.1 — a tool with an enum-like check
#
# Write `set_priority(ticket_id: str, level: str)` that accepts only `"low"`, `"medium"`
# or `"high"`. On a bad level, raise `ToolError(..., kind="invalid_value", hint=...)`.
# On success return `{"ticket_id": ..., "level": ...}`.

# %% exercise
@tool
def set_priority(ticket_id: str, level: str) -> dict:
    """Set a ticket's priority to low, medium, or high."""
    ### BEGIN SOLUTION
    if level not in {"low", "medium", "high"}:
        raise ToolError(f"{level!r} is not a valid level", kind="invalid_value",
                        hint="Use one of: low, medium, high.")
    return {"ticket_id": ticket_id, "level": level}
    ### END SOLUTION

# %% check
assert set_priority.run({"ticket_id": "T1", "level": "high"})["data"]["level"] == "high"
bad = set_priority.run({"ticket_id": "T1", "level": "urgent"})
assert bad["ok"] is False and bad["error"] == "invalid_value" and bad["hint"]
assert set_priority.run({"ticket_id": "T1"})["error"] == "invalid_arguments"
print("✅ set_priority validates its input")

# %% [markdown]
# ## Exercise 2.2 — make a write idempotent
#
# Writes get retried (a timeout, a dropped connection). A retried "refund" must not
# refund twice. Implement `make_refund_tool()` returning a `@tool`-decorated function
# `refund(order_id, amount)` that records each `order_id` it has refunded, with its
# result, in the closure dict `done`; a second call for the same order returns the
# **first** result with `{"already_done": True}` added and does **not** refund or record
# again — even if the retry carries a different amount.

# %% exercise
def make_refund_tool():
    done = {}                         # order_id -> result, the idempotency record
    @tool
    def refund(order_id: str, amount: float) -> dict:
        """Refund an order. Safe to retry."""
        ### BEGIN SOLUTION
        if order_id in done:
            return {**done[order_id], "already_done": True}
        result = {"order_id": order_id, "refunded": amount}
        done[order_id] = result
        return result
        ### END SOLUTION
    return refund

# %% check
refund = make_refund_tool()
first = refund.run({"order_id": "ORD-1", "amount": 20.0})
again = refund.run({"order_id": "ORD-1", "amount": 20.0})
garbled = refund.run({"order_id": "ORD-1", "amount": 99.0})   # a retry with a changed body
other = refund.run({"order_id": "ORD-2", "amount": 5.0})
assert first["data"]["refunded"] == 20.0 and not first["data"].get("already_done")
assert again["data"].get("already_done") is True
first_again = {k: v for k, v in again["data"].items() if k != "already_done"}
assert first_again == first["data"], "a retry must return the first result"
assert garbled["data"]["refunded"] == 20.0 and garbled["data"].get("already_done") is True, \
    "a retry must not overwrite the first refund (no double refund)"
assert other["data"]["refunded"] == 5.0 and not other["data"].get("already_done")
print("✅ refund is idempotent:", first["data"], "→", again["data"])

# %% [markdown]
# ## Exercise 2.3 — see the loop recover from a tool error
#
# Give an `Agent` the `get_order` tool and a **scripted model** that: (1) calls
# `get_order` for a missing order `ORD-9`, then (2) after seeing the `not_found` error,
# answers with the text `"I couldn't find order ORD-9 — can you confirm the number?"`.
# Build the `FakeLLM` script in `script` so the assertion passes.

# %% exercise
### BEGIN SOLUTION
script = [call("get_order", order_id="ORD-9"),
          "I couldn't find order ORD-9 — can you confirm the number?"]
### END SOLUTION

# %% check
r = Agent(FakeLLM(script), tools=[get_order]).run("where is my order ORD-9?")
assert r.done and "confirm" in r.text.lower()
tool_result = [m for m in r.messages if m["role"] == "tool"][0]["content"]
assert "not_found" in tool_result
print(r.transcript())
print("✅ the agent recovered from a not-found error instead of crashing")

# %% [markdown]
# ## The one-minute version
# In a design review, when you "design the interface", write one full tool contract on
# the board: name, description, arguments with an enum, the success shape, the error
# cases, and an idempotency key for writes. One concrete contract beats a list of tool
# names — it shows you have thought about what the model will do wrong.
