# %% [markdown]
# # 03 · State and control
#
# Two things separate a demo loop from something you would run: it remembers the
# conversation across turns, and it protects itself — a step budget, and not calling
# the same tool over and over. This notebook adds both, and shows a `policy` model that
# actually reacts to tool results (so you can drive multi-step behaviour without a rigid
# script).

# %%
from agentcore import Agent, FakeLLM, call, calls, text, tool

@tool
def get_balance(account_id: str) -> dict:
    """Return an account balance."""
    return {"account_id": account_id, "balance": 1234.5}

# %% [markdown]
# ## Multi-turn memory
# `Agent.run` returns the full transcript in `result.messages`. Pass it back as
# `history` on the next turn and the model sees the earlier exchange.

# %%
agent = Agent(FakeLLM([call("get_balance", account_id="a1"), "It's 1234.5.",
                       "You asked about account a1."]), tools=[get_balance])
first = agent.run("balance for a1?")
second = agent.run("which account did I just ask about?", history=first.messages)
print("second answer:", second.text)
print("history carried", len(second.messages), "messages")

# %% [markdown]
# ## Exercise 3.1 — a policy that reacts to results
#
# A **policy** decides each turn from the messages: `(messages, tools) -> Response`.
# Write `balance_policy` that: if the messages after the last user turn contain a tool
# result, answer with `text("Your balance is on file.")`; else if the last user message
# mentions `"balance"`, return `call("get_balance", account_id="a1")`; else
# `text("How can I help?")`.

# %% exercise
def balance_policy(messages, tools):
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    after = messages[last_user + 1:]
    ### BEGIN SOLUTION
    if any(m.get("role") == "tool" for m in after):
        return text("Your balance is on file.")
    if last_user >= 0 and "balance" in str(messages[last_user]["content"]).lower():
        return call("get_balance", account_id="a1")
    return text("How can I help?")
    ### END SOLUTION

# %% check
r = Agent(FakeLLM(policy=balance_policy), tools=[get_balance]).run("what's my balance?")
assert r.done and r.text == "Your balance is on file."
assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "tool", "assistant"]
assert Agent(FakeLLM(policy=balance_policy), tools=[get_balance]).run("hello").text == "How can I help?"
print("✅ policy drives the loop without a fixed script")

# %% [markdown]
# ## Exercise 3.2 — stop a repeated tool call
#
# A confused model can call the same tool with the same arguments forever, burning
# tokens. Write `dedupe(tool_calls, seen)`: `seen` is a set of signatures already run
# (use `tc.signature()`). Return only the calls whose signature is **new**, and add
# them to `seen`. (In a real loop you would feed the model a "you already have this"
# note for the dropped ones; here we just filter.)

# %% exercise
def dedupe(tool_calls, seen: set) -> list:
    fresh = []
    ### BEGIN SOLUTION
    for tc in tool_calls:
        sig = tc.signature()
        if sig not in seen:
            seen.add(sig)
            fresh.append(tc)
    ### END SOLUTION
    return fresh

# %% check
seen = set()
batch = [call("get_balance", account_id="a1"), call("get_balance", account_id="a1"),
         call("get_balance", account_id="a2")]
fresh = dedupe(batch, seen)
assert [tc.args["account_id"] for tc in fresh] == ["a1", "a2"]   # the duplicate a1 is gone
assert dedupe([call("get_balance", account_id="a1")], seen) == []  # already seen
print("✅ dedupe drops repeated calls")

# %% [markdown]
# ## Exercise 3.3 — reason about the budget
#
# `Agent(max_steps=N)` caps model calls per turn. For a model that **never** answers
# (always returns a tool call), how many `assistant` messages appear in the transcript
# when `max_steps=4`, and is `result.done` True or False? Set the two variables, then
# the check confirms them against a real run.

# %% exercise
### BEGIN SOLUTION
expected_assistant_messages = 4
expected_done = False
### END SOLUTION

# %% check
runaway = FakeLLM(policy=lambda m, t: calls(call("get_balance", account_id="a1")))
r = Agent(runaway, tools=[get_balance], max_steps=4).run("go")
assert sum(1 for m in r.messages if m["role"] == "assistant") == expected_assistant_messages
assert r.done is expected_done
print(f"✅ {expected_assistant_messages} model calls, done={expected_done} — the budget saved you")

# %% [markdown]
# ## The one-minute version
# Every design review asks about state. Name the kinds: the **conversation** (the
# transcript), small **working state** (what stage a task is at), and **budgets**
# (steps, and in production tokens and time). Say a loop without a step budget is a cost
# incident waiting to happen — and that you cap it in code, not in the prompt.
