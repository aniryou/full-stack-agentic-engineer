# %% [markdown]
# # 01 · The agent loop, from scratch
#
# An "agent" sounds complicated. But it is only a loop around a model call:
#
# 1. Show the model the conversation and the tools that it can use.
# 2. The model replies with a **final answer**, or with one or more **tool calls**.
# 3. If the reply has tool calls, run them and add the results. Then go back to step 1.
# 4. Stop at a final answer, or when the loop has used all of the step budget.
#
# That is all of it. In this notebook, you build the loop yourself. Then you compare it
# with the packaged `Agent`. Everything runs offline against a scripted fake model.

# %%
from agentcore import FakeLLM, call, calls, text, tool
from agentcore.tools import tool_message

# A scripted model: turn 1 asks for a tool, turn 2 gives the final answer.
llm = FakeLLM([call("get_time", city="Singapore"), "It is 4:00pm in Singapore."])
r1 = llm.generate([{"role": "user", "content": "what time is it in SG?"}])
print("turn 1 →", "tool call:", [(t.name, t.args) for t in r1.tool_calls])
r2 = llm.generate([{"role": "user", "content": "..."}])
print("turn 2 →", "text:", r2.text)

# %% [markdown]
# ## A tool to call
# `@tool` reads the function signature. It uses the signature to make the schema that
# the model sees. It validates the arguments before the tool runs. It also puts the
# result in a wrapper, `{"ok": ..., "data"/"error": ...}`. Notebook 02 is about tools.
# Here, you need only one tool.

# %%
@tool
def get_time(city: str) -> dict:
    """Return the current time in a city."""
    return {"city": city, "time": "16:00"}

print("schema:", get_time.schema)
print("run   :", get_time.run({"city": "Singapore"}))

# %% [markdown]
# ## Exercise 1.1 — the termination check
#
# The loop must know when to stop. Write `is_final(resp)`. Return `True` when the
# response of the model is a final answer (it has no tool calls). Return `False` when
# the model wants tools.

# %% exercise
def is_final(resp) -> bool:
    ### BEGIN SOLUTION
    return not resp.tool_calls
    ### END SOLUTION

# %% check
assert is_final(text("done")) is True
assert is_final(calls(call("get_time", city="X"))) is False
print("✅ is_final works")

# %% [markdown]
# ## Exercise 1.2 — run one round of tool calls
#
# Write `run_tools(tool_calls, registry)`. For each requested call, the function finds
# the tool in `registry` (a `{name: Tool}` dict) and runs it. It returns a list of
# **tool messages** (use `tool_message(tc, result)`). If a tool name is unknown, add a
# tool message whose result is `{"ok": False, "error": "unknown_tool", "message": ...}`.
# The loop must never crash only because the model hallucinated a tool name.

# %% exercise
def run_tools(tool_calls, registry) -> list:
    messages = []
    ### BEGIN SOLUTION
    for tc in tool_calls:
        tool = registry.get(tc.name)
        if tool is None:
            result = {"ok": False, "error": "unknown_tool", "message": f"no tool named {tc.name!r}"}
        else:
            result = tool.run(tc.args)
        messages.append(tool_message(tc, result))
    ### END SOLUTION
    return messages

# %% check
registry = {"get_time": get_time}
msgs = run_tools([call("get_time", city="SG"), call("ghost")], registry)
assert msgs[0]["role"] == "tool" and '"ok": true' in msgs[0]["content"].lower()
assert "unknown_tool" in msgs[1]["content"]
print("✅ run_tools works:", [m["content"] for m in msgs])

# %% [markdown]
# ## Exercise 1.3 — the loop itself
#
# Put the parts together. Write `run_loop(llm, registry, user_message, max_steps=6)`:
#
# * Start `messages` with a system message, then the user message.
# * Do these steps up to `max_steps` times. Call
#   `llm.generate(messages, tools=schemas)`. Add the response as a message
#   (`resp.as_message()`). If the response is final, return `(resp.text, messages)`. If
#   not, add the tool messages from `run_tools` and continue.
# * If the loop uses all of the budget, return `("(stopped: max steps)", messages)`.

# %% exercise
def run_loop(llm, registry, user_message, max_steps=6):
    schemas = [t.schema for t in registry.values()]
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}]
    ### BEGIN SOLUTION
    for _ in range(max_steps):
        resp = llm.generate(messages, tools=schemas)
        messages.append(resp.as_message())
        if is_final(resp):
            return resp.text, messages
        messages += run_tools(resp.tool_calls, registry)
    return "(stopped: max steps)", messages
    ### END SOLUTION

# %% check
llm = FakeLLM([call("get_time", city="Singapore"), "It is 4:00pm in Singapore."])
answer, messages = run_loop(llm, {"get_time": get_time}, "what time is it in SG?")
assert answer == "It is 4:00pm in Singapore."
assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
# a runaway model that never answers must stop at the budget
runaway = FakeLLM(policy=lambda m, t: calls(call("get_time", city="X")))
ans2, _ = run_loop(runaway, {"get_time": get_time}, "loop", max_steps=3)
assert ans2 == "(stopped: max steps)"
# the budget counts model calls: max_steps=3 means exactly 3 calls, never a 4th
assert runaway.call_count == 3, f"max_steps=3 but the loop called the model {runaway.call_count} times"
print("✅ run_loop works — you just built an agent")

# %% [markdown]
# ## The packaged version
# `agentcore.Agent` is the same loop that you wrote, with two more conveniences: a
# `Result` object and an `on_confirm` hook (Notebook 04). Read `agentcore/agent.py`. Now
# it is possible that the code looks familiar to you.

# %%
from agentcore import Agent
r = Agent(FakeLLM([call("get_time", city="Singapore"), "It's 4pm."]), tools=[get_time]).run("time in SG?")
print(r.transcript())
print("\nsteps:", r.steps, "| done:", r.done, "| answer:", r.text)

# %% [markdown]
# ## The one-minute version
# When someone asks *"how does an agent work?"*, draw this loop with a budget. Show the
# model call, the tool execution, the results added to the messages, and the next round.
# Then say that **the harness sets the bounds** of the intelligence. The model only
# proposes. Your code validates, runs, and decides when to stop.
