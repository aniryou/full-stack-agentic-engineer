# %% [markdown]
# # 05 · Going live on Mistral
#
# The loop you built in notebooks 01–04 never mentions a provider. It only needs
# something with a `.generate(messages, tools) -> Response` method. `FakeLLM` is that
# for offline practice; `MistralLLM` is that for Mistral's API. **Swapping one for the
# other is one line** — that is the whole point of keeping the loop provider-agnostic.
#
# This notebook shows the swap and the small amount of translation the adapter does
# between our shapes and Mistral's. It runs **offline**: we exercise the pure
# conversion functions and drive the loop with a fake Mistral-shaped client. A final,
# guarded cell calls the real API only if you have set `MISTRAL_API_KEY`.

# %%
import json
from agentcore import Agent, tool
from agentcore.mistral_llm import parse_response, to_mistral_messages, to_mistral_tools

# %% [markdown]
# ## 1. Tools → Mistral's function shape
# Our tool schema is `{"name","description","parameters"}`. Mistral (like the OpenAI
# shape) wants each tool wrapped as `{"type":"function","function": {...}}`.

# %%
@tool
def get_balance(account_id: str) -> dict:
    """Return the balance for an account."""
    return {"account_id": account_id, "balance": 1234.5}

print(json.dumps(to_mistral_tools([get_balance.schema]), indent=2))

# %% [markdown]
# ## 2. Messages → Mistral's format
# System, user and tool messages already match Mistral. Only our **assistant tool
# calls** need reshaping: we carry `args` as a dict, Mistral wants
# `function.arguments` as a JSON *string*.

# %%
demo = [
    {"role": "user", "content": "balance for a1?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_balance", "args": {"account_id": "a1"}}]},
    {"role": "tool", "name": "get_balance", "tool_call_id": "c1", "content": '{"balance": 1234.5}'},
]
converted = to_mistral_messages(demo)
print("assistant tool call becomes:", json.dumps(converted[1]["tool_calls"][0], indent=2))

# %% [markdown]
# ## 3. Mistral's reply → our `Response`
# A Mistral reply is `response.choices[0].message` with `.content` and `.tool_calls`
# (each `tool_call` has `.id` and `.function.name` / `.function.arguments`). The
# adapter turns that back into the `Response` the loop understands.

# %%
fake_reply = {"choices": [{"message": {"content": "", "tool_calls": [
    {"id": "call_1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}]}}]}
r = parse_response(fake_reply)
print("parsed →", [(tc.name, tc.args, tc.id) for tc in r.tool_calls])

# %% [markdown]
# ## 4. The whole loop, offline, against a Mistral-shaped client
# To prove the swap works without spending a token, here is a stand-in that returns
# Mistral-shaped dicts. Notice `Agent` is unchanged from notebook 04 — only the model
# object differs.

# %%
class FakeMistralClient:
    """Returns Mistral-shaped replies from a script (stands in for the real API)."""
    model_name = "mistral-large-latest"
    def __init__(self, replies): self.replies, self.i = replies, 0
    def generate(self, messages, tools=None):
        reply = self.replies[self.i]; self.i += 1
        return parse_response(reply)

replies = [
    {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}]}}]},
    {"choices": [{"message": {"content": "Your balance is SGD 1,234.50.", "tool_calls": None}}]},
]
result = Agent(FakeMistralClient(replies), tools=[get_balance]).run("what's my balance on a1?")
print(result.transcript())
print("\nanswer:", result.text)

# %% [markdown]
# ## Exercise 5.1 — implement the tool converter
# Without calling the library's `to_mistral_tools`, write `my_to_mistral_tools(schemas)`
# that wraps each of our tool schemas as `{"type": "function", "function": <schema>}`,
# returning `None` for an empty or missing list.

# %% exercise
def my_to_mistral_tools(schemas):
    ### BEGIN SOLUTION
    if not schemas:
        return None
    return [{"type": "function", "function": s} for s in schemas]
    ### END SOLUTION

# %% check
assert my_to_mistral_tools([get_balance.schema]) == to_mistral_tools([get_balance.schema])
assert my_to_mistral_tools([]) is None
print("✅ tool converter matches the library")

# %% [markdown]
# ## Exercise 5.2 — parse a Mistral tool-call reply
# Write `first_tool_call(reply)` that, given a Mistral-shaped reply dict, returns
# `(name, args_dict)` for the first tool call — parsing the JSON `arguments` string.
# (This is the heart of what `parse_response` does.)

# %% exercise
def first_tool_call(reply):
    ### BEGIN SOLUTION
    tc = reply["choices"][0]["message"]["tool_calls"][0]
    return tc["function"]["name"], json.loads(tc["function"]["arguments"])
    ### END SOLUTION

# %% check
name, args = first_tool_call(fake_reply)
assert name == "get_balance" and args == {"account_id": "a1"}
print("✅ parsed the tool call:", name, args)

# %% [markdown]
# ## Exercise 5.3 — pick the right model
# Part of the Mistral deployment story is choosing the *cheapest model that clears the
# bar*. Map each need to a model string (see `docs/MISTRAL.md`): fill in `CHOICES`.
#
# * `"agentic"`   → the flagship, best at multi-step tool use
# * `"reasoning"` → step-by-step reasoning
# * `"cheap_edge"`→ open-weight, runs on-device / cheapest
# * `"coding"`    → code generation and autocomplete

# %% exercise
CHOICES = {}
### BEGIN SOLUTION
CHOICES = {
    "agentic": "mistral-large-latest",
    "reasoning": "magistral-medium-latest",
    "cheap_edge": "ministral-8b-latest",
    "coding": "codestral-latest",
}
### END SOLUTION

# %% check
assert CHOICES["agentic"] == "mistral-large-latest"
assert CHOICES["reasoning"].startswith("magistral")
assert CHOICES["cheap_edge"].startswith("ministral")
assert CHOICES["coding"].startswith("codestral")
print("✅ model choices:", CHOICES)

# %% [markdown]
# ## 5. Run it for real (only if you have a key)
# This cell calls the live API when `MISTRAL_API_KEY` is set, and otherwise prints how
# to enable it — so the notebook still runs clean in CI and offline.

# %%
import os
if os.environ.get("MISTRAL_API_KEY"):
    from agentcore.mistral_llm import MistralLLM
    agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[get_balance],
                  instruction="You are a bank assistant. Use tools for every fact.")
    live = agent.run("what's the balance on account a1?")
    print(live.transcript())
    print("\nlive answer:", live.text)
else:
    print("No MISTRAL_API_KEY set — skipping the live call.")
    print("To run it: pip install mistralai ; export MISTRAL_API_KEY=... ; re-run this cell.")

# %% [markdown]
# ## The one-minute version (Mistral)
# The deployment question is rarely "which model is smartest" — it is *which model
# clears the bar at the lowest cost and the right deployment posture*. The small Mistral
# models are **open-weight (Apache-2.0)**, so the same agent can run on weights in your
# own VPC or on-prem when the data is regulated or must stay in one jurisdiction. In a
# design review, say: start on `la Plateforme` with `mistral-large` for the hard agent,
# route routine turns to `mistral-small`, reach for `magistral` only where reasoning
# pays, and self-host `small`/`ministral` when the data cannot leave your environment.
