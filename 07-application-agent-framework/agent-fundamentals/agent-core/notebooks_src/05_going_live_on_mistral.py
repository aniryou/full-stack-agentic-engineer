# %% [markdown]
# # 05 · Going live on Mistral
#
# The loop that you built in notebooks 01–04 never mentions a provider. It only needs an
# object with a `.generate(messages, tools) -> Response` method. `FakeLLM` is that
# object for offline practice. `MistralLLM` is that object for the API of Mistral. **To
# replace one with the other, you change one line.** That is the full reason to keep the
# loop provider-agnostic.
#
# This notebook shows the swap. It also shows the small translation that the adapter
# does between our shapes and the shapes of Mistral. The adapter is
# `agentcore/mistral_llm.py`, the only provider-specific code of the lab, and
# `docs/MISTRAL.md` is the reference. The notebook runs **offline** at T0. It runs the
# pure conversion functions, and it drives the loop with a fake Mistral-shaped client.
#
# The last cell calls the real API only when the optional `mistral` extra is installed
# and `MISTRAL_API_KEY` is set. If one of the two conditions is false, the cell stops
# with a labelled message. This stop is not a failure.

# %%
import json
from agentcore import Agent, tool
from agentcore.mistral_llm import parse_response, to_mistral_messages, to_mistral_tools

# %% [markdown]
# ## 1. Tools → Mistral's function shape
# Our tool schema is `{"name","description","parameters"}`. Mistral wants each tool in a
# wrapper, `{"type":"function","function": {...}}`, as the OpenAI shape does.

# %%
@tool
def get_balance(account_id: str) -> dict:
    """Return the balance for an account."""
    return {"account_id": account_id, "balance": 1234.5}

print(json.dumps(to_mistral_tools([get_balance.schema]), indent=2))

# %% [markdown]
# ## 2. Messages → Mistral's format
# System, user and tool messages already match the format of Mistral. Only our
# **assistant tool calls** need a new shape. We keep `args` as a dict, but Mistral wants
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
# A Mistral reply is `response.choices[0].message`, with `.content` and `.tool_calls`.
# Each `tool_call` has `.id` and `.function.name` / `.function.arguments`. The adapter
# changes that reply back into the `Response` that the loop understands.

# %%
fake_reply = {"choices": [{"message": {"content": "", "tool_calls": [
    {"id": "call_1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}]}}]}
r = parse_response(fake_reply)
print("parsed →", [(tc.name, tc.args, tc.id) for tc in r.tool_calls])

# %% [markdown]
# ## 4. The whole loop, offline, against a Mistral-shaped client
# The next cell shows that the swap works, and it costs no token. It uses a stand-in
# that returns Mistral-shaped dicts. Look at `Agent`: it is unchanged from notebook 04.
# Only the model object is different.

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
# Write `my_to_mistral_tools(schemas)`. Do not call the `to_mistral_tools` of the
# library. Your function puts each of our tool schemas in a wrapper,
# `{"type": "function", "function": <schema>}`. For an empty list or no list, it returns
# `None`.

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
# Write `first_tool_call(reply)`. The argument is a Mistral-shaped reply dict. The
# function returns `(name, args_dict)` for the first tool call. It parses the JSON
# `arguments` string. This function does the main part of what `parse_response` does.

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
# ## Exercise 5.3 — route a turn to the cheapest model that clears the bar
# One part of the deployment story is to select the *cheapest model that clears the
# bar*. When the data must stay in your environment, the model must also have weights
# that you can run yourself. `CATALOGUE` is a snapshot of the table in
# `docs/MISTRAL.md`. Its prices are list prices per million tokens, from 2026-09-19, and
# illustrative. Make sure that they are correct before you quote them.
#
# Here, `open_weight` means "you can run it yourself without a separate licence".
# Mistral publishes the weights of Medium. But the modified MIT licence of Medium asks
# for a commercial licence above a revenue threshold (`docs/MISTRAL.md`). Thus the
# snapshot marks Medium false. The strings are not important. The rule is important.
#
# Write
# `pick_model(catalogue, need, self_host=False, input_tokens=5_000, output_tokens=300)`.
# It returns the **name** of the model that agrees with these conditions:
#
# * The `can` set of the model contains `need`.
# * If `self_host` is true, the `open_weight` of the model is true.
# * The model has the lowest cost of these models for one call of that size:
#   `(input_tokens × input + output_tokens × output) / 1e6`.
# * If two models have the same cost, the function compares their names to break the tie.
#
# If no model qualifies, the function returns `None`. The check runs your rule against
# the snapshot and against a few hundred random catalogues.

# %%
CATALOGUE = [   # $ per 1M tokens (input, output); `can`: what the model is good enough at here
    {"name": "mistral-large-latest",    "input": 0.50, "output": 1.50, "open_weight": True,  "can": {"chat", "tools", "long_context"}},
    {"name": "mistral-medium-latest",   "input": 1.50, "output": 7.50, "open_weight": False, "can": {"chat", "tools", "code"}},
    {"name": "mistral-small-latest",    "input": 0.15, "output": 0.60, "open_weight": True,  "can": {"chat", "tools"}},
    {"name": "magistral-medium-latest", "input": 2.00, "output": 5.00, "open_weight": False, "can": {"chat", "reasoning"}},
    {"name": "codestral-latest",        "input": 0.30, "output": 0.90, "open_weight": False, "can": {"code"}},
    {"name": "ministral-8b-2512",       "input": 0.15, "output": 0.15, "open_weight": True,  "can": {"chat"}},
]

# %% exercise
def pick_model(catalogue, need, self_host=False, input_tokens=5_000, output_tokens=300):
    ### BEGIN SOLUTION
    ok = [m for m in catalogue if need in m["can"] and (m["open_weight"] or not self_host)]
    if not ok:
        return None
    cost = lambda m: (input_tokens * m["input"] + output_tokens * m["output"]) / 1e6  # noqa: E731
    return min(ok, key=lambda m: (cost(m), m["name"]))["name"]
    ### END SOLUTION

# %% check
import random as _random

def _check_pick(catalogue, need, self_host, i, o):
    got = pick_model(catalogue, need, self_host=self_host, input_tokens=i, output_tokens=o)
    ok = [m for m in catalogue if need in m["can"] and (m["open_weight"] or not self_host)]
    if not ok:
        assert got is None, f"no model can do {need!r} (self_host={self_host}), expected None, got {got!r}"
        return
    by_name = {m["name"]: m for m in catalogue}
    assert got in by_name, f"{got!r} is not a model name in the catalogue"
    m = by_name[got]
    assert need in m["can"], f"{got} cannot do {need!r}"
    assert m["open_weight"] or not self_host, f"{got} has no open weights, but the data must stay in your environment"
    cost = lambda x: (i * x["input"] + o * x["output"]) / 1e6  # noqa: E731
    cheapest = min((cost(x), x["name"]) for x in ok)
    assert (cost(m), got) == cheapest, f"{cheapest[1]} clears the bar for {need!r} at ${cheapest[0]:.6f} a call, cheaper than {got} (${cost(m):.6f})"

for need in ("chat", "tools", "code", "reasoning", "long_context", "vision"):
    for self_host in (False, True):
        for i, o in ((5_000, 300), (500, 4_000)):
            _check_pick(CATALOGUE, need, self_host, i, o)
_rng = _random.Random(5)
_skills = ["chat", "tools", "code", "reasoning"]
for _ in range(300):
    cat = [{"name": f"m{k}", "input": round(_rng.uniform(0.01, 3), 2), "output": round(_rng.uniform(0.01, 10), 2),
            "open_weight": _rng.random() < 0.5, "can": set(_rng.sample(_skills, _rng.randint(1, 3)))}
           for k in range(_rng.randint(0, 6))]
    _check_pick(cat, _rng.choice(_skills), _rng.random() < 0.5, _rng.choice([100, 5_000, 50_000]), _rng.choice([10, 300, 8_000]))
print("✅ routing rule holds on the snapshot and on 300 random catalogues")
print("   tools, managed API:", pick_model(CATALOGUE, "tools"), "| tools, self-hosted:", pick_model(CATALOGUE, "tools", self_host=True),
      "| reasoning, self-hosted:", pick_model(CATALOGUE, "reasoning", self_host=True))

# %% [markdown]
# ## 5. Run it for real (only if you have a key)
# This cell calls the live API when two conditions are true. First, the `mistral` extra
# is installed (`pip install -e ".[mistral]"` in this lab, or `pip install mistralai`).
# Second, `MISTRAL_API_KEY` is set. If one of the two conditions is false, the cell
# prints a labelled stop and ends with no error. Thus the notebook still runs in CI and offline. Mistral bills a live
# call per token.

# %%
import os
from agentcore.mistral_llm import MistralLLM, MistralUnavailable

try:
    llm = MistralLLM(model="mistral-large-latest")
except MistralUnavailable as why:
    print("[live call skipped — T0 path] everything above ran offline against the fake client.")
    print("   reason:", why)
else:
    agent = Agent(llm, tools=[get_balance], instruction="You are a bank assistant. Use tools for every fact.")
    live = agent.run("what's the balance on account a1?")
    print(live.transcript())
    print("\nlive answer:", live.text)

# %% [markdown]
# ## The one-minute version (Mistral)
# The deployment question is rarely "which model is smartest". It is *which model clears
# the bar at the lowest cost and the correct deployment posture*. Several Mistral models
# are **open-weight (Apache-2.0)**: Large 3, Small 4 and the Ministral 3 family,
# 2026-09-19 (verify). Thus, when the data is regulated or must stay in one
# jurisdiction, the same agent can run on weights in your own VPC or on-prem.
#
# In a design review, say this:
#
# * Start on the managed API with the cheapest model that passes your evals for the tool
#   loop.
# * Route the few turns that need step-by-step reasoning to a reasoning model.
# * Self-host an open-weight model when the data cannot leave your environment.
#
# This is the rule of exercise 5.3, with your eval results as the `can` column. The loop
# does not change. Only the model object changes.
