# %% [markdown]
# # 05 · Going live on Mistral
#
# The loop you built in notebooks 01–04 never mentions a provider. It only needs
# something with a `.generate(messages, tools) -> Response` method. `FakeLLM` is that
# for offline practice; `MistralLLM` is that for Mistral's API. **Swapping one for the
# other is one line** — that is the whole point of keeping the loop provider-agnostic.
#
# This notebook shows the swap and the small amount of translation the adapter does
# between our shapes and Mistral's (`agentcore/mistral_llm.py`, the lab's only
# provider-specific code; `docs/MISTRAL.md` is the reference). It runs **offline** at T0:
# we exercise the pure conversion functions and drive the loop with a fake Mistral-shaped
# client. The final cell calls the real API only when the optional `mistral` extra is
# installed and `MISTRAL_API_KEY` is set; otherwise it stops with a labelled message,
# which is not a failure.

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
# ## Exercise 5.3 — route a turn to the cheapest model that clears the bar
# Part of the deployment story is choosing the *cheapest model that clears the bar* —
# and, when the data must stay in your environment, only a model whose weights you can
# run yourself. `CATALOGUE` is a snapshot of the table in `docs/MISTRAL.md` (list prices
# per million tokens, 2026-09-19, illustrative — verify before quoting). `open_weight`
# here means "you may run it yourself without a separate licence": Medium's weights are
# published, but its modified MIT licence asks for a commercial licence above a revenue
# threshold (`docs/MISTRAL.md`), so the snapshot marks it false. The strings do not
# matter; the rule does.
#
# Write `pick_model(catalogue, need, self_host=False, input_tokens=5_000, output_tokens=300)`
# that returns the **name** of the model whose `can` set contains `need` (and, if
# `self_host`, whose `open_weight` is true) with the lowest cost for one call of that
# size — `(input_tokens × input + output_tokens × output) / 1e6` — breaking a tie by
# name, or `None` when no model qualifies. The check runs your rule against the snapshot
# and against a few hundred random catalogues.

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
# This cell calls the live API when the `mistral` extra is installed
# (`pip install -e ".[mistral]"` in this lab, or `pip install mistralai`) and
# `MISTRAL_API_KEY` is set. Without them it prints a labelled stop and ends cleanly, so
# the notebook still runs in CI and offline. A live call is billed per token.

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
# The deployment question is rarely "which model is smartest" — it is *which model
# clears the bar at the lowest cost and the right deployment posture*. Several Mistral
# models are **open-weight (Apache-2.0)** — Large 3, Small 4, the Ministral 3 family,
# 2026-09-19 (verify) — so the same agent can run on weights in your own VPC or on-prem
# when the data is regulated or must stay in one jurisdiction. In a design review, say:
# start on the managed API with the cheapest model that passes your evals for the tool
# loop, route the few turns that need step-by-step reasoning to a reasoning model, and
# self-host an open-weight model when the data cannot leave your environment — the
# exercise 5.3 rule, with your eval results as the `can` column. The loop does not change;
# only the model object does.
