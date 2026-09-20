"""mistral_model.py — the FakeModel swap: a Mistral model decides the next step.

    pip install mistralai            # export MISTRAL_API_KEY=...
    agent = Agent(store, queue, MistralModel(tools=TOOL_SPECS), tools={"charge": pay})

The adapter does one job: turn the run's journal into a chat (goal, then every
earlier decision as an assistant tool call with its recorded result) and ask
the model, via function calling, what to do next. It returns the same dict
shape FakeModel does — {"tool": name, "args": {...}} or {"final": text} — so
durable.py does not change.

Two things worth knowing about the Mistral API here:
  * tool_call ids must be exactly 9 alphanumeric characters, so the journal's
    idempotency key ("run_ab12cd:3") is hashed to a stable 9-char id;
  * the SDK is a namespace package: `from mistralai.client import Mistral`.
Model aliases: mistral-medium-latest (default), mistral-small-latest (cheap steps),
mistral-large-latest — check https://docs.mistral.ai/getting-started/models/models_overview.
"""

import hashlib
import json
import os

DEFAULT_INSTRUCTIONS = (
    "You are an operations agent working one step at a time. Use a tool when the goal needs one; "
    "when the goal is met, reply with a short plain-text summary and no tool call. "
    "Never repeat a tool call whose result you already have."
)


def tool_call_id(key):
    """Mistral requires 9 alphanumeric chars; derive them deterministically from the journal key."""
    return hashlib.sha1(key.encode()).hexdigest()[:9]


def to_messages(goal, journal, instructions=DEFAULT_INSTRUCTIONS):
    """Journal → chat messages. In the journal a tool decision is always followed by its intent,
    so each pair becomes an assistant tool call plus the matching tool result. This is the whole
    prompt on a retry: recorded facts, never the model's memory."""
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": goal}]
    for i, step in enumerate(journal):
        if step["type"] != "decision" or "final" in step:
            continue
        intent = journal[i + 1]                       # the intent journaled right after this decision
        cid = tool_call_id(intent["key"])
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": step["tool"], "arguments": json.dumps(step["args"])}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "name": step["tool"],
                         "content": json.dumps(intent.get("result") if intent["done"] else {"pending": True})})
    return messages


def function_tools(tools):
    """{"charge": {"description", "parameters"}} → the tools list the API expects."""
    return [{"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
            for name, spec in tools.items()]


class MistralModel:
    def __init__(self, tools, model=None, client=None, instructions=DEFAULT_INSTRUCTIONS):
        self.tools, self.model, self.instructions = tools, model or os.environ.get("MISTRAL_MODEL", "mistral-medium-latest"), instructions
        self.calls = 0
        if client is None:
            from mistralai.client import Mistral          # lazy: tests use a fake client
            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.client = client

    def decide(self, goal, journal):
        self.calls += 1
        resp = self.client.chat.complete(model=self.model, messages=to_messages(goal, journal, self.instructions),
                                         tools=function_tools(self.tools), tool_choice="auto", parallel_tool_calls=False)
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = call.function.arguments
            return {"tool": call.function.name, "args": json.loads(args) if isinstance(args, str) else dict(args)}
        text = msg.content if isinstance(msg.content, str) else "".join(getattr(c, "text", "") for c in msg.content or [])
        return {"final": text.strip()}


# The tool schema the demo and tests use. `key` is injected by the agent, so it is not exposed to the model.
TOOL_SPECS = {
    "charge": {
        "description": "Charge the customer's card. Requires human approval before it executes.",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number", "description": "amount in SGD"}},
                       "required": ["amount"]},
    },
}
