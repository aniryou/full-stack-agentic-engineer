"""The Mistral adapter's conversions, tested offline — no SDK, no key, no network.

We test the four pure functions that do the real work; the thin ``MistralLLM``
class around them just calls the SDK, which we don't exercise here.
"""
import json

from agentcore.mistral_llm import parse_response, to_mistral_messages, to_mistral_tools


def test_tools_wrap_into_function_shape():
    schemas = [{"name": "get_balance", "description": "Balance.",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}]
    out = to_mistral_tools(schemas)
    assert out == [{"type": "function", "function": schemas[0]}]
    assert to_mistral_tools(None) is None and to_mistral_tools([]) is None


def test_assistant_tool_calls_become_json_arguments():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_balance", "args": {"id": "a1"}}]},
        {"role": "tool", "name": "get_balance", "tool_call_id": "c1", "content": '{"ok": true}'},
    ]
    out = to_mistral_messages(messages)
    # system/user/tool pass through unchanged
    assert out[0] == messages[0] and out[1] == messages[1] and out[3] == messages[3]
    # assistant tool call: args dict -> JSON string under function.arguments
    tc = out[2]["tool_calls"][0]
    assert tc["type"] == "function" and tc["id"] == "c1"
    assert tc["function"]["name"] == "get_balance"
    assert json.loads(tc["function"]["arguments"]) == {"id": "a1"}


def test_parse_text_response():
    resp = {"choices": [{"message": {"content": "Your balance is 1234.5.", "tool_calls": None}}]}
    r = parse_response(resp)
    assert r.text == "Your balance is 1234.5." and r.tool_calls == []


def test_parse_tool_call_response():
    resp = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "call_1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}
    ]}}]}
    r = parse_response(resp)
    assert r.text is None                       # content "" alongside tool calls → no final text yet
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.name == "get_balance" and tc.args == {"account_id": "a1"} and tc.id == "call_1"


def test_parsed_response_drives_the_loop_with_no_mistral_installed():
    # Prove the adapter's output type is exactly what Agent expects, using a fake
    # client that returns Mistral-shaped dicts — so the whole loop runs offline.
    from agentcore import Agent, tool
    from agentcore.mistral_llm import parse_response

    @tool
    def get_balance(account_id: str) -> dict:
        """Balance."""
        return {"balance": 1234.5}

    scripted = [
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}]}}]},
        {"choices": [{"message": {"content": "Your balance is 1234.5.", "tool_calls": None}}]},
    ]

    class FakeMistral:                      # stands in for MistralLLM without the SDK
        model_name = "mistral-large-latest"
        def __init__(self): self.i = 0
        def generate(self, messages, tools=None):
            r = parse_response(scripted[self.i]); self.i += 1
            return r

    r = Agent(FakeMistral(), tools=[get_balance]).run("balance for a1?")
    assert r.done and "1234.5" in r.text
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "tool", "assistant"]
