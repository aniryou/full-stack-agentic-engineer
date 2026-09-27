"""The Mistral provider adapter, tested offline — no key, no network, and no SDK needed.

The pure conversion functions do the real work; ``MistralLLM`` is driven through an
injected client, and its two T0 stops (no client installed, no key) are checked with
the environment the tests control. One test constructs the real SDK client (no call is
made) and is skipped when the optional ``mistral`` extra is not installed.
"""
import json
import sys

import pytest

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


class RecordingClient:
    """Stands in for the SDK client: ``chat.complete(**kwargs)`` records the request and replies from a script."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = self

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


def test_injected_client_drives_the_loop_and_sees_mistral_shapes():
    from agentcore import Agent, tool
    from agentcore.mistral_llm import MistralLLM

    @tool
    def get_balance(account_id: str) -> dict:
        """Balance."""
        return {"balance": 1234.5}

    client = RecordingClient([
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "get_balance", "arguments": '{"account_id": "a1"}'}}]}}]},
        {"choices": [{"message": {"content": "Your balance is 1234.5.", "tool_calls": None}}]},
    ])
    llm = MistralLLM(model="some-model", client=client, temperature=0.2)
    r = Agent(llm, tools=[get_balance]).run("balance for a1?")
    assert r.done and "1234.5" in r.text and llm.calls == 2
    first, second = client.requests
    assert first["model"] == "some-model" and first["tool_choice"] == "auto" and first["temperature"] == 0.2
    assert first["tools"] == [{"type": "function", "function": get_balance.schema}]
    sent = next(m for m in second["messages"] if m["role"] == "assistant")    # our tool call went back as JSON text
    assert json.loads(sent["tool_calls"][0]["function"]["arguments"]) == {"account_id": "a1"}
    assert any(m["role"] == "tool" and m["tool_call_id"] == "c1" for m in second["messages"])


def test_request_leaves_out_tools_and_temperature_when_not_set():
    from agentcore.mistral_llm import MistralLLM

    llm = MistralLLM(client=RecordingClient([]), tool_choice="any")
    assert llm.request([{"role": "user", "content": "hi"}]) == {"model": "mistral-large-latest",
                                                               "messages": [{"role": "user", "content": "hi"}]}
    with_tools = llm.request([{"role": "user", "content": "hi"}], [{"name": "t", "description": "", "parameters": {}}])
    assert with_tools["tool_choice"] == "any" and "temperature" not in with_tools


def test_no_key_is_a_labelled_stop(monkeypatch):
    from agentcore.mistral_llm import MistralLLM, MistralUnavailable

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(MistralUnavailable, match="MISTRAL_API_KEY"):
        MistralLLM()


def test_no_client_installed_is_a_labelled_stop(monkeypatch):
    from agentcore.mistral_llm import MistralLLM, MistralUnavailable

    for name in ("mistralai", "mistralai.client"):
        monkeypatch.setitem(sys.modules, name, None)                        # makes the import fail
    with pytest.raises(MistralUnavailable, match=r"\[mistral\]"):
        MistralLLM(api_key="not-a-real-key")


def test_the_installed_client_constructs_without_a_call():
    from agentcore.mistral_llm import MistralLLM, MistralUnavailable, _sdk_client_class

    try:
        client_class = _sdk_client_class()
    except MistralUnavailable:
        pytest.skip("the optional mistral extra is not installed")
    llm = MistralLLM(api_key="not-a-real-key")                              # constructing makes no request
    assert isinstance(llm._client, client_class) and callable(llm._client.chat.complete)
