"""An MCP client (Primer §3.2) and the adapter that makes remote tools look local. Teaching subset.

``McpClient`` does the three things a host needs: it stamps every request with
the ``_meta`` block and the mirrored headers, it turns the two embedded
server→client shapes back into ordinary control flow (``input_required`` →
ask the user and re-send; ``task`` → poll, answer, wait), and it raises typed
errors for 401/403 that carry the ``WWW-Authenticate`` details the OAuth
flow (``agentlab.auth``) needs.

``McpToolset`` wraps ``tools/list`` entries as agentlab ``Tool`` objects so an
``LlmAgent`` cannot tell a remote tool from a local one — the property that
lets the same agent code run against a registry of servers.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Mapping

from ..agents.tools import ToolContext, ToolResult, ToolSpec
from ..auth.oauth import parse_www_authenticate
from . import protocol as p
from .protocol import McpError
from .transport import Transport

InputHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class Unauthorized(McpError):
    """HTTP 401: no or invalid token. ``challenge['resource_metadata']`` says where to start the OAuth chain."""

    @property
    def challenge(self) -> dict[str, str]:
        return parse_www_authenticate(self.headers.get("www-authenticate"))


class Forbidden(McpError):
    """HTTP 403: policy denial or ``insufficient_scope`` (then ``required_scope`` names what to step up to)."""

    @property
    def challenge(self) -> dict[str, str]:
        return parse_www_authenticate(self.headers.get("www-authenticate"))

    @property
    def required_scope(self) -> str | None:
        return self.challenge.get("scope") if self.challenge.get("error") == "insufficient_scope" else None


class TaskFailed(McpError):
    pass


def _error_for(status: int, headers: Mapping[str, str], payload: Mapping[str, Any]) -> McpError:
    err = payload.get("error") or {}
    code, message, data = err.get("code", p.INTERNAL_ERROR), err.get("message", f"HTTP {status}"), err.get("data")
    cls = Unauthorized if status == 401 else Forbidden if status == 403 else McpError
    return cls(code, message, data, http_status=status, headers=headers)


class McpClient:
    def __init__(self, transport: Transport, *, agent_identity: str | None = None, bearer_token: str | None = None,
                 capabilities: dict[str, Any] | None = None, protocol_version: str = p.PROTOCOL_VERSION,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, task_timeout_s: float = 30.0):
        self.transport = transport
        self.url = transport.base_url
        self.agent_identity = agent_identity
        self.bearer_token = bearer_token
        self.capabilities = p.client_capabilities() if capabilities is None else capabilities
        self.protocol_version = protocol_version
        self._sleep = sleep
        self.task_timeout_s = task_timeout_s
        self._next_id = 0

    # -- wire ---------------------------------------------------------------
    def build_request(self, method: str, params: dict[str, Any] | None = None) -> tuple[dict[str, str], bytes]:
        """Headers and body for one request. Exposed so a notebook can tamper with the mirrored headers."""
        params = dict(params or {})
        params["_meta"] = p.make_meta(self.capabilities, self.protocol_version)
        self._next_id += 1
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   p.HEADER_PROTOCOL_VERSION: self.protocol_version, p.HEADER_METHOD: method}
        if method == "tools/call":
            headers[p.HEADER_NAME] = str(params.get("name"))
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if self.agent_identity:
            headers[p.HEADER_AGENT_IDENTITY] = self.agent_identity
        return headers, json.dumps(p.request(self._next_id, method, params)).encode()

    async def send(self, headers: Mapping[str, str], body: bytes) -> Any:
        """POST prepared bytes; return the JSON-RPC result or raise the typed error the response maps to."""
        status, rheaders, rbody = await self.transport.request("POST", self.url, headers, body)
        payload = json.loads(rbody) if rbody else {}
        if status >= 400 or "error" in payload:
            raise _error_for(status, rheaders, payload)
        return payload.get("result")

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return await self.send(*self.build_request(method, params))

    async def fetch_json(self, url: str) -> dict[str, Any]:
        """GET a JSON document through the same transport (used for the protected-resource metadata)."""
        status, _, body = await self.transport.request("GET", url, {"Accept": "application/json"}, b"")
        if status != 200:
            raise McpError(p.INTERNAL_ERROR, f"GET {url} -> {status}", http_status=status)
        return json.loads(body)

    # -- methods --------------------------------------------------------------
    async def discover(self) -> dict[str, Any]:
        return await self.request("server/discover")

    async def list_tools(self) -> list[dict[str, Any]]:
        return (await self.request("tools/list"))["tools"]

    async def call_tool(self, name: str, args: dict[str, Any] | None = None,
                        on_input_required: InputHandler | None = None) -> dict[str, Any]:
        """Call a tool and return its final ``CallToolResult``, however many round trips that takes.

        ``on_input_required(key, params)`` answers an elicitation with a
        ``{"action": "accept", "content": {...}}`` dict (see ``protocol.accept``).
        """
        answers: dict[str, Any] = {}
        while True:
            params: dict[str, Any] = {"name": name, "arguments": dict(args or {})}
            if answers:
                params["inputResponses"] = answers
            result = await self.request("tools/call", params)
            kind = result.get("resultType")
            if kind == "input_required":
                answers.update(await self._answer(result["inputRequests"], on_input_required))
                continue                                  # MRTR: same call again, now with the answers
            if kind == "task":
                return await self._await_task(result["task"], on_input_required)
            return result

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return await self.request("tasks/get", {"taskId": task_id})

    async def update_task(self, task_id: str, input_responses: dict[str, Any]) -> dict[str, Any]:
        return await self.request("tasks/update", {"taskId": task_id, "inputResponses": input_responses})

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        return await self.request("tasks/cancel", {"taskId": task_id})

    # -- embedded interactions ---------------------------------------------------
    async def _answer(self, requests: Mapping[str, Any], handler: InputHandler | None) -> dict[str, Any]:
        if handler is None:
            raise McpError(p.INVALID_PARAMS, f"server needs input for {sorted(requests)} and no on_input_required handler was given")
        return {key: await handler(key, req.get("params", {})) for key, req in requests.items()}

    async def _await_task(self, task: dict[str, Any], handler: InputHandler | None) -> dict[str, Any]:
        """Poll at the server's suggested interval; answer input via tasks/update; stop on a terminal status."""
        task_id = task["taskId"]
        deadline = time.monotonic() + self.task_timeout_s
        while True:
            status = task["status"]
            if status == "completed":
                return task["result"]
            if status in ("failed", "cancelled"):
                raise TaskFailed(p.INTERNAL_ERROR, f"task {task_id} {status}: {(task.get('error') or {}).get('message', task.get('statusMessage'))}",
                                 data=task.get("error"))
            if status == "input_required":
                task = await self.update_task(task_id, await self._answer(task["inputRequests"], handler))
                continue
            if time.monotonic() > deadline:
                await self.cancel_task(task_id)
                raise TaskFailed(p.INTERNAL_ERROR, f"task {task_id} did not finish within {self.task_timeout_s}s; cancelled")
            await self._sleep(task.get("pollIntervalMs", 500) / 1000)
            task = await self.get_task(task_id)


# ------------------------------------------------------------- toolset
class RemoteTool:
    """A ``tools/list`` entry as a agentlab ``Tool``: same ``spec`` shape, ``run`` goes over MCP.

    ``destructiveHint`` becomes ``requires_confirmation`` so the local Runner
    pauses for approval before a destructive remote call — annotations are
    untrusted hints, so they may make the host *more* careful, never less.
    """

    def __init__(self, client: McpClient, descriptor: Mapping[str, Any], on_input_required: InputHandler | None = None):
        self.client = client
        self.on_input_required = on_input_required
        side_effect = p.side_effect_from(descriptor.get("annotations"))
        self.spec = ToolSpec(name=descriptor["name"], description=descriptor.get("description", ""),
                             input_schema=descriptor.get("inputSchema") or {"type": "object", "properties": {}},
                             side_effect=side_effect, requires_confirmation=bool((descriptor.get("annotations") or {}).get("destructiveHint")))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        try:
            result = await self.client.call_tool(self.spec.name, args, on_input_required=self.on_input_required)
        except Unauthorized as e:
            return ToolResult.failure("unauthorized", e.message, hint="The agent must obtain a token for this server first.")
        except Forbidden as e:
            hint = f"Step up to scope {e.required_scope}." if e.required_scope else "Tell the user this action is not permitted."
            return ToolResult.failure("forbidden", e.message, hint=hint)
        except McpError as e:
            return ToolResult.failure("mcp_error", e.message, retryable=e.http_status >= 500)
        latency = (time.perf_counter() - t0) * 1000
        if result.get("isError"):
            structured = result.get("structuredContent") or {}
            return ToolResult.failure(structured.get("error", "tool_failure"), structured.get("message") or result["content"][0]["text"],
                                      retryable=bool(structured.get("retryable")), hint=structured.get("hint"), latency_ms=latency)
        data = result.get("structuredContent") if "structuredContent" in result else result["content"][0]["text"]
        return ToolResult.success(data, latency_ms=latency)


class McpToolset:
    """``tools = await McpToolset(client).load()`` → a list an ``LlmAgent`` accepts as ``tools=``."""

    def __init__(self, client: McpClient, on_input_required: InputHandler | None = None):
        self.client = client
        self.on_input_required = on_input_required
        self.tools: list[RemoteTool] = []

    async def load(self) -> list[RemoteTool]:
        self.tools = [RemoteTool(self.client, d, self.on_input_required) for d in await self.client.list_tools()]
        return self.tools
