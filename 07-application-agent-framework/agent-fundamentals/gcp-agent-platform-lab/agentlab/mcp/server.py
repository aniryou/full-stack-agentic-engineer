"""An MCP server over agentlab tools (notebook 05). Teaching subset, not a conformant implementation.

Read top to bottom:

1. how a tool asks the user something (``NeedsInput``) or hands back a durable
   handle (``LongRunning``) — the two embedded server→client interactions;
2. ``McpServer.handle_jsonrpc``: authenticate → parse → mirrored-header check →
   version check → dispatch. Stateless: nothing survives between requests
   except tool state and the task store;
3. the six methods (``server/discover``, ``tools/list``, ``tools/call``,
   ``tasks/get|update|cancel``);
4. ``_Tasks``: background execution on ``TaskRecord`` / ``TaskStore``.

Authorization is the MCP subset of OAuth 2.1: a bearer token whose audience
is this server's canonical URL, scopes per tool, and a 401 that tells the
client where the protected-resource metadata lives. The inbound token is
never forwarded: a tool that needs a downstream credential exchanges it
(RFC 8693) so the backend sees the user's ``sub`` and this server in ``act``;
passing it through would let a token minted for *this* audience be replayed
against another, which is exactly what audience binding exists to prevent.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlparse

from ..agents.state import TaskRecord, TaskStatus, TaskStore
from ..agents.tools import Identity, Tool, ToolContext, ToolResult
from ..auth.oauth import OAuthError, WELL_KNOWN_PRM, identity_from_claims
from . import protocol as p
from .protocol import JsonRpcRequest, McpError

CONFIRM_SCHEMA = {"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]}


# ------------------------------------------------ what a tool can signal
class NeedsInput(BaseException):
    """Raise from a tool to ask the user something; the call resumes when the client re-sends with the answer.

    ``BaseException`` on purpose: ``FunctionTool.run`` turns every ``Exception``
    into a ``tool_failure`` result so stack traces never reach the model. A
    request for input is control flow, not a failure, and has to cross that
    boundary — the same reason ``asyncio.CancelledError`` is a ``BaseException``.
    """

    def __init__(self, key: str, message: str, requested_schema: dict[str, Any] | None = None):
        super().__init__(message)
        self.key = key
        self.message = message
        self.requested_schema = requested_schema or CONFIRM_SCHEMA

    def as_input_requests(self) -> dict[str, Any]:
        return {self.key: p.elicitation_request(self.message, self.requested_schema)}


@dataclass
class LongRunning:
    """Return this from a tool to say 'run ``work`` in the background and give the client a task'.

    ``work(ctx)`` receives a ``ToolContext`` (so it can read ``input_responses``
    and report ``progress``) and may raise ``NeedsInput`` mid-flight; the task
    then waits in ``input_required`` and ``work`` is re-invoked with the answer.
    Work must therefore be idempotent up to its elicitation point — which is
    what ``TaskRecord.completed_steps`` is for in a real implementation.
    """
    work: Callable[[ToolContext], Awaitable[Any] | Any]
    status_message: str = "accepted"


def input_responses(ctx: ToolContext) -> dict[str, Any]:
    """Answers the client sent back (``params.inputResponses``), keyed like the requests."""
    return dict(ctx.extras.get("mcp", {}).get("input_responses") or {})


def confirmed(ctx: ToolContext, key: str = "confirm") -> bool | None:
    """``True``/``False`` once the user answered the boolean field named ``key``, ``None`` if not asked yet.

    Declining or cancelling counts as ``False``: only an explicit accept is a yes.
    """
    responses = input_responses(ctx)
    if key not in responses:
        return None
    answer = p.accepted_content(responses[key])
    return bool(answer and answer.get(key))


def progress(ctx: ToolContext, message: str) -> None:
    """Update the task's ``statusMessage`` (no-op outside a task)."""
    hook = ctx.extras.get("mcp", {}).get("progress")
    if hook is not None:
        hook(message)


# --------------------------------------------------------------- plumbing
@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class _Call:
    """Everything needed to (re-)execute one tool call, in the foreground or as a task."""
    tool: Tool
    args: dict[str, Any]
    identity: Identity | None
    agent: str
    input_responses: dict[str, Any] = field(default_factory=dict)
    work: Callable[[ToolContext], Any] | None = None     # set once a LongRunning marker was returned

    def context(self, task_id: str | None = None, progress_hook: Callable[[str], None] | None = None) -> ToolContext:
        return ToolContext(session_id=f"mcp:{task_id or 'call'}", tenant=self.identity.tenant if self.identity else "default",
                           user=self.identity, agent_name=self.agent,
                           extras={"mcp": {"input_responses": self.input_responses, "task_id": task_id, "progress": progress_hook}})


TokenVerifier = Callable[[str], Mapping[str, Any]]


class McpServer:
    """Serves agentlab tools over MCP. Build once, attach to any transport."""

    def __init__(self, name: str, tools: list[Tool] | None = None, *, resource_url: str | None = None,
                 token_verifier: TokenVerifier | None = None, authorization_servers: list[str] | None = None,
                 scopes_supported: list[str] | None = None, long_running: list[str] | None = None,
                 task_store: TaskStore | None = None, poll_interval_ms: int = 500, task_ttl_ms: int = 3_600_000,
                 version: str = "0.1.0"):
        self.name = name
        self.version = version
        # The canonical URL is the token audience even when reached through a private address.
        self.resource_url = resource_url or f"https://{name}.mcp.example/mcp"
        self.token_verifier = token_verifier
        self.authorization_servers = list(authorization_servers or [])
        self.scopes_supported = list(scopes_supported or [])
        self._tools: dict[str, Tool] = {}
        self._long_running: set[str] = set()
        self._tasks = _Tasks(task_store or TaskStore(), self._invoke, poll_interval_ms, task_ttl_ms)
        for tool in tools or []:
            self.add_tool(tool, long_running=tool.spec.name in set(long_running or []))

    # -- registration and discovery -------------------------------------------
    def add_tool(self, tool: Tool, *, long_running: bool = False) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.spec.name}")
        self._tools[tool.spec.name] = tool
        if long_running:
            self._long_running.add(tool.spec.name)

    @property
    def endpoint_path(self) -> str:
        return urlparse(self.resource_url).path or "/mcp"

    @property
    def metadata_url(self) -> str:
        return self.resource_url.rstrip("/") + WELL_KNOWN_PRM

    def protected_resource_metadata(self) -> dict[str, Any]:
        """RFC 9728 document: public, so an unauthenticated client can learn where to get a token."""
        return {"resource": self.resource_url, "authorization_servers": self.authorization_servers,
                "scopes_supported": self.scopes_supported, "bearer_methods_supported": ["header"]}

    def tool_descriptor(self, tool: Tool) -> dict[str, Any]:
        d: dict[str, Any] = {"name": tool.spec.name, "description": tool.spec.description,
                             "inputSchema": tool.spec.input_schema, "annotations": p.annotations_for(tool.spec.side_effect)}
        if tool.spec.name in self._long_running:
            d["execution"] = {"taskSupport": "optional"}   # Tasks extension: may become a task if the client asks
        return d

    # -- the request pipeline -----------------------------------------------
    async def handle_jsonrpc(self, headers: Mapping[str, str], payload: Any) -> Response:
        """One request in, one response out. Headers are matched case-insensitively."""
        hdrs = {k.lower(): v for k, v in headers.items()}
        request_id = payload.get("id") if isinstance(payload, dict) else None
        try:
            identity = self._authenticate(hdrs)
            req = p.parse_request(payload)
            self._check_mirrored_headers(hdrs, req)
            self._check_version(req)
            result = await self._dispatch(req, identity, hdrs.get(p.HEADER_AGENT_IDENTITY.lower(), "mcp-client"))
            return Response(200, {}, p.success(req.id, p.complete_result(result)))
        except McpError as e:
            return Response(e.http_status, e.headers, p.error(request_id, e))

    def _authenticate(self, hdrs: Mapping[str, str]) -> Identity | None:
        """Bearer token → verified claims → audience check → Identity. No verifier means an open (local) server."""
        if self.token_verifier is None:
            return None
        challenge = f'Bearer resource_metadata="{self.metadata_url}"'
        auth = hdrs.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise McpError(p.UNAUTHORIZED, "bearer token required", headers={"WWW-Authenticate": challenge})
        token = auth[7:].strip()
        try:
            claims = self.token_verifier(token)
        except (OAuthError, ValueError) as e:
            raise McpError(p.UNAUTHORIZED, f"invalid token: {e}",
                           headers={"WWW-Authenticate": f'{challenge}, error="invalid_token"'}) from None
        if claims.get("aud") != self.resource_url:
            # A token for another server is not "almost right": accepting it would make audience binding meaningless.
            raise McpError(p.UNAUTHORIZED, f"token audience {claims.get('aud')!r} is not this server ({self.resource_url})",
                           headers={"WWW-Authenticate": f'{challenge}, error="invalid_token"'})
        return identity_from_claims(claims, token=token)

    @staticmethod
    def _check_mirrored_headers(hdrs: Mapping[str, str], req: JsonRpcRequest) -> None:
        """Header and body must agree; otherwise a gateway would authorise one call and the server run another."""
        expected = {p.HEADER_PROTOCOL_VERSION: req.protocol_version, p.HEADER_METHOD: req.method}
        if req.method == "tools/call":
            expected[p.HEADER_NAME] = req.tool_name
        for header, body_value in expected.items():
            header_value = hdrs.get(header.lower())
            if header_value is None or body_value is None or header_value != body_value:
                raise McpError(p.HEADER_MISMATCH, f"HeaderMismatch: {header}={header_value!r} but body says {body_value!r}",
                               data={"header": header, "headerValue": header_value, "bodyValue": body_value})

    @staticmethod
    def _check_version(req: JsonRpcRequest) -> None:
        if req.protocol_version not in p.SUPPORTED_VERSIONS:
            raise McpError(p.UNSUPPORTED_PROTOCOL_VERSION, f"UnsupportedProtocolVersion: {req.protocol_version!r}",
                           data={"supportedVersions": list(p.SUPPORTED_VERSIONS)})

    async def _dispatch(self, req: JsonRpcRequest, identity: Identity | None, agent: str) -> dict[str, Any]:
        if req.method == "server/discover":
            return self._discover()
        if req.method == "tools/list":
            return {"tools": [self.tool_descriptor(t) for t in self._tools.values()]}
        if req.method == "tools/call":
            return await self._tools_call(req, identity, agent)
        if req.method == "tasks/get":
            return self._tasks.get(self._task_id(req))
        if req.method == "tasks/update":
            responses = req.params.get("inputResponses")
            if not isinstance(responses, dict):
                raise McpError(p.INVALID_PARAMS, "tasks/update needs inputResponses")
            return self._tasks.update(self._task_id(req), responses)
        if req.method == "tasks/cancel":
            return self._tasks.cancel(self._task_id(req))
        raise McpError(p.METHOD_NOT_FOUND, f"method not found: {req.method}", data={"methods": list(p.METHODS)})

    # -- methods ----------------------------------------------------------------
    def _discover(self) -> dict[str, Any]:
        return {"protocolVersions": list(p.SUPPORTED_VERSIONS),
                "capabilities": {"tools": {"listChanged": False}, "extensions": {p.TASKS_EXTENSION: {}}},
                "serverInfo": {"name": self.name, "version": self.version}}

    async def _tools_call(self, req: JsonRpcRequest, identity: Identity | None, agent: str) -> dict[str, Any]:
        tool = self._tools.get(req.tool_name or "")
        if tool is None:
            raise McpError(p.INVALID_PARAMS, f"unknown tool {req.tool_name!r}", data={"tools": list(self._tools)})
        self._authorize(identity, tool)
        caps = req.client_capabilities
        call = _Call(tool, dict(req.params.get("arguments") or {}), identity, agent, dict(req.params.get("inputResponses") or {}))
        if tool.spec.name in self._long_running and p.supports_tasks(caps):
            return self._tasks.start(call)
        try:
            outcome = await self._invoke(call)
            if isinstance(outcome, LongRunning):
                # The tool decided at runtime that this call is long: a task if the client can take one, else run it inline.
                call = replace(call, work=outcome.work)
                if p.supports_tasks(caps):
                    return self._tasks.start(call, outcome.status_message)
                outcome = await self._invoke(call)
        except NeedsInput as need:
            if not p.supports_elicitation(caps):
                return p.call_tool_result(ToolResult.failure(
                    "input_required", f"{tool.spec.name} needs user input and this client did not declare elicitation",
                    hint="Ask the user directly, then call again with the answer in the arguments."))
            return p.input_required_result(need.as_input_requests())
        return outcome

    def _authorize(self, identity: Identity | None, tool: Tool) -> None:
        """Per-tool scope → 403 with the scope the client should step up to. Only meaningful when tokens are verified."""
        scope = tool.spec.required_scope
        if self.token_verifier is None or not scope or (identity is not None and identity.has_scope(scope)):
            return
        raise McpError(p.FORBIDDEN, f"{tool.spec.name} requires scope {scope}",
                       headers={"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{scope}"'})

    @staticmethod
    def _task_id(req: JsonRpcRequest) -> str:
        task_id = req.params.get("taskId")
        if not isinstance(task_id, str):
            raise McpError(p.INVALID_PARAMS, "taskId is required")
        return task_id

    # -- execution ------------------------------------------------------------------
    async def _invoke(self, call: _Call, task_id: str | None = None,
                      progress_hook: Callable[[str], None] | None = None) -> dict[str, Any] | LongRunning:
        """Run the tool (or its resumed ``work``). Raises ``NeedsInput``; returns a CallToolResult or a ``LongRunning`` marker."""
        ctx = call.context(task_id, progress_hook)
        if call.work is None:
            result = await call.tool.run(call.args, ctx)
            if isinstance(result.data, LongRunning):
                return result.data if task_id is None else await self._invoke(replace(call, work=result.data.work), task_id, progress_hook)
            return p.call_tool_result(result)
        t0 = time.perf_counter()
        try:
            out = call.work(ctx)
            if inspect.isawaitable(out):
                out = await out
            result = ToolResult.success(out, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001 - same boundary FunctionTool.run draws: no stack traces to the client
            result = ToolResult.failure("tool_failure", f"{type(e).__name__}: {e}", latency_ms=(time.perf_counter() - t0) * 1000)
        return p.call_tool_result(result)


# ------------------------------------------------------------------ tasks
def task_summary(rec: TaskRecord) -> dict[str, Any]:
    return {"taskId": rec.id, "status": rec.status.value, "statusMessage": rec.status_message,
            "ttlMs": rec.ttl_ms, "pollIntervalMs": rec.poll_interval_ms}


class _Tasks:
    """The Tasks extension: durable records in a ``TaskStore``, work on asyncio tasks, cooperative cancel.

    The record is the truth a client polls; the asyncio task is just the worker.
    TTL is reported, not enforced (a real server sweeps expired records).
    """

    def __init__(self, store: TaskStore, invoke: Callable[..., Awaitable[Any]], poll_interval_ms: int, ttl_ms: int):
        self.store = store
        self._invoke = invoke
        self.poll_interval_ms = poll_interval_ms
        self.ttl_ms = ttl_ms
        self._calls: dict[str, _Call] = {}
        self._handles: dict[str, asyncio.Task[None]] = {}

    def start(self, call: _Call, status_message: str = "accepted") -> dict[str, Any]:
        rec = self.store.create(TaskRecord(kind=call.tool.spec.name, status_message=status_message,
                                           checkpoint={"tool": call.tool.spec.name, "arguments": call.args},
                                           poll_interval_ms=self.poll_interval_ms, ttl_ms=self.ttl_ms))
        self._spawn(rec.id, call)
        return p.task_result(task_summary(rec))

    def get(self, task_id: str) -> dict[str, Any]:
        rec = self._record(task_id)
        out = task_summary(rec)
        if rec.status == TaskStatus.COMPLETED:
            out["result"] = rec.result
        elif rec.status == TaskStatus.FAILED:
            out["error"] = rec.error
        elif rec.status == TaskStatus.INPUT_REQUIRED:
            out["inputRequests"] = rec.input_requests
        return out

    def update(self, task_id: str, responses: dict[str, Any]) -> dict[str, Any]:
        rec = self._record(task_id)
        if rec.status != TaskStatus.INPUT_REQUIRED:
            raise McpError(p.INVALID_PARAMS, f"task {task_id} is {rec.status.value}, not input_required")
        rec.input_responses.update(responses)
        rec.input_requests = {}
        rec.mark(TaskStatus.WORKING, "resuming with input")
        self.store.put(rec)
        call = replace(self._calls[task_id], input_responses=dict(rec.input_responses))
        self._spawn(task_id, call)
        return task_summary(rec)

    def cancel(self, task_id: str) -> dict[str, Any]:
        rec = self._record(task_id)
        if rec.status.terminal:
            return task_summary(rec)          # idempotent: cancelling a finished task changes nothing
        rec.mark(TaskStatus.CANCELLED, "cancelled by client")
        self.store.put(rec)
        handle = self._handles.get(task_id)
        if handle is not None and not handle.done():
            handle.get_loop().call_soon_threadsafe(handle.cancel)   # the worker sees CancelledError at its next await
        return task_summary(rec)

    # -- internals ----------------------------------------------------------------
    def _record(self, task_id: str) -> TaskRecord:
        try:
            return self.store.get(task_id)
        except KeyError:
            raise McpError(p.INVALID_PARAMS, f"unknown taskId {task_id!r}", http_status=404) from None

    def _spawn(self, task_id: str, call: _Call) -> None:
        self._calls[task_id] = call
        self._handles[task_id] = asyncio.create_task(self._drive(task_id, call), name=f"mcp-task:{task_id}")

    async def _drive(self, task_id: str, call: _Call) -> None:
        try:
            outcome = await self._invoke(call, task_id, lambda msg: self._set_message(task_id, msg))
        except NeedsInput as need:
            self._finish(task_id, TaskStatus.INPUT_REQUIRED, need.message, input_requests=need.as_input_requests())
        except asyncio.CancelledError:
            self._finish(task_id, TaskStatus.CANCELLED, "cancelled")
        except Exception as e:  # noqa: BLE001
            self._finish(task_id, TaskStatus.FAILED, "failed", error={"code": p.INTERNAL_ERROR, "message": f"{type(e).__name__}: {e}"})
        else:
            self._finish(task_id, TaskStatus.COMPLETED, "done", result=outcome)

    def _finish(self, task_id: str, status: TaskStatus, message: str, **fields: Any) -> None:
        rec = self.store.get(task_id)
        if rec.status.terminal:
            return                            # a cancel that landed first wins; never resurrect a finished task
        rec.mark(status, message)
        for name, value in fields.items():
            setattr(rec, name, value)
        self.store.put(rec)

    def _set_message(self, task_id: str, message: str) -> None:
        rec = self.store.get(task_id)
        if rec.status == TaskStatus.WORKING:
            rec.status_message = message
            self.store.put(rec)
