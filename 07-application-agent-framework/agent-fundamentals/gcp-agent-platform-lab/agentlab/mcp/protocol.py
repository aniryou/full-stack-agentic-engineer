"""Wire shapes of the Model Context Protocol, 2026-07-28 revision (notebook 05).

Checked against the revision's changelog and schema in the
modelcontextprotocol/modelcontextprotocol repository on 2026-09-26 (verify);
``docs/MCP_REVISIONS.md`` says what differs from the 2025 revisions and where
this subset departs from the spec. Teaching subset, not a conformant implementation. It keeps exactly the parts an
architect has to reason about in a design discussion:

* JSON-RPC 2.0 request / result / error envelopes (no batches, no notifications);
* the per-request ``_meta`` block that replaced the ``initialize`` handshake:
  every request names its protocol version and the client's capabilities, so
  a server can answer statelessly and an intermediary can route by version;
* the HTTP headers that mirror body fields (``MCP-Protocol-Version``,
  ``Mcp-Method``, ``Mcp-Name``) so a gateway can apply per-tool policy without
  parsing JSON — and the ``HeaderMismatch`` error servers raise when the two
  disagree;
* the two embedded server→client shapes: ``input_required`` (elicitation via
  MRTR, the client re-sends the call with ``inputResponses``) and ``task``
  (the Tasks extension's durable handle);
* tool annotations derived from ``ToolSpec.side_effect`` so the same tool
  definition behaves identically locally and remotely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..agents.tools import SideEffect, ToolResult

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED_VERSIONS: tuple[str, ...] = (PROTOCOL_VERSION,)

# ``params._meta`` keys carried on every request (the revision has no session).
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"

# Streamable-HTTP headers that mirror body fields for intermediaries.
HEADER_PROTOCOL_VERSION = "MCP-Protocol-Version"
HEADER_METHOD = "Mcp-Method"
HEADER_NAME = "Mcp-Name"
HEADER_AGENT_IDENTITY = "X-Agent-Identity"   # lab stand-in for the SPIFFE id an mTLS gateway extracts

METHODS = ("server/discover", "tools/list", "tools/call", "tasks/get", "tasks/update", "tasks/cancel")

# ---------------------------------------------------------------- error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# The 2026-07-28 revision reserves -32020..-32099 for codes the spec defines (verify):
HEADER_MISMATCH = -32020                 # HeaderMismatch
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021   # MissingRequiredClientCapability (defined, not raised here)
UNSUPPORTED_PROTOCOL_VERSION = -32022    # UnsupportedProtocolVersion
# Lab codes in the legacy -32000..-32019 range, which the revision says new implementations
# SHOULD NOT use; the HTTP status (401 / 403) and WWW-Authenticate carry the OAuth meaning.
UNAUTHORIZED = -32001
FORBIDDEN = -32003

# HTTP status that accompanies each JSON-RPC error on the HTTP transport.
HTTP_STATUS_FOR_CODE = {
    PARSE_ERROR: 400, INVALID_REQUEST: 400, INVALID_PARAMS: 400,
    METHOD_NOT_FOUND: 404, INTERNAL_ERROR: 500,
    HEADER_MISMATCH: 400, MISSING_REQUIRED_CLIENT_CAPABILITY: 400, UNSUPPORTED_PROTOCOL_VERSION: 400,
    UNAUTHORIZED: 401, FORBIDDEN: 403,
}


class McpError(Exception):
    """A JSON-RPC error, on either side of the wire.

    Servers raise it to abort a request; clients raise it when a response
    carries an ``error`` member. ``headers`` lets an error carry HTTP headers
    such as ``WWW-Authenticate``.
    """

    def __init__(self, code: int, message: str, data: Any = None, *,
                 http_status: int | None = None, headers: Mapping[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.http_status = http_status if http_status is not None else HTTP_STATUS_FOR_CODE.get(code, 400)
        self.headers = dict(headers or {})

    def to_error(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


# ------------------------------------------------------------------ envelopes
@dataclass
class JsonRpcRequest:
    id: Any
    method: str
    params: dict[str, Any]

    @property
    def meta(self) -> dict[str, Any]:
        return self.params.get("_meta") or {}

    @property
    def protocol_version(self) -> str | None:
        return self.meta.get(META_PROTOCOL_VERSION)

    @property
    def client_capabilities(self) -> dict[str, Any]:
        return self.meta.get(META_CLIENT_CAPABILITIES) or {}

    @property
    def tool_name(self) -> str | None:
        return self.params.get("name") if self.method == "tools/call" else None


def parse_request(payload: Any) -> JsonRpcRequest:
    """Validate the JSON-RPC 2.0 envelope. Batches and notifications are out of scope."""
    if not isinstance(payload, dict):
        raise McpError(INVALID_REQUEST, "expected a single JSON-RPC request object (batches are not supported)")
    if payload.get("jsonrpc") != "2.0" or not isinstance(payload.get("method"), str):
        raise McpError(INVALID_REQUEST, "jsonrpc must be '2.0' and method must be a string")
    if "id" not in payload:
        raise McpError(INVALID_REQUEST, "notifications are not supported; every request needs an id")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise McpError(INVALID_PARAMS, "params must be an object")
    return JsonRpcRequest(id=payload["id"], method=payload["method"], params=params)


def request(id: Any, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params}


def success(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def error(id: Any, err: McpError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": err.to_error()}


def make_meta(capabilities: dict[str, Any], protocol_version: str = PROTOCOL_VERSION) -> dict[str, Any]:
    """The ``_meta`` block a client attaches to every request."""
    return {META_PROTOCOL_VERSION: protocol_version, META_CLIENT_CAPABILITIES: capabilities}


def client_capabilities(*, tasks: bool = True, elicitation: bool = True) -> dict[str, Any]:
    caps: dict[str, Any] = {}
    if elicitation:
        caps["elicitation"] = {}
    if tasks:
        caps["extensions"] = {TASKS_EXTENSION: {}}
    return caps


def supports_tasks(capabilities: Mapping[str, Any]) -> bool:
    return TASKS_EXTENSION in (capabilities.get("extensions") or {})


def supports_elicitation(capabilities: Mapping[str, Any]) -> bool:
    return "elicitation" in capabilities


# ---------------------------------------------------------------- tool shapes
def annotations_for(side_effect: SideEffect) -> dict[str, bool]:
    """Tool annotations (hints, never a security boundary) from the agentlab side-effect class."""
    return {
        "readOnlyHint": side_effect == SideEffect.READ,
        "destructiveHint": side_effect == SideEffect.IRREVERSIBLE,
        "idempotentHint": side_effect != SideEffect.IRREVERSIBLE,
    }


def side_effect_from(annotations: Mapping[str, Any] | None) -> SideEffect:
    a = annotations or {}
    if a.get("readOnlyHint"):
        return SideEffect.READ
    if a.get("destructiveHint"):
        return SideEffect.IRREVERSIBLE
    return SideEffect.REVERSIBLE


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def call_tool_result(result: ToolResult) -> dict[str, Any]:
    """``CallToolResult``: text content for any client, structured content for typed ones.

    A tool *failure* is still a successful JSON-RPC response with ``isError``:
    the model should see it and react; protocol errors are for broken requests.
    """
    out: dict[str, Any] = {"content": [{"type": "text", "text": result.to_content()}], "isError": not result.ok}
    if result.ok:
        out["structuredContent"] = _jsonable(result.data)
    else:
        out["structuredContent"] = {"error": result.error.type, "message": result.error.message,
                                    "retryable": result.error.retryable, "hint": result.error.hint}
    return out


def elicitation_request(message: str, requested_schema: dict[str, Any]) -> dict[str, Any]:
    """One entry of ``inputRequests``: what the server would have asked the user directly."""
    return {"method": "elicitation/create", "params": {"message": message, "requestedSchema": requested_schema}}


def input_required_result(input_requests: dict[str, Any]) -> dict[str, Any]:
    return {"resultType": "input_required", "inputRequests": input_requests}


def task_result(task: dict[str, Any]) -> dict[str, Any]:
    return {"resultType": "task", "task": task}


def accept(**content: Any) -> dict[str, Any]:
    """An elicitation answer: the user accepted and filled the requested fields."""
    return {"action": "accept", "content": content}


def decline() -> dict[str, Any]:
    return {"action": "decline"}


def accepted_content(response: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The submitted fields if the user accepted, else ``None`` (declined, cancelled or missing)."""
    if not response or response.get("action") != "accept":
        return None
    return dict(response.get("content") or {})
