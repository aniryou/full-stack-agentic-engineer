"""Teaching subset of the Model Context Protocol (2026-07-28 shape) plus an egress gateway. Not a conformant implementation."""
from .client import Forbidden, InputHandler, McpClient, McpToolset, RemoteTool, TaskFailed, Unauthorized
from .gateway import (IDENTITY_REQUIRED, POLICY_DENIED, SCREENING_BLOCKED, UNKNOWN_DESTINATION, AuditRecord, Decision,
                      Gateway, Policy, Rule)
from .protocol import (HEADER_AGENT_IDENTITY, HEADER_METHOD, HEADER_MISMATCH, HEADER_NAME, HEADER_PROTOCOL_VERSION,
                       INVALID_PARAMS, META_CLIENT_CAPABILITIES, META_PROTOCOL_VERSION, METHOD_NOT_FOUND, PROTOCOL_VERSION,
                       SUPPORTED_VERSIONS, TASKS_EXTENSION, UNSUPPORTED_PROTOCOL_VERSION, McpError, accept, accepted_content,
                       annotations_for, call_tool_result, client_capabilities, decline, make_meta)
from .server import LongRunning, McpServer, NeedsInput, confirmed, input_responses, progress, task_summary
from .transport import HttpTransport, InProcessTransport, LocalHttpServer, Transport, handle

__all__ = [
    "PROTOCOL_VERSION", "SUPPORTED_VERSIONS", "TASKS_EXTENSION", "META_PROTOCOL_VERSION", "META_CLIENT_CAPABILITIES",
    "HEADER_PROTOCOL_VERSION", "HEADER_METHOD", "HEADER_NAME", "HEADER_AGENT_IDENTITY",
    "HEADER_MISMATCH", "UNSUPPORTED_PROTOCOL_VERSION", "METHOD_NOT_FOUND", "INVALID_PARAMS",
    "McpError", "accept", "accepted_content", "annotations_for", "call_tool_result", "client_capabilities", "decline", "make_meta",
    "McpServer", "NeedsInput", "LongRunning", "confirmed", "input_responses", "progress", "task_summary",
    "Transport", "InProcessTransport", "HttpTransport", "LocalHttpServer", "handle",
    "McpClient", "McpToolset", "RemoteTool", "InputHandler", "Unauthorized", "Forbidden", "TaskFailed",
    "Gateway", "Policy", "Rule", "Decision", "AuditRecord",
    "UNKNOWN_DESTINATION", "POLICY_DENIED", "SCREENING_BLOCKED", "IDENTITY_REQUIRED",
]
