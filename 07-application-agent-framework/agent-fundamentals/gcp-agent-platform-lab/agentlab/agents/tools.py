"""Tools: contracts a model can call.

Design rules baked in here (Primer §3.1):

* the schema is derived from a typed Python signature (pydantic) and arguments
  are validated *before* execution — the model's output is never trusted to be
  well-formed;
* every tool declares a side-effect class (read / reversible / irreversible);
  irreversible tools require confirmation by default;
* results are structured; errors say whether they are retryable and what the
  model should do next;
* writes are idempotent when the caller supplies an idempotency key.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError, create_model

from .budget import Budget


class SideEffect(str, Enum):
    READ = "read"
    REVERSIBLE = "reversible"      # a write that can be undone or is idempotent by key
    IRREVERSIBLE = "irreversible"  # money moved, email sent, record deleted


# ------------------------------------------------------------------ results
@dataclass
class ToolError:
    type: str                     # "invalid_arguments" | "not_found" | "transient" | "forbidden" | "tool_failure" | ...
    message: str
    retryable: bool = False       # may the *runtime* retry the same call?
    hint: str | None = None       # what the *model* should do next


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: ToolError | None = None
    latency_ms: float = 0.0
    from_idempotency_cache: bool = False

    @classmethod
    def success(cls, data: Any, latency_ms: float = 0.0) -> "ToolResult":
        return cls(ok=True, data=data, latency_ms=latency_ms)

    @classmethod
    def failure(cls, type: str, message: str, retryable: bool = False, hint: str | None = None, latency_ms: float = 0.0) -> "ToolResult":
        return cls(ok=False, error=ToolError(type, message, retryable, hint), latency_ms=latency_ms)

    def to_content(self, max_chars: int = 2_000) -> str:
        """Compact JSON for the model's context, truncated with an explicit marker."""
        if self.ok:
            payload: Any = {"ok": True, "data": self.data}
        else:
            e = self.error
            payload = {"ok": False, "error": e.type, "message": e.message, "retryable": e.retryable}
            if e.hint:
                payload["hint"] = e.hint
        s = json.dumps(payload, default=str, separators=(",", ":"))
        if len(s) > max_chars:
            s = s[: max_chars - 40] + f'..."[truncated {len(s) - max_chars + 40} chars]"}}'
        return s


# ---------------------------------------------------------------- identity
@dataclass
class Identity:
    subject: str
    tenant: str = "default"
    scopes: set[str] = field(default_factory=set)
    token: str | None = None      # opaque credential the tool layer may exchange downstream

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass
class ToolContext:
    session_id: str = "session"
    tenant: str = "default"
    user: Identity | None = None
    agent_name: str = "agent"
    idempotency_key: str | None = None
    budget: Budget | None = None
    depth: int = 0
    span: Any = None                       # tracing span, if any
    extras: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------- specs
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: SideEffect = SideEffect.READ
    requires_confirmation: bool = False
    required_scope: str | None = None

    def to_model_schema(self) -> dict[str, Any]:
        """What the model sees. Keep it small: it is paid for on every turn."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class ToolArgumentError(ValueError):
    pass


class ToolPermanentError(RuntimeError):
    """Raise from a tool function for a non-retryable failure (e.g. not found)."""

    def __init__(self, message: str, type: str = "tool_failure", hint: str | None = None):
        super().__init__(message)
        self.type = type
        self.hint = hint


class ToolTransientError(RuntimeError):
    """Raise from a tool function for a retryable failure (timeout, 503, rate limit)."""


class IdempotencyStore:
    """Remembers results of writes by key so a retried call cannot double-post."""

    def __init__(self) -> None:
        self._results: dict[str, ToolResult] = {}

    def get(self, key: str) -> ToolResult | None:
        return self._results.get(key)

    def put(self, key: str, result: ToolResult) -> None:
        self._results[key] = result

    def __len__(self) -> int:
        return len(self._results)


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(x) for x in schema]
    return schema


class FunctionTool:
    """Wraps a typed Python function as a tool.

    The JSON schema comes from the signature; ``ctx: ToolContext`` parameters are
    injected, not exposed to the model.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        side_effect: SideEffect = SideEffect.READ,
        requires_confirmation: bool | None = None,
        required_scope: str | None = None,
        idempotency: IdempotencyStore | None = None,
        max_result_chars: int = 2_000,
    ):
        self.fn = fn
        self.max_result_chars = max_result_chars
        self.idempotency = idempotency
        sig = inspect.signature(fn)
        fields: dict[str, Any] = {}
        self._inject_ctx = False
        for pname, param in sig.parameters.items():
            if pname == "ctx" or param.annotation is ToolContext:
                self._inject_ctx = True
                continue
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[pname] = (ann, default)
        self.model: type[BaseModel] = create_model(f"{fn.__name__}_args", **fields)  # type: ignore[call-overload]
        schema = _strip_titles(self.model.model_json_schema())
        if requires_confirmation is None:
            requires_confirmation = side_effect == SideEffect.IRREVERSIBLE
        self.spec = ToolSpec(
            name=name or fn.__name__,
            description=(description or inspect.getdoc(fn) or "").strip(),
            input_schema=schema,
            side_effect=side_effect,
            requires_confirmation=requires_confirmation,
            required_scope=required_scope,
        )

    # -- validation -----------------------------------------------------
    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.model(**(args or {})).model_dump()
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ToolArgumentError(problems) from None
        except TypeError as e:
            raise ToolArgumentError(str(e)) from None

    # -- execution ------------------------------------------------------
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        try:
            clean = self.validate(args)
        except ToolArgumentError as e:
            return ToolResult.failure("invalid_arguments", str(e), retryable=False,
                                      hint="Fix the arguments to match the schema and call again.",
                                      latency_ms=(time.perf_counter() - t0) * 1000)
        if self.spec.required_scope and (ctx.user is None or not ctx.user.has_scope(self.spec.required_scope)):
            return ToolResult.failure("forbidden", f"requires scope {self.spec.required_scope}", retryable=False,
                                      hint="Tell the user this action is not permitted for their account.",
                                      latency_ms=(time.perf_counter() - t0) * 1000)
        key = ctx.idempotency_key
        if self.idempotency is not None and key and self.spec.side_effect != SideEffect.READ:
            cached = self.idempotency.get(key)
            if cached is not None:
                return ToolResult(ok=cached.ok, data=cached.data, error=cached.error, latency_ms=0.0, from_idempotency_cache=True)
        try:
            if self._inject_ctx:
                out = self.fn(**clean, ctx=ctx)
            else:
                out = self.fn(**clean)
            if inspect.isawaitable(out):
                out = await out
            result = ToolResult.success(out, latency_ms=(time.perf_counter() - t0) * 1000)
        except ToolPermanentError as e:
            result = ToolResult.failure(e.type, str(e), retryable=False, hint=e.hint, latency_ms=(time.perf_counter() - t0) * 1000)
        except ToolTransientError as e:
            result = ToolResult.failure("transient", str(e), retryable=True, hint="The system is temporarily unavailable; try once more or tell the user.", latency_ms=(time.perf_counter() - t0) * 1000)
        except asyncio.TimeoutError:
            result = ToolResult.failure("timeout", f"{self.spec.name} timed out", retryable=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001 - boundary: never leak stack traces to the model
            result = ToolResult.failure("tool_failure", f"{type(e).__name__}: {e}", retryable=False,
                                        hint="Do not retry with the same arguments; explain the failure to the user.",
                                        latency_ms=(time.perf_counter() - t0) * 1000)
        if self.idempotency is not None and key and self.spec.side_effect != SideEffect.READ and result.ok:
            self.idempotency.put(key, result)
        return result


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: SideEffect = SideEffect.READ,
    requires_confirmation: bool | None = None,
    required_scope: str | None = None,
    idempotency: IdempotencyStore | None = None,
) -> Any:
    """Decorator: ``@tool`` or ``@tool(side_effect=SideEffect.REVERSIBLE)``."""

    def wrap(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, side_effect=side_effect,
                            requires_confirmation=requires_confirmation, required_scope=required_scope,
                            idempotency=idempotency)

    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    """Name → tool, plus the schema list handed to the model."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.spec.name in self._tools:
            raise ValueError(f"duplicate tool name {t.spec.name}")
        self._tools[t.spec.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.spec.to_model_schema() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
