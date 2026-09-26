"""A tool is an ordinary Python function the model is allowed to call.

The ``@tool`` decorator does three small things a real agent framework does:

1. reads the function's signature to build a little JSON schema (what the model
   is shown so it knows the arguments);
2. checks the arguments before running (the model's output is never trusted to
   be well-formed);
3. wraps the return value in a structured result — ``{"ok": True, "data": ...}``
   on success, ``{"ok": False, "error": ..., "message": ...}`` on failure — so
   the model gets something it can act on instead of a stack trace.

One exception is deliberately *not* wrapped: ``NotImplementedError``. It means
"this tool is not written yet" (every blank exercise raises it), which is a bug in
the program, not a failure the model should hear about and route around — so it
propagates and stops the run where the missing code is.

That is the whole idea. No pydantic, no async, ~100 lines.
"""
from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_args, get_origin


class ToolError(Exception):
    """Raise this inside a tool for an expected failure the model should hear about.

        raise ToolError("no such account", kind="not_found",
                        hint="Ask the user to check the account id.")
    """

    def __init__(self, message: str, kind: str = "tool_error", hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _json_type(ann: Any) -> str:
    """Map a Python annotation to a JSON-schema type: ``int`` → "integer",
    ``list[str]`` → "array", ``int | None`` → "integer"; anything else → "string"."""
    if isinstance(ann, str):                     # an annotation we could not evaluate
        name = ann.replace("Optional[", "").split("|")[0].split("[")[0].strip()
        ann = {t.__name__: t for t in _JSON_TYPES}.get(name, ann)
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        rest = [a for a in get_args(ann) if a is not type(None)]
        return _json_type(rest[0]) if len(rest) == 1 else "string"
    return _JSON_TYPES.get(origin or ann, "string")


def _signature(f: Callable[..., Any]) -> inspect.Signature:
    # Under `from __future__ import annotations` every annotation is a *string*
    # ("int", not int); eval_str=True turns them back into types.
    try:
        return inspect.signature(f, eval_str=True)
    except Exception:                            # e.g. a name only defined later
        return inspect.signature(f)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., Any]
    required: list[str] = field(default_factory=list)
    confirm: bool = False          # if True, the loop asks a human before running it

    def run(self, args: dict[str, Any] | None) -> dict[str, Any]:
        args = args or {}
        unknown = [a for a in args if a not in self.schema["parameters"]["properties"]]
        if unknown and not _takes_kwargs(self.fn):
            return {"ok": False, "error": "invalid_arguments",
                    "message": f"unknown argument(s): {', '.join(unknown)}",
                    "hint": "Use only the arguments in the schema and call again."}
        missing = [p for p in self.required if p not in args or args[p] is None]
        if missing:
            return {"ok": False, "error": "invalid_arguments",
                    "message": f"missing required argument(s): {', '.join(missing)}",
                    "hint": "Fix the arguments to match the schema and call again."}
        try:
            return {"ok": True, "data": self.fn(**args)}
        except ToolError as e:
            out = {"ok": False, "error": e.kind, "message": str(e)}
            if e.hint:
                out["hint"] = e.hint
            return out
        except NotImplementedError:                  # an unwritten tool is a bug, not a result
            raise
        except Exception as e:                       # boundary: never leak a traceback to the model
            return {"ok": False, "error": "tool_failure", "message": f"{type(e).__name__}: {e}"}


def _takes_kwargs(f: Callable[..., Any]) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(f).parameters.values())


def tool(fn: Callable[..., Any] | None = None, *, confirm: bool = False) -> Any:
    """Use as ``@tool`` or ``@tool(confirm=True)``."""

    def wrap(f: Callable[..., Any]) -> Tool:
        sig = _signature(f)
        props: dict[str, dict] = {}
        required: list[str] = []
        for name, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue                             # *args / **kwargs are not named arguments
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
            props[name] = {"type": _json_type(ann)}
            if p.default is inspect.Parameter.empty:
                required.append(name)
        schema = {
            "name": f.__name__,
            "description": (inspect.getdoc(f) or "").strip().split("\n\n")[0],
            "parameters": {"type": "object", "properties": props, "required": required},
        }
        return Tool(f.__name__, schema["description"], schema, f, required, confirm)

    return wrap(fn) if fn is not None else wrap


def tool_message(tc: "Any", result: dict[str, Any]) -> dict[str, Any]:
    """Format a tool's result as a message to append to the transcript."""
    return {"role": "tool", "name": tc.name, "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
