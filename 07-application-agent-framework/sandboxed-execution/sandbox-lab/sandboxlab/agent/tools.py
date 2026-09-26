"""tools.py — ``run_code`` and ``fetch_url``, each routed through the boundary that owns it.

One idea: a tool is where the model's wish meets an enforcement point. ``run_code`` hands the
code to an executor (process sandbox, Docker, a Kubernetes Job or warm pod) with fixed budgets and
returns the 07.1 result shape — the exit reason as the error kind, truncated output in ``data``.
``fetch_url`` never opens a socket to the URL: it asks the egress proxy, which knows the
allowlist and holds the credentials (PRIMER §4). The naive variants exist for contrast, so the
scenarios can show the same attack succeeding without the boundary.
"""
from __future__ import annotations

import http.client
import inspect
import json
from urllib.parse import urlsplit

from ..audit import AuditEvent, execution_event
from ..process import Budgets, ExecResult
from ..proxy.client import fetch
from .loop import Context, Tool


def run_code_tool(executor, budgets: Budgets | None = None, *, prelude: str = "") -> Tool:
    """``run_code(code)``: destructive tier by definition (identity primer §6.2). ``prelude`` is
    prepended (e.g. ``proxy.client.SANDBOX_CLIENT`` so code can call ``proxy_get``)."""
    b = budgets or Budgets()
    replay: dict[str, dict] = {}

    def fn(args: dict, ctx: Context) -> dict:
        if ctx.idempotency_key in replay:                    # a redelivered step: same answer, no second run
            return replay[ctx.idempotency_key]
        code = prelude + args["code"]
        if "idempotency_key" in inspect.signature(executor.run).parameters:   # Kubernetes runners key the Job
            r: ExecResult = executor.run(code, b, idempotency_key=ctx.idempotency_key)
        else:
            r = executor.run(code, b)
        ctx.budget.sandbox_cpu_s += r.cpu_s if r.cpu_s is not None else r.wall_s
        ctx.audit.emit(execution_event(r, agent=ctx.agent, session_id=ctx.session_id, invocation_id=ctx.invocation_id,
                                       code=code, idempotency_key=ctx.idempotency_key))
        out = r.to_tool_result()
        replay[ctx.idempotency_key] = out
        return out

    return Tool("run_code", "Run a Python 3 program in a sandbox and return its output.",
                {"code": {"type": "string"}}, fn, tier="destructive", required=["code"])


def fetch_url_tool(proxy_url: str, routes: dict[str, str]) -> Tool:
    """``fetch_url(url)`` through the proxy. ``routes`` maps URL prefixes the model may use to proxy
    routes (``{"https://api.weather.example/": "api-stub"}``); anything else is sent to the proxy as
    an absolute-form request, where the allowlist refuses it and the refusal is audited."""

    def fn(args: dict, ctx: Context) -> dict:
        url = args["url"]
        for prefix, route in routes.items():
            if url.startswith(prefix):
                path = f"/{route}/" + url[len(prefix):].lstrip("/")
                break
        else:
            u = urlsplit(url)
            path = f"http://{u.hostname or ''}{u.path or '/'}" + (f"?{u.query}" if u.query else "")
        try:
            status, body = fetch(proxy_url, path)
        except OSError as e:
            return {"ok": False, "error": "egress_error", "message": f"{type(e).__name__}: {e}"}
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = body[:4000].decode("utf-8", "replace")
        if status == 403:
            return {"ok": False, "error": "egress_denied", "message": (data or {}).get("reason", "denied"),
                    "hint": "Only the allowlisted APIs are reachable."}
        if status >= 400:
            return {"ok": False, "error": f"http_{status}", "message": str(data)[:500]}
        return {"ok": True, "data": data}

    return Tool("fetch_url", "Fetch a URL (allowlisted APIs only) and return the response body.",
                {"url": {"type": "string"}}, fn, tier="external", required=["url"])


# ---- the naive versions (for contrast only) --------------------------------------------------------------------
def naive_fetch_url_tool(resolve: dict[str, tuple[str, int]], credential: str, routes: dict[str, tuple[str, int]]) -> Tool:
    """What people write first: the agent process fetches the URL itself, attaching its own API key
    to anything that looks like the API. ``resolve`` stands in for DNS so an 'attacker' host lands
    on a local listener; ``routes`` for the real API."""

    def fn(args: dict, ctx: Context) -> dict:
        u = urlsplit(args["url"])
        host = u.hostname or ""
        target = routes.get(host) or resolve.get(host)
        if target is None:
            return {"ok": False, "error": "tool_failure", "message": f"cannot resolve {host}"}
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        headers = {"Host": host, "Authorization": f"Bearer {credential}"}      # the key goes wherever the URL says
        ctx.audit.emit(AuditEvent("egress", ctx.agent, "own", tool="fetch_url", decision="allow",
                                  reasons=["no egress policy"], session_id=ctx.session_id, extra={"host": host}))
        try:
            c = http.client.HTTPConnection(*target, timeout=3)
            c.request("GET", path, headers=headers)
            r = c.getresponse()
            body = r.read()
            c.close()
            try:
                return {"ok": True, "data": json.loads(body)}
            except ValueError:
                return {"ok": True, "data": body[:4000].decode("utf-8", "replace")}
        except (OSError, http.client.HTTPException) as e:
            return {"ok": False, "error": "tool_failure", "message": f"{type(e).__name__}: {e}"}

    return Tool("fetch_url", "Fetch a URL and return the response body.", {"url": {"type": "string"}}, fn,
                tier="external", required=["url"])
