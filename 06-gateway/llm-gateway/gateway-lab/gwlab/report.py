"""Tables for the ledger, chargeback, reconciliation and spans — what an operator reads after a run.

The one idea: every number the gateway produces should be answerable from its own records: who spent what
(the ledger by tenant, key or target), who pays for a shared GPU (chargeback by tokens or by GPU-seconds),
whether the bill agrees with the provider (reconciliation against the provider's counters, with the rows
whose counts were estimated flagged), and where a slow request spent its time (a trace's spans).
"""
from __future__ import annotations

from .gateway import metering


def table(rows: list, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cells = [[(fmt.get(c, "{}").format(r.get(c)) if r.get(c) is not None else "-") for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)), "  ".join("-" * w for w in widths)]
    out += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(out)


def ledger_table(rows: list, last: int = 10) -> str:
    return table(rows[-last:], ["tenant", "alias", "target", "status", "prompt_tokens", "completion_tokens", "cached_tokens",
                                "usage_source", "cost_usd", "cache", "attempts", "error"],
                 {"cost_usd": "{:.6f}"})


def totals_table(rows: list, key: str = "tenant") -> str:
    t = metering.totals(rows, key)
    return table([{key: k, **v} for k, v in t.items()],
                 [key, "requests", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd", "estimated_rows"],
                 {"cost_usd": "{:.6f}"})


def chargeback_table(rows: list, total_usd: float, prefill_s_per_token: float, decode_s_per_token: float) -> str:
    by_tok = metering.chargeback(rows, total_usd, "tokens")
    by_gpu = metering.chargeback(rows, total_usd, "gpu_seconds", prefill_s_per_token, decode_s_per_token)
    return table([{"tenant": t, "by_tokens": by_tok[t], "by_gpu_seconds": by_gpu[t]} for t in by_tok],
                 ["tenant", "by_tokens", "by_gpu_seconds"], {"by_tokens": "${:.4f}", "by_gpu_seconds": "${:.4f}"})


def reconcile_table(rec: dict) -> str:
    return table([{"provider": p, **v} for p, v in rec.items()],
                 ["provider", "ledger_prompt", "provider_prompt", "prompt_gap", "ledger_completion", "provider_completion",
                  "completion_gap", "estimated_rows"])


def spans_tree(spans: list) -> str:
    """One trace as an indented tree: name, kind, duration and the attributes that matter."""
    by_parent: dict = {}
    for s in spans:
        by_parent.setdefault(s.parent_span_id, []).append(s)
    keys = ("gen_ai.provider.name", "gen_ai.request.model", "gen_ai.response.model", "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens", "gen_ai.usage.cache_read.input_tokens", "gwlab.target", "gwlab.cost_usd",
            "gwlab.cache", "error.type")
    out = []

    def walk(parent: str, depth: int):
        for s in sorted(by_parent.get(parent, []), key=lambda x: x.start_ns):
            attrs = ", ".join(f"{k.split('.', 1)[1] if k.startswith('gen_ai.') else k}={s.attributes[k]}"
                              for k in keys if k in s.attributes)
            kind = {1: "internal", 2: "server", 3: "client"}.get(s.kind, s.kind)
            out.append(f"{'  ' * depth}{s.name} [{kind}] {s.duration_s * 1e3:.1f} ms  {attrs}")
            walk(s.span_id, depth + 1)
    roots = {s.parent_span_id for s in spans} - {s.span_id for s in spans}
    for r in sorted(roots):
        walk(r, 0)
    return "\n".join(out)
