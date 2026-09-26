# MCP revisions: what the lab models, and what differs from 2025

The `agentlab.mcp` package and notebook 05 follow the shape of the **2026-07-28** revision of
the Model Context Protocol. Many servers and SDKs in use still speak a 2025 revision, so a
design has to say which one it assumes. This page compares them and lists where the lab's
teaching subset departs from the spec.

Checked on 2026-09-26 against each revision's changelog and the 2026-07-28 schema in the
`modelcontextprotocol/modelcontextprotocol` repository (verify: the spec moves; re-read the
changelog of the revision you target before relying on a row).

## What changed, revision by revision

| Concern | 2025-03-26 | 2025-06-18 | 2025-11-25 | 2026-07-28 (what the lab models) |
|---|---|---|---|---|
| Connection set-up | `initialize` handshake + `notifications/initialized`; Streamable HTTP may assign `Mcp-Session-Id` | same | same | no handshake and no sessions: every request carries its protocol version and client capabilities in `params._meta`; `server/discover` advertises versions, capabilities and identity |
| Version negotiation | once, in `initialize` | once, then the `MCP-Protocol-Version` header on every later HTTP request | same as 2025-06-18 | per request, in `_meta`; a mismatch is `UnsupportedProtocolVersion` (-32022) |
| Headers an intermediary can route on | `Mcp-Session-Id` | + `MCP-Protocol-Version` | same | + `Mcp-Method` and `Mcp-Name` required on POST; a header that disagrees with the body is `HeaderMismatch` (-32020) |
| JSON-RPC batching | added | removed | — | not supported (the lab rejects batches too) |
| Asking the user or the client for something mid-call | server-initiated requests (`sampling/createMessage`, `roots/list`) | + `elicitation/create` as a server-initiated request | + URL-mode elicitation | Multi Round-Trip Requests: the call returns `resultType: "input_required"` with `inputRequests`; the client retries the same call with `inputResponses`. Roots, Sampling and Logging are deprecated |
| Long-running work | progress notifications only | same | experimental tasks in the core protocol (`tasks/result` blocks, `tasks/list`) | the Tasks extension `io.modelcontextprotocol/tasks`: a task handle, polling with `tasks/get`, input with `tasks/update`, `tasks/cancel` |
| Tool results | `content`; tool annotations (`readOnlyHint`, `destructiveHint` …) added | + `structuredContent`, `outputSchema`, resource links | JSON Schema 2020-12 as the default dialect | every result carries `resultType`; list results carry `ttlMs` and `cacheScope` |
| Authorization | OAuth 2.1 framework introduced | the server is an OAuth resource server with Protected Resource Metadata (RFC 9728); clients MUST send the `resource` indicator (RFC 8707) | + Client ID Metadata Documents recommended, OpenID Connect discovery, incremental scope consent | Dynamic Client Registration deprecated in favour of Client ID Metadata Documents; clients MUST validate `iss` (RFC 9207) when present |
| Server-to-client streams | Streamable HTTP replaces HTTP+SSE; resumable with `Last-Event-ID` | same | servers may close SSE streams for polling | resumability removed (re-issue the request); the GET stream and `resources/subscribe` replaced by `subscriptions/listen` |
| Resource not found | -32002 | -32002 | -32002 | -32602 (Invalid Params) |

For a design review, the three rows that change an architecture are the first three: without
sessions a load balancer can send any request to any replica, and with `Mcp-Method` / `Mcp-Name`
in headers a gateway can enforce per-tool policy without parsing the JSON body (notebook 05,
section 6). MRTR and the Tasks extension turn the two awkward cases — "ask the user" and "this
takes ten minutes" — into ordinary request/response, which is what lets the server stay stateless.

## Where the lab's subset departs from the spec

The lab keeps what a design discussion needs; these are the known gaps, so nobody mistakes the
lab for a conformant implementation:

- **`resultType` on complete results.** The 2026-07-28 schema requires every result to carry
  `resultType` (`"complete"` for an ordinary one). The lab's server omits it on complete results
  and sets it only for `"input_required"` and `"task"` — the same thing a client must assume for
  an earlier-revision server, which treats a missing field as `"complete"`.
- **The task handle** is returned as `resultType: "task"` with a `task` object. `ResultType` is an
  open string in the schema, so an extension can add values; check the Tasks extension's own
  page for the exact shape before relying on it (verify).
- **Authorization error codes.** `Unauthorized` (-32001) and `Forbidden` (-32003) are lab codes in
  the legacy -32000..-32019 range, which new implementations SHOULD NOT use; the HTTP status
  (401 / 403) and the `WWW-Authenticate` challenge are what carry the OAuth meaning.
- **Out of scope:** stdio, notifications, the request-scoped SSE stream for progress,
  `subscriptions/listen`, `ttlMs` / `cacheScope` on list results, `clientInfo` / `serverInfo`
  in `_meta`, resources and prompts. `MissingRequiredClientCapability` (-32021) is defined but
  never raised: a client without elicitation gets a tool error telling the model to ask the user.

## Talking to a 2025 server

An agent that must reach both kinds of server needs a client that can do the `initialize`
handshake and keep the session id for 2025 servers, and send `_meta` per request for 2026 ones.
The 2026-07-28 revision names the probe: call `server/discover` first (the changelog calls
it out for stdio in particular); a server that answers speaks the new revision, one that
returns method-not-found gets the old handshake. The lab speaks only 2026-07-28 on both sides:
its server rejects a client that sends `protocol_version="2025-11-25"` with -32022 and lists
the versions it supports (see `tests/test_mcp.py`).
