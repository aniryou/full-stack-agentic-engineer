# MCP revisions: what the lab models, and what differs from 2025

The `agentlab.mcp` package and notebook 05 have the shape of the **2026-07-28** revision of
the Model Context Protocol. Many servers and SDKs in use still use a 2025 revision. Thus a
design must say which revision it assumes. This page compares the revisions. It also lists
where the subset that the lab teaches is different from the spec.

The last check of this page was on 2026-09-26. It compared the page with the changelog of each
revision and with the 2026-07-28 schema in the `modelcontextprotocol/modelcontextprotocol`
repository (verify). The spec changes. Before you rely on a row, read the changelog of the
revision that you target again.

## What changed, revision by revision

| Concern | 2025-03-26 | 2025-06-18 | 2025-11-25 | 2026-07-28 (what the lab models) |
|---|---|---|---|---|
| Connection set-up | `initialize` handshake + `notifications/initialized`. Streamable HTTP can assign `Mcp-Session-Id`. | same | same | No handshake and no sessions. Every request carries its protocol version and client capabilities in `params._meta`. `server/discover` announces versions, capabilities and identity. |
| Version negotiation | once, in `initialize` | once, then the `MCP-Protocol-Version` header on every later HTTP request | same as 2025-06-18 | Per request, in `_meta`. A mismatch is `UnsupportedProtocolVersion` (-32022). |
| Headers an intermediary can route on | `Mcp-Session-Id` | + `MCP-Protocol-Version` | same | + `Mcp-Method` and `Mcp-Name`, necessary on POST. A header that disagrees with the body is `HeaderMismatch` (-32020). |
| JSON-RPC batching | added | removed | — | Not supported. The lab also rejects batches. |
| A request to the user or the client for something during a call | server-initiated requests (`sampling/createMessage`, `roots/list`) | + `elicitation/create` as a server-initiated request | + URL-mode elicitation | Multi Round-Trip Requests: the call returns `resultType: "input_required"` with `inputRequests`. Then the client retries the same call with `inputResponses`. Roots, Sampling and Logging are deprecated. |
| Long-running work | progress notifications only | same | experimental tasks in the core protocol (`tasks/result` blocks, `tasks/list`) | The Tasks extension `io.modelcontextprotocol/tasks`: a task handle, a poll with `tasks/get`, input with `tasks/update`, and `tasks/cancel`. |
| Tool results | `content`. This revision added tool annotations (`readOnlyHint`, `destructiveHint` …). | + `structuredContent`, `outputSchema`, resource links | JSON Schema 2020-12 as the default dialect | Every result carries `resultType`. List results carry `ttlMs` and `cacheScope`. |
| Authorization | This revision added the OAuth 2.1 framework. | The server is an OAuth resource server with Protected Resource Metadata (RFC 9728). Clients MUST send the `resource` indicator (RFC 8707). | + Client ID Metadata Documents (recommended), OpenID Connect discovery, incremental scope consent | Dynamic Client Registration is deprecated, and Client ID Metadata Documents are the preferred replacement. Clients MUST validate `iss` (RFC 9207) when it is present. |
| Server-to-client streams | Streamable HTTP replaces HTTP+SSE. A stream can resume with `Last-Event-ID`. | same | Servers can close SSE streams, and then the client polls. | This revision removes resumability (send the request again). `subscriptions/listen` replaces the GET stream and `resources/subscribe`. |
| Resource not found | -32002 | -32002 | -32002 | -32602 (Invalid Params) |

For a design review, the first three rows are the rows that change an architecture. Without
sessions, a load balancer can send any request to any replica. With `Mcp-Method` / `Mcp-Name`
in headers, a gateway can enforce a per-tool policy without a parse of the JSON body (notebook
05, section 6). MRTR and the Tasks extension change the two difficult cases into an ordinary
request and response. The two cases are "ask the user" and "this takes ten minutes". This
change is what lets the server stay stateless.

## Where the lab's subset departs from the spec

The lab keeps what a design discussion needs. This list gives the known gaps, so that no
reader thinks that the lab is a conformant implementation:

- **The task handle** comes back as `resultType: "task"` with a `task` object. `ResultType` is
  an open string in the schema, so an extension can add values. Before you rely on this shape,
  examine the exact shape on the Tasks extension's own page (verify).
- **Authorization error codes.** `Unauthorized` (-32001) and `Forbidden` (-32003) are lab codes
  in the legacy -32000..-32019 range. The spec marks that range as NOT RECOMMENDED for new
  implementations. The HTTP status (401 / 403) and the `WWW-Authenticate` challenge are the
  parts that carry the OAuth meaning.
- **Out of scope:** stdio, notifications, the request-scoped SSE stream for progress,
  `subscriptions/listen`, `ttlMs` / `cacheScope` on list results, `clientInfo` / `serverInfo`
  in `_meta`, resources and prompts. The lab defines `MissingRequiredClientCapability` (-32021)
  but never raises it. A client without elicitation gets a tool error. The error tells the
  model to ask the user.

## Talking to a 2025 server

An agent that must reach both kinds of server needs a client with two abilities. For a 2025
server, the client can do the `initialize` handshake and keep the session id. For a 2026
server, it can send `_meta` per request.

The 2026-07-28 revision names the probe: call `server/discover` first. The changelog names
this probe for stdio in particular. A server that answers uses the new revision. A server that returns method-not-found gets the old handshake.

A 2025 server sends no `resultType`. The 2026-07-28 schema tells the client to treat an absent
field as `"complete"`, and the lab's client does this. The lab's server sets `"complete"` on
every ordinary result. The lab uses only 2026-07-28 on both sides. Its server rejects a client
that sends `protocol_version="2025-11-25"` with -32022. The server also lists the versions
that it supports (see `tests/test_mcp.py`).
