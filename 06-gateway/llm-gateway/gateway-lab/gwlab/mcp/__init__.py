"""The gateway as an MCP client (PRIMER §8), over HTTP: a fake authorization server and MCP server, and the
client side of the MCP authorization spec (revision 2026-07-28, verify) plus DPoP nonces (RFC 9449 §8, §9).

    from gwlab.mcp import McpServers, McpClient
"""
from .client import McpClient, as_metadata_urls, parse_www_authenticate, pkce_challenge, prm_urls, union_scopes
from .server import McpServers

__all__ = ["McpClient", "McpServers", "as_metadata_urls", "parse_www_authenticate", "pkce_challenge", "prm_urls",
           "union_scopes"]
