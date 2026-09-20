
# Agent Registry = the inventory the governance plane reasons about (primer §9, §10). Agent
# Gateway only forwards to *registered* resources, and IAP binds IAM allow/deny policies to the
# registered MCP server so "which agent may call which server" is a policy, not a network fact.
#
# Cloud Run services deployed with --functional-type=mcp-server are auto-registered under
# /mcpServers as well; this explicit registration keeps a deterministic, reviewable entry in
# Terraform. Both may coexist (the gateway allows either), or set deletion_policy = "ABANDON"
# and import the auto-registered one instead.

resource "google_agent_registry_service" "mcp" {
  project      = var.project_id
  location     = var.region
  service_id   = var.mcp_service_name
  display_name = "Acme Tickets MCP server"
  description  = "OAuth 2.1 resource server exposing get_ticket / list_tickets (read-only) and refund_ticket (destructive)."

  interfaces {
    url              = local.mcp_server_url
    protocol_binding = "JSONRPC" # MCP over Streamable HTTP is JSON-RPC 2.0
  }

  # TOOL_SPEC advertises the tools and their annotations. Gateways and VPC-SC key `readOnlyHint`
  # to mcp.tool.isReadOnly, which is why the server sets the hints truthfully.
  # VERIFY: the exact JSON schema expected for TOOL_SPEC content (this mirrors the MCP
  # tools/list result shape; the provider example uses {"tools":[]}).
  mcp_server_spec {
    type = "TOOL_SPEC"
    content = jsonencode({
      tools = [
        {
          name        = "get_ticket"
          description = "Get one ticket by ID."
          annotations = { title = "Get ticket", readOnlyHint = true, destructiveHint = false }
        },
        {
          name        = "list_tickets"
          description = "List tickets visible to the caller (own tickets when delegated)."
          annotations = { title = "List tickets", readOnlyHint = true, destructiveHint = false }
        },
        {
          name        = "refund_ticket"
          description = "Refund a ticket (destructive). Requires tickets:write and a delegated user token."
          annotations = { title = "Refund ticket", readOnlyHint = false, destructiveHint = true, idempotentHint = false }
        },
      ]
    })
  }

  depends_on = [google_project_service.apis]
}

locals {
  # registry_resource = projects/P/locations/L/mcpServers/<ID>; the IAP binding wants the ID.
  mcp_registry_resource_parts = split("/", google_agent_registry_service.mcp.registry_resource)
  mcp_server_registry_id      = element(local.mcp_registry_resource_parts, length(local.mcp_registry_resource_parts) - 1)
}

# IAP policy on the registered MCP server: only this agent may egress to it through the
# gateway. Equivalent gcloud (grant_agent_iam.sh):
#   gcloud iap web add-iam-policy-binding --resource-type=agent-registry --mcp-server=<ID> \
#     --region=<REGION> --member=principal://... --role=roles/iap.egressor
resource "google_iap_agent_registry_mcp_server_iam_member" "agent_egress" {
  count = local.agent_identity_known ? 1 : 0

  project       = var.project_id
  location      = var.region
  mcp_server_id = local.mcp_server_registry_id
  role          = var.mcp_access_role
  member        = local.agent_principal
}
