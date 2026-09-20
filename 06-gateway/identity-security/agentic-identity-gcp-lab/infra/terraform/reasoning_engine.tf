
# The agent runtime: Agent Engine (resource type reasoningEngines) with
# spec.identity_type = "AGENT_IDENTITY" — the one line that turns "a container with a service
# account" into a first-class, certificate-bound principal (primer §3.3).
#
# Two deployment paths, pick one (README.md):
#   A. Python SDK (default): infra/scripts/deploy_agent_engine.py packages the ADK app with
#      vertexai.Client(...).agent_engines.create(agent=AdkApp(...), config={"identity_type":
#      AGENT_IDENTITY, ...}). Then set var.agent_engine_id so the IAM bindings resolve.
#   B. Terraform (var.deploy_agent_with_terraform = true): source-based deployment. Terraform
#      needs a .tar.gz of the app (base64-inlined via filebase64), produced by
#      infra/scripts/build_agent_source.sh, and `class_methods` must be declared by hand so SDK
#      clients can discover the ADK methods (provider docs). Path B keeps the whole identity ->
#      policy graph in one plan; path A is what the Google docs and ADK CLI do.

locals {
  # Environment the deployed app reads through agentsec.config.Settings.from_env().
  agent_env = merge(
    {
      AGENTSEC_PROFILE                = "gcp"
      AGENTSEC_PROJECT_ID             = var.project_id
      AGENTSEC_PROJECT_NUMBER         = var.project_number
      AGENTSEC_LOCATION               = var.region
      AGENTSEC_MODEL                  = var.agent_model
      AGENTSEC_MODEL_ARMOR_TEMPLATE   = google_model_armor_template.strict.name
      AGENTSEC_AUTH_PROVIDER          = google_agent_identity_auth_provider.crm.name
      AGENTSEC_AUTH_PROVIDER_LOCATION = local.auth_provider_location
      AGENTSEC_MCP_URL                = local.mcp_server_url
      AGENTSEC_AUDIT_LOG              = var.audit_log_name
      AGENTSEC_POLICY                 = "policies/support-agent.yaml" # bundled by build_agent_source.sh
    },
    var.org_id == null ? {} : { AGENTSEC_ORG_ID = var.org_id },
  )

  # Standard ADK AdkApp method surface (from the provider docs / ADK cli_deploy.py). Required
  # only for the Terraform path; the SDK path registers these automatically.
  adk_class_methods = [
    {
      name        = "get_session"
      api_mode    = ""
      description = "Retrieve session by ID"
      parameters = {
        type       = "object"
        required   = ["user_id", "session_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" } }
      }
    },
    {
      name        = "async_get_session"
      api_mode    = "async"
      description = "Retrieve session asynchronously by ID"
      parameters = {
        type       = "object"
        required   = ["user_id", "session_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" } }
      }
    },
    {
      name        = "list_sessions"
      api_mode    = ""
      description = "List all sessions for a user"
      parameters = {
        type       = "object"
        required   = ["user_id"]
        properties = { user_id = { type = "string" } }
      }
    },
    {
      name        = "async_list_sessions"
      api_mode    = "async"
      description = "List all sessions for a user asynchronously"
      parameters = {
        type       = "object"
        required   = ["user_id"]
        properties = { user_id = { type = "string" } }
      }
    },
    {
      name        = "create_session"
      api_mode    = ""
      description = "Create a new session"
      parameters = {
        type       = "object"
        required   = ["user_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" }, state = { type = "object" } }
      }
    },
    {
      name        = "async_create_session"
      api_mode    = "async"
      description = "Create a new session asynchronously"
      parameters = {
        type       = "object"
        required   = ["user_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" }, state = { type = "object" } }
      }
    },
    {
      name        = "delete_session"
      api_mode    = ""
      description = "Delete session by ID"
      parameters = {
        type       = "object"
        required   = ["user_id", "session_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" } }
      }
    },
    {
      name        = "async_delete_session"
      api_mode    = "async"
      description = "Delete session asynchronously by ID"
      parameters = {
        type       = "object"
        required   = ["user_id", "session_id"]
        properties = { user_id = { type = "string" }, session_id = { type = "string" } }
      }
    },
    {
      name        = "stream_query"
      api_mode    = "stream"
      description = "Stream queries from the agent"
      parameters = {
        type     = "object"
        required = ["message", "user_id"]
        properties = {
          message    = { description = "Message string or object" }
          user_id    = { type = "string" }
          session_id = { type = "string" }
          run_config = { type = "object" }
        }
      }
    },
    {
      name        = "async_stream_query"
      api_mode    = "async_stream"
      description = "Stream queries asynchronously from the agent"
      parameters = {
        type     = "object"
        required = ["message", "user_id"]
        properties = {
          message        = { description = "Message string or object" }
          user_id        = { type = "string" }
          session_id     = { type = "string" }
          session_events = { type = "array", items = { type = "object" } }
          run_config     = { type = "object" }
        }
      }
    },
    {
      name        = "streaming_agent_run_with_events"
      api_mode    = "async_stream"
      description = "Stream agent run with events asynchronously"
      parameters = {
        type       = "object"
        required   = ["request_json"]
        properties = { request_json = { type = "string" } }
      }
    },
  ]
}

resource "google_vertex_ai_reasoning_engine" "agent" {
  count = var.deploy_agent_with_terraform ? 1 : 0

  project      = var.project_id
  region       = var.region
  display_name = var.agent_display_name
  description  = "agentsec reference support agent (ADK) running with Agent Identity"
  labels       = local.labels

  # Optional CMEK for the engine and its sub-resources (sessions, memory).
  dynamic "encryption_spec" {
    for_each = var.kms_key_name == null ? [] : [var.kms_key_name]
    content {
      kms_key_name = encryption_spec.value
    }
  }

  spec {
    # THE line. With AGENT_IDENTITY `service_account` must stay unset: the platform issues a
    # SPIFFE certificate for spiffe://<trust domain>/resources/aiplatform/.../reasoningEngines/<id>.
    identity_type   = "AGENT_IDENTITY"
    agent_framework = "google-adk"
    class_methods   = jsonencode(local.adk_class_methods)

    source_code_spec {
      # Input-only base64 tarball. Layout produced by build_agent_source.sh:
      #   deploy/agent_engine_app.py  agentsec/  policies/  requirements-gcp.txt
      inline_source {
        source_archive = filebase64(var.agent_source_archive)
      }

      python_spec {
        entrypoint_module = "deploy.agent_engine_app"
        entrypoint_object = "app"
        requirements_file = "requirements-gcp.txt"
        version           = "3.11"
      }
    }

    deployment_spec {
      min_instances         = 0
      max_instances         = var.agent_max_instances
      container_concurrency = 9

      resource_limits = {
        cpu    = "1"
        memory = "2Gi"
      }

      dynamic "env" {
        for_each = local.agent_env
        content {
          name  = env.key
          value = env.value
        }
      }

      # The one secret, resolved by the platform at deploy time (never in the image or state).
      secret_env {
        name = "AGENTSEC_SECRET_DEMO"
        secret_ref {
          secret  = google_secret_manager_secret.demo.secret_id
          version = "latest"
        }
      }

      # Route all outbound traffic through the egress gateway when it is enabled.
      dynamic "agent_gateway_config" {
        for_each = var.enable_agent_gateway ? [1] : []
        content {
          agent_to_anywhere_config {
            agent_gateway = google_network_services_agent_gateway.egress[0].id
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.demo,
    google_secret_manager_secret_iam_member.reasoning_engine_service_agent,
  ]
}
