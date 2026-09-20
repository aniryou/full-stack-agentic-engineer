
# Model Armor: model-side screening of prompts and responses (primer §6). Two layers:
#  1. a *template* the application (ModelArmorScreener) and the Agent Gateway reference
#     explicitly — INSPECT_AND_BLOCK, strict;
#  2. a project *floor setting* that applies to every Gemini call in the project regardless of
#     what the caller asked for — the safety net for agents nobody wired a template into.
# Precedence (docs/sources.md): request template > floor > Gemini built-in safety.

locals {
  rai_filters = ["HATE_SPEECH", "HARASSMENT", "DANGEROUS", "SEXUALLY_EXPLICIT"]
}

resource "google_model_armor_template" "strict" {
  project     = var.project_id
  location    = local.model_armor_location
  template_id = var.model_armor_template_id
  labels      = local.labels

  filter_config {
    # Prompt injection & jailbreak — the ASI01 (goal hijack) detector.
    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "MEDIUM_AND_ABOVE"
    }

    # Links that tool results / users try to smuggle in (exfiltration, phishing).
    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }

    # Sensitive Data Protection, basic infoTypes (credit cards, keys, ...).
    sdp_settings {
      basic_config {
        filter_enforcement = "ENABLED"
      }
    }

    rai_settings {
      dynamic "rai_filters" {
        for_each = local.rai_filters
        content {
          filter_type      = rai_filters.value
          confidence_level = "MEDIUM_AND_ABOVE"
        }
      }
    }
  }

  template_metadata {
    enforcement_type        = "INSPECT_AND_BLOCK"
    log_sanitize_operations = true # findings land in Cloud Logging -> the audit sink
    log_template_operations = true
  }

  depends_on = [google_project_service.apis]
}

# Project floor for the Vertex AI integration. Starts in INSPECT_ONLY so you can watch the
# findings without breaking anyone; flip var.model_armor_floor_inspect_and_block = true once the
# false-positive rate is understood (primer §11.3). gcloud equivalent:
#   gcloud model-armor floorsettings update --full-uri=projects/P/locations/global/floorSetting \
#     --add-integrated-services=VERTEX_AI
resource "google_model_armor_floorsetting" "project" {
  parent   = "projects/${var.project_id}"
  location = "global"

  enable_floor_setting_enforcement = true
  integrated_services              = var.model_armor_integrated_services # REST enum: AI_PLATFORM

  filter_config {
    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "MEDIUM_AND_ABOVE"
    }
    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }
    sdp_settings {
      basic_config {
        filter_enforcement = "ENABLED"
      }
    }
    rai_settings {
      dynamic "rai_filters" {
        for_each = local.rai_filters
        content {
          filter_type      = rai_filters.value
          confidence_level = "MEDIUM_AND_ABOVE"
        }
      }
    }
  }

  # inspect_only / inspect_and_block are mutually exclusive in the provider: set exactly one.
  ai_platform_floor_setting {
    inspect_only         = var.model_armor_floor_inspect_and_block ? null : true
    inspect_and_block    = var.model_armor_floor_inspect_and_block ? true : null
    enable_cloud_logging = true
  }

  depends_on = [google_project_service.apis]
}
