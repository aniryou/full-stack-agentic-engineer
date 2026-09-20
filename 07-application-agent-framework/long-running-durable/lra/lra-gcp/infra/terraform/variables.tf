variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-southeast1"
}

variable "image" {
  type        = string
  description = "Container image for both services, e.g. asia-southeast1-docker.pkg.dev/PROJECT/lra/lra:latest"
}

variable "firestore_database" {
  type    = string
  default = "(default)"
}

variable "firestore_location" {
  type    = string
  default = "asia-southeast1"
}

variable "gemini_model" {
  type    = string
  default = "gemini-3.1-flash-lite" # pin to what your project has enabled
}

variable "gemini_location" {
  type    = string
  default = "global"
}

variable "enable_bigquery_sink" {
  type    = bool
  default = false
}

variable "bq_location" {
  type    = string
  default = "asia-southeast1"
}
