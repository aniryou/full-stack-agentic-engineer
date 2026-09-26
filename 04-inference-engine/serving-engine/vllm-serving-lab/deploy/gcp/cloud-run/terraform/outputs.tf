output "service_url" {
  description = "Base URL of the OpenAI-compatible API."
  value       = google_cloud_run_v2_service.vllm.uri
}

output "served_model_name" {
  description = "Value for the OpenAI `model` field."
  value       = local.served_model_name
}

output "vllm_args" {
  description = "The exact `vllm serve` arguments the container runs with."
  value       = local.vllm_args
}

output "smoke_test" {
  description = "List models with an identity token (you must be in invoker_members)."
  value       = "curl -sS -H \"Authorization: Bearer $(gcloud auth print-identity-token)\" ${google_cloud_run_v2_service.vllm.uri}/v1/models"
}

output "benchmark" {
  description = "Measure it with the lab's load generator (same command as against the fake server)."
  value       = "SERVELAB_BEARER=$(gcloud auth print-identity-token) python -m servelab bench --url ${google_cloud_run_v2_service.vllm.uri} --rate 2 -n 60 --input-len 512 --output-len 128 --slo-ttft-ms 1000 --slo-tpot-ms 60"
}

output "service_account" {
  description = "Identity the container runs as."
  value       = google_service_account.vllm.email
}
