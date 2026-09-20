output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "worker_url" {
  value = google_cloud_run_v2_service.worker.uri
}

output "tasks_queue" {
  value = google_cloud_tasks_queue.steps.id
}

output "workflow" {
  value = google_workflows_workflow.research_approval.id
}
