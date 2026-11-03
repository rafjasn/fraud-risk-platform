output "api_url" {
  description = "Public API address."
  value       = module.compute.api_url
}

output "repository_urls" {
  description = "Push images here, tagged with image_tag (and <tag>-sagemaker for the training entrypoint)."
  value       = module.storage.repository_urls
}

output "bucket_names" {
  description = "Data, model-output, and MLflow buckets."
  value       = module.storage.bucket_names
}

output "cluster_arn" {
  description = "ECS cluster for one-off tasks."
  value       = module.compute.cluster_arn
}

output "task_definition_arns" {
  description = "Task definitions for aws ecs run-task: ingestion, seed, jobs, db-provision."
  value       = module.compute.task_definition_arns
}

output "run_task_network_configuration" {
  description = "--network-configuration value for aws ecs run-task."
  value = format(
    "awsvpcConfiguration={subnets=[%s],securityGroups=[%s],assignPublicIp=DISABLED}",
    join(",", module.network.private_subnet_ids),
    module.compute.tasks_security_group_id,
  )
}

output "state_machine_arn" {
  description = "Start a simulated month with aws stepfunctions start-execution."
  value       = module.workflow.state_machine_arn
}
