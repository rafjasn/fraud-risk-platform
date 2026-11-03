output "cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.this.arn
}

output "task_definition_arns" {
  description = "Task definition ARNs by task: ingestion, seed, jobs, api, mlflow, db-provision."
  value       = { for name, definition in aws_ecs_task_definition.this : name => definition.arn }
}

output "task_role_arns" {
  description = "Task role ARNs by task."
  value       = { for name, role in aws_iam_role.task : name => role.arn }
}

output "execution_role_arn" {
  description = "Task execution role ARN."
  value       = aws_iam_role.execution.arn
}

output "tasks_security_group_id" {
  description = "Security group of one-off tasks."
  value       = aws_security_group.this["tasks"].id
}

output "training_security_group_id" {
  description = "Security group of SageMaker training jobs, allowed to reach MLflow."
  value       = aws_security_group.this["training"].id
}

output "mlflow_tracking_uri" {
  description = "MLflow address inside the VPC."
  value       = local.mlflow_tracking_uri
}

output "mlflow_artifact_uri" {
  description = "S3 location of training artifacts."
  value       = local.mlflow_artifact_uri
}

output "api_url" {
  description = "Public API address."
  value       = "${var.api_certificate_arn == null ? "http" : "https"}://${aws_lb.api.dns_name}"
}

output "api_internal_url" {
  description = "API address inside the VPC, used by verify-deployment."
  value       = local.api_internal_url
}
