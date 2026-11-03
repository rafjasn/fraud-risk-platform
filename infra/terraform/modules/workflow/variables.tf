variable "name" {
  description = "Name prefix for workflow resources."
  type        = string
}

variable "region" {
  description = "AWS region."
  type        = string
}

variable "account_id" {
  description = "AWS account ID, for ARNs and trust conditions."
  type        = string
}

variable "cluster_arn" {
  description = "ECS cluster of the jobs tasks."
  type        = string
}

variable "private_subnet_ids" {
  description = "Subnets of jobs tasks and training jobs."
  type        = list(string)
}

variable "tasks_security_group_id" {
  description = "Security group of jobs tasks."
  type        = string
}

variable "training_security_group_id" {
  description = "Security group of training jobs."
  type        = string
}

variable "jobs_task_definition_arn" {
  description = "Jobs task definition run by each ECS step."
  type        = string
}

variable "jobs_task_role_arn" {
  description = "Jobs task role, passed to ECS."
  type        = string
}

variable "execution_role_arn" {
  description = "Task execution role, passed to ECS."
  type        = string
}

variable "sagemaker_image" {
  description = "Training image with the SageMaker entrypoint."
  type        = string
}

variable "training_repository_arn" {
  description = "ECR repository of the training image."
  type        = string
}

variable "image_tag" {
  description = "Image tag, recorded as the training code's Git commit."
  type        = string
}

variable "bucket_names" {
  description = "Bucket names: data, models, mlflow."
  type        = map(string)
}

variable "bucket_arns" {
  description = "Bucket ARNs: data, models, mlflow."
  type        = map(string)
}

variable "mlflow_tracking_uri" {
  description = "MLflow address inside the VPC."
  type        = string
}

variable "mlflow_artifact_uri" {
  description = "S3 location of training artifacts."
  type        = string
}

variable "mlflow_experiment_name" {
  description = "MLflow experiment of training cycles."
  type        = string
  default     = "fraud-risk-training-v2"
}

variable "training_instance_type" {
  description = "SageMaker training instance type."
  type        = string
}

variable "training_threads" {
  description = "OpenMP/OpenBLAS threads in training jobs; match the instance's vCPUs."
  type        = number
  default     = 8
}

variable "training_volume_gb" {
  description = "Training volume size."
  type        = number
}

variable "training_max_runtime_hours" {
  description = "Maximum training job runtime."
  type        = number
}

variable "training_trials" {
  description = "Optuna trials per recipe."
  type        = number
}

variable "schedule_expression" {
  description = "When the month workflow starts."
  type        = string
}

variable "schedule_enabled" {
  description = "Whether the schedule starts months."
  type        = bool
}

variable "log_retention_days" {
  description = "Workflow log retention."
  type        = number
}
