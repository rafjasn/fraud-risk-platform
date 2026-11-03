variable "name" {
  description = "Name prefix for compute resources."
  type        = string
}

variable "region" {
  description = "AWS region, for log configuration and SDK clients."
  type        = string
}

variable "vpc_id" {
  description = "VPC ID."
  type        = string
}

variable "public_subnet_ids" {
  description = "Subnets of the public load balancer."
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Subnets of tasks, services, and EFS mount targets."
  type        = list(string)
}

variable "image_tag" {
  description = "Immutable image tag, by convention the Git commit."
  type        = string
}

variable "repository_urls" {
  description = "ECR repository URLs: ingestion, training, serving."
  type        = map(string)
}

variable "bucket_names" {
  description = "Bucket names: data, models, mlflow."
  type        = map(string)
}

variable "bucket_arns" {
  description = "Bucket ARNs: data, models, mlflow."
  type        = map(string)
}

variable "database" {
  description = "Database connection details from the database module."
  type = object({
    address           = string
    security_group_id = string
    master_secret_arn = string
    roles_secret_arn  = string
    url_secret_arns   = map(string)
    provision_sql     = string
  })
}

variable "mlflow_experiment_name" {
  description = "MLflow experiment of training cycles."
  type        = string
  default     = "fraud-risk-training-v2"
}

variable "api_certificate_arn" {
  description = "ACM certificate for HTTPS; null serves HTTP only."
  type        = string
  default     = null
}

variable "api_desired_count" {
  description = "Number of API tasks."
  type        = number
}

variable "deletion_protection" {
  description = "Protect the load balancer from deletion."
  type        = bool
}

variable "alarm_topic_arn" {
  description = "Optional SNS topic for alarm notifications."
  type        = string
  default     = null
}

variable "api_latency_alarm_seconds" {
  description = "p95 latency threshold of the API alarm."
  type        = number
}

variable "log_retention_days" {
  description = "Log retention."
  type        = number
}
