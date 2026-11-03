variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "eu-central-1"
}

variable "availability_zones" {
  description = "Two availability zones in the region, for subnets, RDS, and EFS."
  type        = list(string)
  default     = ["eu-central-1a", "eu-central-1b"]

  validation {
    condition     = length(var.availability_zones) == 2
    error_message = "Provide exactly two availability zones."
  }
}

variable "account_id" {
  description = "The AWS account ID, passed in so planning needs no account lookup."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "The account ID has 12 digits."
  }
}

variable "environment" {
  description = "Short environment name used in resource names, such as dev."
  type        = string
  default     = "dev"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,11}$", var.environment))
    error_message = "Use 2-12 lowercase letters, digits, or hyphens, starting with a letter."
  }
}

variable "image_tag" {
  description = "Immutable image tag pushed to every repository, by convention the Git commit. The SageMaker image uses <tag>-sagemaker."
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$", var.image_tag)) && var.image_tag != "latest"
    error_message = "Use an immutable tag such as a Git commit, not latest."
  }
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.40.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class; the source database holds about 24 million transactions."
  type        = string
  default     = "db.t4g.medium"
}

variable "db_allocated_storage_gb" {
  description = "Initial RDS storage; autoscaling may double it."
  type        = number
  default     = 100
}

variable "db_multi_az" {
  description = "Run RDS with a standby in the second availability zone."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Protect the database and load balancer from deletion. Disable before terraform destroy."
  type        = bool
  default     = true
}

variable "api_certificate_arn" {
  description = "ACM certificate for HTTPS on the API load balancer; without one, the API is served over HTTP."
  type        = string
  default     = null
}

variable "api_desired_count" {
  description = "API tasks; one matches the single-worker local setup and the shared state directory."
  type        = number
  default     = 1
}

variable "alarm_topic_arn" {
  description = "Optional SNS topic notified by the API alarms."
  type        = string
  default     = null
}

variable "api_latency_alarm_seconds" {
  description = "p95 target response time that raises the API latency alarm."
  type        = number
  default     = 0.5
}

variable "training_instance_type" {
  description = "SageMaker training instance; images are built for x86_64."
  type        = string
  default     = "ml.m5.2xlarge"
}

variable "training_volume_gb" {
  description = "SageMaker training volume for downloaded snapshot channels."
  type        = number
  default     = 100
}

variable "training_max_runtime_hours" {
  description = "SageMaker stops a training job after this many hours."
  type        = number
  default     = 8
}

variable "training_trials" {
  description = "Optuna trials per recipe in each workflow-started training job."
  type        = number
  default     = 20
}

variable "month_schedule_expression" {
  description = "EventBridge Scheduler expression that starts a simulated month."
  type        = string
  default     = "rate(1 day)"
}

variable "month_schedule_enabled" {
  description = "Start simulated months on the schedule; months run only on demand when false."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Retention of service, job, and workflow logs."
  type        = number
  default     = 30
}
