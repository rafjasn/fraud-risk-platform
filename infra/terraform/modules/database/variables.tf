variable "name" {
  description = "Name prefix for database resources."
  type        = string
}

variable "vpc_id" {
  description = "VPC of the database security group."
  type        = string
}

variable "subnet_ids" {
  description = "Private subnets in two availability zones."
  type        = list(string)
}

variable "instance_class" {
  description = "RDS instance class."
  type        = string
}

variable "allocated_storage_gb" {
  description = "Initial storage in GiB."
  type        = number
}

variable "multi_az" {
  description = "Run a standby instance."
  type        = bool
}

variable "deletion_protection" {
  description = "Protect the instance from deletion; when disabled, destroy skips the final snapshot."
  type        = bool
}
