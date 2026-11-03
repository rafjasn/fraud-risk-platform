variable "name" {
  description = "Name prefix for network resources."
  type        = string
}

variable "region" {
  description = "AWS region, for the S3 endpoint service name."
  type        = string
}

variable "cidr" {
  description = "VPC CIDR block."
  type        = string
}

variable "availability_zones" {
  description = "Availability zones, one public and one private subnet each."
  type        = list(string)
}
