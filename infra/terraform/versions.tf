terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }
}

# State is local by default. For shared use, add an S3 backend with
# `use_lockfile = true`; see README.md.
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "fraud-risk-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
