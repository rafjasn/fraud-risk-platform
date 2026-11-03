# ECS on Fargate runs the API and MLflow as services and the CLI as one-off tasks.
# An EFS access point is the shared state directory (releases, deployment.json, cycle
# reports), mounted like Compose's bind mount: read-write for jobs, read-only for the
# API. Services find each other through a private DNS namespace.

locals {
  namespace           = "${var.name}.internal"
  mlflow_tracking_uri = "http://mlflow.${local.namespace}:5000"
  api_internal_url    = "http://api.${local.namespace}:8000"
  mlflow_artifact_uri = "s3://${var.bucket_names.mlflow}/training-v2"
  images = {
    ingestion = "${var.repository_urls.ingestion}:${var.image_tag}"
    training  = "${var.repository_urls.training}:${var.image_tag}"
    serving   = "${var.repository_urls.serving}:${var.image_tag}"
  }
  log_groups = toset(["ingestion", "seed", "jobs", "api", "mlflow", "db-provision"])
}

resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "this" {
  for_each          = local.log_groups
  name              = "/ecs/${var.name}/${each.key}"
  retention_in_days = var.log_retention_days
}

resource "aws_service_discovery_private_dns_namespace" "this" {
  name        = local.namespace
  description = "Private names of the API and MLflow"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "this" {
  for_each = toset(["api", "mlflow"])
  name     = each.key

  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.this.id
    routing_policy = "MULTIVALUE"

    dns_records {
      type = "A"
      ttl  = 10
    }
  }
}
