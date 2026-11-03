mock_provider "aws" {
  source = "./tests/mocks"
}

# Distinct identities where a test must tell two resources apart.
override_resource {
  target = aws_iam_role.task["api"]
  values = { arn = "arn:aws:iam::123456789012:role/fraud-test-api-task" }
}

override_resource {
  target = aws_iam_role.task["jobs"]
  values = { arn = "arn:aws:iam::123456789012:role/fraud-test-jobs-task" }
}

override_resource {
  target = aws_lb_listener.https[0]
  values = { arn = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:listener/app/mock/0123456789abcdef/0000000000000443" }
}

variables {
  name               = "fraud-test"
  region             = "eu-central-1"
  vpc_id             = "vpc-0123"
  public_subnet_ids  = ["subnet-public-a", "subnet-public-b"]
  private_subnet_ids = ["subnet-private-a", "subnet-private-b"]
  image_tag          = "0123abc"
  repository_urls = {
    ingestion = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/fraud-test/ingestion"
    training  = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/fraud-test/training"
    serving   = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/fraud-test/serving"
  }
  bucket_names = { data = "fraud-test-data-1", models = "fraud-test-models-1", mlflow = "fraud-test-mlflow-1" }
  bucket_arns = {
    data   = "arn:aws:s3:::fraud-test-data-1"
    models = "arn:aws:s3:::fraud-test-models-1"
    mlflow = "arn:aws:s3:::fraud-test-mlflow-1"
  }
  database = {
    address           = "fraud-test.abc123.eu-central-1.rds.amazonaws.com"
    security_group_id = "sg-database"
    master_secret_arn = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:rds-master"
    roles_secret_arn  = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:roles"
    url_secret_arns = {
      source_app    = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:source-app-url"
      source_reader = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:source-reader-url"
      platform_app  = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:platform-app-url"
      mlflow_app    = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:mlflow-app-url"
    }
    provision_sql = "SELECT 1;"
  }
  api_desired_count         = 1
  deletion_protection       = true
  api_latency_alarm_seconds = 0.5
  log_retention_days        = 30
}

run "state_directory_is_shared_read_write_for_jobs_and_read_only_for_the_api" {
  command = apply

  module {
    source = "./modules/compute"
  }

  assert {
    condition = [
      for mount in jsondecode(aws_ecs_task_definition.this["jobs"].container_definitions)[0].mountPoints :
      mount.readOnly if mount.containerPath == "/state"
    ] == [false]
    error_message = "Jobs mount the state directory read-write."
  }

  assert {
    condition = [
      for mount in jsondecode(aws_ecs_task_definition.this["api"].container_definitions)[0].mountPoints :
      mount.readOnly if mount.containerPath == "/state"
    ] == [true]
    error_message = "The API mounts the state directory read-only."
  }

  assert {
    condition = alltrue([
      for name in ["ingestion", "seed", "mlflow", "db-provision"] :
      length(aws_ecs_task_definition.this[name].volume) == 1
    ])
    error_message = "Only jobs and the API mount the state directory."
  }

  assert {
    condition = [
      for statement in jsondecode(aws_efs_file_system_policy.state.policy).Statement :
      statement.Action if try(statement.Principal.AWS, null) == aws_iam_role.task["api"].arn
    ] == [["elasticfilesystem:ClientMount"]]
    error_message = "The file system lets the API role mount but not write."
  }

  assert {
    condition     = contains([for statement in jsondecode(aws_efs_file_system_policy.state.policy).Statement : statement.Sid], "DenyInsecureTransport")
    error_message = "The file system refuses NFS without TLS."
  }
}

run "containers_are_hardened_and_get_credentials_only_as_secrets" {
  command = apply

  module {
    source = "./modules/compute"
  }

  assert {
    condition = alltrue([
      for definition in aws_ecs_task_definition.this :
      jsondecode(definition.container_definitions)[0].readonlyRootFilesystem &&
      jsondecode(definition.container_definitions)[0].user != "root"
    ])
    error_message = "Every container runs as a non-root user with a read-only root file system."
  }

  assert {
    condition = alltrue(flatten([
      for definition in aws_ecs_task_definition.this : [
        for variable in jsondecode(definition.container_definitions)[0].environment :
        !contains(["DATABASE_URL", "SOURCE_DATABASE_URL", "PGPASSWORD", "MLFLOW_BACKEND_STORE_URI"], variable.name)
      ]
    ]))
    error_message = "Connection strings and passwords must come from Secrets Manager, not plain environment variables."
  }

  assert {
    condition = {
      for secret in jsondecode(aws_ecs_task_definition.this["api"].container_definitions)[0].secrets :
      secret.name => secret.valueFrom
      } == {
      DATABASE_URL        = var.database.url_secret_arns.platform_app
      SOURCE_DATABASE_URL = var.database.url_secret_arns.source_reader
    }
    error_message = "The API reads the source database with the read-only role."
  }

  assert {
    condition = contains(
      [for secret in jsondecode(aws_ecs_task_definition.this["seed"].container_definitions)[0].secrets : secret.valueFrom],
      var.database.url_secret_arns.source_app,
    )
    error_message = "Only the seed task writes the source database."
  }

  assert {
    condition = {
      for variable in jsondecode(aws_ecs_task_definition.this["jobs"].container_definitions)[0].environment :
      variable.name => variable.value if contains(["STATE_DIR", "WORK_DIR", "MLFLOW_TRACKING_URI"], variable.name)
      } == {
      STATE_DIR           = "/state"
      WORK_DIR            = "/tmp/work"
      MLFLOW_TRACKING_URI = "http://mlflow.fraud-test.internal:5000"
    }
    error_message = "Jobs keep state on EFS, scratch on task storage, and reach MLflow by private DNS."
  }
}

run "task_roles_grant_only_what_each_task_uses" {
  command = apply

  module {
    source = "./modules/compute"
  }

  assert {
    condition     = !contains(keys(aws_iam_role_policy.task), "api")
    error_message = "The API needs no AWS permissions beyond its file system and secrets."
  }

  assert {
    condition = contains(
      flatten([for statement in jsondecode(aws_iam_role_policy.task["jobs"].policy).Statement : statement.Resource if contains(statement.Action, "s3:GetObject")]),
      "arn:aws:s3:::fraud-test-models-1/sagemaker/*",
    )
    error_message = "Jobs read SageMaker outputs to import releases."
  }

  assert {
    condition = alltrue([
      for statement in jsondecode(aws_iam_role_policy.task["jobs"].policy).Statement :
      !contains(statement.Resource, "arn:aws:s3:::fraud-test-models-1/*")
    ])
    error_message = "Jobs read only the sagemaker/ prefix of the models bucket."
  }

  assert {
    condition = alltrue(flatten([
      for policy in aws_iam_role_policy.task : [
        for statement in jsondecode(policy.policy).Statement : !contains(statement.Action, "*") && !contains(statement.Action, "s3:*")
      ]
    ]))
    error_message = "Task policies list their actions explicitly."
  }
}

run "network_access_is_limited_to_named_clients" {
  command = apply

  module {
    source = "./modules/compute"
  }

  assert {
    condition = alltrue([
      for rule in aws_vpc_security_group_ingress_rule.database :
      rule.cidr_ipv4 == null && rule.from_port == 5432 && rule.security_group_id == "sg-database"
    ]) && length(aws_vpc_security_group_ingress_rule.database) == 3
    error_message = "PostgreSQL accepts only the tasks, API, and MLflow security groups."
  }

  assert {
    condition     = toset(keys(aws_vpc_security_group_ingress_rule.internal)) == toset(["alb-to-api", "tasks-to-api", "tasks-to-mlflow", "training-to-mlflow", "tasks-to-efs", "api-to-efs"])
    error_message = "MLflow is reachable only from tasks and training jobs; EFS only from tasks and the API."
  }

  assert {
    condition     = length(aws_ecs_service.this["mlflow"].load_balancer) == 0
    error_message = "MLflow is not behind the public load balancer."
  }
}

run "http_api_hides_metrics_and_rolls_back_failed_releases" {
  command = apply

  module {
    source = "./modules/compute"
  }

  assert {
    condition     = aws_lb_listener.http.default_action[0].type == "forward" && length(aws_lb_listener.https) == 0
    error_message = "Without a certificate the API is served over HTTP."
  }

  assert {
    condition = (
      aws_lb_listener_rule.hide_metrics.listener_arn == aws_lb_listener.http.arn &&
      one(aws_lb_listener_rule.hide_metrics.condition).path_pattern[0].values == toset(["/metrics"]) &&
      aws_lb_listener_rule.hide_metrics.action[0].fixed_response[0].status_code == "404"
    )
    error_message = "The load balancer answers /metrics with 404."
  }

  assert {
    condition     = aws_lb_target_group.api.health_check[0].path == "/health"
    error_message = "The load balancer checks /health."
  }

  assert {
    condition = alltrue([
      for service in aws_ecs_service.this :
      service.deployment_circuit_breaker[0].enable && service.deployment_circuit_breaker[0].rollback
    ])
    error_message = "A deployment that fails its health checks rolls back."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.api_latency.extended_statistic == "p95" && aws_cloudwatch_metric_alarm.api_latency.threshold == 0.5
    error_message = "The latency alarm watches p95 against the configured threshold."
  }
}

run "https_api_redirects_http" {
  command = apply

  module {
    source = "./modules/compute"
  }

  variables {
    api_certificate_arn = "arn:aws:acm:eu-central-1:123456789012:certificate/abc"
  }

  assert {
    condition     = aws_lb_listener.http.default_action[0].type == "redirect" && length(aws_lb_listener.https) == 1
    error_message = "With a certificate, HTTP redirects to HTTPS."
  }

  assert {
    condition     = aws_lb_listener_rule.hide_metrics.listener_arn == aws_lb_listener.https[0].arn
    error_message = "The /metrics rule sits on the listener that forwards traffic."
  }

  assert {
    condition     = toset(keys(aws_vpc_security_group_ingress_rule.alb)) == toset(["80", "443"])
    error_message = "The load balancer accepts HTTP and HTTPS."
  }
}
