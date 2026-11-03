# Every container runs as 1000:1000 with a read-only root file system; /tmp is task
# storage. One-off tasks take their command as an override, for example
# ["month", "--stop-before-training"] for jobs or ["seed-source", "--raw-dir",
# "/tmp/raw"] for seed. The default commands only report status.

locals {
  url = var.database.url_secret_arns

  base_environment = {
    AWS_DEFAULT_REGION        = var.region
    FRAUD_DATA_URI            = "s3://${var.bucket_names.data}"
    HOME                      = "/tmp"
    MPLCONFIGDIR              = "/tmp/matplotlib"
    MLFLOW_DISABLE_AGENT_HINT = "1"
  }

  tasks = {
    # Snapshot commands: initial-snapshot, next-snapshot, snapshots, verify-snapshot.
    ingestion = {
      image       = local.images.ingestion
      cpu         = 2048
      memory      = 8192
      storage_gb  = 100
      user        = "1000:1000"
      entry_point = null
      command     = ["snapshots", "list"]
      state       = null
      port        = null
      environment = { STATE_DIR = "/tmp/state", FRAUD_DUCKDB_MEMORY_LIMIT = "4GB" }
      secrets     = { DATABASE_URL = local.url.platform_app, SOURCE_DATABASE_URL = local.url.source_reader }
    }
    # Writes the source database: init-db, then seed-source with --raw-dir /tmp/raw.
    seed = {
      image       = local.images.ingestion
      cpu         = 2048
      memory      = 8192
      storage_gb  = 150
      user        = "1000:1000"
      entry_point = null
      command     = ["init-db"]
      state       = null
      port        = null
      environment = { STATE_DIR = "/tmp/state" }
      secrets     = { DATABASE_URL = local.url.platform_app, SOURCE_DATABASE_URL = local.url.source_app }
    }
    # Everything that reads or writes the shared state directory: month, replay,
    # monitor, sagemaker-inputs, import-release, promote, rollback, verify-deployment.
    jobs = {
      image       = local.images.training
      cpu         = 4096
      memory      = 16384
      storage_gb  = 100
      user        = "1000:1000"
      entry_point = null
      command     = ["status"]
      state       = "read-write"
      port        = null
      environment = {
        STATE_DIR                 = "/state"
        WORK_DIR                  = "/tmp/work"
        FRAUD_DUCKDB_MEMORY_LIMIT = "8GB"
        MLFLOW_TRACKING_URI       = local.mlflow_tracking_uri
        MLFLOW_ARTIFACT_URI       = local.mlflow_artifact_uri
        MLFLOW_EXPERIMENT_NAME    = var.mlflow_experiment_name
        TRAINING_IMAGE_ID         = local.images.training
        TRAINING_GIT_COMMIT       = var.image_tag
        API_URL                   = local.api_internal_url
        OMP_NUM_THREADS           = "4"
        OPENBLAS_NUM_THREADS      = "4"
      }
      secrets = { DATABASE_URL = local.url.platform_app, SOURCE_DATABASE_URL = local.url.source_reader }
    }
    api = {
      image       = local.images.serving
      cpu         = 1024
      memory      = 2048
      storage_gb  = 21
      user        = "1000:1000"
      entry_point = null
      command     = null
      state       = "read-only"
      port        = 8000
      environment = { STATE_DIR = "/state" }
      secrets     = { DATABASE_URL = local.url.platform_app, SOURCE_DATABASE_URL = local.url.source_reader }
    }
    mlflow = {
      image       = local.images.training
      cpu         = 1024
      memory      = 4096
      storage_gb  = 21
      user        = "1000:1000"
      entry_point = ["sh", "-c"]
      command = [join(" ", [
        "exec mlflow server --host 0.0.0.0 --port 5000 --workers 1",
        "--backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\"",
        "--serve-artifacts --artifacts-destination s3://${var.bucket_names.mlflow}/proxied",
        "--allowed-hosts mlflow.${local.namespace}:5000,localhost:*",
      ])]
      state       = null
      port        = 5000
      environment = {}
      secrets     = { MLFLOW_BACKEND_STORE_URI = local.url.mlflow_app }
    }
    # Creates roles and databases as the RDS master user; repeatable.
    db-provision = {
      image       = "public.ecr.aws/docker/library/postgres:17.6-bookworm"
      cpu         = 256
      memory      = 512
      storage_gb  = 21
      user        = "postgres"
      entry_point = ["sh", "-c"]
      command = [join(" ", [
        "printf '%s' \"$PROVISION_SQL\" > /tmp/provision.sql &&",
        "exec psql -X",
        "-v source_app_password=\"$SOURCE_APP_PASSWORD\"",
        "-v source_reader_password=\"$SOURCE_READER_PASSWORD\"",
        "-v platform_app_password=\"$PLATFORM_APP_PASSWORD\"",
        "-v mlflow_app_password=\"$MLFLOW_APP_PASSWORD\"",
        "-f /tmp/provision.sql",
      ])]
      state = null
      port  = null
      environment = {
        PGHOST        = var.database.address
        PGDATABASE    = "fraud"
        PGSSLMODE     = "require"
        PROVISION_SQL = var.database.provision_sql
      }
      secrets = {
        PGUSER                 = "${var.database.master_secret_arn}:username::"
        PGPASSWORD             = "${var.database.master_secret_arn}:password::"
        SOURCE_APP_PASSWORD    = "${var.database.roles_secret_arn}:source_app::"
        SOURCE_READER_PASSWORD = "${var.database.roles_secret_arn}:source_reader::"
        PLATFORM_APP_PASSWORD  = "${var.database.roles_secret_arn}:platform_app::"
        MLFLOW_APP_PASSWORD    = "${var.database.roles_secret_arn}:mlflow_app::"
      }
    }
  }
}

resource "aws_ecs_task_definition" "this" {
  for_each                 = local.tasks
  family                   = "${var.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = contains(keys(local.task_policies), each.key) ? aws_iam_role.task[each.key].arn : null

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  dynamic "ephemeral_storage" {
    for_each = each.value.storage_gb > 20 ? [each.value.storage_gb] : []
    content {
      size_in_gib = ephemeral_storage.value
    }
  }

  volume {
    name = "tmp"
  }

  dynamic "volume" {
    for_each = each.value.state == null ? [] : ["state"]
    content {
      name = "state"

      efs_volume_configuration {
        file_system_id     = aws_efs_file_system.state.id
        transit_encryption = "ENABLED"

        authorization_config {
          access_point_id = aws_efs_access_point.state.id
          iam             = "ENABLED"
        }
      }
    }
  }

  container_definitions = jsonencode([merge(
    {
      name                   = each.key
      image                  = each.value.image
      essential              = true
      user                   = each.value.user
      readonlyRootFilesystem = true
      environment = [
        for name, value in merge(local.base_environment, each.value.environment) :
        { name = name, value = value }
      ]
      secrets = [for name, arn in each.value.secrets : { name = name, valueFrom = arn }]
      mountPoints = concat(
        [{ sourceVolume = "tmp", containerPath = "/tmp", readOnly = false }],
        each.value.state == null ? [] : [{
          sourceVolume  = "state"
          containerPath = "/state"
          readOnly      = each.value.state == "read-only"
        }],
      )
      portMappings = each.value.port == null ? [] : [{ containerPort = each.value.port, protocol = "tcp" }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.this[each.key].name
          awslogs-region        = var.region
          awslogs-stream-prefix = each.key
        }
      }
    },
    each.value.entry_point == null ? {} : { entryPoint = each.value.entry_point },
    each.value.command == null ? {} : { command = each.value.command },
  )])
}
