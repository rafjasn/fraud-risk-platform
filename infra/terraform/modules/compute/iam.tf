# The execution role pulls images, writes logs, and reads the connection secrets.
# Each task role grants only what its commands touch; the seed and db-provision tasks
# need no AWS access beyond their secrets, so they have no task role.

locals {
  ecs_tasks_trust = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  data_bucket   = var.bucket_arns.data
  mlflow_bucket = var.bucket_arns.mlflow
  models_bucket = var.bucket_arns.models

  task_policies = {
    # Snapshots and reports in the data bucket.
    ingestion = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.data_bucket]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${local.data_bucket}/*"]
      },
    ]
    # Monthly coordination, MLflow artifacts, SageMaker inputs and outputs, and the
    # task-token callback to Step Functions. EFS access is granted by the file system.
    jobs = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.data_bucket, local.mlflow_bucket, local.models_bucket]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${local.data_bucket}/*", "${local.mlflow_bucket}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${local.models_bucket}/sagemaker/*"]
      },
      {
        # Task tokens are not tied to a resource ARN.
        Effect   = "Allow"
        Action   = ["states:SendTaskSuccess", "states:SendTaskFailure"]
        Resource = ["*"]
      },
    ]
    # Releases and deployment state come from EFS; predictions go to PostgreSQL.
    api = []
    mlflow = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.mlflow_bucket]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${local.mlflow_bucket}/*"]
      },
    ]
  }

  secret_arns = concat(
    [var.database.master_secret_arn, var.database.roles_secret_arn],
    values(var.database.url_secret_arns),
  )
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution"
  assume_role_policy = local.ecs_tasks_trust
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "read-connection-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = local.secret_arns
    }]
  })
}

resource "aws_iam_role" "task" {
  for_each           = local.task_policies
  name               = "${var.name}-${each.key}-task"
  assume_role_policy = local.ecs_tasks_trust
}

resource "aws_iam_role_policy" "task" {
  for_each = { for name, statements in local.task_policies : name => statements if length(statements) > 0 }
  name     = "${each.key}-access"
  role     = aws_iam_role.task[each.key].id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = each.value
  })
}
