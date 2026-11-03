locals {
  partition = "aws"

  data   = var.bucket_arns.data
  models = var.bucket_arns.models
  mlflow = var.bucket_arns.mlflow

  jobs_task_family = replace(var.jobs_task_definition_arn, "/:[0-9]+$/", ":*")
}

# --- SageMaker training job -------------------------------------------------------

resource "aws_iam_role" "sagemaker" {
  name = "${var.name}-sagemaker-training"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = var.account_id } }
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker" {
  name = "training-job"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListStagedInputs"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.data]
        # Prefix channels (configuration, incumbent) are listed; snapshot channels
        # are manifest files naming each key.
        Condition = { StringLike = { "s3:prefix" = ["training-inputs/*"] } }
      },
      {
        Sid    = "ReadInputs"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${local.data}/training-inputs/*",
          "${local.data}/snapshots/*",
          "${local.data}/objects/*",
        ]
      },
      {
        Sid      = "WriteModelOutput"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${local.models}/sagemaker/*"]
      },
      {
        Sid      = "LogMlflowArtifacts"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${local.mlflow}/training-v2/*"]
      },
      {
        Sid      = "ListMlflowArtifacts"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.mlflow]
      },
      {
        Sid      = "PullTrainingImage"
        Effect   = "Allow"
        Action   = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
        Resource = [var.training_repository_arn]
      },
      {
        Sid      = "AuthenticateToEcr"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Sid      = "WriteJobLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = ["arn:${local.partition}:logs:${var.region}:${var.account_id}:log-group:/aws/sagemaker/TrainingJobs*"]
      },
      {
        Sid       = "PublishJobMetrics"
        Effect    = "Allow"
        Action    = ["cloudwatch:PutMetricData"]
        Resource  = ["*"]
        Condition = { StringLike = { "cloudwatch:namespace" = "/aws/sagemaker/*" } }
      },
      {
        # Network interfaces in private subnets, so the job reaches MLflow.
        Sid    = "RunInVpc"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:CreateNetworkInterfacePermission",
          "ec2:DeleteNetworkInterface",
          "ec2:DeleteNetworkInterfacePermission",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcs",
          "ec2:DescribeDhcpOptions",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
        ]
        Resource = ["*"]
      },
    ]
  })
}

# --- Step Functions workflow ------------------------------------------------------

resource "aws_iam_role" "workflow" {
  name = "${var.name}-month-workflow"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = var.account_id } }
    }]
  })
}

resource "aws_iam_role_policy" "workflow" {
  name = "run-month"
  role = aws_iam_role.workflow.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "RunJobsTasks"
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = [local.jobs_task_family]
      },
      {
        Sid       = "TrackJobsTasks"
        Effect    = "Allow"
        Action    = ["ecs:StopTask", "ecs:DescribeTasks"]
        Resource  = ["*"]
        Condition = { ArnEquals = { "ecs:cluster" = var.cluster_arn } }
      },
      {
        Sid       = "PassJobsRoles"
        Effect    = "Allow"
        Action    = ["iam:PassRole"]
        Resource  = [var.jobs_task_role_arn, var.execution_role_arn]
        Condition = { StringEquals = { "iam:PassedToService" = "ecs-tasks.amazonaws.com" } }
      },
      {
        Sid    = "RunTrainingJobs"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:AddTags",
          "sagemaker:ListTags",
        ]
        Resource = ["arn:${local.partition}:sagemaker:${var.region}:${var.account_id}:training-job/*"]
      },
      {
        Sid       = "PassTrainingRole"
        Effect    = "Allow"
        Action    = ["iam:PassRole"]
        Resource  = [aws_iam_role.sagemaker.arn]
        Condition = { StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" } }
      },
      {
        # The .sync integrations learn of completion through these managed rules.
        Sid    = "AwaitCompletion"
        Effect = "Allow"
        Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = [
          "arn:${local.partition}:events:${var.region}:${var.account_id}:rule/StepFunctionsGetEventsForECSTaskRule",
          "arn:${local.partition}:events:${var.region}:${var.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
        ]
      },
      {
        Sid    = "DeliverLogsAndTraces"
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups",
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets",
        ]
        Resource = ["*"]
      },
    ]
  })
}

# --- EventBridge Scheduler ----------------------------------------------------------

resource "aws_iam_role" "scheduler" {
  name = "${var.name}-month-scheduler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = var.account_id } }
    }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  name = "start-month"
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["states:StartExecution"]
      Resource = [aws_sfn_state_machine.month.arn]
    }]
  })
}
