# One simulated month as a Step Functions Standard workflow:
#   PrepareMonth   ECS jobs task: `month --stop-before-training` (replay, advance,
#                  monitor, with the coordinator's guards); returns the snapshot.
#   StageInputs    ECS jobs task: `sagemaker-inputs`; returns InputDataConfig.
#   Train          SageMaker training job, awaited with .sync.
#   ImportRelease  ECS jobs task: `import-release` of the job's model.tar.gz.
# Promotion stays an operator step. A failed execution is resumed with Step
# Functions redrive. A second start while a month is still training does not advance
# again: PrepareMonth resumes the same snapshot, and SageMaker refuses the repeated
# job name.

locals {
  ecs_task = {
    Cluster         = var.cluster_arn
    TaskDefinition  = var.jobs_task_definition_arn
    LaunchType      = "FARGATE"
    PlatformVersion = "LATEST"
    NetworkConfiguration = {
      AwsvpcConfiguration = {
        Subnets        = var.private_subnet_ids
        SecurityGroups = [var.tasks_security_group_id]
        AssignPublicIp = "DISABLED"
      }
    }
  }

  # The container returns its command's JSON result with this token.
  task_token = [{ Name = "STEP_FUNCTIONS_TASK_TOKEN", "Value.$" = "$$.Task.Token" }]

  # Capacity and throttling errors from RunTask or CreateTrainingJob, not job failures.
  ecs_retry       = [{ ErrorEquals = ["ECS.AmazonECSException"], IntervalSeconds = 30, MaxAttempts = 3, BackoffRate = 2 }]
  sagemaker_retry = [{ ErrorEquals = ["SageMaker.AmazonSageMakerException"], IntervalSeconds = 60, MaxAttempts = 2, BackoffRate = 2 }]

  definition = {
    Comment = "One simulated month: prepare in ECS, train on SageMaker, import the result. Promotion stays manual."
    StartAt = "PrepareMonth"
    States = {
      PrepareMonth = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.waitForTaskToken"
        Parameters = merge(local.ecs_task, {
          Overrides = {
            ContainerOverrides = [{
              Name        = "jobs"
              Command     = ["month", "--stop-before-training"]
              Environment = local.task_token
            }]
          }
        })
        # A task that stops without reporting fails the state after this long.
        TimeoutSeconds = 6 * 3600
        ResultPath     = "$.month"
        Retry          = local.ecs_retry
        Next           = "TrainOrFinish"
      }
      TrainOrFinish = {
        Type = "Choice"
        Choices = [
          { Variable = "$.month.status", StringEquals = "complete", Next = "SimulationComplete" },
          { Variable = "$.month.status", StringEquals = "holdout_reached", Next = "HoldoutReached" },
        ]
        Default = "StageInputs"
      }
      SimulationComplete = {
        Type    = "Succeed"
        Comment = "The simulation has reached the end of the source data."
      }
      HoldoutReached = {
        Type    = "Succeed"
        Comment = "The next snapshot reaches the reserved holdout; run the jobs task with final-evaluation."
      }
      StageInputs = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.waitForTaskToken"
        Parameters = merge(local.ecs_task, {
          Overrides = {
            ContainerOverrides = [{
              Name        = "jobs"
              "Command.$" = "States.Array('sagemaker-inputs', '--snapshot', $.month.snapshot_id, '--trials', '${var.training_trials}')"
              Environment = local.task_token
            }]
          }
        })
        TimeoutSeconds = 3600
        ResultPath     = "$.staged"
        Retry          = local.ecs_retry
        Next           = "Train"
      }
      Train = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters = {
          # Job names are unique forever, so a redrive or retry needs its own name.
          "TrainingJobName.$" = "States.Format('{}-r{}-t{}', $.staged.training_job_name, $$.Execution.RedriveCount, $$.State.RetryCount)"
          AlgorithmSpecification = {
            TrainingImage     = var.sagemaker_image
            TrainingInputMode = "File"
            MetricDefinitions = [
              for name in ["validation_average_precision", "validation_recall_at_top_0.5pct", "evaluation_candidate_average_precision"] :
              { Name = name, Regex = "${replace(name, ".", "\\.")}=([0-9.]+)" }
            ]
          }
          RoleArn             = aws_iam_role.sagemaker.arn
          "InputDataConfig.$" = "$.staged.input_data_config"
          OutputDataConfig    = { S3OutputPath = "s3://${var.bucket_names.models}/sagemaker/" }
          ResourceConfig      = { InstanceCount = 1, InstanceType = var.training_instance_type, VolumeSizeInGB = var.training_volume_gb }
          StoppingCondition   = { MaxRuntimeInSeconds = var.training_max_runtime_hours * 3600 }
          VpcConfig           = { SecurityGroupIds = [var.training_security_group_id], Subnets = var.private_subnet_ids }
          Environment = {
            MLFLOW_TRACKING_URI       = var.mlflow_tracking_uri
            MLFLOW_ARTIFACT_URI       = var.mlflow_artifact_uri
            MLFLOW_EXPERIMENT_NAME    = var.mlflow_experiment_name
            MLFLOW_DISABLE_AGENT_HINT = "1"
            TRAINING_IMAGE_ID         = var.sagemaker_image
            TRAINING_GIT_COMMIT       = var.image_tag
            OMP_NUM_THREADS           = tostring(var.training_threads)
            OPENBLAS_NUM_THREADS      = tostring(var.training_threads)
          }
          Tags = [{ Key = "TrainingRequest", "Value.$" = "$.staged.job_id" }]
        }
        ResultSelector = {
          "TrainingJobName.$"   = "$.TrainingJobName"
          "TrainingJobStatus.$" = "$.TrainingJobStatus"
          "ModelArtifacts.$"    = "$.ModelArtifacts"
        }
        ResultPath = "$.training"
        Retry      = local.sagemaker_retry
        Next       = "ImportRelease"
      }
      ImportRelease = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = merge(local.ecs_task, {
          Overrides = {
            ContainerOverrides = [{
              Name        = "jobs"
              "Command.$" = "States.Array('import-release', '--archive', $.training.ModelArtifacts.S3ModelArtifacts)"
            }]
          }
        })
        ResultPath = null
        Retry      = local.ecs_retry
        Next       = "Trained"
      }
      Trained = {
        Type    = "Succeed"
        Comment = "The cycle report recommends whether to promote; promotion is an operator decision."
      }
    }
  }
}

resource "aws_cloudwatch_log_group" "workflow" {
  name              = "/aws/vendedlogs/states/${var.name}-month"
  retention_in_days = var.log_retention_days
}

resource "aws_sfn_state_machine" "month" {
  name       = "${var.name}-month"
  type       = "STANDARD"
  role_arn   = aws_iam_role.workflow.arn
  definition = jsonencode(local.definition)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.workflow.arn}:*"
    include_execution_data = true
    level                  = "ERROR"
  }

  tracing_configuration {
    enabled = true
  }
}

# Off by default: months then start only on demand. Scheduler retries are disabled,
# because a retried start would begin another month.
resource "aws_scheduler_schedule" "month" {
  name                = "${var.name}-month"
  schedule_expression = var.schedule_expression
  state               = var.schedule_enabled ? "ENABLED" : "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.month.arn
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({})

    retry_policy {
      maximum_retry_attempts       = 0
      maximum_event_age_in_seconds = 3600
    }
  }
}
