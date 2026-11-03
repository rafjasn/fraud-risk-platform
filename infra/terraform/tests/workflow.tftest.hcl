mock_provider "aws" {
  source = "./tests/mocks"
}

variables {
  name                       = "fraud-test"
  region                     = "eu-central-1"
  account_id                 = "123456789012"
  cluster_arn                = "arn:aws:ecs:eu-central-1:123456789012:cluster/fraud-test"
  private_subnet_ids         = ["subnet-private-a", "subnet-private-b"]
  tasks_security_group_id    = "sg-tasks"
  training_security_group_id = "sg-training"
  jobs_task_definition_arn   = "arn:aws:ecs:eu-central-1:123456789012:task-definition/fraud-test-jobs:7"
  jobs_task_role_arn         = "arn:aws:iam::123456789012:role/fraud-test-jobs-task"
  execution_role_arn         = "arn:aws:iam::123456789012:role/fraud-test-ecs-execution"
  sagemaker_image            = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/fraud-test/training:0123abc-sagemaker"
  training_repository_arn    = "arn:aws:ecr:eu-central-1:123456789012:repository/fraud-test/training"
  image_tag                  = "0123abc"
  bucket_names               = { data = "fraud-test-data-1", models = "fraud-test-models-1", mlflow = "fraud-test-mlflow-1" }
  bucket_arns = {
    data   = "arn:aws:s3:::fraud-test-data-1"
    models = "arn:aws:s3:::fraud-test-models-1"
    mlflow = "arn:aws:s3:::fraud-test-mlflow-1"
  }
  mlflow_tracking_uri        = "http://mlflow.fraud-test.internal:5000"
  mlflow_artifact_uri        = "s3://fraud-test-mlflow-1/training-v2"
  training_instance_type     = "ml.m5.2xlarge"
  training_volume_gb         = 100
  training_max_runtime_hours = 8
  training_trials            = 20
  schedule_expression        = "rate(1 day)"
  schedule_enabled           = false
  log_retention_days         = 30
}

run "month_prepares_in_ecs_trains_on_sagemaker_and_imports" {
  command = apply

  module {
    source = "./modules/workflow"
  }

  assert {
    condition     = jsondecode(aws_sfn_state_machine.month.definition).StartAt == "PrepareMonth"
    error_message = "A month starts by preparing in ECS."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.PrepareMonth.Resource == "arn:aws:states:::ecs:runTask.waitForTaskToken" &&
      jsondecode(aws_sfn_state_machine.month.definition).States.PrepareMonth.Parameters.Overrides.ContainerOverrides[0].Command == ["month", "--stop-before-training"] &&
      jsondecode(aws_sfn_state_machine.month.definition).States.PrepareMonth.Parameters.Overrides.ContainerOverrides[0].Environment[0]["Value.$"] == "$$.Task.Token"
    )
    error_message = "PrepareMonth runs the month coordinator up to training and waits for its reported snapshot."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.TrainOrFinish.Choices[0].StringEquals == "complete" &&
      jsondecode(aws_sfn_state_machine.month.definition).States.TrainOrFinish.Choices[1].StringEquals == "holdout_reached" &&
      jsondecode(aws_sfn_state_machine.month.definition).States[jsondecode(aws_sfn_state_machine.month.definition).States.TrainOrFinish.Choices[1].Next].Type == "Succeed" &&
      jsondecode(aws_sfn_state_machine.month.definition).States.TrainOrFinish.Default == "StageInputs"
    )
    error_message = "The workflow ends when the simulation is complete or reaches the holdout, and trains otherwise."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.StageInputs.Parameters.Overrides.ContainerOverrides[0]["Command.$"] ==
      "States.Array('sagemaker-inputs', '--snapshot', $.month.snapshot_id, '--trials', '20')"
    )
    error_message = "StageInputs stages the snapshot the month reported."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.Train.Resource == "arn:aws:states:::sagemaker:createTrainingJob.sync" &&
      jsondecode(aws_sfn_state_machine.month.definition).States.Train.Parameters["InputDataConfig.$"] == "$.staged.input_data_config" &&
      strcontains(jsondecode(aws_sfn_state_machine.month.definition).States.Train.Parameters["TrainingJobName.$"], "$$.Execution.RedriveCount")
    )
    error_message = "Train waits for a SageMaker job on the staged channels, with a new name per redrive."
  }

  assert {
    condition     = jsondecode(aws_sfn_state_machine.month.definition).States.Train.Parameters.AlgorithmSpecification.TrainingImage == var.sagemaker_image
    error_message = "Training runs the SageMaker entrypoint image."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.Train.Parameters.VpcConfig.SecurityGroupIds == ["sg-training"] &&
      jsondecode(aws_sfn_state_machine.month.definition).States.Train.Parameters.Environment.MLFLOW_TRACKING_URI == var.mlflow_tracking_uri
    )
    error_message = "Training jobs run in the VPC so they reach MLflow."
  }

  assert {
    condition = (
      jsondecode(aws_sfn_state_machine.month.definition).States.ImportRelease.Parameters.Overrides.ContainerOverrides[0]["Command.$"] ==
      "States.Array('import-release', '--archive', $.training.ModelArtifacts.S3ModelArtifacts)"
    )
    error_message = "The job's model.tar.gz is imported as a verified release."
  }

  assert {
    condition     = !strcontains(aws_sfn_state_machine.month.definition, "'promote'") && !strcontains(aws_sfn_state_machine.month.definition, "\"promote\"")
    error_message = "Promotion stays an operator step outside the workflow."
  }

  assert {
    condition     = aws_sfn_state_machine.month.logging_configuration[0].level == "ERROR" && aws_sfn_state_machine.month.tracing_configuration[0].enabled
    error_message = "The workflow logs errors and traces executions."
  }
}

run "roles_are_scoped_to_the_workflow" {
  command = apply

  module {
    source = "./modules/workflow"
  }

  assert {
    condition = [
      for statement in jsondecode(aws_iam_role_policy.workflow.policy).Statement :
      statement.Resource if statement.Sid == "RunJobsTasks"
    ] == [["arn:aws:ecs:eu-central-1:123456789012:task-definition/fraud-test-jobs:*"]]
    error_message = "The workflow may run only the jobs task family."
  }

  assert {
    condition = alltrue([
      for statement in jsondecode(aws_iam_role_policy.workflow.policy).Statement :
      try(statement.Condition.StringEquals["iam:PassedToService"] != null, false) if contains(statement.Action, "iam:PassRole")
    ])
    error_message = "Roles can be passed only to their intended service."
  }

  assert {
    condition = toset(flatten([
      for statement in jsondecode(aws_iam_role_policy.sagemaker.policy).Statement :
      statement.Resource if contains(statement.Action, "s3:GetObject") && statement.Sid == "ReadInputs"
      ])) == toset([
      "arn:aws:s3:::fraud-test-data-1/training-inputs/*",
      "arn:aws:s3:::fraud-test-data-1/snapshots/*",
      "arn:aws:s3:::fraud-test-data-1/objects/*",
    ])
    error_message = "Training jobs read only staged inputs and snapshot files."
  }

  assert {
    condition = [
      for statement in jsondecode(aws_iam_role_policy.sagemaker.policy).Statement :
      statement.Resource if contains(statement.Action, "s3:PutObject") && statement.Sid == "WriteModelOutput"
    ] == [["arn:aws:s3:::fraud-test-models-1/sagemaker/*"]]
    error_message = "Training jobs write model output only under sagemaker/."
  }

  assert {
    condition = alltrue(flatten([
      for policy in [aws_iam_role_policy.sagemaker.policy, aws_iam_role_policy.workflow.policy, aws_iam_role_policy.scheduler.policy] : [
        for statement in jsondecode(policy).Statement : !contains(statement.Action, "*")
      ]
    ]))
    error_message = "Policies list their actions explicitly."
  }

  assert {
    condition = alltrue([
      for role in [aws_iam_role.sagemaker, aws_iam_role.workflow, aws_iam_role.scheduler] :
      jsondecode(role.assume_role_policy).Statement[0].Condition.StringEquals["aws:SourceAccount"] == "123456789012"
    ])
    error_message = "Service roles can be assumed only on behalf of this account."
  }
}

run "schedule_is_off_by_default_and_never_retries" {
  command = apply

  module {
    source = "./modules/workflow"
  }

  assert {
    condition     = aws_scheduler_schedule.month.state == "DISABLED"
    error_message = "Months start only on demand unless the schedule is enabled."
  }

  assert {
    condition     = aws_scheduler_schedule.month.target[0].retry_policy[0].maximum_retry_attempts == 0
    error_message = "A retried start could begin a second month."
  }
}

run "schedule_can_be_enabled" {
  command = apply

  module {
    source = "./modules/workflow"
  }

  variables {
    schedule_enabled = true
  }

  assert {
    condition     = aws_scheduler_schedule.month.state == "ENABLED"
    error_message = "Enabling the schedule starts months on it."
  }
}
