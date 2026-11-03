# The whole stack, wired together, with every AWS call mocked.
mock_provider "aws" {
  source = "./tests/mocks"
}

mock_provider "random" {}

variables {
  account_id = "123456789012"
  image_tag  = "0123abc"
}

run "stack_plans_with_defaults" {
  command = apply

  assert {
    condition     = startswith(output.api_url, "http://")
    error_message = "Without a certificate the API URL is HTTP."
  }

  assert {
    condition     = strcontains(output.run_task_network_configuration, "assignPublicIp=DISABLED")
    error_message = "One-off tasks run in private subnets."
  }

  assert {
    condition     = toset(keys(output.task_definition_arns)) == toset(["ingestion", "seed", "jobs", "api", "mlflow", "db-provision"])
    error_message = "Every Compose role has a task definition."
  }

  assert {
    condition     = module.workflow.schedule_state == "DISABLED"
    error_message = "The default deployment starts no months on its own."
  }
}

run "rejects_mutable_image_tags" {
  command = plan

  variables {
    image_tag = "latest"
  }

  expect_failures = [var.image_tag]
}

run "rejects_malformed_account_ids" {
  command = plan

  variables {
    account_id = "1234"
  }

  expect_failures = [var.account_id]
}

run "rejects_long_environment_names" {
  command = plan

  variables {
    environment = "a-very-long-environment"
  }

  expect_failures = [var.environment]
}
