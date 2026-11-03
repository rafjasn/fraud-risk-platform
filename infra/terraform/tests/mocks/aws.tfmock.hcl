# Realistic identifiers for mocked resources. The AWS provider validates ARN formats
# even when mocked, and the random strings it would otherwise generate are not ARNs.

mock_resource "aws_iam_role" {
  defaults = { arn = "arn:aws:iam::123456789012:role/mock" }
}

mock_resource "aws_cloudwatch_log_group" {
  defaults = { arn = "arn:aws:logs:eu-central-1:123456789012:log-group:mock" }
}

mock_resource "aws_sfn_state_machine" {
  defaults = { arn = "arn:aws:states:eu-central-1:123456789012:stateMachine:mock" }
}

mock_resource "aws_lb" {
  defaults = {
    arn        = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:loadbalancer/app/mock/0123456789abcdef"
    arn_suffix = "app/mock/0123456789abcdef"
    dns_name   = "mock-0123456789.eu-central-1.elb.amazonaws.com"
  }
}

mock_resource "aws_lb_target_group" {
  defaults = {
    arn        = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:targetgroup/mock/0123456789abcdef"
    arn_suffix = "targetgroup/mock/0123456789abcdef"
  }
}

mock_resource "aws_lb_listener" {
  defaults = { arn = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:listener/app/mock/0123456789abcdef/0000000000000080" }
}

mock_resource "aws_service_discovery_service" {
  defaults = { arn = "arn:aws:servicediscovery:eu-central-1:123456789012:service/srv-mock" }
}

mock_resource "aws_ecs_cluster" {
  defaults = { arn = "arn:aws:ecs:eu-central-1:123456789012:cluster/mock" }
}

mock_resource "aws_ecs_task_definition" {
  defaults = { arn = "arn:aws:ecs:eu-central-1:123456789012:task-definition/mock:1" }
}

mock_resource "aws_efs_file_system" {
  defaults = { arn = "arn:aws:elasticfilesystem:eu-central-1:123456789012:file-system/fs-mock" }
}

mock_resource "aws_efs_access_point" {
  defaults = { arn = "arn:aws:elasticfilesystem:eu-central-1:123456789012:access-point/fsap-mock" }
}

mock_resource "aws_secretsmanager_secret" {
  defaults = { arn = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:mock-AbCdEf" }
}

mock_resource "aws_s3_bucket" {
  defaults = { arn = "arn:aws:s3:::mock-bucket" }
}

mock_resource "aws_ecr_repository" {
  defaults = {
    arn            = "arn:aws:ecr:eu-central-1:123456789012:repository/mock"
    repository_url = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/mock"
  }
}

mock_resource "aws_db_instance" {
  defaults = {
    address = "fraud-test.abc123.eu-central-1.rds.amazonaws.com"
    port    = 5432
    master_user_secret = [{
      secret_arn    = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:rds-master-AbCdEf"
      kms_key_id    = "alias/aws/secretsmanager"
      secret_status = "active"
    }]
  }
}
