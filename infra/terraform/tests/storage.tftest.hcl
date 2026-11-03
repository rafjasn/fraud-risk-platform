# Offline: the mocked provider never calls AWS.
mock_provider "aws" {
  source = "./tests/mocks"
}

variables {
  name = "fraud-test"
}

run "buckets_are_private_encrypted_versioned_and_tls_only" {
  command = apply

  module {
    source = "./modules/storage"
  }

  assert {
    condition     = toset(keys(aws_s3_bucket.this)) == toset(["data", "models", "mlflow"])
    error_message = "Expected the data, models, and mlflow buckets."
  }

  assert {
    condition = alltrue([
      for block in aws_s3_bucket_public_access_block.this :
      block.block_public_acls && block.block_public_policy && block.ignore_public_acls && block.restrict_public_buckets
    ])
    error_message = "Every bucket must block all public access."
  }

  assert {
    condition = alltrue([
      for config in aws_s3_bucket_server_side_encryption_configuration.this :
      one(config.rule).apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    ])
    error_message = "Every bucket must encrypt objects by default."
  }

  assert {
    condition     = alltrue([for versioning in aws_s3_bucket_versioning.this : versioning.versioning_configuration[0].status == "Enabled"])
    error_message = "Every bucket must keep object versions."
  }

  assert {
    condition = alltrue([
      for policy in aws_s3_bucket_policy.tls_only :
      jsondecode(policy.policy).Statement[0].Effect == "Deny" &&
      jsondecode(policy.policy).Statement[0].Condition.Bool["aws:SecureTransport"] == "false"
    ])
    error_message = "Every bucket must refuse requests without TLS."
  }
}

run "image_tags_are_immutable_and_scanned" {
  command = apply

  module {
    source = "./modules/storage"
  }

  assert {
    condition     = toset(keys(aws_ecr_repository.this)) == toset(["ingestion", "training", "serving"])
    error_message = "Expected one repository per image."
  }

  assert {
    condition = alltrue([
      for repository in aws_ecr_repository.this :
      repository.image_tag_mutability == "IMMUTABLE" && repository.image_scanning_configuration[0].scan_on_push
    ])
    error_message = "Repositories must keep tags immutable and scan images on push."
  }
}
