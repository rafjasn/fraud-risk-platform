resource "aws_efs_file_system" "state" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = { Name = "${var.name}-state" }
}

resource "aws_efs_mount_target" "state" {
  count           = length(var.private_subnet_ids)
  file_system_id  = aws_efs_file_system.state.id
  subnet_id       = var.private_subnet_ids[count.index]
  security_groups = [aws_security_group.this["efs"].id]
}

# Containers run as 1000:1000, as Compose runs them as the local user.
resource "aws_efs_access_point" "state" {
  file_system_id = aws_efs_file_system.state.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/platform-state"

    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "0750"
    }
  }
}

# Only the jobs role may write; the API role may only mount, so it reads releases
# and deployment.json but cannot change them. Plain NFS without TLS is refused.
resource "aws_efs_file_system_policy" "state" {
  file_system_id = aws_efs_file_system.state.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "JobsReadWrite"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.task["jobs"].arn }
        Action    = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
        Resource  = aws_efs_file_system.state.arn
        Condition = { StringEquals = { "elasticfilesystem:AccessPointArn" = aws_efs_access_point.state.arn } }
      },
      {
        Sid       = "ApiReadOnly"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.task["api"].arn }
        Action    = ["elasticfilesystem:ClientMount"]
        Resource  = aws_efs_file_system.state.arn
        Condition = { StringEquals = { "elasticfilesystem:AccessPointArn" = aws_efs_access_point.state.arn } }
      },
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = { AWS = "*" }
        Action    = "*"
        Resource  = aws_efs_file_system.state.arn
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
    ]
  })
}
