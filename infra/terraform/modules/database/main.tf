# One PostgreSQL instance hosts the source, platform, and MLflow databases with
# separate owners, as in Compose. The master password is managed by RDS; application
# role passwords are generated here and kept in Secrets Manager, so they are in the
# Terraform state too: keep state in an encrypted, access-controlled backend.

locals {
  roles = {
    source_app    = "fraud_source"
    source_reader = "fraud_source"
    platform_app  = "fraud_platform"
    mlflow_app    = "mlflow"
  }
}

resource "aws_db_subnet_group" "this" {
  name       = var.name
  subnet_ids = var.subnet_ids
}

# Ingress rules are added by the compute module for each client security group.
resource "aws_security_group" "this" {
  name        = "${var.name}-database"
  description = "PostgreSQL, reachable only from platform services and jobs"
  vpc_id      = var.vpc_id
}

resource "aws_db_parameter_group" "this" {
  name   = "${var.name}-postgres17"
  family = "postgres17"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "5000"
  }
}

resource "aws_db_instance" "this" {
  identifier                          = var.name
  engine                              = "postgres"
  engine_version                      = "17"
  instance_class                      = var.instance_class
  allocated_storage                   = var.allocated_storage_gb
  max_allocated_storage               = var.allocated_storage_gb * 2
  storage_type                        = "gp3"
  storage_encrypted                   = true
  db_name                             = "fraud"
  username                            = "fraud_admin"
  manage_master_user_password         = true
  db_subnet_group_name                = aws_db_subnet_group.this.name
  vpc_security_group_ids              = [aws_security_group.this.id]
  parameter_group_name                = aws_db_parameter_group.this.name
  multi_az                            = var.multi_az
  publicly_accessible                 = false
  deletion_protection                 = var.deletion_protection
  skip_final_snapshot                 = !var.deletion_protection
  final_snapshot_identifier           = "${var.name}-final"
  backup_retention_period             = 7
  copy_tags_to_snapshot               = true
  auto_minor_version_upgrade          = true
  iam_database_authentication_enabled = false
  performance_insights_enabled        = true
  enabled_cloudwatch_logs_exports     = ["postgresql", "upgrade"]
}

resource "random_password" "role" {
  for_each = local.roles
  length   = 32
  special  = false
}

# Role passwords, read by the db-provision task.
resource "aws_secretsmanager_secret" "roles" {
  name                    = "${var.name}/database/roles"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "roles" {
  secret_id     = aws_secretsmanager_secret.roles.id
  secret_string = jsonencode({ for role in keys(local.roles) : role => random_password.role[role].result })
}

# One connection URL per role, injected into containers as DATABASE_URL and similar.
resource "aws_secretsmanager_secret" "url" {
  for_each                = local.roles
  name                    = "${var.name}/database/${each.key}-url"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "url" {
  for_each  = local.roles
  secret_id = aws_secretsmanager_secret.url[each.key].id
  secret_string = format(
    "%s://%s:%s@%s:%d/%s?sslmode=require&connect_timeout=5",
    each.key == "mlflow_app" ? "postgresql+psycopg" : "postgresql",
    each.key,
    random_password.role[each.key].result,
    aws_db_instance.this.address,
    aws_db_instance.this.port,
    each.value,
  )
}
