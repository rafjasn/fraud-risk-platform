output "address" {
  description = "Database host name."
  value       = aws_db_instance.this.address
}

output "security_group_id" {
  description = "Security group that clients' ingress rules attach to."
  value       = aws_security_group.this.id
}

output "master_secret_arn" {
  description = "RDS-managed secret with the master username and password."
  value       = aws_db_instance.this.master_user_secret[0].secret_arn
}

output "roles_secret_arn" {
  description = "Secret with each application role's password, keyed by role."
  value       = aws_secretsmanager_secret.roles.arn
}

output "url_secret_arns" {
  description = "Connection URL secrets by role: source_app, source_reader, platform_app, mlflow_app."
  value       = { for role, secret in aws_secretsmanager_secret.url : role => secret.arn }
}

output "provision_sql" {
  description = "psql script that creates the roles and databases."
  value       = file("${path.module}/provision.sql")
}
