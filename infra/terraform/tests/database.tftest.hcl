mock_provider "aws" {
  source = "./tests/mocks"
}

mock_provider "random" {
  mock_resource "random_password" {
    defaults = { result = "generatedpassword0123456789abcd" }
  }
}

variables {
  name                 = "fraud-test"
  vpc_id               = "vpc-0123"
  subnet_ids           = ["subnet-a", "subnet-b"]
  instance_class       = "db.t4g.medium"
  allocated_storage_gb = 100
  multi_az             = false
  deletion_protection  = true
}

run "database_is_private_encrypted_and_requires_tls" {
  command = apply

  module {
    source = "./modules/database"
  }

  assert {
    condition     = !aws_db_instance.this.publicly_accessible && aws_db_instance.this.storage_encrypted
    error_message = "The database must be private and encrypted."
  }

  assert {
    condition     = aws_db_instance.this.manage_master_user_password && aws_db_instance.this.engine_version == "17"
    error_message = "RDS must manage the master password of PostgreSQL 17."
  }

  assert {
    condition     = contains([for parameter in aws_db_parameter_group.this.parameter : "${parameter.name}=${parameter.value}"], "rds.force_ssl=1")
    error_message = "The database must refuse connections without TLS."
  }

  assert {
    condition     = aws_db_instance.this.deletion_protection && !aws_db_instance.this.skip_final_snapshot
    error_message = "A protected database keeps a final snapshot."
  }
}

run "each_role_connects_to_its_database_over_tls" {
  command = apply

  module {
    source = "./modules/database"
  }

  assert {
    condition = nonsensitive(aws_secretsmanager_secret_version.url["platform_app"].secret_string) == join("", [
      "postgresql://platform_app:generatedpassword0123456789abcd@",
      "fraud-test.abc123.eu-central-1.rds.amazonaws.com:5432/fraud_platform?sslmode=require&connect_timeout=5",
    ])
    error_message = "The platform URL must name its role, database, and TLS."
  }

  assert {
    condition     = startswith(nonsensitive(aws_secretsmanager_secret_version.url["mlflow_app"].secret_string), "postgresql+psycopg://mlflow_app:")
    error_message = "MLflow needs a SQLAlchemy URL with the psycopg driver."
  }

  assert {
    condition     = strcontains(nonsensitive(aws_secretsmanager_secret_version.url["source_reader"].secret_string), "/fraud_source?")
    error_message = "The source reader connects to the source database."
  }

  assert {
    condition     = toset(keys(jsondecode(nonsensitive(aws_secretsmanager_secret_version.roles.secret_string)))) == toset(["source_app", "source_reader", "platform_app", "mlflow_app"])
    error_message = "The roles secret holds one password per application role."
  }
}

run "provisioning_takes_passwords_as_variables" {
  command = apply

  module {
    source = "./modules/database"
  }

  assert {
    condition     = !strcontains(output.provision_sql, "local-") && strcontains(output.provision_sql, ":'platform_app_password'")
    error_message = "The RDS script must use generated passwords, not the local development ones."
  }

  assert {
    condition     = strcontains(output.provision_sql, "GRANT %I TO CURRENT_USER WITH INHERIT TRUE, SET TRUE")
    error_message = "The non-superuser master must be able to SET each role before creating its database."
  }
}

run "disposable_database_skips_final_snapshot" {
  command = apply

  module {
    source = "./modules/database"
  }

  variables {
    deletion_protection = false
  }

  assert {
    condition     = !aws_db_instance.this.deletion_protection && aws_db_instance.this.skip_final_snapshot
    error_message = "An unprotected database can be destroyed without a final snapshot."
  }
}
