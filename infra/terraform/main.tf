# AWS counterpart of the Compose stack; see README.md for the deployment sequence
# and what has and has not been verified.

locals {
  name = "fraud-${var.environment}"
}

module "network" {
  source             = "./modules/network"
  name               = local.name
  region             = var.region
  cidr               = var.vpc_cidr
  availability_zones = var.availability_zones
}

module "storage" {
  source = "./modules/storage"
  name   = local.name
}

module "database" {
  source               = "./modules/database"
  name                 = local.name
  vpc_id               = module.network.vpc_id
  subnet_ids           = module.network.private_subnet_ids
  instance_class       = var.db_instance_class
  allocated_storage_gb = var.db_allocated_storage_gb
  multi_az             = var.db_multi_az
  deletion_protection  = var.deletion_protection
}

module "compute" {
  source                    = "./modules/compute"
  name                      = local.name
  region                    = var.region
  vpc_id                    = module.network.vpc_id
  public_subnet_ids         = module.network.public_subnet_ids
  private_subnet_ids        = module.network.private_subnet_ids
  image_tag                 = var.image_tag
  repository_urls           = module.storage.repository_urls
  bucket_names              = module.storage.bucket_names
  bucket_arns               = module.storage.bucket_arns
  api_certificate_arn       = var.api_certificate_arn
  api_desired_count         = var.api_desired_count
  deletion_protection       = var.deletion_protection
  alarm_topic_arn           = var.alarm_topic_arn
  api_latency_alarm_seconds = var.api_latency_alarm_seconds
  log_retention_days        = var.log_retention_days
  database = {
    address           = module.database.address
    security_group_id = module.database.security_group_id
    master_secret_arn = module.database.master_secret_arn
    roles_secret_arn  = module.database.roles_secret_arn
    url_secret_arns   = module.database.url_secret_arns
    provision_sql     = module.database.provision_sql
  }
}

module "workflow" {
  source                     = "./modules/workflow"
  name                       = local.name
  region                     = var.region
  account_id                 = var.account_id
  cluster_arn                = module.compute.cluster_arn
  private_subnet_ids         = module.network.private_subnet_ids
  tasks_security_group_id    = module.compute.tasks_security_group_id
  training_security_group_id = module.compute.training_security_group_id
  jobs_task_definition_arn   = module.compute.task_definition_arns["jobs"]
  jobs_task_role_arn         = module.compute.task_role_arns["jobs"]
  execution_role_arn         = module.compute.execution_role_arn
  sagemaker_image            = "${module.storage.repository_urls["training"]}:${var.image_tag}-sagemaker"
  training_repository_arn    = module.storage.repository_arns["training"]
  image_tag                  = var.image_tag
  bucket_names               = module.storage.bucket_names
  bucket_arns                = module.storage.bucket_arns
  mlflow_tracking_uri        = module.compute.mlflow_tracking_uri
  mlflow_artifact_uri        = module.compute.mlflow_artifact_uri
  training_instance_type     = var.training_instance_type
  training_volume_gb         = var.training_volume_gb
  training_max_runtime_hours = var.training_max_runtime_hours
  training_trials            = var.training_trials
  schedule_expression        = var.month_schedule_expression
  schedule_enabled           = var.month_schedule_enabled
  log_retention_days         = var.log_retention_days
}
