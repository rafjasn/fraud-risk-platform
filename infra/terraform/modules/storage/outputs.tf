output "bucket_names" {
  description = "Bucket names by role: data, models, mlflow."
  value       = { for key, bucket in aws_s3_bucket.this : key => bucket.bucket }
}

output "bucket_arns" {
  description = "Bucket ARNs by role: data, models, mlflow."
  value       = { for key, bucket in aws_s3_bucket.this : key => bucket.arn }
}

output "repository_urls" {
  description = "ECR repository URLs by image: ingestion, training, serving."
  value       = { for key, repository in aws_ecr_repository.this : key => repository.repository_url }
}

output "repository_arns" {
  description = "ECR repository ARNs by image."
  value       = { for key, repository in aws_ecr_repository.this : key => repository.arn }
}
