output "state_machine_arn" {
  description = "Month workflow ARN."
  value       = aws_sfn_state_machine.month.arn
}

output "sagemaker_role_arn" {
  description = "Execution role of training jobs."
  value       = aws_iam_role.sagemaker.arn
}

output "schedule_state" {
  description = "Whether the schedule starts months."
  value       = aws_scheduler_schedule.month.state
}
