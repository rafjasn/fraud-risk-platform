# The API is public behind an Application Load Balancer; MLflow is internal only.
# /metrics stays private: the load balancer answers it with 404.

resource "aws_lb" "api" {
  name                       = "${var.name}-api"
  load_balancer_type         = "application"
  internal                   = false
  security_groups            = [aws_security_group.this["alb"].id]
  subnets                    = var.public_subnet_ids
  drop_invalid_header_fields = true
  enable_deletion_protection = var.deletion_protection
}

resource "aws_lb_target_group" "api" {
  name                 = "${var.name}-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = var.vpc_id
  deregistration_delay = 30

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = var.api_certificate_arn == null ? "forward" : "redirect"
    target_group_arn = var.api_certificate_arn == null ? aws_lb_target_group.api.arn : null

    dynamic "redirect" {
      for_each = var.api_certificate_arn == null ? [] : ["https"]
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.api_certificate_arn == null ? 0 : 1
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.api_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener_rule" "hide_metrics" {
  listener_arn = var.api_certificate_arn == null ? aws_lb_listener.http.arn : aws_lb_listener.https[0].arn
  priority     = 10

  condition {
    path_pattern {
      values = ["/metrics"]
    }
  }

  action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "Not found"
      status_code  = "404"
    }
  }
}

locals {
  services = {
    api = {
      desired_count  = var.api_desired_count
      security_group = "api"
      port           = 8000
      load_balanced  = true
    }
    mlflow = {
      desired_count  = 1
      security_group = "mlflow"
      port           = 5000
      load_balanced  = false
    }
  }
}

resource "aws_ecs_service" "this" {
  for_each                          = local.services
  name                              = each.key
  cluster                           = aws_ecs_cluster.this.id
  task_definition                   = aws_ecs_task_definition.this[each.key].arn
  desired_count                     = each.value.desired_count
  launch_type                       = "FARGATE"
  platform_version                  = "LATEST"
  propagate_tags                    = "SERVICE"
  health_check_grace_period_seconds = each.value.load_balanced ? 120 : null

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.this[each.value.security_group].id]
    assign_public_ip = false
  }

  dynamic "load_balancer" {
    for_each = each.value.load_balanced ? [each.key] : []
    content {
      target_group_arn = aws_lb_target_group.api.arn
      container_name   = load_balancer.value
      container_port   = each.value.port
    }
  }

  service_registries {
    registry_arn = aws_service_discovery_service.this[each.key].arn
  }

  # A release that fails its health checks is rolled back to the previous task set.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.http, aws_efs_mount_target.state]
}

locals {
  alarm_actions = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
  api_dimensions = {
    LoadBalancer = aws_lb.api.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "api_latency" {
  alarm_name          = "${var.name}-api-p95-latency"
  alarm_description   = "API p95 response time above ${var.api_latency_alarm_seconds}s in 3 of 5 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  dimensions          = local.api_dimensions
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  threshold           = var.api_latency_alarm_seconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "api_errors" {
  alarm_name          = "${var.name}-api-server-errors"
  alarm_description   = "More than 1% of API responses were 5xx in 3 of 5 minutes"
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  threshold           = 1
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "error_rate"
    expression  = "IF(requests > 0, 100 * errors / requests, 0)"
    label       = "5xx responses (%)"
    return_data = true
  }

  metric_query {
    id = "errors"

    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      dimensions  = local.api_dimensions
      period      = 60
      stat        = "Sum"
    }
  }

  metric_query {
    id = "requests"

    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      dimensions  = local.api_dimensions
      period      = 60
      stat        = "Sum"
    }
  }
}
