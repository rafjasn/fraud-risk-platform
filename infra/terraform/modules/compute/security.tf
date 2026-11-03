# Clients reach each listener only through these groups. Outbound traffic is open:
# tasks need RDS, EFS, S3, Secrets Manager, ECR, CloudWatch, and the IBM download.

resource "aws_security_group" "this" {
  for_each    = toset(["alb", "api", "mlflow", "tasks", "training", "efs"])
  name        = "${var.name}-${each.key}"
  description = "${var.name} ${each.key}"
  vpc_id      = var.vpc_id
}

resource "aws_vpc_security_group_egress_rule" "all" {
  for_each          = toset(["api", "mlflow", "tasks", "training"])
  security_group_id = aws_security_group.this[each.key].id
  description       = "Outbound to AWS services and the internet through NAT"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_ingress_rule" "alb" {
  for_each          = toset(var.api_certificate_arn == null ? ["80"] : ["80", "443"])
  security_group_id = aws_security_group.this["alb"].id
  description       = "Public API traffic"
  ip_protocol       = "tcp"
  from_port         = tonumber(each.key)
  to_port           = tonumber(each.key)
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_egress_rule" "alb_to_api" {
  security_group_id            = aws_security_group.this["alb"].id
  description                  = "Forward to API tasks"
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
  referenced_security_group_id = aws_security_group.this["api"].id
}

locals {
  # destination group => [port, [source groups]]
  internal_ingress = {
    api    = [8000, ["alb", "tasks"]]
    mlflow = [5000, ["tasks", "training"]]
    efs    = [2049, ["tasks", "api"]]
  }
  internal_rules = merge([
    for target, rule in local.internal_ingress : {
      for source in rule[1] : "${source}-to-${target}" => {
        target = target
        port   = rule[0]
        source = source
      }
    }
  ]...)
}

resource "aws_vpc_security_group_ingress_rule" "internal" {
  for_each                     = local.internal_rules
  security_group_id            = aws_security_group.this[each.value.target].id
  description                  = "${each.value.source} to ${each.value.target}"
  ip_protocol                  = "tcp"
  from_port                    = each.value.port
  to_port                      = each.value.port
  referenced_security_group_id = aws_security_group.this[each.value.source].id
}

resource "aws_vpc_security_group_ingress_rule" "database" {
  for_each                     = toset(["tasks", "api", "mlflow"])
  security_group_id            = var.database.security_group_id
  description                  = "PostgreSQL from ${each.key}"
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  referenced_security_group_id = aws_security_group.this[each.key].id
}
