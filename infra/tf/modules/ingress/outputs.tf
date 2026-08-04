output "alb_dns_name" {
  value = aws_lb.ingress.dns_name
}

output "app_url" {
  value = "https://${var.domain_name}"
}

output "target_group_arn" {
  value = aws_lb_target_group.ingress.arn
}
