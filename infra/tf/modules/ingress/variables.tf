variable "vpc_id" {
  description = "VPC ID where the ALB and target group live"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for the ALB (needs at least 2 AZs)"
  type        = list(string)
}

variable "worker_asg_name" {
  description = "Name of the existing worker ASG to attach the target group to"
  type        = string
}

variable "domain_name" {
  description = "Full subdomain to expose, e.g. sleeman01.fursa.click"
  type        = string
  default     = "sleeman01.fursa.click"
}

variable "hosted_zone_name" {
  description = "Shared Route 53 hosted zone name (looked up, not managed)"
  type        = string
  default     = "fursa.click"
}

variable "ingress_node_port" {
  description = "NodePort the Nginx Ingress Controller's Service listens on for HTTP"
  type        = number
  default     = 30080
}

variable "name_prefix" {
  description = "Resource name prefix, matching the sleeman- convention"
  type        = string
  default     = "sleeman-polyai-k8s"
}

variable "cluster_sg_id" {
  description = "Security group ID of the k8s cluster nodes, to allow ALB traffic in on the NodePorts"
  type        = string
}
