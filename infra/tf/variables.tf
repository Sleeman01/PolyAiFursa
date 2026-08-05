variable "region" {
  description = "AWS region to deploy into"
  type        = string
}

variable "cluster_name" {
  description = "Name prefix for cluster resources"
  type        = string
  default     = "polyai-k8s"
}

variable "instance_key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "alert_email" {
  description = "Email address to subscribe to cluster alert notifications"
  type        = string
}
