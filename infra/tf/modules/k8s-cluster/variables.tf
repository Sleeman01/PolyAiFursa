variable "cluster_name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "instance_key_name" {
  type = string
}

variable "region" {
  type = string
}
