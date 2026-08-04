terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket = "sleeman-polyai-k8s-tfstate"
    key    = "k8s-cluster/terraform.tfstate"
    region = "us-east-1"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "sleeman-${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.region}a", "${var.region}b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]

  map_public_ip_on_launch = true 
  


  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Project = "sleeman-polyai-t006"
  }
}

module "k8s_cluster" {
  source = "./modules/k8s-cluster"

  cluster_name       = var.cluster_name
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnets
  instance_key_name  = var.instance_key_name
  region             = var.region
}

module "ingress" {
  source = "./modules/ingress"

  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnets
  worker_asg_name   = module.k8s_cluster.asg_name
  cluster_sg_id     = module.k8s_cluster.cluster_sg_id

  domain_name       = "sleeman01.fursa.click"
  hosted_zone_name  = "fursa.click"
  ingress_node_port = 30080
  name_prefix       = "sleeman-${var.cluster_name}"
}
