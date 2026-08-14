terraform {
  required_version = ">= 1.5"

  backend "s3" {
    bucket         = "financial-agent-terraform-state"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "financial-agent"
      Environment = "dev"
      ManagedBy   = "terraform"
    }
  }
}

locals {
  project_name = "financial-agent"
  environment  = "dev"
}

module "networking" {
  source       = "../../modules/networking"
  project_name = local.project_name
  environment  = local.environment
}

module "registry" {
  source       = "../../modules/registry"
  project_name = local.project_name
}

module "database" {
  source       = "../../modules/database"
  project_name = local.project_name
  environment  = local.environment
  vpc_id       = module.networking.vpc_id
  subnet_ids   = module.networking.private_subnet_ids
  db_password  = var.db_password
}

module "cache" {
  source       = "../../modules/cache"
  project_name = local.project_name
  environment  = local.environment
  subnet_ids   = module.networking.private_subnet_ids
}

module "compute" {
  source       = "../../modules/compute"
  project_name = local.project_name
  environment  = local.environment
  subnet_ids   = module.networking.private_subnet_ids
  api_image    = "${module.registry.repository_urls["api"]}:latest"
  ui_image     = "${module.registry.repository_urls["ui"]}:latest"
}

variable "db_password" {
  type      = string
  sensitive = true
}
