variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}"
  subnet_ids = var.subnet_ids
}

resource "aws_db_instance" "pgvector" {
  identifier     = "${var.project_name}-pgvector-${var.environment}"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.environment == "prod" ? "db.t3.medium" : "db.t3.micro"

  allocated_storage = 20
  storage_encrypted = true

  db_name  = "financial_agent"
  username = "postgres"
  password = var.db_password

  db_subnet_group_name = aws_db_subnet_group.main.name
  skip_final_snapshot  = var.environment != "prod"

  tags = {
    Environment = var.environment
  }
}

variable "db_password" {
  type      = string
  sensitive = true
}

output "db_endpoint" {
  value = aws_db_instance.pgvector.endpoint
}
