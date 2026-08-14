variable "project_name" {
  type = string
}

variable "image_names" {
  type    = list(string)
  default = ["api", "ui"]
}

resource "aws_ecr_repository" "repos" {
  for_each = toset(var.image_names)

  name                 = "${var.project_name}/${each.value}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

output "repository_urls" {
  value = { for k, v in aws_ecr_repository.repos : k => v.repository_url }
}
