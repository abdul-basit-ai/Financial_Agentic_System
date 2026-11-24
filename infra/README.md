# Infrastructure

The AWS Terraform prototype is archived under `archive/aws` and is unsupported.
It contains placeholder account, network, database, state-backend, and resource
settings. Do not run `terraform init`, `plan`, or `apply` from the archived tree.

The supported deployment direction is the Supabase-backed Python API and worker
with the Streamlit UI. Use the Docker Compose and Kubernetes examples in the
repository root and follow their external-secret setup instructions. No cloud
resources are created or managed by this repository.
