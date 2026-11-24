from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive" / "aws"
active_terraform = [path for path in ROOT.rglob("*.tf") if ARCHIVE not in path.parents]
readme = (ROOT / "README.md").read_text(encoding="utf-8")

if active_terraform:
    paths = ", ".join(str(path.relative_to(ROOT)) for path in active_terraform)
    raise SystemExit(f"Unsupported active Terraform configuration found: {paths}")

if "The AWS Terraform prototype is archived" not in readme:
    raise SystemExit("Infrastructure archive notice is missing.")

print(
    "Infrastructure target check passed: only the documented Supabase path is active."
)
