"""Prompt loader and versioning registry for agent nodes."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(prompt_name: str) -> tuple[str, str, str]:
    """Loads a prompt template and returns (content, version_tag, sha256_hash)."""
    file_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()
    sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return content, prompt_name, sha256_hash