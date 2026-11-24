"""Versioned benchmark inputs and result provenance."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

METRICS_VERSION: Final = "finqa-strict-results-v1"
SHA256 = r"^[a-f0-9]{64}$"


def hash_file(path: Path) -> str:
    with path.open("rb") as source:
        digest = sha256()
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
        return digest.hexdigest()


def hash_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def id_set_digest(record_ids: Iterable[str]) -> str:
    ids = list(record_ids)
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Benchmark record IDs must be nonempty and unique")
    return hash_json(sorted(ids))


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_sha256: str = Field(pattern=SHA256)
    record_ids_sha256: str = Field(pattern=SHA256)
    split: str = Field(min_length=1)
    scope: Literal["known-record", "open-corpus"]
    corpus_id: str = Field(min_length=1)
    metrics_version: Literal["finqa-strict-results-v1"] = METRICS_VERSION
    display_abs_tol: float = Field(default=0, ge=0, allow_inf_nan=False)
    display_rel_tol: float = Field(default=0, ge=0, allow_inf_nan=False)
    memory: Literal["disabled"] = "disabled"


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config: RunConfig
    execution: Literal["candidate-runtime", "scorer-replay"]
    code_sha256: str = Field(pattern=SHA256)
    models: tuple[tuple[str, str], ...] = ()
    input_states_sha256: str | None = Field(default=None, pattern=SHA256)

    def artifact(self) -> dict[str, object]:
        content = self.model_dump(mode="json")
        return {"manifest": content, "manifest_sha256": hash_json(content)}


def load_manifest(data: dict[str, object]) -> RunManifest:
    content = data.get("manifest")
    if content is None or data.get("manifest_sha256") != hash_json(content):
        raise ValueError("Benchmark requires a valid versioned manifest and checksum")
    return cast(RunManifest, RunManifest.model_validate(content))


def code_digest(root: Path) -> str:
    files = sorted(
        (path.relative_to(root).as_posix(), hash_file(path))
        for name in ("agent", "evaluation", "ingestion", "knowledge")
        for path in (root / name).rglob("*")
        if path.suffix in {".py", ".json", ".j2", ".txt"}
        and "__pycache__" not in path.parts
    )
    return hash_json(files)
